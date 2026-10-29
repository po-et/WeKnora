package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/alicebob/miniredis/v2"
	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// Fork-only diagnostic. ArchiveTask supplies a replayable task without claiming
// retry exhaustion. FastForward expires Redis TTLs, not three hours of wall time.
// Real Asynq workers invoke the real FAQ service, but only in dry-run mode: this
// observes progress ownership, not production database or imported-data loss.
type faqReplayProbeEnqueuer struct {
	client *asynq.Client
	last   *asynq.TaskInfo
}

func (e *faqReplayProbeEnqueuer) Enqueue(task *asynq.Task, opts ...asynq.Option) (*asynq.TaskInfo, error) {
	info, err := e.client.Enqueue(task, opts...)
	if err == nil {
		e.last = info
	}
	return info, err
}

type faqReplayProbeResult struct {
	queueID string
	err     error
}

func TestFAQReplayOwnershipProbe(t *testing.T) {
	if os.Getenv("WK_FAQ_REPLAY_PROBE") != "1" {
		t.Skip("fork-only replay ownership diagnostic")
	}
	for _, scenario := range []struct {
		name        string
		replacement bool
		reuseTaskID bool
	}{
		{name: "no_replacement"},
		{name: "different_task_id", replacement: true},
		{name: "reused_task_id", replacement: true, reuseTaskID: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			f := newFAQWriteFixture(t)
			sqlDB, err := f.db.DB()
			require.NoError(t, err)
			t.Cleanup(func() { _ = sqlDB.Close() })
			mini := miniredis.RunT(t)
			f.svc.redisClient = redis.NewClient(&redis.Options{Addr: mini.Addr()})
			t.Cleanup(func() { _ = f.svc.redisClient.Close() })
			f.svc.tenantRepo = &faqImportTenantLookup{}
			client := asynq.NewClient(asynq.RedisClientOpt{Addr: mini.Addr()})
			t.Cleanup(func() { _ = client.Close() })
			enqueuer := &faqReplayProbeEnqueuer{client: client}
			f.svc.task = enqueuer
			inspector := asynq.NewInspector(asynq.RedisClientOpt{Addr: mini.Addr()})
			t.Cleanup(func() { _ = inspector.Close() })
			ctx, cancel := context.WithTimeout(f.ctx, 2*time.Minute)
			t.Cleanup(cancel)

			enqueue := func(taskID string, count int) (*asynq.TaskInfo, types.FAQImportPayload) {
				entries := make([]types.FAQEntryPayload, count)
				for i := range entries {
					entries[i] = types.FAQEntryPayload{
						StandardQuestion: fmt.Sprintf("How do I complete step %d for %s?", i+1, taskID),
						Answers:          []string{fmt.Sprintf("Follow step %d.", i+1)},
					}
				}
				got, err := f.svc.UpsertFAQEntries(ctx, f.kb.ID, &types.FAQBatchUpsertPayload{
					TaskID: taskID, Mode: types.FAQBatchModeAppend, DryRun: true, Entries: entries,
				})
				require.NoError(t, err)
				require.Equal(t, taskID, got)
				require.NotNil(t, enqueuer.last)
				info := enqueuer.last
				require.Equal(t, types.QueueMaintenance, info.Queue)
				require.Equal(t, 3, info.MaxRetry)
				require.Equal(t, 2*time.Hour, info.Timeout)
				var payload types.FAQImportPayload
				require.NoError(t, json.Unmarshal(info.Payload, &payload))
				require.NotEmpty(t, payload.InstanceID)
				require.Equal(t, taskID+":"+payload.InstanceID, info.ID)
				return info, payload
			}
			readProgress := func(taskID string) *types.FAQImportProgress {
				progress, err := f.svc.GetFAQImportProgress(ctx, taskID)
				require.NoError(t, err)
				require.NotNil(t, progress)
				return progress
			}
			readMarker := func() *runningFAQImportInfo {
				marker, err := f.svc.getRunningFAQImportInfo(ctx, f.kb.ID)
				require.NoError(t, err)
				return marker
			}

			a, aPayload := enqueue("probe-a", 1)
			require.NoError(t, inspector.ArchiveTask(types.QueueMaintenance, a.ID))
			archived, err := inspector.GetTaskInfo(types.QueueMaintenance, a.ID)
			require.NoError(t, err)
			require.Equal(t, asynq.TaskStateArchived, archived.State)
			require.Zero(t, archived.Retried, "archive is an artificial precondition, not retry exhaustion")
			require.NotNil(t, readMarker())
			require.Equal(t, types.FAQImportStatusPending, readProgress(aPayload.TaskID).Status)
			mini.FastForward(3*time.Hour + time.Second)
			require.Nil(t, readMarker())
			require.False(t, mini.Exists(getFAQImportProgressKey(aPayload.TaskID)))
			archived, err = inspector.GetTaskInfo(types.QueueMaintenance, a.ID)
			require.NoError(t, err)
			require.Equal(t, asynq.TaskStateArchived, archived.State)
			t.Logf("FAQ_REPLAY_PRECONDITION archived=%s retried=%d virtual_ttl_advance=3h1s marker_absent=true progress_absent=true",
				a.ID, archived.Retried)

			var b *asynq.TaskInfo
			var bPayload types.FAQImportPayload
			var bPending *types.FAQImportProgress
			var bMarker *runningFAQImportInfo
			if scenario.replacement {
				bTaskID := "probe-b"
				if scenario.reuseTaskID {
					bTaskID = aPayload.TaskID
				}
				b, bPayload = enqueue(bTaskID, 2)
				require.NotEqual(t, a.ID, b.ID)
				require.NotEqual(t, aPayload.InstanceID, bPayload.InstanceID)
				bPending = readProgress(bPayload.TaskID)
				require.Equal(t, types.FAQImportStatusPending, bPending.Status)
				require.Equal(t, 2, bPending.Total)
				bMarker = readMarker()
				require.NotNil(t, bMarker)
				require.Equal(t, bPayload.InstanceID, bMarker.InstanceID)
			}

			blockedB := make(chan struct{})
			releaseB := make(chan struct{})
			enteredB := make(chan struct{})
			var blockedOnce, releaseOnce, enteredOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(releaseB) }) }
			results := make(chan faqReplayProbeResult, 16)
			server := asynq.NewServer(asynq.RedisClientOpt{Addr: mini.Addr()}, asynq.Config{
				Concurrency: 2, Queues: map[string]int{types.QueueMaintenance: 1},
				TaskCheckInterval: 10 * time.Millisecond,
			})
			t.Cleanup(func() {
				release()
				stopped := make(chan struct{})
				go func() { server.Shutdown(); close(stopped) }()
				select {
				case <-stopped:
				case <-time.After(15 * time.Second):
					t.Error("Asynq server cleanup exceeded 15 seconds")
				}
			})
			mux := asynq.NewServeMux()
			mux.HandleFunc(types.TypeFAQImport, func(workerCtx context.Context, task *asynq.Task) error {
				id, _ := asynq.GetTaskID(workerCtx)
				if b != nil && id == b.ID {
					blockedOnce.Do(func() { close(blockedB) })
					select {
					case <-releaseB:
					case <-workerCtx.Done():
						results <- faqReplayProbeResult{queueID: id, err: workerCtx.Err()}
						return workerCtx.Err()
					}
					enteredOnce.Do(func() { close(enteredB) })
				}
				err := f.svc.ProcessFAQImport(workerCtx, task)
				results <- faqReplayProbeResult{queueID: id, err: err}
				return err
			})
			require.NoError(t, server.Start(mux))
			if b != nil {
				select {
				case <-blockedB:
				case <-time.After(20 * time.Second):
					t.Fatal("replacement task did not reach the pre-service gate")
				}
				require.Equal(t, bPending, readProgress(bPayload.TaskID))
			}
			awaitResult := func(queueID string) {
				select {
				case result := <-results:
					require.Equal(t, queueID, result.queueID)
					require.NoError(t, result.err)
				case <-time.After(20 * time.Second):
					t.Fatalf("FAQ handler did not return for %s", queueID)
				}
				require.Eventually(t, func() bool {
					_, err := inspector.GetTaskInfo(types.QueueMaintenance, queueID)
					return errors.Is(err, asynq.ErrTaskNotFound)
				}, 20*time.Second, 10*time.Millisecond, "completed queue task must be acknowledged")
			}
			logState := func(phase string) {
				entry := map[string]interface{}{
					"phase": phase, "scenario": scenario.name,
					"a_queue_id": a.ID, "a_progress": readProgress(aPayload.TaskID),
					"marker": readMarker(),
				}
				if b != nil {
					entry["b_queue_id"] = b.ID
					entry["b_progress_before_replay"] = bPending
					entry["b_progress"] = readProgress(bPayload.TaskID)
					select {
					case <-enteredB:
						entry["b_entered_service"] = true
					default:
						entry["b_entered_service"] = false
					}
				}
				encoded, err := json.Marshal(entry)
				require.NoError(t, err)
				t.Logf("FAQ_REPLAY_OBSERVATION %s", encoded)
			}

			require.NoError(t, inspector.RunTask(types.QueueMaintenance, a.ID))
			awaitResult(a.ID)
			aProgress := readProgress(aPayload.TaskID)
			require.Equal(t, types.FAQImportStatusCompleted, aProgress.Status)
			require.Equal(t, 1, aProgress.Total)
			require.Equal(t, 1, aProgress.Processed)
			require.Equal(t, 1, aProgress.SuccessCount)
			if b == nil {
				require.Nil(t, readMarker())
				logState("a_completed_without_replacement")
				return
			}
			select {
			case <-enteredB:
				t.Fatal("replacement entered the FAQ service before its gate was released")
			default:
			}
			require.Equal(t, bMarker, readMarker(), "old replay must not remove B's running marker")
			if scenario.reuseTaskID {
				require.Equal(t, aProgress, readProgress(bPayload.TaskID))
				require.NotEqual(t, bPending, readProgress(bPayload.TaskID), "observe A overwriting B's pending progress")
			} else {
				require.Equal(t, bPending, readProgress(bPayload.TaskID), "different task IDs preserve B's progress")
			}
			logState("a_completed_while_b_before_service")
			release()
			awaitResult(b.ID)
			bFinal := readProgress(bPayload.TaskID)
			require.Equal(t, types.FAQImportStatusCompleted, bFinal.Status)
			if !scenario.reuseTaskID {
				require.Equal(t, 2, bFinal.Total)
				require.Equal(t, 2, bFinal.Processed)
				require.Equal(t, 2, bFinal.SuccessCount)
				require.Equal(t, []int{0, 1}, bFinal.ValidEntryIndices)
			}
			require.Nil(t, readMarker())
			// Log all counts and validation indices, without asserting that B
			// revalidated every entry after possibly consuming A's saved progress.
			logState("b_returned_after_release")
		})
	}
}
