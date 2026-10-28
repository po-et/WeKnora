package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// Fork-only diagnostic: a real queue and worker process around the existing
// dry-run fixture. It does not exercise model calls or partially written imports.
// The helper registers the FAQ handler directly, not the full application mux,
// HTTP/RBAC layer, dead-letter callback, or application shutdown orchestration.
type processProbeTenant struct {
	interfaces.TenantRepository
	redis *redis.Client
	mode  string
}

func (r *processProbeTenant) GetTenantByID(ctx context.Context, id uint64) (*types.Tenant, error) {
	taskID, _ := asynq.GetTaskID(ctx)
	retry, _ := asynq.GetRetryCount(ctx)
	if err := r.redis.RPush(ctx, "probe:attempts:"+taskID, fmt.Sprintf("%s:%d", r.mode, retry)).Err(); err != nil {
		return nil, err
	}
	switch r.mode {
	case "block":
		<-ctx.Done()
		return nil, ctx.Err()
	case "fail":
		return nil, errors.New("probe dependency unavailable")
	default:
		return &types.Tenant{ID: id}, nil
	}
}

func TestFAQProcessProbeWorker(t *testing.T) {
	mode := os.Getenv("WK_FAQ_PROBE_WORKER")
	if mode == "" {
		t.Skip("fork-only child process")
	}
	f := newFAQWriteFixture(t)
	f.svc.redisClient = redis.NewClient(&redis.Options{Addr: os.Getenv("WK_FAQ_PROBE_REDIS")})
	t.Cleanup(func() { _ = f.svc.redisClient.Close() })
	f.svc.tenantRepo = &processProbeTenant{redis: f.svc.redisClient, mode: mode}
	server := asynq.NewServer(asynq.RedisClientOpt{Addr: os.Getenv("WK_FAQ_PROBE_REDIS")}, asynq.Config{
		Concurrency: 1,
		Queues:      map[string]int{types.QueueMaintenance: 1},
	})
	mux := asynq.NewServeMux()
	mux.HandleFunc(types.TypeFAQImport, f.svc.ProcessFAQImport)
	require.NoError(t, server.Run(mux))
}

type probeRecordingEnqueuer struct {
	client *asynq.Client
	last   *asynq.TaskInfo
}

func (e *probeRecordingEnqueuer) Enqueue(task *asynq.Task, opts ...asynq.Option) (*asynq.TaskInfo, error) {
	info, err := e.client.Enqueue(task, opts...)
	if err == nil {
		e.last = info
	}
	return info, err
}

type probeChild struct {
	cmd  *exec.Cmd
	done chan error
	dead bool
}

func startProbeChild(t *testing.T, mode string) *probeChild {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestFAQProcessProbeWorker$", "-test.timeout=9m", "-test.v")
	cmd.Env = append(os.Environ(), "WK_FAQ_PROBE_WORKER="+mode)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	require.NoError(t, cmd.Start())
	p := &probeChild{cmd: cmd, done: make(chan error, 1)}
	go func() { p.done <- cmd.Wait() }()
	t.Cleanup(func() {
		if !p.dead {
			_ = p.cmd.Process.Kill()
			select {
			case <-p.done:
				p.dead = true
			case <-time.After(5 * time.Second):
				t.Errorf("probe child %d could not be reaped after cleanup kill", p.cmd.Process.Pid)
			}
		}
	})
	return p
}

func stopProbeChild(t *testing.T, p *probeChild, signal os.Signal) {
	t.Helper()
	require.NoError(t, p.cmd.Process.Signal(signal))
	select {
	case err := <-p.done:
		p.dead = true
		if signal == syscall.SIGTERM {
			require.NoError(t, err)
		} else if signal == syscall.SIGKILL {
			var exitErr *exec.ExitError
			require.ErrorAs(t, err, &exitErr)
			status, ok := exitErr.Sys().(syscall.WaitStatus)
			require.True(t, ok)
			require.True(t, status.Signaled())
			require.Equal(t, syscall.SIGKILL, status.Signal())
		}
		t.Logf("PROCESS_PROBE_CHILD_EXIT pid=%d signal=%s error=%v", p.cmd.Process.Pid, signal, err)
	case <-time.After(25 * time.Second):
		t.Fatal("probe child did not stop within bound")
	}
}

func TestFAQProcessRecoveryProbe(t *testing.T) {
	addr := os.Getenv("WK_FAQ_PROBE_REDIS")
	if addr == "" {
		t.Skip("requires dedicated disposable real Redis")
	}
	f := newFAQWriteFixture(t)
	f.svc.redisClient = redis.NewClient(&redis.Options{Addr: addr})
	t.Cleanup(func() { _ = f.svc.redisClient.Close() })
	client := asynq.NewClient(asynq.RedisClientOpt{Addr: addr})
	t.Cleanup(func() { _ = client.Close() })
	enqueuer := &probeRecordingEnqueuer{client: client}
	f.svc.task = enqueuer
	f.svc.tenantRepo = &faqImportTenantLookup{}
	inspector := asynq.NewInspector(asynq.RedisClientOpt{Addr: addr})
	t.Cleanup(func() { _ = inspector.Close() })
	ctx := f.ctx
	require.NoError(t, f.svc.redisClient.Ping(ctx).Err())
	start := time.Now()
	var taskID, queueID string
	observe := func(phase string) {
		// Bound diagnostic reads without changing task/lease/retry deadlines.
		observeCtx, observeCancel := context.WithTimeout(f.ctx, 5*time.Second)
		defer observeCancel()
		observationErrors := make(map[string]string)
		recordError := func(operation string, err error) {
			if err != nil {
				observationErrors[operation] = err.Error()
			}
		}
		info, err := inspector.GetTaskInfo(types.QueueMaintenance, queueID)
		state := "unknown"
		retried := -1
		if err == nil {
			state, retried = info.State.String(), info.Retried
		} else if errors.Is(err, asynq.ErrTaskNotFound) {
			state = "absent"
		} else {
			recordError("queue_lookup", err)
		}
		progress, err := f.svc.GetFAQImportProgress(observeCtx, taskID)
		recordError("progress", err)
		marker, err := f.svc.getRunningFAQImportInfo(observeCtx, f.kb.ID)
		recordError("marker", err)
		attempts, err := f.svc.redisClient.LRange(observeCtx, "probe:attempts:"+queueID, 0, -1).Result()
		recordError("attempts", err)
		markerTTL, err := f.svc.redisClient.TTL(observeCtx, getFAQImportRunningKey(f.kb.ID)).Result()
		recordError("marker_ttl", err)
		progressTTL, err := f.svc.redisClient.TTL(observeCtx, getFAQImportProgressKey(taskID)).Result()
		recordError("progress_ttl", err)
		entry := map[string]interface{}{
			"phase": phase, "observed_at": time.Now().UTC().Format(time.RFC3339Nano), "elapsed_seconds": time.Since(start).Seconds(),
			"task_id": taskID, "asynq_id": queueID, "queue_state": state, "retried": retried,
			"progress": progress, "marker": marker, "attempts": attempts,
			"marker_ttl_seconds": markerTTL.Seconds(), "progress_ttl_seconds": progressTTL.Seconds(),
			"observation_errors": observationErrors,
		}
		if enqueuer.last != nil {
			entry["max_retry"] = enqueuer.last.MaxRetry
			entry["timeout_seconds"] = enqueuer.last.Timeout.Seconds()
		}
		data, marshalErr := json.Marshal(entry)
		require.NoError(t, marshalErr)
		t.Logf("PROCESS_PROBE %s", data)
	}
	defer func() {
		if t.Failed() {
			observe("failure_snapshot")
		}
	}()
	enqueue := func(name string) {
		var err error
		taskID, err = f.svc.UpsertFAQEntries(ctx, f.kb.ID, &types.FAQBatchUpsertPayload{
			TaskID: name, Mode: types.FAQBatchModeAppend, DryRun: true,
			Entries: []types.FAQEntryPayload{{StandardQuestion: "How do I reset my password?", Answers: []string{"Use the reset link."}}},
		})
		require.NoError(t, err)
		// A live healthy worker may complete and remove the marker before
		// Upsert returns. Capture identity from the real Enqueue response.
		require.NotNil(t, enqueuer.last)
		require.Equal(t, types.QueueMaintenance, enqueuer.last.Queue)
		require.Equal(t, 3, enqueuer.last.MaxRetry)
		require.Equal(t, 2*time.Hour, enqueuer.last.Timeout)
		queueID = enqueuer.last.ID
		var queuedPayload types.FAQImportPayload
		require.NoError(t, json.Unmarshal(enqueuer.last.Payload, &queuedPayload))
		require.Equal(t, taskID, queuedPayload.TaskID)
		require.Equal(t, taskID+":"+queuedPayload.InstanceID, queueID)
		observe("enqueued")
	}
	awaitCompleted := func() {
		require.Eventually(t, func() bool {
			progress, err := f.svc.GetFAQImportProgress(ctx, taskID)
			return err == nil && progress != nil && progress.Status == types.FAQImportStatusCompleted
		}, 4*time.Minute, 500*time.Millisecond)
		progress, err := f.svc.GetFAQImportProgress(ctx, taskID)
		require.NoError(t, err)
		require.Equal(t, 1, progress.SuccessCount)
		require.Equal(t, 100, progress.Progress)
		marker, err := f.svc.getRunningFAQImportInfo(ctx, f.kb.ID)
		require.NoError(t, err)
		require.Nil(t, marker)
		observe("completed")
	}
	for _, signal := range []syscall.Signal{syscall.SIGTERM, syscall.SIGKILL} {
		enqueue(fmt.Sprintf("probe-signal-%d", signal))
		blocked := startProbeChild(t, "block")
		require.Eventually(t, func() bool {
			return f.svc.redisClient.LLen(ctx, "probe:attempts:"+queueID).Val() > 0
		}, 20*time.Second, 200*time.Millisecond)
		observe("before_" + signal.String())
		stopProbeChild(t, blocked, signal)
		observe("after_" + signal.String())
		healthy := startProbeChild(t, "healthy")
		awaitCompleted()
		stopProbeChild(t, healthy, syscall.SIGTERM)
	}

	enqueue("probe-final-failure")
	failing := startProbeChild(t, "fail")
	require.Eventually(t, func() bool {
		info, err := inspector.GetTaskInfo(types.QueueMaintenance, queueID)
		return err == nil && info.State == asynq.TaskStateArchived
	}, 5*time.Minute, 500*time.Millisecond)
	observe("archived_after_dependency_failure")
	progress, err := f.svc.GetFAQImportProgress(ctx, taskID)
	require.NoError(t, err)
	require.Equal(t, types.FAQImportStatusPending, progress.Status)
	marker, err := f.svc.getRunningFAQImportInfo(ctx, f.kb.ID)
	require.NoError(t, err)
	require.NotNil(t, marker)
	_, err = f.svc.UpsertFAQEntries(ctx, f.kb.ID, &types.FAQBatchUpsertPayload{
		Mode: types.FAQBatchModeAppend, DryRun: true,
		Entries: []types.FAQEntryPayload{{StandardQuestion: "Another question", Answers: []string{"An answer"}}},
	})
	require.ErrorContains(t, err, "已有导入任务")
	stopProbeChild(t, failing, syscall.SIGTERM)

	// This is the same Inspector operation used by the existing admin run_now
	// action, not an HTTP/RBAC test and not manual deletion of Redis keys.
	require.NoError(t, inspector.RunTask(types.QueueMaintenance, queueID))
	observe("admin_run_now")
	healthy := startProbeChild(t, "healthy")
	awaitCompleted()
	enqueue("probe-next-import")
	awaitCompleted()
	stopProbeChild(t, healthy, syscall.SIGTERM)
}
