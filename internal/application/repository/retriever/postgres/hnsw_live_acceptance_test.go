package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/stretchr/testify/require"
	gormpostgres "gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// This acceptance probe belongs to the fork verification branch. It uses a
// fresh CI database, not a developer's configured application database.
type vectorQueryRecorder struct {
	logger.Interface
	mu  sync.Mutex
	sql string
}

func (r *vectorQueryRecorder) Trace(ctx context.Context, begin time.Time, fc func() (string, int64), err error) {
	query, _ := fc()
	if strings.Contains(query, "AS candidates") && !strings.HasPrefix(query, "EXPLAIN") {
		r.mu.Lock()
		r.sql = query
		r.mu.Unlock()
	}
}

func (r *vectorQueryRecorder) lastSQL() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sql
}

func planHas(value any, key, want string) bool {
	switch value := value.(type) {
	case map[string]any:
		if value[key] == want {
			return true
		}
		for _, child := range value {
			if planHas(child, key, want) {
				return true
			}
		}
	case []any:
		for _, child := range value {
			if planHas(child, key, want) {
				return true
			}
		}
	}
	return false
}

func actualVectorPlan(t *testing.T, db *gorm.DB, query string) any {
	t.Helper()
	require.NotEmpty(t, query)
	var raw string
	// Match the public query's transaction-local settings (TopK=1 expands
	// to 100). In particular ef_search affects the planner's HNSW costs.
	require.NoError(t, db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Exec("SET LOCAL hnsw.ef_search = 100").Error; err != nil {
			return err
		}
		if err := tx.Exec("SET LOCAL hnsw.iterative_scan = strict_order").Error; err != nil {
			return err
		}
		return tx.Raw("EXPLAIN (FORMAT JSON) " + query).Row().Scan(&raw)
	}))
	var plan any
	require.NoError(t, json.Unmarshal([]byte(raw), &plan))
	t.Logf("public VectorRetrieve plan: %s", raw)
	return plan
}

func TestHNSWLiveAcceptance(t *testing.T) {
	dsn := os.Getenv("WEKNORA_HNSW_ACCEPTANCE_DSN")
	if dsn == "" {
		t.Skip("requires the disposable fork CI database")
	}
	require.Equal(t, "1", os.Getenv("WEKNORA_HNSW_ALLOW_SCHEMA_SETUP"))
	recorder := &vectorQueryRecorder{Interface: logger.Default.LogMode(logger.Silent)}
	db, err := gorm.Open(gormpostgres.New(gormpostgres.Config{DSN: dsn, PreferSimpleProtocol: true}), &gorm.Config{Logger: recorder})
	require.NoError(t, err)
	sqlDB, err := db.DB()
	require.NoError(t, err)
	t.Cleanup(func() { _ = sqlDB.Close() })
	sqlDB.SetMaxOpenConns(5)
	var database string
	require.NoError(t, db.Raw("SELECT current_database()").Scan(&database).Error)
	require.Equal(t, "hnsw_acceptance", database, "never point this probe at an application database")
	var exists bool
	require.NoError(t, db.Raw("SELECT to_regclass('embeddings') IS NOT NULL").Scan(&exists).Error)
	require.False(t, exists, "requires a fresh database; this probe never removes existing tables")
	require.NoError(t, db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error)
	require.NoError(t, db.Exec(`CREATE TABLE embeddings (
		id bigserial PRIMARY KEY, created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
		source_id text NOT NULL, source_type integer NOT NULL DEFAULT 0, chunk_id text,
		knowledge_id text, knowledge_base_id text, tag_id text, content text NOT NULL,
		dimension integer NOT NULL, embedding halfvec NOT NULL, is_enabled boolean,
		UNIQUE (source_id, source_type))`).Error)
	t.Setenv("AUTO_MIGRATE", "false")
	loader := NewPostgresRetrieveEngineRepository(db).(*pgRepository)
	rng := rand.New(rand.NewSource(3298))
	var vector []float32
	for offset := 0; offset < 4096; offset += 128 {
		rows := make([]*types.IndexInfo, 0, 128)
		embeddings := make(map[string][]float32, 128)
		for i := offset; i < offset+128; i++ {
			id := fmt.Sprintf("s%d", i+1)
			v := make([]float32, 1536)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			if i == 0 {
				vector = append([]float32(nil), v...)
			}
			embeddings[id] = v
			rows = append(rows, &types.IndexInfo{SourceID: id, ChunkID: id, KnowledgeID: "doc", KnowledgeBaseID: "kb", Content: "article " + id, IsEnabled: true})
		}
		require.NoError(t, loader.BatchSave(context.Background(), rows, map[string]any{"embedding": embeddings}))
	}
	require.NoError(t, db.Exec("ANALYZE embeddings").Error)
	params := types.RetrieveParams{Embedding: vector, TopK: 1, KnowledgeBaseIDs: []string{"kb"}}

	t.Run("public_query_and_planner", func(t *testing.T) {
		before := NewPostgresRetrieveEngineRepository(db).(*pgRepository)
		result, err := before.VectorRetrieve(context.Background(), params)
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.Len(t, result[0].Results, 1)
		require.Equal(t, "s1", result[0].Results[0].SourceID)
		baselinePlan := actualVectorPlan(t, db, recorder.lastSQL())
		require.True(t, planHas(baselinePlan, "Node Type", "Seq Scan"))
		require.NoError(t, db.Raw("SELECT to_regclass('embeddings_embedding_idx_1536') IS NOT NULL").Scan(&exists).Error)
		require.False(t, exists, "AUTO_MIGRATE=false must leave the database unchanged")

		t.Setenv("AUTO_MIGRATE", "true")
		after := NewPostgresRetrieveEngineRepository(db).(*pgRepository)
		result, err = after.VectorRetrieve(context.Background(), params)
		require.NoError(t, err, "the request must remain usable while the background build is scheduled")
		require.Len(t, result, 1)
		require.NotEmpty(t, result[0].Results)
		require.Eventually(t, func() bool {
			var valid bool
			err := db.Raw("SELECT COALESCE((SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass('embeddings_embedding_idx_1536')),false)").Scan(&valid).Error
			return err == nil && valid
		}, 2*time.Minute, 200*time.Millisecond)
		result, err = after.VectorRetrieve(context.Background(), params)
		require.NoError(t, err)
		require.Len(t, result, 1)
		require.NotEmpty(t, result[0].Results)
		// HNSW promises approximate results, not an exact Top-1. Observe its
		// recall smoke separately from deterministic storage assertions.
		t.Logf("ANN smoke: known source=s1, returned source=%s score=%f", result[0].Results[0].SourceID, result[0].Results[0].Score)
		indexedPlan := actualVectorPlan(t, db, recorder.lastSQL())
		require.True(t, planHas(indexedPlan, "Index Name", "embeddings_embedding_idx_1536"), "normal planner must select HNSW; do not force enable_seqscan off")

		// Preserve the newer main behavior: re-indexing updates content, and a
		// repeated source in one batch retains its final value.
		index := &types.IndexInfo{SourceID: "s1", ChunkID: "c1", KnowledgeID: "doc", KnowledgeBaseID: "kb", Content: "updated article", IsEnabled: true}
		additional := map[string]any{"embedding": map[string][]float32{"s1": vector}}
		require.NoError(t, after.Save(context.Background(), index, additional))
		var stored string
		require.NoError(t, db.Raw("SELECT content FROM embeddings WHERE source_id='s1' AND source_type=0").Scan(&stored).Error)
		require.Equal(t, "updated article", stored)
		first, last := *index, *index
		first.Content, last.Content = "old batch value", "final batch value"
		require.NoError(t, after.BatchSave(context.Background(), []*types.IndexInfo{&first, &last}, additional))
		require.NoError(t, db.Raw("SELECT content FROM embeddings WHERE source_id='s1' AND source_type=0").Scan(&stored).Error)
		require.Equal(t, "final batch value", stored)
		result, err = after.VectorRetrieve(context.Background(), params)
		require.NoError(t, err)
		require.NotEmpty(t, result[0].Results)
	})

	t.Run("pinned_session_cleanup_and_healthy_query_after_failure", func(t *testing.T) {
		// A startup setting is the configured default that RESET must restore.
		timedDB, err := gorm.Open(gormpostgres.New(gormpostgres.Config{
			DSN: dsn + "&options=-c%20statement_timeout%3D3000", PreferSimpleProtocol: true,
		}), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
		require.NoError(t, err)
		pool, err := timedDB.DB()
		require.NoError(t, err)
		pool.SetMaxOpenConns(1)
		pool.SetMaxIdleConns(1)
		t.Cleanup(func() { _ = pool.Close() })
		var initialPID, afterPID int
		var configured, restored string
		require.NoError(t, timedDB.Raw("SELECT pg_backend_pid()").Scan(&initialPID).Error)
		require.NoError(t, timedDB.Raw("SHOW statement_timeout").Scan(&configured).Error)
		require.Equal(t, "3s", configured)
		repo := NewPostgresRetrieveEngineRepository(timedDB).(*pgRepository)
		require.NoError(t, repo.createHNSWIndex(2048))
		var built bool
		require.NoError(t, timedDB.Raw("SELECT COALESCE((SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass('embeddings_embedding_idx_2048')),false)").Scan(&built).Error)
		require.True(t, built, "exercise actual successful DDL before checking cleanup")
		require.NoError(t, timedDB.Raw("SHOW statement_timeout").Scan(&restored).Error)
		require.Equal(t, configured, restored)

		// A real DDL failure must not poison the pooled connection, leak the
		// advisory lock or break the next healthy retrieval.
		require.NoError(t, timedDB.Exec("SET default_transaction_read_only=on").Error)
		err = repo.createHNSWIndex(2304)
		require.ErrorContains(t, err, "read-only")
		require.ErrorContains(t, err, "could not build embeddings_embedding_idx_2304")
		require.NoError(t, timedDB.Exec("SET default_transaction_read_only=off").Error)
		require.NoError(t, timedDB.Raw("SHOW statement_timeout").Scan(&restored).Error)
		require.Equal(t, configured, restored)
		require.NoError(t, timedDB.Raw("SELECT pg_backend_pid()").Scan(&afterPID).Error)
		require.Equal(t, initialPID, afterPID, "verify recovery on the same physical connection")
		lease, err := sqlDB.Conn(context.Background())
		require.NoError(t, err)
		defer lease.Close()
		for _, dim := range []int{2048, 2304} {
			var locked, unlocked bool
			require.NoError(t, lease.QueryRowContext(context.Background(), "SELECT pg_try_advisory_lock($1,$2)", hnswAdvisoryLockKey, dim).Scan(&locked))
			require.True(t, locked, "the builder must release its session lock")
			require.NoError(t, lease.QueryRowContext(context.Background(), "SELECT pg_advisory_unlock($1,$2)", hnswAdvisoryLockKey, dim).Scan(&unlocked))
			require.True(t, unlocked)
		}
		result, err := repo.VectorRetrieve(context.Background(), params)
		require.NoError(t, err)
		require.NotEmpty(t, result[0].Results)
	})

	// A hand-built subset index is valid but cannot cover all rows of the
	// dimension. This is an independent catalog/planner boundary, not a mock.
	t.Run("narrow_hand_built_index_is_not_full_coverage", func(t *testing.T) {
		require.NoError(t, db.Exec(`CREATE INDEX hand_built_subset_512 ON embeddings USING hnsw
			((embedding::halfvec(512)) halfvec_cosine_ops)
			WHERE dimension=512 AND knowledge_base_id='only-one-kb'`).Error)
		repo := NewPostgresRetrieveEngineRepository(db).(*pgRepository)
		require.NoError(t, repo.createHNSWIndex(512))
		var valid bool
		require.NoError(t, db.Raw("SELECT COALESCE((SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass('embeddings_embedding_idx_512')),false)").Scan(&valid).Error)
		require.True(t, valid, "a narrower operator index cannot replace full-dimension coverage")
	})
}
