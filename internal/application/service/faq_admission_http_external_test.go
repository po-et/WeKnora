package service_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	service "github.com/Tencent/WeKnora/internal/application/service"
	"github.com/Tencent/WeKnora/internal/handler"
	"github.com/Tencent/WeKnora/internal/types"
	"github.com/gin-gonic/gin"
	"github.com/hibiken/asynq"
	"github.com/stretchr/testify/require"
)

// Fork-only acceptance: real GetImportProgress, including its TaskTenantID
// check and JSON response. This does not mount full authentication/RBAC routes.
func TestFAQAdmissionHTTPProgressPreserved(t *testing.T) {
	if os.Getenv("WK_FAQ_ADMISSION_HTTP") != "1" {
		t.Skip("fork-only FAQ admission HTTP acceptance")
	}
	svc, ctx, taskID, triggerOld := service.FAQAdmissionHTTPFixture(t)
	faq := handler.NewFAQHandler(svc, nil)
	router := gin.New()
	router.GET("/api/v1/faq/import/progress/:task_id", faq.GetImportProgress)
	read := func(phase string) *types.FAQImportProgress {
		request := httptest.NewRequest(http.MethodGet, "/api/v1/faq/import/progress/"+taskID, nil).WithContext(ctx)
		recorder := httptest.NewRecorder()
		router.ServeHTTP(recorder, request)
		t.Logf("FAQ_ADMISSION_HTTP phase=%s status=%d body=%s", phase, recorder.Code, recorder.Body.String())
		require.Equal(t, http.StatusOK, recorder.Code)
		var response struct {
			Success bool                     `json:"success"`
			Data    *types.FAQImportProgress `json:"data"`
		}
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
		require.True(t, response.Success)
		require.NotNil(t, response.Data)
		require.Equal(t, types.FAQImportStatusPending, response.Data.Status)
		require.Equal(t, 2, response.Data.Total)
		return response.Data
	}
	before := read("before_old_replay")
	processErr := triggerOld()
	// Observe HTTP first even on the old implementation: the baseline must
	// expose the completed/total1 response before its error assertion fails.
	after := read("after_old_replay")
	require.Equal(t, before, after)
	require.ErrorIs(t, processErr, asynq.SkipRetry)
}
