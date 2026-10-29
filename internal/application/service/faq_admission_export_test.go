package service

import (
	"context"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
)

// FAQAdmissionHTTPFixture is a fork-only test bridge. It lets an external test
// invoke the real HTTP handler without introducing a service -> handler cycle.
func FAQAdmissionHTTPFixture(t *testing.T) (interfaces.KnowledgeService, context.Context, string, func() error) {
	t.Helper()
	f := newFAQAdmissionFixture(t)
	old, taskID := f.prepareReplacement(t, true)
	requestCtx := types.WithCaller(context.Background(), types.Caller{
		TenantID: f.kb.TenantID, UserID: "owner", Role: types.TenantRoleAdmin,
	})
	requestCtx = types.WithExecutionTenant(requestCtx, f.kb.TenantID)
	return f.svc, requestCtx, taskID, func() error {
		return f.svc.ProcessFAQImport(f.ctx, old)
	}
}
