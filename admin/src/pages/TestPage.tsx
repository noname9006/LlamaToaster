import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError, createAdminTestViewApi } from "../api";
import type { AdminTestSummary } from "../types";
import { SignIn } from "../components/SignIn";
import { TestDetail } from "../../../client/src/pages/TestDetail";
import { TestViewProvider } from "../../../client/src/api/testView";

type Phase =
  | { kind: "loading" }
  | { kind: "unauthorized" }
  | { kind: "not_found" }
  | { kind: "error"; message: string }
  | { kind: "ready"; summary: AdminTestSummary };

// One test, observed. The body is the main site's own TestDetail page, source
// for source -- same tables, charts, live polling while it runs -- pointed at
// the console's read-only mirrors and told to hide every control that would
// change the test. All this file adds is the strip above it saying whose test
// it is.
export function TestPage() {
  const { id = "" } = useParams();
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });

  // The summary doubles as the page's gate. TestDetail itself doesn't surface a
  // failed load (it just renders an empty shell), so checking here first is
  // what turns "signed out" and "no such test" into proper screens -- and it
  // supplies the owner the curve needs (see createAdminTestViewApi).
  useEffect(() => {
    let cancelled = false;
    setPhase({ kind: "loading" });
    api
      .getTestSummary(id)
      .then((summary) => {
        if (!cancelled) setPhase({ kind: "ready", summary });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 403) setPhase({ kind: "unauthorized" });
        else if (err instanceof ApiError && err.status === 404) setPhase({ kind: "not_found" });
        else setPhase({ kind: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const ownerUserId = phase.kind === "ready" ? phase.summary.userId : null;
  // Stable across TestDetail's 2s polling renders -- a fresh object each render
  // would make every consumer of the context re-render for nothing.
  const view = useMemo(
    () => ({
      api: createAdminTestViewApi(ownerUserId),
      readOnly: true,
      testPath: (testId: string) => `/tests/${testId}`,
    }),
    [ownerUserId]
  );

  if (phase.kind === "unauthorized") return <SignIn />;

  if (phase.kind === "not_found") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm text-muted">
          No test with id <code className="text-fg">{id}</code>.
        </p>
        <Link to="/" className="text-sm text-accent hover:underline">
          ← All tests
        </Link>
      </div>
    );
  }

  if (phase.kind === "error") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm text-danger">Could not load this test: {phase.message}</p>
        <Link to="/" className="text-sm text-accent hover:underline">
          ← All tests
        </Link>
      </div>
    );
  }

  if (phase.kind === "loading") {
    return <div className="flex min-h-screen items-center justify-center text-sm text-muted">Loading…</div>;
  }

  const { summary } = phase;
  return (
    // Same page padding as the main site's <main>, so the shared view is laid
    // out exactly as its owner sees it.
    <div className="w-full px-8 py-8">
      <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-border bg-surface px-3.5 py-2 text-xs text-muted">
        <Link to="/" className="font-semibold text-accent hover:underline">
          ← All tests
        </Link>
        <span>
          Observing <b className="text-fg">{summary.userDisplayName ?? "an unknown user"}</b>'s test
        </span>
        <span>read-only — you can watch this test but not change it</span>
      </div>
      <TestViewProvider value={view}>
        <TestDetail />
      </TestViewProvider>
    </div>
  );
}
