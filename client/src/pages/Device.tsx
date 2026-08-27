import { useEffect, useState } from "react";
import { AddMachinePanel } from "../components/AddMachinePanel";

// MULTIUSER_PLAN.md §3.1's "Add machine" screen -- kept as its own
// bookmarkable /device URL for the case where enrolment starts somewhere the
// browser wasn't already open (a headless box set up over SSH, a code
// relayed from someone else). The actual UI lives in components/
// AddMachinePanel.tsx so the Workers page can render the very same panel at
// the top of its own page (the same component either way, not two separate
// screens).
export function Device() {
  // Reachable directly by URL even when nothing links to it on a
  // Stage-1-only (no AUTH_ENABLED) deployment -- deviceApprovalRoutes are
  // only ever registered server-side when AUTH_ENABLED (server/src/index.ts),
  // so this page needs to know that too rather than attempting fetches
  // against routes that don't exist.
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    import("../api/client")
      .then(({ api }) => api.getAuthStatus())
      .then((s) => setAuthEnabled(s.authEnabled))
      .catch(() => setAuthEnabled(false));
  }, []);

  if (authEnabled === null) {
    return <p className="text-sm text-muted">Loading…</p>;
  }
  if (!authEnabled) {
    return (
      <div>
        <h1 className="text-2xl font-semibold text-fg">Add a machine</h1>
        <p className="mt-4 max-w-lg text-sm text-muted">
          Device enrolment needs user accounts to be enabled on this deployment. Set up a new
          worker with the shared worker token instead — see the setup commands on the Workers
          page.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Add a machine</h1>
      <div className="mt-4">
        <AddMachinePanel />
      </div>
    </div>
  );
}