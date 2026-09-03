import { useEffect, useState } from "react";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { useDeviceEnrolment } from "../api/useDeviceEnrolment";
import { WorkerCard } from "../components/WorkerCard";
import { AddMachinePanel } from "../components/AddMachinePanel";
import { IconChevronDown } from "../components/icons";

export function Workers() {
  const { order, status, loaded, refresh } = useWorkerStatuses();
  // Device enrolment needs AUTH_ENABLED (server-side routes + client gate);
  // checked once here so the collapsible add-machine section simply doesn't
  // exist on a Stage-1-only deployment rather than showing a dead panel.
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    import("../api/client")
      .then(({ api }) => api.getAuthStatus())
      .then((s) => setAuthEnabled(s.authEnabled))
      .catch(() => setAuthEnabled(false));
  }, []);

  // Lifted up (rather than left inside AddMachinePanel) so the compact code
  // box next to the collapsed summary below and the full panel underneath
  // share one code/status/approve state instead of drifting independently.
  const enrolment = useDeviceEnrolment();
  const [detailsOpen, setDetailsOpen] = useState(false);
  useEffect(() => {
    if (loaded && order.length === 0) setDetailsOpen(true);
  }, [loaded, order.length]);
  // Once a typed code resolves to something the user needs to act on
  // (pending confirmation, or an already-approved message), pop the panel
  // open so they see it even if they typed the code while collapsed.
  useEffect(() => {
    if (enrolment.status?.state === "pending" || enrolment.status?.state === "approved" || enrolment.approved) {
      setDetailsOpen(true);
    }
  }, [enrolment.status, enrolment.approved]);

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-fg">Workers</h1>
          <p className="mt-2 max-w-2xl text-sm text-muted">
            llama.cpp build management per worker. Installing, activating, and deleting a build is
            always a manual click here — nothing downloads or switches on its own. Installable
            versions are checked against GitHub every few minutes and refreshed whenever this page
            loads.
          </p>
        </div>
      </div>

      {authEnabled && (
        <details
          className="group mt-4 rounded-xl border border-border bg-surface"
          open={detailsOpen}
          onToggle={(e) => setDetailsOpen(e.currentTarget.open)}
        >
          <summary className="flex cursor-pointer items-center gap-2 px-5 py-3 text-sm font-semibold text-fg select-none">
            Add a machine
            <span className="text-xs font-normal text-muted">
              — connect your machine by running a single terminal command
            </span>
            {/* Kept visible (and interactive) while the section is
                collapsed, so a code from a headless setup can be entered
                without expanding this whole panel first -- stopPropagation
                keeps clicking/typing here from toggling the <details>. A
                <span>, not a <div>, to stay within <summary>'s phrasing
                content model while still getting flex layout from the
                Tailwind class. */}
            <span className="ml-auto flex items-center gap-3">
              <input
                type="text"
                value={enrolment.codeInput}
                onChange={(e) => enrolment.handleCodeChange(e.target.value)}
                onClick={(e) => e.stopPropagation()}
                placeholder="ABCD-EFGH"
                maxLength={9}
                aria-label="Enrolment code"
                className="w-32 rounded-lg border border-border bg-surface px-2.5 py-1 text-center font-mono text-xs uppercase tracking-widest text-fg outline-none focus:border-accent/50"
              />
              <IconChevronDown
                width={14}
                height={14}
                className="text-muted transition-transform group-open:rotate-180"
              />
            </span>
          </summary>
          <div className="border-t border-border px-5 py-4">
            <AddMachinePanel enrolment={enrolment} />
          </div>
        </details>
      )}

      {loaded && order.length === 0 && (
        <p className="mt-6 text-sm text-muted">No workers configured.</p>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {order.map((id) => {
          const worker = status[id];
          if (!worker) return null;
          return <WorkerCard key={id} worker={worker} onRefresh={() => refresh()} />;
        })}
      </div>
    </div>
  );
}