import { useEffect, useState } from "react";
import { useWorkerStatuses } from "../api/useWorkerStatus";
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
          open={loaded && order.length === 0}
        >
          <summary className="flex cursor-pointer items-center gap-2 px-5 py-3 text-sm font-semibold text-fg select-none">
            Add a machine
            <span className="text-xs font-normal text-muted">
              — connect a new GPU box by running one command on it
            </span>
            <IconChevronDown
              width={14}
              height={14}
              className="ml-auto text-muted transition-transform group-open:rotate-180"
            />
          </summary>
          <div className="border-t border-border px-5 py-4">
            <AddMachinePanel />
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