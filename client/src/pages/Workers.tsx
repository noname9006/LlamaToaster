import { Link } from "react-router-dom";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { WorkerCard } from "../components/WorkerCard";
import { IconPlus } from "../components/icons";

export function Workers() {
  const { order, status, loaded, refresh } = useWorkerStatuses();

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
        <Link
          to="/device"
          className="flex flex-none items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent"
        >
          <IconPlus width={14} height={14} />
          Add machine
        </Link>
      </div>

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
