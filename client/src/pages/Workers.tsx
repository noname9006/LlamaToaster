import { useWorkerStatuses } from "../api/useWorkerStatus";
import { WorkerCard } from "../components/WorkerCard";

export function Workers() {
  const { order, status, loaded, refresh } = useWorkerStatuses();

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Workers</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        llama.cpp build management per worker. Installing, activating, and deleting a build is
        always a manual click here — nothing downloads or switches on its own. Installable
        versions are checked against GitHub every few minutes and refreshed whenever this page
        loads.
      </p>

      {loaded && order.length === 0 && (
        <p className="mt-6 text-sm text-muted">No workers configured.</p>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {order.map((name) => {
          const s = status[name];
          if (!s) return null;
          return <WorkerCard key={name} worker={s.worker} status={s} onRefresh={() => refresh(s.worker)} />;
        })}
      </div>
    </div>
  );
}
