import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWorkerStatuses } from "../api/useWorkerStatus";
import { StatCard } from "../components/StatCard";
import { RunStatusPill, WorkerStatusPill, ElapsedSince } from "../components/StatusPill";
import { TokSpeedDemo } from "../components/TokSpeedDemo";
import type { Model, Run } from "../types";
import { shortId } from "../utils";

export function Dashboard() {
  const [models, setModels] = useState<Model[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loaded, setLoaded] = useState(false);
  const { order, status } = useWorkerStatuses();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [m, r] = await Promise.all([api.listModels(), api.listRuns()]);
      if (cancelled) return;
      setModels(m);
      setRuns(r);
      setLoaded(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const running = runs.filter((r) => r.status === "running").length;
  const recent = runs.slice(0, 8);

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Dashboard</h1>

      <section className="mt-6 flex flex-wrap gap-4">
        <StatCard label="Models" value={models.length} />
        <StatCard label="Runs" value={runs.length} />
        <StatCard label="Running" value={running} />
      </section>

      {order.length > 0 && (
        <section className="mt-6 flex flex-wrap gap-3">
          {order.map((name) => {
            const s = status[name];
            return (
              <div
                key={name}
                className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
              >
                <span className="text-sm font-medium text-fg">{name}</span>
                <WorkerStatusPill inaccessible={s?.inaccessible ?? false} loading={s?.loading} />
                {s?.info?.current_run && (
                  <span className="text-xs text-muted">
                    · {s.info.current_run.model_filename} · <ElapsedSince startedAt={s.info.current_run.started_at} />
                  </span>
                )}
              </div>
            );
          })}
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Recent runs</h2>
        {loaded && recent.length === 0 && <p className="mt-2 text-sm text-muted">No runs yet.</p>}
        <div className="mt-2 flex flex-col divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
          {recent.map((r) => (
            <Link
              key={r.id}
              to={`/runs/${r.id}`}
              className="flex items-center gap-3 px-4 py-3 text-sm transition-colors hover:bg-white/5"
            >
              <code className="text-muted">{shortId(r.id)}</code>
              <span className="text-fg">{r.worker_name}</span>
              <span className="text-muted">{r.llama_cpp_backend}</span>
              <span className="ml-auto flex items-center gap-2">
                {r.items_total ? (
                  <span className="text-xs text-muted">
                    {r.items_done}/{r.items_total}
                  </span>
                ) : null}
                <RunStatusPill status={r.status} />
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="mt-8">
        <TokSpeedDemo />
      </section>
    </div>
  );
}
