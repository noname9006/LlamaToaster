import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { StatCard } from "../components/StatCard";
import { TestStatusPill, StatusPill, type PillTone } from "../components/StatusPill";
import { TokSpeedDemo } from "../components/TokSpeedDemo";
import type { Test, Worker } from "../types";
import { shortId, formatGpuLabel } from "../utils";

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.2): "machines, not users" for
// every number EXCEPT the "Users" stat card below -- every other number and
// machine shown here comes from api.listWorkers()/api.listTests(), both
// already scoped to the caller's own account once authenticated (Stage 4's
// own §4.3/§4.5 scoping) and unfiltered in single-tenant mode (AUTH_ENABLED
// off), unchanged from before that scoping existed. "Users" is a deliberate,
// later exception (operator request): a single unscoped GET /api/stats call
// (server/src/routes/stats.ts) returning just a total account count, no
// per-user data -- every signed-in user sees the same platform-wide number,
// not just a superadmin. A superadmin otherwise sees exactly this same
// personal dashboard, no branch, no admin link -- the full cross-tenant view
// (every user's own machines/runs) still lives entirely on the separate
// admin origin (§5.1).
const WORKER_STATUS_TONE: Record<Worker["status"], PillTone> = {
  offline: "danger",
  idle: "muted",
  busy: "accent",
};

function MachineRow({ worker }: { worker: Worker }) {
  const gpu = worker.hardware?.gpu[0];
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-fg">{worker.displayName}</span>
          <StatusPill label={worker.status} tone={WORKER_STATUS_TONE[worker.status]} />
        </div>
        <div className="mt-0.5 truncate text-xs text-muted">
          {[worker.backend, gpu ? formatGpuLabel(gpu) : null].filter(Boolean).join(" · ") || "no details reported yet"}
        </div>
      </div>
      {worker.status === "busy" && worker.activeJobProgress && (
        <span className="flex-none text-xs text-muted">
          {worker.activeJobProgress.detail ?? worker.activeJobProgress.phase}
        </span>
      )}
    </div>
  );
}

export function Dashboard() {
  const [runs, setRuns] = useState<Test[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [userCount, setUserCount] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  // Self-rescheduling poll (same shape as Workers page's useWorkerStatuses)
  // rather than a one-shot fetch -- a worker's build install/activate here
  // used to look permanently stuck at its pre-job state since nothing ever
  // refetched after the initial mount.
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const [r, w, s] = await Promise.all([api.listTests(), api.listWorkers(), api.getStats()]);
        if (cancelled) return;
        setRuns(r);
        setWorkers(w);
        setUserCount(s.users);
        setLoaded(true);
      } finally {
        if (!cancelled) timerRef.current = window.setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  // A lifetime count, not a live online/total ratio -- see this file's own
  // header comment and MULTIUSER_PLAN.md §5.2: an online/total ratio makes
  // the tile flicker as boxes sleep and turns a headline number into a
  // status widget; per-machine liveness belongs in the strip below, where
  // it's actionable. Abandoned enrolments (never heartbeated) don't count; a
  // machine that's merely powered off right now still does.
  const machinesEverConnected = workers.filter((w) => w.lastHeartbeatAt !== null).length;
  // Terminal outcomes only (done/failed/failed_oom/cancelled) -- a still-
  // queued or in-flight item hasn't been performed yet, so items_total alone
  // (which counts those too) overcounts. Matches the admin dashboard's own
  // "Tests performed" definition (server/src/db/repo.ts's adminRepo.stats).
  const testsTotal = runs.reduce((sum, r) => sum + (r.items_done ?? 0) + (r.items_failed ?? 0) + (r.items_cancelled ?? 0), 0);
  // "Runs" in this app's own vocabulary means repeats (-r), not sweep
  // combinations or job rows -- see TestDetail.tsx's own totalRuns (items x
  // repeats) for the per-run version of this same math. A raw runs.length
  // (job-row count) would undercount relative to testsTotal above, since
  // one job's several sweep combinations each get counted as one test but
  // physically execute `repeats` times.
  const totalRuns = runs.reduce((sum, r) => sum + (r.items_total ?? 0) * (r.config?.sweep?.repeats ?? 1), 0);
  const distinctModelsTested = new Set(runs.map((r) => r.model_id)).size;
  const recent = runs.slice(0, 8);

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Dashboard</h1>

      <section className="grid grid-cols-2 gap-4 md:grid-cols-5">
        <StatCard label="Users" value={userCount ?? "—"} />
        <StatCard label="Machines" value={machinesEverConnected} />
        <StatCard label="Models tested" value={distinctModelsTested} />
        <StatCard label="Tests performed" value={testsTotal} />
        <StatCard label="Total runs" value={totalRuns} />
      </section>

      {loaded && workers.length === 0 ? (
        <section className="mt-6 rounded-xl border border-border bg-surface px-6 py-8 text-center">
          <p className="text-sm font-semibold text-fg">No machines yet.</p>
          <p className="mt-1 text-sm text-muted">LlamaToaster runs benchmarks on your own hardware.</p>
          <Link
            to="/device"
            className="mt-4 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:bg-accent/90"
          >
            Connect your first machine
          </Link>
          <p className="mt-2 text-xs text-muted">Takes about a minute.</p>
        </section>
      ) : (
        <section className="mt-6 grid gap-3 md:grid-cols-2">
          {workers.map((w) => (
            <MachineRow key={w.id} worker={w} />
          ))}
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Recent tests</h2>
        {loaded && recent.length === 0 && <p className="mt-2 text-sm text-muted">No tests yet.</p>}
        <div className="mt-2 flex flex-col divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
          {recent.map((r) => (
            <Link
              key={r.id}
              to={`/tests/${r.id}`}
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
                <TestStatusPill status={r.status} />
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
