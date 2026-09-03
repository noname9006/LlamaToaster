import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { StatCard } from "../components/StatCard";
import { TestStatusPill, StatusPill, type PillTone } from "../components/StatusPill";
import { TokSpeedDemo } from "../components/TokSpeedDemo";
import { SETUP_OS_LABELS, platformToSetupOS } from "../components/WorkerCard";
import { IconX } from "../components/icons";
import type { AdminStats, Test, Worker } from "../types";
import { shortId, formatGpuLabel, formatBytes } from "../utils";

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.2) originally made this page
// "machines, not users": every stat card except "Users" derived from the
// caller's own api.listWorkers()/api.listTests() (Stage 4's §4.3/§4.5
// scoping), with only a total account count shown platform-wide. Later
// operator request reversed that for the stat-card row specifically: all six
// cards now come from one GET /api/stats call (server/src/routes/stats.ts),
// which is the same unscoped repo.adminRepo.stats() query the admin surface
// uses -- aggregate counts only, no per-user breakdown, so it carries the
// same "just a count" reasoning that already justified exposing the users
// total here. Every signed-in user sees the same platform-wide numbers, not
// just a superadmin. The machine list and "Recent tests" section BELOW the
// stat cards are still the caller's own, via listWorkers()/listTests() --
// only the headline totals went platform-wide. The full cross-tenant *table*
// view (every user's own machines/runs, filterable) still lives entirely on
// the separate admin origin (§5.1).
const WORKER_STATUS_TONE: Record<Worker["status"], PillTone> = {
  offline: "danger",
  idle: "muted",
  busy: "accent",
};

const HIDDEN_WORKERS_STORAGE_KEY = "llamatoaster:dashboard:hidden-workers";

function readHiddenWorkers(): string[] {
  try {
    const raw = localStorage.getItem(HIDDEN_WORKERS_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function writeHiddenWorkers(ids: string[]): void {
  try {
    localStorage.setItem(HIDDEN_WORKERS_STORAGE_KEY, JSON.stringify(ids));
  } catch {
    /* localStorage unavailable (private browsing, quota) -- hiding just won't survive a reload */
  }
}

// worker_id is the real FK (MULTIUSER_PLAN.md §1.2); worker_name is only a
// point-in-time snapshot taken when the test ran, so a run predating that
// column (or a since-renamed worker) falls back to matching the name that
// was true back then.
function testMatchesWorker(r: Test, worker: Worker): boolean {
  return r.worker_id ? r.worker_id === worker.id : r.worker_name === worker.displayName;
}

function MachineCard({
  worker,
  modelsTested,
  testsPerformed,
  selected,
  hidden,
  onSelect,
  onHide,
  onUnhide,
}: {
  worker: Worker;
  modelsTested: number;
  testsPerformed: number;
  selected: boolean;
  hidden: boolean;
  onSelect: () => void;
  onHide: () => void;
  onUnhide: () => void;
}) {
  const gpu = worker.hardware?.gpu[0];
  const osLabel = SETUP_OS_LABELS.find((o) => o.key === platformToSetupOS(worker.platform))?.label ?? worker.platform ?? "unknown OS";
  const hardwareBits = [
    worker.hardware?.cpu.brand || worker.hardware?.cpu.manufacturer,
    worker.hardware?.mem_total_bytes ? formatBytes(worker.hardware.mem_total_bytes) : null,
    gpu ? formatGpuLabel(gpu) : worker.hardware ? "no GPU" : null,
  ].filter(Boolean);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={`flex cursor-pointer flex-col gap-2 rounded-lg border px-4 py-3 text-left transition-colors ${
        selected ? "border-accent/50 bg-accent/10" : "border-border bg-surface hover:border-accent/30"
      } ${hidden ? "border-dashed opacity-60" : ""}`}
    >
      <div className="flex items-center gap-2">
        <span className="truncate text-sm font-medium text-fg">{worker.displayName}</span>
        <StatusPill label={worker.status} tone={WORKER_STATUS_TONE[worker.status]} />
        <span className="ml-auto flex-none">
          {hidden ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onUnhide();
              }}
              className="text-xs font-semibold text-muted hover:text-accent"
            >
              Unhide
            </button>
          ) : (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onHide();
              }}
              className="text-xs font-semibold text-muted hover:text-danger"
            >
              Hide
            </button>
          )}
        </span>
      </div>
      <div className="truncate text-xs text-muted">
        {osLabel}
        {hardwareBits.length ? ` · ${hardwareBits.join(" · ")}` : ""}
      </div>
      <div className="mt-1 flex items-center gap-4 text-xs">
        <span className="text-fg">
          {modelsTested} <span className="text-muted">models tested</span>
        </span>
        <span className="text-fg">
          {testsPerformed} <span className="text-muted">tests performed</span>
        </span>
      </div>
      {worker.status === "busy" && worker.activeJobProgress && (
        <span className="text-xs text-muted">{worker.activeJobProgress.detail ?? worker.activeJobProgress.phase}</span>
      )}
    </div>
  );
}

export function Dashboard() {
  const [runs, setRuns] = useState<Test[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
  const [hiddenWorkerIds, setHiddenWorkerIds] = useState<string[]>(() => readHiddenWorkers());
  const [showHidden, setShowHidden] = useState(false);
  const [hideFailed, setHideFailed] = useState(false);
  const [hideCancelled, setHideCancelled] = useState(false);
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
        setStats(s);
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

  function hideWorker(id: string): void {
    setHiddenWorkerIds((prev) => {
      if (prev.includes(id)) return prev;
      const next = [...prev, id];
      writeHiddenWorkers(next);
      return next;
    });
  }

  function unhideWorker(id: string): void {
    setHiddenWorkerIds((prev) => {
      const next = prev.filter((x) => x !== id);
      writeHiddenWorkers(next);
      return next;
    });
  }

  const visibleWorkers = showHidden ? workers : workers.filter((w) => !hiddenWorkerIds.includes(w.id));

  // Per-machine "models tested" / "tests performed" -- the app-wide stat
  // cards above went platform-wide (see this file's own header comment), but
  // a rig card still needs its OWN counts, which only this page's already-
  // scoped runs/workers can answer.
  const workerStats = useMemo(() => {
    const map = new Map<string, { models: Set<string>; tests: number }>();
    for (const w of workers) map.set(w.id, { models: new Set(), tests: 0 });
    for (const r of runs) {
      for (const w of workers) {
        if (!testMatchesWorker(r, w)) continue;
        const entry = map.get(w.id)!;
        entry.models.add(r.model_id);
        entry.tests += (r.items_done ?? 0) + (r.items_failed ?? 0) + (r.items_cancelled ?? 0);
      }
    }
    return map;
  }, [workers, runs]);

  const selectedWorker = selectedWorkerId ? workers.find((w) => w.id === selectedWorkerId) : undefined;

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (selectedWorker && !testMatchesWorker(r, selectedWorker)) return false;
      if (hideFailed && r.status === "failed") return false;
      if (hideCancelled && r.status === "cancelled") return false;
      return true;
    });
  }, [runs, selectedWorker, hideFailed, hideCancelled]);

  const recent = filteredRuns.slice(0, 8);

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Dashboard</h1>

      <section className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Users" value={stats?.users ?? "—"} />
        <StatCard label="Machines" value={stats?.machines ?? "—"} />
        <StatCard label="Models tested" value={stats?.modelsTested ?? "—"} />
        <StatCard label="Quants options" value={stats?.quants ?? "—"} />
        <StatCard label="Tests performed" value={stats?.tests ?? "—"} />
        <StatCard label="Total runs" value={stats?.runs ?? "—"} />
      </section>

      <section className="mt-6">
        {workers.length > 0 && hiddenWorkerIds.length > 0 && (
          <div className="mb-2 flex justify-end">
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input
                type="checkbox"
                checked={showHidden}
                onChange={(e) => setShowHidden(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Show hidden ({hiddenWorkerIds.length})
            </label>
          </div>
        )}
        {loaded && workers.length === 0 ? (
          <div className="rounded-xl border border-border bg-surface px-6 py-8 text-center">
            <p className="text-sm font-semibold text-fg">No machines yet.</p>
            <p className="mt-1 text-sm text-muted">LlamaToaster runs benchmarks on your own hardware.</p>
            <Link
              to="/device"
              className="mt-4 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-fg hover:bg-accent/90"
            >
              Connect your first machine
            </Link>
            <p className="mt-2 text-xs text-muted">Takes about a minute.</p>
          </div>
        ) : visibleWorkers.length === 0 ? (
          <div className="rounded-xl border border-border bg-surface px-6 py-8 text-center">
            <p className="text-sm font-semibold text-fg">All machines are hidden.</p>
            <button
              type="button"
              onClick={() => setShowHidden(true)}
              className="mt-2 text-sm text-accent hover:underline"
            >
              Show hidden machines
            </button>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visibleWorkers.map((w) => {
              const s = workerStats.get(w.id);
              return (
                <MachineCard
                  key={w.id}
                  worker={w}
                  modelsTested={s?.models.size ?? 0}
                  testsPerformed={s?.tests ?? 0}
                  selected={selectedWorkerId === w.id}
                  hidden={hiddenWorkerIds.includes(w.id)}
                  onSelect={() => setSelectedWorkerId((cur) => (cur === w.id ? null : w.id))}
                  onHide={() => hideWorker(w.id)}
                  onUnhide={() => unhideWorker(w.id)}
                />
              );
            })}
          </div>
        )}
      </section>

      <section className="mt-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Recent tests</h2>
          <div className="flex flex-wrap items-center gap-4">
            {selectedWorker && (
              <button
                type="button"
                onClick={() => setSelectedWorkerId(null)}
                className="flex items-center gap-1 rounded-lg border border-accent/40 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent hover:bg-accent/20"
              >
                {selectedWorker.displayName}
                <IconX width={12} height={12} />
              </button>
            )}
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input
                type="checkbox"
                checked={hideFailed}
                onChange={(e) => setHideFailed(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Hide failed
            </label>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input
                type="checkbox"
                checked={hideCancelled}
                onChange={(e) => setHideCancelled(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Hide cancelled
            </label>
          </div>
        </div>
        {loaded && recent.length === 0 && (
          <p className="mt-2 text-sm text-muted">{runs.length === 0 ? "No tests yet." : "No tests match the current filters."}</p>
        )}
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
