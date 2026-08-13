import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { RunStatusPill, StatusDot, StatusCircleStrip, buildProgressUnits, describeItemPhase } from "../components/StatusPill";
import { Th, toggleSort, type SortState } from "../components/Th";
import type { Run, RunConfig, RunItem } from "../types";
import { shortId, formatElapsed, formatFlashAttn } from "../utils";

const TERMINAL_ITEM_STATUSES = new Set(["done", "failed", "failed_oom", "cancelled"]);

const COLUMN_DESCRIPTIONS: Record<string, string> = {
  id: "Unique run identifier — click it to open this run's detail page.",
  worker: "Which configured machine executed this run (e.g. a GPU box vs a CPU worker).",
  model: "The GGUF model file benchmarked in this run.",
  backend: "llama.cpp compute backend the worker was configured with for this run (e.g. cpu, vulkan, cuda, rocm, sycl — whatever that worker actually uses).",
  build: "llama.cpp build tag that was active on the worker when this run was triggered.",
  params:
    "Every value swept per flag — p=n_prompt, n=n_gen, t=threads, ngl=n_gpu_layers, b=batch_size, " +
    "ub=ubatch_size, ctk/ctv=K/V cache type, fa=flash attention — × repeats per combination.",
  status:
    "running / scheduled (queued behind another run on the same worker, starts automatically once it's free) / done / partial (some sweep combinations failed, or the run was lost/reconciled after some completed) / failed / cancelled (stopped by user, or lost with nothing completed), plus how many of the sweep's tests have completed.",
  started: "When this run was triggered.",
};

function runSortValue(r: Run, key: string): string | number {
  switch (key) {
    case "id":
      return r.id;
    case "worker":
      return r.worker_name;
    case "model":
      return r.model_filename || r.model_id;
    case "backend":
      return r.llama_cpp_backend;
    case "build":
      return r.llama_cpp_build;
    case "status":
      return r.status;
    case "started":
    default:
      return r.started_at;
  }
}

// Compact rendering of every dimension in the sweep -- a single value prints
// bare, multiple values print as a bracketed list, so e.g. flash_attn
// ["on","off"] (2 combinations from otherwise-fixed params) reads as
// "fa[on,off]" instead of expanding into two full param strings.
function formatSweepParams(sweep: RunConfig["sweep"]): string {
  const fmt = (arr: (number | string)[]): string => (arr.length > 1 ? `[${arr.join(",")}]` : String(arr[0]));
  // Normalize + dedupe: a sweep poisoned by the old seeding bug can hold
  // e.g. ["true", "on"], which are the same value twice, not two combinations.
  const faValues = [...new Set(sweep.flash_attn.map((v) => formatFlashAttn(String(v))))];
  return (
    `p${fmt(sweep.n_prompt)} n${fmt(sweep.n_gen)} t${fmt(sweep.threads)} ngl${fmt(sweep.n_gpu_layers)} ` +
    `b${fmt(sweep.batch_size)} ub${fmt(sweep.ubatch_size)} ctk${fmt(sweep.cache_type_k)} ctv${fmt(sweep.cache_type_v)} ` +
    `fa${fmt(faValues)} ×${sweep.repeats}`
  );
}

export function Runs() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  // Keyed by run id -- only populated for currently-running runs (see the
  // poll effect below). Holds every run_item so the row can both find the
  // current one (mini status line) and render the full bucketed status-dot
  // strip under Params, without a second fetch for the same data.
  const [runItems, setRunItems] = useState<Record<string, RunItem[] | undefined>>({});
  const timerRef = useRef<number | null>(null);

  const [workerFilter, setWorkerFilter] = useState("");
  const [backendFilter, setBackendFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [sort, setSort] = useState<SortState>({ key: "started", dir: "desc" });

  const workerOptions = useMemo(() => Array.from(new Set(runs.map((r) => r.worker_name))).sort(), [runs]);
  const backendOptions = useMemo(() => Array.from(new Set(runs.map((r) => r.llama_cpp_backend))).sort(), [runs]);
  const statusOptions = useMemo(() => Array.from(new Set(runs.map((r) => r.status))).sort(), [runs]);

  const visibleRuns = useMemo(() => {
    let list = runs;
    if (workerFilter) list = list.filter((r) => r.worker_name === workerFilter);
    if (backendFilter) list = list.filter((r) => r.llama_cpp_backend === backendFilter);
    if (statusFilter) list = list.filter((r) => r.status === statusFilter);
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...list].sort((a, b) => {
      const av = runSortValue(a, sort.key);
      const bv = runSortValue(b, sort.key);
      if (typeof av === "number" && typeof bv === "number") return dir * (av - bv);
      return dir * String(av).localeCompare(String(bv));
    });
  }, [runs, workerFilter, backendFilter, statusFilter, sort]);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const list = await api.listRuns();
      if (cancelled) return;
      setRuns(list);
      setLoaded(true);

      // Only running runs need live item data -- done/partial/failed runs'
      // items are all terminal already, and the per-run detail page covers
      // the full item list for those. Fetched in parallel since there's
      // normally at most one or two running at once (one per worker,
      // workers run one bench at a time).
      const running = list.filter((r) => r.status === "running");
      if (running.length > 0) {
        const entries = await Promise.all(
          running.map(async (r) => {
            try {
              const detail = await api.getRun(r.id);
              return [r.id, detail.items] as const;
            } catch {
              return [r.id, undefined] as const;
            }
          })
        );
        if (cancelled) return;
        setRunItems(Object.fromEntries(entries));
      }

      // Keep polling while anything is running *or* scheduled -- a
      // scheduled run flips to running on its own (server-side, once the
      // run ahead of it on that worker finishes, see routes/runs.ts's
      // dispatchScheduledRun) with nothing else prompting a refresh, so this
      // page needs to keep checking even when nothing it can currently see
      // is "running" yet.
      if (list.some((r) => r.status === "running" || r.status === "scheduled")) {
        timerRef.current = window.setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  const hasRunning = runs.some((r) => r.status === "running");
  useEffect(() => {
    if (!hasRunning) return;
    const tickId = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(tickId);
  }, [hasRunning]);

  const filtersActive = workerFilter || backendFilter || statusFilter;

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Runs</h1>
      {loaded && runs.length === 0 && <p className="mt-4 text-sm text-muted">No runs yet.</p>}
      {runs.length > 0 && (
        <>
          <div className="mt-4 flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Worker</span>
              <select
                value={workerFilter}
                onChange={(e) => setWorkerFilter(e.target.value)}
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              >
                <option value="">All</option>
                {workerOptions.map((w) => (
                  <option key={w} value={w}>
                    {w}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Backend</span>
              <select
                value={backendFilter}
                onChange={(e) => setBackendFilter(e.target.value)}
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              >
                <option value="">All</option>
                {backendOptions.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Status</span>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              >
                <option value="">All</option>
                {statusOptions.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            {filtersActive && (
              <button
                type="button"
                onClick={() => {
                  setWorkerFilter("");
                  setBackendFilter("");
                  setStatusFilter("");
                }}
                className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-accent"
              >
                Clear filters
              </button>
            )}
          </div>

          {visibleRuns.length === 0 ? (
            <p className="mt-4 text-sm text-muted">No runs match the current filters.</p>
          ) : (
            <div className="mt-4 overflow-x-auto rounded-xl border border-border bg-surface">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
                    <Th label="ID" description={COLUMN_DESCRIPTIONS.id} sortKey="id" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Worker" description={COLUMN_DESCRIPTIONS.worker} sortKey="worker" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Model" description={COLUMN_DESCRIPTIONS.model} sortKey="model" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Backend" description={COLUMN_DESCRIPTIONS.backend} sortKey="backend" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Build" description={COLUMN_DESCRIPTIONS.build} sortKey="build" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Params" description={COLUMN_DESCRIPTIONS.params} className="!px-4 !py-2.5" />
                    <Th label="Status" description={COLUMN_DESCRIPTIONS.status} sortKey="status" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                    <Th label="Started" description={COLUMN_DESCRIPTIONS.started} sortKey="started" sort={sort} onSort={(k) => setSort((s) => toggleSort(s, k))} className="!px-4 !py-2.5" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {visibleRuns.map((r) => {
                    const items = r.status === "running" ? runItems[r.id] : undefined;
                    const current =
                      items?.find((it) => !TERMINAL_ITEM_STATUSES.has(it.status) && it.status !== "queued") ??
                      items?.find((it) => it.status === "queued");
                    return (
                      <tr key={r.id} className="hover:bg-white/5">
                        <td className="px-4 py-2.5">
                          <Link to={`/runs/${r.id}`} className="text-accent hover:underline">
                            <code>{shortId(r.id)}</code>
                          </Link>
                        </td>
                        <td className="px-4 py-2.5 text-fg">{r.worker_name}</td>
                        <td className="px-4 py-2.5 text-muted">{r.model_filename || `${shortId(r.model_id)}…`}</td>
                        <td className="px-4 py-2.5 text-muted">{r.llama_cpp_backend}</td>
                        <td className="px-4 py-2.5 text-muted">{r.llama_cpp_build}</td>
                        <td className="px-4 py-2.5 text-muted whitespace-nowrap font-mono text-xs">
                          {formatSweepParams(r.config.sweep)}
                          {r.status === "running" && items && items.length > 0 && (
                            <StatusCircleStrip units={buildProgressUnits(items, r.config.sweep.repeats)} />
                          )}
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex flex-wrap items-center gap-2">
                            <RunStatusPill status={r.status} />
                            {r.items_total ? (
                              <span className="text-xs text-muted">
                                {r.items_done}/{r.items_total}
                                {r.items_failed ? ` (${r.items_failed} failed)` : ""}
                                {r.items_cancelled ? ` (${r.items_cancelled} cancelled)` : ""}
                              </span>
                            ) : null}
                            {r.status === "running" && (
                              <span className="text-xs text-muted">{formatElapsed(now - r.started_at)}</span>
                            )}
                          </div>
                          {r.status === "running" && current && (
                            <div className="mt-1 flex items-center gap-1.5">
                              <StatusDot status={current.status} />
                              <span className="text-xs text-muted">
                                step {current.idx + 1}/{r.items_total}
                              </span>
                              <span className="max-w-xs truncate text-xs text-muted" title={describeItemPhase(current)}>
                                {describeItemPhase(current)}
                              </span>
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-2.5 text-muted">{new Date(r.started_at).toLocaleString()}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {runs.some((r) => r.status === "running" || r.status === "scheduled") && (
        <p className="mt-2 text-sm text-muted">Polling every 5s (a run is in progress or scheduled)…</p>
      )}
    </div>
  );
}
