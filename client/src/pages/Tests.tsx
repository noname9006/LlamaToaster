import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import {
  TestStatusPill,
  StatusDot,
  StatusPill,
  StatusCircle,
  StatusCircleStrip,
  buildProgressUnits,
  describeItemPhase,
  TEST_STATUS_TONE,
  memberChipLabel,
  MEMBER_CIRCLE_TONE,
} from "../components/StatusPill";
import { Th, toggleSort, type SortState } from "../components/Th";
import type { Test, TestConfig, TestItem, TestStatus } from "../types";
import { shortId, formatElapsed, formatFlashAttn } from "../utils";

const TERMINAL_ITEM_STATUSES = new Set(["done", "failed", "failed_oom", "cancelled"]);

// One row's worth of chain: every test sharing a root_run_id (Test A's own
// id -- see Benchmark.tsx's startStage/repo.ts's chain insert, which points
// every child at its root and never at itself). A standalone test (no
// tuning/refine/sweep chain, e.g. most runtime/comparison rows) is simply a
// group of one, and renders exactly as a bare test did before this grouping
// existed.
interface TestGroup {
  rootId: string;
  members: Test[]; // ascending by started_at, so members[0] is the chain's own start and members.at(-1) is its most advanced stage
}

// The stage the row should represent right now: whichever member is
// actually in flight, else the furthest one the chain has reached. This is
// NOT what the ID cell links to -- the ID always points at the chain's own
// root page -- but it drives every other cell (backend/build/status/params)
// so the row shows what the chain is doing *now*, not stale facts from
// Test A.
function currentMember(members: Test[]): Test {
  return members.find((r) => r.status === "running") ?? members.find((r) => r.status === "scheduled") ?? members[members.length - 1];
}

function aggregateItems(members: Test[]): { total: number; done: number; failed: number; cancelled: number } {
  return members.reduce(
    (acc, r) => ({
      total: acc.total + (r.items_total ?? 0),
      done: acc.done + (r.items_done ?? 0),
      failed: acc.failed + (r.items_failed ?? 0),
      cancelled: acc.cancelled + (r.items_cancelled ?? 0),
    }),
    { total: 0, done: 0, failed: 0, cancelled: 0 }
  );
}

// A probe batch's members are independent siblings (each search mode either
// fits or doesn't on its own), unlike a tuning->refine->sweep chain's
// sequential, DEPENDENT stages -- a chain stage failing should still read as
// a hard "failed" (nothing after it will ever run), but a probe scenario
// failing/being stopped should not drag an otherwise-successful batch's
// headline status down to "failed"/"cancelled". Gated on every member
// sharing kind "probe" so chain display is completely untouched.
function isProbeBatch(members: Test[]): boolean {
  return members.length > 1 && members.every((m) => m.kind === "probe");
}

// Mirrors finalizeTest's own precedence (server/src/db/repo.ts) one level up,
// across sibling tests instead of items within one test: a batch stopped
// partway through reads "partial" (some scenarios did complete) rather than
// whatever status currentMember() happens to have picked.
function rowStatus(members: Test[]): TestStatus {
  if (!isProbeBatch(members)) return currentMember(members).status;
  if (members.some((m) => m.status === "running")) return "running";
  if (members.some((m) => m.status === "scheduled")) return "scheduled";
  const anyCancelled = members.some((m) => m.status === "cancelled");
  const anyDone = members.some((m) => m.status === "done");
  if (anyCancelled) return anyDone ? "partial" : "cancelled";
  if (members.every((m) => m.status === "done")) return "done";
  return "partial";
}

const COLUMN_DESCRIPTIONS: Record<string, string> = {
  id: "Unique test identifier — click it to open this test's detail page. A multi-stage chain (Test A/B/C), or a batch of probe search modes run together, shares one id here; the small chips below Params (A/B/C, or each mode's name) jump to an individual stage/scenario.",
  worker: "Which configured machine executed this test (e.g. a GPU box vs a CPU worker).",
  model: "The GGUF model file benchmarked in this test.",
  backend: "llama.cpp compute backend the worker was configured with for this test (e.g. cpu, vulkan, cuda, rocm, sycl — whatever that worker actually uses).",
  build: "llama.cpp build tag that was active on the worker when this test was triggered.",
  params:
    "Every value swept per flag — p=n_prompt, n=n_gen, t=threads, ngl=n_gpu_layers, b=batch_size, " +
    "ub=ubatch_size, ctk/ctv=K/V cache type, fa=flash attention — × repeats per combination.",
  status:
    "running / scheduled (queued behind another test on the same worker, starts automatically once it's free) / done / partial (some sweep combinations failed, or the test was lost/reconciled after some completed — for a probe batch, some scenarios completed before the rest were stopped/failed) / failed / cancelled (stopped by user, or lost with nothing completed), plus how many of the sweep's tests have completed.",
  started: "When this test was triggered.",
};

// Same columns as a bare test, but resolved against a whole chain: id/worker/
// model/started are the chain's own facts (root id, root's pairing, root's
// start time), while backend/build/status track whichever member
// currentMember() says is "now" -- see that function's comment.
function groupSortValue(g: TestGroup, key: string): string | number {
  const root = g.members[0];
  const cur = currentMember(g.members);
  switch (key) {
    case "id":
      return g.rootId;
    case "worker":
      return root.worker_name;
    case "model":
      return root.model_filename || root.model_id;
    case "backend":
      return cur.llama_cpp_backend;
    case "build":
      return cur.llama_cpp_build;
    case "status":
      return rowStatus(g.members);
    case "started":
    default:
      return root.started_at;
  }
}

// Compact rendering of every dimension in the sweep -- a single value prints
// bare, multiple values print as a bracketed list, so e.g. flash_attn
// ["on","off"] (2 combinations from otherwise-fixed params) reads as
// "fa[on,off]" instead of expanding into two full param strings.
function formatSweepParams(sweep: TestConfig["sweep"]): string {
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

// A probe's `sweep` block is vestigial -- it exists only to satisfy the
// trigger route's validation and create the tracked run_item (see
// Benchmark.tsx's verifyPlacement), so printing it here would show made-up
// numbers. The probe spec is what the test actually did, and it rides along
// on the test's own config, needing no extra fetch. Returns null for any
// other kind, which keeps using formatSweepParams.
function formatProbeParams(config: TestConfig): string | null {
  const probe = config.probe;
  if (!probe) return null;
  const [k, v] = probe.kv_pair;
  const moe = probe.placement.n_cpu_moe ? ` moe${probe.placement.n_cpu_moe}` : "";
  return `ctx${probe.candidate_ctx.toLocaleString()} ngl${probe.placement.ngl}${moe} ${k}/${v}`;
}

export function Tests() {
  const [runs, setRuns] = useState<Test[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  // Keyed by test id -- only populated for currently-running tests (see the
  // poll effect below). Holds every run_item so the row can both find the
  // current one (mini status line) and render the full bucketed status-dot
  // strip under Params, without a second fetch for the same data.
  const [runItems, setRunItems] = useState<Record<string, TestItem[] | undefined>>({});
  const timerRef = useRef<number | null>(null);

  const [workerFilter, setWorkerFilter] = useState("");
  const [backendFilter, setBackendFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [sort, setSort] = useState<SortState>({ key: "started", dir: "desc" });

  const workerOptions = useMemo(() => Array.from(new Set(runs.map((r) => r.worker_name))).sort(), [runs]);
  const backendOptions = useMemo(() => Array.from(new Set(runs.map((r) => r.llama_cpp_backend))).sort(), [runs]);

  // §0.5's chain: every test sharing a root_run_id (Test A's own id, per
  // repo.ts's chain insert) collapses to one row. A standalone test (no
  // parent, no children -- most runtime/comparison rows) is a group of one
  // and renders exactly as before this grouping existed.
  const groups = useMemo<TestGroup[]>(() => {
    const byRoot = new Map<string, Test[]>();
    for (const r of runs) {
      const key = r.root_run_id ?? r.id;
      const bucket = byRoot.get(key);
      if (bucket) bucket.push(r);
      else byRoot.set(key, [r]);
    }
    return Array.from(byRoot.entries()).map(([rootId, members]) => ({
      rootId,
      members: [...members].sort((a, b) => a.started_at - b.started_at),
    }));
  }, [runs]);

  // Derived from the same rollup the rows themselves display (rowStatus),
  // not the raw per-test status column -- a probe batch can show "partial"
  // as its headline status without any single underlying test row actually
  // carrying that value, and the filter has to offer what's on screen.
  const statusOptions = useMemo(() => Array.from(new Set(groups.map((g) => rowStatus(g.members)))).sort(), [groups]);

  const visibleGroups = useMemo(() => {
    let list = groups;
    if (workerFilter) list = list.filter((g) => g.members[0].worker_name === workerFilter);
    if (backendFilter) list = list.filter((g) => currentMember(g.members).llama_cpp_backend === backendFilter);
    if (statusFilter) list = list.filter((g) => rowStatus(g.members) === statusFilter);
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...list].sort((a, b) => {
      const av = groupSortValue(a, sort.key);
      const bv = groupSortValue(b, sort.key);
      if (typeof av === "number" && typeof bv === "number") return dir * (av - bv);
      return dir * String(av).localeCompare(String(bv));
    });
  }, [groups, workerFilter, backendFilter, statusFilter, sort]);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const list = await api.listTests();
      if (cancelled) return;
      setRuns(list);
      setLoaded(true);

      // Only running tests need live item data -- done/partial/failed tests'
      // items are all terminal already, and the per-test detail page covers
      // the full item list for those. Fetched in parallel since there's
      // normally at most one or two running at once (one per worker,
      // workers run one bench at a time).
      const running = list.filter((r) => r.status === "running");
      if (running.length > 0) {
        const entries = await Promise.all(
          running.map(async (r) => {
            try {
              const detail = await api.getTest(r.id);
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
      // scheduled test flips to running on its own (server-side, once the
      // test ahead of it on that worker finishes, see routes/tests.ts's
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
      <h1 className="text-2xl font-semibold text-fg">Tests</h1>
      {loaded && runs.length === 0 && <p className="mt-4 text-sm text-muted">No tests yet.</p>}
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

          {visibleGroups.length === 0 ? (
            <p className="mt-4 text-sm text-muted">No tests match the current filters.</p>
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
                  {visibleGroups.map((g) => {
                    const root = g.members[0];
                    const cur = currentMember(g.members);
                    const agg = aggregateItems(g.members);
                    const items = cur.status === "running" ? runItems[cur.id] : undefined;
                    const current =
                      items?.find((it) => !TERMINAL_ITEM_STATUSES.has(it.status) && it.status !== "queued") ??
                      items?.find((it) => it.status === "queued");
                    return (
                      <tr key={g.rootId} className="hover:bg-white/5">
                        <td className="px-4 py-2.5">
                          <Link to={`/tests/${g.rootId}`} className="text-accent hover:underline">
                            <code>{shortId(g.rootId)}</code>
                          </Link>
                        </td>
                        <td className="px-4 py-2.5 text-fg">{root.worker_name}</td>
                        <td className="px-4 py-2.5 text-muted">{root.model_filename || `${shortId(root.model_id)}…`}</td>
                        <td className="px-4 py-2.5 text-muted">{cur.llama_cpp_backend}</td>
                        <td className="px-4 py-2.5 text-muted">{cur.llama_cpp_build}</td>
                        <td className="px-4 py-2.5 text-muted whitespace-nowrap font-mono text-xs">
                          {cur.kind === "probe" && (
                            <span className="mr-1.5 rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-semibold not-italic text-accent">
                              probe
                            </span>
                          )}
                          {formatProbeParams(cur.config) ?? formatSweepParams(cur.config.sweep)}
                          {cur.status === "running" && items && items.length > 0 && (
                            <StatusCircleStrip units={buildProgressUnits(items, cur.config.sweep.repeats)} />
                          )}
                          {g.members.length > 1 && (
                            <div className="mt-1.5 flex flex-wrap items-center gap-1">
                              {g.members.map((m) => (
                                <Link
                                  key={m.id}
                                  to={`/tests/${m.id}`}
                                  title={`${memberChipLabel(m)}: ${m.status}`}
                                  className="inline-flex items-center gap-1"
                                >
                                  <StatusCircle tone={MEMBER_CIRCLE_TONE[m.status]} />
                                  <StatusPill label={memberChipLabel(m)} tone={TEST_STATUS_TONE[m.status]} />
                                </Link>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex flex-wrap items-center gap-2">
                            <TestStatusPill status={rowStatus(g.members)} />
                            {agg.total ? (
                              <span className="text-xs text-muted">
                                {agg.done}/{agg.total}
                                {agg.failed ? ` (${agg.failed} failed)` : ""}
                                {agg.cancelled ? ` (${agg.cancelled} cancelled)` : ""}
                              </span>
                            ) : null}
                            {cur.status === "running" && (
                              <span className="text-xs text-muted">{formatElapsed(now - root.started_at)}</span>
                            )}
                          </div>
                          {cur.status === "running" && current && (
                            <div className="mt-1 flex items-center gap-1.5">
                              <StatusDot status={current.status} />
                              <span className="text-xs text-muted">
                                step {current.idx + 1}/{cur.items_total}
                              </span>
                              <span className="max-w-xs truncate text-xs text-muted" title={describeItemPhase(current)}>
                                {describeItemPhase(current)}
                              </span>
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-2.5 text-muted">{new Date(root.started_at).toLocaleString()}</td>
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
        <p className="mt-2 text-sm text-muted">Polling every 5s (a test is in progress or scheduled)…</p>
      )}
    </div>
  );
}
