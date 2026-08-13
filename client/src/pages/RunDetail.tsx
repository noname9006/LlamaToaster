import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import type { ChartConfiguration } from "chart.js";
import { api } from "../api/client";
import {
  RunStatusPill,
  StatusCircleStrip,
  buildItemRepeatUnits,
  describeItemPhase,
  REPEAT_PROGRESS_RE,
} from "../components/StatusPill";
import { Chart } from "../components/Chart";
import { Th, toggleSort, type SortState } from "../components/Th";
import { IconInfo } from "../components/icons";
import type {
  Run,
  ResultRow,
  RunItem,
  GpuMemoryAccuracyLevel,
  GpuMemoryMeasurementSource,
  Model,
  WorkerLlamaCppInfo,
} from "../types";
import { shortId, formatElapsed, formatFlashAttn } from "../utils";

interface ColDef {
  label: string;
  description?: string;
  sortKey?: string;
  // Explicit left/right padding override for columns needing asymmetric
  // spacing -- e.g. n_prompt/threads pull their right edge in tight to the
  // column that immediately follows (n_gen/ngl), which pulls its own left
  // edge in to match, collapsing the visual gap between the pair without
  // touching padding on either column's *other* side. Defaults to the
  // standard px-2 when omitted.
  padX?: string;
}

// One merged table, visible in full as soon as a run is triggered (items are
// pre-created 'queued' for the whole sweep -- see server/src/routes/runs.ts's
// trigger handler), instead of a separate live "Tests" table and a final
// "Results" table that only grew rows as tests completed. `sortKey` is
// omitted for columns that aren't meaningfully sortable ("live t/s" is a
// transient in-flight reading, "detail" is free text/error prose). Each
// row's own status is shown as a color strip beneath it, not a column --
// see the always-rendered indicator row below the main <tr>.
const MERGED_COLUMN_DEFS: ColDef[] = [
  { label: "#", description: "Position of this combination within the sweep.", sortKey: "idx" },
  { label: "detail", description: "Current action / progress text, or the error message if this test failed." },
  { label: "test", description: "pp = prompt processing, tg = text generation, pg = both in one llama-bench call.", sortKey: "test" },
  { label: "n_prompt", description: "-p — prompt tokens processed before generating.", sortKey: "n_prompt", padX: "!pl-1 !pr-0.5" },
  { label: "n_gen", description: "-n — tokens generated after the prompt.", sortKey: "n_gen", padX: "!pl-0.5 !pr-1" },
  { label: "threads", description: "-t — CPU threads used for compute.", sortKey: "threads", padX: "!pl-1 !pr-0.5" },
  { label: "ngl", description: "-ngl — model layers offloaded to GPU (999 = all, i.e. the requested value).", sortKey: "ngl", padX: "!pl-0.5 !pr-1" },
  {
    label: "offload",
    description:
      "Base model: layers actually loaded onto the GPU / this model's total transformer layer count, both read from llama.cpp's own runtime output (not calculated) — loaded may be less than ngl if the model has fewer layers than requested. On an MTP row this is always the base model's own figures, never the draft model's — see the ngld column for the draft's.",
    sortKey: "layers_loaded",
  },
  { label: "batch", description: "-b — logical batch size for prompt processing.", sortKey: "batch" },
  { label: "ubatch", description: "-ub — physical batch size (must be ≤ batch size).", sortKey: "ubatch" },
  { label: "ctk", description: "-ctk — K cache quantization type.", sortKey: "ctk" },
  { label: "ctv", description: "-ctv — V cache quantization type.", sortKey: "ctv" },
  { label: "fa", description: "-fa — flash attention kernel, on or off.", sortKey: "fa" },
  {
    label: "mtp",
    description:
      "Multi-token-prediction speculative decoding, on or off. An \"on\" combination runs via llama-server instead of llama-bench, since llama-bench itself has no MTP support. " +
      "When on, also shows the draft head's acceptance rate (accepted/drafted tokens from llama-server's /metrics, tg test only) -- hover for the raw counts.",
    sortKey: "mtp",
  },
  {
    label: "ngld",
    description:
      "-ngld — MTP draft model's own layers offloaded to GPU, independent of ngl above (the requested value). Only meaningful for an \"on\" mtp combination with a separate --model-draft file — see the offload d column for what was actually loaded.",
    sortKey: "ngld",
  },
  {
    label: "offload d",
    description:
      "Draft model: layers actually loaded onto the GPU / this model's total transformer layer count, both read from llama.cpp's own runtime output (not calculated) — loaded may be less than ngld if the draft model has fewer layers than requested. Only meaningful for an \"on\" mtp combination with a separate --model-draft file.",
    sortKey: "layers_loaded_draft",
  },
  {
    label: "PP tok/s",
    description:
      "Prompt processing (prefill) speed. While running this shows the live current value (blue); once finished, the average across repeats — hover for stddev.",
    sortKey: "pp_tps",
  },
  {
    label: "TG tok/s",
    description:
      "Generation (decode) speed. While running this shows the live current value (blue); once finished, the average across repeats — hover for stddev.",
    sortKey: "tg_tps",
  },
  {
    label: "TTFT",
    description:
      "Estimated time to first token, derived as prompt tokens ÷ PP speed. Not a directly measured request latency — this app benchmarks locally via llama-bench, not through a live serving request.",
    sortKey: "ttft",
  },
  { label: "ram free", description: "Free system RAM immediately before this test started.", sortKey: "ram_free" },
  { label: "ram avg", description: "Average RAM used by the process while this test ran.", sortKey: "ram_avg" },
  { label: "ram max", description: "Peak RAM used by the process during this test.", sortKey: "ram_max" },
  { label: "vram total", description: "Total GPU memory (VRAM) capacity reported by the backend.", sortKey: "vram_total" },
  { label: "vram free", description: "Free GPU memory immediately before this test started (best-effort).", sortKey: "vram_free" },
  { label: "vram avg", description: "Average GPU memory used while this test ran (best-effort).", sortKey: "vram_avg" },
  { label: "vram max", description: "Peak GPU memory used during this test (best-effort).", sortKey: "vram_max" },
];

const LEGACY_COLUMN_DESCRIPTIONS: Record<string, string> = {
  test: "pp = prompt processing, tg = text generation, pg = both in one llama-bench call.",
  n_prompt: "-p — prompt tokens processed before generating.",
  n_gen: "-n — tokens generated after the prompt.",
  threads: "-t — CPU threads used for compute.",
  ngl: "-ngl — model layers offloaded to GPU (999 = all).",
  batch: "-b — logical batch size for prompt processing.",
  ubatch: "-ub — physical batch size (must be ≤ batch size).",
  ctk: "-ctk — K cache quantization type.",
  ctv: "-ctv — V cache quantization type.",
  fa: "-fa — flash attention kernel, on or off.",
  "avg tps": "Average tokens/second across all repeats, as reported by llama-bench.",
  stddev: "Standard deviation of tokens/second across repeats — lower is more consistent.",
  "ram MiB": "Peak RAM used by the process during this test.",
  "vram MiB": "Peak GPU memory used during this test (best-effort — n/a if not measurable on this box).",
};

const TERMINAL_ITEM_STATUSES = new Set(["done", "failed", "failed_oom", "cancelled"]);
const RUNNING_ITEM_STATUSES = new Set(["loading", "processing", "generating", "benchmarking"]);

type StatusFilter = "all" | "queued" | "running" | "done" | "failed" | "cancelled";

const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "queued", label: "Queued" },
  { value: "running", label: "Running" },
  { value: "done", label: "Done" },
  { value: "failed", label: "Failed" },
  { value: "cancelled", label: "Cancelled" },
];

function matchesStatusFilter(status: RunItem["status"], filter: StatusFilter): boolean {
  switch (filter) {
    case "queued":
      return status === "queued";
    case "running":
      return RUNNING_ITEM_STATUSES.has(status);
    case "done":
      return status === "done";
    case "failed":
      return status === "failed" || status === "failed_oom";
    case "cancelled":
      return status === "cancelled";
    default:
      return true;
  }
}

// Mirrors shared/sweep.ts's deriveTestType -- kept as a tiny local inline
// rather than importing across the client/shared boundary for one 3-line
// cosmetic label (see client/src/types.ts's comment on why client files
// don't poke ../../../shared directly).
function testTypeLabel(nPrompt: number, nGen: number): string {
  if (nGen === 0) return "pp";
  if (nPrompt === 0) return "tg";
  return "pg";
}

// Mirrors server/src/routes/results.ts's fmtSpecDecode (the CSV export's own
// formatting for this same pair of counters) -- kept as a tiny local inline
// for the same client/shared-boundary reason as testTypeLabel above.
// undefined means /metrics never confirmed anything (unreachable, or no
// counters at all); drafted: 0 renders as "0/0" with no percentage -- the
// draft head loaded but drafted nothing, a distinct, worse condition than a
// missing reading.
function fmtSpecDecode(drafted: number | null | undefined, accepted: number | null | undefined): string | undefined {
  if (drafted == null || accepted == null) return undefined;
  const rate = drafted > 0 ? ` (${((accepted / drafted) * 100).toFixed(1)}%)` : "";
  return `${accepted}/${drafted}${rate}`;
}

// suspect_count/suspect_samples are only ever populated by the llama-server/
// MTP path (worker/src/serverBench.ts) -- a nonzero count means at least one
// repeat's reading was computable but outside plausible bounds (see
// MAX_PLAUSIBLE_*_TOKENS_PER_SECOND there), most often the known
// llama-server MTP timing bug. Rather than silently averaging those out
// (the old behavior, which could make an entire test_type vanish if every
// repeat was suspect), the worker now always reports a real number here --
// this just makes sure it's visibly flagged rather than looking identical
// to a clean measurement.
function resultSuspectTitle(r: ResultRow): string | undefined {
  if (!r.suspect_count) return undefined;
  const raw = r.suspect_samples?.map((v) => v.toFixed(1)).join(", ") ?? "unknown";
  const total = r.sample_count ?? r.suspect_count;
  return `${r.suspect_count} of ${total} repeat(s) reported an implausible speed and were excluded from this average ` +
    `(raw flagged values: ${raw} tok/s) -- likely the known llama-server MTP timing bug`;
}

// Keyed off measurement_source (never backend_type) so the UI is driven
// entirely by the API's own accuracy metadata rather than hardcoding
// per-backend rules -- the same value/source pair means the same tooltip
// regardless of which backend produced it. Returns undefined for "exact"
// (no tooltip needed) and whenever source is missing (nothing measured at
// all -- the cell itself already reads "n/a"/"—", a tooltip would be noise).
const ACCURACY_SOURCE_EXPLANATIONS: Record<GpuMemoryMeasurementSource, string> = {
  process_gpu_usage: "This value is based on direct process GPU memory reporting.",
  driver_reported_memory:
    "This value is a whole-GPU driver-reported reading (not isolated to this benchmark process) and may include memory used by other running programs.",
  backend_allocation_tracking:
    "This value is estimated from the backend's own internal allocation tracking, not a direct OS/driver measurement.",
  memory_budget_estimate:
    "This value is estimated using the backend's memory budget/heap information and may differ from actual process GPU memory usage.",
  unified_memory_estimate:
    "This value is estimated using unified memory allocation information and may not represent GPU-only memory usage.",
};

function gpuMemoryAccuracyTitle(
  accuracy: GpuMemoryAccuracyLevel | undefined,
  source: GpuMemoryMeasurementSource | null | undefined
): string | undefined {
  if (!accuracy || accuracy === "exact" || !source) return undefined;
  return ACCURACY_SOURCE_EXPLANATIONS[source];
}

// Real SVG circle+"i" (see icons.tsx) in place of the old literal " ⓘ"
// Unicode glyph, which renders as a distorted/non-circular blob in several
// UI fonts. The tooltip lives on the wrapping <span>, not the <svg> itself --
// a bare `title` attribute on an inline SVG root isn't reliably honored as a
// hover tooltip across browsers the way it is on ordinary HTML elements.
function AccuracyIcon({ title }: { title: string | undefined }) {
  if (!title) return null;
  return (
    <span title={title} className="inline-flex flex-none translate-y-[1px] text-muted/70">
      <IconInfo width={11} height={11} />
    </span>
  );
}

interface MemoryCells {
  ramFree: string;
  ramAvg: string;
  ramMax: string;
  vramTotal: string;
  vramFree: string;
  vramAvg: string;
  vramMax: string;
}

// Once an item is terminal, its `results` row (matched by idx) has the true
// final avg/peak/free-before figures. While still running, the ram/vram
// "avg" column shows the sampler's current live reading as a live-updating
// proxy instead -- ram/vram "max" isn't known until the item finishes.
// vram total has no run_items live-tick column at all (it's a static
// hardware fact, not something that needs sampling over time) -- it behaves
// like ram/vram "max" here (dash while running, real value once terminal),
// not like "free" (which run_items *does* track from the item's first tick).
function memoryCells(item: RunItem, result: ResultRow | undefined): MemoryCells {
  const terminal = TERMINAL_ITEM_STATUSES.has(item.status);
  const ramFree = result?.ram_free_before_mib ?? item.ram_free_before_mib;
  const ramAvg = terminal ? result?.ram_avg_mib ?? item.ram_avg_mib : item.ram_mib;
  const ramMax = result?.ram_peak_mib ?? item.ram_peak_mib;
  const vramTotal = result?.gpu_memory_total_mb;
  const vramFree = result?.vram_free_before_mib ?? item.vram_free_before_mib;
  const vramAvg = terminal ? result?.vram_avg_mib ?? item.vram_avg_mib : item.vram_mib;
  const vramMax = result?.vram_peak_mib ?? item.vram_peak_mib;

  const dash = (v: number | null | undefined) => (v == null ? "—" : String(v));
  // VRAM: once terminal, still-null means genuinely unmeasured on this box
  // ("n/a", same convention the old final Results table used) -- while
  // still running, null just means not sampled yet ("—").
  const vramCell = (v: number | null | undefined) => (v == null ? (terminal ? "n/a" : "—") : String(v));

  return {
    ramFree: dash(ramFree),
    ramAvg: dash(ramAvg),
    ramMax: dash(ramMax),
    vramTotal: vramCell(vramTotal),
    vramFree: vramCell(vramFree),
    vramAvg: vramCell(vramAvg),
    vramMax: vramCell(vramMax),
  };
}

// Mirrors memoryCells' terminal/live fallback logic but returns raw numbers
// (nullish -> -1, sorts to one end) instead of display strings, for the
// header-click sort below. `results` may hold up to two rows for one idx
// (a pp row and a tg row from the same llama-bench process, see
// worker/src/index.ts's runSweepItem) -- picked apart by test_type below.
function mergedSortValue(item: RunItem, results: ResultRow[] | undefined, key: string): number | string {
  const anyResult = results?.[0];
  const ppResult = results?.find((r) => r.test_type === "pp");
  const tgResult = results?.find((r) => r.test_type === "tg");
  switch (key) {
    case "idx":
      return item.idx;
    case "test":
      return testTypeLabel(item.n_prompt, item.n_gen);
    case "n_prompt":
      return item.n_prompt;
    case "n_gen":
      return item.n_gen;
    case "threads":
      return item.n_threads;
    case "ngl":
      return item.n_gpu_layers;
    case "layers_loaded":
      return anyResult?.gpu_layers_loaded ?? -1;
    case "batch":
      return item.batch_size;
    case "ubatch":
      return item.ubatch_size;
    case "ctk":
      return item.cache_type_k;
    case "ctv":
      return item.cache_type_v;
    case "fa":
      return item.flash_attn;
    case "mtp":
      return item.mtp;
    case "ngld":
      return item.n_gpu_layers_draft;
    case "layers_loaded_draft":
      return anyResult?.gpu_layers_loaded_draft ?? -1;
    case "pp_tps":
      return ppResult?.avg_tps ?? -1;
    case "tg_tps":
      return tgResult?.avg_tps ?? -1;
    case "ttft":
      return ppResult && ppResult.n_prompt > 0 ? ppResult.n_prompt / ppResult.avg_tps : -1;
    case "ram_free":
      return anyResult?.ram_free_before_mib ?? item.ram_free_before_mib ?? -1;
    case "ram_avg":
      return anyResult?.ram_avg_mib ?? item.ram_avg_mib ?? item.ram_mib ?? -1;
    case "ram_max":
      return anyResult?.ram_peak_mib ?? item.ram_peak_mib ?? -1;
    case "vram_total":
      return anyResult?.gpu_memory_total_mb ?? -1;
    case "vram_free":
      return anyResult?.vram_free_before_mib ?? item.vram_free_before_mib ?? -1;
    case "vram_avg":
      return anyResult?.vram_avg_mib ?? item.vram_avg_mib ?? item.vram_mib ?? -1;
    case "vram_max":
      return anyResult?.vram_peak_mib ?? item.vram_peak_mib ?? -1;
    default:
      return item.idx;
  }
}

export function RunDetail() {
  const { id = "" } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [results, setResults] = useState<ResultRow[]>([]);
  const [items, setItems] = useState<RunItem[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [paused, setPaused] = useState(false);
  const [controlPending, setControlPending] = useState(false);
  const [controlError, setControlError] = useState("");
  // Subheader hardware line (arch/cpu/gpu/os) -- not on Run itself, so
  // fetched separately from the same per-worker endpoint WorkerCard/NewRun
  // already use. Best-effort: stays undefined (line just omits those bits)
  // if the worker's unreachable, same as everywhere else this is fetched.
  const [workerInfo, setWorkerInfo] = useState<WorkerLlamaCppInfo | undefined>(undefined);
  // Only fetched when a run actually named an MTP draft model -- resolves
  // config.mtp_model_id (an id, not a filename) for the subheader's "draft
  // model: X" bit. The full registry is more than this needs, but there's no
  // get-one-model-by-id endpoint and this list is already fetched the same
  // way by Models.tsx/NewRun.tsx.
  const [models, setModels] = useState<Model[]>([]);
  const timerRef = useRef<number | null>(null);
  // null = natural idx order (the sweep's own ordering) -- a running sweep's
  // default view shouldn't reshuffle mid-run; sorting only kicks in once the
  // user clicks a header.
  const [sort, setSort] = useState<SortState | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  // Freezes each item's last live pp/tg reading, keyed by idx -- the worker
  // reuses one `live_tps` column for whichever phase is currently running,
  // so by the time an item moves from "processing" to "generating" the pp
  // reading is already overwritten server-side and gone from `items`. The
  // real averaged pp result only exists once the *whole* item (both phases)
  // finishes (see shared/types.ts's RunItemTerminalInput -- results are only
  // sent in the terminal update, never mid-item), so without this the PP
  // column would blank out for the entire tg phase instead of holding the
  // last thing it actually measured. Mutated directly during render (not in
  // an effect) so it's never a poll cycle behind what's on screen.
  const lastLiveRef = useRef<Record<number, { pp?: number; tg?: number }>>({});
  useEffect(() => {
    lastLiveRef.current = {};
  }, [id]);

  // Keeps polling while the run is still in flight so results/status show up
  // without a manual refresh once the worker reports each item -- same
  // pattern as the Runs list page. 2s (not 5s) while running so the live
  // test list doesn't look stale against the worker's own ~2s tick cadence.
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const data = await api.getRun(id);
      if (cancelled) return;
      setRun(data.run);
      setResults(data.results ?? []);
      setItems(data.items ?? []);
      setPaused(data.paused ?? false);
      // Keep polling while scheduled too -- a scheduled run flips to
      // running on its own once whatever's ahead of it on that worker
      // finishes (see server/src/routes/runs.ts's dispatchScheduledRun),
      // with no other signal telling this page to refresh.
      if (data.run.status === "running" || data.run.status === "scheduled") {
        timerRef.current = window.setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [id]);

  useEffect(() => {
    if (!run?.worker_name) return;
    let cancelled = false;
    api
      .getWorkerLlamaCpp(run.worker_name)
      .then((info) => {
        if (!cancelled) setWorkerInfo(info);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [run?.worker_name]);

  useEffect(() => {
    if (!run?.config.mtp_model_id) return;
    let cancelled = false;
    api
      .listModels()
      .then((list) => {
        if (!cancelled) setModels(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [run?.config.mtp_model_id]);

  async function handlePauseToggle() {
    if (!run) return;
    setControlPending(true);
    setControlError("");
    try {
      if (paused) {
        await api.resumeWorker(run.worker_name);
        setPaused(false);
      } else {
        await api.pauseWorker(run.worker_name);
        setPaused(true);
      }
    } catch (err) {
      setControlError(err instanceof Error ? err.message : String(err));
    } finally {
      setControlPending(false);
    }
  }

  async function handleStop() {
    if (!run) return;
    const confirmed = window.confirm(
      "Stop this run? The test currently in flight is killed immediately; every remaining sweep combination is marked cancelled. This can't be undone."
    );
    if (!confirmed) return;
    setControlPending(true);
    setControlError("");
    try {
      await api.stopWorker(run.worker_name);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : String(err));
    } finally {
      setControlPending(false);
    }
  }

  useEffect(() => {
    if (run?.status !== "running") return;
    const tickId = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(tickId);
  }, [run?.status]);

  // Every distinct reason any VRAM figure in this run is approximate rather
  // than an exact per-process reading -- surfaced as a standing banner above
  // the table (see below) rather than relying solely on each cell's hover
  // tooltip, which is easy to miss entirely.
  const vramAccuracyNotes = new Set<string>();
  for (const r of results) {
    for (const t of [
      gpuMemoryAccuracyTitle(r.gpu_memory_total_accuracy, r.gpu_memory_total_source),
      gpuMemoryAccuracyTitle(r.gpu_memory_free_start_accuracy, r.gpu_memory_free_start_source),
      gpuMemoryAccuracyTitle(r.gpu_memory_model_avg_accuracy, r.gpu_memory_model_avg_source),
      gpuMemoryAccuracyTitle(r.gpu_memory_model_peak_accuracy, r.gpu_memory_model_peak_source),
    ]) {
      if (t) vramAccuracyNotes.add(t);
    }
  }

  // A single sweep item can now yield up to two results (a pp row and a tg
  // row from one llama-bench process, see worker/src/index.ts's
  // runSweepItem) sharing the same idx -- group rather than overwrite.
  // Hoisted above chartConfig so both the chart and the merged table below
  // group on the same idx keys.
  const resultByIdx = new Map<number, ResultRow[]>();
  for (const r of results) {
    const existing = resultByIdx.get(r.idx);
    if (existing) existing.push(r);
    else resultByIdx.set(r.idx, [r]);
  }
  const chartIdxs = [...resultByIdx.keys()].sort((a, b) => a - b);
  // Grouped bar chart: one category per sweep combination (idx), with PP
  // and TG bars paired side by side instead of interleaved same-colored
  // bars. barPercentage/categoryPercentage are fixed fractions of whatever
  // width Chart.js allocates per category, not measured pixel widths, so
  // they're correct on the very first paint and always fit the container
  // regardless of sweep size -- categoryPercentage narrows as the test
  // count grows so labels/bars don't get crushed, and stays capped for
  // small sweeps so a couple of bars don't stretch across the whole chart.
  const categoryPercentage = chartIdxs.length <= 3 ? 0.35 : chartIdxs.length <= 8 ? 0.55 : 0.85;
  const chartConfig: ChartConfiguration<"bar"> | null = chartIdxs.length
    ? {
        type: "bar",
        data: {
          labels: chartIdxs.map((idx) => {
            const r = resultByIdx.get(idx)![0];
            return `#${idx + 1} p${r.n_prompt} n${r.n_gen} t${r.n_threads} ngl${r.n_gpu_layers}`;
          }),
          datasets: [
            {
              label: "PP tok/s",
              data: chartIdxs.map(
                (idx) => resultByIdx.get(idx)!.find((r) => r.test_type === "pp")?.avg_tps ?? null
              ),
              backgroundColor: "#6ea8fe",
              barPercentage: 0.9,
              categoryPercentage,
            },
            {
              label: "TG tok/s",
              data: chartIdxs.map(
                (idx) => resultByIdx.get(idx)!.find((r) => r.test_type === "tg")?.avg_tps ?? null
              ),
              backgroundColor: "#86ef9e",
              barPercentage: 0.9,
              categoryPercentage,
            },
          ],
        },
        options: {
          maintainAspectRatio: false,
          responsive: true,
          scales: { y: { beginAtZero: true } },
          plugins: { legend: { display: true } },
        },
      }
    : null;

  const itemsCompleted = items.filter((it) => TERMINAL_ITEM_STATUSES.has(it.status)).length;
  // Items report in idx order (one llama-bench process at a time -- see
  // worker/src/index.ts's per-item loop), so the first non-terminal one is
  // whichever is currently in flight.
  const currentItem = items.find((it) => !TERMINAL_ITEM_STATUSES.has(it.status) && it.status !== "queued");

  // Mutated directly during render (not in an effect) so it's never a poll
  // cycle behind -- must run before the ETA calc and table cells below, both
  // of which read it. See lastLiveRef's own declaration above for why this
  // freeze is needed at all.
  for (const it of items) {
    if (it.status === "processing" && it.live_tps != null) {
      lastLiveRef.current[it.idx] = { ...lastLiveRef.current[it.idx], pp: it.live_tps };
    } else if (it.status === "generating" && it.live_tps != null) {
      lastLiveRef.current[it.idx] = { ...lastLiveRef.current[it.idx], tg: it.live_tps };
    }
  }

  // "Runs" in this app's own vocabulary means repeats (-r), not sweep
  // combinations -- see buildProgressUnits' comment in StatusPill.tsx. One
  // test item is one combination executed `repeats` times.
  const repeats = run?.config.sweep.repeats ?? 1;
  const totalRuns = items.length * repeats;

  // Estimated time to finish every remaining run, recalculated on every
  // poll. Preferably sourced from a stable pp/tg tok/s reading (from
  // `results`), which only exists once at least one item has finished --
  // before that, falls back to the currently-running item's own live/frozen
  // reading (see below), since a sweep with a single combination and many
  // repeats would otherwise never cross that gate before the run is
  // basically over.
  let ppSpeedSum = 0;
  let ppSpeedCount = 0;
  let tgSpeedSum = 0;
  let tgSpeedCount = 0;
  for (const rs of resultByIdx.values()) {
    const pp = rs.find((r) => r.test_type === "pp");
    const tg = rs.find((r) => r.test_type === "tg");
    if (pp) {
      ppSpeedSum += pp.avg_tps;
      ppSpeedCount++;
    }
    if (tg) {
      tgSpeedSum += tg.avg_tps;
      tgSpeedCount++;
    }
  }
  const avgPpTps = ppSpeedCount > 0 ? ppSpeedSum / ppSpeedCount : null;
  const avgTgTps = tgSpeedCount > 0 ? tgSpeedSum / tgSpeedCount : null;

  // Fallback: no item has finished yet, so borrow the currently-running
  // item's own live reading (the exact value its PP/TG cells are showing --
  // `it.live_tps` for whichever phase is active right now, `lastLiveRef`'s
  // frozen snapshot for the other one). A single in-flight sample is
  // noisier than a multi-repeat average, so this is only used when no real
  // average exists for that phase, and the UI marks it as a rough estimate.
  let usingLiveFallback = false;
  let effectivePpTps = avgPpTps;
  let effectiveTgTps = avgTgTps;
  if (avgPpTps == null && avgTgTps == null && currentItem) {
    const livePp =
      currentItem.status === "processing" && currentItem.live_tps != null
        ? currentItem.live_tps
        : lastLiveRef.current[currentItem.idx]?.pp;
    const liveTg =
      currentItem.status === "generating" && currentItem.live_tps != null
        ? currentItem.live_tps
        : lastLiveRef.current[currentItem.idx]?.tg;
    if (livePp != null || liveTg != null) {
      effectivePpTps = livePp ?? null;
      effectiveTgTps = liveTg ?? null;
      usingLiveFallback = true;
    }
  }

  // ttft (prompt tokens ÷ measured pp speed) plus generation time (gen
  // tokens ÷ measured tg speed) -- per-item rather than a flat average
  // seconds-per-run, since sweep combinations can have very different
  // n_prompt/n_gen values.
  const estimateRepeatSeconds = (it: RunItem): number => {
    let s = 0;
    if (it.n_prompt > 0 && effectivePpTps) s += it.n_prompt / effectivePpTps;
    if (it.n_gen > 0 && effectiveTgTps) s += it.n_gen / effectiveTgTps;
    return s;
  };
  let etaMs: number | null = null;
  if (effectivePpTps != null || effectiveTgTps != null) {
    let remainingSeconds = 0;
    for (const it of items) {
      if (TERMINAL_ITEM_STATUSES.has(it.status)) continue;
      let remainingRepeats = repeats;
      if (RUNNING_ITEM_STATUSES.has(it.status)) {
        const match = REPEAT_PROGRESS_RE.exec(it.detail ?? "");
        const doneRepeats = match ? Math.max(0, Number(match[1]) - 1) : 0;
        remainingRepeats = Math.max(0, repeats - doneRepeats);
      }
      remainingSeconds += remainingRepeats * estimateRepeatSeconds(it);
    }
    etaMs = remainingSeconds * 1000;
  }

  const statusCounts = useMemo(() => {
    const c: Record<StatusFilter, number> = {
      all: items.length,
      queued: 0,
      running: 0,
      done: 0,
      failed: 0,
      cancelled: 0,
    };
    for (const it of items) {
      if (it.status === "queued") c.queued++;
      else if (RUNNING_ITEM_STATUSES.has(it.status)) c.running++;
      else if (it.status === "done") c.done++;
      else if (it.status === "cancelled") c.cancelled++;
      else c.failed++;
    }
    return c;
  }, [items]);

  const visibleItems = useMemo(() => {
    const filtered = items.filter((it) => matchesStatusFilter(it.status, statusFilter));
    if (!sort) return filtered;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = mergedSortValue(a, resultByIdx.get(a.idx), sort.key);
      const bv = mergedSortValue(b, resultByIdx.get(b.idx), sort.key);
      if (typeof av === "number" && typeof bv === "number") return dir * (av - bv);
      return dir * String(av).localeCompare(String(bv));
    });
  }, [items, statusFilter, sort, resultByIdx]);

  // Subheader hardware line: architecture/OS come from the worker's own
  // /llama-cpp+/hardware fetch (workerInfo, best-effort -- omitted if that
  // worker's unreachable); VRAM/RAM totals come from this run's own results
  // instead (system_memory_total_mb/gpu_memory_total_mb are static hardware
  // facts sampled once per test, see ResultRow's own doc comments) since
  // WorkerLlamaCppInfo's HardwareInfo has no memory-size fields at all.
  // backend_device_name is preferred over hw.gpu for the GPU label since
  // it's Run's own two-tier-detected value for whatever actually ran this
  // benchmark, not just whatever's currently in the box.
  const hw = workerInfo?.hardware;
  const cpuLabel = hw?.cpu.brand || hw?.cpu.manufacturer || undefined;
  const gpuDeviceLabel =
    run?.backend_device_name || (hw?.gpu.length ? hw.gpu.map((g) => g.model).join(", ") : undefined);
  const gpuLabel = gpuDeviceLabel && run ? `${gpuDeviceLabel} (${run.llama_cpp_backend})` : gpuDeviceLabel;
  const vramTotalMib = results.find((r) => r.gpu_memory_total_mb != null)?.gpu_memory_total_mb;
  const ramTotalMib = results.find((r) => r.system_memory_total_mb != null)?.system_memory_total_mb;
  const hardwareBits = [
    workerInfo?.arch,
    cpuLabel ? `${cpuLabel}${hw?.cpu.cores ? ` (${hw.cpu.cores} threads)` : ""}` : undefined,
    gpuLabel,
    vramTotalMib != null ? `${(vramTotalMib / 1024).toFixed(1)} GB VRAM` : undefined,
    ramTotalMib != null ? `${(ramTotalMib / 1024).toFixed(1)} GB RAM` : undefined,
    workerInfo?.platform,
  ].filter((v): v is string => Boolean(v));

  const draftModel = run?.config.mtp_model_id ? models.find((m) => m.id === run.config.mtp_model_id) : undefined;

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">
        Run <code className="text-lg text-muted">{shortId(id)}</code>
      </h1>
      {run && (
        <div className="mt-2 flex flex-col gap-1 text-sm text-muted">
          <p>
            Worker: <b className="text-fg">{run.worker_name}</b>
            {hardwareBits.length > 0 && ` · ${hardwareBits.join(", ")}`}
          </p>
          <p>Llama.cpp version: {run.llama_cpp_build}</p>
          <p>
            Model: <span className="text-fg">{run.model_filename || `${shortId(run.model_id)}…`}</span>
            {draftModel && ` (draft model: ${draftModel.filename})`}
          </p>
          <p className="flex flex-wrap items-center gap-2">
            <RunStatusPill status={run.status} />
            {run.status === "running" && <span>· {formatElapsed(now - run.started_at)} elapsed</span>}
            {run.status === "running" && paused && <span className="text-warning">· paused</span>}
            {run.error && <span className="text-danger">· {run.error}</span>}
          </p>
        </div>
      )}

      {run?.status === "running" && (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handlePauseToggle}
            disabled={controlPending}
            title={
              paused
                ? "Resume before the next sweep combination"
                : "Pause before the next sweep combination -- the one currently in flight (including all its repeats) finishes first"
            }
            className="rounded-lg border border-border px-3 py-1.5 text-sm font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-50"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            type="button"
            onClick={handleStop}
            disabled={controlPending}
            className="rounded-lg border border-danger/40 px-3 py-1.5 text-sm font-semibold text-danger hover:bg-danger-bg disabled:opacity-50"
          >
            Stop
          </button>
          {controlError && <span className="text-sm text-danger">{controlError}</span>}
        </div>
      )}

      {items.length > 0 && (
        <section className="mt-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
              Tests ({itemsCompleted} of {items.length}) · Total runs: {totalRuns}
            </h2>
            <div className="flex flex-wrap items-baseline gap-3">
              {run?.status === "running" && etaMs != null && (
                <span
                  className="text-xs text-muted"
                  title={
                    usingLiveFallback
                      ? "Rough estimate from the current test's live in-progress speed (no test has finished yet to average) -- will firm up once the first test completes"
                      : "Estimated from the average measured pp/tg speed and each remaining test's token counts -- recalculated as more tests finish"
                  }
                >
                  Est. remaining{usingLiveFallback ? " (rough)" : ""}: {formatElapsed(etaMs)}
                </span>
              )}
              {currentItem && (
                <span className="text-xs text-muted">
                  Test {currentItem.idx + 1} of {items.length} — {describeItemPhase(currentItem)}
                </span>
              )}
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {STATUS_FILTERS.map((f) => (
              <button
                key={f.value}
                type="button"
                onClick={() => setStatusFilter(f.value)}
                className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                  statusFilter === f.value
                    ? "border-accent/40 bg-accent/15 text-accent"
                    : "border-border text-muted hover:border-accent/40 hover:text-fg"
                }`}
              >
                {f.label} ({statusCounts[f.value]})
              </button>
            ))}
          </div>
          {vramAccuracyNotes.size > 0 && (
            <div className="mt-2 flex items-start gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs text-muted">
              <IconInfo width={13} height={13} className="mt-0.5 flex-none" />
              <p>
                VRAM columns marked <IconInfo width={11} height={11} className="inline -translate-y-px" /> are
                approximate: {[...vramAccuracyNotes].join(" ")}
              </p>
            </div>
          )}
          {visibleItems.length === 0 ? (
            <p className="mt-3 text-sm text-muted">No tests match the current filter.</p>
          ) : (
          <div className="mt-2 overflow-x-auto rounded-xl border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border uppercase tracking-wide text-muted">
                  {MERGED_COLUMN_DEFS.map((c) => (
                    <Th
                      key={c.label}
                      label={c.label}
                      description={c.description}
                      sortKey={c.sortKey}
                      sort={sort}
                      onSort={c.sortKey ? (k) => setSort((s) => toggleSort(s, k)) : undefined}
                      className={`${c.padX ?? "!px-1"} !py-1.5`}
                    />
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {visibleItems.map((it) => {
                  const itemResults = resultByIdx.get(it.idx);
                  const anyResult = itemResults?.[0];
                  const ppResult = itemResults?.find((r) => r.test_type === "pp");
                  const tgResult = itemResults?.find((r) => r.test_type === "tg");
                  const mem = memoryCells(it, anyResult);
                  const isTerminal = TERMINAL_ITEM_STATUSES.has(it.status);
                  const detailText = it.error || (isTerminal ? it.detail || "" : describeItemPhase(it));
                  const showLivePp = !isTerminal && it.status === "processing" && it.live_tps != null;
                  const showLiveTg = !isTerminal && it.status === "generating" && it.live_tps != null;
                  // Last live reading seen for a phase that's since moved on
                  // (pp once tg starts running, or either while the item is
                  // finalizing) -- see lastLiveRef above for why this can't
                  // just be read off `it.live_tps` at this point.
                  const frozenPp = lastLiveRef.current[it.idx]?.pp;
                  const frozenTg = lastLiveRef.current[it.idx]?.tg;
                  const ttftSeconds =
                    ppResult && ppResult.n_prompt > 0 ? ppResult.n_prompt / ppResult.avg_tps : null;
                  // spec_drafted/spec_accepted only ever live on the tg row of
                  // an MTP item (see fmtSpecDecode's own comment) -- the full
                  // "accepted/drafted (rate%)" string goes in the tooltip,
                  // the bare percentage in the cell itself to keep the column
                  // narrow across the (much more common) non-MTP rows.
                  const mtpAcceptTitle =
                    it.mtp === "on" ? fmtSpecDecode(tgResult?.spec_drafted, tgResult?.spec_accepted) : undefined;
                  const mtpAcceptPct =
                    tgResult?.spec_drafted && tgResult?.spec_accepted != null
                      ? (tgResult.spec_accepted / tgResult.spec_drafted) * 100
                      : null;
                  // Repeat-progress dots for this specific set of parameters,
                  // one per repeat regardless of the item's own status --
                  // see buildItemRepeatUnits (queued = all grey, done = all
                  // solid, running = live per-repeat progress, etc.).
                  const statusUnits = buildItemRepeatUnits(it, repeats);
                  return (
                    <Fragment key={it.id}>
                    <tr className="!border-b-0">
                      <td className="px-1 py-1.5 align-top text-muted">{it.idx + 1}</td>
                      <td className={`px-1 py-1.5 align-top ${it.error ? "text-danger" : "text-muted"}`}>
                        {/* Fixed to fit "processing" (the longest single word ever
                            shown here) plus a hair of rendering buffer -- multi-word
                            phrases like "generating tokens" wrap onto a second line
                            via whitespace-normal instead of stretching the column;
                            break-words is just a fallback for the rare longer outlier
                            (an error message, or "benchmarking" on older
                            --progress-less llama-bench builds). */}
                        <div className="w-[11ch] whitespace-normal break-words" title={detailText}>
                          {detailText.length > 120 ? `${detailText.slice(0, 120)}…` : detailText}
                        </div>
                      </td>
                      <td className="px-1 py-1.5 text-fg">{testTypeLabel(it.n_prompt, it.n_gen)}</td>
                      <td className="pl-1 pr-0.5 py-1.5 text-muted">{it.n_prompt}</td>
                      <td className="pl-0.5 pr-1 py-1.5 text-muted">{it.n_gen}</td>
                      <td className="pl-1 pr-0.5 py-1.5 text-muted">{it.n_threads}</td>
                      <td className="pl-0.5 pr-1 py-1.5 text-muted">{it.n_gpu_layers}</td>
                      <td className="px-1 py-1.5 text-muted">
                        {anyResult?.gpu_layers_loaded != null && anyResult?.total_model_layers != null
                          ? `${anyResult.gpu_layers_loaded}/${anyResult.total_model_layers}`
                          : isTerminal
                            ? "n/a"
                            : "—"}
                      </td>
                      <td className="px-1 py-1.5 text-muted">{it.batch_size}</td>
                      <td className="px-1 py-1.5 text-muted">{it.ubatch_size}</td>
                      <td className="px-1 py-1.5 text-muted">{it.cache_type_k}</td>
                      <td className="px-1 py-1.5 text-muted">{it.cache_type_v}</td>
                      <td className="px-1 py-1.5 text-muted">{formatFlashAttn(it.flash_attn)}</td>
                      <td className="px-1 py-1.5 text-muted">
                        {it.mtp === "on" ? (
                          <span title={mtpAcceptTitle}>
                            MTP{mtpAcceptPct != null ? ` ${mtpAcceptPct.toFixed(1)}%` : ""}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="px-1 py-1.5 text-muted">{it.mtp === "on" ? it.n_gpu_layers_draft : "—"}</td>
                      <td className="px-1 py-1.5 text-muted">
                        {it.mtp !== "on"
                          ? "—"
                          : anyResult?.gpu_layers_loaded_draft != null && anyResult?.total_model_layers_draft != null
                            ? `${anyResult.gpu_layers_loaded_draft}/${anyResult.total_model_layers_draft}`
                            : isTerminal
                              ? "n/a"
                              : "—"}
                      </td>
                      <td className="px-1 py-1.5">
                        {showLivePp ? (
                          <span className="text-accent">{it.live_tps!.toFixed(1)}</span>
                        ) : ppResult ? (
                          <span
                            className={ppResult.suspect_count ? "text-warning" : "text-fg"}
                            title={
                              resultSuspectTitle(ppResult) ??
                              `avg ${ppResult.avg_tps.toFixed(2)} ± ${ppResult.stddev_tps.toFixed(2)} tok/s`
                            }
                          >
                            {ppResult.avg_tps.toFixed(2)}
                            {ppResult.suspect_count ? " ⚠" : ""}
                          </span>
                        ) : !isTerminal && frozenPp != null ? (
                          <span
                            className="text-accent/70"
                            title="Last measured prompt-processing speed -- generation is still running, final average isn't in yet"
                          >
                            {frozenPp.toFixed(1)}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="px-1 py-1.5">
                        {showLiveTg ? (
                          <span className="text-accent">{it.live_tps!.toFixed(1)}</span>
                        ) : tgResult ? (
                          <span
                            className={tgResult.suspect_count ? "text-warning" : "text-fg"}
                            title={
                              resultSuspectTitle(tgResult) ??
                              `avg ${tgResult.avg_tps.toFixed(2)} ± ${tgResult.stddev_tps.toFixed(2)} tok/s`
                            }
                          >
                            {tgResult.avg_tps.toFixed(2)}
                            {tgResult.suspect_count ? " ⚠" : ""}
                          </span>
                        ) : !isTerminal && frozenTg != null ? (
                          <span
                            className="text-accent/70"
                            title="Last measured generation speed -- finalizing this test, final average isn't in yet"
                          >
                            {frozenTg.toFixed(1)}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td
                        className="px-1 py-1.5 text-muted"
                        title="Derived: prompt tokens ÷ PP speed -- not a directly measured request latency"
                      >
                        {ttftSeconds != null ? `${ttftSeconds.toFixed(2)}s` : "—"}
                      </td>
                      <td className="px-1 py-1.5 text-muted">{mem.ramFree}</td>
                      <td className="px-1 py-1.5 text-muted">{mem.ramAvg}</td>
                      <td className="px-1 py-1.5 text-muted">{mem.ramMax}</td>
                      <td className="px-1 py-1.5 text-muted">
                        <span className="inline-flex items-center gap-0.5">
                          {mem.vramTotal}
                          <AccuracyIcon
                            title={gpuMemoryAccuracyTitle(anyResult?.gpu_memory_total_accuracy, anyResult?.gpu_memory_total_source)}
                          />
                        </span>
                      </td>
                      <td className="px-1 py-1.5 text-muted">
                        <span className="inline-flex items-center gap-0.5">
                          {mem.vramFree}
                          <AccuracyIcon
                            title={gpuMemoryAccuracyTitle(anyResult?.gpu_memory_free_start_accuracy, anyResult?.gpu_memory_free_start_source)}
                          />
                        </span>
                      </td>
                      <td className="px-1 py-1.5 text-muted">
                        <span className="inline-flex items-center gap-0.5">
                          {mem.vramAvg}
                          <AccuracyIcon
                            title={gpuMemoryAccuracyTitle(anyResult?.gpu_memory_model_avg_accuracy, anyResult?.gpu_memory_model_avg_source)}
                          />
                        </span>
                      </td>
                      <td className="px-1 py-1.5 text-muted">
                        <span className="inline-flex items-center gap-0.5">
                          {mem.vramMax}
                          <AccuracyIcon
                            title={gpuMemoryAccuracyTitle(anyResult?.gpu_memory_model_peak_accuracy, anyResult?.gpu_memory_model_peak_source)}
                          />
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td colSpan={MERGED_COLUMN_DEFS.length} className="px-2 pb-1.5 pt-0">
                        <StatusCircleStrip units={statusUnits} />
                      </td>
                    </tr>
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          )}
        </section>
      )}

      {/* Historical runs from before run_items existed have results but no
          items -- fall back to a plain results table rather than showing
          nothing for them. */}
      {items.length === 0 && results.length > 0 && (
        <section className="mt-6">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Results</h2>
          <div className="mt-2 overflow-x-auto rounded-xl border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border uppercase tracking-wide text-muted">
                   {["test", "n_prompt", "n_gen", "threads", "ngl", "batch", "ubatch", "ctk", "ctv", "fa", "Tok/s", "stddev", "ram MiB", "vram MiB"].map(
                    (h) => (
                      <Th key={h} label={h} description={LEGACY_COLUMN_DESCRIPTIONS[h]} className="!px-3 !py-2" />
                    )
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {results.map((r) => (
                  <tr key={r.id}>
                    <td className="px-3 py-2 text-fg">{r.test_type}</td>
                    <td className="px-3 py-2 text-muted">{r.n_prompt}</td>
                    <td className="px-3 py-2 text-muted">{r.n_gen}</td>
                    <td className="px-3 py-2 text-muted">{r.n_threads}</td>
                    <td className="px-3 py-2 text-muted">{r.n_gpu_layers}</td>
                    <td className="px-3 py-2 text-muted">{r.batch_size}</td>
                    <td className="px-3 py-2 text-muted">{r.ubatch_size}</td>
                    <td className="px-3 py-2 text-muted">{r.cache_type_k}</td>
                    <td className="px-3 py-2 text-muted">{r.cache_type_v}</td>
                    <td className="px-3 py-2 text-muted">{formatFlashAttn(r.flash_attn)}</td>
                    <td className="px-3 py-2 text-fg">{r.avg_tps.toFixed(2)}</td>
                    <td className="px-3 py-2 text-muted">{r.stddev_tps.toFixed(2)}</td>
                    <td className="px-3 py-2 text-muted">{r.ram_peak_mib}</td>
                    <td className="px-3 py-2 text-muted">{r.vram_peak_mib === null ? "n/a" : r.vram_peak_mib}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {results.length > 0 && (
        <>
          <div className="relative mt-6 h-72 rounded-xl border border-border bg-surface p-4">
            {chartConfig && <Chart config={chartConfig} />}
          </div>
          <p className="mt-3 text-sm">
            <a href={api.exportUrl("csv", [id])} download className="text-accent hover:underline">
              Export run CSV
            </a>
          </p>
        </>
      )}
    </div>
  );
}
