import type { ResultRow, SweepConfig, TestItem } from "../../types";
import type { SpeedCandidate, SpeedReading, SpeedRunSpec, ThreadSweepItem, ThreadSweepSpec } from "../../../../shared/speedRun.js";
import type { FlowRun } from "./useOptimizeFlow";

// Turning a flow run's items/results into the numbers the cards show.

/** The one-combination grid every flow run carries; the spec decides what
 * actually runs, the server only needs a valid template item. */
export function templateSweep(repeats: number): Omit<SweepConfig, "model_id"> {
  return {
    n_prompt: [512],
    n_gen: [128],
    threads: [-1],
    n_gpu_layers: [0],
    batch_size: [2048],
    ubatch_size: [512],
    cache_type_k: ["f16"],
    cache_type_v: ["f16"],
    flash_attn: ["on"],
    mtp: ["off"],
    n_gpu_layers_draft: [0],
    n_cpu_moe: [0],
    repeats: Math.max(1, Math.min(25, repeats)),
  };
}

// llama-bench rows carry no per-repeat count of their own -- every repeat it
// was asked for produced the mean -- so the run's repeats stand in.
function reading(r: ResultRow | undefined, repeats: number): SpeedReading | null {
  if (!r || !Number.isFinite(r.avg_tps)) return null;
  const samples = r.sample_count ?? (r.repeat_samples?.length || repeats || 1);
  return { mean: r.avg_tps, stddev: r.stddev_tps ?? 0, samples };
}

export interface ItemOutcome {
  idx: number;
  status: TestItem["status"] | "pending";
  error: string | null;
  pp: SpeedReading | null;
  tg: SpeedReading | null;
  sharedPeakMib: number | null;
  vramPeakMib: number | null;
  ramPeakMib: number | null;
  ppRow?: ResultRow;
  tgRow?: ResultRow;
}

export function itemOutcomes(run: FlowRun, count: number, repeats: number): ItemOutcome[] {
  return Array.from({ length: count }, (_, idx) => {
    const item = run.items.find((i) => i.idx === idx);
    const rows = run.results.filter((r) => r.idx === idx);
    const ppRow = rows.find((r) => r.test_type === "pp");
    const tgRow = rows.find((r) => r.test_type === "tg");
    const any = ppRow ?? tgRow;
    return {
      idx,
      status: item?.status ?? "pending",
      error: item?.error ?? null,
      pp: reading(ppRow, repeats),
      tg: reading(tgRow, repeats),
      sharedPeakMib: any?.gpu_memory_shared_peak_mib ?? null,
      vramPeakMib: any?.vram_peak_mib ?? null,
      ramPeakMib: any?.ram_peak_mib ?? null,
      ppRow,
      tgRow,
    };
  });
}

export function speedSpecOf(run: FlowRun | null): SpeedRunSpec | null {
  return (run?.test.config as { speed_run?: SpeedRunSpec } | undefined)?.speed_run ?? null;
}

export function threadSpecOf(run: FlowRun | null): ThreadSweepSpec | null {
  return (run?.test.config as { thread_sweep?: ThreadSweepSpec } | undefined)?.thread_sweep ?? null;
}

export interface SpeedRunView {
  spec: SpeedRunSpec;
  rows: { candidate: SpeedCandidate; outcome: ItemOutcome }[];
}

export function speedRunView(run: FlowRun | null): SpeedRunView | null {
  const spec = speedSpecOf(run);
  if (!run || !spec) return null;
  const outcomes = itemOutcomes(run, spec.candidates.length, spec.repeats);
  return { spec, rows: spec.candidates.map((candidate, i) => ({ candidate, outcome: outcomes[i] })) };
}

export interface ThreadRunView {
  spec: ThreadSweepSpec;
  rows: { item: ThreadSweepItem; outcome: ItemOutcome; reading: SpeedReading | null }[];
}

export function threadRunView(run: FlowRun | null): ThreadRunView | null {
  const spec = threadSpecOf(run);
  if (!run || !spec) return null;
  const outcomes = itemOutcomes(run, spec.items.length, spec.repeats);
  return {
    spec,
    rows: spec.items.map((item, i) => ({ item, outcome: outcomes[i], reading: item.role === "t" ? outcomes[i].tg : outcomes[i].pp })),
  };
}

export function isFailed(status: ItemOutcome["status"]): boolean {
  return status === "failed" || status === "failed_oom" || status === "failed_timeout" || status === "failed_unsupported";
}

/** Spill worth mentioning: more than a quarter GB of the process's GPU memory in system RAM. */
export function spilledMib(o: ItemOutcome): number | null {
  return o.sharedPeakMib != null && o.sharedPeakMib > 256 ? o.sharedPeakMib : null;
}
