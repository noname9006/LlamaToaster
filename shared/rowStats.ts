// One formula for every speed row's statistics, used by the worker when it
// builds a row and by the server when it stores one (RESULT_PROVENANCE_PLAN.md
// Phase 3). Before this, three worker paths each computed mean/stddev and
// sample_count their own way, and the server stored whatever arrived.

/** Mean and sample (n-1) standard deviation -- llama-bench's own formula. */
export function meanAndSampleStddev(values: readonly number[]): { mean: number; stddev: number } {
  if (values.length === 0) return { mean: 0, stddev: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (values.length < 2) return { mean, stddev: 0 };
  const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (values.length - 1);
  return { mean, stddev: Math.sqrt(variance) };
}

/**
 * Per-repetition rates from llama-bench's JSON `samples_ns`. Its `samples_ts`
 * are printed rounded (about six significant digits), so a mean over them
 * misses `avg_ts` by ~1e-4 relative; the integer nanosecond timings reproduce
 * `avg_ts`/`stddev_ts` to the precision they are printed with. llama-bench
 * counts n_prompt + n_gen tokens per repetition (depth tokens are not timed).
 */
export function llamaBenchSampleRates(nPrompt: number, nGen: number, samplesNs: unknown): number[] | undefined {
  if (!Array.isArray(samplesNs) || samplesNs.length === 0) return undefined;
  const tokens = nPrompt + nGen;
  if (!(tokens > 0)) return undefined;
  const rates: number[] = [];
  for (const ns of samplesNs) {
    if (typeof ns !== "number" || !Number.isFinite(ns) || ns <= 0) return undefined;
    rates.push((tokens * 1e9) / ns);
  }
  return rates;
}

// llama-bench prints avg_ts/stddev_ts with six decimals, so its own figures
// carry up to 5e-7 of rounding; a value computed from the same doubles on both
// sides agrees exactly. Anything past this is a different formula or input.
const ABS_TOLERANCE = 1e-5;
const REL_TOLERANCE = 1e-6;

export function statsAgree(a: number, b: number): boolean {
  return Math.abs(a - b) <= ABS_TOLERANCE + REL_TOLERANCE * Math.max(Math.abs(a), Math.abs(b));
}

/** What a row needs for its statistics to be reconciled -- a subset of IngestResultInput. */
export interface RowStatsInput {
  avg_tps: number;
  stddev_tps: number;
  sample_count?: number;
  suspect_count?: number;
  suspect_samples?: number[];
  clean_samples?: number[];
  repeats_requested?: number;
}

export interface ReconciledRowStats {
  avg_tps: number;
  stddev_tps: number;
  /** Repeats that produced any computable reading (clean + suspect). */
  sample_count: number | null;
  suspect_count: number | null;
  /** The worker's own figures, kept only when they disagree with the recomputation. */
  worker_avg_tps: number | null;
  worker_stddev_tps: number | null;
  /** 1 = the worker's figures disagreed and were replaced; 0 = they agreed; null = nothing to recompute from. */
  stats_mismatch: 0 | 1 | null;
  /** 1 = every requested repeat produced a reading; 0 = some produced none; null = the worker sent no request count. */
  repeats_complete: 0 | 1 | null;
}

/**
 * The server's view of a row's statistics. Recomputes mean and stddev from the
 * clean samples when the worker sent them; an all-suspect row (no clean
 * sample) keeps the worker's figures, because what its average stands for
 * differs by path (serverBench averages the flagged values, the fill curve
 * reports 0) and is already flagged through suspect_count.
 */
export function reconcileRowStats(row: RowStatsInput): ReconciledRowStats {
  const suspectCount = row.suspect_samples?.length ?? row.suspect_count ?? null;
  const clean = row.clean_samples;
  let avg = row.avg_tps;
  let stddev = row.stddev_tps;
  let workerAvg: number | null = null;
  let workerStddev: number | null = null;
  let mismatch: 0 | 1 | null = null;
  let sampleCount: number | null = row.sample_count ?? null;

  if (clean !== undefined) {
    sampleCount = clean.length + (suspectCount ?? 0);
    if (clean.length > 0) {
      const recomputed = meanAndSampleStddev(clean);
      if (statsAgree(recomputed.mean, row.avg_tps) && statsAgree(recomputed.stddev, row.stddev_tps)) {
        mismatch = 0;
      } else {
        mismatch = 1;
        workerAvg = row.avg_tps;
        workerStddev = row.stddev_tps;
      }
      avg = recomputed.mean;
      stddev = recomputed.stddev;
    }
  }

  let complete: 0 | 1 | null = null;
  if (row.repeats_requested !== undefined && row.repeats_requested > 0 && sampleCount !== null) {
    complete = sampleCount >= row.repeats_requested ? 1 : 0;
  }

  return {
    avg_tps: avg,
    stddev_tps: stddev,
    sample_count: sampleCount,
    suspect_count: suspectCount,
    worker_avg_tps: workerAvg,
    worker_stddev_tps: workerStddev,
    stats_mismatch: mismatch,
    repeats_complete: complete,
  };
}

/**
 * "3 of 5 repeats produced a reading" for a row whose repeats did not all
 * produce one; undefined for a complete row or one predating the fields.
 */
export function incompleteRepeatsNote(row: {
  repeats_complete?: boolean | null;
  repeats_requested?: number | null;
  sample_count?: number | null;
}): string | undefined {
  if (row.repeats_complete !== false || row.repeats_requested == null) return undefined;
  const got = row.sample_count ?? 0;
  return `${got} of ${row.repeats_requested} repeat(s) produced a reading; the others produced none`;
}
