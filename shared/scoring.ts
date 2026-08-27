// BENCHMARKING_PLAN_V8.md §0.3 (scoring primitives) + M3 (goal-parameterized
// profile scoring). Pure functions over stored rows: changing goals re-runs
// this, never re-measurement.
//
// Everything here consumes rows the caller has already narrowed to a single
// scoring universe: runs of kind 'sweep', one method_version, and (N3)
// comparison members excluded -- trimmed comparison grids must not compete
// with Test A's full-grid profiles.

import type { CaveatFlag, TestType } from "./types.js";
import type { GoalsConfig } from "./goals.js";
import { WORKLOAD_WEIGHTS, defaultGoals } from "./goals.js";
import { configHash, type ConfigHashInput } from "./configHash.js";
import type { MaxCtxEstimate } from "./vramEstimate.js";

// The row shape scoring actually reads. Deliberately narrower than ResultRow
// so the worker, server and client can all feed it without importing the
// whole result model.
export interface ScoringRow {
  idx: number;
  test_type: TestType;
  n_prompt: number;
  n_gen: number;
  n_depth?: number | null;
  n_threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  n_gpu_layers_draft: number;
  n_cpu_moe: number;
  concurrency?: number | null;
  engine?: string | null;
  method_version?: number | null;
  avg_tps: number;
  stddev_tps: number;
  sample_count?: number;
  suspect_count?: number;
  repeat_samples?: number[];
  caveat_flags?: CaveatFlag[];
  vram_peak_mib: number | null;
  ram_peak_mib: number;
  gpu_memory_total_mb: number | null;
  system_memory_total_mb: number | null;
}

export type RejectionGate = "stability" | "suspect_samples" | "missing_pp_or_tg" | "caveat_flagged";

export type ProfileId = "max_speed" | "balanced" | "max_context" | "low_memory";

export interface ScoredConfig {
  /** §0.4 hash with n_depth pinned to 0 -- the tuple's depth-blind identity. */
  key: string;
  /** Every §0.4 axis except n_depth, which selects rows rather than identifying them. */
  axes: ConfigHashInput;
  /** The run-item indices that produced this tuple's rows. */
  itemIndices: number[];
  /** Raw measured rates at the selected depths. */
  pp: number;
  tg: number;
  /** Depth the tg (and pp, where available) reading was taken at. */
  referenceDepth: number;
  /** Normalized (§0.3): X / P90(per-config medians of X). */
  ppHat: number;
  tgHat: number;
  /** max(VRAM_peak/VRAM_total, RAM_peak/RAM_total) over non-null inputs; null when neither exists. */
  pressure: number | null;
  vramPeakMib: number | null;
  ramPeakMib: number;
  /** M1's inverse estimate for this placement, when the caller could compute one. */
  maxCtx: MaxCtxEstimate | null;
  /** min(1, maxCtx.tokens / target_ctx) when both exist; null otherwise. */
  fit: number | null;
  caveatFlags: CaveatFlag[];
}

export interface ProfileCard {
  id: ProfileId;
  title: string;
  config: ScoredConfig;
  /** True when the stability gate was waived to produce this card (§0.3's last paragraph). */
  unverified: boolean;
  /** Why this card ranks the way it does -- rendered verbatim next to the card. */
  basis: string;
}

export interface HiddenProfile {
  id: ProfileId;
  title: string;
  reason: string;
}

export interface ScoringResult {
  profiles: ProfileCard[];
  /** Named, never silently absent (M3). */
  hidden: HiddenProfile[];
  /** Which gate removed each tuple; zero-profile outcomes render these. */
  tallies: Record<RejectionGate, number>;
  /** Tuples that survived every gate. */
  scoredCount: number;
  /** Tuples seen before gating. */
  candidateCount: number;
  referenceDepth: number | null;
  /** The single method_version this scoring pass covers (§0.1). */
  methodVersion: number | null;
  /** True when no memory totals existed anywhere, so PRESSURE is undefined (§0.3 degradation). */
  pressureUnavailable: boolean;
  weights: { wPP: number; wTG: number };
  goals: GoalsConfig;
}

export interface ScoringInput {
  rows: ScoringRow[];
  /** Terminal status per run-item idx -- a failed/failed_oom item disqualifies its tuple. */
  itemStatusByIdx?: Record<number, string>;
  goals?: GoalsConfig;
  /** Sweep repeats, used as the n fallback for llama-bench rows (which report no sample_count). */
  repeats?: number;
  /** N6 -- scoring-side exclusion of the first M repeats, recorded in the run's config. */
  discardFirstRepeats?: number;
  /**
   * Tokens the TG reference depth aims at (default 50 % of the stated target,
   * else 50 % of trained context). The nearest available depth wins.
   */
  referenceDepthTokens?: number | null;
  /** M1 per-placement inverse estimate; the caller owns weights/placement knowledge. */
  maxCtxFor?: (axes: ConfigHashInput) => MaxCtxEstimate | null;
}

const STDDEV_FLOOR_TPS = 0.5;
const STDDEV_FRACTION = 0.1;
const MIN_SAMPLES = 3;

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Nearest-rank P90 over per-config medians -- explicitly not the raw max, so
// one repeat hit by a timer bug (~1e6 tok/s) cannot pin the denominator.
export function p90(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil(0.9 * sorted.length);
  return sorted[Math.max(0, Math.min(sorted.length - 1, rank - 1))];
}

// Sample (n-1) standard deviation -- llama-bench's own formula, matched here
// so a scoring-side recomputation after N6's discard is comparable with the
// stddev the worker reported.
export function sampleStddev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function tupleAxes(row: ScoringRow): ConfigHashInput {
  return {
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    // Depth is deliberately pinned to 0 in the tuple's identity: depth
    // selects WHICH tg row feeds the comparison (§0.3), it does not make two
    // otherwise-identical configurations different configurations.
    n_depth: 0,
    threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    engine: (row.engine as "bench" | "server" | undefined) ?? (row.mtp === "on" ? "server" : "bench"),
    concurrency: row.concurrency ?? 1,
  };
}

export function tupleKey(row: ScoringRow): string {
  return configHash(tupleAxes(row));
}

interface RowStats {
  mean: number;
  stddev: number;
  n: number;
  suspect: number;
}

// N6's "discard first M repeats" is a scoring-side exclusion over the stored
// per-repeat samples -- never post-hoc trimming outside the stored config,
// and never applied when the row carries no per-repeat detail to trim.
function statsFor(row: ScoringRow, repeats: number | undefined, discardFirst: number): RowStats {
  const samples = row.repeat_samples;
  if (discardFirst > 0 && samples && samples.length > discardFirst) {
    const kept = samples.slice(discardFirst);
    return {
      mean: kept.reduce((a, b) => a + b, 0) / kept.length,
      stddev: sampleStddev(kept),
      n: kept.length,
      suspect: row.suspect_count ?? 0,
    };
  }
  return {
    mean: row.avg_tps,
    stddev: row.stddev_tps,
    n: row.sample_count ?? samples?.length ?? repeats ?? 0,
    suspect: row.suspect_count ?? 0,
  };
}

// §0.3's P90 denominator is explicitly "per-config medians", not the mean
// used elsewhere for the card's displayed rate -- a single outlier repeat
// (e.g. a timer-bug ~1e6 tok/s reading that still passed the suspect filter)
// pulls a mean further than a median, and the denominator is exactly where
// the spec guards against that.
function medianStatFor(row: ScoringRow, discardFirst: number): number {
  const samples = row.repeat_samples;
  if (samples && samples.length > 0) {
    const kept = discardFirst > 0 && samples.length > discardFirst ? samples.slice(discardFirst) : samples;
    if (kept.length > 0) return median(kept);
  }
  return row.avg_tps;
}

function passesStability(stats: RowStats): boolean {
  if (stats.n < MIN_SAMPLES) return false;
  // Absolute floor stops near-zero means exploding the ratio.
  return stats.stddev <= Math.max(STDDEV_FRACTION * stats.mean, STDDEV_FLOOR_TPS);
}

function pressureOf(row: ScoringRow | undefined, peer: ScoringRow | undefined): number | null {
  const candidates: number[] = [];
  for (const r of [row, peer]) {
    if (!r) continue;
    if (r.vram_peak_mib != null && r.gpu_memory_total_mb != null && r.gpu_memory_total_mb > 0) {
      candidates.push(r.vram_peak_mib / r.gpu_memory_total_mb);
    }
    if (r.system_memory_total_mb != null && r.system_memory_total_mb > 0) {
      candidates.push(r.ram_peak_mib / r.system_memory_total_mb);
    }
  }
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

interface TupleBucket {
  key: string;
  axes: ConfigHashInput;
  rows: ScoringRow[];
}

function nearestDepthRow(rows: ScoringRow[], target: number, excludeSwa: boolean): ScoringRow | undefined {
  const usable = excludeSwa ? rows.filter((r) => !(r.caveat_flags ?? []).includes("swa")) : rows;
  if (usable.length === 0) return undefined;
  let best = usable[0];
  let bestDelta = Math.abs((best.n_depth ?? 0) - target);
  for (const row of usable.slice(1)) {
    const delta = Math.abs((row.n_depth ?? 0) - target);
    if (delta < bestDelta) {
      best = row;
      bestDelta = delta;
    }
  }
  return best;
}

export function scoreProfiles(input: ScoringInput): ScoringResult {
  const goals = input.goals ?? defaultGoals();
  const weights = WORKLOAD_WEIGHTS[goals.workload] ?? WORKLOAD_WEIGHTS.even;
  const discardFirst = input.discardFirstRepeats ?? 0;
  const tallies: Record<RejectionGate, number> = {
    stability: 0,
    suspect_samples: 0,
    missing_pp_or_tg: 0,
    caveat_flagged: 0,
  };

  // §0.1 -- rows of different method_version are never averaged together.
  // The pass covers the most recent vintage present; older rows are reported
  // through the caller's own "mixed vintages" surface, not silently mixed in.
  const versions = new Set(input.rows.map((r) => r.method_version ?? null));
  let methodVersion: number | null = null;
  for (const v of versions) {
    if (v != null && (methodVersion == null || v > methodVersion)) methodVersion = v;
  }
  const rows = input.rows.filter((r) => (r.method_version ?? null) === methodVersion);

  const buckets = new Map<string, TupleBucket>();
  for (const row of rows) {
    if (row.test_type !== "pp" && row.test_type !== "tg") continue;
    const key = tupleKey(row);
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = { key, axes: tupleAxes(row), rows: [] };
      buckets.set(key, bucket);
    }
    bucket.rows.push(row);
  }

  // The pinned reference depth: nearest available to the requested target
  // across every tuple, so all cards compare TG at one depth.
  const depthsPresent = [...new Set(rows.map((r) => r.n_depth ?? 0))].sort((a, b) => a - b);
  const requestedDepth = input.referenceDepthTokens ?? null;
  let referenceDepth: number | null = null;
  if (depthsPresent.length > 0) {
    if (requestedDepth == null) {
      referenceDepth = depthsPresent[depthsPresent.length - 1];
    } else {
      referenceDepth = depthsPresent.reduce((best, d) =>
        Math.abs(d - requestedDepth) < Math.abs(best - requestedDepth) ? d : best
      );
    }
  }

  const candidateCount = buckets.size;
  const eligible: ScoredConfig[] = [];
  const stabilityOnlyFailures: ScoredConfig[] = [];
  // §0.3 P90 denominator input, keyed by tuple: per-config MEDIANS, kept
  // separate from ScoredConfig.pp/tg (which stay the mean -- "raw measured
  // rate" -- for display) so normalization and display never silently share
  // one statistic that was only validated for one of the two uses.
  const medianByKey = new Map<string, { pp: number; tg: number }>();

  for (const bucket of buckets.values()) {
    const ppRows = bucket.rows.filter((r) => r.test_type === "pp");
    const tgRows = bucket.rows.filter((r) => r.test_type === "tg");

    // A failed/failed_oom item in the tuple's sub-run disqualifies it; a
    // `skipped` item does not -- it was never measured. Both land in the
    // missing_pp_or_tg tally, which is the closed registry's bucket for "this
    // tuple never produced the pair scoring needs".
    const itemFailed = bucket.rows.some((r) => {
      const status = input.itemStatusByIdx?.[r.idx];
      return status === "failed" || status === "failed_oom";
    });
    if (itemFailed || ppRows.length === 0 || tgRows.length === 0) {
      tallies.missing_pp_or_tg++;
      continue;
    }

    const targetDepth = referenceDepth ?? 0;
    // §0.3 -- swa rows are excluded from the TG reference-depth comparison:
    // depth past the sliding window is structurally misleading.
    const tgRow = nearestDepthRow(tgRows, targetDepth, true);
    if (!tgRow) {
      tallies.caveat_flagged++;
      continue;
    }
    const ppRow = nearestDepthRow(ppRows, tgRow.n_depth ?? 0, false)!;

    const ppStats = statsFor(ppRow, input.repeats, discardFirst);
    const tgStats = statsFor(tgRow, input.repeats, discardFirst);

    const pressure = pressureOf(tgRow, ppRow);
    const maxCtx = input.maxCtxFor?.(bucket.axes) ?? null;
    const fit =
      maxCtx && maxCtx.confidence !== "unknown" && goals.target_ctx != null && goals.target_ctx > 0
        ? Math.min(1, maxCtx.tokens / goals.target_ctx)
        : null;

    const config: ScoredConfig = {
      key: bucket.key,
      axes: bucket.axes,
      itemIndices: [...new Set(bucket.rows.map((r) => r.idx))].sort((a, b) => a - b),
      pp: ppStats.mean,
      tg: tgStats.mean,
      referenceDepth: tgRow.n_depth ?? 0,
      ppHat: 0,
      tgHat: 0,
      pressure,
      vramPeakMib: tgRow.vram_peak_mib,
      ramPeakMib: tgRow.ram_peak_mib,
      maxCtx,
      fit,
      caveatFlags: [...new Set([...(ppRow.caveat_flags ?? []), ...(tgRow.caveat_flags ?? [])])],
    };

    if (ppStats.suspect > 0 || tgStats.suspect > 0) {
      tallies.suspect_samples++;
      continue;
    }
    medianByKey.set(config.key, {
      pp: medianStatFor(ppRow, discardFirst),
      tg: medianStatFor(tgRow, discardFirst),
    });
    if (!passesStability(ppStats) || !passesStability(tgStats)) {
      tallies.stability++;
      stabilityOnlyFailures.push(config);
      continue;
    }
    eligible.push(config);
  }

  const normalizeAgainst = eligible.length > 0 ? eligible : stabilityOnlyFailures;
  const ppDenominator = p90(normalizeAgainst.map((c) => medianByKey.get(c.key)!.pp));
  const tgDenominator = p90(normalizeAgainst.map((c) => medianByKey.get(c.key)!.tg));
  for (const config of [...eligible, ...stabilityOnlyFailures]) {
    config.ppHat = ppDenominator > 0 ? config.pp / ppDenominator : 0;
    config.tgHat = tgDenominator > 0 ? config.tg / tgDenominator : 0;
  }

  const pressureUnavailable = [...eligible, ...stabilityOnlyFailures].every((c) => c.pressure == null);
  const result: ScoringResult = {
    profiles: [],
    hidden: [],
    tallies,
    scoredCount: eligible.length,
    candidateCount,
    referenceDepth,
    methodVersion,
    pressureUnavailable,
    weights,
    goals,
  };

  if (eligible.length === 0) {
    // §0.3 -- if every tuple failed only stability, an additional
    // `unverified` Max Speed card may be emitted with that gate waived,
    // visibly distinct and excluded from downstream defaults.
    const onlyStability =
      stabilityOnlyFailures.length > 0 &&
      tallies.suspect_samples === 0 &&
      tallies.caveat_flagged === 0 &&
      tallies.missing_pp_or_tg === 0;
    if (onlyStability) {
      const best = argmax(stabilityOnlyFailures, (c) => combined(c, weights));
      if (best) {
        result.profiles.push({
          id: "max_speed",
          title: "Max Speed",
          config: best,
          unverified: true,
          basis:
            "every configuration failed only the stability gate, so this card is shown with that gate waived — unverified, and excluded from defaults",
        });
      }
    }
    return result;
  }

  emitCards(result, eligible, goals, weights);
  return result;
}

function combined(config: ScoredConfig, weights: { wPP: number; wTG: number }): number {
  return weights.wPP * config.ppHat + weights.wTG * config.tgHat;
}

function argmax<T>(items: T[], score: (item: T) => number): T | undefined {
  let best: T | undefined;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const item of items) {
    const s = score(item);
    if (s > bestScore) {
      best = item;
      bestScore = s;
    }
  }
  return best;
}

function peakSum(config: ScoredConfig): number {
  return (config.vramPeakMib ?? 0) + config.ramPeakMib;
}

function emitCards(
  result: ScoringResult,
  eligible: ScoredConfig[],
  goals: GoalsConfig,
  weights: { wPP: number; wTG: number }
): void {
  const showMaxSpeed = goals.goal === "max_speed" || goals.goal === "balanced";
  const showBalanced = goals.goal === "balanced";
  const showMaxContext = goals.goal === "max_context";

  if (showMaxSpeed) {
    const best = argmax(eligible, (c) => combined(c, weights))!;
    result.profiles.push({
      id: "max_speed",
      title: "Max Speed",
      config: best,
      unverified: false,
      basis: `argmax of ${weights.wPP.toFixed(2)}·P̂P + ${weights.wTG.toFixed(
        2
      )}·T̂G — measured speed only`,
    });
  } else {
    result.hidden.push({
      id: "max_speed",
      title: "Max Speed",
      reason: `not produced under the “${goalLabel(goals)}” goal`,
    });
  }

  if (showBalanced) {
    // M3's pinned reduction: with FIT null, f = 0.4 and the rule collapses to
    // argmax(0.8*S + 0.2*(1 - PRESSURE)) -- byte-for-byte the fixed-weight
    // behavior, which is what "skippable" means.
    const best = argmax(eligible, (c) => {
      const f = c.fit != null ? 0.4 * c.fit : 0.4;
      const pressureTerm = c.pressure != null ? 0.2 * (1 - c.pressure) : 0;
      return (0.4 + f) * combined(c, weights) + pressureTerm;
    })!;
    result.profiles.push({
      id: "balanced",
      title: "Balanced",
      config: best,
      unverified: false,
      basis:
        best.fit != null
          ? "argmax((0.4 + 0.4·FIT)·S + 0.2·(1 − PRESSURE)) — FIT from the affordability estimate"
          : "argmax(0.8·S + 0.2·(1 − PRESSURE)) — no target stated, so FIT's weight is redistributed to speed",
    });
  } else {
    result.hidden.push({
      id: "balanced",
      title: "Balanced",
      reason: `not produced under the “${goalLabel(goals)}” goal`,
    });
  }

  if (showMaxContext) {
    const floorFrac = goals.speed_floor_frac ?? 0.5;
    const bestTg = Math.max(...eligible.map((c) => c.tg));
    const overFloor = eligible.filter((c) => bestTg > 0 && c.tg >= floorFrac * bestTg);
    if (overFloor.length > 0) {
      const best = overFloor.reduce((a, b) => {
        const at = a.maxCtx?.tokens ?? -1;
        const bt = b.maxCtx?.tokens ?? -1;
        if (bt > at) return b;
        if (bt < at) return a;
        // Tie-break lower VRAM_peak + RAM_peak; with no memory totals the
        // tie-break is omitted and order stays stable by estimated tokens.
        return peakSum(b) < peakSum(a) ? b : a;
      });
      result.profiles.push({
        id: "max_context",
        title: "Max Context",
        config: best,
        unverified: false,
        basis: `argmax estimated affordable tokens among configurations clearing the ≥ ${Math.round(
          floorFrac * 100
        )} % TG floor — ranked on estimated fit, not measured at your target. Verify with a probe run.`,
      });
    } else {
      result.hidden.push({
        id: "max_context",
        title: "Max Context",
        reason: `no configuration cleared the ≥ ${Math.round(floorFrac * 100)} % usable-speed floor`,
      });
    }
  } else {
    result.hidden.push({
      id: "max_context",
      title: "Max Context",
      reason: "only produced under the “Max context” goal",
    });
  }

  // Low Memory: shown whenever PRESSURE is computable for >= 1 config. On
  // machines reporting no memory totals the §0.3 degradation applies -- a
  // speed card and a stated reason, never a silent collapse.
  const withPressure = eligible.filter((c) => c.pressure != null);
  if (withPressure.length > 0) {
    const pool = withPressure.filter((c) => c.tgHat >= 0.9 && c.ppHat >= 0.8);
    const candidates = pool.length > 0 ? pool : withPressure;
    const best = candidates.reduce((a, b) => {
      if ((b.pressure ?? 1) < (a.pressure ?? 1)) return b;
      if ((b.pressure ?? 1) > (a.pressure ?? 1)) return a;
      return peakSum(b) < peakSum(a) ? b : a;
    });
    result.profiles.push({
      id: "low_memory",
      title: "Low Memory",
      config: best,
      unverified: false,
      basis:
        pool.length > 0
          ? "argmin PRESSURE among configurations at T̂G ≥ 0.9 and P̂P ≥ 0.8"
          : "argmin PRESSURE — no configuration cleared the T̂G ≥ 0.9 / P̂P ≥ 0.8 speed band, so the whole eligible set was considered",
    });
  } else {
    result.hidden.push({
      id: "low_memory",
      title: "Low Memory",
      reason: "this machine reported no VRAM or RAM totals, so memory pressure is undefined here",
    });
  }
}

export function goalLabel(goals: GoalsConfig): string {
  switch (goals.goal) {
    case "max_speed":
      return "Max tok/s";
    case "max_context":
      return "Max context";
    default:
      return "Balanced";
  }
}

// Every profile card carries this: the boundary is that quality is never
// measured by any of these rankings.
export const QUALITY_NOT_MEASURED_DISCLAIMER =
  "Ranked on measured speed and memory — output quality was not measured.";
