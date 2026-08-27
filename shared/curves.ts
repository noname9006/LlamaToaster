// BENCHMARKING_PLAN_V8.md N1 (context curves) and N5 (concurrency knee).
// Both are DERIVED READS over ordinary `results` rows -- no new table, and in
// N5's case no stored verdict: the knee is computed on read, so a later
// re-measurement moves it without anyone having to invalidate a cached
// answer.

import type { CaveatFlag, TestType } from "./types.js";

export interface CurveSourceRow {
  id: string;
  /** The run and item that produced this row -- a point IS one item's pp/tg pair. */
  run_id: string;
  idx: number;
  test_type: TestType;
  n_prompt: number;
  n_gen: number;
  n_depth?: number | null;
  avg_tps: number;
  stddev_tps: number;
  vram_peak_mib: number | null;
  ram_peak_mib: number;
  ttft_ms_p50?: number | null;
  ttft_ms_p95?: number | null;
  ttft_n?: number | null;
  e2e_ms_mean?: number | null;
  method_version?: number | null;
  caveat_flags?: CaveatFlag[];
  concurrency?: number | null;
  engine?: string | null;
  created_at: number;
}

export interface CurvePoint {
  /** n_depth + n_prompt -- server rows keep n_depth = 0, so this is n_prompt there. */
  effectiveCtx: number;
  pp: number | null;
  tg: number | null;
  ttftMs: number | null;
  /** 1 on a choreographed cold point: single-shot by construction, labeled as such. */
  ttftN: number | null;
  e2eMs: number | null;
  vramPeakMib: number | null;
  methodVersion: number | null;
  caveatFlags: CaveatFlag[];
  /**
   * True when a later measurement of the same effective context exists. The
   * superseded point is rendered greyed -- never averaged with its successor.
   */
  superseded: boolean;
  /** The item this point came from -- what makes two same-second points distinct. */
  runId: string;
  idx: number;
  /** Stable identity used as the second key of the (created_at, id) total order. */
  pointId: string;
  /** Dropped from the curve with an explanatory tooltip when the cache did not hold. */
  excluded: boolean;
  excludedReason: string | null;
  createdAt: number;
}

// A curve point pairs the pp and tg rows measured at one effective context.
// When two choreographed points share an effective context (a re-measure of
// the same cell) the greatest (created_at, id) wins -- a TOTAL order, since
// second-resolution timestamps can collide -- and the superseded point stays
// visible, greyed, rather than being averaged in or silently dropped.
export function buildCurve(rows: CurveSourceRow[]): CurvePoint[] {
  // A point is ONE ITEM's pp/tg pair. Grouping on the item (not on a
  // timestamp) is what makes two measurements of the same cell in the same
  // second two distinct points rather than one merged average -- the exact
  // collision the (created_at, id) tie-break exists to survive.
  const byCell = new Map<string, { ctx: number; rows: CurveSourceRow[] }>();
  for (const row of rows) {
    if (row.test_type !== "pp" && row.test_type !== "tg") continue;
    const ctx = (row.n_depth ?? 0) + row.n_prompt;
    const key = `${row.run_id}|${row.idx}|${ctx}`;
    let cell = byCell.get(key);
    if (!cell) {
      cell = { ctx, rows: [] };
      byCell.set(key, cell);
    }
    cell.rows.push(row);
  }

  const points: CurvePoint[] = [];
  for (const cell of byCell.values()) {
    const pp = cell.rows.find((r) => r.test_type === "pp");
    const tg = cell.rows.find((r) => r.test_type === "tg");
    const flags = [...new Set(cell.rows.flatMap((r) => r.caveat_flags ?? []))];
    const ttftRow = cell.rows.find((r) => r.ttft_ms_p50 != null) ?? pp ?? tg!;
    // The eviction detector's verdict has already been written onto the row
    // as a caveat flag by the worker; the curve honors it by dropping the
    // point with its reason, never by silently keeping a fast repeat.
    const evicted = flags.includes("cache_evicted");
    const shifted = flags.includes("context_shift");
    points.push({
      effectiveCtx: cell.ctx,
      pp: pp?.avg_tps ?? null,
      tg: tg?.avg_tps ?? null,
      ttftMs: ttftRow?.ttft_ms_p50 ?? null,
      ttftN: ttftRow?.ttft_n ?? null,
      e2eMs: (tg ?? pp)?.e2e_ms_mean ?? null,
      vramPeakMib: (tg ?? pp)?.vram_peak_mib ?? null,
      methodVersion: (pp ?? tg)!.method_version ?? null,
      caveatFlags: flags,
      superseded: false,
      runId: cell.rows[0].run_id,
      idx: cell.rows[0].idx,
      pointId: cell.rows.map((r) => r.id).sort()[0],
      excluded: evicted || shifted,
      excludedReason: evicted
        ? "the prefix cache did not hold: a warm repeat re-prefilled, so this point's generation numbers are not comparable"
        : shifted
          ? "a context shift appeared in the logs and this build has no --no-context-shift, which silently corrupts TTFT comparability"
          : null,
      createdAt: Math.max(...cell.rows.map((r) => r.created_at)),
    });
  }

  // (created_at, id) is a TOTAL order -- second-resolution timestamps can
  // collide, and "the greatest wins" has to mean something even then.
  const order = (a: CurvePoint, b: CurvePoint): number =>
    a.createdAt !== b.createdAt ? a.createdAt - b.createdAt : a.pointId < b.pointId ? -1 : a.pointId > b.pointId ? 1 : 0;
  points.sort((a, b) => (a.effectiveCtx !== b.effectiveCtx ? a.effectiveCtx - b.effectiveCtx : order(a, b)));

  // Every point but the greatest at each effective context is superseded:
  // rendered greyed, never averaged into its successor.
  const winner = new Map<number, CurvePoint>();
  for (const point of points) {
    const held = winner.get(point.effectiveCtx);
    if (!held || order(point, held) > 0) winner.set(point.effectiveCtx, point);
  }
  for (const point of points) {
    if (winner.get(point.effectiveCtx) !== point) point.superseded = true;
  }
  return points;
}

// N1's sizing ladder. Points beyond the trained context or beyond what M1
// says the machine affords render as "unavailable -- reason", never as
// silent gaps.
export const CURVE_LADDER = [2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072] as const;

export interface LadderCell {
  effectiveCtx: number;
  measured: boolean;
  available: boolean;
  unavailableReason: string | null;
}

export function buildLadder(
  measuredContexts: number[],
  limits: { trainedCtx?: number | null; affordableTokens?: number | null }
): LadderCell[] {
  const measured = new Set(measuredContexts);
  return CURVE_LADDER.map((effectiveCtx) => {
    let unavailableReason: string | null = null;
    if (limits.trainedCtx != null && effectiveCtx > limits.trainedCtx) {
      unavailableReason = `unavailable — trained ctx ${limits.trainedCtx}`;
    } else if (limits.affordableTokens != null && limits.affordableTokens > 0 && effectiveCtx > limits.affordableTokens) {
      unavailableReason = `unavailable — this placement affords about ${limits.affordableTokens} tokens`;
    }
    return {
      effectiveCtx,
      measured: measured.has(effectiveCtx),
      available: unavailableReason == null,
      unavailableReason,
    };
  });
}

// --- N5: the concurrency knee ----------------------------------------------

export interface KneeSample {
  slots: number;
  /** TTFT p95 in milliseconds, from raw per-stream samples. */
  ttftP95Ms: number | null;
  /** Aggregate tok/s across all slots. */
  aggregateTps: number | null;
}

export interface KneeResult {
  samples: KneeSample[];
  /** The smallest slot count whose TTFT p95 exceeds 2x its slots=1 value. */
  knee: number | null;
  /** 2x the slots=1 value -- the threshold the chart draws. */
  thresholdMs: number | null;
  /** Paired with the chart as an sr-only sentence, so the bars are never colour-only. */
  summary: string;
}

export function deriveKnee(rows: CurveSourceRow[]): KneeResult {
  const bySlots = new Map<number, { ttft: number[]; tps: number }>();
  for (const row of rows) {
    const slots = row.concurrency ?? 1;
    let entry = bySlots.get(slots);
    if (!entry) {
      entry = { ttft: [], tps: 0 };
      bySlots.set(slots, entry);
    }
    if (row.ttft_ms_p95 != null) entry.ttft.push(row.ttft_ms_p95);
    if (row.test_type === "tg") entry.tps += row.avg_tps;
  }
  const samples: KneeSample[] = [...bySlots.entries()]
    .map(([slots, entry]) => ({
      slots,
      ttftP95Ms: entry.ttft.length > 0 ? Math.max(...entry.ttft) : null,
      aggregateTps: entry.tps > 0 ? entry.tps : null,
    }))
    .sort((a, b) => a.slots - b.slots);

  const solo = samples.find((s) => s.slots === 1);
  if (!solo || solo.ttftP95Ms == null) {
    return {
      samples,
      knee: null,
      thresholdMs: null,
      summary: "no slots = 1 baseline was measured, so a knee cannot be derived from these rows",
    };
  }
  const thresholdMs = 2 * solo.ttftP95Ms;
  const knee = samples.find((s) => s.slots > 1 && s.ttftP95Ms != null && s.ttftP95Ms > thresholdMs)?.slots ?? null;
  return {
    samples,
    knee,
    thresholdMs,
    summary:
      knee != null
        ? `knee at slots = ${knee} — TTFT p95 doubles past this point`
        : `no knee within the measured ladder — TTFT p95 stayed under ${Math.round(thresholdMs)} ms at every slot count`,
  };
}
