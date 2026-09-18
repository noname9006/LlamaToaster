// The Benchmark chain's sweep stage (Test C): prefill speed measured ALONG one
// context's fill, not at a single prompt length.
//
// One llama-server is started per sweep item with -c = ctx. A single prompt of
// fillTarget(ctx) tokens is then sent in growing prefixes -- fillBoundaries()
// -- with the prompt cache on, so each request only prefills the slice
// appended since the previous one. That slice's own prompt_n / prompt_ms is
// the prefill rate at that depth of the context, which is exactly what a
// whole-prompt reading averages away: attention cost grows with what is
// already in the KV cache, so the last slice of a nearly-full context is the
// slowest one a real workload pays.
//
// Each slice lands as an ordinary pp result row: n_depth = tokens already in
// the context when the slice started, n_prompt = the slice's own size (the
// same meaning llama-bench's -d gives the pair), stamped
// FILL_CURVE_METHOD_VERSION so it is never averaged with a whole-prompt pp.

import type { FillCurveSpec } from "./types.js";

// The prompt stops 0.5 % short of the context so the last request -- prompt
// plus the one generated token every request asks for -- is guaranteed to
// fit, whatever the tokenizer adds on the server side.
export const FILL_CURVE_HEADROOM_FRAC = 0.005;
export const DEFAULT_FILL_CURVE_STEPS = 16;
export const MIN_FILL_CURVE_STEPS = 2;
export const MAX_FILL_CURVE_STEPS = 64;
export const MIN_FILL_CURVE_CTX = 256;
export const MAX_FILL_CURVE_CTX = 4_194_304;

/** Tokens the prompt fills the context to: ctx less the 0.5 % headroom. */
export function fillTarget(ctx: number): number {
  return Math.floor(ctx * (1 - FILL_CURVE_HEADROOM_FRAC));
}

export function fillCurveSteps(spec: Pick<FillCurveSpec, "steps">): number {
  return spec.steps ?? DEFAULT_FILL_CURVE_STEPS;
}

/**
 * Cumulative prompt lengths, strictly increasing, ending exactly at `fill`:
 * request i sends the first boundaries[i] tokens of the prompt. Evenly spaced;
 * collapses to fewer steps when the fill is too small to give every step at
 * least one token.
 */
export function fillBoundaries(fill: number, steps: number): number[] {
  const total = Math.max(0, Math.trunc(fill));
  const count = Math.max(1, Math.min(Math.trunc(steps), total));
  const out: number[] = [];
  for (let i = 1; i <= count; i++) {
    const b = Math.round((total * i) / count);
    if (b > 0 && (out.length === 0 || b > out[out.length - 1])) out.push(b);
  }
  return out;
}

/** Returns an error message, or null when the spec is valid. */
export function validateFillCurveSpec(spec: FillCurveSpec, trainedCtx: number | null): string | null {
  if (!Number.isInteger(spec.ctx) || spec.ctx < MIN_FILL_CURVE_CTX || spec.ctx > MAX_FILL_CURVE_CTX) {
    return `fill_curve.ctx must be an integer between ${MIN_FILL_CURVE_CTX} and ${MAX_FILL_CURVE_CTX}`;
  }
  if (trainedCtx != null && trainedCtx > 0 && spec.ctx > trainedCtx) {
    return `fill_curve.ctx exceeds the model's trained context (${trainedCtx})`;
  }
  if (
    spec.steps !== undefined &&
    (!Number.isInteger(spec.steps) || spec.steps < MIN_FILL_CURVE_STEPS || spec.steps > MAX_FILL_CURVE_STEPS)
  ) {
    return `fill_curve.steps must be an integer between ${MIN_FILL_CURVE_STEPS} and ${MAX_FILL_CURVE_STEPS}`;
  }
  return null;
}

// --- reading the curves back ------------------------------------------------

export interface FillCurveRow {
  idx: number;
  test_type: string;
  n_prompt: number;
  n_depth?: number | null;
  avg_tps: number;
  stddev_tps: number;
  n_gpu_layers: number;
  cache_type_k: string;
  cache_type_v: string;
  sample_count?: number | null;
  suspect_count?: number | null;
  method_version?: number | null;
}

export interface FillCurvePoint {
  /** Tokens in the context when the slice started. */
  depth: number;
  /** Tokens in the context when the slice finished -- the x axis. */
  fill: number;
  tps: number;
  stddev: number;
  /** Repeats that were excluded because the prefix cache did not hold. */
  suspect: number;
}

export interface FillCurve {
  idx: number;
  ngl: number;
  kv: string;
  points: FillCurvePoint[];
}

/** One curve per sweep item, points in fill order. Non-fill rows are ignored. */
export function groupFillCurves(rows: readonly FillCurveRow[], methodVersion: number): FillCurve[] {
  const byIdx = new Map<number, FillCurve>();
  for (const r of rows) {
    if (r.test_type !== "pp" || r.method_version !== methodVersion) continue;
    let curve = byIdx.get(r.idx);
    if (!curve) {
      curve = { idx: r.idx, ngl: r.n_gpu_layers, kv: `${r.cache_type_k}/${r.cache_type_v}`, points: [] };
      byIdx.set(r.idx, curve);
    }
    const depth = r.n_depth ?? 0;
    curve.points.push({
      depth,
      fill: depth + r.n_prompt,
      tps: r.avg_tps,
      stddev: r.stddev_tps,
      suspect: r.suspect_count ?? 0,
    });
  }
  const curves = [...byIdx.values()].sort((a, b) => a.idx - b.idx);
  for (const c of curves) c.points.sort((a, b) => a.fill - b.fill);
  return curves;
}
