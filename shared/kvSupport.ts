// Step 4 of the optimization flow (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md):
// which K/V cache type pairs this build + backend + model actually run on
// the GPU. One llama-fit-params call per pair and FA mode at -c 4096 -ngl 99
// (no search, no weights, ~0.5 s each); the scheduler's split count against
// the same model's f16/f16 baseline is the fallback signal, verified against
// the 2026-09-29 real-load sweep.

import { KV_CACHE_TYPES, kvRelativeSize, kvSizeKey, isQuantizedKv, type FitFaMode, type ParsedFitOutput } from "./fitParams.js";

export type KvSupportStatus = "ok" | "cpu_fallback" | "invalid" | "fa_disabled" | "cuda_slow";

export interface KvSupportRow {
  ctk: string;
  ctv: string;
  fa: FitFaMode;
  status: KvSupportStatus;
  splits: number | null;
  baseline_splits: number | null;
  detail: string | null;
}

/** All 64 ordered (K, V) pairs of the non-32-bit cache types. */
export function kvCandidatePairs(): [string, string][] {
  const out: [string, string][] = [];
  for (const k of KV_CACHE_TYPES) for (const v of KV_CACHE_TYPES) out.push([k, v]);
  return out;
}

// CUDA compiles flash-attention vector kernels only for these pairs by
// default (GGML_CUDA_FA_QUANTS off). Other pairs stay on the GPU -- splits
// can't see it -- but convert K/V to f16 on every decode step.
const CUDA_FA_VEC_PAIRS = new Set(["f16/f16", "bf16/bf16", "q8_0/q8_0", "q4_0/q4_0"]);

export function classifyKvProbe(input: {
  ctk: string;
  ctv: string;
  fa: FitFaMode;
  probe: ParsedFitOutput;
  /** The same model's f16/f16 probe at the same FA mode. */
  baseline: ParsedFitOutput | null;
  backend: string;
}): KvSupportRow {
  const { ctk, ctv, fa, probe, baseline, backend } = input;
  const baseSplits = baseline?.splits ?? null;
  const row = (status: KvSupportStatus, detail: string | null): KvSupportRow => ({
    ctk,
    ctv,
    fa,
    status,
    splits: probe.splits,
    baseline_splits: baseSplits,
    detail,
  });
  if (probe.error || probe.splits == null) {
    return row(
      "invalid",
      probe.error ??
        (isQuantizedKv(ctv) && fa === "off"
          ? "a quantized V cache needs flash attention"
          : "the context could not be created")
    );
  }
  if (fa === "on" && probe.faDisabled) {
    return row("fa_disabled", "flash attention has no GPU kernel for this pair and was turned off");
  }
  if (baseSplits != null && probe.splits > baseSplits) {
    return row("cpu_fallback", `attention runs on the CPU (${probe.splits} graph splits vs ${baseSplits} for f16/f16)`);
  }
  if (/cuda/i.test(backend) && fa === "on" && !CUDA_FA_VEC_PAIRS.has(`${ctk}/${ctv}`)) {
    return row("cuda_slow", "runs on the GPU, but CUDA converts this pair to f16 on every decode step unless built with GGML_CUDA_FA_QUANTS");
  }
  return row("ok", null);
}

/** Pairs usable in the flow: they run on the GPU at this FA mode. */
export function isUsableKv(row: Pick<KvSupportRow, "status">): boolean {
  return row.status === "ok" || row.status === "cuda_slow";
}

export interface KvSizeGroup {
  /** kvSizeKey of every pair in the group. */
  key: string;
  relSize: number;
  /** The pair whose fit answer stands for the group. */
  rep: [string, string];
  fa: FitFaMode;
  pairs: [string, string][];
}

const TYPE_PREFERENCE = ["f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl", "bf16"];

function pairRank([k, v]: [string, string]): number {
  // Symmetric pairs first, then the more common type.
  return (k === v ? 0 : 100) + TYPE_PREFERENCE.indexOf(k) * 10 + TYPE_PREFERENCE.indexOf(v);
}

/**
 * Collapses usable pairs into fit-map groups by cache size. The fit answer
 * depends only on the bytes per cell, so one call per size covers every pair
 * of that size. FA on is preferred: a quantized V needs it, and FA off makes
 * a quantized K fall back to the CPU on most backends.
 */
export function kvSizeGroups(rows: readonly KvSupportRow[], equalHeadDims = true): KvSizeGroup[] {
  const usable = rows.filter(isUsableKv);
  const byKey = new Map<string, { on: [string, string][]; off: [string, string][] }>();
  for (const r of usable) {
    const key = kvSizeKey(r.ctk, r.ctv, equalHeadDims);
    const g = byKey.get(key) ?? { on: [], off: [] };
    const list = r.fa === "on" ? g.on : g.off;
    if (!list.some(([k, v]) => k === r.ctk && v === r.ctv)) list.push([r.ctk, r.ctv]);
    byKey.set(key, g);
  }
  const groups: KvSizeGroup[] = [];
  for (const [key, g] of byKey) {
    const fa: FitFaMode = g.on.length > 0 ? "on" : "off";
    const pairs = (fa === "on" ? g.on : g.off).sort((a, b) => pairRank(a) - pairRank(b));
    groups.push({ key, relSize: kvRelativeSize(pairs[0][0], pairs[0][1]), rep: pairs[0], fa, pairs });
  }
  return groups.sort((a, b) => b.relSize - a.relSize);
}

export function pairLabel(ctk: string, ctv: string): string {
  return `${ctk} / ${ctv}`;
}

export const KV_QUALITY_WARNING =
  "Quantized KV caches use less memory, which can fit more layers or a longer context and may improve speed — but they can slightly reduce output quality.";
