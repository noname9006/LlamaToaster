// The optimization flow's view logic (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md),
// kept out of React so it can be tested: what kind of machine this is, the
// fit-map job spec, turning stored fit points into the "Fit paths" chart's
// series, the default selection, the one-line insight under the chart, and
// the candidates the confirm step proposes.

import { fitCtxStops, isFitMargin, isQuantizedKv, type FitFaMode, type FitMarginMib, type FitPoint } from "./fitParams.js";
import { kvSizeGroups, pairLabel, type KvSupportRow } from "./kvSupport.js";
import { describePlacement, fromFit, type MoeInfo } from "./moePlacement.js";
import type { SpeedCandidate, ThreadArgs } from "./speedRun.js";

export type MachineClass = "dgpu" | "unified" | "cpu";

export function machineClassOf(input: { hasGpu: boolean; unified: boolean }): MachineClass {
  if (!input.hasGpu) return "cpu";
  return input.unified ? "unified" : "dgpu";
}

export interface FitMapSpec {
  margin_mib: FitMarginMib;
  /** Context stops, ascending, all >= 4096. The worker walks them max first. */
  ctx_stops: number[];
  machine: MachineClass;
  /** Step 4: detect which K/V pairs run on the GPU. */
  kv_detect: boolean;
  /** Step 5: rerun the map once per supported K/V size. */
  kv_fit: boolean;
  target_ctx?: number | null;
}

export const MAX_FIT_CTX_STOPS = 12;

export function validateFitMapSpec(raw: unknown, trainedCtx: number | null): string | null {
  if (!raw || typeof raw !== "object") return "fit_map must be an object";
  const s = raw as Partial<FitMapSpec>;
  if (!isFitMargin(s.margin_mib)) return "fit_map.margin_mib must be 512, 1024 or 1536";
  if (s.machine !== "dgpu" && s.machine !== "unified" && s.machine !== "cpu") return "fit_map.machine must be dgpu, unified or cpu";
  if (!Array.isArray(s.ctx_stops) || s.ctx_stops.length === 0 || s.ctx_stops.length > MAX_FIT_CTX_STOPS) {
    return `fit_map.ctx_stops must list 1..${MAX_FIT_CTX_STOPS} contexts`;
  }
  const ceiling = trainedCtx && trainedCtx > 0 ? Math.max(4096, trainedCtx) : 4_194_304;
  for (const c of s.ctx_stops) {
    if (!Number.isInteger(c) || c < 4096 || c > ceiling) return `fit_map.ctx_stops values must be integers in [4096, ${ceiling}]`;
  }
  if (typeof s.kv_detect !== "boolean" || typeof s.kv_fit !== "boolean") return "fit_map.kv_detect and kv_fit must be booleans";
  if (s.kv_fit && !s.kv_detect) return "fit_map.kv_fit needs kv_detect";
  if (s.target_ctx != null && (!Number.isInteger(s.target_ctx) || s.target_ctx < 1)) return "fit_map.target_ctx must be a positive integer";
  return null;
}

export function defaultFitMapSpec(input: { trainedCtx: number | null; machine: MachineClass; margin: FitMarginMib; targetCtx?: number | null }): FitMapSpec {
  return {
    margin_mib: input.margin,
    ctx_stops: fitCtxStops(input.trainedCtx),
    machine: input.machine,
    kv_detect: true,
    kv_fit: true,
    target_ctx: input.targetCtx ?? null,
  };
}

// --- the Fit paths chart ------------------------------------------------------

export interface FitSeries {
  id: string;
  label: string;
  subtitle: string;
  /** "default" = f16 FA on, the reference the insight compares against. */
  kind: "default" | "fa_off" | "kv";
  fa: FitFaMode;
  ctk: string;
  ctv: string;
  /** Every pair whose answer this path stands for. */
  pairs: [string, string][];
  /** "q4_0 / q4_0, iq4_nl / q4_0, ..." -- for tooltips. */
  pairsText: string;
  /** Rank in the accent ramp (0 = largest cache); null for the f16 paths. */
  rampIndex: number | null;
  points: FitPoint[];
}

export function fitPointKey(p: Pick<FitPoint, "ctx" | "fa" | "ctk" | "ctv">): string {
  return `${p.ctx}|${p.fa}|${p.ctk}|${p.ctv}`;
}

/**
 * Groups points into chart paths: the two f16 paths from step 1, then one
 * path per K/V size from step 5 (smallest cache darkest). Each KV path lists
 * every supported pair of its size, since fit answers per byte, not per type.
 */
export function buildFitSeries(points: readonly FitPoint[], kvRows: readonly KvSupportRow[]): FitSeries[] {
  const by = new Map<string, FitPoint[]>();
  for (const p of points) {
    const k = `${p.ctk}|${p.ctv}|${p.fa}`;
    by.set(k, [...(by.get(k) ?? []), p]);
  }
  const sortPts = (l: FitPoint[]) => [...l].sort((a, b) => a.ctx - b.ctx);
  const out: FitSeries[] = [];
  const on = by.get("f16|f16|on");
  if (on) {
    out.push({ id: "f16-on", label: "f16 · FA on", subtitle: "default cache", kind: "default", fa: "on", ctk: "f16", ctv: "f16", pairs: [["f16", "f16"]], pairsText: "f16 / f16", rampIndex: null, points: sortPts(on) });
  }
  const off = by.get("f16|f16|off");
  if (off) {
    out.push({ id: "f16-off", label: "f16 · FA off", subtitle: "default cache, flash attention off", kind: "fa_off", fa: "off", ctk: "f16", ctv: "f16", pairs: [["f16", "f16"]], pairsText: "f16 / f16", rampIndex: null, points: sortPts(off) });
  }
  const groups = kvSizeGroups(kvRows);
  const kvSeries: FitSeries[] = [];
  for (const g of groups) {
    const [k, v] = g.rep;
    if (k === "f16" && v === "f16") continue;
    const pts = by.get(`${k}|${v}|${g.fa}`);
    if (!pts) continue;
    const lossy = isQuantizedKv(k) || isQuantizedKv(v);
    kvSeries.push({
      id: `kv-${k}-${v}-${g.fa}`,
      label: g.pairs.length > 1 ? `${pairLabel(k, v)} +${g.pairs.length - 1}` : pairLabel(k, v),
      subtitle: `FA ${g.fa}${lossy ? " · lossy KV" : ""}${g.pairs.length > 1 ? ` · ${g.pairs.length} pairs of this size` : ""}`,
      pairsText: g.pairs.map(([a, b]) => pairLabel(a, b)).join(", "),
      kind: "kv",
      fa: g.fa,
      ctk: k,
      ctv: v,
      pairs: g.pairs,
      rampIndex: 0,
      points: sortPts(pts),
    });
  }
  // Ramp: darker = smaller cache. kvSizeGroups already sorts largest first.
  kvSeries.forEach((s, i) => (s.rampIndex = i));
  return [...out, ...kvSeries];
}

/**
 * The KV paths worth showing before "show all": at some context they fit
 * more layers than both f16 paths, or the same layers with clearly less
 * system RAM, or they fit where f16 doesn't. At most `limit`, the biggest
 * gains first; the f16 paths are always shown.
 */
export function usefulSeries(series: readonly FitSeries[], moe: MoeInfo | null, limit = 4): Set<string> {
  const base = series.filter((s) => s.kind !== "kv");
  const gain = (s: FitSeries): number => {
    let best = 0;
    for (const p of s.points) {
      if (!fits(p)) continue;
      const refs = base.map((b) => b.points.find((x) => x.ctx === p.ctx)).filter((x): x is FitPoint => !!x && fits(x));
      if (refs.length === 0) {
        best = Math.max(best, 1000);
        continue;
      }
      const top = Math.max(...refs.map((r) => pointScore(r, moe)));
      const dy = pointScore(p, moe) - top;
      if (dy > 0) best = Math.max(best, 100 * dy);
      else if (dy === 0) {
        const ram = Math.min(...refs.filter((r) => pointScore(r, moe) === top).map((r) => r.host_used_mib ?? 0));
        const saved = ram - (p.host_used_mib ?? 0);
        if (saved > 256) best = Math.max(best, saved / 1024);
      }
    }
    return best;
  };
  const ranked = series
    .filter((s) => s.kind === "kv")
    .map((s) => ({ s, g: gain(s) }))
    .filter((x) => x.g > 0)
    .sort((a, b) => b.g - a.g || (a.s.rampIndex ?? 0) - (b.s.rampIndex ?? 0))
    .slice(0, limit);
  return new Set([...base.map((s) => s.id), ...ranked.map((x) => x.s.id)]);
}

export function fits(p: FitPoint): boolean {
  return p.verdict === "full" || p.verdict === "partial" || p.verdict === "cpu_only";
}

/** Where a point sits on the y axis. MoE models count layers whose experts
 * are on the GPU; points still short with every expert on the CPU fall in a
 * lower "dense layers only" band. */
export interface PointY {
  band: "main" | "dense";
  value: number;
}

export function pointY(p: FitPoint, moe: MoeInfo | null): PointY {
  const layers = p.layers_gpu ?? 0;
  if (!moe?.isMoe) return { band: "main", value: layers };
  const all = p.layers_total ?? moe.nBlocks + 1;
  const isAll = p.ngl === -1 || (p.ngl != null && p.ngl >= all) || layers >= all;
  if (!isAll) return { band: "dense", value: layers };
  // Fit counts the output layer among the overflowing ones, so layers that
  // keep their experts are all - overflow, capped at the MoE block count.
  return { band: "main", value: Math.max(0, Math.min(moe.moeLayers.length, all - p.overflow_layers)) };
}

/** Comparable score: any expert-phase point beats any dense-phase point. */
export function pointScore(p: FitPoint, moe: MoeInfo | null): number {
  const y = pointY(p, moe);
  return y.band === "main" ? 10_000 + y.value : y.value;
}

export function ctxLabel(ctx: number): string {
  if (ctx >= 1024 && ctx % 1024 === 0) return `${ctx / 1024}k`;
  if (ctx >= 1000) return `${Math.round(ctx / 1024)}k`;
  return String(ctx);
}

export interface FitSelection {
  seriesId: string;
  ctx: number;
}

/** The target context's stop (or the nearest stop at/above it, else the largest). */
export function targetStop(stops: readonly number[], target: number | null | undefined): number | null {
  if (stops.length === 0) return null;
  const sorted = [...stops].sort((a, b) => a - b);
  if (target == null) return sorted[sorted.length - 1];
  return sorted.find((s) => s >= target) ?? sorted[sorted.length - 1];
}

/** Default selection: the target context, and the setup with the most layers
 * there (tie -> less system RAM). Falls back to the largest fitting context. */
export function defaultFitSelection(series: readonly FitSeries[], targetCtx: number | null, moe: MoeInfo | null): FitSelection | null {
  const stops = [...new Set(series.flatMap((s) => s.points.map((p) => p.ctx)))];
  const tries = [targetStop(stops, targetCtx), ...[...stops].sort((a, b) => b - a)].filter((x): x is number => x != null);
  for (const ctx of tries) {
    let best: { s: FitSeries; p: FitPoint } | null = null;
    for (const s of series) {
      const p = s.points.find((x) => x.ctx === ctx);
      if (!p || !fits(p)) continue;
      if (
        !best ||
        pointScore(p, moe) > pointScore(best.p, moe) ||
        (pointScore(p, moe) === pointScore(best.p, moe) && (p.host_used_mib ?? 0) < (best.p.host_used_mib ?? 0))
      ) {
        best = { s, p };
      }
    }
    if (best) return { seriesId: best.s.id, ctx };
  }
  return null;
}

export function mibToGb(mib: number | null | undefined): number | null {
  return mib == null ? null : mib / 1024;
}

function gb(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GB`;
}

/** The sentence under the chart, comparing the selection with the default
 * (f16 · FA on) at the same context. */
export function fitInsight(input: {
  series: readonly FitSeries[];
  selection: FitSelection;
  moe: MoeInfo | null;
  machine: MachineClass;
}): string {
  const { series, selection, moe, machine } = input;
  const sel = series.find((s) => s.id === selection.seriesId);
  const p = sel?.points.find((x) => x.ctx === selection.ctx);
  if (!sel || !p) return "";
  const at = ctxLabel(p.ctx);
  if (!fits(p)) {
    const why = p.reason === "ram" ? "system RAM" : machine === "dgpu" ? "VRAM" : "memory";
    return `At ${at}, ${sel.label} doesn't fit: it needs more ${why} than this machine has${p.error ? ` (${p.error})` : ""}.`;
  }
  const lossy = sel.kind === "kv" && (isQuantizedKv(sel.ctk) || isQuantizedKv(sel.ctv));
  const quality = lossy ? " Quantized KV can slightly reduce output quality." : "";
  const inferred = p.inferred ? " (inferred, not called)" : "";
  if (machine !== "dgpu") {
    const def = series.find((s) => s.kind === "default")?.points.find((x) => x.ctx === p.ctx);
    if (sel.kind !== "default" && def && def.total_mib != null && p.total_mib != null) {
      const saved = def.total_mib - p.total_mib;
      return `At ${at}, ${sel.label} needs ${gb(p.total_mib)}${saved > 50 ? `, ${gb(saved)} less than the default` : ""}.${quality}${inferred}`;
    }
    return `At ${at}, ${sel.label} needs ${p.total_mib != null ? gb(p.total_mib) : "an unknown amount of memory"}.${quality}${inferred}`;
  }
  const y = pointY(p, moe);
  const unit = moe?.isMoe && y.band === "main" ? "expert layer" : "layer";
  const ramPart = (mib: number | null) => (mib != null && mib > 50 ? `${gb(mib)} in system RAM` : "nothing in system RAM");
  if (sel.kind === "default") {
    const total = p.layers_total ?? 0;
    const lay = moe?.isMoe ? describeFitPlacement(p, moe) : `${p.layers_gpu} of ${total} layers on the GPU`;
    return `At ${at}, the default cache fits ${lay}, with ${ramPart(p.host_used_mib)}.${inferred}`;
  }
  const def = series.find((s) => s.kind === "default")?.points.find((x) => x.ctx === p.ctx);
  if (!def || !fits(def)) {
    return `At ${at}, ${sel.label} fits where the default cache doesn't — ${describeFitPlacement(p, moe)}.${quality}${inferred}`;
  }
  const dy = pointScore(p, moe) - pointScore(def, moe);
  const dRam = (def.host_used_mib ?? 0) - (p.host_used_mib ?? 0);
  const parts: string[] = [];
  // pointScore puts every expert-phase point 10,000 above every dense-phase
  // one, so a gap that large means the two sit in different phases.
  if (dy > 5000) parts.push("keeps experts on the GPU where the default can't");
  else if (dy < -5000) parts.push("needs every expert on the CPU where the default keeps some on the GPU");
  else if (dy > 0) parts.push(`fits ${dy} more ${unit}${dy === 1 ? "" : "s"} than the default`);
  else if (dy < 0) parts.push(`fits ${-dy} fewer ${unit}${dy === -1 ? "" : "s"} than the default`);
  else parts.push("fits the same layers as the default");
  if (dRam > 50) parts.push(`saves ${gb(dRam)} of RAM`);
  else if (dRam < -50) parts.push(`uses ${gb(-dRam)} more RAM`);
  return `At ${at}, ${sel.label} ${parts.join(" and ")}.${quality}${inferred}`;
}

export function describeFitPlacement(p: FitPoint, moe: MoeInfo | null): string {
  const total = p.layers_total ?? 0;
  if (!moe?.isMoe) {
    if (p.layers_gpu == null) return "unknown placement";
    return p.layers_gpu >= total && total > 0 ? `all ${total} layers on GPU` : `${p.layers_gpu} of ${total} layers on GPU`;
  }
  // Fit's overflow count includes the output pseudo-layer, so the exact
  // ladder state (which blocks, which boundary fraction) is described instead.
  const state = fromFit({ ngl: p.ngl, ot: p.ot }, moe);
  if (state) return describePlacement(state, moe);
  const y = pointY(p, moe);
  if (y.band === "dense") return `${p.layers_gpu} layers on GPU, all experts on CPU`;
  if (p.overflow_layers === 0) return `all ${total} layers on GPU, experts included`;
  return `all ${total} layers · experts of about ${p.overflow_layers} on CPU`;
}

// --- confirm step candidates -------------------------------------------------

export function candidateFromPoint(p: FitPoint, label: string, source: SpeedCandidate["source"], threads: ThreadArgs | null): SpeedCandidate {
  const ngl = p.ngl == null || p.ngl < 0 ? p.layers_total ?? 999 : p.ngl;
  return {
    label,
    source,
    ngl,
    ot: p.ot,
    ctk: p.ctk,
    ctv: p.ctv,
    fa: p.fa,
    threads,
    predicted: { dev_used_mib: p.dev_used_mib, host_used_mib: p.host_used_mib, layers_gpu: p.layers_gpu },
  };
}

/**
 * Up to three suggestions at the target context: the most layers with an f16
 * cache, the most layers with a quantized cache, and the setup that reaches
 * the longest context (its placement at the target). Duplicates merge.
 */
export function proposeCandidates(input: {
  series: readonly FitSeries[];
  targetCtx: number;
  moe: MoeInfo | null;
  threads: ThreadArgs | null;
}): SpeedCandidate[] {
  const { series, moe, threads } = input;
  const stops = [...new Set(series.flatMap((s) => s.points.map((p) => p.ctx)))];
  const at = targetStop(stops, input.targetCtx);
  if (at == null) return [];
  const bestOf = (list: readonly FitSeries[]) => {
    let best: { s: FitSeries; p: FitPoint } | null = null;
    for (const s of list) {
      const p = s.points.find((x) => x.ctx === at);
      if (!p || !fits(p)) continue;
      if (!best || pointScore(p, moe) > pointScore(best.p, moe) || (pointScore(p, moe) === pointScore(best.p, moe) && (p.host_used_mib ?? 0) < (best.p.host_used_mib ?? 0))) {
        best = { s, p };
      }
    }
    return best;
  };
  const out: SpeedCandidate[] = [];
  const f16 = bestOf(series.filter((s) => s.kind !== "kv"));
  if (f16) out.push(candidateFromPoint(f16.p, `Most layers · ${f16.s.label}`, "fit", threads));
  const kv = bestOf(series.filter((s) => s.kind === "kv"));
  if (kv && (!f16 || pointScore(kv.p, moe) > pointScore(f16.p, moe) || (kv.p.host_used_mib ?? 0) < (f16.p.host_used_mib ?? 0) - 256)) {
    out.push(candidateFromPoint(kv.p, `Most layers · ${kv.s.label}`, "fit_kv", threads));
  }
  // The setup whose own path reaches the longest fitting context.
  let longest: { s: FitSeries; max: number; score: number } | null = null;
  for (const s of series) {
    const max = Math.max(0, ...s.points.filter(fits).map((p) => p.ctx));
    const atMax = s.points.find((p) => p.ctx === max);
    const score = atMax ? pointScore(atMax, moe) : -1;
    if (!longest || max > longest.max || (max === longest.max && score > longest.score)) longest = { s, max, score };
  }
  if (longest) {
    const p = longest.s.points.find((x) => x.ctx === at);
    if (p && fits(p)) out.push(candidateFromPoint(p, `Longest context · ${longest.s.label} (fits ${ctxLabel(longest.max)})`, "fit_ctx", threads));
  }
  const seen = new Set<string>();
  return out.filter((c) => {
    const k = JSON.stringify([c.ngl, c.ot, c.ctk, c.ctv, c.fa]);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Fit-map job cost in seconds, for the run button (measured ~0.5 s for a
 * call that needs no change, ~4 s for a partial-offload search). */
export function estimateFitMapSeconds(input: { stops: number; machine: MachineClass; kvDetect: boolean; kvFit: boolean; kvKnown: boolean }): number {
  const faModes = 2;
  const perCall = input.machine === "dgpu" ? 3.5 : 0.6;
  let s = input.stops * faModes * perCall;
  if (input.kvDetect && !input.kvKnown) s += 128 * 0.5;
  if (input.kvFit) s += 12 * input.stops * perCall * 0.7;
  return s;
}
