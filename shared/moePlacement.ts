// Step M of the optimization flow (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md):
// MoE detection and the placement ladder the real-load searches walk.
//
// The ladder copies llama.cpp's own fit (common/fit.cpp), so a fit answer maps
// onto it exactly and the real-load search can start from it:
//   * dense phase -- reached only when even "all routed experts on CPU"
//     doesn't fit: -ngl n from the back, every GPU block keeps its experts on
//     the CPU. Between n and n+1 sits ATTN on the next block (attention on
//     GPU, its whole FFN incl. router and shared experts on CPU).
//   * expert phase -- every layer on the GPU, experts of MoE blocks 0..e-1 on
//     the GPU and the rest on the CPU, e = 0 .. |moeLayers|. Between e and e+1
//     sit UP then GATE on block e.
// Memory grows strictly along the list (measured 2026-10-04 on two models).
//
// Layer indexing follows fit: "layers" are blocks 0..n_layer-1 plus the
// output layer, and ngl counts from the back, so n_layer + 1 = every layer.

import type { TensorLayerBreakdown } from "./vramEstimate.js";

export interface MoeInfo {
  isMoe: boolean;
  /** Block indices holding routed-expert tensors, ascending. */
  moeLayers: number[];
  /** Mean expert bytes per MoE block. */
  expertBytesPerLayer: number;
  /** Transformer blocks (n_layer). */
  nBlocks: number;
}

export function detectMoe(tlb: TensorLayerBreakdown | null | undefined): MoeInfo {
  if (!tlb || !Array.isArray(tlb.moe)) return { isMoe: false, moeLayers: [], expertBytesPerLayer: 0, nBlocks: tlb?.dense?.length ?? 0 };
  const moeLayers = tlb.moe.map((b, i) => (b > 0 ? i : -1)).filter((i) => i >= 0);
  const total = moeLayers.reduce((a, i) => a + tlb.moe[i], 0);
  return {
    isMoe: moeLayers.length > 0,
    moeLayers,
    expertBytesPerLayer: moeLayers.length ? total / moeLayers.length : 0,
    nBlocks: tlb.dense.length,
  };
}

export type Boundary = "none" | "ATTN" | "UP" | "GATE";

export interface PlacementState {
  phase: "dense" | "expert";
  /** -ngl passed to llama.cpp. */
  ngl: number;
  /** Expert phase: MoE blocks 0..e-1 keep their experts on the GPU. */
  expertsGpu: number;
  boundary: Boundary;
}

const EXPS = (i: number) => `blk\\.${i}\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU`;
const FRACTION: Record<Exclude<Boundary, "none">, (i: number) => string> = {
  GATE: (i) => `blk\\.${i}\\.ffn_down.*=CPU`,
  UP: (i) => `blk\\.${i}\\.ffn_(gate|gate_up|down).*=CPU`,
  ATTN: (i) => `blk\\.${i}\\.ffn_(gate|up|gate_up|down).*=CPU`,
};

/** Every layer on the GPU. */
export function nglAll(nBlocks: number): number {
  return nBlocks + 1;
}

/** Blocks resident on the GPU at -ngl n (output layer counts as one). */
export function gpuBlocks(nBlocks: number, ngl: number): number[] {
  const out: number[] = [];
  for (let i = Math.max(0, nBlocks - ngl + 1); i < nBlocks; i++) out.push(i);
  return out;
}

/**
 * The ordered placement states for one model, smallest memory first. Dense
 * models get the plain -ngl 0..all list. `dropFractions` removes boundary
 * states fit-params didn't size strictly above the state below them (UP on
 * fused gate_up models, for one).
 */
export function placementLadder(
  moe: MoeInfo,
  opts: { dropFractions?: ReadonlySet<Boundary> } = {}
): PlacementState[] {
  const all = nglAll(moe.nBlocks);
  if (!moe.isMoe) {
    return Array.from({ length: all + 1 }, (_, n) => ({ phase: "dense", ngl: n, expertsGpu: 0, boundary: "none" }));
  }
  const drop = opts.dropFractions ?? new Set<Boundary>();
  const moeSet = new Set(moe.moeLayers);
  const out: PlacementState[] = [];
  for (let n = 0; n < all; n++) {
    out.push({ phase: "dense", ngl: n, expertsGpu: 0, boundary: "none" });
    // ATTN on the block that would join next (only meaningful if it's a MoE
    // block -- a dense block's next step is simply n+1).
    const next = moe.nBlocks - n;
    if (n >= 1 && next >= 0 && moeSet.has(next) && !drop.has("ATTN")) {
      out.push({ phase: "dense", ngl: n, expertsGpu: 0, boundary: "ATTN" });
    }
  }
  for (let e = 0; e <= moe.moeLayers.length; e++) {
    out.push({ phase: "expert", ngl: all, expertsGpu: e, boundary: "none" });
    if (e < moe.moeLayers.length) {
      if (!drop.has("UP")) out.push({ phase: "expert", ngl: all, expertsGpu: e, boundary: "UP" });
      if (!drop.has("GATE")) out.push({ phase: "expert", ngl: all, expertsGpu: e, boundary: "GATE" });
    }
  }
  return out;
}

/** llama.cpp args for one state, in fit's own -ot form (comma-joined). */
export function toArgs(state: PlacementState, moe: MoeInfo): { ngl: number; ot: string | null } {
  if (!moe.isMoe) return { ngl: state.ngl, ot: null };
  const moeSet = new Set(moe.moeLayers);
  const entries: string[] = [];
  if (state.phase === "dense") {
    const blocks = gpuBlocks(moe.nBlocks, state.ngl);
    if (state.boundary === "ATTN") {
      const k = moe.nBlocks - state.ngl;
      entries.push(FRACTION.ATTN(k));
      for (const b of blocks) if (moeSet.has(b)) entries.push(EXPS(b));
      return { ngl: state.ngl + 1, ot: entries.join(",") };
    }
    for (const b of blocks) if (moeSet.has(b)) entries.push(EXPS(b));
    return { ngl: state.ngl, ot: entries.length ? entries.join(",") : null };
  }
  const cpuExperts = moe.moeLayers.slice(state.expertsGpu);
  cpuExperts.forEach((b, j) => {
    if (j === 0 && state.boundary !== "none") entries.push(FRACTION[state.boundary](b));
    else entries.push(EXPS(b));
  });
  return { ngl: state.ngl, ot: entries.length ? entries.join(",") : null };
}

function stateEquals(a: PlacementState, b: PlacementState): boolean {
  return a.phase === b.phase && a.ngl === b.ngl && a.expertsGpu === b.expertsGpu && a.boundary === b.boundary;
}

export function ladderIndex(ladder: readonly PlacementState[], s: PlacementState): number {
  return ladder.findIndex((x) => stateEquals(x, s));
}

const ENTRY_RE = /^blk\\\.(\d+)\\\.(.+)=CPU$/;

function classifyEntry(rest: string): "exps" | Exclude<Boundary, "none"> | null {
  if (rest === "ffn_(up|down|gate_up|gate)_(ch|)exps") return "exps";
  if (rest === "ffn_down.*") return "GATE";
  if (rest === "ffn_(gate|gate_up|down).*") return "UP";
  if (rest === "ffn_(gate|up|gate_up|down).*") return "ATTN";
  return null;
}

/**
 * Maps a fit answer (fitted -ngl and -ot) onto a ladder state; null for a
 * shape fit never emits. Expert-phase answers map exactly. In the dense phase
 * fit can keep one GPU block's experts on the GPU (measured 2026-10-08: at
 * -ngl 5 on Qwen3.6-35B-A3B fit offloads 37..40 while blocks 36..39 are the
 * GPU blocks), which sits between two ladder states; it maps to the lower one
 * (-ngl n, every GPU block dense-only) -- a seed the real-load search moves up
 * from, never a stored answer.
 */
export function fromFit(args: { ngl: number | null; ot: string | null }, moe: MoeInfo): PlacementState | null {
  if (args.ngl == null) return null;
  const all = nglAll(moe.nBlocks);
  const ngl = args.ngl < 0 ? all : Math.min(args.ngl, all);
  if (!moe.isMoe) return { phase: "dense", ngl, expertsGpu: 0, boundary: "none" };
  const entries = (args.ot ?? "")
    .split(",")
    .map((e) => e.trim())
    .filter(Boolean)
    .map((e) => {
      const m = ENTRY_RE.exec(e);
      return m ? { block: Number(m[1]), kind: classifyEntry(m[2]) } : null;
    });
  if (entries.some((e) => e == null || e.kind == null)) return null;
  const parsed = entries as { block: number; kind: "exps" | Exclude<Boundary, "none"> }[];
  const moeSet = new Set(moe.moeLayers);
  if (ngl < all) {
    const attn = parsed.find((e) => e.kind === "ATTN");
    return attn ? { phase: "dense", ngl: ngl - 1, expertsGpu: 0, boundary: "ATTN" } : { phase: "dense", ngl, expertsGpu: 0, boundary: "none" };
  }
  // Expert phase: the first MoE block with an entry is the boundary.
  const real = parsed.filter((e) => moeSet.has(e.block)).sort((a, b) => a.block - b.block);
  if (real.length === 0) return { phase: "expert", ngl: all, expertsGpu: moe.moeLayers.length, boundary: "none" };
  const first = real[0];
  const e = moe.moeLayers.indexOf(first.block);
  const boundary: Boundary = first.kind === "exps" ? "none" : first.kind;
  // ATTN on the front block with every layer on the GPU is still the dense
  // phase: its step up is "every layer dense-only" (expert phase, e = 0).
  if (boundary === "ATTN") return e === 0 ? { phase: "dense", ngl: all - 1, expertsGpu: 0, boundary: "ATTN" } : null;
  return { phase: "expert", ngl: all, expertsGpu: e, boundary };
}

/** Plain-language placement, used by the result cards and tables. */
export function describePlacement(state: PlacementState, moe: MoeInfo): string {
  const all = nglAll(moe.nBlocks);
  if (!moe.isMoe) return state.ngl >= all ? `all ${all} layers on GPU` : `${state.ngl} of ${all} layers on GPU`;
  if (state.phase === "dense") {
    const base = `${state.ngl} layers on GPU, all experts on CPU`;
    return state.boundary === "ATTN" ? `${base} (+ attention of 1 more layer)` : base;
  }
  const cpu = moe.moeLayers.length - state.expertsGpu;
  if (cpu === 0) return `all ${all} layers on GPU, experts included`;
  const first = moe.moeLayers[state.expertsGpu];
  const last = moe.moeLayers[moe.moeLayers.length - 1];
  const span = first === last ? `layer ${first}` : `layers ${first}–${last}`;
  const frac = state.boundary === "GATE" ? ` (layer ${first}: up+gate on GPU)` : state.boundary === "UP" ? ` (layer ${first}: up on GPU)` : "";
  return `all ${all} layers on GPU · experts of ${span} on CPU${frac}`;
}
