// A header-only stand-in for one llama-fit-params call: from a model's GGUF
// header facts (per-block weight bytes, per-block cache kinds, vocabulary) and
// a machine's free memory, predict what `--fit` answers -- the fallback for
// when llama-fit-params cannot run on the model (worker/src/headerFit.ts).
// docs/plans/HEADER_FIT_PLAN.md; calibrated against llama-fit-params b9503
// (Vulkan) on 12 models, ~300 fixed placements (-ngl / -ncmoe) at 4k..256k.
//
// Placement, as llama.cpp does it: -ngl N puts the OUTPUT layer on the GPU
// first, then the last N-1 transformer blocks; "layers" counts the output
// layer as one, so a full offload is n_layer + 1. A MoE model first keeps
// every block's attention on the GPU and moves routed-expert tensors to the
// CPU from the last block backward; when even that does not fit, it drops
// blocks with every GPU block's experts kept CPU-side ("dense-only" layers).
// The fit target is device memory used <= free - margin.
//
// The compute buffer is the largest of the graph's stages (fit reserves the
// prompt graph for n_ubatch = 512 tokens with 512 outputs):
//   logits      vocab x 512 x f32, plus a copy of the output weight when the
//               output layer stays on the CPU (op offload runs CPU-resident
//               matmuls on the GPU from a copied weight);
//   attention   the KQ mask (512 x ctx f16) + graph scratch;
//   CPU cache   a full-attention block on the CPU: its K/V are copied in,
//               (largest full layer's K+V + mask) x ctx;
//   experts     a GPU block whose experts stay on the CPU: its expert tensors
//               are copied in for prompt processing, one at a time -- ggml's
//               scheduler starts a new split at every CPU-resident weight so
//               the previous copy's memory is reused -- so the largest single
//               expert tensor + the mask + 120 MiB. Fit on 309 llama-fit-params
//               b12058 answers (Vulkan + ROCm agree) with experts offloaded:
//               mean |error| 47 MiB, p90 +43. A few families hold two copies
//               (Ling-3.0-flash, MiniMax-M2 HQ4_K: up to 740 MiB more).
//               Rows without max_moe fall back to the whole block's experts,
//               the earlier b9503 reading (5022 MiB on a 5058 MiB GLM-5.3-Flash
//               block).
// Measured spread: within ~100 MiB on most placements; the attention stage of
// some MoE graphs (Qwen3.6-35B, Gemma-4 26B) runs ~250 MiB above it at 256k.

import type { KvLayout, TensorLayerBreakdown } from "./vramEstimate.js";
import { hasTiedOutput } from "./vramEstimate.js";

const MIB = 1048576;
const UBATCH = 512;
// Graph scratch beside the logits tensor, MiB.
const LOGITS_SCRATCH_MIB = 12;
// Attention-stage scratch beside the mask, MiB: 240 on 2-9B-class models,
// a few MiB on tiny ones (SmolLM2-135M: 104 MiB compute at 4k, mask + 13 at
// 128k) -- capped at half the logits tensor, which tracks model width.
const ATTENTION_SCRATCH_MIB = 240;
// Compute buffer of a model with no per-token cache (embedding, encoder,
// classifier): no mask growth and no full-vocabulary logits -- 36-82 MiB
// measured across contexts on LFM2-600M, EmbeddingGemma-2, Clef-0.8B.
const NO_CACHE_COMPUTE_MIB = 64;
// CPU-cache stage scratch, MiB.
const CPU_CACHE_SCRATCH_MIB = 64;
// MoE graphs pay more per token in that stage than the copied K/V + mask
// explain: +2 KiB measured on Gemma-4 26B-A4B, +0..1 KiB on Qwen3.6-35B-A3B
// (quant-dependent). The larger value, so the fallback errs to fewer layers.
const MOE_CPU_CACHE_EXTRA_BYTES = 2048;
// Experts-stage scratch beside the copied expert tensor and the mask, MiB.
const EXPERT_COPY_SCRATCH_MIB = 120;
// Host compute buffer beside the mask-sized term, MiB.
const HOST_COMPUTE_MIB = 16;

export interface FitSimInput {
  breakdown: TensorLayerBreakdown;
  layout: KvLayout;
  nLayer: number;
  ctx: number;
  freeMib: number;
  marginMib: number;
  /** Bytes per KV element (f16 = 2). */
  bytesK?: number;
  bytesV?: number;
  /** "fit" (default): llama.cpp --fit's search. "plain": the largest plain
   * -ngl whose device claim is <= free - margin (no expert moves). */
  mode?: "fit" | "plain";
}

export interface FitSimPoint {
  verdict: "full" | "partial" | "cpu_only";
  /** -ngl: GPU layers, the output layer counted as one. */
  layers_gpu: number;
  layers_total: number;
  /** Blocks whose routed experts stay on the CPU. */
  overflow_layers: number;
  dev_used_mib: number;
  host_used_mib: number;
  kv_mib: number;
  compute_mib: number;
}

/** How much of the boundary block (the frontmost of the `movedBlocks`) goes
 * to the CPU, as llama.cpp --fit splits it (common/fit.cpp): "gate" moves its
 * ffn_down* tensors, "up" its ffn_gate* + ffn_down*, "attn" every ffn_gate* /
 * ffn_up* / ffn_down* tensor -- dense, shared and routed alike, so "attn" also
 * moves the router and shared experts. Unset: its routed experts only. */
export type BoundarySplit = "gate" | "up" | "attn";

/** A placement: the output layer + the last `blocks` blocks on the GPU
 * (output only when `output`), experts of `movedBlocks` GPU blocks (from the
 * back) on the CPU, the frontmost of them split per `boundary`.
 * `movedBytes` (research scripts only) moves that many more bytes of the next
 * block's experts. */
export interface Placement {
  output: boolean;
  blocks: number;
  movedBlocks: number;
  boundary?: BoundarySplit;
  movedBytes?: number;
}

/** Whether the row carries the per-block ffn group sums a boundary split needs. */
function hasFfnGroups(b: TensorLayerBreakdown): boolean {
  const n = b.dense.length;
  return b.ffn_gate?.length === n && b.ffn_up?.length === n && b.ffn_down?.length === n;
}

/** Bytes of block `i` a boundary split puts on the CPU. */
function boundaryCpuBytes(b: TensorLayerBreakdown, i: number, split: BoundarySplit): number {
  const gate = b.ffn_gate?.[i] ?? 0;
  const up = b.ffn_up?.[i] ?? 0;
  const down = b.ffn_down?.[i] ?? 0;
  return split === "gate" ? down : split === "up" ? gate + down : gate + up + down;
}

function swaCells(window: number, ctx: number): number {
  return Math.min(ctx, Math.ceil((window + UBATCH) / 256) * 256);
}

/** Per-block cache bytes at `ctx`, and which blocks are full attention.
 * Rows without `kinds` spread the totals evenly over the blocks. */
export function blockCaches(layout: KvLayout, nLayer: number, ctx: number, bytesK = 2, bytesV = 2): { bytes: number[]; full: boolean[] } {
  if (layout.no_cache) return { bytes: new Array(nLayer).fill(0), full: new Array(nLayer).fill(false) };
  const full = (layout.full_k * bytesK + layout.full_v * bytesV) * ctx;
  const swa = (layout.swa_k * bytesK + layout.swa_v * bytesV) * swaCells(layout.window, ctx);
  const state = layout.state_bytes ?? 0;
  const kinds = layout.kinds && layout.kinds.length === nLayer ? layout.kinds : null;
  if (!kinds) {
    const each = (full + swa + state) / Math.max(1, nLayer);
    return { bytes: new Array(nLayer).fill(each), full: new Array(nLayer).fill(layout.full_k > 0) };
  }
  const count = (c: string) => [...kinds].filter((k) => k === c).length;
  const per: Record<string, number> = {
    f: full / Math.max(1, count("f")),
    s: swa / Math.max(1, count("s")),
    r: state / Math.max(1, count("r")),
    n: 0,
  };
  return { bytes: [...kinds].map((k) => per[k] ?? 0), full: [...kinds].map((k) => k === "f") };
}

type SimModel = {
  n: number;
  budget: number;
  evaluate: (p: Placement) => { devMib: number };
  finish: (p: Placement) => FitSimPoint;
};

function buildModel(input: FitSimInput): SimModel | null {
  const { breakdown: b, layout, ctx } = input;
  const n = input.nLayer;
  if (n <= 0 || b.dense.length !== n || b.moe.length !== n) return null;
  const bK = input.bytesK ?? 2;
  const bV = input.bytesV ?? 2;
  const caches = blockCaches(layout, n, ctx, bK, bV);
  const tied = hasTiedOutput(b);
  // The output layer's weights: output.weight, or the tied copy of the token
  // embedding that lands on the GPU beside it.
  const outputBytes = b.output + (tied ? b.embed : 0);
  const outputWeightMib = (tied ? (b.max_embed ?? b.embed) : (b.max_output ?? b.output)) / MIB;
  const maskMib = (ctx * UBATCH * 2) / MIB;
  const logitsMib = (layout.vocab ? (layout.vocab * UBATCH * 4) / MIB : 0) + LOGITS_SCRATCH_MIB;
  const fullMaxBytes = (layout.full_max ?? (layout.full_k + layout.full_v) / Math.max(1, caches.full.filter(Boolean).length)) * Math.max(bK, bV);
  // What the experts stage copies in: the largest expert tensor, or on rows
  // read before max_moe existed, the whole block's experts.
  const expertCopyMib = b.max_moe != null && b.max_moe > 0 ? b.max_moe / MIB + EXPERT_COPY_SCRATCH_MIB : Math.max(0, ...b.moe) / MIB;
  const moeModel = b.moe.some((x) => x > 0);
  const totalWeights = b.dense.reduce((a, x) => a + x, 0) + b.moe.reduce((a, x) => a + x, 0) + b.embed + b.output + b.other;
  const totalCacheMib = caches.bytes.reduce((a, x) => a + x, 0) / MIB;
  const budget = input.freeMib - input.marginMib;

  const computeMib = (p: Placement): number => {
    const from = n - p.blocks;
    if (layout.no_cache) return NO_CACHE_COMPUTE_MIB + (p.output ? 0 : outputWeightMib);
    const attentionScratch = Math.min(ATTENTION_SCRATCH_MIB, (logitsMib - LOGITS_SCRATCH_MIB) / 2);
    let stage = Math.max(p.output ? logitsMib : logitsMib + outputWeightMib, maskMib + attentionScratch);
    if (caches.full.some((f, i) => f && i < from)) {
      const perToken = fullMaxBytes + UBATCH * 2 + (moeModel ? MOE_CPU_CACHE_EXTRA_BYTES : 0);
      stage = Math.max(stage, (ctx * perToken) / MIB + CPU_CACHE_SCRATCH_MIB);
    }
    if ((p.movedBlocks > 0 || (p.movedBytes ?? 0) > 0)) stage = Math.max(stage, expertCopyMib + maskMib);
    return stage;
  };

  const evaluate = (p: Placement) => {
    const from = n - p.blocks;
    let weights = p.output ? outputBytes : 0;
    let cache = 0;
    for (let i = from; i < n; i++) {
      weights += b.dense[i] + b.moe[i];
      cache += caches.bytes[i];
    }
    for (let i = n - p.movedBlocks; i < n; i++) weights -= b.moe[i];
    weights -= p.movedBytes ?? 0;
    if (p.boundary && p.movedBlocks > 0) {
      const i = n - p.movedBlocks;
      weights += b.moe[i] - boundaryCpuBytes(b, i, p.boundary);
    }
    const compute = computeMib(p);
    return { devMib: weights / MIB + cache / MIB + compute, weights, cacheMib: cache / MIB, compute };
  };

  const finish = (p: Placement): FitSimPoint => {
    const ev = evaluate(p);
    // Everything not on the device; the tied embedding keeps its input copy in
    // RAM, so its device copy is extra.
    const hostWeights = totalWeights - ev.weights + (p.output && tied ? b.embed : 0);
    const layersGpu = p.blocks + (p.output ? 1 : 0);
    const overflow = p.movedBlocks + ((p.movedBytes ?? 0) > 0 ? 1 : 0);
    return {
      verdict: layersGpu === 0 ? "cpu_only" : layersGpu === n + 1 && overflow === 0 ? "full" : "partial",
      layers_gpu: layersGpu,
      layers_total: n + 1,
      overflow_layers: overflow,
      dev_used_mib: Math.round(ev.devMib),
      host_used_mib: Math.round(hostWeights / MIB + (totalCacheMib - ev.cacheMib) + (layout.no_cache ? 0 : maskMib) + HOST_COMPUTE_MIB),
      kv_mib: Math.round(totalCacheMib),
      compute_mib: Math.round(ev.compute),
    };
  };

  return { n, budget, evaluate, finish };
}

export function simulateFit(input: FitSimInput): FitSimPoint | null {
  const m = buildModel(input);
  if (!m) return null;
  const { n, budget, evaluate, finish } = m;
  const b = input.breakdown;
  const fits = (p: Placement) => evaluate(p).devMib <= budget;
  // -ngl k: the output layer once k >= 1, then k - 1 blocks.
  const plain = (k: number, movedBlocks = 0): Placement => ({ output: k >= 1, blocks: Math.max(0, Math.min(n, k - 1)), movedBlocks: Math.min(movedBlocks, Math.max(0, k - 1)) });

  if (input.mode === "plain") {
    for (let k = n + 1; k >= 0; k--) if (fits(plain(k))) return finish(plain(k));
    return finish(plain(0));
  }

  const all = plain(n + 1);
  if (fits(all)) return finish(all);

  const hasMoe = b.moe.some((x) => x > 0);
  if (hasMoe) {
    // Every block on the GPU, routed experts moved from the back one whole
    // block at a time. Then fit's "one more layer" step on the boundary block
    // (the frontmost moved one): try keeping its up tensors on the GPU ("up"),
    // and if that fits its gate tensors too ("gate"); when "up" does not fit,
    // it tries "attn", which moves that block's whole FFN, router and shared
    // experts included. Those tries need used < budget, not <=.
    //
    // Fit never moves the experts of the front-most GPU block: it lays layers
    // out with the output layer as the LAST index (n), so the override it
    // writes for "the output layer" is blk.<n>, a block that does not exist,
    // and the one it means to keep whole is the front block instead (seen in
    // b12058's -ot: a 32-block Mixnu ends with blk\.32\.ffn_...=CPU). The same
    // shift is why its "(K overflowing)" counts one more than the blocks it
    // moves; overflow_layers here counts real blocks.
    const below = (p: Placement) => evaluate(p).devMib < budget;
    for (let moved = 0; moved <= n - 1; moved++) {
      const p = { ...all, movedBlocks: moved };
      if (!fits(p)) continue;
      if (moved === 0 || !hasFfnGroups(b)) return finish(p);
      const up: Placement = { ...p, boundary: "up" };
      if (below(up)) {
        const gate: Placement = { ...p, boundary: "gate" };
        return finish(below(gate) ? gate : up);
      }
      const attn: Placement = { ...p, boundary: "attn" };
      return finish(below(attn) ? attn : p);
    }
    // Dense-only: drop blocks, every GPU block's experts on the CPU but the
    // front-most one's (see above).
    for (let k = n; k >= 1; k--) {
      const p = plain(k, Math.max(0, k - 2));
      if (fits(p)) return finish(p);
    }
    return finish(plain(0));
  }

  for (let k = n; k >= 1; k--) if (fits(plain(k))) return finish(plain(k));
  return finish(plain(0));
}

/** The memory picture of one fixed placement (no search) -- what
 * llama-fit-params reports for an explicit -ngl / -ncmoe. */
export function simulateFitAt(input: Omit<FitSimInput, "freeMib" | "marginMib" | "mode">, p: Placement): FitSimPoint | null {
  const m = buildModel({ ...input, freeMib: 0, marginMib: 0 });
  return m ? m.finish(p) : null;
}
