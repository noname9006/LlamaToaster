// Cheap, honestly-approximate VRAM-fit estimate shared by the client's
// pre-flight banner (NewRun.tsx, warning + auto-cap) and the worker's
// post-run discrepancy check (index.ts's finalizeSweepItemResult) -- one
// formula so the two numbers can never silently drift apart. Motivated by a
// diagnosed real case: a 14.2GB MoE model on a 10GB GPU had llama.cpp report
// "31/31 layers offloaded" (no error) while the actually-measured
// vram_peak_mib stayed under 1GB the whole run -- Windows' NVIDIA driver had
// silently backed the overcommitted CUDA allocation with system RAM instead
// of erroring, and llama.cpp's own "offloaded X/Y" line only ever reflects
// buffer *assignment*, never actual VRAM residency (the OS hides the
// fallback from the CUDA runtime itself, so there's no more-detailed log
// line to go parse instead).

export const VRAM_ESTIMATE_FIXED_OVERHEAD_MIB = 512;
// Deliberately generous: the real diagnosed case above was off by roughly
// 25x, so flagging anything under half the estimate still catches that
// failure mode while staying quiet on ordinary estimate error -- the flat
// per-layer average below is never exact, especially for a MoE model whose
// real per-layer footprint isn't uniform.
export const VRAM_DISCREPANCY_RATIO = 0.5;

const BYTES_PER_MIB = 1024 * 1024;

export interface VramNeedEstimateInput {
  modelSizeBytes: number;
  // n_layer + 1 (the output layer) -- matches llama.cpp's own "offloaded
  // X/Y" runtime convention, see shared/types.ts's ResultRow.total_model_layers.
  totalModelLayers: number;
  requestedNgl: number;
}

// Flat per-layer average of the whole on-disk file -- deliberately simple,
// not a real per-tensor breakdown. Weakest for MoE models specifically: a
// MoE layer's true footprint depends on how many of ITS experts actually end
// up GPU-resident (see --n-cpu-moe), which this formula has no way to see --
// it just assumes every layer is the same size as the file-wide average.
// Good enough for an order-of-magnitude warning, not a precise budget. Null
// whenever an input can't produce a meaningful estimate (nothing requested,
// or a size/layer-count of 0) -- callers treat null as "don't show a banner"
// rather than a false "fits" or "doesn't fit".
export function estimateVramNeededMib(input: VramNeedEstimateInput): number | null {
  if (input.modelSizeBytes <= 0 || input.totalModelLayers <= 0 || input.requestedNgl <= 0) return null;
  const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
  const neededBytes = bytesPerLayer * input.requestedNgl + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB * BYTES_PER_MIB;
  return Math.round(neededBytes / BYTES_PER_MIB);
}

// Inverse of the above: the largest ngl whose estimate still fits under
// freeVramMib. Used both by the pre-flight banner (a starting suggestion)
// and the auto-cap action (NewRun.tsx). Clamped to [0, totalModelLayers].
export function estimateSafeNgl(modelSizeBytes: number, totalModelLayers: number, freeVramMib: number): number {
  if (modelSizeBytes <= 0 || totalModelLayers <= 0) return 0;
  const bytesPerLayer = modelSizeBytes / totalModelLayers;
  const usableBytes = Math.max(0, freeVramMib - VRAM_ESTIMATE_FIXED_OVERHEAD_MIB) * BYTES_PER_MIB;
  const safeLayers = Math.floor(usableBytes / bytesPerLayer);
  return Math.max(0, Math.min(totalModelLayers, safeLayers));
}

// True when a claimed-successful offload's OBSERVED vram_peak_mib is
// implausibly low relative to what estimateVramNeededMib predicted for that
// many requested layers -- the signal that llama.cpp's own "offloaded X/Y"
// line was honest about intent but wrong about outcome (see this file's top
// comment). estimatedMib must already be a non-null estimate (callers skip
// this check entirely when estimateVramNeededMib returned null -- nothing
// to compare against).
export function isVramDiscrepancy(estimatedMib: number, observedPeakMib: number): boolean {
  return observedPeakMib < estimatedMib * VRAM_DISCREPANCY_RATIO;
}

export interface ResidentGpuLayersInput {
  modelSizeBytes: number;
  totalModelLayers: number;
  claimedNgl: number;
  // Peak VRAM reading sampled while this run's process was live -- the same
  // figure isVramDiscrepancy already judged implausibly low.
  observedPeakMib: number;
  // Whole-adapter VRAM already in use before the process spawned
  // (gpu_memory_total minus vram_free_before), so only the run-attributable
  // growth above that baseline counts toward residency. Null when either
  // component wasn't readable -- an un-attributable peak can't be turned
  // into a number here without counting other processes' allocations as
  // model layers, so callers get null back instead of a guess.
  baselineUsedMib: number | null;
}

// Inverse problem of estimateVramNeededMib above: instead of predicting how
// much VRAM a claimed offload needs, derive how many of the claimed layers
// were ACTUALLY resident from the telemetry the sampler did see --
//
//   resident ~= floor((vram_peak - baseline_used) / per-layer file share)
//
// clamped to [0, claimedNgl]. Only ever meaningful once isVramDiscrepancy has
// fired (when it hasn't, llama.cpp's own claim stands and callers leave this
// unset rather than displaying a noisy near-claim figure as if it were a
// measurement). Deliberately coarse for exactly the same reasons as
// estimateVramNeededMib -- flat per-layer average, KV-cache/compute-buffer
// bytes not separable from weights, other processes allocating mid-run
// inflate the delta -- an order-of-magnitude display figure ("~0/33 actually
// resident" next to a claimed 33/33), never a precise count. Null whenever
// any input needed to attribute the peak is missing.
export function estimateResidentGpuLayers(input: ResidentGpuLayersInput): number | null {
  if (input.modelSizeBytes <= 0 || input.totalModelLayers <= 0 || input.claimedNgl <= 0) return null;
  if (!Number.isFinite(input.observedPeakMib) || input.baselineUsedMib == null) return null;
  const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
  const runDeltaMib = Math.max(0, input.observedPeakMib - input.baselineUsedMib);
  const layers = Math.floor((runDeltaMib * BYTES_PER_MIB) / bytesPerLayer);
  return Math.max(0, Math.min(input.claimedNgl, layers));
}

// Preferred sibling of estimateResidentGpuLayers above, used whenever
// worker/src/bench.ts's parseModelBufferSizes actually found llama.cpp's own
// per-backend "model buffer size" lines for this item (see that function's
// doc comment). Strictly better than the vram-sample-based estimate: instead
// of a flat file-size-per-layer average compared against an OS-level VRAM
// sample (vulnerable to other processes, driver-counter timing, and the
// unverified nvidia-smi pid-matching in worker/src/vram.ts), this divides
// llama.cpp's own reported GPU-vs-CPU buffer split proportionally across the
// claimed layers -- ground truth about which backend the allocator actually
// used, straight from the process that made the decision. Still an estimate
// (individual layers aren't equally sized, and a MoE model's real per-layer
// footprint varies with how many experts loaded -- same caveat
// estimateVramNeededMib documents), just a better-informed one. Null when
// there's nothing to divide (no claim, or llama.cpp reported zero bytes on
// every backend, which parseModelBufferSizes already turns into null rather
// than a real ModelBufferSizes for the caller to reach here with).
export function estimateResidentGpuLayersFromBufferSizes(
  claimedNgl: number,
  gpuMib: number,
  cpuMib: number
): number | null {
  const totalMib = gpuMib + cpuMib;
  if (claimedNgl <= 0 || totalMib <= 0) return null;
  const layers = Math.round((claimedNgl * gpuMib) / totalMib);
  return Math.max(0, Math.min(claimedNgl, layers));
}

// --- M1 -- Affordability inversion: given this GPU, these weights, this KV
// type -- what is the largest context that still fits and stays practical?
// Same advisory-only posture as everything else in this file: it ranks and
// annotates, it never gates a config out of a sweep.

const DEFAULT_MAX_CTX_SCRATCH_MIB = 256;
const DEFAULT_ACTIVATIONS_HEADROOM_FRAC = 0.1;

export interface MaxCtxInput {
  // VRAM total (RAM total for CPU workers).
  totalMib: number;
  // VRAM-resident weight share for the chosen placement (ngl / n-cpu-moe);
  // null when only a full-offload number exists.
  weightsMib: number | null;
  // Graphs + driver scratch; default 256.
  scratchMib?: number;
  activationsHeadroomFrac?: number;
  // Default 1; KV splits across slots.
  parallelSlots?: number;
  // Per-token KV inputs, identical fields to the forward estimator.
  nLayer: number;
  nHeadKv: number;
  headDimK?: number;
  headDimV?: number;
  // Consumed only by the dK/dV fallback below; required together for it,
  // absent => confidence drops a level.
  nEmbd?: number;
  nHead?: number;
  cacheTypeK: string;
  cacheTypeV: string;
  slidingWindow?: number;
  // M1's own stated fallback when confidence can only be "unknown": offer
  // the full-trained_ctx load as a conservative floor candidate, explicitly
  // labeled conservative -- never invent a weights number instead.
  trainedCtx?: number | null;
}

export interface MaxCtxEstimate {
  tokens: number;
  confidence: "good" | "rough" | "unknown";
  binding: "kv" | "weights-placement" | null;
  // Populated only alongside confidence "unknown", when trainedCtx was
  // supplied: the conservative floor candidate M1 names as the fallback.
  // Never a substitute estimate -- callers must label it conservative.
  conservativeFloorTokens?: number | null;
}

// Bytes per element per KV-cache value, from ggml's own block layouts
// (block size 32 for every quantized type here). The M1 worked example pins
// two of them: f16 = 2 B and q8_0 = 136/128 = 1.0625 B per element.
function bytesPerVal(cacheType: string): number | null {
  switch (cacheType) {
    case "f32":
      return 4;
    case "bf16":
    case "f16":
      return 2;
    case "q8_0":
      return 34 / 32;
    case "q5_1":
      return 24 / 32;
    case "q5_0":
      return 22 / 32;
    case "q4_1":
      return 20 / 32;
    case "q4_0":
    case "iq4_nl":
      return 18 / 32;
    default:
      return null;
  }
}

export interface KvGeometry {
  nLayer: number;
  nHeadKv: number;
  headDimK?: number;
  headDimV?: number;
  nEmbd?: number;
  nHead?: number;
  cacheTypeK: string;
  cacheTypeV: string;
}

export interface KvBytesPerToken {
  bytes: number | null;
  confidence: "good" | "rough" | "unknown";
}

// The per-token KV cost both directions of the estimator share:
//   kvBytesPerTok = nLayer * nHeadKv * (dK*bytesPerVal(K) + dV*bytesPerVal(V))
// where dK = headDimK ?? nEmbd/nHead and dV = headDimV ?? headDimK ?? dK.
// The nEmbd/nHead fallback is a both-or-neither input pair, and using it
// drops confidence a level -- exactly as the forward estimator derives them.
export function kvBytesPerToken(g: KvGeometry): KvBytesPerToken {
  const bK = bytesPerVal(g.cacheTypeK);
  const bV = bytesPerVal(g.cacheTypeV);
  if (bK == null || bV == null || g.nLayer <= 0 || g.nHeadKv <= 0) {
    return { bytes: null, confidence: "unknown" };
  }
  let dK = g.headDimK;
  let dV = g.headDimV ?? g.headDimK;
  if ((dK == null || dV == null) && g.nEmbd != null && g.nHead != null && g.nHead > 0) {
    dK = dK ?? g.nEmbd / g.nHead;
    dV = dV ?? dK;
  }
  if (dK == null || dV == null) return { bytes: null, confidence: "unknown" };
  const confidence = g.headDimK == null || g.headDimV == null ? "rough" : "good";
  return { bytes: g.nLayer * g.nHeadKv * (dK * bK + dV * bV), confidence };
}

// Forward direction, in MiB: what a KV cache of `tokens` tokens costs across
// `parallelSlots` slots. Null when the geometry can't produce a number --
// callers annotate with "unknown", never a fabricated figure.
export function estimateKvCacheMib(
  g: KvGeometry & { tokens: number; parallelSlots?: number }
): number | null {
  const perToken = kvBytesPerToken(g);
  if (perToken.bytes == null || g.tokens <= 0) return null;
  return (perToken.bytes * g.tokens * Math.max(1, g.parallelSlots ?? 1)) / BYTES_PER_MIB;
}

// M1's "weightsMib === null -> interpolate from any prior run's
// vram_free_before_mib/gpu_memory_model_peak pair for this model+machine":
// a measured peak minus that item's own KV cost and scratch is the
// VRAM-resident weight share for the placement that produced it. Never
// invents a weights number -- null in, null out.
export function residentWeightsMibFromPeak(
  input: KvGeometry & {
    vramPeakMib: number | null;
    contextTokens: number;
    parallelSlots?: number;
    scratchMib?: number;
  }
): number | null {
  if (input.vramPeakMib == null || input.vramPeakMib <= 0) return null;
  const kvMib = estimateKvCacheMib({ ...input, tokens: input.contextTokens });
  if (kvMib == null) return null;
  const scratch = input.scratchMib ?? DEFAULT_MAX_CTX_SCRATCH_MIB;
  return Math.max(0, Math.round(input.vramPeakMib - kvMib - scratch));
}

export function maxAffordableContext(input: MaxCtxInput): MaxCtxEstimate {
  const scratchMib = input.scratchMib ?? DEFAULT_MAX_CTX_SCRATCH_MIB;
  const headroomFrac = input.activationsHeadroomFrac ?? DEFAULT_ACTIVATIONS_HEADROOM_FRAC;
  const slots = Math.max(1, input.parallelSlots ?? 1);

  // One formula for both directions -- see kvBytesPerToken above.
  const perToken = kvBytesPerToken(input);
  if (perToken.bytes == null || input.totalMib <= 0 || input.weightsMib == null) {
    const conservativeFloorTokens =
      input.trainedCtx != null && input.trainedCtx > 0 ? Math.floor(input.trainedCtx) : null;
    return { tokens: 0, confidence: "unknown", binding: null, conservativeFloorTokens };
  }
  const confidence: MaxCtxEstimate["confidence"] = perToken.confidence;

  const usableMib = input.totalMib * (1 - headroomFrac) - input.weightsMib - scratchMib;
  const kvBytesPerTok = perToken.bytes;
  if (usableMib <= 0 || kvBytesPerTok <= 0) {
    const binding: MaxCtxEstimate["binding"] =
      input.weightsMib + scratchMib > input.totalMib * (1 - headroomFrac) ? "weights-placement" : "kv";
    return { tokens: 0, confidence, binding };
  }

  if (input.slidingWindow != null && input.slidingWindow > 0) {
    // SWA -- direction matters because this is the inverse. The forward
    // estimator's naive all-layers-full-context error overestimates cost;
    // here the same naivety would overestimate affordable tokens, which is
    // the unsafe direction. Assume the Gemma-style ~1 global : 5 local
    // interleave: local layers contribute their window-bound bytes and the
    // global budget is solved for.
    const g = Math.ceil(input.nLayer / 6);
    const rawTokens = (usableMib * 2 ** 20) / kvBytesPerTok;
    const tokens = Math.max(0, rawTokens - (input.nLayer - g) * input.slidingWindow) / g;
    return { tokens: Math.floor(tokens), confidence: "rough", binding: "kv" };
  }

  const tokens = Math.floor((usableMib * 2 ** 20) / (kvBytesPerTok * slots));
  const binding: MaxCtxEstimate["binding"] =
    input.weightsMib + scratchMib > input.totalMib * (1 - headroomFrac) ? "weights-placement" : "kv";
  return { tokens, confidence, binding };
}

// --- Dual-pool (VRAM + RAM) fit check and auto-suggested placement ---------
//
// Extends the single-pool inversion above to what a partial -ngl offload
// actually creates: some layers' weights and KV cache sit in VRAM, the rest
// sit in system RAM, and both budgets must be checked independently -- except
// on a unified-memory machine (Metal, or any GPU with vram_dynamic:true),
// where VRAM and RAM are literally the same bytes and a split two-pool check
// would double-count them.
//
// Reuses kvBytesPerToken/estimateKvCacheMib UNCHANGED: because that formula
// is linear in nLayer, the GPU share of KV cache is the same call with
// nLayer:ngl and the CPU share is nLayer:(kvLayerCount-ngl) -- no new KV math
// needed, just re-parameterizing the existing one twice.
//
// Deliberately uses the PLAIN (non-SWA) kvBytesPerToken path even when
// slidingWindow is set: maxAffordableContext's SWA branch above assumes
// nLayer is the WHOLE model's own fixed global/local layer pattern, which an
// arbitrary GPU-resident subset does not preserve. The plain formula
// overestimates KV need for an SWA model instead -- the SAFE direction for a
// forward fit check (unlike maxAffordableContext's own inverse direction,
// where overestimating affordable tokens is the unsafe one). Confidence
// drops one level whenever slidingWindow is set, so the UI still hedges even
// though the arithmetic is deliberately simple.

export const RAM_ESTIMATE_FIXED_OVERHEAD_MIB = 512;

export interface PoolReading {
  freeMib: number | null;
  totalMib: number | null;
}

export interface DualPoolInput {
  modelSizeBytes: number;
  // n_layer + 1, same convention as estimateVramNeededMib.
  totalModelLayers: number;
  // Raw n_layer (NO +1) -- only transformer layers carry a KV cache.
  kvLayerCount: number;
  ngl: number;
  ctxTokens: number;
  nHeadKv: number;
  headDimK?: number;
  headDimV?: number;
  nEmbd?: number;
  nHead?: number;
  cacheTypeK: string;
  cacheTypeV: string;
  slidingWindow?: number;
  parallelSlots?: number;
  vram: PoolReading;
  ram: PoolReading;
  // Metal, or any visible GPU with vram_dynamic:true.
  unifiedPool: boolean;
}

export interface PoolFit {
  weightsMib: number;
  kvMib: number;
  neededMib: number;
  freeMib: number | null;
  // null = insufficient data (no live reading yet) -- never treated as
  // "doesn't fit". Callers must branch on all three states explicitly.
  fits: boolean | null;
}

export interface DualPoolFit {
  gpu: PoolFit;
  cpu: PoolFit;
  unifiedPool: boolean;
  fits: boolean | null;
  confidence: "good" | "rough" | "unknown";
}

function dualPoolConfidence(input: DualPoolInput): "good" | "rough" | "unknown" {
  const base = kvBytesPerToken({
    nLayer: input.kvLayerCount,
    nHeadKv: input.nHeadKv,
    headDimK: input.headDimK,
    headDimV: input.headDimV,
    nEmbd: input.nEmbd,
    nHead: input.nHead,
    cacheTypeK: input.cacheTypeK,
    cacheTypeV: input.cacheTypeV,
  }).confidence;
  if (base === "unknown") return "unknown";
  return input.slidingWindow != null && input.slidingWindow > 0 ? "rough" : base;
}

export function computeDualPoolFit(input: DualPoolInput): DualPoolFit {
  const ngl = Math.max(0, Math.min(Math.round(input.ngl), input.totalModelLayers));
  const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
  const gpuWeightsMib = (bytesPerLayer * ngl) / BYTES_PER_MIB;
  const cpuWeightsMib = (bytesPerLayer * (input.totalModelLayers - ngl)) / BYTES_PER_MIB;

  // ngl's range includes the output layer, which carries no KV -- clamping
  // against kvLayerCount (not totalModelLayers) matches llama.cpp's own
  // offload order, where the output layer is offloaded only once ngl exceeds
  // n_layer, so this correctly counts only real KV-bearing layers per side.
  const gpuKvLayers = Math.min(ngl, input.kvLayerCount);
  const cpuKvLayers = input.kvLayerCount - gpuKvLayers;
  const kvGeometry = {
    nHeadKv: input.nHeadKv,
    headDimK: input.headDimK,
    headDimV: input.headDimV,
    nEmbd: input.nEmbd,
    nHead: input.nHead,
    cacheTypeK: input.cacheTypeK,
    cacheTypeV: input.cacheTypeV,
  };
  // estimateKvCacheMib returns null when nLayer<=0 -- the ?? 0 fallback is
  // correct here: zero GPU-resident KV layers genuinely need zero GPU KV
  // bytes, not "unknown".
  const gpuKvMib =
    estimateKvCacheMib({ ...kvGeometry, nLayer: gpuKvLayers, tokens: input.ctxTokens, parallelSlots: input.parallelSlots }) ??
    0;
  const cpuKvMib =
    estimateKvCacheMib({ ...kvGeometry, nLayer: cpuKvLayers, tokens: input.ctxTokens, parallelSlots: input.parallelSlots }) ??
    0;

  const confidence = dualPoolConfidence(input);

  if (input.unifiedPool) {
    // ngl is always pinned > 0 on a unified-memory machine by the caller's
    // own lock rule, so the overhead term is unconditional here -- gating it
    // on ngl>0 would make total need depend on ngl on a pool where it
    // physically can't (VRAM and RAM are the same bytes).
    const neededMib = gpuWeightsMib + cpuWeightsMib + gpuKvMib + cpuKvMib + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB;
    const freeMib = input.ram.freeMib ?? input.vram.freeMib;
    const fit: PoolFit = {
      weightsMib: gpuWeightsMib + cpuWeightsMib,
      kvMib: gpuKvMib + cpuKvMib,
      neededMib,
      freeMib,
      fits: freeMib == null ? null : neededMib <= freeMib,
    };
    return { gpu: fit, cpu: fit, unifiedPool: true, fits: fit.fits, confidence };
  }

  // Split pool (discrete GPU). All comparisons use FREE, not total, mib --
  // matching estimateSafeNgl's own precedent, deliberately not the total-mib
  // convention maxAffordableContext/GoalQuestionnaire's affordabilityFor use.
  const gpuNeededMib = gpuWeightsMib + gpuKvMib + (ngl > 0 ? VRAM_ESTIMATE_FIXED_OVERHEAD_MIB : 0);
  const cpuNeededMib = cpuWeightsMib + cpuKvMib + RAM_ESTIMATE_FIXED_OVERHEAD_MIB;
  const gpu: PoolFit = {
    weightsMib: gpuWeightsMib,
    kvMib: gpuKvMib,
    neededMib: gpuNeededMib,
    freeMib: input.vram.freeMib,
    // Needing 0 bytes from a pool always fits, even without a live reading
    // for it -- otherwise every CPU-only worker (which never reports a VRAM
    // total/free at all, since it has no GPU) would be stuck at "unknown"
    // forever purely because ngl is pinned at 0, never reaching "ok".
    fits: gpuNeededMib === 0 ? true : input.vram.freeMib == null ? null : gpuNeededMib <= input.vram.freeMib,
  };
  const cpu: PoolFit = {
    weightsMib: cpuWeightsMib,
    kvMib: cpuKvMib,
    neededMib: cpuNeededMib,
    freeMib: input.ram.freeMib,
    fits: input.ram.freeMib == null ? null : cpuNeededMib <= input.ram.freeMib,
  };
  const fits = gpu.fits == null || cpu.fits == null ? null : gpu.fits && cpu.fits;
  return { gpu, cpu, unifiedPool: false, fits, confidence };
}

// Matches MIN_PROBE_CTX (server/src/routes/measurements.ts) -- every
// suggestion below must also be a legal later probe candidate.
export const MIN_FEASIBLE_CTX = 256;

export type SuggestedConfigLabel =
  | "target_ctx_reduce_offload"
  | "max_offload_reduce_ctx"
  | "balanced"
  | "minimum_viable";

export interface SuggestedConfig {
  label: SuggestedConfigLabel;
  ngl: number;
  ctx: number;
}

export interface SuggestionResult {
  // cannot_run: confirmed nothing fits, ever. unknown: insufficient data
  // (missing live reading or KV geometry) -- distinct from cannot_run, never
  // conflated. ok: 1-3 configs, guaranteed non-empty (see the fallback
  // below).
  outcome: "cannot_run" | "unknown" | "ok";
  configs: SuggestedConfig[];
}

export interface SuggestInput {
  modelSizeBytes: number;
  totalModelLayers: number;
  kvLayerCount: number;
  currentNgl: number;
  currentCtx: number;
  trainedCtx?: number | null;
  nHeadKv: number;
  headDimK?: number;
  headDimV?: number;
  nEmbd?: number;
  nHead?: number;
  cacheTypeK: string;
  cacheTypeV: string;
  slidingWindow?: number;
  parallelSlots?: number;
  vram: PoolReading;
  ram: PoolReading;
  unifiedPool: boolean;
  noGpu: boolean;
}

// Three deterministic auto-suggested placements for when the caller's
// current (ngl, ctx) doesn't fit: A keeps ctx and reduces offload, B
// maximizes offload and reduces ctx, C balances both. See the plan this
// implements (dreamy-drifting-zephyr) for the full derivation.
export function suggestPlacementConfigs(input: SuggestInput): SuggestionResult {
  const fitAt = (ngl: number, ctx: number): DualPoolFit =>
    computeDualPoolFit({
      modelSizeBytes: input.modelSizeBytes,
      totalModelLayers: input.totalModelLayers,
      kvLayerCount: input.kvLayerCount,
      ngl,
      ctxTokens: ctx,
      nHeadKv: input.nHeadKv,
      headDimK: input.headDimK,
      headDimV: input.headDimV,
      nEmbd: input.nEmbd,
      nHead: input.nHead,
      cacheTypeK: input.cacheTypeK,
      cacheTypeV: input.cacheTypeV,
      slidingWindow: input.slidingWindow,
      parallelSlots: input.parallelSlots,
      vram: input.vram,
      ram: input.ram,
      unifiedPool: input.unifiedPool,
    });

  // A missing reading is "we don't know", never "0 MiB free" -- guard BEFORE
  // estimateSafeNgl rather than coercing with `?? 0` inside the call, which
  // would silently force nglCeiling to 0 and cascade into a false
  // cannot_run instead of unknown.
  if (!input.noGpu && !input.unifiedPool && input.vram.freeMib == null) {
    return { outcome: "unknown", configs: [] };
  }
  // ram.freeMib is non-null whenever WorkerVramInfo exists at all -- null
  // here means no reading has arrived yet at all (e.g. an offline worker).
  if (!input.unifiedPool && input.ram.freeMib == null) {
    return { outcome: "unknown", configs: [] };
  }

  // Step 0: the hard ceiling -- max ngl whose WEIGHTS ALONE fit VRAM at
  // ctx=0. On a unified pool this isn't a meaningful VRAM-only question
  // (all weights draw from the one shared pool regardless of split), so the
  // ceiling is simply every layer, matching the caller's own lock rule that
  // always pins ngl there.
  const nglCeiling = input.noGpu
    ? 0
    : input.unifiedPool
      ? input.totalModelLayers
      : estimateSafeNgl(input.modelSizeBytes, input.totalModelLayers, input.vram.freeMib ?? 0);

  const floorFit = fitAt(0, MIN_FEASIBLE_CTX).fits;
  if (floorFit === false) return { outcome: "cannot_run", configs: [] };
  if (floorFit === null) return { outcome: "unknown", configs: [] };

  const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
  const trainedCtxOrInf = input.trainedCtx != null && input.trainedCtx > 0 ? input.trainedCtx : Infinity;
  const currentCtxOrInf = input.currentCtx > 0 ? input.currentCtx : Infinity;

  // Config A's own search primitive, reused verbatim for the guaranteed
  // fallback at the bottom (same loop, different fixed ctx).
  const searchDescending = (ctx: number, from: number): SuggestedConfig | null => {
    for (let ngl = Math.min(input.currentNgl, from); ngl >= 0; ngl--) {
      if (fitAt(ngl, ctx).fits === true) return { label: "target_ctx_reduce_offload", ngl, ctx };
    }
    return null;
  };

  // Config A -- target context, reduce offload.
  const A = searchDescending(input.currentCtx, nglCeiling);

  // Config B -- max layers on VRAM, reduce context.
  let B: SuggestedConfig | null = null;
  {
    const ngl = nglCeiling;
    let ctx: number;
    if (ngl === 0) {
      // Degenerate: not even one layer's weights fit VRAM at ctx=0, so
      // there's no meaningful GPU KV budget to invert -- this is simply
      // "everything on CPU", same starting point as the floor check.
      const bound = Math.min(trainedCtxOrInf, currentCtxOrInf);
      ctx = Number.isFinite(bound) ? Math.max(MIN_FEASIBLE_CTX, bound) : MIN_FEASIBLE_CTX;
    } else {
      const gpuWeightsMib = (bytesPerLayer * ngl) / BYTES_PER_MIB;
      // activationsHeadroomFrac:0 -- vram.freeMib is ALREADY the reserved
      // budget; maxAffordableContext's own 10% headroom is calibrated
      // against every existing caller's use of TOTAL vram, and stacking it
      // on an already-free-mib figure would double-apply headroom silently.
      const vramCeil = maxAffordableContext({
        totalMib: input.vram.freeMib ?? 0,
        weightsMib: gpuWeightsMib,
        nLayer: Math.min(ngl, input.kvLayerCount),
        nHeadKv: input.nHeadKv,
        headDimK: input.headDimK,
        headDimV: input.headDimV,
        nEmbd: input.nEmbd,
        nHead: input.nHead,
        cacheTypeK: input.cacheTypeK,
        cacheTypeV: input.cacheTypeV,
        activationsHeadroomFrac: 0,
      });
      const upperBound = Math.min(trainedCtxOrInf, currentCtxOrInf);
      const capped = Number.isFinite(upperBound) ? Math.min(vramCeil.tokens, upperBound) : vramCeil.tokens;
      ctx = Math.max(MIN_FEASIBLE_CTX, capped);
    }
    // Halve ctx until BOTH pools fit -- vramCeil above only ever checked
    // VRAM; RAM for the (totalModelLayers-ngl) CPU-resident layers hasn't
    // been checked at all yet.
    while (fitAt(ngl, ctx).fits !== true && ctx > MIN_FEASIBLE_CTX) {
      ctx = Math.max(MIN_FEASIBLE_CTX, Math.floor(ctx / 2));
    }
    if (fitAt(ngl, ctx).fits === true) B = { label: "max_offload_reduce_ctx", ngl, ctx };
  }

  // Config C -- balanced: midpoint of A and B, then ratchet ctx down by
  // half, then ngl down by single layers, until it fits. Reuses only
  // primitives A/B already needed -- nothing new invented. A.ngl <= B.ngl
  // and B.ctx <= A.ctx always hold by construction, so the midpoint always
  // sits strictly between them on both axes.
  let C: SuggestedConfig | null = null;
  if (A && B) {
    let ngl = Math.round((A.ngl + B.ngl) / 2);
    let ctx = Math.round((A.ctx + B.ctx) / 2);
    while (fitAt(ngl, ctx).fits !== true && ctx > MIN_FEASIBLE_CTX) {
      ctx = Math.max(MIN_FEASIBLE_CTX, Math.floor(ctx / 2));
    }
    while (fitAt(ngl, ctx).fits !== true && ngl > 0) {
      ngl -= 1;
    }
    if (fitAt(ngl, ctx).fits === true) C = { label: "balanced", ngl, ctx };
  }

  const seen = new Set<string>();
  const configs: SuggestedConfig[] = [];
  for (const c of [A, B, C]) {
    if (!c) continue;
    const key = `${c.ngl}:${c.ctx}`;
    if (seen.has(key)) continue;
    seen.add(key);
    configs.push(c);
  }

  // Guaranteed non-empty: floorFit===true already PROVES (ngl=0,
  // ctx=MIN_FEASIBLE_CTX) fits, but A/B/C can each independently miss it (A
  // only tries ctx=currentCtx; B is pinned at ngl=nglCeiling, which can be
  // sized so tightly against VRAM that not even MIN_FEASIBLE_CTX of KV fits
  // there). Without this, outcome:"ok" could legitimately pair with an
  // empty configs list. Only ever engaged in this rare all-three-empty edge
  // case -- normal cases still show up to 3 cards, not 4.
  if (configs.length === 0) {
    const fallback = searchDescending(MIN_FEASIBLE_CTX, nglCeiling);
    if (fallback) configs.push({ ...fallback, label: "minimum_viable" });
  }

  return { outcome: "ok", configs };
}
