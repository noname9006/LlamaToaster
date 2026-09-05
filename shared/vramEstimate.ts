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

// Real on-disk byte sizes per tensor, bucketed by what placeWeightBytes below
// needs to place them: which transformer block (if any) each belongs to, and
// whether it's a MoE expert tensor (subject to --n-cpu-moe) or not. Built by
// worker/src/gguf.ts's readGgufInfo from each tensor_info entry's OWN
// declared byte offset -- consecutive tensors' offsets subtracted give the
// exact stored size regardless of quantization format, so this needs no
// per-type block-size table and never goes stale against a new GGUF quant
// format. Defined here (not gguf.ts) since it's consumed by placeWeightBytes
// below and by every client call site, none of which may import from
// worker/.
export interface TensorLayerBreakdown {
  // Bytes per transformer block (blk.N.*), index-aligned to block index N,
  // EXCLUDING any MoE expert tensors that block holds -- length n_layer.
  dense: number[];
  // MoE expert tensor (blk.N.ffn_*_exps.*) bytes per block, same indexing as
  // dense -- 0 for a block with no experts (dense model, or a hybrid
  // architecture's dense-only layers).
  moe: number[];
  // token_embd.* -- llama.cpp keeps this CPU-resident unless every
  // transformer block is itself GPU-resident (ngl >= n_layer), regardless of
  // how many layers are offloaded short of that.
  embed: number;
  // output_norm.* + output.* (or output.* alone when the model ties output
  // to token_embd, per architecture) -- llama.cpp's own "+1" pseudo-layer,
  // offloaded only once requested ngl exceeds n_layer.
  output: number;
  // Anything matched by neither bucket above (rope frequency tables, misc
  // top-level tensors) -- always treated as CPU-resident; typically tiny.
  other: number;
}

export interface WeightPlacementBytes {
  gpuBytes: number;
  cpuBytes: number;
}

// The accurate replacement for the flat file-size/layer-count average below:
// places every real tensor bucket according to llama.cpp's OWN documented
// offload rules (confirmed against its source, see the PR/issue discussion
// on input-embedding placement), not an assumption that every layer is the
// same size.
//
//   - dense block tensors: GPU iff the block is among the LAST `ngl` blocks
//     (llama.cpp's i_gpu_start = n_layer - ngl; layers closer to the output
//     offload first as ngl grows).
//   - MoE expert tensors: forced CPU for the FIRST `nCpuMoe` blocks
//     regardless of the dense rule (--n-cpu-moe), otherwise follow the same
//     dense rule as their block's other tensors.
//   - token_embd: GPU only once EVERY transformer block is GPU-resident
//     (ngl >= nLayer) -- llama.cpp keeps it CPU otherwise, since there's
//     little benefit to offloading a lookup-only tensor short of that.
//   - output/output_norm: GPU only once ngl exceeds nLayer (this codebase's
//     own "+1" convention for the output pseudo-layer, e.g.
//     ResultRow.total_model_layers).
//   - other: always CPU (typically tiny -- rope frequency tables etc.).
export function placeWeightBytes(
  breakdown: TensorLayerBreakdown,
  nLayer: number,
  ngl: number,
  nCpuMoe = 0
): WeightPlacementBytes {
  const clampedNgl = Math.max(0, Math.min(Math.round(ngl), nLayer + 1));
  const gpuBlocks = Math.min(clampedNgl, nLayer);
  const gpuFrom = nLayer - gpuBlocks; // blocks [gpuFrom, nLayer) are GPU-resident
  let gpuBytes = 0;
  let cpuBytes = 0;
  for (let i = 0; i < nLayer; i++) {
    const dense = breakdown.dense[i] ?? 0;
    const moe = breakdown.moe[i] ?? 0;
    const onGpu = i >= gpuFrom;
    if (onGpu) gpuBytes += dense;
    else cpuBytes += dense;
    // --n-cpu-moe pins this block's experts to CPU regardless of the dense
    // rule above, but only when that block would otherwise be GPU-resident --
    // a block the dense rule already sends to CPU needs no separate pin.
    if (onGpu && i < nCpuMoe) cpuBytes += moe;
    else if (onGpu) gpuBytes += moe;
    else cpuBytes += moe;
  }
  if (gpuBlocks >= nLayer) gpuBytes += breakdown.embed;
  else cpuBytes += breakdown.embed;
  if (clampedNgl > nLayer) gpuBytes += breakdown.output;
  else cpuBytes += breakdown.output;
  cpuBytes += breakdown.other;
  return { gpuBytes, cpuBytes };
}

export const VRAM_ESTIMATE_FIXED_OVERHEAD_MIB = 512;
// Deliberately generous: the real diagnosed case above was off by roughly
// 25x, so flagging anything under half the estimate still catches that
// failure mode while staying quiet on ordinary estimate error -- the flat
// per-layer average below is never exact, especially for a MoE model whose
// real per-layer footprint isn't uniform.
export const VRAM_DISCREPANCY_RATIO = 0.5;

// --- Measured host-backed fallback (see detectHostBackedFallback below) -----
//
// Every constant here was calibrated against a real 19-load sweep on an AMD
// Radeon RX 6600 XT (8176MiB) running the Vulkan build b10819 with
// Qwen3.6-35B-A3B-UD-IQ4_NL (17205MiB, 41 layers). The two raw datasets --
// an ngl sweep at fixed context and a batch/context sweep at fixed ngl -- are
// what these numbers are fitted to, and any change to them should be made
// against a comparable measurement rather than by intuition:
//
//   ngl   dedicated   shared      gen tok/s     (ctx 1024, b2048/ub512)
//     0         277       28           8.92
//     4        2014      427          11.31
//     6        2822      427          11.62
//     8        3442      639          11.90
//    10        3610     1279          12.05   <- fastest config measured
//    12        3434     2270          10.93
//    26        3736     7647           3.54
//    41        4069    13258           6.87
//
// Two facts drive the design. First, `shared` is FLAT (427-810MiB) across
// every configuration where the placement is genuinely resident -- it does
// not scale with ngl, and it moves the WRONG way with batch size (the two
// smallest batches measured the highest shared usage) -- so it is buffer
// overhead, not spill. Second, once the driver starts backing weights with
// system RAM, `shared` grows by one layer's worth per added layer (382MiB
// measured, against a 420MiB file-derived per-layer size), and it matches
// the missing dedicated bytes to within 1%: at ngl 26 the shortfall against
// the estimate was 7565MiB and shared measured 7647MiB.
//
// Hence: subtract an overhead allowance, and call whatever is left over
// spill once it exceeds a whole layer.

/**
 * How much of a layer's worth of system RAM has to appear per ADDED layer
 * before the placement is called host-backed. Dimensionless on purpose: it is
 * a fraction of whatever one layer of THIS model costs, so it carries across
 * models, GPUs, drivers and quantizations without recalibration.
 *
 * The measured separation on the calibration sweep is wide and unambiguous --
 * clean intervals ran 0.00-0.47 layers per layer added, spilling ones
 * 0.76-1.18, with nothing in between:
 *
 *   ngl 2->4    0.00      ngl  8->10   0.76
 *   ngl 4->6    0.00      ngl 10->12   1.18
 *   ngl 6->8    0.25      ngl 12->14   0.78
 *   ngl 0->2    0.47      ngl 14->18   0.99
 *                         ngl 18->26   0.91
 *                         ngl 26->41   0.89
 *
 * The one clean interval that comes close (0->2) is the GPU compute buffers
 * appearing the moment ANY layer lands on the device -- a one-off step, not a
 * slope, which is why a rung at ngl 0 is never used as a reference.
 */
export const HOST_BACKED_SLOPE_RATIO = 0.5;

/**
 * Minimum layer span the slope may be measured over.
 *
 * The driver hands out memory in granules, not in exact layer-sized pieces --
 * measured at ~224MiB on the reference machine, which is 0.53 of that model's
 * 420MiB layer. A single-layer delta is therefore mostly quantization noise,
 * and a full 1-layer-resolution sweep (42 settings, 7 runs each) shows the two
 * regimes overlapping outright at that resolution:
 *
 *   span 1   clean up to +1.05   spilling down to +0.38   -> NO threshold works
 *   span 2   clean up to +0.31   spilling down to +0.62   -> clean separation
 *   span 3   clean up to +0.35   spilling down to +0.71
 *
 * The tell is the alternation in the clean region -- +0.54, -0.53, +1.05,
 * -0.53 -- one granule appearing and disappearing between adjacent rungs.
 * Two layers halves that noise to +/-0.27, which is what opens the gap the
 * threshold sits in. A reference closer than this is skipped in favour of a
 * further one; if none exists, the bootstrap decides instead.
 */
export const HOST_BACKED_MIN_SLOPE_SPAN = 2;

/**
 * Bootstrap threshold for the FIRST rung of a phase, which by definition has
 * no lower rung to take a slope against: what fraction of a placement's own
 * predicted GPU footprint may be system-RAM-backed before it is called
 * host-backed.
 *
 * Also dimensionless, and relative to the model's own size rather than to any
 * machine -- a 2GiB model and a 200GiB one are judged on the same scale. The
 * measured separation is narrower than the slope's, which is exactly why this
 * is only the fallback:
 *
 *   clean    14% (ngl 6)  16% (ngl 8)  19% (ngl 4)  25% (ngl 4 @131072 ctx)
 *            27% (ngl 10) 28% (ngl 4 @ b512)
 *   spilling 41% (ngl 12) 44% (ngl 13) 46% (ngl 14) 58% (ngl 18) 68% (ngl 26)
 *            75% (ngl 41)
 *
 * Denominated in the ESTIMATOR's footprint rather than the file's per-layer
 * size on purpose: the clean band has to hold at large contexts too, where KV
 * cache legitimately adds system-RAM-backed bytes (the 131072-token rung above
 * would read 48% against weights alone, and be convicted).
 */
export const HOST_BACKED_BOOTSTRAP_FRAC = 0.35;

const BYTES_PER_MIB = 1024 * 1024;

export interface VramNeedEstimateInput {
  modelSizeBytes: number;
  // n_layer + 1 (the output layer) -- matches llama.cpp's own "offloaded
  // X/Y" runtime convention, see shared/types.ts's ResultRow.total_model_layers.
  totalModelLayers: number;
  requestedNgl: number;
  // --n-cpu-moe -- see shared/sweep.ts's SweepItem.n_cpu_moe. Only consumed
  // alongside tensorBreakdown below; the flat fallback path has no way to
  // see which bytes are MoE-expert weights at all.
  nCpuMoe?: number;
  // Real per-tensor byte breakdown (worker/src/gguf.ts's readGgufInfo, see
  // ModelMetadata.tensor_layer_bytes) -- when present, drives the accurate
  // placeWeightBytes path below instead of the flat per-layer average.
  // Absent/null for a model registered before this field existed (or a
  // "local" entry the app never had file bytes to parse), in which case this
  // falls back to the flat average exactly as before.
  tensorBreakdown?: TensorLayerBreakdown | null;
}

// Flat per-layer average of the whole on-disk file -- the fallback for a
// model with no real tensorBreakdown yet (see VramNeedEstimateInput). Weakest
// for MoE models specifically: a MoE layer's true footprint depends on how
// many of ITS experts actually end up GPU-resident (see --n-cpu-moe), which
// this formula has no way to see -- it just assumes every layer is the same
// size as the file-wide average. It also silently misattributes a share of
// the (often huge) token-embedding/output tensors to every requested layer,
// even though llama.cpp keeps those CPU-resident short of a full-model
// offload -- systematically overestimating VRAM need, which is the "less
// memory needed than the estimate" bug the tensor-based path above exists to
// fix. Good enough for an order-of-magnitude warning when no better data
// exists, not a precise budget. Null whenever an input can't produce a
// meaningful estimate (nothing requested, or a size/layer-count of 0) --
// callers treat null as "don't show a banner" rather than a false "fits" or
// "doesn't fit".
function estimateVramNeededMibFlat(input: VramNeedEstimateInput): number | null {
  if (input.modelSizeBytes <= 0 || input.totalModelLayers <= 0 || input.requestedNgl <= 0) return null;
  const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
  const neededBytes = bytesPerLayer * input.requestedNgl + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB * BYTES_PER_MIB;
  return Math.round(neededBytes / BYTES_PER_MIB);
}

// GPU-resident weight bytes for the requested placement, in MiB. Prefers the
// accurate tensor-based placeWeightBytes whenever a real tensorBreakdown is
// available (see TensorLayerBreakdown's doc comment for why this needs no
// per-quant-format table and stays correct for MoE/uneven-layer models),
// falling back to the flat per-layer average otherwise. See
// estimateVramNeededMibFlat above for that fallback's known weaknesses.
export function estimateVramNeededMib(input: VramNeedEstimateInput): number | null {
  if (input.requestedNgl <= 0) return null;
  if (input.tensorBreakdown && input.totalModelLayers > 0) {
    const nLayer = input.totalModelLayers - 1;
    if (nLayer > 0) {
      const { gpuBytes } = placeWeightBytes(input.tensorBreakdown, nLayer, input.requestedNgl, input.nCpuMoe ?? 0);
      return Math.round(gpuBytes / BYTES_PER_MIB + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB);
    }
  }
  return estimateVramNeededMibFlat(input);
}

// Inverse of the above: the largest ngl whose estimate still fits under
// freeVramMib. Used both by the pre-flight banner (a starting suggestion)
// and the auto-cap action (NewRun.tsx). Clamped to [0, totalModelLayers].
// Ignores nCpuMoe deliberately when a tensorBreakdown is given: this answers
// "how many layers can I add", and a caller not yet offloading any MoE
// experts to CPU should see the same ceiling regardless of what --n-cpu-moe
// might later be set to -- treating every candidate ngl as nCpuMoe:0 (the
// worst case for GPU bytes needed) keeps the suggestion conservative rather
// than optimistic.
export function estimateSafeNgl(
  modelSizeBytes: number,
  totalModelLayers: number,
  freeVramMib: number,
  tensorBreakdown?: TensorLayerBreakdown | null
): number {
  if (modelSizeBytes <= 0 || totalModelLayers <= 0) return 0;
  const usableBytes = Math.max(0, freeVramMib - VRAM_ESTIMATE_FIXED_OVERHEAD_MIB) * BYTES_PER_MIB;
  if (tensorBreakdown) {
    const nLayer = totalModelLayers - 1;
    if (nLayer > 0) {
      // Bytes are non-decreasing in ngl (each step only ever adds a block's
      // worth of tensors), so a linear scan for the largest fitting ngl is
      // safe and simple -- nLayer is at most a few hundred even for the
      // deepest real models.
      let best = 0;
      for (let ngl = 0; ngl <= totalModelLayers; ngl++) {
        const { gpuBytes } = placeWeightBytes(tensorBreakdown, nLayer, ngl, 0);
        if (gpuBytes > usableBytes) break;
        best = ngl;
      }
      return best;
    }
  }
  const bytesPerLayer = modelSizeBytes / totalModelLayers;
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

/** One measured rung, as this detector needs to see it. */
export interface HostBackedRungSample {
  ngl: number;
  /** MEASURED per-process system-RAM-backed GPU allocation (Windows WDDM
   * "Shared Usage" / Linux amdgpu GTT). Null wherever no such counter exists
   * -- which is what makes this whole check unavailable rather than passing. */
  sharedPeakMib: number | null;
  /** MEASURED per-process dedicated VRAM peak -- this process only, never the
   * whole adapter. */
  dedicatedPeakMib: number | null;
}

/**
 * Fraction of the best clean prompt-processing rate below which prefill is
 * considered to have fallen off a cliff.
 *
 * A SECOND, EARLIER boundary than the weights one, and a different mechanism:
 * the batch compute buffer moves to host memory before any weight does, and
 * prompt evaluation -- which is dominated by that buffer -- collapses when it
 * happens, while token generation (bandwidth-bound per token, not
 * buffer-bound) carries on improving for several more layers.
 *
 * Measured on the reference machine, one layer apart:
 *
 *   ngl 6   pp 69.2 t/s   TTFT  2583 ms   tg 12.23
 *   ngl 7   pp 17.7 t/s   TTFT 10056 ms   tg 12.28   <- 3.9x collapse
 *   ngl 10  pp 18.9 t/s   TTFT  9389 ms   tg 13.10   <- fastest tg, prefill still ruined
 *
 * pp never recovers (17-24 t/s all the way to full offload). For a 130-token
 * prompt with 80 generated that makes ngl 6 finish 1.5x sooner end to end than
 * the layer count with the best tok/s -- which is why this is reported rather
 * than folded into the pass/fail rule: which one is "best" depends on whether
 * the workload is prefill- or decode-heavy, and only the caller knows that.
 *
 * 0.5 discriminates a 3.9x step, so its exact value is not load-bearing;
 * anything from 0.3 to 0.8 identifies the same rung on this data.
 */
export const PREFILL_CLIFF_RATIO = 0.5;

/**
 * True when this rung's prompt-processing rate has collapsed relative to the
 * best rate seen at a genuinely-resident placement. `bestCleanPpTps` must come
 * from rungs the host-backed check left alone -- comparing against a rung that
 * was already over the cliff would hide it.
 */
export function isPrefillCliff(ppTps: number | null, bestCleanPpTps: number | null): boolean {
  if (ppTps == null || bestCleanPpTps == null || bestCleanPpTps <= 0) return false;
  return ppTps < bestCleanPpTps * PREFILL_CLIFF_RATIO;
}

export interface HostBackedFallbackInput {
  rung: HostBackedRungSample;
  /** Rungs already measured in this run AT THE SAME CONTEXT. Same context is
   * the whole requirement: a different context legitimately moves the KV
   * cache, which would show up in `shared` and be indistinguishable from
   * spilled weights. The layer phase pins context while it searches, so its
   * rungs are all mutually comparable by construction. */
  priorSameCtx: HostBackedRungSample[];
  /** One layer's worth of weights: the model file's own size divided by its
   * layer count. Deliberately a fact from disk rather than an estimate --
   * it is the unit the slope is measured in, so it must not inherit the
   * estimator's bias. Null when the layer count is unknown. */
  perLayerMib: number | null;
  /** computeDualPoolFit's predicted GPU footprint for this rung -- weights
   * plus GPU-side KV plus overhead. Only the BOOTSTRAP uses it, as the
   * denominator that makes its threshold model-relative; the slope path never
   * touches the estimator at all. */
  estimatedGpuMib?: number | null;
}

export interface HostBackedFallbackVerdict {
  hostBacked: boolean;
  /** Which evidence decided it -- "slope" when a comparable lower-ngl rung
   * existed, "ratio" for the single-rung bootstrap, null when neither could
   * run. Also how a caller tells a MEASURED verdict from an absent one. */
  method: "slope" | "ratio" | null;
  /** Layers' worth of system RAM appearing per layer added, against the
   * reference rung. Null unless method is "slope". */
  slopeRatio: number | null;
  /** Roughly how many of this rung's claimed layers are host-backed relative
   * to the reference, for display. Null unless method is "slope". */
  spilledLayers: number | null;
}

const UNAVAILABLE_HOST_BACKED: HostBackedFallbackVerdict = {
  hostBacked: false,
  method: null,
  slopeRatio: null,
  spilledLayers: null,
};

/**
 * MEASURED silent-fallback detection: does system RAM demonstrably hold this
 * placement's weights?
 *
 * isVramDiscrepancy above can only ever INFER a fallback, by noticing that
 * an observed VRAM peak came in far below what was predicted -- which cannot
 * distinguish "the driver silently paged the model to host RAM" from "our
 * estimate was too pessimistic", and fails the rung either way. This one
 * reads the OS's own per-process counter for system-RAM-backed GPU memory,
 * so a rung is only ever failed when the missing bytes have actually been
 * found somewhere else. An estimate that is merely wrong produces no shared
 * usage and is no longer capable of failing anything.
 *
 * Returns hostBacked:false (with null figures) whenever the check cannot run
 * -- no counter on this platform/backend, no estimate, or a rung with
 * nothing claimed on the GPU. Callers fall back to isVramDiscrepancy there;
 * "couldn't measure" must never read as "measured clean".
 */
export function detectHostBackedFallback(input: HostBackedFallbackInput): HostBackedFallbackVerdict {
  const { rung, perLayerMib } = input;
  if (rung.sharedPeakMib == null || rung.ngl <= 0) return UNAVAILABLE_HOST_BACKED;

  // --- Primary: the SLOPE ---------------------------------------------------
  //
  // Overhead does not move with ngl; spilled weights move with it one-for-one.
  // So the discriminating quantity is how much system RAM appears PER LAYER
  // ADDED, measured against the nearest comparable rung below this one, and
  // expressed as a fraction of what one layer of this model weighs.
  //
  // The reference is the CLOSEST lower-ngl rung rather than the lowest, so
  // this is a local gradient rather than a whole-range average -- a reference
  // that is itself already spilling then understates the slope, which loses
  // detections but never invents them. Under-detection is the safe direction:
  // it leaves the ladder searching, where a false positive would fail a
  // placement that genuinely works.
  //
  // ngl 0 is never a reference: going from no GPU layers to some allocates the
  // compute buffers, a one-off step that would read as a steep slope.
  // Closest usable reference, but never closer than HOST_BACKED_MIN_SLOPE_SPAN
  // -- an adjacent rung's delta is dominated by the driver's allocation
  // granularity rather than by placement. Closest-of-the-eligible keeps the
  // gradient local; a wider span would average across the onset.
  const reference = input.priorSameCtx
    .filter(
      (p) => p.sharedPeakMib != null && p.ngl > 0 && p.ngl <= rung.ngl - HOST_BACKED_MIN_SLOPE_SPAN
    )
    .sort((a, b) => b.ngl - a.ngl)[0];
  if (reference && perLayerMib != null && perLayerMib > 0) {
    const deltaNgl = rung.ngl - reference.ngl;
    const deltaShared = rung.sharedPeakMib - reference.sharedPeakMib!;
    const slopeRatio = deltaShared / deltaNgl / perLayerMib;
    return {
      hostBacked: slopeRatio > HOST_BACKED_SLOPE_RATIO,
      method: "slope",
      slopeRatio,
      // Only the growth ABOVE the reference is attributable from this pair --
      // whatever the reference itself was already spilling is invisible here.
      spilledLayers: deltaShared > 0 ? Math.round(deltaShared / perLayerMib) : 0,
    };
  }

  // --- Bootstrap: the first rung of a phase has no slope --------------------
  //
  // Judged against its OWN predicted footprint, so the scale is the model's
  // rather than the machine's. This matters more than it looks: whatever
  // verdict the first rung gets becomes the reference every later slope is
  // measured from, and a spilling rung accepted here anchors the whole search
  // in the wrong place -- the slope only ever sees growth ABOVE its reference,
  // so it cannot notice that the reference was already dirty.
  //
  // An earlier version compared shared against dedicated (host-backed when
  // more of the process's GPU memory was system RAM than VRAM). Scale-free and
  // constant-free, but far too permissive to anchor on: simulated against the
  // measured sweep it passed a rung already spilling four layers, and the
  // ladder then converged there instead of descending to the real boundary.
  if (input.estimatedGpuMib == null || input.estimatedGpuMib <= 0) return UNAVAILABLE_HOST_BACKED;
  return {
    hostBacked: rung.sharedPeakMib / input.estimatedGpuMib > HOST_BACKED_BOOTSTRAP_FRAC,
    method: "ratio",
    slopeRatio: null,
    spilledLayers: null,
  };
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
  // <architecture>.attention.sliding_window_pattern / .shared_kv_layers --
  // see worker/src/gguf.ts's GgufInfo doc comments for what each means and
  // SWA_PATTERN_FALLBACK/swaGlobalLayersInSuffix for how they refine the SWA
  // branch below. Both undefined (the common case) reproduces the exact
  // pre-existing behavior: the hardcoded ~1-in-6 fallback, no shared layers.
  slidingWindowPattern?: number;
  sharedKvLayers?: number;
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

// Fallback global:local interleave period, used only when a model's own
// header doesn't resolve a real one (worker/src/gguf.ts's
// GgufInfo.sliding_window_pattern -- null either because the key is absent,
// or because the architecture writes it as a full per-layer array rather
// than a scalar period, confirmed live on a real Gemma-4 GGUF). Matches
// Gemma-3's own default (llama.cpp's llama_model_gemma3::load_arch_hparams),
// but real periods vary meaningfully by architecture -- confirmed against
// llama.cpp's own hparam loaders: gemma2/openai-moe use 2, cohere2/llama4/
// olmo2 use 4, gemma3 (and, empirically, a real gemma-4-12b GGUF's per-layer
// array) use 6. This is a documented fallback, not a universally-correct
// default: it silently UNDERcounts global layers -- and so underestimates
// VRAM need, the unsafe direction -- for a period-2/4 architecture whose
// file doesn't declare its own pattern.
const SWA_PATTERN_FALLBACK = 6;

// Global (full-attention) layer count within a GPU-resident SUFFIX of
// `subsetLen` blocks, for the Gemma-style hybrid attention pattern where the
// model's LAST block is always global and full-attention blocks repeat every
// `period`-th block counting backward from there (see the module's SWA doc
// comments below for the fuller rationale, and SWA_PATTERN_FALLBACK's own
// comment for what `period` defaults to and why). Because that phase is
// anchored at the model's own final block -- never at wherever a subset
// happens to start -- this count depends only on subsetLen (and period), not
// on the size of the model the subset was taken from: it's exact (given the
// pattern assumption) whether the subset is the whole model
// (maxAffordableContext below) or the GPU-resident tail of a partial --ngl
// offload (computeDualPoolFit's gpuKvLayers further down), since llama.cpp
// always offloads the LAST `ngl` blocks first -- i.e. the GPU side is always
// a suffix ending at the same final block the pattern is anchored to. The
// same anchoring lets it also count global layers within a model's
// shared-KV TRAILING block (see maxAffordableContext/computeDualPoolFit's
// sharedKvLayers handling) -- that block is itself just another such suffix.
function swaGlobalLayersInSuffix(subsetLen: number, period: number = SWA_PATTERN_FALLBACK): number {
  return subsetLen <= 0 ? 0 : Math.ceil(subsetLen / Math.max(1, period));
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
    // the unsafe direction. Uses the model's own declared global:local
    // interleave period when available, falling back to the Gemma-3-style
    // ~1-in-6 assumption otherwise -- see SWA_PATTERN_FALLBACK's doc comment.
    // Local layers contribute their window-bound bytes and the global budget
    // is solved for.
    const period = input.slidingWindowPattern ?? SWA_PATTERN_FALLBACK;
    // Shared (KV-reusing) layers are themselves the model's own trailing
    // suffix (llama.cpp reuses an earlier layer's cache for exactly its LAST
    // `sharedKvLayers` blocks -- see worker/src/gguf.ts's
    // GgufInfo.shared_kv_layers), so subtracting a same-anchor suffix count
    // from the whole-model count is exact given the pattern assumption, the
    // same trick computeDualPoolFit's GPU/CPU split uses below. Clamped to
    // nLayer-1 (never nLayer itself): llama.cpp's own gemma3n/gemma4 loader
    // asserts at least 2 layers keep an independent KV cache, so a model
    // literally cannot have zero real KV-allocating layers.
    const sharedInSuffix = Math.min(Math.max(0, input.nLayer - 1), Math.max(0, input.sharedKvLayers ?? 0));
    const totalGlobal = swaGlobalLayersInSuffix(input.nLayer, period);
    const globalInSharedTail = swaGlobalLayersInSuffix(sharedInSuffix, period);
    // Clamped to >=1: a model whose ENTIRE global-layer set happened to fall
    // inside its shared tail would make this 0, which this inversion can't
    // divide by. Treating one local layer as if it scaled with context
    // charges MORE bytes per requested token than that layer really needs,
    // which can only shrink the inverted "max affordable context" -- the
    // safe direction, consistent with this whole branch's
    // never-overestimate-affordable-tokens posture.
    const g = Math.max(1, totalGlobal - globalInSharedTail);
    const effectiveLayers = input.nLayer - sharedInSuffix;
    const perLayerBytes = kvBytesPerToken({ ...input, nLayer: 1 }).bytes ?? 0;
    // Only the g GLOBAL layers scale with the requested context; the
    // (effectiveLayers - g) local layers are capped at slidingWindow tokens
    // each no matter how large the context gets, so their cost is fixed --
    // pay it out of the budget first, then divide what's left among just the
    // g layers that actually grow. Dividing the whole budget by
    // (perLayerBytes * effectiveLayers) here (as this used to) instead of by
    // (perLayerBytes * g) undercounts each global layer's real share by a
    // factor of effectiveLayers/g -- for a 49-layer model with 1-in-6 global
    // layers, roughly 49x too little.
    const localFixedCostMib = ((effectiveLayers - g) * input.slidingWindow * perLayerBytes) / 2 ** 20;
    const globalBudgetMib = Math.max(0, usableMib - localFixedCostMib);
    const tokens = (globalBudgetMib * 2 ** 20) / (perLayerBytes * g);
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
// Reuses kvBytesPerToken/estimateKvCacheMib UNCHANGED for the non-SWA case:
// because that formula is linear in nLayer, the GPU share of KV cache is the
// same call with nLayer:ngl and the CPU share is nLayer:(kvLayerCount-ngl) --
// no new KV math needed, just re-parameterizing the existing one twice.
//
// When slidingWindow IS set, splits each side's KV further into its global
// (full-attention) and local (window-bound) layer counts via
// swaGlobalLayersInSuffix, rather than falling back to the plain formula's
// blanket overestimate: llama.cpp always offloads the LAST `ngl` blocks
// first (placeWeightBytes' own i_gpu_start convention), and the Gemma-style
// pattern's global layer is always the model's LAST block -- so the
// GPU-resident side is always a suffix ending at that same anchor, making
// swaGlobalLayersInSuffix(gpuKvLayers) exact (given the pattern assumption),
// not just a whole-model approximation. The CPU side's global count is
// simply whatever's left: swaGlobalLayersInSuffix(kvLayerCount) -
// swaGlobalLayersInSuffix(gpuKvLayers) -- exact for the same reason, since
// both counts are taken from the same anchored pattern. Confidence still
// drops one level whenever slidingWindow is set (dualPoolConfidence below):
// even with a real slidingWindowPattern period read from the header, which
// layers carry NO independent KV at all still relies on sharedKvLayers being
// exactly the model's trailing N blocks (see swaGlobalLayersInSuffix's own
// doc comment) rather than something read layer-by-layer.
//
// sharedKvLayers (when set) further excludes the model's trailing
// KV-reusing blocks from both the global and local counts entirely -- see
// GgufInfo.shared_kv_layers's doc comment -- using the same suffix-anchored
// subtraction trick, since that trailing block is itself just another suffix
// of the same anchored pattern.

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
  // See MaxCtxInput's matching fields for what each means.
  slidingWindowPattern?: number;
  sharedKvLayers?: number;
  parallelSlots?: number;
  vram: PoolReading;
  ram: PoolReading;
  // Metal, or any visible GPU with vram_dynamic:true.
  unifiedPool: boolean;
  // --n-cpu-moe -- see shared/sweep.ts's SweepItem.n_cpu_moe. Only consumed
  // alongside tensorBreakdown below.
  nCpuMoe?: number;
  // Real per-tensor byte breakdown -- see estimateVramNeededMib's own doc
  // comment (VramNeedEstimateInput.tensorBreakdown) for what this changes and
  // why. Absent/null falls back to the flat per-layer average exactly as
  // before.
  tensorBreakdown?: TensorLayerBreakdown | null;
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
  let gpuWeightsMib: number;
  let cpuWeightsMib: number;
  if (input.tensorBreakdown && input.kvLayerCount > 0) {
    const { gpuBytes, cpuBytes } = placeWeightBytes(input.tensorBreakdown, input.kvLayerCount, ngl, input.nCpuMoe ?? 0);
    gpuWeightsMib = gpuBytes / BYTES_PER_MIB;
    cpuWeightsMib = cpuBytes / BYTES_PER_MIB;
  } else {
    const bytesPerLayer = input.modelSizeBytes / input.totalModelLayers;
    gpuWeightsMib = (bytesPerLayer * ngl) / BYTES_PER_MIB;
    cpuWeightsMib = (bytesPerLayer * (input.totalModelLayers - ngl)) / BYTES_PER_MIB;
  }

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
  let gpuKvMib: number;
  let cpuKvMib: number;
  if (input.slidingWindow != null && input.slidingWindow > 0) {
    // See this function's doc comment above: the GPU side is always a
    // suffix ending at the model's last block, so swaGlobalLayersInSuffix
    // applies directly to it; the CPU side's global count is just the
    // model-wide total minus the GPU side's share. Uses the model's own
    // declared period when available, falling back to the Gemma-3-style
    // ~1-in-6 assumption otherwise -- see SWA_PATTERN_FALLBACK's doc comment.
    const period = input.slidingWindowPattern ?? SWA_PATTERN_FALLBACK;
    // The shared (KV-reusing) tail is itself a suffix of the whole model,
    // anchored at the same final block -- so splitting it across the GPU/CPU
    // boundary is the same min()-against-gpuKvLayers trick as gpuKvLayers
    // itself above: whichever of it lands within the GPU's own suffix stays
    // there, the rest falls to the CPU side.
    const sharedInSuffix = Math.min(input.kvLayerCount, Math.max(0, input.sharedKvLayers ?? 0));
    const gpuSharedInSuffix = Math.min(gpuKvLayers, sharedInSuffix);
    const cpuSharedInSuffix = sharedInSuffix - gpuSharedInSuffix;
    const totalGlobal = swaGlobalLayersInSuffix(input.kvLayerCount, period);
    const totalGlobalShared = swaGlobalLayersInSuffix(sharedInSuffix, period);
    const gpuGlobalAll = swaGlobalLayersInSuffix(gpuKvLayers, period);
    const gpuGlobalShared = swaGlobalLayersInSuffix(gpuSharedInSuffix, period);
    // Shared layers carry no independent KV cache at all (see
    // GgufInfo.shared_kv_layers) -- excluded from every count below rather
    // than priced as global OR local. When sharedKvLayers is 0/undefined
    // (the common case), sharedInSuffix/gpuSharedInSuffix/*Shared above are
    // all 0 and every line below reduces to exactly the pre-existing
    // formula.
    const gpuGlobal = Math.max(0, gpuGlobalAll - gpuGlobalShared);
    const cpuGlobal = Math.max(0, totalGlobal - totalGlobalShared - gpuGlobal);
    const gpuLocal = Math.max(0, gpuKvLayers - gpuSharedInSuffix - gpuGlobal);
    const cpuLocal = Math.max(0, cpuKvLayers - cpuSharedInSuffix - cpuGlobal);
    const perLayerBytes = kvBytesPerToken({ ...kvGeometry, nLayer: 1 }).bytes ?? 0;
    const localTokens = Math.min(input.ctxTokens, input.slidingWindow);
    const slots = Math.max(1, input.parallelSlots ?? 1);
    gpuKvMib = (perLayerBytes * (gpuGlobal * input.ctxTokens + gpuLocal * localTokens) * slots) / BYTES_PER_MIB;
    cpuKvMib = (perLayerBytes * (cpuGlobal * input.ctxTokens + cpuLocal * localTokens) * slots) / BYTES_PER_MIB;
  } else {
    gpuKvMib =
      estimateKvCacheMib({ ...kvGeometry, nLayer: gpuKvLayers, tokens: input.ctxTokens, parallelSlots: input.parallelSlots }) ??
      0;
    cpuKvMib =
      estimateKvCacheMib({ ...kvGeometry, nLayer: cpuKvLayers, tokens: input.ctxTokens, parallelSlots: input.parallelSlots }) ??
      0;
  }

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
  // See MaxCtxInput's matching fields for what each means.
  slidingWindowPattern?: number;
  sharedKvLayers?: number;
  parallelSlots?: number;
  vram: PoolReading;
  ram: PoolReading;
  unifiedPool: boolean;
  noGpu: boolean;
  // See DualPoolInput's matching fields -- threaded through to every internal
  // fitAt/estimateSafeNgl/maxAffordableContext call below.
  nCpuMoe?: number;
  tensorBreakdown?: TensorLayerBreakdown | null;
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
      slidingWindowPattern: input.slidingWindowPattern,
      sharedKvLayers: input.sharedKvLayers,
      parallelSlots: input.parallelSlots,
      vram: input.vram,
      ram: input.ram,
      unifiedPool: input.unifiedPool,
      nCpuMoe: input.nCpuMoe,
      tensorBreakdown: input.tensorBreakdown,
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
      : estimateSafeNgl(input.modelSizeBytes, input.totalModelLayers, input.vram.freeMib ?? 0, input.tensorBreakdown);

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
      const gpuWeightsMib =
        input.tensorBreakdown && input.kvLayerCount > 0
          ? placeWeightBytes(input.tensorBreakdown, input.kvLayerCount, ngl, input.nCpuMoe ?? 0).gpuBytes / BYTES_PER_MIB
          : (bytesPerLayer * ngl) / BYTES_PER_MIB;
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
