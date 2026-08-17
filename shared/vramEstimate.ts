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
