import { describe, expect, it } from "vitest";
import {
  estimateResidentGpuLayers,
  estimateResidentGpuLayersFromBufferSizes,
  estimateSafeNgl,
  estimateVramNeededMib,
  isVramDiscrepancy,
  placeWeightBytes,
  VRAM_ESTIMATE_FIXED_OVERHEAD_MIB,
  type TensorLayerBreakdown,
} from "./vramEstimate.js";

const BYTES_PER_MIB = 1024 * 1024;

// One clean per-layer figure: a 10,000MiB file across 10 layers means every
// layer "costs" exactly 1000MiB in the flat-average model.
const TEN_GB_FILE = 10_000 * BYTES_PER_MIB;

describe("estimateResidentGpuLayers", () => {
  it("reports ~0 for the sysmem-fallback fingerprint: peak never left the pre-spawn baseline", () => {
    // The diagnosed real case (run 2bf0ce32): claimed 33/33 offloaded while
    // whole-adapter used memory stayed pinned at its pre-spawn level for the
    // entire run -- the driver served every allocation from system RAM.
    const est = estimateResidentGpuLayers({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      claimedNgl: 10,
      observedPeakMib: 2064,
      baselineUsedMib: 2064,
    });
    expect(est).toBe(0);
  });

  it("floors partial residency -- a fraction of one layer's worth of VRAM is not a resident layer", () => {
    const est = estimateResidentGpuLayers({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      claimedNgl: 10,
      observedPeakMib: 2064 + 2500, // 2.5 layers' worth of run-attributable growth
      baselineUsedMib: 2064,
    });
    expect(est).toBe(2);
  });

  it("never claims more layers than were requested", () => {
    const est = estimateResidentGpuLayers({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      claimedNgl: 3,
      observedPeakMib: 20_000, // far more growth than 3 layers could explain
      baselineUsedMib: 0,
    });
    expect(est).toBe(3);
  });

  it("treats a peak below the baseline as zero, not negative", () => {
    // Whole-adapter readings can DROP during a run when other processes free
    // memory -- that must read as "no evidence of residency", not NaN/negative.
    const est = estimateResidentGpuLayers({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      claimedNgl: 10,
      observedPeakMib: 1500,
      baselineUsedMib: 2064,
    });
    expect(est).toBe(0);
  });

  it("refuses to guess without a pre-run baseline to attribute the peak against", () => {
    // Without subtracting what the desktop/other processes already held, the
    // whole peak would masquerade as model residency -- null beats a wrong
    // number (the UI renders nothing instead).
    const est = estimateResidentGpuLayers({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      claimedNgl: 10,
      observedPeakMib: 2064,
      baselineUsedMib: null,
    });
    expect(est).toBeNull();
  });

  it("returns null for degenerate inputs (no claim, empty file, no layers)", () => {
    expect(
      estimateResidentGpuLayers({
        modelSizeBytes: TEN_GB_FILE,
        totalModelLayers: 10,
        claimedNgl: 0,
        observedPeakMib: 5000,
        baselineUsedMib: 1000,
      })
    ).toBeNull();
    expect(
      estimateResidentGpuLayers({
        modelSizeBytes: 0,
        totalModelLayers: 10,
        claimedNgl: 5,
        observedPeakMib: 5000,
        baselineUsedMib: 1000,
      })
    ).toBeNull();
    expect(
      estimateResidentGpuLayers({
        modelSizeBytes: TEN_GB_FILE,
        totalModelLayers: 0,
        claimedNgl: 5,
        observedPeakMib: 5000,
        baselineUsedMib: 1000,
      })
    ).toBeNull();
  });
});

describe("estimateResidentGpuLayersFromBufferSizes", () => {
  it("reproduces the diagnosed sysmem-fallback fingerprint from llama.cpp's own buffer report: all weight bytes on CPU", () => {
    // worker/src/bench.ts's parseModelBufferSizes for run 2bf0ce32 item 1
    // would have reported this exact split if it had been captured at the
    // time: llama.cpp claimed 33/33 offloaded but its own allocator put
    // every byte in CPU_Mapped.
    expect(estimateResidentGpuLayersFromBufferSizes(33, 0, 5880)).toBe(0);
  });

  it("splits proportionally to the real GPU/CPU byte split, not a flat per-layer average", () => {
    // 3/4 of the reported bytes landed on GPU -> ~3/4 of the claimed layers
    // read as resident, regardless of how big any single layer actually is.
    expect(estimateResidentGpuLayersFromBufferSizes(8, 3000, 1000)).toBe(6);
  });

  it("a fully GPU-resident buffer reads as fully resident", () => {
    expect(estimateResidentGpuLayersFromBufferSizes(33, 4368, 0)).toBe(33);
  });

  it("never claims more layers than were requested, and never fewer than zero", () => {
    expect(estimateResidentGpuLayersFromBufferSizes(0, 4368, 0)).toBeNull();
  });

  it("returns null when llama.cpp reported zero bytes on every backend", () => {
    expect(estimateResidentGpuLayersFromBufferSizes(10, 0, 0)).toBeNull();
  });
});

describe("estimateVramNeededMib + isVramDiscrepancy + estimateResidentGpuLayers round trip", () => {
  it("the reported log's numbers reproduce: flagged discrepancy with ~=0 resident", () => {
    // Qwen3.5-9B Q4_K_M-shaped inputs from run 2bf0ce32 item 1:
    // ~5880MiB estimated need vs an observed 2064MiB peak -> discrepancy
    // fires, and the same telemetry derives ~0 actually-resident layers.
    const modelSizeBytes = Math.round(6.16 * 1024 * BYTES_PER_MIB); // ~5880MiB over 33 layers + fixed overhead
    const estimatedMib = estimateVramNeededMib({
      modelSizeBytes,
      totalModelLayers: 33,
      requestedNgl: 33,
    });
    expect(estimatedMib).not.toBeNull();
    expect(isVramDiscrepancy(estimatedMib!, 2064)).toBe(true);

    // Sanity on the estimate itself: 33 equal shares of the file plus the
    // fixed overhead, rounded to MiB.
    const expectedMib = Math.round((modelSizeBytes + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB * BYTES_PER_MIB) / BYTES_PER_MIB);
    expect(estimatedMib).toBe(expectedMib);

    const residentEst = estimateResidentGpuLayers({
      modelSizeBytes,
      totalModelLayers: 33,
      claimedNgl: 33,
      observedPeakMib: 2064,
      baselineUsedMib: 24576 - 22512,
    });
    expect(residentEst).toBe(0);
  });

  it("a genuinely GPU-resident run does not flag and would not be annotated", () => {
    const modelSizeBytes = TEN_GB_FILE;
    const estimatedMib = estimateVramNeededMib({
      modelSizeBytes,
      totalModelLayers: 10,
      requestedNgl: 10,
    });
    // Peak comfortably above half the estimate -> claim stands unchallenged
    // (no discrepancy), which is exactly when gpu_layers_resident_est stays
    // null in the worker.
    expect(isVramDiscrepancy(estimatedMib!, estimatedMib! - 1)).toBe(false);
  });
});

// The bug this file's tensor-aware path exists to fix: the flat file-size/
// layer-count average spreads a share of the (often huge) token-embedding
// and output tensors across every requested layer, even though llama.cpp
// keeps those CPU-resident short of a full/near-full offload -- reliably
// OVERESTIMATING VRAM need, i.e. "the estimate says more memory is needed
// than llama.cpp actually uses". A real per-tensor breakdown must not repeat
// that mistake.
describe("placeWeightBytes -- real tensor ownership, not a flat per-layer average", () => {
  // 4 transformer blocks, each 100 bytes of dense weights; a comparatively
  // enormous 10,000-byte token embedding (the large-vocab case that most
  // exposes the flat average's error) and a small 50-byte output.
  const breakdown: TensorLayerBreakdown = {
    dense: [100, 100, 100, 100],
    moe: [0, 0, 0, 0],
    embed: 10_000,
    output: 50,
    other: 0,
  };

  it("keeps token_embd off the GPU for any partial offload, unlike a flat average", () => {
    // A flat average over (4 dense blocks + embed + output) would put a
    // large chunk of the 10,000-byte embedding on GPU as soon as ANY layer is
    // requested. Real llama.cpp placement doesn't move it until every block
    // is GPU-resident.
    const { gpuBytes } = placeWeightBytes(breakdown, 4, 2);
    expect(gpuBytes).toBe(200); // just the 2 GPU-resident blocks' dense bytes
  });

  it("moves token_embd to GPU only once every transformer block is offloaded", () => {
    const allBlocks = placeWeightBytes(breakdown, 4, 4);
    expect(allBlocks.gpuBytes).toBe(400 + 10_000); // 4 blocks' dense + embed, NOT output yet
    expect(allBlocks.cpuBytes).toBe(50); // just output

    const fullPlusOutput = placeWeightBytes(breakdown, 4, 5); // the "+1" convention
    expect(fullPlusOutput.gpuBytes).toBe(400 + 10_000 + 50);
    expect(fullPlusOutput.cpuBytes).toBe(0);
  });

  it("offloads the LAST ngl blocks (closest to output), not the first", () => {
    const uneven: TensorLayerBreakdown = { dense: [10, 20, 30, 40], moe: [0, 0, 0, 0], embed: 0, output: 0, other: 0 };
    // ngl=2 should take blocks [2,3] (30+40=70), not blocks [0,1] (10+20=30).
    expect(placeWeightBytes(uneven, 4, 2).gpuBytes).toBe(70);
  });

  it("--n-cpu-moe pins the first N blocks' experts to CPU even when those blocks are GPU-resident", () => {
    const moeBreakdown: TensorLayerBreakdown = {
      dense: [10, 10, 10, 10],
      moe: [200, 200, 200, 200],
      embed: 0,
      output: 0,
      other: 0,
    };
    // Full offload (ngl=4), n_cpu_moe=2: blocks 0-1's experts forced CPU,
    // blocks 2-3's experts (and every block's dense bytes) stay GPU.
    const { gpuBytes, cpuBytes } = placeWeightBytes(moeBreakdown, 4, 4, 2);
    expect(gpuBytes).toBe(10 * 4 + 200 * 2); // all dense + blocks 2,3's experts
    expect(cpuBytes).toBe(200 * 2); // blocks 0,1's experts only
  });

  it("n_cpu_moe on a block that's already CPU-resident (dense rule) changes nothing", () => {
    const moeBreakdown: TensorLayerBreakdown = {
      dense: [10, 10, 10, 10],
      moe: [200, 200, 200, 200],
      embed: 0,
      output: 0,
      other: 0,
    };
    // ngl=1 offloads only block 3; n_cpu_moe=2 targets blocks 0-1, which are
    // already CPU-resident via the dense rule alone -- pinning them again is
    // a no-op, and block 3 (the only GPU-resident block) isn't targeted at
    // all since 3 >= n_cpu_moe(2).
    const withMoe = placeWeightBytes(moeBreakdown, 4, 1, 2);
    const withoutMoe = placeWeightBytes(moeBreakdown, 4, 1, 0);
    expect(withMoe.gpuBytes).toBe(10 + 200); // block 3's dense + experts, unaffected
    expect(withMoe.gpuBytes).toBe(withoutMoe.gpuBytes);
  });

  it("puts everything on CPU at ngl=0", () => {
    const { gpuBytes, cpuBytes } = placeWeightBytes(breakdown, 4, 0);
    expect(gpuBytes).toBe(0);
    expect(cpuBytes).toBe(400 + 10_000 + 50);
  });

  it("the 'other' bucket always stays CPU-resident regardless of offload", () => {
    const withOther: TensorLayerBreakdown = { dense: [0], moe: [0], embed: 0, output: 0, other: 123 };
    expect(placeWeightBytes(withOther, 1, 1).cpuBytes).toBe(123);
    expect(placeWeightBytes(withOther, 1, 2).cpuBytes).toBe(123); // even at full "+1" offload
  });
});

describe("estimateVramNeededMib with a real tensorBreakdown", () => {
  // Same shape as the placeWeightBytes suite above: a large embedding that a
  // flat average would wrongly spread across every requested layer.
  const breakdown: TensorLayerBreakdown = {
    dense: [100, 100, 100, 100],
    moe: [0, 0, 0, 0],
    embed: 10_000,
    output: 50,
    other: 0,
  };

  it("does not include any of the embedding bytes for a partial offload", () => {
    const mib = estimateVramNeededMib({
      modelSizeBytes: 100_000, // deliberately inconsistent with breakdown -- must be ignored once tensorBreakdown is present
      totalModelLayers: 5, // n_layer(4) + 1
      requestedNgl: 2,
      tensorBreakdown: breakdown,
    });
    expect(mib).toBe(Math.round(200 / BYTES_PER_MIB + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB));
  });

  it("falls back to the flat average when no tensorBreakdown is supplied (unchanged behavior)", () => {
    const withBreakdown = estimateVramNeededMib({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      requestedNgl: 5,
      tensorBreakdown: null,
    });
    const withoutField = estimateVramNeededMib({
      modelSizeBytes: TEN_GB_FILE,
      totalModelLayers: 10,
      requestedNgl: 5,
    });
    expect(withBreakdown).toBe(withoutField);
  });
});

describe("estimateSafeNgl with a real tensorBreakdown", () => {
  it("finds a materially different (larger) safe ngl than the flat average once a huge embedding is excluded", () => {
    // 4 blocks of 1000 bytes each, plus a 40,000-byte embedding that a flat
    // average would spread across every layer -- the flat model badly
    // underestimates how many blocks actually fit once the embedding is
    // correctly excluded from the offload cost.
    const breakdown: TensorLayerBreakdown = {
      dense: [1000, 1000, 1000, 1000],
      moe: [0, 0, 0, 0],
      embed: 40_000,
      output: 0,
      other: 0,
    };
    const totalModelLayers = 5; // n_layer(4) + 1
    const modelSizeBytes = 4000 + 40_000; // matches breakdown's total, for a fair flat-average comparison
    const freeMib = (2500 + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB * BYTES_PER_MIB) / BYTES_PER_MIB; // budget for ~2.5 blocks' worth

    const flat = estimateSafeNgl(modelSizeBytes, totalModelLayers, freeMib);
    const accurate = estimateSafeNgl(modelSizeBytes, totalModelLayers, freeMib, breakdown);
    expect(accurate).toBe(2); // 2 blocks (2000 bytes) fit; a 3rd would need 3000
    expect(accurate).toBeGreaterThan(flat); // flat average's embedding dilution underestimates the ceiling
  });

  it("never returns a layer count whose real GPU bytes exceed the budget", () => {
    const breakdown: TensorLayerBreakdown = {
      dense: [500, 500, 500],
      moe: [0, 0, 0],
      embed: 100,
      output: 0,
      other: 0,
    };
    const safe = estimateSafeNgl(2000, 4, (1100 + VRAM_ESTIMATE_FIXED_OVERHEAD_MIB * BYTES_PER_MIB) / BYTES_PER_MIB, breakdown);
    const { gpuBytes } = placeWeightBytes(breakdown, 3, safe);
    expect(gpuBytes).toBeLessThanOrEqual(1100);
  });
});
