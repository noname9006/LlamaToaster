import { describe, expect, it } from "vitest";
import {
  estimateResidentGpuLayers,
  estimateResidentGpuLayersFromBufferSizes,
  estimateVramNeededMib,
  isVramDiscrepancy,
  VRAM_ESTIMATE_FIXED_OVERHEAD_MIB,
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
