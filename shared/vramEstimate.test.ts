import { describe, expect, it } from "vitest";
import {
  estimateResidentGpuLayers,
  estimateResidentGpuLayersFromBufferSizes,
  estimateSafeNgl,
  estimateVramNeededMib,
  detectHostBackedFallback,
  HOST_BACKED_BOOTSTRAP_FRAC,
  isPrefillCliff,
  HOST_BACKED_SLOPE_RATIO,
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

// Every number below is a REAL measurement from a 19-load calibration sweep on
// an AMD Radeon RX 6600 XT (8176MiB) running llama.cpp b10819 (Vulkan) with
// Qwen3.6-35B-A3B-UD-IQ4_NL -- 17205MiB across 41 layers, so ~420MiB/layer
// straight from the file. dedicated/shared are what Windows' per-process WDDM
// counters reported during generation:
//
//   ngl    0    2    4    6    8   10   12   14   18    26    41
//   ded  277 1136 2014 2822 3442 3610 3434 3584 3543  3736  4069
//   shr   28  427  427  427  639 1279 2270 2928 4594  7647 13258
//   tps 8.92 10.5 11.3 11.6 11.9 12.1 10.9 10.4 9.25  3.54  6.87
describe("detectHostBackedFallback", () => {
  const PER_LAYER_MIB = 17205 / 41;
  const SWEEP: Record<number, { ded: number; shr: number }> = {
    0: { ded: 277, shr: 28 }, 2: { ded: 1136, shr: 427 }, 4: { ded: 2014, shr: 427 },
    6: { ded: 2822, shr: 427 }, 8: { ded: 3442, shr: 639 }, 10: { ded: 3610, shr: 1279 },
    12: { ded: 3434, shr: 2270 }, 14: { ded: 3584, shr: 2928 }, 18: { ded: 3543, shr: 4594 },
    26: { ded: 3736, shr: 7647 }, 41: { ded: 4069, shr: 13258 },
  };
  const sample = (ngl: number) => ({ ngl, sharedPeakMib: SWEEP[ngl].shr, dedicatedPeakMib: SWEEP[ngl].ded });
  const between = (lower: number, upper: number) =>
    detectHostBackedFallback({ rung: sample(upper), priorSameCtx: [sample(lower)], perLayerMib: PER_LAYER_MIB });

  // The whole basis of the method: overhead does not move with ngl, spilled
  // weights move with it one-for-one. Measured, the two regimes do not
  // overlap -- clean intervals 0.00-0.25, spilling ones 0.76-1.18.
  it.each([[2, 4], [4, 6], [6, 8]])("reads a clean interval %i->%i as a flat slope", (lo, hi) => {
    const v = between(lo, hi);
    expect(v.method).toBe("slope");
    expect(v.slopeRatio!).toBeLessThan(HOST_BACKED_SLOPE_RATIO);
    expect(v.hostBacked).toBe(false);
  });

  it.each([[8, 10], [10, 12], [12, 14], [14, 18], [18, 26], [26, 41]])(
    "reads a spilling interval %i->%i as ~one layer per layer",
    (lo, hi) => {
      const v = between(lo, hi);
      expect(v.slopeRatio!).toBeGreaterThan(HOST_BACKED_SLOPE_RATIO);
      expect(v.hostBacked).toBe(true);
    }
  );

  // ngl 0 -> 2 measures 0.47: the GPU compute buffers appearing the moment any
  // layer lands on the device. A one-off step, not a slope -- which is why a
  // zero-layer rung is never allowed to be the reference.
  it("never uses a zero-layer rung as the reference", () => {
    const v = detectHostBackedFallback({
      rung: sample(2), priorSameCtx: [sample(0)], perLayerMib: PER_LAYER_MIB, estimatedGpuMib: 1514,
    });
    expect(v.method).toBe("ratio");
  });

  it("attributes the growth above the reference, in layers", () => {
    // 26 vs 14: 4719MiB more system RAM for 12 more layers -> ~11 layers' worth.
    expect(between(14, 26).spilledLayers).toBe(11);
  });

  it("prefers the CLOSEST lower rung, so the slope stays local", () => {
    const v = detectHostBackedFallback({
      rung: sample(12),
      priorSameCtx: [sample(2), sample(10), sample(4)],
      perLayerMib: PER_LAYER_MIB,
    });
    // Against ngl 10 (the closest), not ngl 2: (2270-1279)/2/420 = 1.18.
    expect(v.slopeRatio!).toBeCloseTo(1.18, 1);
  });

  // A reference that is itself already spilling flattens the apparent slope.
  // That loses detections; it never invents them -- the safe direction, since
  // the ladder keeps searching rather than failing a placement that works.
  it("under-detects rather than over-detects when the reference is itself dirty", () => {
    expect(between(12, 14).hostBacked).toBe(true);
  });

  // Adjacent rungs are refused as references: at 1-layer resolution the two
  // regimes overlap outright (clean reaching +1.05, spilling dipping to +0.38)
  // because the driver's ~224MiB allocation granule is half a layer here.
  // These are real adjacent pairs from the 1-layer sweep -- both would be
  // misclassified if a span of 1 were allowed.
  it.each([
    ["a clean pair that reads as a steep slope", 6, 452, 7, 894],
    ["a spilling pair that reads as flat", 12, 1925, 13, 2083],
  ])("refuses an adjacent reference: %s", (_label, lowNgl, lowShr, hiNgl, hiShr) => {
    const v = detectHostBackedFallback({
      rung: { ngl: hiNgl, sharedPeakMib: hiShr, dedicatedPeakMib: 3900 },
      priorSameCtx: [{ ngl: lowNgl, sharedPeakMib: lowShr, dedicatedPeakMib: 3800 }],
      perLayerMib: PER_LAYER_MIB,
    });
    expect(v.method).not.toBe("slope");
  });

  it("reaches past an adjacent rung to a wide-enough one", () => {
    // ngl 11 against ngl 9 (span 2) rather than ngl 10 (span 1):
    // (1565-786)/2/420 = 0.93 -> correctly spilling. Against ngl 10 alone the
    // single-layer delta would read 1.51, right verdict for the wrong reason;
    // the pair below it (12 vs 11) would read 0.86 and 13 vs 12 only 0.38.
    const v = detectHostBackedFallback({
      rung: { ngl: 11, sharedPeakMib: 1565, dedicatedPeakMib: 3790 },
      priorSameCtx: [
        { ngl: 9, sharedPeakMib: 786, dedicatedPeakMib: 3865 },
        { ngl: 10, sharedPeakMib: 931, dedicatedPeakMib: 4015 },
      ],
      perLayerMib: PER_LAYER_MIB,
    });
    expect(v.method).toBe("slope");
    expect(v.slopeRatio!).toBeCloseTo(0.93, 2);
    expect(v.hostBacked).toBe(true);
  });

  describe("single-rung bootstrap", () => {
    // computeDualPoolFit's own figures for this model at ctx 1024.
    const EST = (ngl: number) => Math.round(706 + 404 * ngl);
    const alone = (ngl: number) =>
      detectHostBackedFallback({
        rung: sample(ngl), priorSameCtx: [], perLayerMib: PER_LAYER_MIB, estimatedGpuMib: EST(ngl),
      });

    it.each([12, 14, 18, 26, 41])("convicts ngl %i with no reference to slope against", (ngl) => {
      expect(alone(ngl)).toMatchObject({ hostBacked: true, method: "ratio" });
    });

    it.each([2, 4, 6, 8, 10])("stays silent at ngl %i with no reference", (ngl) => {
      expect(alone(ngl).hostBacked).toBe(false);
    });

    // max_gpu opens at every layer, so this is the rung the bootstrap must
    // catch for the search to start bracketing in the right place at all.
    it("convicts max_gpu's opening rung", () => {
      expect(alone(41)).toMatchObject({ hostBacked: true, method: "ratio" });
    });

    // The anchoring failure this threshold exists to prevent: the estimate's
    // safe-ngl landing point on this machine is ~13, already spilling four
    // layers. Passing it would anchor every later slope on a dirty reference.
    it("convicts the estimate's own landing point, which is already spilling", () => {
      expect(
        detectHostBackedFallback({
          rung: { ngl: 13, sharedPeakMib: 2599, dedicatedPeakMib: 3509 },
          priorSameCtx: [], perLayerMib: PER_LAYER_MIB, estimatedGpuMib: EST(13),
        }).hostBacked
      ).toBe(true);
    });

    // ...while a large context, whose KV legitimately lands partly in system
    // RAM, stays clean -- which is why the denominator is the full predicted
    // footprint and not the weights alone.
    it("leaves a 131072-token context at 4 layers alone", () => {
      expect(
        detectHostBackedFallback({
          rung: { ngl: 4, sharedPeakMib: 810, dedicatedPeakMib: 2393 },
          priorSameCtx: [], perLayerMib: PER_LAYER_MIB, estimatedGpuMib: 3276,
        }).hostBacked
      ).toBe(false);
    });

    it("is unavailable with no estimate to judge against", () => {
      expect(
        detectHostBackedFallback({ rung: sample(41), priorSameCtx: [], perLayerMib: PER_LAYER_MIB }).method
      ).toBeNull();
    });
  });

  it("is unavailable -- never 'clean' -- with no shared-memory counter", () => {
    expect(
      detectHostBackedFallback({
        rung: { ngl: 26, sharedPeakMib: null, dedicatedPeakMib: 3736 },
        priorSameCtx: [sample(10)],
        perLayerMib: PER_LAYER_MIB,
      })
    ).toEqual({ hostBacked: false, method: null, slopeRatio: null, spilledLayers: null });
  });

  it("falls back to the bootstrap when the per-layer size is unknown", () => {
    const v = detectHostBackedFallback({
      rung: sample(26), priorSameCtx: [sample(10)], perLayerMib: null, estimatedGpuMib: 11213,
    });
    expect(v.method).toBe("ratio");
    expect(v.hostBacked).toBe(true);
  });

  it("never judges a rung that asked for nothing on the GPU", () => {
    expect(detectHostBackedFallback({ rung: sample(0), priorSameCtx: [], perLayerMib: PER_LAYER_MIB }).method).toBeNull();
  });
});

// Prefill's own cliff, from the same 1-layer sweep. It sits THREE layers below
// the weights knee and is a far larger signal -- a 3.9x step, against the
// weights slope's 0.31-vs-0.62.
describe("isPrefillCliff", () => {
  const BEST_CLEAN = 69.2; // ngl 6, the best prompt rate measured

  it("fires on the measured collapse at ngl 7", () => {
    expect(isPrefillCliff(17.7, BEST_CLEAN)).toBe(true);
  });

  it("keeps firing where generation looks its best", () => {
    // ngl 10 is the FASTEST rung by gen tok/s (13.10) and still has prefill
    // ruined at 18.9 t/s -- the case that makes this worth reporting
    // separately rather than folding into one verdict.
    expect(isPrefillCliff(18.9, BEST_CLEAN)).toBe(true);
  });

  it.each([58.8, 63.1, 65.6, 67.1, 69.2])("leaves a healthy rate (%s t/s) alone", (pp) => {
    expect(isPrefillCliff(pp, BEST_CLEAN)).toBe(false);
  });

  it("cannot fire before a clean reference exists", () => {
    expect(isPrefillCliff(17.7, null)).toBe(false);
  });

  it("says nothing when this rung has no prompt timing", () => {
    expect(isPrefillCliff(null, BEST_CLEAN)).toBe(false);
  });
});
