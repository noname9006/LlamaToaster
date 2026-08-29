import { describe, expect, it } from "vitest";
import {
  computeDualPoolFit,
  suggestPlacementConfigs,
  estimateKvCacheMib,
  MIN_FEASIBLE_CTX,
  type DualPoolInput,
  type SuggestInput,
  type TensorLayerBreakdown,
} from "./vramEstimate.js";

const BYTES_PER_MIB = 1024 * 1024;

// A clean 10-layer model (9 real transformer layers + 1 output layer, same
// n_layer+1 convention as estimateVramNeededMib): 1000 MiB/layer, f16/f16 KV,
// nHeadKv 4, head dim 128 both sides -- kv bytes/token/layer = 4*(128*2+128*2)
// = 2048 bytes, so kv cost is easy to hand-verify at any (ngl, ctx).
const MODEL = {
  modelSizeBytes: 10_000 * BYTES_PER_MIB,
  totalModelLayers: 10,
  kvLayerCount: 9,
  nHeadKv: 4,
  headDimK: 128,
  headDimV: 128,
  cacheTypeK: "f16",
  cacheTypeV: "f16",
} as const;

function dualInput(overrides: Partial<DualPoolInput>): DualPoolInput {
  return {
    ...MODEL,
    ngl: 5,
    ctxTokens: 1000,
    vram: { freeMib: 100_000, totalMib: 100_000 },
    ram: { freeMib: 100_000, totalMib: 100_000 },
    unifiedPool: false,
    ...overrides,
  };
}

describe("computeDualPoolFit", () => {
  it("splits weights and KV so the two pools sum back to the unsplit whole (linearity)", () => {
    const ngl = 4;
    const ctx = 2000;
    const fit = computeDualPoolFit(dualInput({ ngl, ctxTokens: ctx }));
    expect(fit.gpu.weightsMib + fit.cpu.weightsMib).toBeCloseTo(MODEL.modelSizeBytes / BYTES_PER_MIB, 6);
    const wholeKv = estimateKvCacheMib({ ...MODEL, nLayer: MODEL.kvLayerCount, tokens: ctx }) ?? 0;
    expect(fit.gpu.kvMib + fit.cpu.kvMib).toBeCloseTo(wholeKv, 6);
  });

  // The reported real-world bug: a flat file-size/layer-count average spreads
  // a share of the (often huge) token-embedding tensor across every
  // requested layer, overstating VRAM need at partial offload. A real
  // tensorBreakdown must report materially less GPU weight for the same
  // (small) ngl once that embedding is correctly excluded.
  it("with a real tensorBreakdown, reports less GPU weight than the flat average when a huge embedding is involved", () => {
    const breakdown: TensorLayerBreakdown = {
      dense: new Array(9).fill(1000 * BYTES_PER_MIB), // 9 layers, matches MODEL.kvLayerCount
      moe: new Array(9).fill(0),
      embed: 900_000 * BYTES_PER_MIB, // dwarfs the transformer blocks, e.g. a huge-vocab model
      output: 100_000 * BYTES_PER_MIB,
      other: 0,
    };
    // Same total file size either way (9000 + 900_000 + 100_000 = 1_009_000
    // MiB across totalModelLayers:10) -- only the PLACEMENT assumption
    // differs, isolating exactly what this fix changes.
    const modelSizeBytes = 1_009_000 * BYTES_PER_MIB;
    const flat = computeDualPoolFit(dualInput({ ngl: 2, modelSizeBytes }));
    const accurate = computeDualPoolFit(dualInput({ ngl: 2, modelSizeBytes, tensorBreakdown: breakdown }));
    expect(accurate.gpu.weightsMib).toBe(2000); // exactly 2 blocks' worth, no embedding share
    expect(accurate.gpu.weightsMib).toBeLessThan(flat.gpu.weightsMib);
  });

  it("with a tensorBreakdown, --n-cpu-moe moves the pinned blocks' expert bytes from GPU to CPU", () => {
    const breakdown: TensorLayerBreakdown = {
      dense: new Array(9).fill(100 * BYTES_PER_MIB),
      moe: new Array(9).fill(500 * BYTES_PER_MIB),
      embed: 0,
      output: 0,
      other: 0,
    };
    const noCpuMoe = computeDualPoolFit(dualInput({ ngl: 9, tensorBreakdown: breakdown, nCpuMoe: 0 }));
    const withCpuMoe = computeDualPoolFit(dualInput({ ngl: 9, tensorBreakdown: breakdown, nCpuMoe: 3 }));
    expect(withCpuMoe.gpu.weightsMib).toBeCloseTo(noCpuMoe.gpu.weightsMib - 3 * 500, 6);
    expect(withCpuMoe.cpu.weightsMib).toBeCloseTo(noCpuMoe.cpu.weightsMib + 3 * 500, 6);
  });

  it("keeps unified-pool need identical for every ngl -- VRAM and RAM are the same bytes", () => {
    const base = { ...MODEL, ctxTokens: 5000, vram: { freeMib: 50_000, totalMib: 50_000 }, ram: { freeMib: 50_000, totalMib: 50_000 }, unifiedPool: true };
    const at0 = computeDualPoolFit({ ...base, ngl: 0 });
    const at5 = computeDualPoolFit({ ...base, ngl: 5 });
    const at10 = computeDualPoolFit({ ...base, ngl: 10 });
    expect(at5.gpu.neededMib).toBeCloseTo(at0.gpu.neededMib, 6);
    expect(at10.gpu.neededMib).toBeCloseTo(at0.gpu.neededMib, 6);
    expect(at0.fits).toBe(at5.fits);
    expect(at0.fits).toBe(at10.fits);
  });

  // With a sliding window set, the GPU side is always the SUFFIX of
  // GPU-resident blocks (llama.cpp offloads the last `ngl` blocks first) and
  // the Gemma-style pattern's global block is always the model's LAST
  // block -- so global-layer counts are exact, not approximated, on both
  // sides: gpuGlobal = ceil(gpuKvLayers/6), cpuGlobal = ceil(kvLayerCount/6)
  // - gpuGlobal. Here kvLayerCount:9, ngl:5 -> gpuKvLayers:5, cpuKvLayers:4,
  // totalGlobal:ceil(9/6)=2, gpuGlobal:ceil(5/6)=1, cpuGlobal:2-1=1. Per-layer
  // cost is MODEL's own 2048 bytes/token/layer (see its doc comment).
  it("splits KV by global (full-ctx) vs local (window-bound) layers when a sliding window is set", () => {
    const ctxTokens = 4000;
    const slidingWindow = 512;
    const plain = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens }));
    const swa = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens, slidingWindow }));

    // gpu: 1 global layer @ full ctx + 4 local layers capped at the window.
    const expectedGpuBytes = 2048 * (1 * ctxTokens + 4 * slidingWindow);
    // cpu: 1 global layer @ full ctx + 3 local layers capped at the window.
    const expectedCpuBytes = 2048 * (1 * ctxTokens + 3 * slidingWindow);
    expect(swa.gpu.kvMib).toBeCloseTo(expectedGpuBytes / BYTES_PER_MIB, 6);
    expect(swa.cpu.kvMib).toBeCloseTo(expectedCpuBytes / BYTES_PER_MIB, 6);

    // Window-capping must strictly reduce need vs the naive all-full-ctx
    // formula -- this is the whole point of the SWA-aware split.
    expect(swa.gpu.kvMib).toBeLessThan(plain.gpu.kvMib);
    expect(swa.cpu.kvMib).toBeLessThan(plain.cpu.kvMib);

    expect(plain.confidence).toBe("good");
    expect(swa.confidence).toBe("rough");
  });

  // ctxTokens below the window means no layer is actually capped yet -- SWA
  // and plain math must agree exactly here, same as the real llama.cpp
  // behavior (a sliding-window cache is identical to a full one up to the
  // window size).
  it("agrees with the plain formula whenever ctx is within the sliding window", () => {
    const swa = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens: 400, slidingWindow: 512 }));
    const plain = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens: 400 }));
    expect(swa.gpu.kvMib).toBeCloseTo(plain.gpu.kvMib, 6);
    expect(swa.cpu.kvMib).toBeCloseTo(plain.cpu.kvMib, 6);
  });

  it("reports fits:null (not false) when a pool's live reading is missing", () => {
    const missingVram = computeDualPoolFit(dualInput({ ngl: 5, vram: { freeMib: null, totalMib: null } }));
    expect(missingVram.gpu.fits).toBeNull();
    expect(missingVram.fits).toBeNull();

    const missingRam = computeDualPoolFit(dualInput({ ngl: 5, ram: { freeMib: null, totalMib: null } }));
    expect(missingRam.cpu.fits).toBeNull();
    expect(missingRam.fits).toBeNull();
  });

  it("treats zero GPU-resident KV layers as zero bytes needed, not unknown", () => {
    const fit = computeDualPoolFit(dualInput({ ngl: 0, ctxTokens: 1000 }));
    expect(fit.gpu.kvMib).toBe(0);
    expect(fit.gpu.weightsMib).toBe(0);
  });

  // Regression case: a real Gemma-4-12B-style model (48 layers, GQA
  // nHeadKv:8/headDim:256, 1024-token sliding window, ngl:0 i.e. CPU-only)
  // at ctx:32768. Before this fix, every one of the 40 window-bound layers
  // was wrongly charged the full 32768-token cost instead of its real
  // 1024-token cap: plain math gives ~24.3 GiB (weights ~11.83 + naive KV 12
  // + 0.5 overhead). The live app was observed showing ~54.8 GiB for this
  // exact model/ctx/ngl -- notably MORE than even this naive-but-correct-
  // head-count figure, which this fix alone doesn't explain (points at a
  // separate metadata discrepancy for that specific GGUF, e.g. a wrong
  // stored n_head_kv). What this test locks in is that the SWA-aware math
  // itself is correct given accurate inputs: only the 8 global layers
  // (ceil(48/6)) scale with ctx, the other 40 stay capped at the window,
  // landing far below the naive figure either way.
  it("keeps a Gemma-3/4-style 48-layer hybrid-attention model's RAM estimate realistic at CPU-only offload", () => {
    const modelSizeBytes = 12.7e9; // Q8_0 file size, decimal GB per HF's own listing
    const input: DualPoolInput = {
      modelSizeBytes,
      totalModelLayers: 49,
      kvLayerCount: 48,
      ngl: 0,
      ctxTokens: 32_768,
      nHeadKv: 8,
      headDimK: 256,
      headDimV: 256,
      cacheTypeK: "f16",
      cacheTypeV: "f16",
      slidingWindow: 1024,
      vram: { freeMib: 5.2 * 1024, totalMib: 5.2 * 1024 },
      ram: { freeMib: 12.2 * 1024, totalMib: 12.2 * 1024 },
      unifiedPool: false,
    };
    const swa = computeDualPoolFit(input);
    const plain = computeDualPoolFit({ ...input, slidingWindow: undefined });

    // Naive (old) behavior: every layer charged the full 32768-token cost.
    const plainGiB = plain.cpu.neededMib / 1024;
    expect(plainGiB).toBeGreaterThan(23);
    expect(plainGiB).toBeLessThan(26);
    // SWA-aware (new) behavior lands well below it.
    const swaGiB = swa.cpu.neededMib / 1024;
    expect(swaGiB).toBeGreaterThan(13);
    expect(swaGiB).toBeLessThan(16);
    expect(swa.cpu.neededMib).toBeLessThan(plain.cpu.neededMib);
  });
});

// Fixture for suggestPlacementConfigs: full offload (ngl=10) doesn't fit
// 8000 MiB of free VRAM at ctx=1000, but ngl=7 does -- hand-verified:
//   ngl=10: weights 10000 + kv(9L,1000tok)≈17.58 + 512 ≈ 10529.6 > 8000 (fails)
//   ngl=8:  weights 8000  + kv(8L,1000tok)≈15.63 + 512 ≈ 8527.6  > 8000 (fails)
//   ngl=7:  weights 7000  + kv(7L,1000tok)≈13.67 + 512 ≈ 7525.7 <= 8000 (fits);
//           CPU side: weights 3000 + kv(2L,1000tok)≈3.91 + 512 ≈ 3515.9 <= 11000 (fits)
// ram.freeMib=11000 is also generous enough for floorFit's own ngl=0 check
// (all 10 layers on CPU: 10000 + ~4.5 + 512 ≈ 10516.5), which this fixture
// doesn't otherwise care about but the cannot_run/unknown precondition does.
const PARTIAL_OFFLOAD_FIXTURE: SuggestInput = {
  ...MODEL,
  currentNgl: 10,
  currentCtx: 1000,
  trainedCtx: null,
  vram: { freeMib: 8000, totalMib: 8000 },
  ram: { freeMib: 11_000, totalMib: 11_000 },
  unifiedPool: false,
  noGpu: false,
};

describe("suggestPlacementConfigs", () => {
  it("Config A keeps the target ctx and finds the largest ngl that fits both pools", () => {
    const result = suggestPlacementConfigs(PARTIAL_OFFLOAD_FIXTURE);
    expect(result.outcome).toBe("ok");
    const a = result.configs.find((c) => c.label === "target_ctx_reduce_offload");
    expect(a).toEqual({ label: "target_ctx_reduce_offload", ngl: 7, ctx: 1000 });
  });

  it("Config B pins ngl at the VRAM-weights-only ceiling and reduces ctx to fit, never above the target", () => {
    // A dedicated fixture (not PARTIAL_OFFLOAD_FIXTURE): currentCtx is huge
    // (1,000,000) so B's own VRAM-driven ceiling at ngl=7 -- well under that
    // -- is what actually binds, forcing the halving loop to engage rather
    // than trivially capping back down to whatever ctx Config A already
    // used (which is what happens, and correctly so, when the target ctx is
    // already below the ceiling -- see the nglCeiling===0 test below for
    // that legitimate collision case).
    const fixture: SuggestInput = {
      ...MODEL,
      currentNgl: 10,
      currentCtx: 1_000_000,
      trainedCtx: null,
      vram: { freeMib: 8000, totalMib: 8000 },
      ram: { freeMib: 1_000_000, totalMib: 1_000_000 }, // generous -- this test isolates the VRAM-side reduction
      unifiedPool: false,
      noGpu: false,
    };
    const result = suggestPlacementConfigs(fixture);
    const b = result.configs.find((c) => c.label === "max_offload_reduce_ctx");
    expect(b).toBeDefined();
    expect(b!.ngl).toBe(7); // nglCeiling for 8000 MiB free at this model
    expect(b!.ctx).toBeGreaterThanOrEqual(MIN_FEASIBLE_CTX);
    expect(b!.ctx).toBeLessThan(fixture.currentCtx);
    const fit = computeDualPoolFit({ ...MODEL, ngl: b!.ngl, ctxTokens: b!.ctx, vram: fixture.vram, ram: fixture.ram, unifiedPool: false });
    expect(fit.fits).toBe(true);
  });

  it("degenerate nglCeiling===0: Config B falls back to an all-CPU placement instead of crashing on a zero-layer inversion", () => {
    const result = suggestPlacementConfigs({
      ...MODEL,
      currentNgl: 10,
      currentCtx: 2000,
      trainedCtx: null,
      vram: { freeMib: 100, totalMib: 100 }, // not even one layer's weights fit
      ram: { freeMib: 11_000, totalMib: 11_000 },
      unifiedPool: false,
      noGpu: false,
    });
    expect(result.outcome).toBe("ok");
    // nglCeiling=0 collapses Config A ("fix ctx, vary ngl" -- only ngl=0 is
    // ever tried) and Config B ("fix ngl=ceiling=0, vary ctx") onto the same
    // point; dedup keeps whichever was pushed first, so only one label
    // survives -- what matters is the (ngl, ctx) pair itself is present.
    expect(result.configs.some((c) => c.ngl === 0 && c.ctx === 2000)).toBe(true);
  });

  it("outcome unknown (not cannot_run) when the VRAM reading is missing", () => {
    const result = suggestPlacementConfigs({ ...PARTIAL_OFFLOAD_FIXTURE, vram: { freeMib: null, totalMib: null } });
    expect(result.outcome).toBe("unknown");
    expect(result.configs).toEqual([]);
  });

  it("outcome unknown (not cannot_run) when the RAM reading is missing", () => {
    const result = suggestPlacementConfigs({ ...PARTIAL_OFFLOAD_FIXTURE, ram: { freeMib: null, totalMib: null } });
    expect(result.outcome).toBe("unknown");
    expect(result.configs).toEqual([]);
  });

  it("outcome cannot_run when nothing fits at any placement or context, ever", () => {
    const result = suggestPlacementConfigs({
      ...MODEL,
      currentNgl: 10,
      currentCtx: 1000,
      trainedCtx: null,
      vram: { freeMib: 10, totalMib: 10 },
      ram: { freeMib: 10, totalMib: 10 }, // far below even the minimum viable RAM need
      unifiedPool: false,
      noGpu: false,
    });
    expect(result.outcome).toBe("cannot_run");
    expect(result.configs).toEqual([]);
  });

  // Engineered so Config A (fixed ctx=1,000,000, every ngl) and Config B
  // (fixed ngl=nglCeiling=5, every ctx down to the floor) both come up empty,
  // even though floorFit (ngl=0, ctx=MIN_FEASIBLE_CTX) is proven to fit --
  // hand-verified: at ngl=5 even ctx=256 already exceeds the 5512 MiB VRAM
  // budget (5000 weights + 512 overhead + ~2.5 KV ≈ 5514.5), so B never
  // finds anything; at ctx=1,000,000 the RAM cost at every ngl in [0,5]
  // dwarfs the 10,517 MiB RAM budget, so A never finds anything either.
  it("guarantees at least one config even when Config A and Config B both come up empty", () => {
    const result = suggestPlacementConfigs({
      ...MODEL,
      currentNgl: 10,
      currentCtx: 1_000_000,
      trainedCtx: null,
      vram: { freeMib: 5512, totalMib: 5512 },
      ram: { freeMib: 10_517, totalMib: 10_517 },
      unifiedPool: false,
      noGpu: false,
    });
    expect(result.outcome).toBe("ok");
    expect(result.configs.length).toBeGreaterThanOrEqual(1);
    expect(result.configs.every((c) => c.label !== "target_ctx_reduce_offload" || c.ctx !== 1_000_000)).toBe(true);
    const fallback = result.configs.find((c) => c.label === "minimum_viable");
    expect(fallback).toBeDefined();
    const fit = computeDualPoolFit({
      ...MODEL,
      ngl: fallback!.ngl,
      ctxTokens: fallback!.ctx,
      vram: { freeMib: 5512, totalMib: 5512 },
      ram: { freeMib: 10_517, totalMib: 10_517 },
      unifiedPool: false,
    });
    expect(fit.fits).toBe(true);
  });

  it("noGpu forces nglCeiling to 0 and never touches the VRAM reading", () => {
    const result = suggestPlacementConfigs({
      ...MODEL,
      currentNgl: 0,
      currentCtx: 1000,
      trainedCtx: null,
      vram: { freeMib: null, totalMib: null }, // absent entirely, as on a real CPU-only worker
      ram: { freeMib: 20_000, totalMib: 20_000 },
      unifiedPool: false,
      noGpu: true,
    });
    expect(result.outcome).toBe("ok");
    expect(result.configs.every((c) => c.ngl === 0)).toBe(true);
  });
});
