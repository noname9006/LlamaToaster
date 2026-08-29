import { describe, expect, it } from "vitest";
import {
  computeDualPoolFit,
  suggestPlacementConfigs,
  estimateKvCacheMib,
  MIN_FEASIBLE_CTX,
  type DualPoolInput,
  type SuggestInput,
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

  it("uses the plain KV formula regardless of a sliding window -- need is unchanged, confidence downgrades", () => {
    const plain = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens: 4000 }));
    const swa = computeDualPoolFit(dualInput({ ngl: 5, ctxTokens: 4000, slidingWindow: 512 }));
    expect(swa.gpu.kvMib).toBeCloseTo(plain.gpu.kvMib, 6);
    expect(swa.cpu.kvMib).toBeCloseTo(plain.cpu.kvMib, 6);
    expect(plain.confidence).toBe("good");
    expect(swa.confidence).toBe("rough");
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
