import { describe, expect, it } from "vitest";
import {
  maxAffordableContext,
  kvBytesPerToken,
  estimateKvCacheMib,
  residentWeightsMibFromPeak,
} from "./vramEstimate.js";

// The plan's own worked example (M1 exit): RX 6600 XT, Qwen3-30B-A3B Q4_K_M
// MoE placement -- 48 layers, n_head_kv 4, head dim 128 both sides, no
// sliding window, resident weight share 2969 MiB, scratch 256, headroom 10%.
const workedExample = {
  totalMib: 8192,
  weightsMib: 2969,
  scratchMib: 256,
  activationsHeadroomFrac: 0.1,
  parallelSlots: 1,
  nLayer: 48,
  nHeadKv: 4,
  headDimK: 128,
  headDimV: 128,
  cacheTypeK: "f16",
  cacheTypeV: "f16",
} as const;

describe("maxAffordableContext (M1)", () => {
  it("reproduces the worked example: ~44.2k at f16/f16 with confidence good and binding kv", () => {
    const est = maxAffordableContext(workedExample);
    // usable = 8192*0.9 - 2969 - 256 ≈ 4148 MiB; 96 KiB/token.
    expect(est.tokens).toBeGreaterThan(44_000);
    expect(est.tokens).toBeLessThan(44_400);
    expect(est.confidence).toBe("good");
    expect(est.binding).toBe("kv");
  });

  it("gives ≈83.3k at q8_0/q8_0 (51 KiB/token)", () => {
    const est = maxAffordableContext({ ...workedExample, cacheTypeK: "q8_0", cacheTypeV: "q8_0" });
    expect(est.tokens).toBeGreaterThan(83_000);
    expect(est.tokens).toBeLessThan(83_600);
    expect(est.confidence).toBe("good");
  });

  it("splits the KV budget across parallel slots", () => {
    const solo = maxAffordableContext(workedExample).tokens;
    const quad = maxAffordableContext({ ...workedExample, parallelSlots: 4 }).tokens;
    expect(quad).toBe(Math.floor(solo / 4));
  });

  it("never invents a weights number: null weights => unknown", () => {
    const est = maxAffordableContext({ ...workedExample, weightsMib: null });
    expect(est.confidence).toBe("unknown");
    expect(est.binding).toBeNull();
    expect(est.tokens).toBe(0);
  });

  it("offers the full-trained_ctx load as a conservative floor candidate when confidence is unknown", () => {
    const est = maxAffordableContext({ ...workedExample, weightsMib: null, trainedCtx: 32_768 });
    expect(est.confidence).toBe("unknown");
    expect(est.conservativeFloorTokens).toBe(32_768);
  });

  it("never offers a conservative floor candidate when confidence is NOT unknown -- it isn't a substitute estimate", () => {
    const est = maxAffordableContext({ ...workedExample, trainedCtx: 32_768 });
    expect(est.confidence).toBe("good");
    expect(est.conservativeFloorTokens).toBeUndefined();
  });

  it("offers no conservative floor candidate when trainedCtx itself is unknown", () => {
    const est = maxAffordableContext({ ...workedExample, weightsMib: null, trainedCtx: null });
    expect(est.conservativeFloorTokens).toBeNull();
  });

  it("drops a confidence level when head dims must come from the nEmbd/nHead fallback", () => {
    const est = maxAffordableContext({
      ...workedExample,
      headDimK: undefined,
      headDimV: undefined,
      nEmbd: 2048,
      nHead: 16,
    });
    // dK = 2048/16 = 128 -- same arithmetic as the worked example, lower trust.
    expect(est.confidence).toBe("rough");
    expect(est.tokens).toBe(maxAffordableContext(workedExample).tokens);
  });

  it("returns unknown when the fallback inputs are absent or incomplete", () => {
    const noDims = maxAffordableContext({ ...workedExample, headDimK: undefined, headDimV: undefined });
    expect(noDims.confidence).toBe("unknown");
    const halfFallback = maxAffordableContext({
      ...workedExample,
      headDimK: undefined,
      nEmbd: 2048,
      nHead: undefined,
    });
    expect(halfFallback.confidence).toBe("unknown");
  });

  it("reports binding weights-placement when the model barely seats before any KV exists", () => {
    const est = maxAffordableContext({ ...workedExample, weightsMib: 7400 });
    expect(est.binding).toBe("weights-placement");
    expect(est.tokens).toBe(0);
  });

  it("handles SWA conservatively: window-bound local layers, solved global budget, rough confidence", () => {
    // Gemma-3-style: sliding window 1024 on a 48-layer model -> g=8 global.
    const plain = { ...workedExample };
    const swa = maxAffordableContext({ ...plain, slidingWindow: 1024 });
    expect(swa.confidence).toBe("rough");
    // Local layers' bytes are capped by the window, so affordable tokens must
    // be finite and positive; naive all-layers math would wildly overcount.
    expect(swa.tokens).toBeGreaterThan(0);
    // The solved-global formula: the 40 local layers' fixed window-bound
    // cost comes out of the budget first, then what's left is split among
    // just the 8 layers that actually scale with context -- NOT divided
    // across all 48 (that would undercount each global layer's real share
    // by ~6x here, and by nLayer/g in general).
    const usableMib = 8192 * 0.9 - 2969 - 256;
    const perLayerBytes = 4 * (128 * 2 + 128 * 2);
    const localFixedCostMib = ((48 - 8) * 1024 * perLayerBytes) / 2 ** 20;
    const globalBudgetMib = Math.max(0, usableMib - localFixedCostMib);
    const expected = Math.floor((globalBudgetMib * 2 ** 20) / (perLayerBytes * 8));
    expect(swa.tokens).toBe(expected);
  });

  // A cohere2/llama4/olmo2-style model (real default period 4, confirmed
  // against llama.cpp's own hparam loaders) declares its own
  // slidingWindowPattern instead of relying on the Gemma-3-style hardcoded
  // fallback -- g should be ceil(48/4)=12 global layers, not 8.
  it("uses a declared slidingWindowPattern instead of the hardcoded ~1-in-6 fallback", () => {
    const swa = maxAffordableContext({ ...workedExample, slidingWindow: 1024, slidingWindowPattern: 4 });
    expect(swa.confidence).toBe("rough");
    const usableMib = 8192 * 0.9 - 2969 - 256;
    const perLayerBytes = 4 * (128 * 2 + 128 * 2);
    const localFixedCostMib = ((48 - 12) * 1024 * perLayerBytes) / 2 ** 20;
    const globalBudgetMib = Math.max(0, usableMib - localFixedCostMib);
    const expected = Math.floor((globalBudgetMib * 2 ** 20) / (perLayerBytes * 12));
    expect(swa.tokens).toBe(expected);
    // Locks in that `g` (not just the fallback constant) actually changes the
    // result -- the exact direction/magnitude of that change is this
    // pre-existing formula's own arithmetic (see the "raw" line above; not
    // something this fix's scope re-derives), so this only asserts it moved,
    // not which way.
    const fallback = maxAffordableContext({ ...workedExample, slidingWindow: 1024 });
    expect(swa.tokens).not.toBe(fallback.tokens);
  });

  // Gemma3n/Gemma4-style KV-cache reuse: sharedKvLayers trailing blocks carry
  // no independent KV cache at all, so they must fall out of BOTH the global
  // and local layer counts entirely -- never priced as either.
  it("excludes sharedKvLayers from the SWA budget entirely, shrinking need (growing affordable tokens)", () => {
    const noShared = maxAffordableContext({ ...workedExample, slidingWindow: 1024 });
    const shared = maxAffordableContext({ ...workedExample, slidingWindow: 1024, sharedKvLayers: 6 });
    // Fewer real KV-allocating layers for the same usable budget -> strictly
    // more affordable context, never less (a shared layer can only reduce
    // need, never increase it).
    expect(shared.tokens).toBeGreaterThan(noShared.tokens);
    // Hand-verified: sharedKvLayers=6 removes exactly 1 global layer
    // (swaGlobalLayersInSuffix(6,6)=1) and 5 local layers from the 48-layer
    // suffix, leaving g=7 global / 40 local of the 42 effective layers.
    const usableMib = 8192 * 0.9 - 2969 - 256;
    const perLayerBytes = 4 * (128 * 2 + 128 * 2);
    const localFixedCostMib = ((42 - 7) * 1024 * perLayerBytes) / 2 ** 20;
    const globalBudgetMib = Math.max(0, usableMib - localFixedCostMib);
    const expected = Math.floor((globalBudgetMib * 2 ** 20) / (perLayerBytes * 7));
    expect(shared.tokens).toBe(expected);
  });

  // sharedKvLayers=0/undefined must be a complete no-op -- every existing SWA
  // caller (no such field yet) gets byte-identical numbers.
  it("sharedKvLayers 0 or undefined reproduce the exact same SWA result", () => {
    const base = maxAffordableContext({ ...workedExample, slidingWindow: 1024 });
    const zero = maxAffordableContext({ ...workedExample, slidingWindow: 1024, sharedKvLayers: 0 });
    expect(zero).toEqual(base);
  });

  it("is advisory-only arithmetic: never negative", () => {
    const est = maxAffordableContext({ ...workedExample, weightsMib: 8000 });
    expect(est.tokens).toBeGreaterThanOrEqual(0);
  });
});

describe("KV forward helpers (the same formula M1 inverts)", () => {
  const geometry = {
    nLayer: 48,
    nHeadKv: 4,
    headDimK: 128,
    headDimV: 128,
    cacheTypeK: "f16",
    cacheTypeV: "f16",
  } as const;

  it("prices f16 at 96 KiB/token and q8_0 at 51 KiB/token, exactly as the worked example states", () => {
    expect(kvBytesPerToken(geometry)).toEqual({ bytes: 98_304, confidence: "good" });
    expect(kvBytesPerToken({ ...geometry, cacheTypeK: "q8_0", cacheTypeV: "q8_0" }).bytes).toBe(52_224);
  });

  it("round-trips against maxAffordableContext: the affordable count costs the usable budget", () => {
    const est = maxAffordableContext({
      totalMib: 8192,
      weightsMib: 2969,
      scratchMib: 256,
      activationsHeadroomFrac: 0.1,
      ...geometry,
    });
    const kvMib = estimateKvCacheMib({ ...geometry, tokens: est.tokens })!;
    const usable = 8192 * 0.9 - 2969 - 256;
    expect(kvMib).toBeLessThanOrEqual(usable);
    expect(kvMib).toBeGreaterThan(usable - 1);
  });

  it("derives a placement's resident weight share from a measured peak, and never invents one", () => {
    const kvMib = estimateKvCacheMib({ ...geometry, tokens: 4096 })!;
    const weights = residentWeightsMibFromPeak({
      ...geometry,
      vramPeakMib: Math.round(3600 + kvMib + 256),
      contextTokens: 4096,
    });
    expect(weights).toBe(3600);
    expect(residentWeightsMibFromPeak({ ...geometry, vramPeakMib: null, contextTokens: 4096 })).toBeNull();
    expect(
      residentWeightsMibFromPeak({ ...geometry, cacheTypeK: "nonsense", vramPeakMib: 5000, contextTokens: 4096 })
    ).toBeNull();
  });

  it("splits KV across parallel slots in the forward direction too", () => {
    const solo = estimateKvCacheMib({ ...geometry, tokens: 4096 })!;
    expect(estimateKvCacheMib({ ...geometry, tokens: 4096, parallelSlots: 4 })).toBeCloseTo(solo * 4, 6);
  });
});
