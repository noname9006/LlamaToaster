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
    // The solved-global formula: tokens = max(0, raw - (48-8)*1024)/8.
    const usableMib = 8192 * 0.9 - 2969 - 256;
    const raw = (usableMib * 2 ** 20) / (48 * 4 * (128 * 2 + 128 * 2));
    const expected = Math.floor(Math.max(0, raw - (48 - 8) * 1024) / 8);
    expect(swa.tokens).toBe(expected);
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
