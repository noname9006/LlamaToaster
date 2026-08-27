import { describe, expect, it } from "vitest";
import {
  normalizeGoals,
  defaultGoals,
  goalsEqualDefaults,
  WORKLOAD_WEIGHTS,
  recommendedKvGrid,
  pruneCacheTypes,
  ensureUnquantizedPairSurvives,
  pairAllowedUnderTolerance,
} from "./goals.js";

describe("goals normalization (M2)", () => {
  it("accepts every default = exactly the previous behavior (skippable)", () => {
    expect(goalsEqualDefaults(undefined)).toBe(true);
    expect(goalsEqualDefaults(defaultGoals())).toBe(true);
    // A skipped questionnaire produces no goals block at all -- the trigger
    // payload stays byte-identical to legacy.
    expect(normalizeGoals(undefined)).toBeUndefined();
  });

  it("round-trips stated answers verbatim", () => {
    const g = normalizeGoals({
      goal: "max_context",
      target_ctx: 65536,
      speed_floor_frac: 0.5,
      workload: "chat",
      kv_tolerance: "q8_0_ok",
    });
    expect(g).toEqual({ goal: "max_context", target_ctx: 65536, speed_floor_frac: 0.5, workload: "chat", kv_tolerance: "q8_0_ok" });
  });

  it("coerces garbage to defaults rather than fabricating answers", () => {
    const g = normalizeGoals({ goal: "turbo", workload: "yolo", target_ctx: -5 });
    expect(g?.goal).toBe("balanced");
    expect(g?.workload).toBe("even");
    expect(g?.target_ctx).toBeNull();
  });

  it("treats 'don't know' targets as target_unverified (null)", () => {
    expect(normalizeGoals({ target_ctx: null })?.target_ctx).toBeNull();
    expect(normalizeGoals({ target_ctx: "unverified" })?.target_ctx).toBeNull();
  });

  it("sets (wPP, wTG) per workload shape", () => {
    expect(WORKLOAD_WEIGHTS.chat).toEqual({ wPP: 0.25, wTG: 0.75 });
    expect(WORKLOAD_WEIGHTS.docs).toEqual({ wPP: 0.7, wTG: 0.3 });
    expect(WORKLOAD_WEIGHTS.even).toEqual({ wPP: 0.5, wTG: 0.5 });
  });
});

describe("KV tolerance pruning (M4)", () => {
  it("prunes exactly the forbidden pairs from the recommended grid", () => {
    // q8_0 ok drops q8_0/q4_0.
    expect(recommendedKvGrid("q8_0_ok")).toEqual([
      ["f16", "f16"],
      ["f16", "q8_0"],
      ["q8_0", "q8_0"],
    ]);
    // f16 only leaves f16/f16 alone.
    expect(recommendedKvGrid("f16_only")).toEqual([["f16", "f16"]]);
    // q4_0 ok keeps everything.
    expect(recommendedKvGrid("q4_0_ok").length).toBe(4);
  });

  it("prunes quantized sides but retains other unquantized pairs in expert mode (tolerance speaks to quantization, not width)", () => {
    const pruned = pruneCacheTypes(["f32", "bf16", "f16", "q8_0", "q4_0"], "q8_0_ok");
    expect(pruned).toEqual(["f32", "bf16", "f16", "q8_0"]);
  });

  it("always leaves at least one unquantized pair surviving", () => {
    for (const tolerance of ["q4_0_ok", "q8_0_ok", "f16_only"] as const) {
      const grid = recommendedKvGrid(tolerance);
      expect(grid.length).toBeGreaterThan(0);
      expect(grid.some(([k, v]) => k === v && ["f32", "bf16", "f16"].includes(k))).toBe(true);
    }
  });

  it("keeps one unquantized pair even when a caller pruned them all away", () => {
    const result = ensureUnquantizedPairSurvives(["q8_0"], ["q8_0"], [["f16", "f16"]], false);
    expect(result.cache_type_k).toContain("f16");
    expect(result.cache_type_v).toContain("f16");
  });

  it("prunes to empty, never back to the forbidden originals, when an axis is entirely quantized types the tolerance rejects", () => {
    // Every value here is quantized and below f16_only's infinite-bit floor
    // -- pruning must never silently let a forbidden quantized type back in
    // just because nothing survived.
    expect(pruneCacheTypes(["q4_0", "q4_1"], "f16_only")).toEqual([]);
    // Composed with ensureUnquantizedPairSurvives, the empty axis still ends
    // up with a real fallback pair, never the forbidden values.
    const result = ensureUnquantizedPairSurvives([], [], [["f16", "f16"]], true);
    expect(result.cache_type_k).toEqual(["f16"]);
    expect(result.cache_type_v).toEqual(["f16"]);
  });

  it("checks both sides of a pair against the tolerance", () => {
    expect(pairAllowedUnderTolerance("q8_0", "q4_0", "q8_0_ok")).toBe(false);
    expect(pairAllowedUnderTolerance("q8_0", "q8_0", "q8_0_ok")).toBe(true);
    expect(pairAllowedUnderTolerance("f16", "q4_0", "f16_only")).toBe(false);
    expect(pairAllowedUnderTolerance("f32", "bf16", "f16_only")).toBe(true);
  });
});
