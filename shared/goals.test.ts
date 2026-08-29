import { describe, expect, it } from "vitest";
import {
  normalizeGoals,
  defaultGoals,
  goalsEqualDefaults,
  WORKLOAD_WEIGHTS,
  KV_PRESETS,
  KV_PRESET_PAIRS,
  kvPresetPairs,
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
      kv_preset: "comprehensive",
    });
    expect(g).toEqual({ goal: "max_context", target_ctx: 65536, speed_floor_frac: 0.5, workload: "chat", kv_preset: "comprehensive" });
  });

  it("coerces garbage to defaults rather than fabricating answers", () => {
    const g = normalizeGoals({ goal: "turbo", workload: "yolo", target_ctx: -5 });
    expect(g?.goal).toBe("balanced");
    expect(g?.workload).toBe("even");
    expect(g?.target_ctx).toBeNull();
    expect(g?.kv_preset).toBe("extended");
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

describe("KV cache presets", () => {
  it("defaults to extended", () => {
    expect(defaultGoals().kv_preset).toBe("extended");
    expect(kvPresetPairs(undefined)).toEqual(KV_PRESET_PAIRS.extended);
  });

  it("grows strictly with tier: compact (3) < basic (4) < extended (7) < comprehensive (12)", () => {
    expect(KV_PRESET_PAIRS.compact.length).toBe(3);
    expect(KV_PRESET_PAIRS.basic.length).toBe(4);
    expect(KV_PRESET_PAIRS.extended.length).toBe(7);
    expect(KV_PRESET_PAIRS.comprehensive.length).toBe(12);
  });

  it("is strictly nested: each tier is a superset of the one before it", () => {
    const tiers: (readonly [string, string][])[] = [
      KV_PRESET_PAIRS.compact as readonly [string, string][],
      KV_PRESET_PAIRS.basic as readonly [string, string][],
      KV_PRESET_PAIRS.extended as readonly [string, string][],
      KV_PRESET_PAIRS.comprehensive as readonly [string, string][],
    ];
    for (let i = 1; i < tiers.length; i++) {
      const smaller = new Set(tiers[i - 1].map(([k, v]) => `${k}/${v}`));
      const bigger = new Set(tiers[i].map(([k, v]) => `${k}/${v}`));
      for (const pair of smaller) expect(bigger.has(pair)).toBe(true);
      // Strict, not equal -- every tier actually adds something new.
      expect(bigger.size).toBeGreaterThan(smaller.size);
    }
  });

  it("every tier includes the f16/f16 baseline pair", () => {
    for (const preset of KV_PRESETS) {
      expect(KV_PRESET_PAIRS[preset].some(([k, v]) => k === "f16" && v === "f16")).toBe(true);
    }
  });

  it("has no duplicate pairs within a tier", () => {
    for (const preset of KV_PRESETS) {
      const seen = new Set(KV_PRESET_PAIRS[preset].map(([k, v]) => `${k}/${v}`));
      expect(seen.size).toBe(KV_PRESET_PAIRS[preset].length);
    }
  });
});
