import { describe, expect, it } from "vitest";
import {
  LEGAL_ENGINE_PAIRS,
  isLegalEnginePair,
  isKnownCacheType,
  CACHE_TYPE_VALUES,
  DEPTH_RULE_REJECTION_MESSAGE,
  engineFromItem,
  pairFromItem,
  specTypeFor,
} from "./engineSpec.js";

describe("engine/spec legality (§0.2)", () => {
  it("enumerates exactly the four legal pairs", () => {
    expect(LEGAL_ENGINE_PAIRS).toEqual([
      { engine: "bench", spec: "off" },
      { engine: "server", spec: "off" },
      { engine: "server", spec: "mtp" },
      { engine: "server", spec: "draft" },
    ]);
  });

  it("never derives a bench/mtp pair -- llama-bench has no speculation", () => {
    for (const pair of LEGAL_ENGINE_PAIRS) {
      if (pair.engine === "bench") expect(pair.spec).toBe("off");
    }
    expect(isLegalEnginePair({ engine: "bench", spec: "mtp" })).toBe(false);
    expect(isLegalEnginePair({ engine: "bench", spec: "draft" })).toBe(false);
  });

  it("accepts every listed pair and rejects unknown ones", () => {
    for (const pair of LEGAL_ENGINE_PAIRS) expect(isLegalEnginePair(pair)).toBe(true);
    expect(isLegalEnginePair({ engine: "server" as never, spec: "turbo" as never })).toBe(false);
  });
});

describe("cache-type closed set (§0.2)", () => {
  it("is the nine-value set", () => {
    expect(CACHE_TYPE_VALUES).toEqual(["f32", "bf16", "f16", "q8_0", "q5_0", "q5_1", "q4_1", "q4_0", "iq4_nl"]);
  });

  it("rejects values outside the set", () => {
    for (const v of CACHE_TYPE_VALUES) expect(isKnownCacheType(v)).toBe(true);
    expect(isKnownCacheType("q6_K")).toBe(false);
    expect(isKnownCacheType("")).toBe(false);
    expect(isKnownCacheType(42)).toBe(false);
  });
});

describe("depth rule copy (§0.2)", () => {
  it("names the fix in the rejection message", () => {
    expect(DEPTH_RULE_REJECTION_MESSAGE).toContain("llama-bench");
    expect(DEPTH_RULE_REJECTION_MESSAGE).toContain("n_depth to 0");
  });
});

describe("item → engine mapping", () => {
  it("routes mtp:on items to the server engine and everything else to llama-bench", () => {
    expect(engineFromItem({ mtp: "on" })).toBe("server");
    expect(engineFromItem({ mtp: "off" })).toBe("bench");
    expect(pairFromItem({ mtp: "on" })).toEqual({ engine: "server", spec: "mtp" });
    expect(pairFromItem({ mtp: "off" })).toEqual({ engine: "bench", spec: "off" });
  });

  it("maps spec modes to llama-server flag values", () => {
    expect(specTypeFor({ engine: "server", spec: "mtp" })).toBe("draft-mtp");
    expect(specTypeFor({ engine: "server", spec: "draft" })).toBe("draft-simple");
    expect(specTypeFor({ engine: "server", spec: "off" })).toBeNull();
    expect(specTypeFor({ engine: "bench", spec: "off" })).toBeNull();
  });
});
