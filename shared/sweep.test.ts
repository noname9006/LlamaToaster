import { describe, expect, it } from "vitest";
import { expandSweep, deriveTestType, validateDepthRule, validateConcurrencyRule } from "./sweep.js";
import type { SweepConfig } from "./types.js";

const baseSweep: Omit<SweepConfig, "model_id"> = {
  n_prompt: [512],
  n_gen: [128],
  threads: [8],
  n_gpu_layers: [0, 99],
  batch_size: [2048],
  ubatch_size: [512],
  cache_type_k: ["f16"],
  cache_type_v: ["f16"],
  flash_attn: ["on", "off"],
  mtp: ["off"],
  n_gpu_layers_draft: [0],
  n_cpu_moe: [0],
  repeats: 1,
};

describe("expandSweep", () => {
  it("produces the full cross-product for a simple sweep, in fixed field order", () => {
    const items = expandSweep(baseSweep);
    // 2 n_gpu_layers x 2 flash_attn = 4 combos; cache types are fixed f16/f16
    // so isValidCombo never rejects anything here.
    expect(items).toHaveLength(4);
    expect(items[0]).toMatchObject({
      idx: 0,
      n_prompt: 512,
      n_gen: 128,
      n_gpu_layers: 0,
      flash_attn: "on",
    });
    // idx is assigned in emission order and must be contiguous.
    expect(items.map((i) => i.idx)).toEqual([0, 1, 2, 3]);
  });

  it("is a pure function -- calling it twice on identical input yields identical output", () => {
    // The server (trigger-time item pre-creation) and the worker (per-item
    // execution loop) both call expandSweep independently on the same sweep
    // JSON and must agree without any registration round trip -- this is the
    // invariant that guarantees it.
    const a = expandSweep(baseSweep);
    const b = expandSweep(structuredClone(baseSweep));
    expect(b).toEqual(a);
  });

  it("skips invalid flash_attn/cache-type combinations (quantized KV cache requires flash_attn on)", () => {
    const sweep: Omit<SweepConfig, "model_id"> = {
      ...baseSweep,
      cache_type_k: ["f16", "q8_0"],
      cache_type_v: ["f16", "q8_0"],
      flash_attn: ["on", "off"],
      n_gpu_layers: [0],
    };
    const items = expandSweep(sweep);
    for (const item of items) {
      if (item.flash_attn === "off") {
        expect(item.cache_type_k).toBe("f16");
        expect(item.cache_type_v).toBe("f16");
      }
    }
    // 4 cache combos x 2 flash_attn = 8, minus the 3 invalid off-combos
    // (f16/q8_0, q8_0/f16, q8_0/q8_0 all rejected when flash_attn is off).
    expect(items).toHaveLength(5);
  });

  it("returns an empty array for an empty axis", () => {
    expect(expandSweep({ ...baseSweep, n_prompt: [] })).toEqual([]);
  });
});

describe("cache_type_pairs (curated, non-rectangular KV grids)", () => {
  it("iterates exactly the coupled pairs instead of the cross product of cache_type_k x cache_type_v", () => {
    const sweep: Omit<SweepConfig, "model_id"> = {
      ...baseSweep,
      // The cross product of these two axes would be 4 pairs (f16/f16,
      // f16/q8_0, q8_0/f16, q8_0/q8_0) -- cache_type_pairs restricts it to
      // just the 2 curated ones, proving the axis arrays no longer drive
      // expansion once pairs are supplied.
      cache_type_k: ["f16", "q8_0"],
      cache_type_v: ["f16", "q8_0"],
      cache_type_pairs: [
        ["f16", "f16"],
        ["q8_0", "q8_0"],
      ],
      flash_attn: ["on"],
      n_gpu_layers: [0],
    };
    const items = expandSweep(sweep);
    expect(items.map((i) => [i.cache_type_k, i.cache_type_v])).toEqual([
      ["f16", "f16"],
      ["q8_0", "q8_0"],
    ]);
  });

  it("still applies isValidCombo per coupled pair (quantized KV needs flash_attn on)", () => {
    const sweep: Omit<SweepConfig, "model_id"> = {
      ...baseSweep,
      cache_type_k: ["f16", "q8_0"],
      cache_type_v: ["f16", "q8_0"],
      cache_type_pairs: [
        ["f16", "f16"],
        ["q8_0", "q8_0"],
      ],
      flash_attn: ["on", "off"],
      n_gpu_layers: [0],
    };
    const items = expandSweep(sweep);
    // f16/f16 survives both flash_attn values; q8_0/q8_0 only survives "on".
    expect(items).toHaveLength(3);
    for (const item of items) {
      if (item.flash_attn === "off") expect(item.cache_type_k).toBe("f16");
    }
  });

  it("falls back to the cross product when cache_type_pairs is absent or empty (legacy behavior unchanged)", () => {
    const withoutPairs = expandSweep(baseSweep);
    const withEmptyPairs = expandSweep({ ...baseSweep, cache_type_pairs: [] });
    expect(withEmptyPairs).toEqual(withoutPairs);
  });
});

describe("deriveTestType", () => {
  it("maps n_gen=0 to pp, n_prompt=0 to tg, and both nonzero to pg", () => {
    expect(deriveTestType(512, 0)).toBe("pp");
    expect(deriveTestType(0, 128)).toBe("tg");
    expect(deriveTestType(512, 128)).toBe("pg");
  });
});

describe("n_depth axis (§0.2)", () => {
  it("defaults an absent axis to [0], preserving legacy expansion order exactly", () => {
    const withAxis = expandSweep({ ...baseSweep, n_depth: [0] });
    const withoutAxis = expandSweep(baseSweep);
    expect(withoutAxis).toEqual(withAxis);
  });

  it("expands the depth axis innermost of all", () => {
    const items = expandSweep({ ...baseSweep, n_depth: [0, 4096] });
    expect(items).toHaveLength(8);
    expect(items.map((i) => i.n_depth)).toEqual([0, 4096, 0, 4096, 0, 4096, 0, 4096]);
  });

  it("rejects depth > 0 on server-engine items (llama-server has no KV-prefill flag)", () => {
    const items = expandSweep({ ...baseSweep, mtp: ["on"], n_depth: [4096] });
    expect(validateDepthRule(items)).toMatch(/n_depth is only supported on the llama-bench engine/);
    expect(validateDepthRule(expandSweep({ ...baseSweep, mtp: ["off"], n_depth: [4096] }))).toBeNull();
    expect(validateDepthRule(expandSweep(baseSweep))).toBeNull();
  });
});

describe("concurrency axis (N5)", () => {
  it("defaults to [1] for legacy payloads", () => {
    const items = expandSweep(baseSweep);
    for (const item of items) {
      expect(item.concurrency).toBe(1);
      // Legacy payloads keep producing identical items apart from the
      // defaulted new fields.
      expect(item).toMatchObject({ idx: item.idx, concurrency: 1, n_depth: 0 });
    }
  });

  it("rejects slots > 1 on llama-bench items (no request lifecycle there)", () => {
    const items = expandSweep({ ...baseSweep, concurrency: [4] });
    expect(validateConcurrencyRule(items)).toMatch(/only supported on the llama-server engine/);
    expect(validateConcurrencyRule(expandSweep({ ...baseSweep, mtp: ["on"], concurrency: [1, 2, 4, 8] }))).toBeNull();
  });
});
