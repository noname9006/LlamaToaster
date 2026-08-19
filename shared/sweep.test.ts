import { describe, expect, it } from "vitest";
import { expandSweep, deriveTestType } from "./sweep.js";
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

describe("deriveTestType", () => {
  it("maps n_gen=0 to pp, n_prompt=0 to tg, and both nonzero to pg", () => {
    expect(deriveTestType(512, 0)).toBe("pp");
    expect(deriveTestType(0, 128)).toBe("tg");
    expect(deriveTestType(512, 128)).toBe("pg");
  });
});
