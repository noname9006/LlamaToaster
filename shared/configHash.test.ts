import { describe, expect, it } from "vitest";
import { configHash } from "./configHash.js";

const base = {
  n_prompt: 512,
  n_gen: 128,
  n_depth: 0,
  threads: 8,
  n_gpu_layers: 99,
  batch_size: 2048,
  ubatch_size: 512,
  cache_type_k: "f16",
  cache_type_v: "f16",
  flash_attn: "on",
  n_gpu_layers_draft: 0,
  n_cpu_moe: 0,
  engine: "server" as const,
  concurrency: 1,
};

describe("configHash (§0.4)", () => {
  it("is deterministic and hex", () => {
    const a = configHash(base);
    const b = configHash({ ...base });
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{64}$/);
  });

  it("excludes spec fields -- the exclusion is what makes twin joins work", () => {
    const withSpec = { ...base, spec: "mtp", spec_type: "draft-mtp", spec_n_max: 4, spec_n_min: 1 };
    // spec/spec_type/spec_n_max/spec_n_min are not ConfigHashInput axes at
    // all; a hash computed from an object carrying them must be identical.
    expect(configHash(withSpec)).toBe(configHash(base));
    const otherSpec = { ...base, spec_n_max: 999 };
    expect(configHash(withSpec)).toBe(configHash(otherSpec));
  });

  it("changes when any included axis changes", () => {
    for (const [axis, value] of [
      ["n_prompt", 1024],
      ["n_gen", 256],
      ["threads", 4],
      ["n_gpu_layers", 0],
      ["batch_size", 4096],
      ["ubatch_size", 256],
      ["cache_type_k", "q8_0"],
      ["cache_type_v", "q8_0"],
      ["flash_attn", "off"],
      ["engine", "bench" as const],
      ["concurrency", 4],
    ] as const) {
      const mutated = { ...base, [axis]: value };
      expect(configHash(mutated)).not.toBe(configHash(base));
    }
  });

  it("applies explicit defaults for absent axes so adding one never changes an existing hash", () => {
    expect(configHash(base)).toBe(
      configHash({
        n_prompt: 512,
        n_gen: 128,
        threads: 8,
        n_gpu_layers: 99,
        batch_size: 2048,
        ubatch_size: 512,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn: "on",
        n_gpu_layers_draft: 0,
        n_cpu_moe: 0,
        engine: "server",
      })
    );
    // Absent n_depth defaults to 0, absent concurrency to 1.
    expect(configHash({ ...base })).toBe(configHash({ ...base, n_depth: undefined, concurrency: undefined }));
  });
});
