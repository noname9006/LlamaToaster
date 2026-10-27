import { describe, expect, it } from "vitest";
import { buildKvLayout } from "./gguf.js";
import { estimateKvLayoutMib, estimateRuntimeExtrasMib, isKvLayout, placeWeightPools } from "./vramEstimate.js";

// Expected MiB figures are llama.cpp b11226's own `llama_kv_cache: size =`
// lines for the real files these headers come from (llama-fit-params -c 0 -v).
const base = {
  nextnLayers: 0,
  keyLength: null,
  valueLength: null,
  keyLengthSwa: null,
  valueLengthSwa: null,
  nEmbd: null,
  nHead: null,
  slidingWindow: null,
  swaPatternScalar: null,
  swaPatternArray: null,
  sharedKvLayers: 0,
  fullAttentionInterval: null,
};
const f16 = { cacheTypeK: "f16", cacheTypeV: "f16" };

describe("buildKvLayout", () => {
  it("counts only every 4th layer of a Qwen3.5 hybrid (interval defaulted)", () => {
    const layout = buildKvLayout({ ...base, architecture: "qwen35", nLayer: 32, headCountKv: [4], keyLength: [256], valueLength: [256] })!;
    expect(layout).toEqual({
      full_k: 8 * 4 * 256,
      full_v: 8 * 4 * 256,
      swa_k: 0,
      swa_v: 0,
      window: 0,
      kinds: "rrrf".repeat(8),
      full_max: 2 * 4 * 256,
    });
    expect(estimateKvLayoutMib(layout, { tokens: 262144, ...f16 })).toBe(8192);
  });

  it("honours an explicit full_attention_interval", () => {
    const layout = buildKvLayout({ ...base, architecture: "x", nLayer: 40, headCountKv: [2], keyLength: [256], valueLength: [256], fullAttentionInterval: 4 })!;
    expect(estimateKvLayoutMib(layout, { tokens: 262144, ...f16 })).toBe(5120);
  });

  it("skips zero-head (recurrent) layers in a per-layer array", () => {
    const layout = buildKvLayout({ ...base, architecture: "lfm2", nLayer: 4, headCountKv: [0, 8, 0, 8], keyLength: [64], valueLength: [64] })!;
    expect(layout.full_k).toBe(2 * 8 * 64);
  });

  it("splits Gemma-4 style per-layer SWA flags, swa head dims and shared KV layers", () => {
    // 6 layers, last 2 share KV; flags: swa,swa,global,swa,swa,global
    const layout = buildKvLayout({
      ...base,
      architecture: "gemma4",
      nLayer: 6,
      headCountKv: [8, 8, 2, 8, 8, 2],
      keyLength: [512],
      valueLength: [512],
      keyLengthSwa: 256,
      valueLengthSwa: 256,
      slidingWindow: 1024,
      swaPatternArray: [1, 1, 0, 1, 1, 0],
      sharedKvLayers: 2,
    })!;
    expect(layout).toEqual({
      full_k: 2 * 512,
      full_v: 2 * 512,
      swa_k: 3 * 8 * 256,
      swa_v: 3 * 8 * 256,
      window: 1024,
      kinds: "ssfsnn",
      full_max: 2 * 2 * 512,
    });
    // SWA cache = window + 512 ubatch cells, padded to 256.
    const mib = estimateKvLayoutMib(layout, { tokens: 262144, ...f16 })!;
    expect(mib).toBeCloseTo((2048 * 2 * 262144 + 12288 * 2 * 1536) / 1048576, 6);
  });

  it("uses the scalar SWA period like llama.cpp set_swa_pattern", () => {
    const layout = buildKvLayout({ ...base, architecture: "gemma3", nLayer: 12, headCountKv: [4], keyLength: [128], valueLength: [128], slidingWindow: 512, swaPatternScalar: 6 })!;
    expect(layout.full_k).toBe(2 * 4 * 128); // layers 5 and 11
    expect(layout.swa_k).toBe(10 * 4 * 128);
  });

  it("never sizes the SWA cache above the context", () => {
    const layout = { full_k: 0, full_v: 0, swa_k: 100, swa_v: 100, window: 4096 };
    expect(estimateKvLayoutMib(layout, { tokens: 1000, ...f16 })).toBeCloseTo((400 * 1000) / 1048576, 9);
  });

  it("returns null without head counts", () => {
    expect(buildKvLayout({ ...base, architecture: "llama", nLayer: 4, headCountKv: null })).toBeNull();
  });
});

describe("isKvLayout", () => {
  it("accepts the shape and rejects junk", () => {
    expect(isKvLayout({ full_k: 1, full_v: 1, swa_k: 0, swa_v: 0, window: 0 })).toBe(true);
    expect(isKvLayout({ full_k: "1", full_v: 1, swa_k: 0, swa_v: 0, window: 0 })).toBe(false);
    expect(isKvLayout({ full_k: -1, full_v: 1, swa_k: 0, swa_v: 0, window: 0 })).toBe(false);
    expect(isKvLayout(null)).toBe(false);
  });
});

describe("recurrent state + compute extras", () => {
  it("matches llama.cpp's recurrent size for Qwen3.5-4B (50.25 MiB)", () => {
    const layout = buildKvLayout({
      ...base,
      architecture: "qwen35",
      nLayer: 32,
      headCountKv: [4],
      keyLength: [256],
      valueLength: [256],
      ssmConvKernel: 4,
      ssmInnerSize: 4096,
      ssmStateSize: 128,
      ssmGroupCount: 16,
      vocab: 248320,
    })!;
    expect(layout.state_bytes! / 1048576).toBeCloseTo(50.25, 6);
    // logits 248320 x 512 x f32 = 485 MiB, + 64 scratch + 50.25 state
    expect(estimateRuntimeExtrasMib(layout)).toBeCloseTo(485 + 64 + 50.25, 6);
  });
});

describe("placeWeightPools", () => {
  const MiB = 1048576;
  const blocks = { dense: [100 * MiB, 100 * MiB], moe: [0, 0], output: 1 * MiB, other: 2 * MiB };
  it("keeps the token embedding in RAM for an untied model, even at full offload", () => {
    const b = { ...blocks, embed: 50 * MiB, output: 50 * MiB };
    const { gpuBytes, cpuBytes } = placeWeightPools(b, 2, 3);
    expect(gpuBytes).toBe(200 * MiB + 50 * MiB); // blocks + separate output.weight
    expect(cpuBytes).toBe(50 * MiB + 2 * MiB); // embedding + misc
  });
  it("puts a copy of a tied embedding on both sides once the output layer is offloaded", () => {
    const b = { ...blocks, embed: 50 * MiB };
    const full = placeWeightPools(b, 2, 3);
    expect(full.gpuBytes).toBe(200 * MiB + 1 * MiB + 50 * MiB);
    expect(full.cpuBytes).toBe(50 * MiB + 2 * MiB);
    // ngl == nLayer: output layer still on the CPU, so no GPU copy.
    expect(placeWeightPools(b, 2, 2).gpuBytes).toBe(200 * MiB);
  });
});

describe("buildKvLayout -- llama.cpp's own per-layer and no-cache keys", () => {
  const qwen = { ...base, architecture: "qwen35", nLayer: 8, headCountKv: [2], keyLength: [256], valueLength: [256], fullAttentionInterval: 4 };

  it("explicit recurrent_layers flags override full_attention_interval", () => {
    // Qwen3.5-2B "Longbook2": interval 4 in the header, every layer flagged recurrent.
    const layout = buildKvLayout({ ...qwen, recurrentLayers: new Array(8).fill(1) })!;
    expect(layout.kinds).toBe("rrrrrrrr");
    expect(layout.full_k).toBe(0);
    const mixed = buildKvLayout({ ...qwen, recurrentLayers: [1, 1, 0, 1, 1, 1, 1, 0] })!;
    expect(mixed.kinds).toBe("rrfrrrrf");
  });

  it("flags a model that keeps no cache", () => {
    expect(buildKvLayout({ ...qwen, noCache: true })!.no_cache).toBe(true);
    expect(buildKvLayout(qwen)!.no_cache).toBeUndefined();
  });
});
