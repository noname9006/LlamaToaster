import { describe, expect, it } from "vitest";
import { blockCaches, simulateFit, simulateFitAt } from "./fitSimulate.js";
import type { KvLayout, TensorLayerBreakdown } from "./vramEstimate.js";

const MiB = 1048576;

// 8 blocks of 200 MiB, a 400 MiB untied output, a 300 MiB embedding; every
// 4th block full attention (kv 4 KiB/token), the rest recurrent.
const dense: TensorLayerBreakdown = {
  dense: new Array(8).fill(200 * MiB),
  moe: new Array(8).fill(0),
  embed: 300 * MiB,
  output: 400 * MiB,
  other: 0,
  max_dense: 50 * MiB,
  max_embed: 300 * MiB,
  max_output: 400 * MiB,
};
const layout: KvLayout = { full_k: 2 * 1024, full_v: 2 * 1024, swa_k: 0, swa_v: 0, window: 0, state_bytes: 8 * MiB, vocab: 100_000, kinds: "rrrfrrrf", full_max: 2048 };
const base = { breakdown: dense, layout, nLayer: 8 };

describe("blockCaches", () => {
  it("puts full-context KV on the full-attention blocks only", () => {
    const c = blockCaches(layout, 8, 4096);
    expect(c.full).toEqual([false, false, false, true, false, false, false, true]);
    expect(c.bytes[3]).toBe(1024 * 2 * 2 * 4096);
    expect(c.bytes[0]).toBe((8 * MiB) / 6);
  });

  it("spreads evenly on a row read before kinds existed", () => {
    const { kinds: _k, ...old } = layout;
    const c = blockCaches(old, 8, 4096);
    expect(new Set(c.bytes).size).toBe(1);
  });
});

describe("simulateFitAt -- llama.cpp's placement", () => {
  it("-ngl 1 is the output layer alone", () => {
    const p = simulateFitAt({ ...base, ctx: 4096 }, { output: true, blocks: 0, movedBlocks: 0 })!;
    expect(p.layers_gpu).toBe(1);
    // output weight + the larger of logits (100k x 512 x 4 B + 12) and the
    // attention stage (4 MiB mask + scratch capped at half the logits).
    expect(p.dev_used_mib).toBe(Math.round(400 + 195.3125 + 12));
  });

  it("a full-attention block on the CPU grows the compute buffer with context", () => {
    const all = simulateFitAt({ ...base, ctx: 262144 }, { output: true, blocks: 8, movedBlocks: 0 })!;
    const part = simulateFitAt({ ...base, ctx: 262144 }, { output: true, blocks: 4, movedBlocks: 0 })!;
    expect(part.compute_mib).toBeGreaterThan(all.compute_mib);
    // (4 KiB KV + 1 KiB mask) x 256k + scratch
    expect(part.compute_mib).toBe(1280 + 64);
  });

  it("-ngl 0 copies the output weight into the compute buffer", () => {
    const p = simulateFitAt({ ...base, ctx: 4096 }, { output: false, blocks: 0, movedBlocks: 0 })!;
    expect(p.compute_mib).toBe(Math.round(195.3125 + 12 + 400));
    expect(p.verdict).toBe("cpu_only");
  });
});

describe("simulateFit -- the search", () => {
  it("full offload when everything fits", () => {
    expect(simulateFit({ ...base, ctx: 4096, freeMib: 8000, marginMib: 1024 })!.verdict).toBe("full");
  });

  it("drops blocks from the front, output kept", () => {
    const p = simulateFit({ ...base, ctx: 4096, freeMib: 2048, marginMib: 512 })!;
    expect(p.verdict).toBe("partial");
    expect(p.dev_used_mib).toBeLessThanOrEqual(2048 - 512);
    const one = simulateFit({ ...base, ctx: 4096, freeMib: 2048 + 200, marginMib: 512 })!;
    expect(one.layers_gpu).toBe(p.layers_gpu + 1);
  });

  it("MoE moves experts from the back before dropping any layer", () => {
    const moe: TensorLayerBreakdown = { ...dense, dense: new Array(8).fill(50 * MiB), moe: new Array(8).fill(400 * MiB), max_moe: 130 * MiB };
    const p = simulateFit({ breakdown: moe, layout, nLayer: 8, ctx: 4096, freeMib: 3000, marginMib: 512 })!;
    expect(p.layers_gpu).toBe(9);
    expect(p.overflow_layers).toBeGreaterThan(0);
    expect(p.dev_used_mib).toBeLessThanOrEqual(3000 - 512);
  });
});

describe("models without a per-token cache", () => {
  it("embedding / encoder / classifier: no KV, flat small compute at any context", () => {
    const enc = { ...layout, no_cache: true };
    const short = simulateFitAt({ ...base, layout: enc, ctx: 4096 }, { output: true, blocks: 8, movedBlocks: 0 })!;
    const long = simulateFitAt({ ...base, layout: enc, ctx: 131072 }, { output: true, blocks: 8, movedBlocks: 0 })!;
    expect(short.kv_mib).toBe(0);
    expect(long.dev_used_mib).toBe(short.dev_used_mib);
    expect(long.compute_mib).toBe(64);
  });
});
