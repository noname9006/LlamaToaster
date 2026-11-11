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

describe("simulateFit -- MoE boundary layer, as --fit splits it", () => {
  // 8 blocks: 50 MiB attention + router/shared, 400 MiB routed experts.
  // Per block the ffn groups: gate 140 (130 experts + 10 router/shared),
  // up 130, down 135 (130 + 5 shared) -- 405, the 5 left being the shared up
  // and router counted in dense.
  const moe: TensorLayerBreakdown = {
    ...dense,
    dense: new Array(8).fill(50 * MiB),
    moe: new Array(8).fill(400 * MiB),
    max_moe: 135 * MiB,
    ffn_gate: new Array(8).fill(140 * MiB),
    ffn_up: new Array(8).fill(130 * MiB),
    ffn_down: new Array(8).fill(135 * MiB),
  };
  const run = (freeMib: number, b = moe) => simulateFit({ breakdown: b, layout, nLayer: 8, ctx: 4096, freeMib, marginMib: 0 })!;
  const at = (p: Parameters<typeof simulateFitAt>[1], b = moe) => simulateFitAt({ breakdown: b, layout, nLayer: 8, ctx: 4096 }, p)!;

  it("keeps the boundary block's up and gate tensors when both fit (GATE)", () => {
    const moved3 = at({ output: true, blocks: 8, movedBlocks: 3 });
    const gate = at({ output: true, blocks: 8, movedBlocks: 3, boundary: "gate" });
    // GATE puts back the 400 MiB of experts and moves only ffn_down (135).
    expect(gate.dev_used_mib - moved3.dev_used_mib).toBe(400 - 135);
    const p = run(gate.dev_used_mib + 1);
    expect(p.overflow_layers).toBe(3);
    expect(p.dev_used_mib).toBe(gate.dev_used_mib);
  });

  it("falls back to ATTN, below plain expert offload, when UP does not fit", () => {
    const moved3 = at({ output: true, blocks: 8, movedBlocks: 3 });
    const up = at({ output: true, blocks: 8, movedBlocks: 3, boundary: "up" });
    const attn = at({ output: true, blocks: 8, movedBlocks: 3, boundary: "attn" });
    expect(up.dev_used_mib - moved3.dev_used_mib).toBe(400 - 140 - 135);
    // ATTN also moves the router and shared parts: 405 - 400 below moved3.
    expect(moved3.dev_used_mib - attn.dev_used_mib).toBe(5);
    const p = run(moved3.dev_used_mib + 50);
    expect(p.dev_used_mib).toBe(attn.dev_used_mib);
    expect(p.overflow_layers).toBe(3);
  });

  it("never moves a byte-exact part of a block (no movedBytes)", () => {
    const moved3 = at({ output: true, blocks: 8, movedBlocks: 3 });
    const moved2 = at({ output: true, blocks: 8, movedBlocks: 2 });
    // Room for 200 of the next block's 400 MiB experts: fit can't use it
    // byte-exactly; GATE needs 265, UP 125 fits.
    const p = run(moved3.dev_used_mib + 200);
    expect(p.dev_used_mib).toBe(moved3.dev_used_mib + 125);
    expect(p.dev_used_mib).toBeLessThan(moved2.dev_used_mib);
  });

  it("rows without the ffn group sums keep whole blocks only", () => {
    const { ffn_gate: _g, ffn_up: _u, ffn_down: _d, ...old } = moe;
    const moved3 = at({ output: true, blocks: 8, movedBlocks: 3 }, old);
    const p = run(moved3.dev_used_mib + 300, old);
    expect(p.dev_used_mib).toBe(moved3.dev_used_mib);
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
