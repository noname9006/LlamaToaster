import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPromptTokens, fetchFillerBlocks, type FillerBlocks } from "./fillerPrompt.js";

// Five blocks of distinct, non-overlapping id ranges: which block a token came
// from is then readable straight off its value, which is what lets these tests
// assert on the register MIX rather than just on counts.
const BLOCKS: FillerBlocks = [
  Array.from({ length: 37 }, (_, i) => 1000 + i),
  Array.from({ length: 53 }, (_, i) => 2000 + i),
  Array.from({ length: 41 }, (_, i) => 3000 + i),
  Array.from({ length: 61 }, (_, i) => 4000 + i),
  Array.from({ length: 29 }, (_, i) => 5000 + i),
];

const blockOf = (token: number) => Math.floor(token / 1000);

function shares(tokens: number[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const t of tokens) counts.set(blockOf(t), (counts.get(blockOf(t)) ?? 0) + 1);
  return counts;
}

describe("filler prompt construction", () => {
  it("produces exactly the requested token count", () => {
    for (const n of [0, 1, 7, 64, 768, 8192]) {
      expect(buildPromptTokens(n, 0, 0, BLOCKS)).toHaveLength(n);
    }
  });

  it("only ever emits ids that came from the blocks", () => {
    const all = new Set(BLOCKS.flat());
    expect(buildPromptTokens(2048, 3, 2, BLOCKS).every((t) => all.has(t))).toBe(true);
  });

  // The register mix has to be identical at every prompt size, or N1's curve
  // conflates depth scaling with a drifting content mix -- and content is worth
  // up to 58% on MoE prefill.
  it("gives every block the same share of the prompt at every size", () => {
    for (const n of [512, 4096, 32_768]) {
      const counts = shares(buildPromptTokens(n, 0, 0, BLOCKS));
      expect(counts.size).toBe(BLOCKS.length);
      const each = [...counts.values()];
      // Equal to within the remainder spread across the leading blocks.
      expect(Math.max(...each) - Math.min(...each)).toBeLessThanOrEqual(1);
      expect(each.reduce((a, b) => a + b, 0)).toBe(n);
    }
  });

  // Prefill computes its expert union per ubatch (512). Contiguous blocks would
  // put one register in each ubatch at large sizes and read ~3% fast.
  it("interleaves finely enough that every ubatch-sized window sees every block", () => {
    const tokens = buildPromptTokens(4096, 0, 0, BLOCKS);
    for (let start = 0; start + 512 <= tokens.length; start += 512) {
      expect(shares(tokens.slice(start, start + 512)).size).toBe(BLOCKS.length);
    }
  });

  it("gives each nonce a distinct sequence, so the prefix cache cannot dedupe streams", () => {
    const a = buildPromptTokens(64, 0, 1, BLOCKS);
    const b = buildPromptTokens(64, 0, 2, BLOCKS);
    expect(a).not.toEqual(b);
    // Distinct from the very first token: a shared prefix is exactly what would
    // let a later stream read artificially fast.
    expect(a[0]).not.toBe(b[0]);
  });

  // N5 spaces nonces as `nonce + repeat * 1000`.
  it("keeps N5's repeat-spaced nonces distinct", () => {
    const prompts = new Set(
      [0, 1000, 2000, 3000].map((nonce) => buildPromptTokens(128, 0, nonce, BLOCKS).join(","))
    );
    expect(prompts.size).toBe(4);
  });

  it("gives each retry offset a distinct sequence, so a retry is not a replay", () => {
    expect(buildPromptTokens(256, 0, 0, BLOCKS)).not.toEqual(buildPromptTokens(256, 137, 0, BLOCKS));
  });

  it("keeps the mix fixed while rotating, so a retry changes order and not composition", () => {
    const plain = shares(buildPromptTokens(1024, 0, 0, BLOCKS));
    const rotated = shares(buildPromptTokens(1024, 137, 3, BLOCKS));
    expect([...rotated.entries()].sort()).toEqual([...plain.entries()].sort());
  });

  it("still fills the prompt when a block tokenized to nothing", () => {
    const withEmpty: FillerBlocks = [BLOCKS[0], [], BLOCKS[2]];
    const tokens = buildPromptTokens(300, 0, 0, withEmpty);
    expect(tokens).toHaveLength(300);
    expect(shares(tokens).size).toBe(2);
  });

  it("refuses to build a prompt with no usable blocks", () => {
    expect(() => buildPromptTokens(64, 0, 0, [])).toThrow(/no usable token blocks/);
    expect(() => buildPromptTokens(64, 0, 0, [[], []])).toThrow(/no usable token blocks/);
  });
});

describe("fetchFillerBlocks", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns one tokenized block per register, asking the server not to add special tokens", async () => {
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        calls.push(String(init.body));
        return Response.json({ tokens: [11, 22, 33] });
      })
    );
    const blocks = await fetchFillerBlocks(8080);
    expect(blocks.length).toBeGreaterThanOrEqual(5);
    expect(blocks.every((b) => b.length === 3)).toBe(true);
    expect(calls.every((b) => JSON.parse(b).add_special === false)).toBe(true);
  });

  // A silent fallback would emit a number measured against different content,
  // and that number can be 70% off. An incomparable row is worse than a failure.
  it("throws rather than degrading when the endpoint is unusable", async () => {
    for (const [stub, pattern] of [
      [
        async () => {
          throw new Error("connect ECONNREFUSED");
        },
        /request failed/,
      ],
      [async () => new Response("nope", { status: 404 }), /returned 404/],
      [async () => Response.json({ tokens: "not an array" }), /returned no tokens/],
      [async () => Response.json({ tokens: [] }), /tokenized to nothing/],
    ] as const) {
      vi.stubGlobal("fetch", vi.fn(stub));
      await expect(fetchFillerBlocks(8080)).rejects.toThrow(pattern);
    }
  });
});
