import { describe, expect, it } from "vitest";
import { contextUsed, ppSpeed, speedCurve, summarizeUsage, tgSpeed, type DaemonUsageRow } from "./daemonUsage.js";

let seq = 0;
function row(over: Partial<DaemonUsageRow> = {}): DaemonUsageRow {
  return {
    id: String(seq++),
    at: 1000 + seq,
    model_id: "m",
    load_hash: "h",
    endpoint: "chat",
    stream: false,
    status: 200,
    cancelled: false,
    processed_tokens: 100,
    cached_tokens: 0,
    generated_tokens: 50,
    prompt_ms: 100,
    predicted_ms: 500,
    total_ms: 620,
    concurrency: 1,
    watched: false,
    ...over,
  };
}

describe("speed readings", () => {
  it("ignores prompts too short to measure", () => {
    expect(ppSpeed(row({ processed_tokens: 6, prompt_ms: 177 }))).toBeNull();
    expect(ppSpeed(row({ processed_tokens: 831, prompt_ms: 1000 }))).toBe(831);
    expect(tgSpeed(row({ generated_tokens: 3 }))).toBeNull();
    expect(tgSpeed(row({ generated_tokens: 100, predicted_ms: 1000 }))).toBe(100);
  });

  it("has no speed where the response had no timings", () => {
    const untimed = row({ endpoint: "messages", prompt_ms: null, predicted_ms: null });
    expect(ppSpeed(untimed)).toBeNull();
    expect(tgSpeed(untimed)).toBeNull();
  });

  it("counts cached tokens as context used", () => {
    expect(contextUsed(row({ processed_tokens: 10, cached_tokens: 90 }))).toBe(100);
    expect(contextUsed(row({ processed_tokens: null, cached_tokens: null }))).toBeNull();
  });
});

describe("speedCurve", () => {
  it("takes the median per power-of-two bucket, in order", () => {
    const curve = speedCurve([
      row({ processed_tokens: 1000, cached_tokens: 0, prompt_ms: 1000 }),
      row({ processed_tokens: 1000, cached_tokens: 0, prompt_ms: 2000 }),
      row({ processed_tokens: 1000, cached_tokens: 0, prompt_ms: 4000 }),
      row({ processed_tokens: 100, cached_tokens: 0, prompt_ms: 100 }),
    ]);
    expect(curve.map((p) => p.from)).toEqual([64, 512]);
    expect(curve[1].pp).toBe(500);
    expect(curve[1].ppCount).toBe(3);
  });

  it("leaves out rows with nothing measurable", () => {
    expect(speedCurve([row({ processed_tokens: 2, generated_tokens: 1 })])).toEqual([]);
  });
});

describe("summarizeUsage", () => {
  it("adds up tokens and the cache share", () => {
    const p = summarizeUsage([row({ processed_tokens: 100, cached_tokens: 300, generated_tokens: 50 }), row({ processed_tokens: 100, cached_tokens: 500, generated_tokens: 50 })]);
    expect(p.tokens).toEqual({ processed: 200, cached: 800, generated: 100 });
    expect(p.cacheShare).toBeCloseTo(0.8);
  });

  it("does not count a failed request's tokens as served", () => {
    const p = summarizeUsage([row({ status: 400, processed_tokens: null, cached_tokens: null, generated_tokens: null, prompt_ms: null, predicted_ms: null })]);
    expect(p.requests).toBe(1);
    expect(p.errors).toBe(1);
    expect(p.tokens.processed).toBe(0);
    expect(p.cacheShare).toBeNull();
    expect(p.observations.some((o) => o.kind === "rejected")).toBe(true);
  });

  it("names no pattern until there is enough traffic", () => {
    const p = summarizeUsage(Array.from({ length: 19 }, () => row()));
    expect(p.shape).toBeNull();
    expect(p.workload).toBeNull();
    expect(p.observations.some((o) => o.kind === "little_data")).toBe(true);
  });

  it.each([
    ["agent_loop", { processed_tokens: 200, cached_tokens: 8000, generated_tokens: 100 }],
    ["agent_loop", { processed_tokens: 60, cached_tokens: 1700, generated_tokens: 40 }],
    ["cold_prompts", { processed_tokens: 8000, cached_tokens: 0, generated_tokens: 100 }],
    ["short_chat", { processed_tokens: 200, cached_tokens: 0, generated_tokens: 100 }],
    ["long_generation", { processed_tokens: 200, cached_tokens: 0, generated_tokens: 2000 }],
    ["batch", { processed_tokens: 200, cached_tokens: 0, generated_tokens: 100, concurrency: 4 }],
  ] as const)("recognizes %s", (shape, over) => {
    expect(summarizeUsage(Array.from({ length: 30 }, () => row(over))).shape).toBe(shape);
  });

  it("gives the later optimization loop the sizes to test", () => {
    const p = summarizeUsage(Array.from({ length: 30 }, () => row({ processed_tokens: 200, cached_tokens: 8000, generated_tokens: 100 })));
    expect(p.workload).toEqual({ prompt_p50: 8200, prompt_p95: 8200, generated_p50: 100, generated_p95: 100, cache_share: 8000 / 8200, concurrency_p95: 1 });
  });

  it("splits the wait between prefill, generation and everything else", () => {
    const p = summarizeUsage([row({ prompt_ms: 200, predicted_ms: 600, total_ms: 1000 })]);
    expect(p.timeShare).toEqual({ prefill: 0.2, generation: 0.6, other: expect.closeTo(0.2) });
  });

  it("keeps untimed requests out of the time split but counts them", () => {
    const p = summarizeUsage([row({ endpoint: "messages", prompt_ms: null, predicted_ms: null })]);
    expect(p.timeShare).toBeNull();
    expect(p.untimedRequests).toBe(1);
    expect(p.speed).toEqual({ pp: null, tg: null });
  });

  it("counts a request cut off as cancelled, not failed", () => {
    const p = summarizeUsage([row({ status: 499, cancelled: true, processed_tokens: null, cached_tokens: null, generated_tokens: null, prompt_ms: null, predicted_ms: null })]);
    expect(p.errors).toBe(0);
    expect(p.cancelled).toBe(1);
  });

  it("only calls a finished response on a generating endpoint untimed", () => {
    const untimed = { prompt_ms: null, predicted_ms: null };
    expect(summarizeUsage([row({ ...untimed, cancelled: true })]).untimedRequests).toBe(0);
    expect(summarizeUsage([row({ ...untimed, endpoint: "embeddings" })]).untimedRequests).toBe(0);
    expect(summarizeUsage([row({ ...untimed, endpoint: "rerank" })]).untimedRequests).toBe(0);
    expect(summarizeUsage([row({ ...untimed, endpoint: "messages" })]).untimedRequests).toBe(1);
  });

  it("states measured facts about slots and context", () => {
    const many = Array.from({ length: 30 }, () => row());
    const kinds = summarizeUsage(many, { slots: 4, nCtx: 100_000, kvMib: 512 }).observations.map((o) => o.kind);
    expect(kinds).toContain("unused_slots");
    expect(kinds).toContain("context_headroom");
    const tight = summarizeUsage(many, { nCtx: 150 }).observations.map((o) => o.kind);
    expect(tight).toContain("context_tight");
    const cold = Array.from({ length: 30 }, () => row({ processed_tokens: 5000, cached_tokens: 0 }));
    expect(summarizeUsage(cold).observations.map((o) => o.kind)).toContain("cold_prompts");
  });

  it("is empty for no rows", () => {
    const p = summarizeUsage([]);
    expect(p.requests).toBe(0);
    expect(p.firstAt).toBeNull();
    expect(p.observations).toEqual([]);
  });
});
