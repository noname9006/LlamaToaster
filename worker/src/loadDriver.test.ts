import { describe, expect, it } from "vitest";
import {
  buildServerArgs,
  contextSizeForSlots,
  DEFAULT_KNEE_SLOTS,
  generationRate,
  prefillRate,
  percentile,
  planConcurrentBatch,
  planCurvePoint,
  summarizeStreams,
  type StreamSample,
} from "./loadDriver.js";
import { buildPromptTokens } from "./fillerPrompt.js";

const item = {
  threads: 8,
  n_gpu_layers: 99,
  batch_size: 2048,
  ubatch_size: 512,
  cache_type_k: "f16",
  cache_type_v: "f16",
  flash_attn: "on",
  n_cpu_moe: 0,
  n_prompt: 8192,
  n_gen: 512,
};

describe("slot accounting (N1/N5)", () => {
  it("sizes -c as slots x (n_depth + n_prompt + n_gen + margin)", () => {
    expect(contextSizeForSlots({ slots: 4, nPrompt: 8192, nGen: 512 })).toBe(4 * (8192 + 512 + 256));
  });

  it("equals the existing sizing at parallel 1", () => {
    expect(contextSizeForSlots({ slots: 1, nPrompt: 8192, nGen: 512 })).toBe(8192 + 512 + 256);
  });

  it("counts depth toward per-slot demand", () => {
    expect(contextSizeForSlots({ slots: 1, nDepth: 16_384, nPrompt: 512, nGen: 128 })).toBe(
      16_384 + 512 + 128 + 256
    );
  });
});

describe("server arguments for the spec-off engine pair", () => {
  it("never passes --spec-type: speculation is a different engine pair (§0.2)", () => {
    const args = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 1 });
    expect(args).not.toContain("--spec-type");
    expect(args).not.toContain("--model-draft");
  });

  it("passes --parallel for the slot count and sizes -c for all of them", () => {
    const args = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 4 });
    expect(args[args.indexOf("--parallel") + 1]).toBe("4");
    expect(args[args.indexOf("-c") + 1]).toBe(String(4 * (8192 + 512 + 256)));
  });

  it("only passes --no-context-shift when the binary probe says it exists (§0.7)", () => {
    const without = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 1 });
    expect(without).not.toContain("--no-context-shift");
    const with_ = buildServerArgs({
      modelPath: "m.gguf",
      port: 8080,
      item,
      slots: 1,
      supportsNoContextShift: true,
    });
    expect(with_).toContain("--no-context-shift");
  });

  it("only passes --fit off when the binary probe says --fit exists (§0.7)", () => {
    const without = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 1 });
    expect(without).not.toContain("--fit");
    const with_ = buildServerArgs({
      modelPath: "m.gguf",
      port: 8080,
      item,
      slots: 1,
      supportsFit: true,
    });
    expect(with_[with_.indexOf("--fit") + 1]).toBe("off");
  });

  it("only passes --mlock when the binary probe says it exists (§0.7)", () => {
    const without = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 1 });
    expect(without).not.toContain("--mlock");
    const with_ = buildServerArgs({ modelPath: "m.gguf", port: 8080, item, slots: 1, supportsMlock: true });
    expect(with_).toContain("--mlock");
  });
});

describe("stream summaries (raw samples, never a server aggregate)", () => {
  const sample = (over: Partial<StreamSample>): StreamSample => ({
    ttftMs: 107_000,
    e2eMs: 120_000,
    tokensPredicted: 512,
    promptN: 8192,
    promptMs: 106_000,
    predictedN: 512,
    predictedMs: 13_000,
    slot: 0,
    ...over,
  });

  const samples: StreamSample[] = [
    sample({}),
    sample({ ttftMs: 108_000, e2eMs: 121_000, promptN: 0, promptMs: null }),
    sample({ ttftMs: 250_000, e2eMs: 270_000, promptN: 0, promptMs: null }),
  ];

  it("computes nearest-rank p50/p95 over the raw TTFT samples", () => {
    const summary = summarizeStreams(samples);
    expect(summary.ttftP50Ms).toBe(108_000);
    expect(summary.ttftP95Ms).toBe(250_000);
    expect(summary.ttftN).toBe(3);
  });

  it("reports the sample count so a single-shot cold point is never rendered as a p50", () => {
    expect(summarizeStreams([samples[0]]).ttftN).toBe(1);
  });

  it("sums per-stream rates into the aggregate -- the streams ran simultaneously", () => {
    const batch: StreamSample[] = [0, 1, 2, 3].map((slot) => sample({
      ttftMs: 1_000,
      e2eMs: 11_000,
      tokensPredicted: 250,
      predictedN: 250,
      predictedMs: 10_000,
      promptMs: 900,
      slot,
    }));
    const summary = summarizeStreams(batch);
    expect(summary.perStreamTps).toBeCloseTo(25, 6);
    expect(summary.aggregateTps).toBeCloseTo(100, 6);
  });

  it("handles an empty batch without inventing numbers", () => {
    expect(summarizeStreams([])).toMatchObject({ ttftP50Ms: null, aggregateTps: null, ttftN: 0 });
  });

  it("percentile is nearest-rank", () => {
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2);
    expect(percentile([1, 2, 3, 4], 0.95)).toBe(4);
    expect(percentile([], 0.5)).toBeNull();
  });
});

describe("N1 choreography", () => {
  // v7: no request classes to interleave any more. The warm repeats this
  // replaced depended on a prefix-cache hit that production never actually
  // got -- and the code could only detect the miss after the fact, while
  // still publishing a generation rate measured over a window that held no
  // generation.
  it("issues one cache-free request per repeat, all identical in shape", () => {
    const plan = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 5 });
    expect(plan).toHaveLength(5);
    expect(plan.every((s) => s.requestClass === "cold_timed")).toBe(true);
    expect(plan.every((s) => s.cachePrompt === false)).toBe(true);
    expect(plan.every((s) => s.promptTokens === 8192 && s.nPredict === 512)).toBe(true);
    expect(plan.filter((s) => s.countsTowardStatistics)).toHaveLength(5);
  });

  it("gives every repeat its own nonce, so none can reuse another's prefix", () => {
    const plan = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 4 });
    expect(new Set(plan.map((s) => s.nonce)).size).toBe(4);
  });

  // Every repeat generates: a curve point that asked for zero tokens is what
  // produced the five-hour run of unusable tg rows (the server now refuses
  // that payload, and the worker refuses the job).
  it("asks for the point's full generation on every repeat, never one token", () => {
    const plan = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 3 });
    expect(plan.every((s) => s.nPredict === 512)).toBe(true);
  });

  it("still emits one measured request at repeats = 1", () => {
    const plan = planCurvePoint({ promptTokens: 4096, nGen: 128, repeats: 1 });
    expect(plan.map((s) => s.requestClass)).toEqual(["cold_timed"]);
  });
});

describe("one request's two rates", () => {
  const base: StreamSample = {
    ttftMs: 2_000,
    e2eMs: 6_000,
    tokensPredicted: 128,
    promptN: 8_192,
    promptMs: 2_000,
    predictedN: 128,
    predictedMs: 4_000,
    slot: 0,
  };

  it("reads each rate from that request's own timings", () => {
    expect(prefillRate(base)!.value).toBeCloseTo(4_096, 6);
    expect(generationRate(base)!.value).toBeCloseTo(32, 6);
    expect(generationRate(base)!.suspect).toBe(false);
  });

  // The production failure in one assertion: one token, a 1 ms wall-clock
  // window after the first chunk, and a decode timer that says 12 tok/s. The
  // old formula divided the token count by that window and published 1000.
  it("takes a one-token generation from the decode timer, not the teardown window", () => {
    const reading = generationRate({
      ...base,
      ttftMs: 136_000,
      e2eMs: 136_001,
      tokensPredicted: 1,
      predictedN: 1,
      predictedMs: 1000 / 12,
    })!;
    expect(reading.value).toBeCloseTo(12, 6);
    expect(reading.suspect).toBe(false);
  });

  it("flags decode that reads faster than the same request's prefill", () => {
    // 8192 tokens in 546 s = 15 tok/s prefill; 128 tokens in 1 s = 128 tok/s
    // decode, corroborated by the wall clock, and still not believable.
    const reading = generationRate({
      ...base,
      promptMs: 546_000,
      ttftMs: 546_000,
      e2eMs: 547_000,
      predictedMs: 1_000,
    })!;
    expect(reading.suspect).toBe(true);
    expect(reading.reason).toContain("prefill");
  });

  // A 16-token prompt times first-request setup rather than prefill, so its
  // pp is real but useless as a bound -- the cross-check must stay out of the
  // probe's way.
  it("does not hold a tiny prompt's prefill rate against a generation reading", () => {
    const reading = generationRate({
      ...base,
      promptN: 16,
      promptMs: 12_000, // 1.3 tok/s -- setup cost, not prefill
      ttftMs: 12_000,
      e2eMs: 12_500,
      tokensPredicted: 16,
      predictedN: 16,
      predictedMs: 500, // 32 tok/s
    })!;
    expect(reading.suspect).toBe(false);
  });

  it("flags a decode timer the wall clock cannot corroborate", () => {
    // 128 tokens in 100 ms (1280 tok/s) against a wall clock that saw 4 s
    // pass after the first chunk -- under the ceiling, and still impossible.
    const reading = generationRate({ ...base, predictedMs: 100 })!;
    expect(reading.suspect).toBe(true);
    expect(reading.reason).toContain("wall clock");
  });

  it("returns nothing at all when the server reported no decode timing", () => {
    expect(generationRate({ ...base, predictedN: null, predictedMs: null })).toBeNull();
    expect(prefillRate({ ...base, promptMs: null })).toBeNull();
    expect(prefillRate({ ...base, promptN: 0 })).toBeNull();
  });
});

describe("N5 concurrent batch", () => {
  it("gives every stream a distinct nonce so none can read a warm shared prefix", () => {
    const batch = planConcurrentBatch({ slots: 4, promptTokens: 32_768, nGen: 512 });
    expect(batch).toHaveLength(4);
    expect(new Set(batch.map((s) => s.nonce)).size).toBe(4);
    // Never nonce 0 -- that is the measured curve prompt.
    expect(batch.every((s) => s.nonce !== 0)).toBe(true);
    const prompts = batch.map((s) => buildPromptTokens(8, 0, s.nonce, [[7, 8, 9, 10]]).join(","));
    expect(new Set(prompts).size).toBe(4);
  });

  it("defaults the ladder to {1, 2, 4, 8}", () => {
    expect([...DEFAULT_KNEE_SLOTS]).toEqual([1, 2, 4, 8]);
  });
});
