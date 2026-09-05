import { describe, expect, it } from "vitest";
import {
  buildServerArgs,
  contextSizeForSlots,
  curveCaveatFlags,
  detectCacheEviction,
  DEFAULT_KNEE_SLOTS,
  percentile,
  planConcurrentBatch,
  planCurvePoint,
  sawContextShift,
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
});

describe("stream summaries (raw samples, never a server aggregate)", () => {
  const samples: StreamSample[] = [
    { ttftMs: 107_000, e2eMs: 120_000, tokensPredicted: 512, promptN: 8192, promptMs: 106_000, slot: 0 },
    { ttftMs: 108_000, e2eMs: 121_000, tokensPredicted: 512, promptN: 0, promptMs: null, slot: 0 },
    { ttftMs: 250_000, e2eMs: 270_000, tokensPredicted: 512, promptN: 0, promptMs: null, slot: 0 },
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
    const batch: StreamSample[] = [0, 1, 2, 3].map((slot) => ({
      ttftMs: 1_000,
      e2eMs: 11_000,
      tokensPredicted: 250,
      promptN: 8192,
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
  it("issues warm/discard, then ONE cold timed prefill, then repeats-1 warm repeats", () => {
    const plan = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 5 });
    // repeats = 5 means five MEASURED requests: the cold prefill is one of
    // them, so four warm repeats follow it.
    expect(plan.map((s) => s.requestClass)).toEqual([
      "warm_discard",
      "cold_timed",
      "warm_repeat",
      "warm_repeat",
      "warm_repeat",
      "warm_repeat",
    ]);
    expect(plan.filter((s) => s.countsTowardStatistics)).toHaveLength(5);
  });

  it("gives the warm/discard request a SHORT NONCE prompt distinct from the measured one", () => {
    const [warm, cold] = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 3 });
    expect(warm.promptTokens).toBeLessThan(cold.promptTokens);
    expect(warm.nonce).not.toBe(cold.nonce);
    expect(warm.cachePrompt).toBe(false);
    expect(warm.countsTowardStatistics).toBe(false);
  });

  it("makes the cold prefill timed DATA: one token, no cache reuse, and it counts", () => {
    const cold = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 3 })[1];
    expect(cold.nPredict).toBe(1);
    expect(cold.cachePrompt).toBe(false);
    expect(cold.countsTowardStatistics).toBe(true);
  });

  it("runs the warm repeats against the cache with the identical prompt", () => {
    const plan = planCurvePoint({ promptTokens: 8192, nGen: 512, repeats: 4 });
    const warmRepeats = plan.filter((s) => s.requestClass === "warm_repeat");
    expect(warmRepeats).toHaveLength(3);
    expect(warmRepeats.every((s) => s.cachePrompt && s.promptTokens === 8192 && s.nPredict === 512)).toBe(true);
  });

  it("still emits the cold point at repeats = 1", () => {
    const plan = planCurvePoint({ promptTokens: 4096, nGen: 128, repeats: 1 });
    expect(plan.map((s) => s.requestClass)).toEqual(["warm_discard", "cold_timed"]);
  });
});

describe("N1 eviction detector", () => {
  it("never condemns the legitimate cold prefill for reporting its full prompt", () => {
    const detection = detectCacheEviction([
      { requestClass: "cold_timed", promptN: 8192 },
      { requestClass: "warm_repeat", promptN: 0 },
      { requestClass: "warm_repeat", promptN: 0 },
    ]);
    expect(detection.evicted).toBe(false);
  });

  it("flags a warm repeat that re-prefilled -- the silent fast-repeat failure mode", () => {
    const detection = detectCacheEviction([
      { requestClass: "cold_timed", promptN: 8192 },
      { requestClass: "warm_repeat", promptN: 8192 },
      { requestClass: "warm_repeat", promptN: 0 },
    ]);
    expect(detection.evicted).toBe(true);
    expect(detection.offendingRepeats).toBe(1);
  });

  it("ignores the warm/discard request entirely", () => {
    expect(
      detectCacheEviction([
        { requestClass: "warm_discard", promptN: 32 },
        { requestClass: "warm_repeat", promptN: 0 },
      ]).evicted
    ).toBe(false);
  });
});

describe("N1 context-shift flag", () => {
  it("flags only when the binary lacks --no-context-shift AND the logs show a shift", () => {
    const log = "slot update_slots: id 0 | task 1 | slot context shift, n_keep = 0";
    expect(sawContextShift(log)).toBe(true);
    expect(
      curveCaveatFlags({ samples: [], serverLog: log, supportsNoContextShift: false })
    ).toContain("context_shift");
    expect(
      curveCaveatFlags({ samples: [], serverLog: log, supportsNoContextShift: true })
    ).not.toContain("context_shift");
  });

  it("stays quiet on an ordinary log", () => {
    expect(sawContextShift("slot launch_slot_: id 0 | processing task")).toBe(false);
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
