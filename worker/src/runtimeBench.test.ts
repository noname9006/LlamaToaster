import { describe, expect, it } from "vitest";
import {
  executeCurvePoint,
  executeKneeLadder,
  probeSucceeded,
  PROBE_MIN_GEN_TPS,
  type StreamedRequestInput,
} from "./runtimeBench.js";
import type { StreamSample } from "./loadDriver.js";
import { CURVE_METHOD_VERSION } from "../../shared/types.js";

// A fake llama-server: every request is recorded, and the reply is shaped by
// whether the caller opted into prefix-cache reuse.
function fakeServer(opts: { evictOnRepeat?: number; coldPromptMs?: number } = {}) {
  const requests: (StreamedRequestInput & { promptLength: number })[] = [];
  let warmRepeatCount = 0;
  const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
    requests.push({ ...input, promptLength: input.promptTokens.length });
    const isWarm = input.cachePrompt;
    let promptN = isWarm ? 0 : input.promptTokens.length;
    if (isWarm) {
      warmRepeatCount++;
      if (opts.evictOnRepeat != null && warmRepeatCount === opts.evictOnRepeat) {
        promptN = input.promptTokens.length; // the cache did not hold
      }
    }
    const ttftMs = isWarm ? 40 : (opts.coldPromptMs ?? 26_400);
    return {
      ttftMs,
      e2eMs: ttftMs + input.nPredict * 25,
      tokensPredicted: input.nPredict,
      promptN,
      promptMs: promptN > 0 ? (opts.coldPromptMs ?? 26_400) : null,
      slot: input.slot ?? 0,
    };
  };
  return { completion, requests };
}

describe("N1 curve-point execution", () => {
  it("runs the three request classes in order and never averages them together", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 8_192,
      nGen: 512,
      repeats: 5,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      completion: server.completion,
    });
    expect(server.requests).toHaveLength(6); // 1 warm/discard + 1 cold + 4 warm
    // The warm/discard request is short AND a different prompt, so it cannot
    // seed the measured prefix into the cache.
    expect(server.requests[0].promptLength).toBe(32);
    expect(server.requests[0].promptTokens[0]).not.toBe(server.requests[1].promptTokens[0]);
    // The cold prefill is the timed data point: full prompt, one token, no reuse.
    expect(server.requests[1].promptLength).toBe(8_192);
    expect(server.requests[1].nPredict).toBe(1);
    expect(server.requests[1].cachePrompt).toBe(false);
    // Warm repeats reuse the identical prompt.
    expect(server.requests.slice(2).every((r) => r.cachePrompt && r.promptLength === 8_192)).toBe(true);

    const pp = execution.results.find((r) => r.test_type === "pp")!;
    const tg = execution.results.find((r) => r.test_type === "tg")!;
    // pp comes from the cold response's OWN timings -- prompt_n / prompt_ms.
    expect(pp.avg_tps).toBeCloseTo((8192 / 26_400) * 1000, 6);
    expect(pp.ttft_n).toBe(1);
    expect(pp.ttft_ms_p50).toBe(26_400);
    expect(pp.ttft_ms_p50).toBe(pp.ttft_ms_p95);
    // tg statistics come only from the warm repeats.
    expect(tg.sample_count).toBe(4);
    expect(tg.repeat_samples).toHaveLength(4);
  });

  it("stamps METHOD_VERSION 2 so ordinary runtime rows can never land in a curve", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 4_096,
      nGen: 128,
      repeats: 3,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      completion: server.completion,
    });
    expect(execution.results.every((r) => r.method_version === CURVE_METHOD_VERSION)).toBe(true);
  });

  it("flags cache_evicted when a warm repeat re-prefills, and says so in the warning", async () => {
    const server = fakeServer({ evictOnRepeat: 2 });
    const execution = await executeCurvePoint({
      effectiveCtx: 8_192,
      nGen: 512,
      repeats: 4,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      completion: server.completion,
    });
    expect(execution.results.every((r) => (r.caveat_flags ?? []).includes("cache_evicted"))).toBe(true);
    expect(execution.warning).toContain("prefix cache did not hold");
  });

  it("does not flag the legitimate cold prefill as an eviction", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 8_192,
      nGen: 512,
      repeats: 4,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      completion: server.completion,
    });
    expect(execution.results.every((r) => (r.caveat_flags ?? []).length === 0)).toBe(true);
    expect(execution.warning).toBeUndefined();
  });

  it("flags context_shift when the binary lacks the flag and the log shows a shift", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 8_192,
      nGen: 512,
      repeats: 3,
      port: 1,
      serverLog: "slot update_slots: slot context shift, n_keep = 0",
      supportsNoContextShift: false,
      completion: server.completion,
    });
    expect(execution.results.every((r) => (r.caveat_flags ?? []).includes("context_shift"))).toBe(true);
  });

  it("marks the pp reading suspect rather than inventing a rate when the server reports no prefill timing", async () => {
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => ({
      ttftMs: 100,
      e2eMs: 200,
      tokensPredicted: input.nPredict,
      promptN: 0,
      promptMs: null,
      slot: 0,
    });
    const execution = await executeCurvePoint({
      effectiveCtx: 2_048,
      nGen: 64,
      repeats: 2,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      completion,
    });
    const pp = execution.results.find((r) => r.test_type === "pp")!;
    expect(pp.avg_tps).toBe(0);
    expect(pp.suspect_count).toBe(1);
  });
});

describe("N5 knee ladder execution", () => {
  it("issues the slot count SIMULTANEOUSLY and records concurrency on every row", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      return {
        ttftMs: 100 * (input.slot ?? 0) + 100,
        e2eMs: 100 * (input.slot ?? 0) + 1_100,
        tokensPredicted: 100,
        promptN: input.promptTokens.length,
        promptMs: 90,
        slot: input.slot ?? 0,
      };
    };
    const rows = await executeKneeLadder({
      nPrompt: 4_096,
      nGen: 128,
      repeats: 1,
      slots: [1, 2, 4],
      port: 1,
      completion,
    });
    expect(maxInFlight).toBe(4);
    expect(rows.map((r) => r.concurrency)).toEqual([1, 2, 4]);
    expect(rows.every((r) => r.ttft_ms_p95 != null && r.ttft_n! > 0)).toBe(true);
  });

  it("never reuses the prefix cache across concurrent streams", async () => {
    const prompts: string[] = [];
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      prompts.push(input.promptTokens.slice(0, 4).join(","));
      expect(input.cachePrompt).toBe(false);
      return {
        ttftMs: 10,
        e2eMs: 20,
        tokensPredicted: 8,
        promptN: input.promptTokens.length,
        promptMs: 5,
        slot: input.slot ?? 0,
      };
    };
    await executeKneeLadder({ nPrompt: 64, nGen: 8, repeats: 1, slots: [4], port: 1, completion });
    expect(new Set(prompts).size).toBe(4);
  });
});

// The ladder itself moved to shared/probeLadder.ts (and is covered by
// shared/probeLadder.test.ts); what stays here is the per-rung verdict, which
// is about one load rather than about the search.
describe("N2 probe success rule", () => {
  it("treats loading-but-crawling as failure -- loading is not the same as usable", () => {
    const crawling = probeSucceeded({ oom: false, vramPeakMib: 7000, gpuTotalMib: 8192, genTps: 0.4 });
    expect(crawling.ok).toBe(false);
    expect(crawling.reason).toContain(`${PROBE_MIN_GEN_TPS} tok/s floor`);
  });

  it("treats a spill past the adapter total as failure", () => {
    const spilled = probeSucceeded({ oom: false, vramPeakMib: 9000, gpuTotalMib: 8192, genTps: 30 });
    expect(spilled).toMatchObject({ ok: false, spill: true });
  });

  it("passes a clean load above the floor", () => {
    expect(probeSucceeded({ oom: false, vramPeakMib: 7000, gpuTotalMib: 8192, genTps: 30 })).toMatchObject({
      ok: true,
      spill: false,
      reason: null,
    });
  });
});
