import { spawn } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  completeWithRetries,
  executeCurvePoint,
  executeKneeLadder,
  executeFillCurve,
  probeSucceeded,
  failedForHostBackedLayers,
  pickDedupPoint,
  toProbeAttemptReport,
  spawnRuntimeServer,
  streamedCompletion,
  LlamaServerOutputError,
  RuntimeServerStartupError,
  type StreamedRequestInput,
} from "./runtimeBench.js";
import type { StreamSample } from "./loadDriver.js";
import { CURVE_METHOD_VERSION, FILL_CURVE_METHOD_VERSION } from "../../shared/types.js";
import type { SweepItem } from "../../shared/sweep.js";

// A fake llama-server. Every request is recorded, and the reply carries the
// same four timings a real one does: its own prefill pair and its own decode
// pair. Rates here are deliberately realistic in SHAPE -- prefill (batched)
// faster than decode (token by token) -- because v7's cross-checks compare
// the two against each other.
function fakeServer(opts: { ppTps?: number; tgTps?: number } = {}) {
  const requests: (StreamedRequestInput & { promptLength: number })[] = [];
  const ppTps = opts.ppTps ?? 310;
  const tgTps = opts.tgTps ?? 40;
  const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
    requests.push({ ...input, promptLength: input.promptTokens.length });
    const promptN = input.promptTokens.length;
    const promptMs = (promptN / ppTps) * 1000;
    const predictedN = input.nPredict;
    const predictedMs = (predictedN / tgTps) * 1000;
    return {
      ttftMs: promptMs,
      e2eMs: promptMs + predictedMs,
      tokensPredicted: predictedN,
      promptN,
      promptMs,
      predictedN,
      predictedMs,
      slot: input.slot ?? 0,
    };
  };
  return { completion, requests };
}

// Supplied explicitly so no test reaches for a real /tokenize -- fetchFillerBlocks
// throws rather than degrading, which is the point of it.
const BLOCKS = [
  Array.from({ length: 31 }, (_, i) => 1000 + i),
  Array.from({ length: 43 }, (_, i) => 2000 + i),
  Array.from({ length: 37 }, (_, i) => 3000 + i),
];

describe("N1 curve-point execution", () => {
  it("runs every repeat cache-free, on its own prompt, and measures both rates from each one", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 8_192,
      nGen: 512,
      repeats: 5,
      port: 1,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    expect(server.requests).toHaveLength(5);
    // No request opts into the prefix cache, every one sends the full prompt
    // and asks for the full generation, and no two prompts are identical --
    // so no repeat can reuse another's prefix even if the flag were ignored.
    expect(server.requests.every((r) => r.cachePrompt === false)).toBe(true);
    expect(server.requests.every((r) => r.promptLength === 8_192)).toBe(true);
    expect(server.requests.every((r) => r.nPredict === 512)).toBe(true);
    expect(new Set(server.requests.map((r) => r.promptTokens.join(","))).size).toBe(5);

    const pp = execution.results.find((r) => r.test_type === "pp")!;
    const tg = execution.results.find((r) => r.test_type === "tg")!;
    // Both rows now draw on EVERY repeat, where v5 had one pp sample and
    // repeats-1 tg samples.
    expect(pp.sample_count).toBe(5);
    expect(tg.sample_count).toBe(5);
    expect(pp.avg_tps).toBeCloseTo(310, 6);
    expect(tg.avg_tps).toBeCloseTo(40, 6);
    expect(pp.suspect_count).toBe(0);
    expect(tg.suspect_count).toBe(0);
    expect(pp.ttft_n).toBe(5);
    expect(execution.warning).toBeUndefined();
  });

  // The production failure this vintage exists for: a CPU load whose prefill
  // ran at ~15 tok/s reported 1000 tok/s of generation, because the rate was
  // computed as tokens / (end - first chunk) and that window held a single
  // token's worth of teardown latency. The same server shape must now produce
  // the server's own decode rate instead.
  it("does not turn a one-token generation's teardown latency into a generation rate", async () => {
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      const promptMs = (input.promptTokens.length / 15) * 1000; // 15 tok/s prefill
      return {
        ttftMs: promptMs,
        e2eMs: promptMs + 1, // 1 ms after the first chunk -- the v5 numerator's whole basis
        tokensPredicted: 1,
        promptN: input.promptTokens.length,
        promptMs,
        predictedN: 1,
        predictedMs: 1000 / 12, // the server's own decode timer: 12 tok/s
        slot: 0,
      };
    };
    const execution = await executeCurvePoint({
      effectiveCtx: 2_048,
      nGen: 1,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion,
    });
    const tg = execution.results.find((r) => r.test_type === "tg")!;
    expect(tg.avg_tps).toBeCloseTo(12, 6);
    expect(tg.avg_tps).toBeLessThan(execution.results.find((r) => r.test_type === "pp")!.avg_tps);
    expect(tg.suspect_count).toBe(0);
  });

  it("flags a generation rate that reads faster than its own request's prefill", async () => {
    const server = fakeServer({ ppTps: 15, tgTps: 1_000 });
    const execution = await executeCurvePoint({
      effectiveCtx: 2_048,
      nGen: 16,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    const tg = execution.results.find((r) => r.test_type === "tg")!;
    expect(tg.suspect_count).toBe(2);
    expect(tg.avg_tps).toBeCloseTo(1_000, 6); // kept and reported, never erased
    expect(execution.warning).toContain("faster than this request's own prefill");
  });

  it("flags a decode timer this side's wall clock cannot corroborate", async () => {
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => ({
      ttftMs: 1_000,
      e2eMs: 2_000, // 64 tokens can't have been decoded in the 1000 ms after the first chunk...
      tokensPredicted: input.nPredict,
      promptN: input.promptTokens.length,
      promptMs: 1_000,
      predictedN: input.nPredict,
      predictedMs: 100, // ...yet the server claims 100 ms for all 64 of them (640 tok/s)
      slot: 0,
    });
    const execution = await executeCurvePoint({
      effectiveCtx: 4_096,
      nGen: 64,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion,
    });
    const tg = execution.results.find((r) => r.test_type === "tg")!;
    expect(tg.suspect_count).toBe(2);
    expect(execution.warning).toContain("wall clock");
  });

  it("says so when a repeat reports no generation timing at all, instead of shrinking the sample", async () => {
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => ({
      ttftMs: 100,
      e2eMs: 200,
      tokensPredicted: input.nPredict,
      promptN: input.promptTokens.length,
      promptMs: 100,
      predictedN: null,
      predictedMs: null,
      slot: 0,
    });
    const execution = await executeCurvePoint({
      effectiveCtx: 2_048,
      nGen: 64,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion,
    });
    expect(execution.results.some((r) => r.test_type === "tg")).toBe(false);
    expect(execution.warning).toContain("no generation timing");
  });

  // The filler prompt has to be built from ids that decode to valid UTF-8 in
  // the model's OWN vocabulary, or llama.cpp's chat-output parser rejects the
  // generation it provokes -- and it has to carry every register, or MoE
  // prefill reads up to 58% fast (see fillerPrompt.ts).
  it("builds every prompt out of the supplied filler blocks, carrying every register", async () => {
    const server = fakeServer();
    const all = new Set(BLOCKS.flat());
    await executeCurvePoint({
      effectiveCtx: 256,
      nGen: 8,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    expect(server.requests.length).toBeGreaterThan(0);
    expect(server.requests.every((r) => r.promptTokens.every((t) => all.has(t)))).toBe(true);
    const measured = server.requests[server.requests.length - 1].promptTokens;
    expect(new Set(measured.map((t) => Math.floor(t / 1000))).size).toBe(BLOCKS.length);
  });

  it("asks the running server to tokenize the filler when no blocks are supplied", async () => {
    const server = fakeServer();
    const fetchMock = vi.fn(async (..._args: unknown[]) => Response.json({ tokens: [9000, 9001, 9002] }));
    vi.stubGlobal("fetch", fetchMock);
    try {
      await executeCurvePoint({
        effectiveCtx: 256,
        nGen: 8,
        repeats: 2,
        port: 1,
        completion: server.completion,
      });
    } finally {
      vi.unstubAllGlobals();
    }
    expect(fetchMock).toHaveBeenCalled();
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/tokenize"))).toBe(true);
    expect(server.requests.every((r) => r.promptTokens.every((t) => t >= 9000))).toBe(true);
  });

  it("stamps the curve vintage so ordinary runtime rows can never land in a curve", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 4_096,
      nGen: 128,
      repeats: 3,
      port: 1,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    expect(execution.results.every((r) => r.method_version === CURVE_METHOD_VERSION)).toBe(true);
    // Must not collide with the fill curve's vintage: those rows share this
    // column AND engine "server", so an equal number would let prefill slices
    // be read as context-curve points.
    expect(CURVE_METHOD_VERSION).not.toBe(FILL_CURVE_METHOD_VERSION);
  });

  it("writes no prefill row at all when the server reports no prefill timing", async () => {
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => ({
      ttftMs: 100,
      e2eMs: 200,
      tokensPredicted: input.nPredict,
      promptN: 0,
      promptMs: null,
      predictedN: input.nPredict,
      predictedMs: 100,
      slot: 0,
    });
    const execution = await executeCurvePoint({
      effectiveCtx: 2_048,
      nGen: 64,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion,
    });
    expect(execution.results.some((r) => r.test_type === "pp")).toBe(false);
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
        predictedN: 100,
        predictedMs: 1_000,
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
      fillerBlocks: BLOCKS,
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
        predictedN: 8,
        predictedMs: 10,
        promptN: input.promptTokens.length,
        promptMs: 5,
        slot: input.slot ?? 0,
      };
    };
    await executeKneeLadder({ nPrompt: 64, nGen: 8, repeats: 1, slots: [4], port: 1, fillerBlocks: BLOCKS, completion });
    expect(new Set(prompts).size).toBe(4);
  });
});

// The ladder itself moved to shared/probeLadder.ts (and is covered by
// shared/probeLadder.test.ts); what stays here is the per-rung verdict, which
// is about one load rather than about the search.
describe("N2 probe success rule", () => {
  const base = { oom: false, vramPeakMib: 7000, gpuTotalMib: 8176, generated: true, genTps: 11, estimatedVramMib: null };

  // The 1 tok/s floor is gone. It rejected a placement on a rate measured over
  // the probe's short fixed request -- a number that describes a nearly empty
  // cache and not the configuration in the row -- while the thing it was proxying
  // for (GPU memory served from system RAM) is now measured directly.
  it("no longer fails a slow load: a rate is reported, not judged", () => {
    expect(probeSucceeded({ ...base, genTps: 0.4, ngl: 0 }).ok).toBe(true);
  });

  it("still fails a load that generated nothing", () => {
    const dead = probeSucceeded({ ...base, generated: false, genTps: null, ngl: 0 });
    expect(dead.ok).toBe(false);
    expect(dead.reason).toContain("generated nothing");
  });

  // Liveness and speed are separate signals since v7: a rung whose rate could
  // not be computed (an older build reporting no decode timing, or a reading
  // that failed its cross-check) still LOADED, and the ladder must not read
  // the missing number as a dead placement.
  it("passes a load that generated tokens but yielded no usable rate", () => {
    expect(probeSucceeded({ ...base, generated: true, genTps: null, ngl: 0 }).ok).toBe(true);
  });

  it("treats a spill past the adapter total as failure", () => {
    expect(probeSucceeded({ ...base, vramPeakMib: 9000, gpuTotalMib: 8192, ngl: 0 })).toMatchObject({ ok: false, spill: true });
  });

  it("passes a clean load", () => {
    expect(probeSucceeded({ ...base, ngl: 0 })).toMatchObject({
      ok: true,
      spill: false,
      vramDiscrepancy: false,
      failCause: null,
      reason: null,
    });
  });

  // Real loads, 2026-09-14: Qwen3.6-35B-A3B-UD-IQ4_NL on a Radeon RX 6600 XT,
  // Vulkan, b10956 -- llama.cpp's own buffer report against the process's
  // dedicated VRAM once loaded (see shared/gpuSpill.test.ts for all twelve).
  describe("measured GPU spill", () => {
    const eightLayers = { gpuBuffers: { deviceMib: 3634.7, contextMib: 215.51 }, loadedDedicatedMib: 2806.27 };

    it("passes a load whose GPU buffers are all in VRAM", () => {
      const result = probeSucceeded({
        ...base, ngl: 4, gpuBuffers: { deviceMib: 2001.57, contextMib: 263.01 }, loadedDedicatedMib: 2017.33,
        loadedDedicatedJitterMib: 3,
      });
      expect(result).toMatchObject({ ok: true, vramDiscrepancy: false, failCause: null });
      expect(result.gpuSpill).toMatchObject({ measured: true, spilled: false });
    });

    // 828MiB of the buffers missing from VRAM at 1,024 tokens, with half the
    // card free -- the driver does not wait for VRAM to run out.
    it("fails a load with its layers in system RAM, and says no smaller context fixes it", () => {
      const result = probeSucceeded({ ...base, ngl: 8, ...eightLayers, loadedDedicatedJitterMib: 12 });
      expect(result).toMatchObject({ ok: false, spill: false, vramDiscrepancy: true, failCause: "layers" });
      expect(result.reason).toContain("828MiB of the 3635MiB");
      expect(result.reason).toContain("no smaller context fixes it");
    });

    // 234MiB of a 1,069MiB compute buffer in system RAM at 262,144 tokens.
    it("fails a context-sized spill as the cache's, so the search tries a smaller context", () => {
      const result = probeSucceeded({
        ...base, ngl: 4, gpuBuffers: { deviceMib: 3319.56, contextMib: 1581 }, loadedDedicatedMib: 3085.23,
        loadedDedicatedJitterMib: 0,
      });
      expect(result).toMatchObject({ ok: false, failCause: "cache" });
      expect(result.reason).toContain("a smaller context can bring it back");
    });

    it("measures a load with no layers on the GPU too", () => {
      const result = probeSucceeded({
        ...base, ngl: 0, gpuBuffers: { deviceMib: 1069, contextMib: 1069 }, loadedDedicatedMib: 822.35,
        loadedDedicatedJitterMib: 0,
      });
      expect(result).toMatchObject({ ok: false, failCause: "cache" });
    });

    // A rerun of 8 layers at 1,024 tokens spilled only 189MiB -- less than that
    // context's own buffers -- but no smaller context exists to try.
    it("files a spill at the smallest context under the layers, and says why", () => {
      const rerun = {
        ...base, ngl: 8, gpuBuffers: { deviceMib: 3634.7, contextMib: 215.51 }, loadedDedicatedMib: 3446,
        loadedDedicatedJitterMib: 2,
      };
      expect(probeSucceeded(rerun).failCause).toBe("cache");
      const atFloor = probeSucceeded({ ...rerun, atSmallestContext: true });
      expect(atFloor).toMatchObject({ ok: false, failCause: "layers" });
      expect(atFloor.reason).toContain("smallest context");
      expect(atFloor.reason).not.toContain("more than this context's entire");
    });

    it("does not call a difference the counter itself could not resolve a spill", () => {
      const result = probeSucceeded({
        ...base, ngl: 4, gpuBuffers: { deviceMib: 2000, contextMib: 263 }, loadedDedicatedMib: 1980,
        loadedDedicatedJitterMib: 40,
      });
      expect(result.ok).toBe(true);
    });

    it("does not consult the estimate once the spill was measured", () => {
      const result = probeSucceeded({
        ...base, ngl: 26, estimatedVramMib: 27000, vramProcessPeakMib: 2017,
        gpuBuffers: { deviceMib: 2001.57, contextMib: 263.01 }, loadedDedicatedMib: 2017.33, loadedDedicatedJitterMib: 0,
      });
      expect(result).toMatchObject({ ok: true, vramDiscrepancy: false });
    });

    // Rule order: "generated nothing" wins, so a spill failure always generated.
    it("reports no generation, not a spill, when a spilled load also generated nothing", () => {
      const result = probeSucceeded({
        ...base,
        generated: false,
        genTps: null,
        ngl: 8,
        ...eightLayers,
        loadedDedicatedJitterMib: 0,
      });
      expect(result.gpuSpill.spilled).toBe(true);
      expect(result).toMatchObject({ ok: false, failCause: null });
      expect(result.reason).toContain("generated nothing");
    });
  });

  describe("where the spill cannot be measured", () => {
    it.each([
      ["no buffer report", { gpuBuffers: null, loadedDedicatedMib: 2000 }],
      ["no per-process VRAM reading", { gpuBuffers: { deviceMib: 3634.7, contextMib: 215.51 }, loadedDedicatedMib: null }],
      ["nothing placed on a GPU", { gpuBuffers: { deviceMib: 0, contextMib: 0 }, loadedDedicatedMib: 0 }],
    ])("stays unmeasured with %s, and passes", (_label, spill) => {
      const result = probeSucceeded({ ...base, ngl: 8, ...spill });
      expect(result.gpuSpill.measured).toBe(false);
      expect(result.ok).toBe(true);
    });

    // A 35B MoE at full offload on an 8GiB card, with no measurement available:
    // the estimate needs far more than the card has, but an estimate cannot tell
    // a real fallback from a pessimistic one, so this stays a warning.
    it("flags an inferred discrepancy as a warning only", () => {
      const result = probeSucceeded({ ...base, vramPeakMib: 6544, gpuTotalMib: 8192, genTps: 6.8, ngl: 41, estimatedVramMib: 27435 });
      expect(result).toMatchObject({ ok: true, vramDiscrepancy: true });
      expect(result.gpuSpill.measured).toBe(false);
    });

    // 3736/11577 = 0.32 (flagged) against the whole-adapter 5872/11577 = 0.507
    // (not flagged): the difference is the ~2100MiB the desktop held.
    it("infers against this process's VRAM, not every process on the adapter", () => {
      const shared = { oom: false, gpuTotalMib: 8176, generated: true, genTps: 3.54, ngl: 26, estimatedVramMib: 11577 };
      expect(probeSucceeded({ ...shared, vramPeakMib: 5872, vramProcessPeakMib: 3736 }).vramDiscrepancy).toBe(true);
      expect(probeSucceeded({ ...shared, vramPeakMib: 5872, vramProcessPeakMib: null }).vramDiscrepancy).toBe(false);
    });

    it("infers nothing when ngl is 0", () => {
      expect(probeSucceeded({ ...base, vramPeakMib: 1791, gpuTotalMib: 8192, ngl: 0, estimatedVramMib: 22315 }).vramDiscrepancy).toBe(false);
    });
  });
});

// The ladder's "why did it fail" signal, read back off a stored outcome so a
// rung reused from a batch sibling answers the same way.
describe("failedForHostBackedLayers", () => {
  it("is true only for a failure probeSucceeded filed under the layers", () => {
    expect(failedForHostBackedLayers({ ok: false, hostBackedFailCause: "layers" })).toBe(true);
    expect(failedForHostBackedLayers({ ok: false, hostBackedFailCause: "cache" })).toBe(false);
    expect(failedForHostBackedLayers({ ok: false, hostBackedFailCause: null })).toBe(false);
    expect(failedForHostBackedLayers({ ok: true, hostBackedFailCause: "layers" })).toBe(false);
  });

  it("agrees with probeSucceeded on the rung it failed", () => {
    const result = probeSucceeded({
      oom: false, vramPeakMib: 4322, gpuTotalMib: 8176, generated: true, genTps: 11.8, ngl: 8, estimatedVramMib: null,
      gpuBuffers: { deviceMib: 3634.7, contextMib: 215.51 }, loadedDedicatedMib: 2806.27, loadedDedicatedJitterMib: 0,
    });
    expect(failedForHostBackedLayers({ ok: result.ok, hostBackedFailCause: result.failCause })).toBe(true);
  });
});

// spawnRuntimeServer's readiness-failure path (the bug worker/src/index.ts's
// runOneProbeLoad used to hit): the child dies before its health port ever
// answers, so the ONLY way its diagnostic output can reach the caller is if
// the thrown error itself carries it -- `server` never gets assigned in that
// caller, since spawnRuntimeServer never returns. spawnFn drives a REAL
// (trivial, script-based) node child process rather than a real llama-server
// binary, mirroring this file's existing "seam tests drive instead of a real
// llama-server" pattern for the completion function above.
describe("spawnRuntimeServer readiness failures", () => {
  const BASE_ITEM: SweepItem = {
    idx: 0,
    n_prompt: 512,
    n_gen: 128,
    n_depth: 0,
    concurrency: 1,
    threads: 4,
    n_gpu_layers: 0,
    batch_size: 512,
    ubatch_size: 512,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
  };

  // Nothing ever listens here, so waitForReady's health-check fetches just
  // fail (ECONNREFUSED) until the fake process's death is observed.
  const DEAD_PORT_BASE = 48173;

  function crashingSpawnFn(stderrText: string) {
    const script = `process.stderr.write(${JSON.stringify(stderrText)}, () => { process.exit(1); });`;
    return () => spawn(process.execPath, ["-e", script], { stdio: ["ignore", "pipe", "pipe"] });
  }

  it("carries the child's real stderr and exit code when it dies before becoming ready, OOM-shaped output", async () => {
    const oomText = "ggml_cuda_host_malloc: failed to allocate 16384.00 MiB of pinned memory: out of memory\n";
    await expect(
      spawnRuntimeServer({
        // A nonexistent binary path -- same pattern as bench.test.ts's
        // benchInput(): the flag-support probe's spawn() hits ENOENT and
        // resolves false quickly, since spawnFn below replaces the real
        // spawn() call entirely and never touches this path.
        llamaServerPath: "/nonexistent/fake-llama-server",
        modelPath: "/models/fake.gguf",
        port: DEAD_PORT_BASE,
        item: BASE_ITEM,
        slots: 1,
        spawnFn: crashingSpawnFn(oomText),
      })
    ).rejects.toThrow(RuntimeServerStartupError);

    try {
      await spawnRuntimeServer({
        llamaServerPath: "/nonexistent/fake-llama-server",
        modelPath: "/models/fake.gguf",
        port: DEAD_PORT_BASE,
        item: BASE_ITEM,
        slots: 1,
        spawnFn: crashingSpawnFn(oomText),
      });
      expect.unreachable("spawnRuntimeServer should have rejected");
    } catch (err) {
      expect(err).toBeInstanceOf(RuntimeServerStartupError);
      const startupErr = err as RuntimeServerStartupError;
      expect(startupErr.stderr).toContain("out of memory");
      expect(startupErr.code).toBe(1);
      expect(startupErr.signal).toBeNull();
    }
  }, 10_000);

  it("still carries stderr for a crash that is NOT OOM-shaped, rather than discarding it", async () => {
    const crashText = "Segmentation fault (core dumped)\n";
    try {
      await spawnRuntimeServer({
        llamaServerPath: "/nonexistent/fake-llama-server",
        modelPath: "/models/fake.gguf",
        port: DEAD_PORT_BASE + 1,
        item: BASE_ITEM,
        slots: 1,
        spawnFn: crashingSpawnFn(crashText),
      });
      expect.unreachable("spawnRuntimeServer should have rejected");
    } catch (err) {
      expect(err).toBeInstanceOf(RuntimeServerStartupError);
      const startupErr = err as RuntimeServerStartupError;
      expect(startupErr.stderr).toContain("Segmentation fault");
      expect(startupErr.code).toBe(1);
    }
  }, 10_000);

  it("still returns a working handle when the child DOES become ready (regression check on the spawnFn refactor)", async () => {
    const port = DEAD_PORT_BASE + 2;
    const script = `
      const http = require('http');
      process.stderr.write('startup diagnostic line\\n');
      const server = http.createServer((req, res) => {
        if (req.url === '/health') { res.writeHead(200); res.end('ok'); }
        else { res.writeHead(404); res.end(); }
      });
      server.listen(${port});
    `;
    const handle = await spawnRuntimeServer({
      llamaServerPath: "/nonexistent/fake-llama-server",
      modelPath: "/models/fake.gguf",
      port,
      item: BASE_ITEM,
      slots: 1,
      spawnFn: () => spawn(process.execPath, ["-e", script], { stdio: ["ignore", "pipe", "pipe"] }),
    });
    try {
      expect(handle.port).toBe(port);
      expect(handle.stderr()).toContain("startup diagnostic line");
    } finally {
      await handle.stop();
    }
  }, 10_000);
});

// The context tests used to have no recovery at all: one LlamaServerOutputError
// propagated out and index.ts marked the whole probe ladder fatal, abandoning
// every remaining placement over a single request. The MTP path had had a retry
// ladder for months.
describe("parser-failure recovery", () => {
  const parserError = () =>
    new LlamaServerOutputError("rejected", "The model produced output that does not match the expected Content-only format");

  const sample = (): StreamSample => ({
    ttftMs: 10,
    e2eMs: 20,
    tokensPredicted: 8,
    predictedN: 8,
    predictedMs: 10,
    promptN: 64,
    promptMs: 5,
    slot: 0,
  });

  function recorder(failures: number) {
    const seen: { first: number; grammar?: string }[] = [];
    let calls = 0;
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      seen.push({ first: input.promptTokens[0], grammar: input.grammar });
      if (calls++ < failures) throw parserError();
      return sample();
    };
    return { completion, seen };
  }

  it("retries on a DIFFERENT prompt, because an identical greedy retry provably fails identically", async () => {
    const { completion, seen } = recorder(1);
    const out = await completeWithRetries({
      completion, port: 1, tokenCount: 64, offset: 0, nonce: 0,
      blocks: BLOCKS, nPredict: 8, cachePrompt: false,
    });
    expect(out.grammarConstrained).toBe(false);
    expect(seen).toHaveLength(2);
    expect(seen[1].first).not.toBe(seen[0].first);
    expect(seen.every((r) => r.grammar === undefined)).toBe(true);
  });

  it("falls back to a grammar only after every unconstrained attempt is rejected, and says so", async () => {
    const { completion, seen } = recorder(3);
    const out = await completeWithRetries({
      completion, port: 1, tokenCount: 64, offset: 0, nonce: 0,
      blocks: BLOCKS, nPredict: 8, cachePrompt: false,
    });
    expect(out.grammarConstrained).toBe(true);
    expect(seen).toHaveLength(4);
    // Three unconstrained attempts, each on its own prompt, then the grammar.
    expect(seen.slice(0, 3).every((r) => r.grammar === undefined)).toBe(true);
    expect(new Set(seen.slice(0, 3).map((r) => r.first)).size).toBe(3);
    expect(seen[3].grammar).toMatch(/^root ::=/);
  });

  it("stays fatal when even the grammar attempt is rejected -- no other placement will differ", async () => {
    const completion = async (): Promise<StreamSample> => {
      throw parserError();
    };
    await expect(
      completeWithRetries({
        completion, port: 1, tokenCount: 64, offset: 0, nonce: 0,
        blocks: BLOCKS, nPredict: 8, cachePrompt: false,
      })
    ).rejects.toBeInstanceOf(LlamaServerOutputError);
  });

  it("does not burn retries on an error that another prompt cannot fix", async () => {
    let calls = 0;
    const completion = async (): Promise<StreamSample> => {
      calls++;
      throw new Error("llama-server /completion returned 503");
    };
    await expect(
      completeWithRetries({
        completion, port: 1, tokenCount: 64, offset: 0, nonce: 0,
        blocks: BLOCKS, nPredict: 8, cachePrompt: false,
      })
    ).rejects.toThrow("503");
    expect(calls).toBe(1);
  });

  it("still yields a curve point whose reading only came back under a grammar", async () => {
    let calls = 0;
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      // Only the very first request needs the grammar -- that request is the
      // cold TIMED prefill.
      if (!input.grammar && calls++ < 3) throw parserError();
      return { ...sample(), promptN: input.promptTokens.length };
    };
    const execution = await executeCurvePoint({
      effectiveCtx: 256,
      nGen: 8,
      repeats: 2,
      port: 1,
      fillerBlocks: BLOCKS,
      completion,
    });
    expect(execution.results.length).toBeGreaterThan(0);
  });
});

// Live-reproduced against a real Qwen3.8 GGUF on llama.cpp b10793:
// llama-server accepts a raw /completion request (HTTP 200, stream opens
// normally) and then emits an {"error":...} SSE frame instead of real content
// -- its chat-output PEG parser hard-failing over one invalid UTF-8 byte in
// the generated text. Confirmed independent of --reasoning-format,
// --skip-chat-parsing, -rea off, --reasoning-budget and ignore_eos (see
// ggml-org/llama.cpp#25072, and fillerPrompt.ts for the mechanism). Before
// this, streamedCompletion had no way to see that frame as anything other
// than "the model generated zero tokens" -- see the two tests below.
describe("streamedCompletion / llama-server SSE error frames", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function sseResponse(dataLines: unknown[]): Response {
    const body = dataLines.map((line) => `data: ${JSON.stringify(line)}\n\n`).join("") + "data: [DONE]\n\n";
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body));
        controller.close();
      },
    });
    return new Response(stream, { status: 200 });
  }

  it("throws a friendly, specific LlamaServerOutputError for the known Content-only parser rejection", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          {
            error: {
              code: 500,
              message: "The model produced output that does not match the expected Content-only format",
              type: "server_error",
            },
          },
        ])
      )
    );
    const call = streamedCompletion({ port: 1, promptTokens: [100, 101, 102], nPredict: 8, cachePrompt: false });
    await expect(call).rejects.toBeInstanceOf(LlamaServerOutputError);
    await expect(call).rejects.toThrow(/one invalid UTF-8 byte/);
    await expect(call).rejects.toThrow(/ggml-org\/llama\.cpp#25072/);
  });

  it("falls back to a plainer message for an error signature it hasn't seen before, without claiming this specific diagnosis", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => sseResponse([{ error: { code: 500, message: "something else entirely broke" } }]))
    );
    const call = streamedCompletion({ port: 1, promptTokens: [1], nPredict: 1, cachePrompt: false });
    await expect(call).rejects.toBeInstanceOf(LlamaServerOutputError);
    await expect(call).rejects.toThrow("llama-server reported an error during generation: something else entirely broke");
    await expect(call).rejects.not.toThrow(/Content-only/);
  });

  it("still parses a normal streamed completion correctly (regression check on separating JSON.parse from the error check)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse([
          { content: " hi", tokens_predicted: 1 },
          { content: " there", tokens_predicted: 2, timings: { prompt_n: 4, prompt_ms: 12 } },
        ])
      )
    );
    const sample = await streamedCompletion({ port: 1, promptTokens: [1, 2, 3, 4], nPredict: 2, cachePrompt: false });
    expect(sample.tokensPredicted).toBe(2);
    expect(sample.promptN).toBe(4);
    expect(sample.promptMs).toBe(12);
  });
});

// The bug this exists to prevent, verbatim: pp/ttft/prefill_cliff/slope were
// computed by probeSucceeded, stored on the outcome, printed in the worker's
// own log -- and never mapped onto the wire shape, so the UI showed a column
// of em-dashes for all of them. Nothing in the type system catches it: every
// field on the report is optional, so an omitted one is still a valid object.
describe("toProbeAttemptReport", () => {
  const outcome = {
    candidateCtx: 4096,
    ngl: 13,
    ok: true,
    oom: false,
    spill: false,
    vramPeakMib: 6761,
    vramProcessPeakMib: 4380,
    ramPeakMib: 16657,
    ramTotalPeakMib: 21000,
    vramSharedPeakMib: 1821,
    vramSharedTotalPeakMib: 2411,
    vramClaimedPeakMib: 6201,
    vramNeededMib: 5958,
    vramFreeMib: 5778,
    ramNeededMib: 12341,
    ramFreeMib: 24479,
    genTps: 10.3,
    ppTps: 18.9,
    ttftMs: 9389,
    prefillCliff: true,
    gpuBuffersMib: 6462.03,
    gpuInSystemRamMib: -15.76,
    gpuSpillJitterMib: 4,
    hostBackedFailCause: null,
    vramDiscrepancy: false,
    gpuLayersResidentEst: 13,
    gpuLayersResidentExact: true,
  };

  it("carries every measurement onto the wire shape", () => {
    const report = toProbeAttemptReport(outcome) as unknown as Record<string, unknown>;
    // Each measured field, and where it has to land.
    const expected: Record<string, unknown> = {
      vram_peak_mib: 6761,
      vram_process_peak_mib: 4380,
      ram_peak_mib: 16657,
      ram_total_peak_mib: 21000,
      vram_shared_peak_mib: 1821,
      vram_shared_total_peak_mib: 2411,
      vram_claimed_peak_mib: 6201,
      gen_tps: 10.3,
      pp_tps: 18.9,
      ttft_ms: 9389,
      prefill_cliff: true,
      gpu_buffers_mib: 6462.03,
      gpu_in_system_ram_mib: -15.76,
      gpu_spill_jitter_mib: 4,
      gpu_layers_resident_est: 13,
      gpu_layers_resident_exact: true,
    };
    for (const [key, value] of Object.entries(expected)) {
      expect({ [key]: report[key] }).toEqual({ [key]: value });
    }
  });

  it("passes nulls through as nulls rather than dropping the keys", () => {
    const report = toProbeAttemptReport({
      ...outcome, ppTps: null, ttftMs: null, gpuInSystemRamMib: null,
    }) as unknown as Record<string, unknown>;
    expect("pp_tps" in report).toBe(true);
    expect(report.pp_tps).toBeNull();
    expect(report.gpu_in_system_ram_mib).toBeNull();
  });
});

describe("pickDedupPoint", () => {
  const sibling = [
    { candidate_ctx: 1024, ngl: 6, seq: 7 },
    { candidate_ctx: 1024, ngl: 7, seq: 3 },
    { candidate_ctx: 1024, ngl: 6, seq: 11 },
  ];

  it("reuses the sibling's n-th load of a point for this probe's n-th load", () => {
    expect(pickDedupPoint(sibling, 1024, 6, 0)?.seq).toBe(7);
    expect(pickDedupPoint(sibling, 1024, 6, 1)?.seq).toBe(11);
  });

  it("never offers one reading twice, so a control stays a real second load", () => {
    expect(pickDedupPoint(sibling, 1024, 7, 1)).toBeUndefined();
    expect(pickDedupPoint(sibling, 2048, 6, 0)).toBeUndefined();
  });

  it("skips a sibling's claim-only load when this load has to judge spill, and reuses it when it does not", () => {
    const withClaimOnly = [
      { candidate_ctx: 2048, ngl: 17, load_kind: "claim_only", seq: 1 },
      { candidate_ctx: 2048, ngl: 17, load_kind: "full", seq: 2 },
    ];
    expect(pickDedupPoint(withClaimOnly, 2048, 17, 0)?.seq).toBe(2);
    expect(pickDedupPoint(withClaimOnly, 2048, 17, 0, false)?.seq).toBe(1);
  });
});

// A fake llama-server that really behaves like a single-slot prompt cache:
// it remembers the last prompt and prefills only what extends past the
// longest common prefix. Prefill is slower the fuller the context already is,
// so the curve has a shape to check.
function prefixCachingServer(opts: { cacheHolds?: boolean } = {}) {
  let cached: number[] = [];
  const requests: StreamedRequestInput[] = [];
  const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
    requests.push(input);
    const prompt = input.promptTokens;
    let common = 0;
    if (input.cachePrompt && opts.cacheHolds !== false) {
      while (common < cached.length && common < prompt.length && cached[common] === prompt[common]) common++;
    }
    const promptN = prompt.length - common;
    // 1000 tok/s on an empty context, falling linearly with depth.
    const rate = 1000 / (1 + common / 10_000);
    cached = [...prompt, 7];
    return {
      ttftMs: 1,
      e2eMs: 2,
      tokensPredicted: 1,
      promptN,
      promptMs: (promptN / rate) * 1000,
      predictedN: 1,
      predictedMs: 25,
      slot: 0,
    };
  };
  return { completion, requests };
}

describe("fill curve execution", () => {
  const ITEM: SweepItem = {
    idx: 0,
    n_prompt: 8_151,
    n_gen: 0,
    n_depth: 0,
    concurrency: 1,
    threads: 4,
    n_gpu_layers: 33,
    batch_size: 2048,
    ubatch_size: 512,
    cache_type_k: "q8_0",
    cache_type_v: "q8_0",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
  };

  it("grows one prompt to ctx - 0.5 % and records each slice at its own depth", async () => {
    const server = prefixCachingServer();
    const execution = await executeFillCurve({
      ctx: 8_192,
      steps: 4,
      repeats: 3,
      port: 1,
      item: ITEM,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    expect(execution.stopped).toBe(false);
    expect(execution.warning).toBeUndefined();
    // 3 repeats x 4 slices, cache on, each a prefix of the full prompt.
    expect(server.requests).toHaveLength(12);
    expect(server.requests.every((r) => r.cachePrompt && r.nPredict === 1)).toBe(true);
    expect(server.requests.slice(0, 4).map((r) => r.promptTokens.length)).toEqual([2_038, 4_076, 6_113, 8_151]);

    const rows = execution.results;
    expect(rows.map((r) => r.n_depth)).toEqual([0, 2_038, 4_076, 6_113]);
    expect(rows.map((r) => (r.n_depth ?? 0) + r.n_prompt)).toEqual([2_038, 4_076, 6_113, 8_151]);
    expect(rows.every((r) => r.test_type === "pp" && r.method_version === FILL_CURVE_METHOD_VERSION)).toBe(true);
    expect(rows.every((r) => r.n_gpu_layers === 33 && r.cache_type_k === "q8_0" && r.sample_count === 3)).toBe(true);
    expect(rows.every((r) => r.suspect_count === 0)).toBe(true);
    // Prefill slows as the context fills.
    for (let i = 1; i < rows.length; i++) expect(rows[i].avg_tps).toBeLessThan(rows[i - 1].avg_tps);
    expect(rows[0].avg_tps).toBeCloseTo(1000, 0);
  });

  it("excludes slices that re-prefilled from the start because the cache did not hold", async () => {
    const server = prefixCachingServer({ cacheHolds: false });
    const execution = await executeFillCurve({
      ctx: 8_192,
      steps: 4,
      repeats: 2,
      port: 1,
      item: ITEM,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    // The first slice starts from an empty context either way, so it holds.
    expect(execution.results[0].suspect_count).toBe(0);
    expect(execution.results.slice(1).every((r) => r.suspect_count === 2 && r.avg_tps === 0)).toBe(true);
    expect(execution.warning).toMatch(/did not hold/);
  });

  it("stops between requests and reports only the slices it measured", async () => {
    const server = prefixCachingServer();
    const execution = await executeFillCurve({
      ctx: 8_192,
      steps: 4,
      repeats: 1,
      port: 1,
      item: ITEM,
      fillerBlocks: BLOCKS,
      completion: server.completion,
      shouldStop: () => server.requests.length >= 2,
    });
    expect(execution.stopped).toBe(true);
    expect(execution.results).toHaveLength(2);
  });
});
