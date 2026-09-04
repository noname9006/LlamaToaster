import { spawn } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  executeCurvePoint,
  executeKneeLadder,
  probeSucceeded,
  spawnRuntimeServer,
  streamedCompletion,
  LlamaServerOutputError,
  RuntimeServerStartupError,
  PROBE_MIN_GEN_TPS,
  type StreamedRequestInput,
} from "./runtimeBench.js";
import type { StreamSample } from "./loadDriver.js";
import { CURVE_METHOD_VERSION } from "../../shared/types.js";
import type { SweepItem } from "../../shared/sweep.js";

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

// Live-reproduced against a real Qwen3.5/3.8 "thinking" GGUF on llama.cpp
// b10793: llama-server accepts a raw /completion request (HTTP 200, stream
// opens normally) and then emits an {"error":...} SSE frame instead of real
// content -- its chat-format output validator rejecting the model's own
// generated text. Confirmed independent of --reasoning-format,
// --skip-chat-parsing, -rea off, --reasoning-budget and ignore_eos (see
// ggml-org/llama.cpp#19869 and friends). Before this, streamedCompletion had
// no way to see that frame as anything other than "the model generated zero
// tokens" -- see the two tests below.
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
    await expect(call).rejects.toThrow(/testing can't run correctly for it on this build/);
    await expect(call).rejects.toThrow(/ggml-org\/llama\.cpp#19869/);
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
