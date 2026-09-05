import { spawn } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  completeWithRetries,
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

// Supplied explicitly so no test reaches for a real /tokenize -- fetchFillerBlocks
// throws rather than degrading, which is the point of it.
const BLOCKS = [
  Array.from({ length: 31 }, (_, i) => 1000 + i),
  Array.from({ length: 43 }, (_, i) => 2000 + i),
  Array.from({ length: 37 }, (_, i) => 3000 + i),
];

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
      fillerBlocks: BLOCKS,
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
      serverLog: "",
      supportsNoContextShift: true,
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
        serverLog: "",
        supportsNoContextShift: true,
        completion: server.completion,
      });
    } finally {
      vi.unstubAllGlobals();
    }
    expect(fetchMock).toHaveBeenCalled();
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/tokenize"))).toBe(true);
    expect(server.requests.every((r) => r.promptTokens.every((t) => t >= 9000))).toBe(true);
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
      fillerBlocks: BLOCKS,
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
      fillerBlocks: BLOCKS,
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
      fillerBlocks: BLOCKS,
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
      fillerBlocks: BLOCKS,
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
      fillerBlocks: BLOCKS,
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
  it("treats loading-but-crawling as failure -- loading is not the same as usable", () => {
    const crawling = probeSucceeded({
      oom: false,
      vramPeakMib: 7000,
      gpuTotalMib: 8192,
      genTps: 0.4,
      ngl: 0,
      estimatedVramMib: null,
    });
    expect(crawling.ok).toBe(false);
    expect(crawling.reason).toContain(`${PROBE_MIN_GEN_TPS} tok/s floor`);
  });

  it("treats a spill past the adapter total as failure", () => {
    const spilled = probeSucceeded({
      oom: false,
      vramPeakMib: 9000,
      gpuTotalMib: 8192,
      genTps: 30,
      ngl: 0,
      estimatedVramMib: null,
    });
    expect(spilled).toMatchObject({ ok: false, spill: true });
  });

  it("passes a clean load above the floor", () => {
    expect(
      probeSucceeded({ oom: false, vramPeakMib: 7000, gpuTotalMib: 8192, genTps: 30, ngl: 0, estimatedVramMib: null })
    ).toMatchObject({
      ok: true,
      spill: false,
      vramDiscrepancy: false,
      reason: null,
    });
  });

  // The real case this exists for: a full-VRAM-oversubscribed load whose
  // generation speed still clears the (deliberately low) tok/s floor because
  // the OS quietly backed the overcommit with system RAM instead of erroring
  // -- confirmed live on an AMD RX 6600 XT (8GB, Vulkan backend) loading a
  // 35B MoE model at full offload: gen tok/s stayed above the floor and VRAM
  // peak never exceeded the card's total, so neither the genTps nor the spill
  // check caught it, yet the estimate (needing far more than the card has)
  // was right and the load was really running from host RAM. `ok` itself does
  // NOT flip here -- see this function's own doc comment for why that
  // decision belongs to the caller's vramDiscrepancyPolicy, not this rule.
  it("flags a VRAM discrepancy when observed peak is far below the estimate for a real offload", () => {
    const result = probeSucceeded({
      oom: false,
      vramPeakMib: 6544,
      gpuTotalMib: 8192,
      genTps: 6.8,
      ngl: 41,
      estimatedVramMib: 27435,
    });
    expect(result.vramDiscrepancy).toBe(true);
    expect(result.ok).toBe(true);
  });

  it("does not flag a discrepancy when ngl is 0 (nothing was requested on GPU)", () => {
    const result = probeSucceeded({
      oom: false,
      vramPeakMib: 1791,
      gpuTotalMib: 8192,
      genTps: 8.7,
      ngl: 0,
      estimatedVramMib: 22315,
    });
    expect(result.vramDiscrepancy).toBe(false);
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

  it("flags a curve point whose reading only came back under a grammar", async () => {
    let calls = 0;
    const completion = async (input: StreamedRequestInput): Promise<StreamSample> => {
      // Only the very first request needs the grammar; the flag must still
      // reach the row, since that request is the cold TIMED prefill.
      if (!input.grammar && calls++ < 3) throw parserError();
      return { ...sample(), promptN: input.promptTokens.length };
    };
    const execution = await executeCurvePoint({
      effectiveCtx: 256,
      nGen: 8,
      repeats: 2,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      fillerBlocks: BLOCKS,
      completion,
    });
    expect(execution.results.length).toBeGreaterThan(0);
    expect(execution.results.every((r) => (r.caveat_flags ?? []).includes("grammar_constrained"))).toBe(true);
  });

  it("leaves the flag off when nothing needed constraining", async () => {
    const server = fakeServer();
    const execution = await executeCurvePoint({
      effectiveCtx: 256,
      nGen: 8,
      repeats: 2,
      port: 1,
      serverLog: "",
      supportsNoContextShift: true,
      fillerBlocks: BLOCKS,
      completion: server.completion,
    });
    expect(execution.results.every((r) => !(r.caveat_flags ?? []).includes("grammar_constrained"))).toBe(true);
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
