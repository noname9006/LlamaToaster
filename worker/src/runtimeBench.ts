// BENCHMARKING_PLAN_V8.md N1 (context curves), N2 (usable-config probe) and
// N5 (concurrency knee): the {engine:"server", spec:"off"} execution paths.
//
// Deliberately separate from serverBench.ts, which drives MTP speculative
// decoding: "N1's server-path choreography ships as separate code paths; its
// warm-cache repeats may not silently change ordinary runtime runs' reuse
// semantics." The pure planning/summarizing logic lives in loadDriver.ts and
// is unit-tested there; this file is the process and HTTP plumbing over it.

import { spawn, type ChildProcess } from "node:child_process";
import type { IngestResultInput, ProbeAttemptReport } from "../../shared/types.js";
import { CURVE_METHOD_VERSION, FILL_CURVE_METHOD_VERSION, SERVER_METHOD_VERSION } from "../../shared/types.js";
import { fillBoundaries, fillTarget } from "../../shared/fillCurve.js";
import type { SweepItem } from "../../shared/sweep.js";
import { isVramDiscrepancy } from "../../shared/vramEstimate.js";
import {
  measureGpuSpill,
  UNMEASURED_GPU_SPILL,
  type GpuBufferReport,
  type GpuSpillVerdict,
  type GrowthSpillVerdict,
  type SpillAnchor,
} from "../../shared/gpuSpill.js";
import {
  appendBoundedOutput,
  collapseTensorLoadSpam,
  MAX_CAPTURED_PROCESS_OUTPUT_CHARS,
  parseModelBufferSizes,
  parseOffloadLayers,
  type BenchLogger,
  type BenchResult,
} from "./bench.js";
import { buildPromptTokens, fetchFillerBlocks, type FillerBlocks } from "./fillerPrompt.js";
import {
  buildServerArgs,
  contextSizeForSlots,
  planConcurrentBatch,
  planCurvePoint,
  summarizeStreams,
  type RequestClass,
  type StreamSample,
} from "./loadDriver.js";
import { supportsFlag } from "./binary-probe.js";
import { PROBE_LADDER_MIN_CTX } from "../../shared/probeLadder.js";

const READY_POLL_INTERVAL_MS = 500;
const READY_TIMEOUT_MS = 120_000;
const STOP_GRACE_MS = 5_000;
const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;

export interface RuntimeServerHandle {
  proc: ChildProcess;
  port: number;
  stderr: () => string;
  stop: () => Promise<{ code: number | null; signal: NodeJS.Signals | null }>;
}

// Thrown by spawnRuntimeServer when the child dies, or never opens its
// health port, before waitForReady is satisfied. A plain Error would lose
// the process's captured output the instant it's thrown -- spawnRuntimeServer
// never gets to construct/return a RuntimeServerHandle in this path, so
// without this the caller has no way to reach stderr()/exit info at all (see
// worker/src/index.ts's runOneProbeLoad, which used to see `server` stuck at
// null for exactly this reason and so could never classify OOM here).
export class RuntimeServerStartupError extends Error {
  readonly stderr: string;
  readonly code: number | null;
  readonly signal: NodeJS.Signals | null;

  constructor(message: string, info: { stderr: string; code: number | null; signal: NodeJS.Signals | null }) {
    super(message);
    this.name = "RuntimeServerStartupError";
    this.stderr = info.stderr;
    this.code = info.code;
    this.signal = info.signal;
  }
}

// Test seam, same spirit as CompletionFn below: lets a test drive a real
// (but trivial, script-based) child process instead of needing a real
// llama-server binary on disk just to exercise the readiness-failure path.
export type SpawnFn = (path: string, args: string[]) => ChildProcess;

export interface SpawnRuntimeServerInput {
  llamaServerPath: string;
  modelPath: string;
  port: number;
  item: SweepItem;
  slots: number;
  mainGpu?: number;
  contextSizeOverride?: number;
  /** Speed runs load the model fully into RAM instead of mmap-ing it. */
  noMmap?: boolean;
  log?: BenchLogger;
  onSpawn?: (proc: ChildProcess) => void;
  spawnFn?: SpawnFn;
}

function defaultSpawn(path: string, args: string[]): ChildProcess {
  return spawn(path, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
}

export async function spawnRuntimeServer(input: SpawnRuntimeServerInput): Promise<RuntimeServerHandle> {
  // §0.7 -- probed once per binary identity. An unsupported flag disables its
  // behavior rather than failing the item.
  const supportsNoContextShift = await supportsFlag(input.llamaServerPath, "--no-context-shift").catch(() => false);
  // Same §0.7 probe pattern for --fit -- see buildServerArgs's own comment
  // for why this is forced off rather than left at llama-server's default.
  const supportsFit = await supportsFlag(input.llamaServerPath, "--fit").catch(() => false);
  // Same §0.7 probe pattern for --mlock -- keeps the model locked in RAM for
  // the run's duration instead of left swappable.
  const supportsMlock = await supportsFlag(input.llamaServerPath, "--mlock").catch(() => false);
  const noMmap = input.noMmap === true && (await supportsFlag(input.llamaServerPath, "--no-mmap").catch(() => false));
  const args = buildServerArgs({
    modelPath: input.modelPath,
    port: input.port,
    item: input.item,
    slots: input.slots,
    mainGpu: input.mainGpu,
    supportsNoContextShift,
    supportsFit,
    supportsMlock,
    contextSizeOverride: input.contextSizeOverride,
    noMmap,
  });
  input.log?.info(`llama-server ${args.join(" ")}`);
  const proc = (input.spawnFn ?? defaultSpawn)(input.llamaServerPath, args);
  input.onSpawn?.(proc);

  let stderr = "";
  proc.stderr?.on("data", (chunk: Buffer) => {
    stderr = appendBoundedOutput(stderr, chunk.toString(), MAX_CAPTURED_PROCESS_OUTPUT_CHARS);
  });
  proc.stdout?.on("data", (chunk: Buffer) => {
    stderr = appendBoundedOutput(stderr, chunk.toString(), MAX_CAPTURED_PROCESS_OUTPUT_CHARS);
  });

  const closed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    proc.on("close", (code, signal) => resolve({ code, signal }));
  });

  try {
    await waitForReady(input.port, Date.now() + READY_TIMEOUT_MS, closed);
  } catch (err) {
    // Still alive means this was a readiness TIMEOUT, not a death -- and
    // since this throw means the caller never gets a handle to stop() it,
    // leaving it running here would leak the process.
    if (proc.exitCode == null && proc.signalCode == null) {
      proc.kill("SIGKILL");
    }
    const exitInfo = await closed;
    const message = err instanceof Error ? err.message : String(err);
    throw new RuntimeServerStartupError(message, { stderr, ...exitInfo });
  }

  return {
    proc,
    port: input.port,
    stderr: () => stderr,
    stop: async () => {
      if (proc.exitCode == null && proc.signalCode == null) {
        proc.kill("SIGTERM");
        const timer = setTimeout(() => proc.kill("SIGKILL"), STOP_GRACE_MS);
        const result = await closed;
        clearTimeout(timer);
        return result;
      }
      return closed;
    },
  };
}

async function waitForReady(
  port: number,
  deadlineAt: number,
  exited: Promise<{ code: number | null; signal: NodeJS.Signals | null }>
): Promise<void> {
  let dead = false;
  void exited.then(() => {
    dead = true;
  });
  while (Date.now() < deadlineAt) {
    if (dead) throw new Error("llama-server exited before it became ready");
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) return;
    } catch {
      /* not listening yet */
    }
    await new Promise((r) => setTimeout(r, READY_POLL_INTERVAL_MS));
  }
  throw new Error(`llama-server did not become ready within ${READY_TIMEOUT_MS}ms`);
}

// --- One streamed request ---------------------------------------------------

export interface StreamedRequestInput {
  port: number;
  promptTokens: number[];
  nPredict: number;
  cachePrompt: boolean;
  slot?: number;
  signal?: AbortSignal;
  /**
   * GBNF constraining what may be sampled. Only ever set by the last-resort
   * retry below, and only for a model that cannot otherwise produce a
   * parseable response at all -- constrained sampling has its own cost, so a
   * reading measured this way is not directly comparable with an
   * unconstrained one.
   */
  grammar?: string;
}

// Thrown when llama-server accepts a /completion request (HTTP 200, stream
// opens normally) but then emits an `{"error":...}` SSE frame mid-stream
// instead of real content -- e.g. its chat-format output validator rejecting
// the model's own generated text. Distinct from a thrown network/HTTP error
// so callers (the N2 ladder) can tell "this model can't generate through
// this endpoint at all" apart from an ordinary too-slow/OOM candidate and
// stop trying further placements instead of re-discovering the same failure
// at every rung.
export class LlamaServerOutputError extends Error {
  constructor(
    message: string,
    /** llama-server's own error text, unmodified -- kept for the log even
     * when `message` above has been rewritten into something readable. */
    public readonly rawMessage: string
  ) {
    super(message);
    this.name = "LlamaServerOutputError";
  }
}

// The one signature we've actually root-caused: llama.cpp's chat-output PEG
// parser hard-failing the whole response over a single invalid-UTF-8 byte in
// the generated text (ggml-org/llama.cpp#25072 -- see fillerPrompt.ts for the
// mechanism and why no server flag or request field makes it tolerant).
//
// The usual cause was our OWN filler prompt, which used to be built from raw
// single-byte token ids and reliably provoked byte-fragment output; that is
// fixed at the source in fillerPrompt.ts. This stays as the guard for the
// residual case (a model that emits an invalid byte anyway -- observed on MTP
// builds), and anything that is not this signature gets a plainer fallback
// rather than falsely claiming this diagnosis.
function describeLlamaServerError(rawMessage: string): string {
  if (/does not match the expected .*format/i.test(rawMessage)) {
    return (
      "llama-server rejected this model's generated output as invalid: llama.cpp's chat-output parser fails a whole " +
      "response over one invalid UTF-8 byte, and no server flag disables that check (ggml-org/llama.cpp#25072). " +
      "This is an upstream llama.cpp limitation, not a hardware or configuration problem."
    );
  }
  return `llama-server reported an error during generation: ${rawMessage}`;
}

// --- Recovering from the parser failure -------------------------------------
//
// Three attempts, each with a different prompt, then one grammar-constrained
// attempt as a last resort. Mirrors the MTP path's ladder (serverBench.ts) --
// which the context tests never had: before this, a single
// LlamaServerOutputError propagated straight out and index.ts marked the whole
// probe ladder fatal, so one unlucky request abandoned every remaining
// placement.
//
// Retrying with a DIFFERENT prompt is the whole point: sampling is greedy
// (temperature 0), so a byte-for-byte identical retry provably fails
// identically -- confirmed live on the MTP path, 3 attempts out of 3.
const MAX_COMPLETION_ATTEMPTS = 3;
// Rotates the filler on each retry. Same constant, and the same reasoning, as
// serverBench.ts's RETRY_PROMPT_SHIFT.
const RETRY_PROMPT_SHIFT = 137;

// The last resort. Confirmed live against b10793: the exact request that fails
// with "does not match the expected Content-only format" succeeds under a
// grammar, because the sampler can no longer reach a token whose bytes are an
// invalid UTF-8 fragment. Deliberately plain ASCII -- the point is to make an
// invalid byte unreachable, not to shape the text.
const FALLBACK_GRAMMAR = 'root ::= [a-zA-Z0-9 ,.;:!?\\n]+';

export interface ResilientCompletionInput {
  completion: CompletionFn;
  port: number;
  tokenCount: number;
  offset: number;
  nonce: number;
  blocks: FillerBlocks;
  nPredict: number;
  cachePrompt: boolean;
  slot?: number;
  log?: BenchLogger;
}

export interface ResilientCompletion {
  sample: StreamSample;
  /** True when only the grammar-constrained attempt produced a reading. */
  grammarConstrained: boolean;
}

export async function completeWithRetries(input: ResilientCompletionInput): Promise<ResilientCompletion> {
  const attemptAt = (offset: number, grammar?: string) =>
    input.completion({
      port: input.port,
      promptTokens: buildPromptTokens(input.tokenCount, offset, input.nonce, input.blocks),
      nPredict: input.nPredict,
      cachePrompt: input.cachePrompt,
      slot: input.slot,
      ...(grammar ? { grammar } : {}),
    });

  for (let attempt = 0; attempt < MAX_COMPLETION_ATTEMPTS; attempt++) {
    const offset = input.offset + attempt * RETRY_PROMPT_SHIFT;
    try {
      return { sample: await attemptAt(offset), grammarConstrained: false };
    } catch (err) {
      // Only the parser failure is worth another prompt; anything else (a dead
      // server, a timeout) fails fast rather than being retried three times.
      if (!(err instanceof LlamaServerOutputError)) throw err;
      input.log?.warn(
        `llama-server rejected its own output at prompt offset ${offset} ` +
          `(attempt ${attempt + 1}/${MAX_COMPLETION_ATTEMPTS}): ${err.rawMessage}`
      );
    }
  }

  input.log?.warn(
    "every unconstrained attempt was rejected; retrying under a grammar so this configuration " +
      "still yields a reading"
  );
  // A failure here is genuine and stays fatal: nothing about this model can
  // generate through this endpoint, and no other placement will differ.
  return { sample: await attemptAt(input.offset, FALLBACK_GRAMMAR), grammarConstrained: true };
}

// TTFT is the arrival of the FIRST streamed chunk, measured here rather than
// derived from n_prompt/pp -- which is precisely the semantics change §0.1
// says increments METHOD_VERSION.
export async function streamedCompletion(input: StreamedRequestInput): Promise<StreamSample> {
  const startedAt = Date.now();
  let ttftMs: number | null = null;
  let tokensPredicted = 0;
  let promptN = 0;
  let promptMs: number | null = null;

  const res = await fetch(`http://127.0.0.1:${input.port}/completion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: input.signal,
    body: JSON.stringify({
      prompt: input.promptTokens,
      n_predict: input.nPredict,
      stream: true,
      cache_prompt: input.cachePrompt,
      ignore_eos: true,
      temperature: 0,
      ...(input.grammar ? { grammar: input.grammar } : {}),
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`llama-server /completion returned ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (ttftMs == null) ttftMs = Date.now() - startedAt;
    buffered += decoder.decode(value, { stream: true });
    let newlineAt = buffered.indexOf("\n");
    while (newlineAt !== -1) {
      const line = buffered.slice(0, newlineAt).trim();
      buffered = buffered.slice(newlineAt + 1);
      newlineAt = buffered.indexOf("\n");
      if (!line.startsWith("data:")) continue;
      const payload = line.slice("data:".length).trim();
      if (!payload || payload === "[DONE]") continue;
      let parsed: {
        timings?: { prompt_n?: number; prompt_ms?: number; predicted_n?: number };
        tokens_predicted?: number;
        error?: { message?: string; code?: number; type?: string };
      };
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue; // a partial SSE frame; the next chunk completes it
      }
      // A hard failure arrives as HTTP 200 + a normal-looking SSE stream that
      // just carries an {"error":...} frame instead of content/timings -- not
      // a thrown fetch/HTTP error, so the checks below would otherwise never
      // see it and this would look identical to "the model generated zero
      // tokens" (see LlamaServerOutputError's doc comment).
      if (parsed.error) {
        const rawMessage = parsed.error.message ?? JSON.stringify(parsed.error);
        throw new LlamaServerOutputError(describeLlamaServerError(rawMessage), rawMessage);
      }
      if (parsed.timings) {
        if (typeof parsed.timings.prompt_n === "number") promptN = parsed.timings.prompt_n;
        if (typeof parsed.timings.prompt_ms === "number") promptMs = parsed.timings.prompt_ms;
      }
      if (typeof parsed.tokens_predicted === "number") tokensPredicted = parsed.tokens_predicted;
    }
  }
  const e2eMs = Date.now() - startedAt;
  return {
    ttftMs: ttftMs ?? e2eMs,
    e2eMs,
    tokensPredicted,
    promptN,
    promptMs,
    slot: input.slot ?? 0,
  };
}

// The seam tests drive instead of a real llama-server.
export type CompletionFn = (input: StreamedRequestInput) => Promise<StreamSample>;

// --- N1: one curve point ----------------------------------------------------

export interface CurvePointExecutionInput {
  effectiveCtx: number;
  nGen: number;
  repeats: number;
  port: number;
  promptOffset?: number;
  completion?: CompletionFn;
  /**
   * Tokenized filler blocks (see fillerPrompt.ts). Omit and they are fetched
   * from the running server, which is the only way to get ids valid in this
   * model's vocabulary; supplied explicitly only by tests.
   */
  fillerBlocks?: FillerBlocks;
  log?: BenchLogger;
}

export interface CurvePointExecution {
  results: IngestResultInput[];
  samples: (StreamSample & { requestClass: RequestClass })[];
  warning?: string;
}

// The choreography itself. Three request classes, never averaged together --
// planCurvePoint owns the sequencing, this owns the arithmetic that turns
// each class into the columns it is allowed to write:
//   * cold_timed  -> ttft_ms_p50/p95 (ttft_n = 1) AND the pp row, since
//                    timings.prompt_ms / prompt_n IS the point's pp value.
//   * warm_repeat -> the tg row's rate, stddev and e2e mean.
export async function executeCurvePoint(input: CurvePointExecutionInput): Promise<CurvePointExecution> {
  const completion = input.completion ?? streamedCompletion;
  const fillerBlocks = input.fillerBlocks ?? (await fetchFillerBlocks(input.port));
  const plan = planCurvePoint({
    promptTokens: input.effectiveCtx,
    nGen: input.nGen,
    repeats: input.repeats,
  });

  const samples: (StreamSample & { requestClass: RequestClass })[] = [];
  for (const step of plan) {
    const outcome = await completeWithRetries({
      completion,
      port: input.port,
      tokenCount: step.promptTokens,
      offset: input.promptOffset ?? 0,
      nonce: step.nonce,
      blocks: fillerBlocks,
      nPredict: step.nPredict,
      cachePrompt: step.cachePrompt,
      log: input.log,
    });
    samples.push({ ...outcome.sample, requestClass: step.requestClass });
  }

  const cold = samples.find((s) => s.requestClass === "cold_timed");
  const warm = samples.filter((s) => s.requestClass === "warm_repeat");

  const results: IngestResultInput[] = [];
  const shared = {
    n_depth: 0,
    n_threads: 0,
    n_gpu_layers: 0,
    batch_size: 0,
    ubatch_size: 0,
    cache_type_k: "",
    cache_type_v: "",
    flash_attn: "",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
    // Choreographed points stamp their own vintage -- cold-timed prefill plus
    // warm-repeat statistics is a real semantics change under §0.1, and that
    // stamp is what keeps ordinary runtime rows out of curves. Bumped again
    // when the filler prompt became a mixed-register passage, which moved
    // measured MoE prefill by 58%.
    method_version: CURVE_METHOD_VERSION,
    prompt_offset: input.promptOffset ?? 0,
    concurrency: 1,
  };

  if (cold) {
    // pp from the cold request's OWN timings, which is how the curve keeps a
    // pp column while §0.2 keeps n_depth = 0 on server rows.
    const ppTps = cold.promptMs != null && cold.promptMs > 0 ? (cold.promptN / cold.promptMs) * 1000 : null;
    results.push({
      ...shared,
      test_type: "pp",
      n_prompt: input.effectiveCtx,
      n_gen: 0,
      avg_tps: ppTps ?? 0,
      stddev_tps: 0,
      sample_count: 1,
      suspect_count: ppTps == null ? 1 : 0,
      // Single-sample by construction; the UI labels it as such rather than
      // implying a p50/p95 over repeats that never happened.
      ttft_ms_p50: cold.ttftMs,
      ttft_ms_p95: cold.ttftMs,
      ttft_n: 1,
      e2e_ms_mean: cold.e2eMs,
    } as IngestResultInput);
  }

  if (warm.length > 0) {
    const summary = summarizeStreams(warm);
    const rates = warm
      .filter((s) => s.e2eMs > s.ttftMs && s.tokensPredicted > 0)
      .map((s) => (s.tokensPredicted / (s.e2eMs - s.ttftMs)) * 1000);
    const mean = rates.length > 0 ? rates.reduce((a, b) => a + b, 0) / rates.length : 0;
    const variance =
      rates.length > 1 ? rates.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (rates.length - 1) : 0;
    results.push({
      ...shared,
      test_type: "tg",
      n_prompt: input.effectiveCtx,
      n_gen: input.nGen,
      avg_tps: mean,
      stddev_tps: Math.sqrt(variance),
      sample_count: rates.length,
      suspect_count: 0,
      repeat_samples: rates,
      e2e_ms_mean: summary.e2eMeanMs ?? undefined,
    } as IngestResultInput);
  }

  const evictedCount = samples.filter((s) => s.requestClass === "warm_repeat" && s.promptN > 0).length;
  return {
    results,
    samples,
    warning:
      evictedCount > 0
        ? `${evictedCount} warm repeat(s) re-prefilled the prompt: the prefix cache did not hold, so this point's generation numbers are not comparable to a cache-warm reading`
        : undefined,
  };
}

// --- Fill curve: prefill speed along one context's fill ---------------------

export interface FillCurveExecutionInput {
  /** The server's -c; the prompt stops at fillTarget(ctx). */
  ctx: number;
  steps: number;
  repeats: number;
  port: number;
  /** The sweep item this curve belongs to -- stamped onto every row. */
  item: SweepItem;
  completion?: CompletionFn;
  fillerBlocks?: FillerBlocks;
  log?: BenchLogger;
  /** Checked before every request; a true return ends the curve early. */
  shouldStop?: () => boolean;
  onProgress?: (progress: { repeat: number; filled: number; fill: number }) => void;
}

export interface FillCurveExecution {
  results: IngestResultInput[];
  warning?: string;
  stopped: boolean;
}

// A slice reading counts only when the server prefilled (about) the slice
// itself. Far more means the prompt cache did not hold and the request
// re-prefilled from an earlier point, so its rate belongs to a different span
// of the context. The slack absorbs a token or two of BOS/prefix-match
// boundary handling.
function sliceHeld(promptN: number, expected: number): boolean {
  return Math.abs(promptN - expected) <= Math.max(2, Math.ceil(expected * 0.01));
}

// Each repeat walks the whole fill once with its own prompt (a different
// nonce), so the first slice of repeat 2 cannot reuse repeat 1's cache. Within
// a repeat, request i sends the first boundaries[i] tokens with the prompt
// cache on -- only the newly appended slice is prefilled, and that request's
// own prompt timings are the slice's rate. Every request asks for one token
// under a plain-ASCII grammar: the output is thrown away, and the grammar
// keeps llama-server's output validator (see LlamaServerOutputError) from
// failing a request over a single generated byte.
export async function executeFillCurve(input: FillCurveExecutionInput): Promise<FillCurveExecution> {
  const completion = input.completion ?? streamedCompletion;
  const blocks = input.fillerBlocks ?? (await fetchFillerBlocks(input.port));
  const fill = fillTarget(input.ctx);
  const boundaries = fillBoundaries(fill, input.steps);
  const repeats = Math.max(1, input.repeats);

  // rates[step] -> per-repeat { rate, held }
  const rates: { rate: number; held: boolean }[][] = boundaries.map(() => []);
  let stopped = false;

  outer: for (let r = 0; r < repeats; r++) {
    const tokens = buildPromptTokens(fill, 0, r + 1, blocks);
    let prev = 0;
    for (let s = 0; s < boundaries.length; s++) {
      if (input.shouldStop?.()) {
        stopped = true;
        break outer;
      }
      const end = boundaries[s];
      input.onProgress?.({ repeat: r, filled: end, fill });
      const sample = await completion({
        port: input.port,
        promptTokens: tokens.slice(0, end),
        nPredict: 1,
        cachePrompt: true,
        grammar: FALLBACK_GRAMMAR,
      });
      const expected = end - prev;
      if (sample.promptMs != null && sample.promptMs > 0 && sample.promptN > 0) {
        rates[s].push({
          rate: (sample.promptN / sample.promptMs) * 1000,
          held: sliceHeld(sample.promptN, expected),
        });
      } else {
        rates[s].push({ rate: 0, held: false });
      }
      if (!sliceHeld(sample.promptN, expected)) {
        input.log?.warn(
          `fill curve: slice ${prev}..${end} prefilled ${sample.promptN} tokens (expected ${expected}) -- excluded`
        );
      }
      prev = end;
    }
  }

  const { item } = input;
  const results: IngestResultInput[] = [];
  let excluded = 0;
  let prev = 0;
  for (let s = 0; s < boundaries.length; s++) {
    const end = boundaries[s];
    const all = rates[s];
    if (all.length === 0) break; // stopped before this slice was ever measured
    const clean = all.filter((x) => x.held).map((x) => x.rate);
    const suspect = all.filter((x) => !x.held).map((x) => x.rate);
    excluded += suspect.length;
    const mean = clean.length > 0 ? clean.reduce((a, b) => a + b, 0) / clean.length : 0;
    const variance =
      clean.length > 1 ? clean.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (clean.length - 1) : 0;
    results.push({
      test_type: "pp",
      n_prompt: end - prev,
      n_gen: 0,
      n_depth: prev,
      n_threads: item.threads,
      n_gpu_layers: item.n_gpu_layers,
      batch_size: item.batch_size,
      ubatch_size: item.ubatch_size,
      cache_type_k: item.cache_type_k,
      cache_type_v: item.cache_type_v,
      flash_attn: item.flash_attn,
      mtp: "off",
      n_gpu_layers_draft: 0,
      n_cpu_moe: item.n_cpu_moe,
      avg_tps: mean,
      stddev_tps: Math.sqrt(variance),
      sample_count: all.length,
      suspect_count: suspect.length,
      suspect_samples: suspect,
      repeat_samples: clean,
      method_version: FILL_CURVE_METHOD_VERSION,
      prompt_offset: 0,
      concurrency: 1,
    } as IngestResultInput);
    prev = end;
  }

  return {
    results,
    stopped,
    warning:
      excluded > 0
        ? `${excluded} slice reading(s) re-prefilled more than their own slice -- the prompt cache did not hold ` +
          `(common on sliding-window models), so those readings were excluded from the curve`
        : undefined,
  };
}

// --- N5: the concurrency ladder --------------------------------------------

export interface KneeExecutionInput {
  nPrompt: number;
  nGen: number;
  repeats: number;
  slots: number[];
  port: number;
  promptOffset?: number;
  completion?: CompletionFn;
  /** See CurvePointExecutionInput.fillerBlocks. */
  fillerBlocks?: FillerBlocks;
  log?: BenchLogger;
  /** Called before each slot count so the caller can restart the server with the right --parallel. */
  beforeSlotCount?: (slots: number) => Promise<number>;
}

// Rows land as ordinary results with `concurrency` set -- the knee itself is
// a derived read (shared/curves.ts's deriveKnee), never a stored verdict.
export async function executeKneeLadder(input: KneeExecutionInput): Promise<IngestResultInput[]> {
  const completion = input.completion ?? streamedCompletion;
  const rows: IngestResultInput[] = [];
  for (const slots of input.slots) {
    const port = (await input.beforeSlotCount?.(slots)) ?? input.port;
    // Re-fetched per slot count because beforeSlotCount restarts the server
    // (same model, so the same blocks -- but this never assumes that).
    const fillerBlocks = input.fillerBlocks ?? (await fetchFillerBlocks(port));
    const batchSamples: StreamSample[] = [];
    for (let repeat = 0; repeat < Math.max(1, input.repeats); repeat++) {
      const plan = planConcurrentBatch({ slots, promptTokens: input.nPrompt, nGen: input.nGen });
      // Simultaneous, not sequential -- that is the whole measurement.
      const batch = await Promise.all(
        plan.map((step) =>
          completeWithRetries({
            completion,
            port,
            tokenCount: step.promptTokens,
            offset: input.promptOffset ?? 0,
            nonce: step.nonce + repeat * 1000,
            blocks: fillerBlocks,
            nPredict: step.nPredict,
            cachePrompt: false,
            slot: step.slot,
            log: input.log,
          })
        )
      );
      batchSamples.push(...batch.map((b) => b.sample));
    }
    const summary = summarizeStreams(batchSamples);
    const shared = {
      n_depth: 0,
      n_threads: 0,
      n_gpu_layers: 0,
      batch_size: 0,
      ubatch_size: 0,
      cache_type_k: "",
      cache_type_v: "",
      flash_attn: "",
      mtp: "off",
      n_gpu_layers_draft: 0,
      n_cpu_moe: 0,
      method_version: SERVER_METHOD_VERSION,
      concurrency: slots,
      ttft_ms_p50: summary.ttftP50Ms,
      ttft_ms_p95: summary.ttftP95Ms,
      ttft_n: summary.ttftN,
      e2e_ms_mean: summary.e2eMeanMs,
    };
    rows.push({
      ...shared,
      test_type: "tg",
      n_prompt: input.nPrompt,
      n_gen: input.nGen,
      // Aggregate throughput across every slot: past the knee this keeps
      // rising while per-user latency collapses, and the chart shows both.
      avg_tps: summary.aggregateTps ?? 0,
      stddev_tps: 0,
      sample_count: batchSamples.length,
      suspect_count: 0,
    } as IngestResultInput);
  }
  return rows;
}

// --- N2: the probe ----------------------------------------------------------

export interface ProbeAttemptOutcome {
  candidateCtx: number;
  ok: boolean;
  oom: boolean;
  spill: boolean;
  /** MemorySampler's vram_peak_mib -- WHOLE-ADAPTER despite the name (every
   * process on the GPU combined). vramProcessPeakMib below is the
   * llama.cpp-only figure. */
  vramPeakMib: number | null;
  /** The probe's own RSS peak (MemorySampler.stop()'s ram_peak_mib) -- the
   * real measured, per-process RAM usage, alongside vramPeakMib. Pairs with
   * ramTotalPeakMib below the same way vramProcessPeakMib pairs with
   * vramPeakMib. */
  ramPeakMib?: number | null;
  /** MemorySampler's vram_process_peak_mib -- this load's own (llama.cpp-
   * only) peak VRAM usage, as distinct from vramPeakMib's whole-adapter
   * reading above. Undefined/null wherever the backend/platform never
   * attributed a reading to this pid during the whole load (not a measured
   * 0) -- more likely here than for vramPeakMib, since per-process
   * attribution can lag a fresh spawn or never catch up on a short load. */
  vramProcessPeakMib?: number | null;
  /** MemorySampler's ram_total_peak_mib -- whole-system RAM in use (every
   * process combined), the RAM-side counterpart to vramPeakMib's
   * whole-adapter reading. */
  ramTotalPeakMib?: number | null;
  /** The probe's own peak system-RAM-backed GPU allocation (MemorySampler's
   * vram_process_shared_peak_mib -- vram.ts's GpuMemoryReading.processShared:
   * Windows WDDM "Shared Usage" or Linux amdgpu GTT). The DIRECT measured
   * answer to "how much silently spilled into system RAM instead of erroring"
   * -- vramDiscrepancy above only ever INFERS that from a needed-vs-peak gap.
   * Null wherever no such counter/file exists at all (not a measured 0). */
  vramSharedPeakMib?: number | null;
  /** MemorySampler's vram_total_shared_peak_mib -- WHOLE-ADAPTER
   * system-RAM-backed GPU memory (every process combined), the total beside
   * vramSharedPeakMib's llama-only figure. Null where no such counter exists. */
  vramSharedTotalPeakMib?: number | null;
  /** MemorySampler's vram_process_claimed_peak_mib -- llama-server's own
   * dedicated + shared GPU memory, summed within one reading and peaked:
   * everything it claimed on the device, however the driver split it between
   * VRAM and system RAM. Null when no per-process dedicated reading was taken. */
  vramClaimedPeakMib?: number | null;
  genTps: number | null;
  /** Fraction of the adapter total still free at this candidate. */
  headroomFrac?: number | null;
  /** The placement this rung loaded at -- the ladder moves ngl too. */
  ngl?: number | null;
  /** computeDualPoolFit's prediction for this rung, and the real free pools. */
  vramNeededMib?: number | null;
  vramFreeMib?: number | null;
  ramNeededMib?: number | null;
  ramFreeMib?: number | null;
  error?: string;
  /** Set when this attempt failed in a way no other (ctx, ngl) rung can fix
   * -- currently just a LlamaServerOutputError (the model's own output
   * rejected by llama-server independent of placement, see runtimeBench.ts).
   * Tells the ladder loop to stop searching immediately instead of re-
   * discovering the identical failure at every remaining rung. */
  fatal?: boolean;
  /** A short, worker-log-only tail of this attempt's captured process output
   * -- populated only when the failure was NOT classified as OOM, so a
   * genuine startup bug is debuggable from the worker log instead of a bare
   * "exited before it became ready". Never sent to the server: the report's
   * own `error` already carries the short message. */
  stderrTail?: string;
  /** N2 batch dedup -- set when this "attempt" was never actually loaded,
   * but reused verbatim from an earlier sibling run's own measurement of
   * this exact (candidateCtx, ngl) point (see worker/src/index.ts's
   * findDedupMatch). Names the sibling run it came from. */
  reusedFromRunId?: string | null;
  /** True when this rung's GPU memory was not all in VRAM: MEASURED when
   * probeSucceeded could hold llama.cpp's buffer report against a per-process
   * dedicated reading (gpuInSystemRamMib is set, and the rung failed), else
   * INFERRED from an observed VRAM peak far below computeDualPoolFit's
   * prediction, which only ever warns. worker/src/index.ts's
   * describeProbeVramDiscrepancy explains the inferred case.
   * vramDiscrepancyPolicy's retry/fail escalation never applies here -- that is
   * the sweep path's. */
  vramDiscrepancy?: boolean;
  /** The measured spill (shared/gpuSpill.ts): every buffer llama.cpp reported
   * putting on the GPU, how much of it this process's dedicated VRAM did not
   * hold once loaded (negative when VRAM holds more -- the normal clean
   * reading), and how far those dedicated readings moved. All null when the
   * spill could not be measured. */
  gpuBuffersMib?: number | null;
  gpuInSystemRamMib?: number | null;
  gpuSpillJitterMib?: number | null;
  /** What `llama-server --list-devices` reported free on the probe's devices,
   * read once before the ladder's first load (binary-probe.ts's
   * readListDevicesFreeMib). gpuBuffersMib below it is the probe's second
   * target. */
  listDevicesFreeMib?: number | null;
  /** "full" -- loaded, prompted, generated, spill judged. "claim_stop" -- stopped
   * once ready because the claim did not fit free VRAM: no generation, no spill
   * verdict. "claim_only" -- stopped once ready because the claim fit and the
   * load could only ever answer that target (probeLadder.ts nextClaimOnly): no
   * generation, no spill verdict. "error" -- failed for a reason that is not memory (readiness
   * timeout, a crash without an out-of-memory signature, a rejected request),
   * so it decides nothing about the claim. */
  loadKind?: "full" | "claim_stop" | "claim_only" | "error";
  /** Target 2's verdict for this load, judged per device where llama.cpp's
   * buffer report and --list-devices both name the devices
   * (claimFitsFreeByDevice), on the totals otherwise. Undefined from a load
   * that never reported buffers. */
  claimFitsFree?: boolean | null;
  /** Spill per phase (shared/gpuSpill.ts measureGrowthSpill): each phase's
   * min(s, d) against the anchor, with its tolerance. Null where the phase had
   * no reading. */
  spillReadyMib?: number | null;
  spillReadyJitterMib?: number | null;
  spillWorkMib?: number | null;
  spillWorkJitterMib?: number | null;
  /** See ProbeAttemptReport.spill_method and its siblings. */
  spillMethod?: "anchor" | "growth" | null;
  spillSharedGrowthMib?: number | null;
  spillUnlandedGrowthMib?: number | null;
  /** The anchor as it stands after an anchor load (built from the first, noise
   * folded in by the second). Worker-only: the ladder loop carries it to every
   * later load; never reported. Undefined on every other load. */
  spillAnchor?: SpillAnchor | null;
  /** The ladder's bounds, repeated on every row -- see ProbeAttemptReport. */
  ladderNglMax?: number | null;
  ladderMaxCtx?: number | null;
  /** Which spill failed this rung: its LAYERS (more in system RAM than the
   * context's whole KV cache and compute buffer, so no smaller context fixes
   * it) or its context's buffers. Null on a pass and on every other kind of
   * failure. */
  hostBackedFailCause?: "layers" | "cache" | null;
  /** Prompt-processing rate for this rung, from llama-server's own
   * timings.prompt_n / prompt_ms. Reported per rung because prefill has its
   * OWN placement cliff, several layers below the weights one -- see
   * PREFILL_CLIFF_RATIO. */
  ppTps?: number | null;
  /** Time to first streamed token, measured client-side from request send.
   * The user-visible face of the prefill cliff: it jumped 2.6s -> 10.1s across
   * a single layer on the reference machine. */
  ttftMs?: number | null;
  /** Set when ppTps collapsed against the best clean rung in this run
   * (isPrefillCliff). Never flips `ok`: the placement works, it is just a bad
   * trade for a prompt-heavy workload. */
  prefillCliff?: boolean;
  /** How many of `ngl`'s claimed layers actually landed in a GPU buffer,
   * per worker/src/index.ts's computeResidentLayers -- the same claimed-vs-
   * landed check the sweep path already surfaces as ResultRow's
   * gpu_layers_resident_est, now available for a probe rung too. EXACT when
   * this build printed its per-layer "assigned to device" lines under -v,
   * a byte-ratio estimate from the post-allocation buffer-size report
   * otherwise (see gpuLayersResidentExact), null when neither was available
   * (ngl<=0, an older build, or the load failed before tensor loading
   * finished). */
  gpuLayersResidentEst?: number | null;
  /** True when gpuLayersResidentEst came from llama.cpp's own per-layer
   * "assigned to device" lines (an exact count), false when it's the
   * coarser buffer-byte-ratio estimate. Meaningless when
   * gpuLayersResidentEst is null. */
  gpuLayersResidentExact?: boolean;
}

/**
 * There is deliberately no MINIMUM generation rate any more.
 *
 * A 1 tok/s floor used to reject "loading is not the same as usable", but it
 * was answering a question the probe can no longer be trusted to ask: every
 * rung generates PROBE_GEN_TOKENS at a PROBE_PROMPT_TOKENS-token prompt
 * whatever `-c` says, so the rate it measures describes a ~512-token context
 * and nothing else. Using
 * a threshold on that to reject a placement means rejecting on a number that
 * was never about the configuration being judged.
 *
 * What actually distinguishes a usable placement from a thrashing one is
 * measured directly now -- probeSucceeded holds llama.cpp's own GPU buffer
 * report against the process's dedicated VRAM rather than inferring a
 * fallback from the symptom. A rung that
 * generates nothing at all still fails below; a rung that generates slowly is
 * reported with its rate and left to the caller.
 */
export { PROBE_GEN_TOKENS, PROBE_PROMPT_TOKENS } from "../../shared/probeLadder.js";

// The ladder that used to live here -- one context axis, x0.75 on failure and
// x1.33 on a roomy success, three loads max -- has been replaced by
// shared/probeLadder.ts, which searches placement as well as context and is
// shared with the client so the two cannot disagree about what a legal
// context is. Only the per-rung success rule below stayed behind, because it
// is about one load's verdict rather than about the search.

// A probe's own success rule, kept next to the ladder that consumes it: no
// OOM, no spill past the adapter total, some generation, and nothing llama.cpp
// put on the GPU being served from system RAM instead.
//
// That last rule is the one a silently oversubscribed placement meets. On a
// driver that backs an overcommitted allocation with system RAM instead of
// erroring (Windows WDDM on any vendor, amdgpu GTT) none of the other three can
// fire: nothing OOMs, dedicated VRAM never passes the adapter total because the
// overflow lands in shared memory, and the load still generates. Observed live:
// max_gpu on a 10GiB RTX 3080 "verified" every layer of a model several GiB
// larger than the card. It is measured by holding llama.cpp's own buffer report
// against this process's dedicated VRAM -- no reference load, no estimate, no
// tuned limit; see shared/gpuSpill.ts.
//
// Where that cannot be measured (no per-process VRAM reading, as on Metal, or a
// build that printed no buffer sizes) the older needed-vs-observed inference
// still runs, and only ever warns: it cannot tell a real fallback from an
// estimate that was merely pessimistic, and failing on it is what 926ab8c
// backed out.
/** "+145MiB", "-3MiB", or "not measured" -- a growth reading, sign always shown. */
export function signedMib(v: number | null | undefined): string {
  if (v == null) return "not measured";
  const r = Math.round(v);
  return `${r >= 0 ? "+" : ""}${r}MiB`;
}

export function probeSucceeded(input: {
  oom: boolean;
  // WHOLE-ADAPTER VRAM peak, every process combined -- read only by the
  // adapter-total rule below, which is a statement about the device.
  vramPeakMib: number | null;
  gpuTotalMib: number | null;
  genTps: number | null;
  // The rung's requested layer count and computeDualPoolFit's predicted GPU
  // need for it: the inference's inputs, consulted only when the spill could
  // not be measured. ngl<=0 or a missing estimate never infers anything.
  ngl: number;
  estimatedVramMib: number | null;
  // This process's own dedicated VRAM peak over the whole load, for that same
  // inference. Null when the platform never attributed a reading to this pid.
  vramProcessPeakMib?: number | null;
  // The measured rule's two sides: llama.cpp's report of what it put on the
  // GPU (bench.ts's parseGpuBufferReport), and this process's dedicated VRAM
  // once the model had loaded, with how far those readings moved
  // (MemorySampler's vram_process_loaded_*).
  gpuBuffers?: GpuBufferReport | null;
  loadedDedicatedMib?: number | null;
  loadedDedicatedJitterMib?: number | null;
  // This rung already runs at the smallest context the probe ever tries, so no
  // smaller context can fix a spill (see GpuSpillInput.atSmallestContext).
  atSmallestContext?: boolean;
  // A spill verdict measured elsewhere (the context test's per-phase trace).
  // When present it replaces the loaded-peak measurement below.
  gpuSpillOverride?: GpuSpillVerdict;
}): {
  ok: boolean;
  spill: boolean;
  vramDiscrepancy: boolean;
  gpuSpill: GpuSpillVerdict;
  /** Which spill failed the rung, if one did -- see ProbeAttemptOutcome.hostBackedFailCause. */
  failCause: "layers" | "cache" | null;
  reason: string | null;
} {
  if (input.oom) {
    return {
      ok: false,
      spill: false,
      vramDiscrepancy: false,
      gpuSpill: UNMEASURED_GPU_SPILL,
      failCause: null,
      reason: "out of memory at this context",
    };
  }
  // Still the whole-adapter reading, and deliberately so: "spilled past the
  // adapter's VRAM total" is a statement about the DEVICE, not about this
  // process's share of it. It never fires under Windows WDDM, whose whole
  // behaviour is to back oversubscription with shared memory rather than push
  // dedicated past the adapter total -- which is why the measured rule exists.
  const spill =
    input.vramPeakMib != null && input.gpuTotalMib != null && input.vramPeakMib > input.gpuTotalMib;
  if (spill) {
    return {
      ok: false,
      spill: true,
      vramDiscrepancy: false,
      gpuSpill: UNMEASURED_GPU_SPILL,
      failCause: null,
      reason: "the allocation spilled past this adapter's VRAM total",
    };
  }
  const gpuSpill = input.gpuSpillOverride ?? measureGpuSpill({
    buffers: input.gpuBuffers ?? null,
    dedicatedMib: input.loadedDedicatedMib ?? null,
    dedicatedJitterMib: input.loadedDedicatedJitterMib ?? null,
    atSmallestContext: input.atSmallestContext,
  });
  // A measurement decides whenever one exists; the inference is strictly worse
  // evidence of the same fact. Per-process where available -- the whole-adapter
  // reading credits the rung with every other process's VRAM.
  const observedMib = input.vramProcessPeakMib ?? input.vramPeakMib;
  const vramDiscrepancy = gpuSpill.measured
    ? gpuSpill.spilled
    : input.ngl > 0 &&
      input.estimatedVramMib != null &&
      observedMib != null &&
      isVramDiscrepancy(input.estimatedVramMib, observedMib);
  // "Generated nothing measurable" is still a failure -- that is not a slow
  // configuration, it is one that did not work. Any positive rate passes.
  if (input.genTps == null) {
    return {
      ok: false,
      spill: false,
      vramDiscrepancy,
      gpuSpill,
      failCause: null,
      reason: "the model loaded but produced no measurable generation",
    };
  }
  if (gpuSpill.spilled && gpuSpill.inSystemRamMib != null && input.gpuBuffers) {
    // Judged as growth over the anchor, what a smaller context can take back is
    // what this context ADDS in KV cache and compute buffer over the anchor's.
    const growth = (gpuSpill as Partial<GrowthSpillVerdict>).unlandedGrowthMib != null ? (gpuSpill as GrowthSpillVerdict) : null;
    const contextBuffersMib = growth?.contextGrowthMib ?? input.gpuBuffers.contextMib;
    const contextMib = Math.round(contextBuffersMib);
    const beyondContext = gpuSpill.inSystemRamMib - (gpuSpill.jitterMib ?? 0) > contextBuffersMib;
    const contextWords = growth ? "what this context adds in KV cache and compute buffer over the anchor" : "this context's entire KV cache and compute buffer";
    const consequence =
      gpuSpill.cause === "cache"
        ? `no more than ${contextWords} (${contextMib}MiB), so a smaller context can bring it back into VRAM`
        : beyondContext
          ? `more than ${contextWords} (${contextMib}MiB), so the model's layers are in system RAM and no smaller ` +
            `context fixes it`
          : `and this is already the smallest context the probe tries, so only fewer layers can bring it back into VRAM`;
    return {
      ok: false,
      spill: false,
      vramDiscrepancy,
      gpuSpill,
      failCause: gpuSpill.cause,
      reason: growth
        ? `${Math.round(gpuSpill.inSystemRamMib)}MiB more of llama.cpp's GPU memory is in system RAM than at the ` +
          `anchor load (shared ${signedMib(growth.sharedGrowthMib)}, claim not in VRAM ` +
          `${signedMib(growth.unlandedGrowthMib)}, tolerance ${Math.round(gpuSpill.jitterMib ?? 0)}MiB) -- ${consequence}`
        : `${Math.round(gpuSpill.inSystemRamMib)}MiB of the ${Math.round(input.gpuBuffers.deviceMib)}MiB llama.cpp ` +
          `put on the GPU is being served from system RAM, not VRAM -- ${consequence}`,
    };
  }
  return { ok: true, spill: false, vramDiscrepancy, gpuSpill, failCause: null, reason: null };
}

/**
 * Did this rung fail BECAUSE its layers were in system RAM, as opposed to its
 * context's buffers, or not fitting at all?
 *
 * The ladder needs the distinction (LadderAttempt.hostBacked): a smaller
 * context cannot rescue a placement whose weights are in system RAM, so a
 * context phase stops walking on the first one -- while a context-sized spill
 * is exactly what a smaller context fixes. Read from the cause probeSucceeded
 * recorded, which a rung reused from a batch sibling carries too.
 */
export function failedForHostBackedLayers(attempt: Pick<ProbeAttemptOutcome, "ok" | "hostBackedFailCause">): boolean {
  return !attempt.ok && attempt.hostBackedFailCause === "layers";
}

// Shared BenchResult shaping so both runtime paths report through
// worker/src/index.ts's existing finalization unchanged.
export function toBenchResult(input: {
  results: IngestResultInput[];
  stderr: string;
  code: number | null;
  signal: NodeJS.Signals | null;
  timedOut?: boolean;
  warning?: string;
}): BenchResult {
  const stderr = collapseTensorLoadSpam(input.stderr);
  return {
    stdout: "",
    stderr,
    code: input.code,
    signal: input.signal,
    timedOut: input.timedOut ?? false,
    results: input.results,
    warning: input.warning,
    // No MTP draft on this path at all: speculation is a different engine
    // pair (§0.2), so there is never a second model's offload line to
    // disambiguate.
    offload: parseOffloadLayers(input.stderr, false),
    modelBufferSizes: parseModelBufferSizes(input.stderr) ?? undefined,
  };
}

/**
 * The batch sibling's reading a probe may reuse for its next load of a point:
 * the n-th load reuses a sibling's n-th load of that point, never the same
 * reading twice -- a control is only a control if it is a second real load.
 * `loadsHere` is how many times this probe has already loaded the point.
 * `needsSpill`: this load has to judge target 1, so a sibling's claim_only load
 * -- stopped before any spill verdict -- cannot stand in for it.
 */
export function pickDedupPoint<P extends { candidate_ctx: number; ngl: number | null; load_kind?: string | null }>(
  points: readonly P[],
  ctx: number,
  ngl: number,
  loadsHere: number,
  needsSpill = true
): P | undefined {
  return points.filter(
    (p) => p.candidate_ctx === ctx && p.ngl === ngl && !(needsSpill && p.load_kind === "claim_only")
  )[loadsHere];
}

export const RUNTIME_DEFAULT_TIMEOUT_MS = DEFAULT_TIMEOUT_MS;
export { contextSizeForSlots };

// The outcome -> wire shape for one probe rung, used by BOTH the live per-rung
// tick and the final result post. Lives here rather than in index.ts purely so
// it can be unit-tested: a field that exists on the outcome but is missing
// from this mapping is computed, logged, and then silently thrown away, which
// is exactly how pp/ttft/slope reached production as a column of em-dashes.
export function toProbeAttemptReport(attempt: ProbeAttemptOutcome): ProbeAttemptReport {
  return {
    candidate_ctx: attempt.candidateCtx,
    ok: attempt.ok,
    oom: attempt.oom,
    spill: attempt.spill,
    ngl: attempt.ngl,
    vram_peak_mib: attempt.vramPeakMib,
    gen_tps: attempt.genTps,
    vram_needed_mib: attempt.vramNeededMib,
    vram_free_mib: attempt.vramFreeMib,
    ram_needed_mib: attempt.ramNeededMib,
    ram_free_mib: attempt.ramFreeMib,
    ram_peak_mib: attempt.ramPeakMib,
    vram_process_peak_mib: attempt.vramProcessPeakMib,
    ram_total_peak_mib: attempt.ramTotalPeakMib,
    vram_shared_peak_mib: attempt.vramSharedPeakMib,
    vram_shared_total_peak_mib: attempt.vramSharedTotalPeakMib,
    vram_claimed_peak_mib: attempt.vramClaimedPeakMib,
    gpu_buffers_mib: attempt.gpuBuffersMib,
    gpu_in_system_ram_mib: attempt.gpuInSystemRamMib,
    gpu_spill_jitter_mib: attempt.gpuSpillJitterMib,
    list_devices_free_mib: attempt.listDevicesFreeMib,
    load_kind: attempt.loadKind,
    claim_fits_free: attempt.claimFitsFree,
    spill_ready_mib: attempt.spillReadyMib,
    spill_ready_jitter_mib: attempt.spillReadyJitterMib,
    spill_work_mib: attempt.spillWorkMib,
    spill_work_jitter_mib: attempt.spillWorkJitterMib,
    spill_method: attempt.spillMethod,
    spill_shared_growth_mib: attempt.spillSharedGrowthMib,
    spill_unlanded_growth_mib: attempt.spillUnlandedGrowthMib,
    ladder_ngl_max: attempt.ladderNglMax,
    ladder_max_ctx: attempt.ladderMaxCtx,
    host_backed_fail: attempt.hostBackedFailCause,
    pp_tps: attempt.ppTps,
    ttft_ms: attempt.ttftMs,
    prefill_cliff: attempt.prefillCliff,
    error: attempt.error,
    reused_from_run_id: attempt.reusedFromRunId,
    vram_discrepancy: attempt.vramDiscrepancy,
    gpu_layers_resident_est: attempt.gpuLayersResidentEst,
    gpu_layers_resident_exact: attempt.gpuLayersResidentExact,
  };
}
