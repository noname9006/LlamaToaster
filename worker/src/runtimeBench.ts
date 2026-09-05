// BENCHMARKING_PLAN_V8.md N1 (context curves), N2 (usable-config probe) and
// N5 (concurrency knee): the {engine:"server", spec:"off"} execution paths.
//
// Deliberately separate from serverBench.ts, which drives MTP speculative
// decoding: "N1's server-path choreography ships as separate code paths; its
// warm-cache repeats may not silently change ordinary runtime runs' reuse
// semantics." The pure planning/summarizing logic lives in loadDriver.ts and
// is unit-tested there; this file is the process and HTTP plumbing over it.

import { spawn, type ChildProcess } from "node:child_process";
import type { IngestResultInput } from "../../shared/types.js";
import { CURVE_METHOD_VERSION, SERVER_METHOD_VERSION, type CaveatFlag } from "../../shared/types.js";
import type { SweepItem } from "../../shared/sweep.js";
import { isVramDiscrepancy } from "../../shared/vramEstimate.js";
import {
  collapseTensorLoadSpam,
  parseModelBufferSizes,
  parseOffloadLayers,
  type BenchLogger,
  type BenchResult,
} from "./bench.js";
import { buildPromptTokens, fetchFillerBlocks, type FillerBlocks } from "./fillerPrompt.js";
import {
  buildServerArgs,
  contextSizeForSlots,
  curveCaveatFlags,
  planConcurrentBatch,
  planCurvePoint,
  summarizeStreams,
  type RequestClass,
  type StreamSample,
} from "./loadDriver.js";
import { supportsFlag } from "./binary-probe.js";

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
  log?: BenchLogger;
  onSpawn?: (proc: ChildProcess) => void;
  spawnFn?: SpawnFn;
}

function defaultSpawn(path: string, args: string[]): ChildProcess {
  return spawn(path, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
}

export async function spawnRuntimeServer(input: SpawnRuntimeServerInput): Promise<RuntimeServerHandle> {
  // §0.7 -- probed once per binary identity. An unsupported flag disables its
  // behavior rather than failing the item; the row then carries a
  // context_shift caveat if the logs show a shift happened anyway.
  const supportsNoContextShift = await supportsFlag(input.llamaServerPath, "--no-context-shift").catch(() => false);
  const args = buildServerArgs({
    modelPath: input.modelPath,
    port: input.port,
    item: input.item,
    slots: input.slots,
    mainGpu: input.mainGpu,
    supportsNoContextShift,
    contextSizeOverride: input.contextSizeOverride,
  });
  input.log?.info(`llama-server ${args.join(" ")}`);
  const proc = (input.spawnFn ?? defaultSpawn)(input.llamaServerPath, args);
  input.onSpawn?.(proc);

  let stderr = "";
  proc.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString();
  });
  proc.stdout?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString();
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
   * retry below, and only for a model that cannot otherwise produce a parseable
   * response at all -- constrained sampling has its own cost, so any row
   * measured this way is flagged grammar_constrained.
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
      "still yields a reading -- the row will be flagged grammar_constrained"
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
  serverLog: string;
  supportsNoContextShift: boolean;
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
  let grammarConstrained = false;
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
    grammarConstrained ||= outcome.grammarConstrained;
    samples.push({ ...outcome.sample, requestClass: step.requestClass });
  }

  const cold = samples.find((s) => s.requestClass === "cold_timed");
  const warm = samples.filter((s) => s.requestClass === "warm_repeat");
  const caveats = curveCaveatFlags({
    samples: samples.map((s) => ({ requestClass: s.requestClass, promptN: s.promptN })),
    serverLog: input.serverLog,
    supportsNoContextShift: input.supportsNoContextShift,
  });
  if (grammarConstrained) caveats.push("grammar_constrained");

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
    caveat_flags: caveats,
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
        ? `${evictedCount} warm repeat(s) re-prefilled the prompt: the prefix cache did not hold, so this point is flagged cache_evicted and drops out of the curve`
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
    let grammarConstrained = false;
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
      if (batch.some((b) => b.grammarConstrained)) grammarConstrained = true;
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
      caveat_flags: grammarConstrained ? (["grammar_constrained"] as CaveatFlag[]) : undefined,
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
  vramPeakMib: number | null;
  /** The probe's own RSS peak (MemorySampler.stop()'s ram_peak_mib) -- the
   * real measured RAM usage, alongside vramPeakMib. */
  ramPeakMib?: number | null;
  /** The probe's own peak system-RAM-backed GPU allocation (MemorySampler's
   * vram_process_shared_peak_mib -- vram.ts's GpuMemoryReading.processShared:
   * Windows WDDM "Shared Usage" or Linux amdgpu GTT). The DIRECT measured
   * answer to "how much silently spilled into system RAM instead of erroring"
   * -- vramDiscrepancy above only ever INFERS that from a needed-vs-peak gap.
   * Null wherever no such counter/file exists at all (not a measured 0). */
  vramSharedPeakMib?: number | null;
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
  /** True when this rung's observed VRAM peak came in far below what
   * computeDualPoolFit predicted the requested ngl needs -- the same
   * silent-sysmem-fallback signature shared/vramEstimate.ts's top comment
   * documents (confirmed on both NVIDIA/CUDA and AMD/Vulkan): llama.cpp's
   * "offloaded X/Y" claim only reflects buffer assignment, never residency,
   * and the OS can back an oversubscribed allocation with system RAM instead
   * of erroring. Set by probeSucceeded below; worker/src/index.ts's ladder
   * loop turns it into a warn/retry/fail outcome per vramDiscrepancyPolicy,
   * same policy the sweep path already applies. */
  vramDiscrepancy?: boolean;
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

/** Gen tok/s floor -- excludes swap-thrash "success". */
export const PROBE_MIN_GEN_TPS = 1;
export const PROBE_GEN_TOKENS = 256;

// The ladder that used to live here -- one context axis, x0.75 on failure and
// x1.33 on a roomy success, three loads max -- has been replaced by
// shared/probeLadder.ts, which searches placement as well as context and is
// shared with the client so the two cannot disagree about what a legal
// context is. Only the per-rung success rule below stayed behind, because it
// is about one load's verdict rather than about the search.

// A probe's own success rule, kept next to the ladder that consumes it: no
// OOM, no spill (vram_peak within total), gen tok/s above the floor, and no
// VRAM discrepancy severe enough to fail under the operator's policy.
//
// vramDiscrepancy is reported as a raw fact regardless of policy -- it does
// NOT by itself flip `ok` here. worker/src/index.ts's ladder loop is what
// turns it into warn/retry/fail (via vram-policy.ts's
// resolveVramDiscrepancyAction, the SAME function/policy the sweep path
// already uses), the same separation of "detect" from "decide" that path
// keeps. Unlike the sweep path, there's no llama.cpp post-allocation
// buffer-size report available here (that parsing is llama-bench-log-specific,
// see bench.ts's parseModelBufferSizes) -- the vram_peak-vs-estimate signal is
// the only one a probe has, so the ladder loop treats it as sufficient on its
// own to trigger the policy's harder actions, not gated behind the sweep
// path's stricter "exactly 0 bytes resident" bar.
export function probeSucceeded(input: {
  oom: boolean;
  vramPeakMib: number | null;
  gpuTotalMib: number | null;
  genTps: number | null;
  // The rung's requested layer count and computeDualPoolFit's own predicted
  // GPU need for it (estimateProbeMemoryNeed's vramMib) -- both null-safe:
  // ngl<=0 or a missing estimate simply never flags a discrepancy (nothing to
  // compare against).
  ngl: number;
  estimatedVramMib: number | null;
}): { ok: boolean; spill: boolean; vramDiscrepancy: boolean; reason: string | null } {
  if (input.oom) return { ok: false, spill: false, vramDiscrepancy: false, reason: "out of memory at this context" };
  const spill =
    input.vramPeakMib != null && input.gpuTotalMib != null && input.vramPeakMib > input.gpuTotalMib;
  if (spill) {
    return { ok: false, spill: true, vramDiscrepancy: false, reason: "the allocation spilled past this adapter's VRAM total" };
  }
  const vramDiscrepancy =
    input.ngl > 0 &&
    input.estimatedVramMib != null &&
    input.vramPeakMib != null &&
    isVramDiscrepancy(input.estimatedVramMib, input.vramPeakMib);
  if (input.genTps == null || input.genTps < PROBE_MIN_GEN_TPS) {
    return {
      ok: false,
      spill: false,
      vramDiscrepancy,
      reason: `generation ran at ${input.genTps == null ? "an unmeasurable rate" : `${input.genTps.toFixed(2)} tok/s`}, below the ${PROBE_MIN_GEN_TPS} tok/s floor -- loading is not the same as usable`,
    };
  }
  return { ok: true, spill: false, vramDiscrepancy, reason: null };
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

export const RUNTIME_DEFAULT_TIMEOUT_MS = DEFAULT_TIMEOUT_MS;
export { contextSizeForSlots };
