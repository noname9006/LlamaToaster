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
import { CURVE_METHOD_VERSION, METHOD_VERSION } from "../../shared/types.js";
import type { SweepItem } from "../../shared/sweep.js";
import {
  collapseTensorLoadSpam,
  parseModelBufferSizes,
  parseOffloadLayers,
  type BenchLogger,
  type BenchResult,
} from "./bench.js";
import {
  buildPromptTokens,
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
      try {
        const parsed = JSON.parse(payload) as {
          timings?: { prompt_n?: number; prompt_ms?: number; predicted_n?: number };
          tokens_predicted?: number;
        };
        if (parsed.timings) {
          if (typeof parsed.timings.prompt_n === "number") promptN = parsed.timings.prompt_n;
          if (typeof parsed.timings.prompt_ms === "number") promptMs = parsed.timings.prompt_ms;
        }
        if (typeof parsed.tokens_predicted === "number") tokensPredicted = parsed.tokens_predicted;
      } catch {
        /* a partial SSE frame; the next chunk completes it */
      }
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
  const plan = planCurvePoint({
    promptTokens: input.effectiveCtx,
    nGen: input.nGen,
    repeats: input.repeats,
  });

  const samples: (StreamSample & { requestClass: RequestClass })[] = [];
  for (const step of plan) {
    const sample = await completion({
      port: input.port,
      promptTokens: buildPromptTokens(step.promptTokens, input.promptOffset ?? 0, step.nonce),
      nPredict: step.nPredict,
      cachePrompt: step.cachePrompt,
    });
    samples.push({ ...sample, requestClass: step.requestClass });
  }

  const cold = samples.find((s) => s.requestClass === "cold_timed");
  const warm = samples.filter((s) => s.requestClass === "warm_repeat");
  const caveats = curveCaveatFlags({
    samples: samples.map((s) => ({ requestClass: s.requestClass, promptN: s.promptN })),
    serverLog: input.serverLog,
    supportsNoContextShift: input.supportsNoContextShift,
  });

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
    // Choreographed points stamp METHOD_VERSION 2 -- cold-timed prefill plus
    // warm-repeat statistics is a real semantics change under §0.1, and that
    // stamp is what keeps ordinary runtime rows out of curves.
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
    const batchSamples: StreamSample[] = [];
    for (let repeat = 0; repeat < Math.max(1, input.repeats); repeat++) {
      const plan = planConcurrentBatch({ slots, promptTokens: input.nPrompt, nGen: input.nGen });
      // Simultaneous, not sequential -- that is the whole measurement.
      const batch = await Promise.all(
        plan.map((step) =>
          completion({
            port,
            promptTokens: buildPromptTokens(step.promptTokens, input.promptOffset ?? 0, step.nonce + repeat * 1000),
            nPredict: step.nPredict,
            cachePrompt: false,
            slot: step.slot,
          })
        )
      );
      batchSamples.push(...batch);
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
      method_version: METHOD_VERSION,
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
  /** A short, worker-log-only tail of this attempt's captured process output
   * -- populated only when the failure was NOT classified as OOM, so a
   * genuine startup bug is debuggable from the worker log instead of a bare
   * "exited before it became ready". Never sent to the server: the report's
   * own `error` already carries the short message. */
  stderrTail?: string;
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

// A probe's own success rule, kept next to the ladder that consumes it:
// no OOM, no spill (vram_peak within total), and gen tok/s above the floor.
export function probeSucceeded(input: {
  oom: boolean;
  vramPeakMib: number | null;
  gpuTotalMib: number | null;
  genTps: number | null;
}): { ok: boolean; spill: boolean; reason: string | null } {
  if (input.oom) return { ok: false, spill: false, reason: "out of memory at this context" };
  const spill =
    input.vramPeakMib != null && input.gpuTotalMib != null && input.vramPeakMib > input.gpuTotalMib;
  if (spill) {
    return { ok: false, spill: true, reason: "the allocation spilled past this adapter's VRAM total" };
  }
  if (input.genTps == null || input.genTps < PROBE_MIN_GEN_TPS) {
    return {
      ok: false,
      spill: false,
      reason: `generation ran at ${input.genTps == null ? "an unmeasurable rate" : `${input.genTps.toFixed(2)} tok/s`}, below the ${PROBE_MIN_GEN_TPS} tok/s floor -- loading is not the same as usable`,
    };
  }
  return { ok: true, spill: false, reason: null };
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
