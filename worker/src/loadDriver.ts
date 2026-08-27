// BENCHMARKING_PLAN_V8.md N5's extraction, and the substrate N1's curve
// choreography runs on. Deliberately a separate module from serverBench.ts:
// that file drives MTP speculative decoding and hardcodes --spec-type
// draft-mtp, while everything here is the {engine:"server", spec:"off"} path.
//
// "This extraction is a prerequisite regardless of N5: the streaming client
// must not grow into it ad hoc."
//
// The I/O-free parts (argument building, slot sizing, percentile summaries,
// the eviction detector, the choreography sequencer) are exported separately
// from the spawn/HTTP parts so they can be tested without a llama-server.

import type { SweepItem } from "../../shared/sweep.js";
import type { CaveatFlag } from "../../shared/types.js";

// Padding above the per-slot demand so llama-server's own context accounting
// (special tokens, internal bookkeeping) never rejects a request sitting
// exactly at the context-size edge. Same constant serverBench.ts uses.
export const CONTEXT_MARGIN = 256;

export const FILLER_TOKEN_RANGE_START = 100;
export const FILLER_TOKEN_RANGE_SIZE = 500;

// Slot accounting (N1/N5, stated identically in both): per-slot demand is the
// full prompt plus the generation budget, and -c covers every slot. At
// parallel 1 this equals the existing sizing.
export function contextSizeForSlots(input: {
  slots: number;
  nDepth?: number;
  nPrompt: number;
  nGen: number;
  margin?: number;
}): number {
  const slots = Math.max(1, input.slots);
  const perSlot = (input.nDepth ?? 0) + input.nPrompt + input.nGen + (input.margin ?? CONTEXT_MARGIN);
  return slots * perSlot;
}

export interface ServerArgsInput {
  modelPath: string;
  port: number;
  item: Pick<
    SweepItem,
    | "threads"
    | "n_gpu_layers"
    | "batch_size"
    | "ubatch_size"
    | "cache_type_k"
    | "cache_type_v"
    | "flash_attn"
    | "n_cpu_moe"
    | "n_prompt"
    | "n_gen"
  >;
  slots: number;
  mainGpu?: number;
  /** §0.7 probe result -- when false the flag is left off and its row flags instead. */
  supportsNoContextShift?: boolean;
  contextSizeOverride?: number;
}

// The {engine:"server", spec:"off"} argument set. No --spec-type here at all:
// §0.2's legal pairs make speculation a different engine pair, and mixing the
// two behind one code path is exactly what "ships as separate code paths"
// forbids.
export function buildServerArgs(input: ServerArgsInput): string[] {
  const { item } = input;
  const contextSize =
    input.contextSizeOverride ??
    contextSizeForSlots({ slots: input.slots, nPrompt: item.n_prompt, nGen: item.n_gen });
  const args = [
    "-m",
    input.modelPath,
    "-t",
    String(item.threads),
    "-ngl",
    String(item.n_gpu_layers),
    "-b",
    String(item.batch_size),
    "-ub",
    String(item.ubatch_size),
    "-ctk",
    item.cache_type_k,
    "-ctv",
    item.cache_type_v,
    "-fa",
    item.flash_attn,
    "-c",
    String(contextSize),
    "--port",
    String(input.port),
    "--host",
    "127.0.0.1",
    "--parallel",
    String(Math.max(1, input.slots)),
    "--metrics",
    "--verbosity",
    "4",
  ];
  if (input.mainGpu != null) args.push("-sm", "none", "-mg", String(input.mainGpu));
  if (item.n_cpu_moe > 0) args.push("--n-cpu-moe", String(item.n_cpu_moe));
  // Probed, never assumed (§0.7): an unsupported flag disables its behavior
  // rather than failing the item, and the row carries a context_shift flag if
  // the logs then show a shift happened anyway.
  if (input.supportsNoContextShift) args.push("--no-context-shift");
  return args;
}

// A synthetic prompt of exactly n tokens, sent pre-tokenized so the count is
// exact rather than tokenizer-dependent. `nonce` shifts the sequence: N5's
// concurrent streams each carry a DISTINCT suffix so the prefix cache cannot
// dedupe them (otherwise later streams read artificially fast against a warm
// prefix), and N1's warm/discard request uses a nonce prompt distinct from
// the measured one so it never seeds the measured prefix into the cache.
export function buildPromptTokens(tokenCount: number, offset = 0, nonce = 0): number[] {
  const tokens: number[] = [];
  for (let i = 0; i < tokenCount; i++) {
    tokens.push(FILLER_TOKEN_RANGE_START + offset + ((i + nonce * 7919) % FILLER_TOKEN_RANGE_SIZE));
  }
  return tokens;
}

// --- Streaming measurement --------------------------------------------------

export interface StreamSample {
  /** Milliseconds from request send to FIRST streamed chunk. */
  ttftMs: number;
  /** Milliseconds from request send to the final chunk. */
  e2eMs: number;
  /** Tokens the server reports it generated. */
  tokensPredicted: number;
  /** Prompt tokens the server says it PREFILLED. 0 on a genuine cache hit. */
  promptN: number;
  /** llama-server's own prefill timing, when it reported one. */
  promptMs: number | null;
  /** Which slot/stream produced this sample (N5). */
  slot: number;
}

export function percentile(values: number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil(fraction * sorted.length);
  return sorted[Math.max(0, Math.min(sorted.length - 1, rank - 1))];
}

export interface StreamSummary {
  ttftP50Ms: number | null;
  ttftP95Ms: number | null;
  ttftN: number;
  e2eMeanMs: number | null;
  /** Aggregate generated tokens per second across every stream in the batch. */
  aggregateTps: number | null;
  /** Mean per-stream generated tokens per second. */
  perStreamTps: number | null;
}

// TTFT p50/p95 come from RAW per-stream samples, never from a server-reported
// aggregate -- that is the whole point of driving the requests ourselves.
export function summarizeStreams(samples: StreamSample[]): StreamSummary {
  if (samples.length === 0) {
    return { ttftP50Ms: null, ttftP95Ms: null, ttftN: 0, e2eMeanMs: null, aggregateTps: null, perStreamTps: null };
  }
  const ttfts = samples.map((s) => s.ttftMs);
  const e2es = samples.map((s) => s.e2eMs);
  const perStream = samples
    .filter((s) => s.e2eMs > s.ttftMs && s.tokensPredicted > 0)
    .map((s) => (s.tokensPredicted / (s.e2eMs - s.ttftMs)) * 1000);
  // Aggregate throughput is what the SERVER delivered in the wall-clock span
  // the batch occupied -- the sum of per-stream rates, since the streams ran
  // simultaneously.
  const aggregate = perStream.length > 0 ? perStream.reduce((a, b) => a + b, 0) : null;
  return {
    ttftP50Ms: percentile(ttfts, 0.5),
    ttftP95Ms: percentile(ttfts, 0.95),
    ttftN: samples.length,
    e2eMeanMs: e2es.reduce((a, b) => a + b, 0) / e2es.length,
    aggregateTps: aggregate,
    perStreamTps: perStream.length > 0 ? aggregate! / perStream.length : null,
  };
}

// --- N1: the eviction detector ----------------------------------------------

export type RequestClass = "warm_discard" | "cold_timed" | "warm_repeat";

// "Any WARM REPEAT (class 3) whose response reports timings.prompt_n > 0
// re-prefilled -- the cache did not hold." The cold request of class 2
// legitimately reports the full prompt and is NEVER evaluated by this
// detector; scoping it to warm repeats is the whole correctness of the rule.
export function detectCacheEviction(
  samples: { requestClass: RequestClass; promptN: number }[]
): { evicted: boolean; offendingRepeats: number } {
  const offenders = samples.filter((s) => s.requestClass === "warm_repeat" && s.promptN > 0);
  return { evicted: offenders.length > 0, offendingRepeats: offenders.length };
}

// llama.cpp logs a context shift when it slides the KV window. If the binary
// has no --no-context-shift to suppress it, the row is flagged: shifted
// contexts silently corrupt TTFT comparability.
const CONTEXT_SHIFT_LOG_RE = /context shift|shifting kv cache|slot context shift/i;

export function sawContextShift(serverLog: string): boolean {
  return CONTEXT_SHIFT_LOG_RE.test(serverLog);
}

export function curveCaveatFlags(input: {
  samples: { requestClass: RequestClass; promptN: number }[];
  serverLog: string;
  supportsNoContextShift: boolean;
}): CaveatFlag[] {
  const flags: CaveatFlag[] = [];
  if (detectCacheEviction(input.samples).evicted) flags.push("cache_evicted");
  if (!input.supportsNoContextShift && sawContextShift(input.serverLog)) flags.push("context_shift");
  return flags;
}

// --- N1: the choreography ---------------------------------------------------

export interface CurveRequestPlanStep {
  requestClass: RequestClass;
  /** Prompt token count for this request. */
  promptTokens: number;
  nPredict: number;
  /** Whether this request opts into prefix-cache reuse. */
  cachePrompt: boolean;
  /** Distinct nonce -- the warm/discard prompt must not seed the measured prefix. */
  nonce: number;
  /** Excluded from statistics by construction. */
  countsTowardStatistics: boolean;
}

// Three request classes, never averaged together:
//   1. warm/discard -- tiny n_predict on a SHORT NONCE PROMPT distinct from
//      the measured one, absorbing CUDA-graph capture / pipeline compile
//      without seeding the measured prefix into the cache.
//   2. cold timed prefill, ONE per point -- full prompt, stream, n_predict 1,
//      ignore_eos. First-chunk arrival IS the TTFT data point, and the same
//      response's timings.prompt_ms / prompt_n is the point's pp value.
//   3. warm repeats x (repeats - 1) -- identical prompt with cache_prompt,
//      which is where generation/E2E statistics come from.
export function planCurvePoint(input: { promptTokens: number; nGen: number; repeats: number }): CurveRequestPlanStep[] {
  const repeats = Math.max(1, input.repeats);
  const steps: CurveRequestPlanStep[] = [
    {
      requestClass: "warm_discard",
      promptTokens: 32,
      nPredict: 8,
      cachePrompt: false,
      nonce: 1,
      countsTowardStatistics: false,
    },
    {
      requestClass: "cold_timed",
      promptTokens: input.promptTokens,
      nPredict: 1,
      cachePrompt: false,
      nonce: 0,
      countsTowardStatistics: true,
    },
  ];
  for (let i = 0; i < repeats - 1; i++) {
    steps.push({
      requestClass: "warm_repeat",
      promptTokens: input.promptTokens,
      nPredict: input.nGen,
      cachePrompt: true,
      nonce: 0,
      countsTowardStatistics: true,
    });
  }
  return steps;
}

// --- N5: the knee ladder ----------------------------------------------------

export const DEFAULT_KNEE_SLOTS = [1, 2, 4, 8] as const;

export interface ConcurrentRequestPlanStep {
  slot: number;
  promptTokens: number;
  nPredict: number;
  /** Distinct per stream so the prefix cache cannot dedupe concurrent requests. */
  nonce: number;
}

export function planConcurrentBatch(input: {
  slots: number;
  promptTokens: number;
  nGen: number;
}): ConcurrentRequestPlanStep[] {
  return Array.from({ length: Math.max(1, input.slots) }, (_, slot) => ({
    slot,
    promptTokens: input.promptTokens,
    nPredict: input.nGen,
    // slot + 1 so no stream ever gets nonce 0 (the measured curve prompt).
    nonce: slot + 1,
  }));
}
