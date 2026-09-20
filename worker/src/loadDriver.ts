// BENCHMARKING_PLAN_V8.md N5's extraction, and the substrate N1's curve
// choreography runs on. Deliberately a separate module from serverBench.ts:
// that file drives MTP speculative decoding and hardcodes --spec-type
// draft-mtp, while everything here is the {engine:"server", spec:"off"} path.
//
// "This extraction is a prerequisite regardless of N5: the streaming client
// must not grow into it ad hoc."
//
// The I/O-free parts (argument building, slot sizing, percentile summaries,
// the choreography sequencer) are exported separately from the spawn/HTTP
// parts so they can be tested without a llama-server.

import type { SweepItem } from "../../shared/sweep.js";

// Padding above the per-slot demand so llama-server's own context accounting
// (special tokens, internal bookkeeping) never rejects a request sitting
// exactly at the context-size edge. Same constant serverBench.ts uses.
export const CONTEXT_MARGIN = 256;

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
  /** §0.7 probe result for --fit -- see buildServerArgs's own comment on why
   * this is forced off rather than left at llama-server's default. Unlike
   * supportsNoContextShift, no row-level caveat exists for "unsupported": a
   * build that doesn't have --fit at all has no auto-adjustment behavior to
   * begin with, so there's nothing left to flag. */
  supportsFit?: boolean;
  /** §0.7 probe result for --mlock -- keeps the model locked in RAM for the
   * run's duration instead of left swappable. No row-level caveat needed:
   * unlike --no-context-shift, there's no observable symptom to flag when a
   * build lacks it. */
  supportsMlock?: boolean;
  contextSizeOverride?: number;
  /** Already probed by the caller -- see spawnRuntimeServer. */
  noMmap?: boolean;
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
    // 5 ("debug"), not 4 ("trace") -- see serverBench.ts's buildArgs for why:
    // only level 5 also prints llama.cpp's own per-layer "assigned to device"
    // ground truth (bench.ts's LAYER_DEVICE_LINE_RE), which is what lets
    // worker/src/index.ts's probe path report an exact resident-layer count
    // instead of a byte-ratio estimate. bench.ts's appendBoundedOutput caps
    // the live capture so level 5's extra per-token trace can't grow it
    // unbounded.
    "--verbosity",
    "5",
  ];
  if (input.mainGpu != null) args.push("-sm", "none", "-mg", String(input.mainGpu));
  if (item.n_cpu_moe > 0) args.push("--n-cpu-moe", String(item.n_cpu_moe));
  // --fit (default "on" on builds that have it) auto-adjusts whichever of
  // -ngl/-c/-ts/-ot were left UNSET to fit free device memory -- it's
  // documented to leave an explicitly-passed value alone. -ngl and -c are
  // always explicit above, but -ot (tensor placement, what --n-cpu-moe is a
  // friendlier alias for) is only ever set when n_cpu_moe>0: at 0, tensor
  // placement is genuinely unset, so a MoE model that doesn't fully fit at
  // the requested -ngl could have --fit silently push some experts to CPU
  // anyway -- exactly the "0 means nothing forced off GPU" guarantee every
  // reading in this app assumes. Forcing it off removes that gap regardless
  // of whether --n-cpu-moe itself would otherwise have shielded -ot from the
  // fitter -- not verified against real source, made moot instead.
  if (input.supportsFit) args.push("--fit", "off");
  // Probed, never assumed (§0.7): an unsupported flag disables its behavior
  // rather than failing the item.
  if (input.supportsNoContextShift) args.push("--no-context-shift");
  if (input.supportsMlock) args.push("--mlock");
  // No --no-mmap by default, unlike the speed-benchmark paths (bench.ts's
  // buildArgs, serverBench.ts's buildArgs): context tests measure spill/growth
  // behavior a real deployment would see, and a real deployment runs with mmap
  // at its default ON. The fill curve IS a speed run, so it opts in.
  if (input.noMmap) args.push("--no-mmap");
  return args;
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
  /**
   * Tokens the server's own decode timer counted, and the ms it says they
   * took -- the generation-only span, which is NOT derivable on this side of
   * the connection: the wall clock can only see "after the first chunk
   * arrived", a window that holds tokensPredicted - 1 tokens at best and
   * nothing at all when only one token was generated. Null when the server
   * reported no such timing (older builds, or a stream that ended early).
   */
  predictedN: number | null;
  predictedMs: number | null;
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

// --- N1: one request's two rates -------------------------------------------

// An absolute ceiling on either side of a pair. Same role and the same
// reasoning as serverBench.ts's constants (the ~1e6 tok/s timer-bug class):
// generous enough that no real load on the hardware this tool targets could
// reach it. Note what it does NOT catch on its own -- the production failure
// that motivated v7 reported 1000 tok/s, three orders of magnitude under the
// tg ceiling, and every reading looked individually "possible". The
// cross-checks below are what catch that class; the ceiling is the backstop.
export const MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND = 200_000;
export const MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND = 5_000;

// Below this many prompt tokens, a prefill reading mostly times first-request
// setup rather than prefill itself -- measured on an RX 6600 XT at 1.3 tok/s
// for a 16-token prompt against ~20 tok/s from 128 tokens up (see the probe's
// PROBE_PROMPT_TOKENS and its commit). Such a pp value is real for what it
// measures but useless as a comparison bound, so the pp-vs-tg cross-check
// below stays out of its way.
export const MIN_PROMPT_TOKENS_FOR_PP_CROSSCHECK = 128;

// How far the server's own decode timer may run ahead of what this side's
// wall clock can corroborate before the reading is called suspect. The wall
// clock sees the span after the FIRST streamed chunk, which holds at most
// tokensPredicted - 1 tokens and may hold fewer still if the transport
// coalesced several frames into that first chunk -- so it is a soft bound,
// not an equality, and the factor is deliberately loose.
const WALL_CLOCK_DISAGREEMENT_FACTOR = 2;

/** Which cross-check a suspect reading failed -- the grouping key for a
 * point's warning, so five flagged repeats read as one counted sentence
 * rather than five near-identical ones. */
export type SuspectKind = "ceiling" | "faster_than_prefill" | "wall_clock";

export interface RateReading {
  value: number;
  /** Kept and reported rather than discarded -- §"suspect readings are kept and flagged". */
  suspect: boolean;
  /** Set together with `reason` whenever suspect; both null when not. */
  kind: SuspectKind | null;
  /** Why it is suspect, with this reading's own numbers in it. */
  reason: string | null;
}

/**
 * Prefill rate from ONE request's own timings. Null when the server reported
 * no usable prefill (a cache hit prefilling nothing, or no timings at all).
 */
export function prefillRate(sample: StreamSample): RateReading | null {
  if (sample.promptMs == null || sample.promptMs <= 0 || sample.promptN <= 0) return null;
  const value = (sample.promptN / sample.promptMs) * 1000;
  if (!Number.isFinite(value)) return null;
  return value > MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND
    ? {
        value,
        suspect: true,
        kind: "ceiling",
        reason: `prefill rate above the ${MAX_PLAUSIBLE_PP_TOKENS_PER_SECOND} tok/s ceiling`,
      }
    : { value, suspect: false, kind: null, reason: null };
}

/**
 * Generation rate from ONE request's own decode timer, cross-checked against
 * that same request's prefill rate and against this side's wall clock.
 *
 * Deliberately NOT computed from (e2eMs - ttftMs): that window starts at the
 * first streamed chunk, so it excludes the first token entirely and is empty
 * when only one token was generated. v5 divided the FULL token count by it
 * and published the result, which is how a CPU load whose prefill ran at 15
 * tok/s came to report 1000 tok/s generation -- one token over a 1 ms window
 * that contained no decoding at all.
 *
 * A single token IS a legitimate sample here, unlike under the old formula:
 * the server's timer measures that token's own decode, not a leftover span.
 */
export function generationRate(sample: StreamSample): RateReading | null {
  if (sample.predictedN == null || sample.predictedMs == null) return null;
  if (sample.predictedN <= 0 || sample.predictedMs <= 0) return null;
  const value = (sample.predictedN / sample.predictedMs) * 1000;
  if (!Number.isFinite(value)) return null;

  if (value > MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND) {
    return {
      value,
      suspect: true,
      kind: "ceiling",
      reason: `generation rate above the ${MAX_PLAUSIBLE_TG_TOKENS_PER_SECOND} tok/s ceiling`,
    };
  }

  // Batched prefill reads the whole prompt in ubatch-sized chunks; decode
  // walks one token at a time through the same weights. Decode faster than
  // prefill, on the same request and the same hardware, means one of the two
  // timers is lying.
  const pp = prefillRate(sample);
  if (pp != null && !pp.suspect && sample.promptN >= MIN_PROMPT_TOKENS_FOR_PP_CROSSCHECK && value > pp.value) {
    return {
      value,
      suspect: true,
      kind: "faster_than_prefill",
      reason: `generation (${value.toFixed(2)} tok/s) read faster than this request's own prefill (${pp.value.toFixed(2)} tok/s)`,
    };
  }

  // Independent corroboration: the span after the first chunk holds at most
  // predictedN - 1 tokens, so the rate it implies is an upper bound the
  // server's own timer should not wildly exceed.
  if (sample.predictedN >= 2 && sample.e2eMs > sample.ttftMs) {
    const wallClock = ((sample.predictedN - 1) / (sample.e2eMs - sample.ttftMs)) * 1000;
    if (wallClock > 0 && value > wallClock * WALL_CLOCK_DISAGREEMENT_FACTOR) {
      return {
        value,
        suspect: true,
        kind: "wall_clock",
        reason: `the server's decode timer (${value.toFixed(2)} tok/s) disagrees with this side's wall clock (at most ${wallClock.toFixed(2)} tok/s)`,
      };
    }
  }

  return { value, suspect: false, kind: null, reason: null };
}

// --- N1: request classes -----------------------------------------------------

// One class only, since v7: every curve request is a cold, cache-free
// measurement of its own. "cold_timed" is kept as the name (rather than
// dropped entirely) because stored samples and the worker's own logs refer to
// it, and because a future class would need to be distinguished FROM it.
export type RequestClass = "cold_timed";

// --- N1: the choreography ---------------------------------------------------

export interface CurveRequestPlanStep {
  requestClass: RequestClass;
  /** Prompt token count for this request. */
  promptTokens: number;
  nPredict: number;
  /** Whether this request opts into prefix-cache reuse. */
  cachePrompt: boolean;
  /** Distinct per repeat, so no two requests share a prefix -- see planCurvePoint. */
  nonce: number;
  /** Excluded from statistics by construction. */
  countsTowardStatistics: boolean;
}

// `repeats` identical-shape requests, every one of them cold:
//   * full prompt, streamed, ignore_eos, n_predict = nGen.
//   * cache_prompt OFF, and a DIFFERENT nonce per repeat, so no request can
//     reuse another's prefix even if llama-server decided to ignore the flag.
//     Belt and braces on purpose: the warm-repeat design this replaced relied
//     on a cache hit it could only detect after the fact (prompt_n > 0), and
//     in production it silently never held -- every repeat re-prefilled while
//     the code went on reporting a "cache-warm" generation rate measured over
//     a window that contained no generation at all.
//   * every repeat therefore yields BOTH rates from its own response timings
//     (pp from prompt_ms/prompt_n, tg from predicted_ms/predicted_n) plus its
//     own TTFT, instead of one pp sample and repeats-1 tg samples.
// Deliberately no throwaway warmup request ahead of the first (removed
// 2026-09-17, was "warm_discard"): a policy call to keep context tests
// consistent -- N2's probe and N5's knee ladder never had a warmup step
// either. Note this makes repeats genuinely expensive at wide contexts: each
// one pays a full prefill. That is what the ETA has always priced
// (pricing.ts prices nPrompt/ppRate * repeats), so estimates get MORE
// accurate here, not less.
export function planCurvePoint(input: { promptTokens: number; nGen: number; repeats: number }): CurveRequestPlanStep[] {
  const repeats = Math.max(1, input.repeats);
  return Array.from({ length: repeats }, (_, i) => ({
    requestClass: "cold_timed" as const,
    promptTokens: input.promptTokens,
    nPredict: input.nGen,
    cachePrompt: false,
    nonce: i,
    countsTowardStatistics: true,
  }));
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
