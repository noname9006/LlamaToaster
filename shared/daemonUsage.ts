// What the served model is actually being used for, from one row per request.
// Pure, so the page, the server and the later optimization loop all read the same
// analysis. Everything here is a measured fact about past traffic: nothing predicts
// that a change would make the model faster.

export type DaemonEndpoint = "chat" | "completions" | "responses" | "messages" | "embeddings" | "rerank";

export interface DaemonUsageRow {
  /** Generated on the worker: the batch is resent until acknowledged. */
  id: string;
  at: number;
  model_id: string;
  load_hash: string;
  endpoint: DaemonEndpoint;
  stream: boolean;
  status: number;
  /** The client went away before the response finished. */
  cancelled: boolean;
  /** Prompt tokens actually evaluated. Null where the response has no such figure. */
  processed_tokens: number | null;
  /** Prompt tokens served from the cache. */
  cached_tokens: number | null;
  generated_tokens: number | null;
  prompt_ms: number | null;
  predicted_ms: number | null;
  total_ms: number;
  /** The most requests in flight at any point during this one, itself included. */
  concurrency: number;
  /** A viewer had the page open, so the Windows GPU counter query was running. */
  watched: boolean;
}

// Below this, prefill speed is mostly noise: on one model 6 tokens read 33 t/s
// and 831 read 701 t/s. Generation is steadier but a handful of tokens still
// measures little beyond the first decode.
export const MIN_PP_TOKENS = 32;
export const MIN_TG_TOKENS = 8;
const MIN_PROFILE_REQUESTS = 20;

export function contextUsed(r: Pick<DaemonUsageRow, "processed_tokens" | "cached_tokens">): number | null {
  if (r.processed_tokens == null && r.cached_tokens == null) return null;
  return (r.processed_tokens ?? 0) + (r.cached_tokens ?? 0);
}

export function ppSpeed(r: DaemonUsageRow): number | null {
  if (r.processed_tokens == null || r.prompt_ms == null || r.processed_tokens < MIN_PP_TOKENS || r.prompt_ms <= 0) return null;
  return r.processed_tokens / (r.prompt_ms / 1000);
}

export function tgSpeed(r: DaemonUsageRow): number | null {
  if (r.generated_tokens == null || r.predicted_ms == null || r.generated_tokens < MIN_TG_TOKENS || r.predicted_ms <= 0) return null;
  return r.generated_tokens / (r.predicted_ms / 1000);
}

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  const rank = (sorted.length - 1) * p;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

function percentiles(values: (number | null)[]): { p50: number | null; p95: number | null; max: number | null } {
  const sorted = values.filter((v): v is number => v != null).sort((a, b) => a - b);
  return { p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), max: sorted.length ? sorted[sorted.length - 1] : null };
}

export interface SpeedCurvePoint {
  /** Lower edge of the bucket, in tokens of context used. */
  from: number;
  to: number;
  pp: number | null;
  ppCount: number;
  tg: number | null;
  tgCount: number;
}

/** Median speed per power-of-two bucket of context used. */
export function speedCurve(rows: DaemonUsageRow[]): SpeedCurvePoint[] {
  const buckets = new Map<number, { pp: number[]; tg: number[] }>();
  for (const r of rows) {
    const ctx = contextUsed(r);
    if (ctx == null || ctx <= 0) continue;
    const exponent = Math.floor(Math.log2(ctx));
    const bucket = buckets.get(exponent) ?? { pp: [], tg: [] };
    const pp = ppSpeed(r);
    const tg = tgSpeed(r);
    if (pp != null) bucket.pp.push(pp);
    if (tg != null) bucket.tg.push(tg);
    buckets.set(exponent, bucket);
  }
  return [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .filter(([, b]) => b.pp.length > 0 || b.tg.length > 0)
    .map(([exponent, b]) => ({
      from: 2 ** exponent,
      to: 2 ** (exponent + 1),
      pp: percentile([...b.pp].sort((x, y) => x - y), 0.5),
      ppCount: b.pp.length,
      tg: percentile([...b.tg].sort((x, y) => x - y), 0.5),
      tgCount: b.tg.length,
    }));
}

export type UsageShape = "agent_loop" | "cold_prompts" | "short_chat" | "long_generation" | "batch" | "mixed";

export const USAGE_SHAPE_LABELS: Record<UsageShape, { title: string; detail: string }> = {
  agent_loop: { title: "Agent loop", detail: "Long prompts, mostly served from the cache. Each turn evaluates only what changed." },
  cold_prompts: { title: "Cold long prompts", detail: "Long prompts evaluated from scratch. Prefill is most of the wait." },
  short_chat: { title: "Short chat", detail: "Short prompts and short answers." },
  long_generation: { title: "Long generation", detail: "Short prompts and long answers. Generation is most of the wait." },
  batch: { title: "Parallel batch", detail: "Several requests at once, sharing throughput across slots." },
  mixed: { title: "Mixed", detail: "No single pattern dominates." },
};

/** The numbers a later optimization loop would size a speed run from. */
export interface UsageWorkload {
  prompt_p50: number;
  prompt_p95: number;
  generated_p50: number;
  generated_p95: number;
  cache_share: number;
  concurrency_p95: number;
}

export interface UsageObservation {
  kind: "unused_slots" | "context_headroom" | "context_tight" | "cold_prompts" | "rejected" | "little_data";
  text: string;
}

export interface UsageProfile {
  requests: number;
  errors: number;
  cancelled: number;
  firstAt: number | null;
  lastAt: number | null;
  tokens: { processed: number; cached: number; generated: number };
  /** Share of prompt tokens the cache served, 0..1. Null with no prompt counts. */
  cacheShare: number | null;
  promptTokens: { p50: number | null; p95: number | null; max: number | null };
  generatedTokens: { p50: number | null; p95: number | null; max: number | null };
  concurrency: { p95: number | null; max: number | null };
  /** Total wait split by what it went on. Null with no timed rows. */
  timeShare: { prefill: number; generation: number; other: number } | null;
  speed: { pp: number | null; tg: number | null };
  /** Requests with no speed readings at all (/v1/messages, unstreamed /v1/responses). */
  untimedRequests: number;
  shape: UsageShape | null;
  workload: UsageWorkload | null;
  observations: UsageObservation[];
}

export interface ProfileContext {
  /** Context size of the loaded model, when known. */
  nCtx?: number | null;
  /** Server slots of the loaded model, when known. */
  slots?: number | null;
  /** KV cache size of the loaded model, when known. */
  kvMib?: number | null;
}

const LONG_PROMPT = 2048;
const CACHED_PROMPT = 512;
const LONG_ANSWER = 256;

function classify(input: { cacheShare: number | null; promptP50: number; generatedP50: number; concurrencyP95: number }): UsageShape {
  if (input.concurrencyP95 >= 2) return "batch";
  // A conversation that keeps growing, however short it still is: live, a loop at
  // 1.8k tokens and 96% cached read as "short chat" against the long-prompt bar alone.
  if (input.promptP50 >= CACHED_PROMPT && (input.cacheShare ?? 0) >= 0.6) return "agent_loop";
  if (input.promptP50 >= LONG_PROMPT) {
    if ((input.cacheShare ?? 0) <= 0.3) return "cold_prompts";
    return "mixed";
  }
  return input.generatedP50 >= LONG_ANSWER ? "long_generation" : "short_chat";
}

export function summarizeUsage(rows: DaemonUsageRow[], ctx: ProfileContext = {}): UsageProfile {
  const served = rows.filter((r) => r.status >= 200 && r.status < 300);
  let processed = 0;
  let cached = 0;
  let generated = 0;
  let prefillMs = 0;
  let generationMs = 0;
  let timedTotalMs = 0;
  let ppTokens = 0;
  let ppMs = 0;
  let tgTokens = 0;
  let tgMs = 0;
  let untimed = 0;
  let coldLarge = 0;
  for (const r of served) {
    processed += r.processed_tokens ?? 0;
    cached += r.cached_tokens ?? 0;
    generated += r.generated_tokens ?? 0;
    if (r.prompt_ms == null && r.predicted_ms == null) {
      // Only a response that finished, on an endpoint that could have reported timings.
      // A cancelled one never got to, and embeddings generate nothing to time.
      if (!r.cancelled && r.endpoint !== "embeddings" && r.endpoint !== "rerank") untimed++;
    } else {
      prefillMs += r.prompt_ms ?? 0;
      generationMs += r.predicted_ms ?? 0;
      timedTotalMs += r.total_ms;
    }
    if (ppSpeed(r) != null) {
      ppTokens += r.processed_tokens as number;
      ppMs += r.prompt_ms as number;
    }
    if (tgSpeed(r) != null) {
      tgTokens += r.generated_tokens as number;
      tgMs += r.predicted_ms as number;
    }
    const total = contextUsed(r);
    if (total != null && total >= LONG_PROMPT && (r.cached_tokens ?? 0) / total < 0.1) coldLarge++;
  }
  const promptSeen = processed + cached > 0;
  const cacheShare = promptSeen ? cached / (processed + cached) : null;
  const promptTokens = percentiles(served.map(contextUsed));
  const generatedTokens = percentiles(served.map((r) => r.generated_tokens));
  const concurrency = percentiles(served.map((r) => r.concurrency));
  const timeShare =
    timedTotalMs > 0
      ? {
          prefill: Math.min(1, prefillMs / timedTotalMs),
          generation: Math.min(1, generationMs / timedTotalMs),
          other: Math.max(0, 1 - (prefillMs + generationMs) / timedTotalMs),
        }
      : null;

  const enough = served.length >= MIN_PROFILE_REQUESTS && promptTokens.p50 != null && promptTokens.p95 != null;
  const workload: UsageWorkload | null = enough
    ? {
        prompt_p50: Math.round(promptTokens.p50 as number),
        prompt_p95: Math.round(promptTokens.p95 as number),
        generated_p50: Math.round(generatedTokens.p50 ?? 0),
        generated_p95: Math.round(generatedTokens.p95 ?? 0),
        cache_share: cacheShare ?? 0,
        concurrency_p95: Math.round(concurrency.p95 ?? 1),
      }
    : null;
  const shape: UsageShape | null = workload
    ? classify({ cacheShare, promptP50: workload.prompt_p50, generatedP50: workload.generated_p50, concurrencyP95: workload.concurrency_p95 })
    : null;

  const observations: UsageObservation[] = [];
  const rejected = rows.filter((r) => r.status === 400).length;
  if (rejected > 0) {
    observations.push({ kind: "rejected", text: `${rejected} request${rejected === 1 ? " was" : "s were"} refused with HTTP 400. A prompt larger than the context is refused this way.` });
  }
  if (!enough && rows.length > 0) {
    observations.push({ kind: "little_data", text: `${served.length} of ${MIN_PROFILE_REQUESTS} requests needed before this describes a pattern.` });
  }
  if (enough) {
    const peak = concurrency.max ?? 1;
    if (ctx.slots != null && ctx.slots > 1 && peak < ctx.slots) {
      observations.push({ kind: "unused_slots", text: `The model has ${ctx.slots} slots and traffic never used more than ${peak} at once.` });
    }
    const p95 = (promptTokens.p95 ?? 0) + (generatedTokens.p95 ?? 0);
    if (ctx.nCtx != null && ctx.nCtx > 0) {
      if (p95 >= ctx.nCtx * 0.9) {
        observations.push({ kind: "context_tight", text: `The 95th-percentile request reaches ${Math.round(p95).toLocaleString("en-US")} of ${ctx.nCtx.toLocaleString("en-US")} context tokens.` });
      } else if (promptTokens.max != null && (promptTokens.max + (generatedTokens.max ?? 0)) * 4 <= ctx.nCtx) {
        const kv = ctx.kvMib != null ? ` The KV cache holds ${Math.round(ctx.kvMib).toLocaleString("en-US")} MiB for it.` : "";
        observations.push({ kind: "context_headroom", text: `The largest request used under a quarter of the ${ctx.nCtx.toLocaleString("en-US")}-token context.${kv}` });
      }
    }
    if (coldLarge / served.length >= 0.25 && (cacheShare ?? 0) < 0.6) {
      observations.push({ kind: "cold_prompts", text: `${coldLarge} of ${served.length} requests evaluated a long prompt almost from scratch. A prompt that changes near its start, such as a timestamp in a system message, defeats the cache.` });
    }
  }

  return {
    requests: rows.length,
    // A request cut off by a drain or the client is cancelled, not failed.
    errors: rows.filter((r) => r.status >= 400 && !r.cancelled).length,
    cancelled: rows.filter((r) => r.cancelled).length,
    firstAt: rows.length ? Math.min(...rows.map((r) => r.at)) : null,
    lastAt: rows.length ? Math.max(...rows.map((r) => r.at)) : null,
    tokens: { processed, cached, generated },
    cacheShare,
    promptTokens,
    generatedTokens,
    concurrency: { p95: concurrency.p95, max: concurrency.max },
    timeShare,
    speed: { pp: ppMs > 0 ? ppTokens / (ppMs / 1000) : null, tg: tgMs > 0 ? tgTokens / (tgMs / 1000) : null },
    untimedRequests: untimed,
    shape,
    workload,
    observations,
  };
}
