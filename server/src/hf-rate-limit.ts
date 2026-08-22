// Hugging Face rate-limit handling -- 5-minute fixed windows
//
// Buckets (from https://huggingface.co/docs/hub/rate-limits):
//   - api        : Hub API  (search, tree, model info)  -- http://huggingface.co/api/*
//   - resolvers  : /resolve/ URLs (file content via huggingface.co/<repo>/resolve/…)
//   - pages      : web browsing, not used here
//
// All buckets share a 5-minute FIXED window (w=300), not a sliding window,
// but we track request timestamps as a sliding window to self-throttle before
// HF returns 429.  On 429, HF sends `RateLimit` and `RateLimit-Policy` IETF
// draft headers with `r=remaining` and `t=secondsUntilReset` / `q=quota`.
// We parse those to learn the true tier (anonymous vs FREE vs PRO vs Team …)
// and to sleep the exact reset value instead of a blind 30s.
//
// Tier limits as documented Sept 2025 (updated in this file when they change):
//   Anonymous : api=500, resolvers=3000  per 300s
//   Free user : api=1000, resolvers=5000
//   PRO       : api=2500, resolvers=12000
//   Team      : api=3000, resolvers=20000
//   Enterprise: api=6000, resolvers=50000  (Enterprise Plus higher)

import { log } from "./log.js";

export type RateLimitBucket = "api" | "resolvers";
const WINDOW_MS = 300_000; // 5 minutes -- fixed window on HF side
const SAFETY_MARGIN = 20; // keep N requests headroom before hitting the window limit

// --- token helpers -----------------------------------------------------------

export function getHfToken(): string | undefined {
  // Priority matches HF docs + common variants seen in the wild.
  const raw =
    process.env.HF_TOKEN ??
    process.env.HF_API_KEY ??
    process.env.HUGGING_FACE_HUB_TOKEN ??
    process.env.HUGGINGFACE_TOKEN ??
    process.env.HUGGINGFACE_HUB_TOKEN;
  const trimmed = raw?.trim();
  return trimmed ? trimmed : undefined;
}

export function getHfAuthHeaders(): Record<string, string> {
  const t = getHfToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export function isHfTokenConfigured(): boolean {
  return !!getHfToken();
}

// --- limit defaults ----------------------------------------------------------

// Per user request: use Free tier when HF_TOKEN is set, Anonymous otherwise.
// Free: api 1000 / resolvers 5000 per 5m ; Anonymous: api 500 / resolvers 3000.
// This is intentionally capped to Free even if the token's actual tier is
// higher (PRO/Team etc) – use HF_API_RATE_LIMIT/HF_RESOLVERS_RATE_LIMIT env
// overrides to opt into higher quotas explicitly.
const FREE_LIMITS: Record<RateLimitBucket, number> = { api: 1000, resolvers: 5000 };
const ANON_LIMITS: Record<RateLimitBucket, number> = { api: 500, resolvers: 3000 };

function defaultLimitFor(bucket: RateLimitBucket): number {
  return isHfTokenConfigured() ? FREE_LIMITS[bucket] : ANON_LIMITS[bucket];
}

function capToConfiguredTier(bucket: RateLimitBucket, quota: number): number {
  // Env override is explicit opt-in to a specific ceiling -- it must win
  // over whatever HF's own RateLimit-Policy header reports, not just at
  // bucket-creation time (ensureBucket) but on every subsequent learned
  // quota too. Returning the override itself (not the raw `quota` param)
  // keeps state.limit pinned to it instead of getting clobbered by the
  // next response's header.
  const override = envOverrideFor(bucket);
  if (override != null) return override;
  const cap = isHfTokenConfigured() ? FREE_LIMITS[bucket] : ANON_LIMITS[bucket];
  // Honor platform-health decreases (quota < cap) but never exceed the
  // configured Free/Anonymous ceiling unless env override is set.
  if (quota > cap) {
    log.debug(`[hf-rate-limit] capping learned ${bucket} quota ${quota} -> ${cap} (Free/Anonymous ceiling; set HF_${bucket.toUpperCase()}_RATE_LIMIT to override)`);
    return cap;
  }
  return quota;
}

// Optional explicit override for local tuning / CI memory (never required).
function envOverrideFor(bucket: RateLimitBucket): number | undefined {
  const key = bucket === "api" ? "HF_API_RATE_LIMIT" : "HF_RESOLVERS_RATE_LIMIT";
  const raw = process.env[key];
  if (!raw) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

// --- per-bucket state --------------------------------------------------------

interface BucketState {
  // Effective quota for this window, learned from RateLimit-Policy q=… or fallback default.
  limit: number;
  // Whether limit was learned from server header (authoritative) vs default guess.
  learned: boolean;
  // Last known remaining from RateLimit r=…, used only for logging/throttling heuristics.
  remaining?: number;
  // Absolute reset timestamp (ms) = now + t*1000 from the last 429/200 header.
  resetAtMs?: number;
  // Timestamps (ms) of requests we admitted in this window, used for sliding-window throttling.
  timestamps: number[];
}

const buckets = new Map<RateLimitBucket, BucketState>();

function ensureBucket(bucket: RateLimitBucket): BucketState {
  let s = buckets.get(bucket);
  const override = envOverrideFor(bucket);
  const def = defaultLimitFor(bucket);
  if (!s) {
    // Env override wins outright; otherwise use default guess (which may bump on token presence).
    if (override != null) s = { limit: override, learned: true, timestamps: [] };
    else s = { limit: def, learned: false, timestamps: [] };
    buckets.set(bucket, s);
    return s;
  }
  if (override != null) {
    if (s.limit !== override) {
      s.limit = override;
      s.learned = true;
    }
    return s;
  }
  // If token appeared after bucket was created (e.g. tests or env injected at runtime)
  // and we haven't yet learned a real quota from headers, bump default.
  if (!s.learned && s.limit !== def) {
    const old = s.limit;
    s.limit = def;
    log.debug(`[hf-rate-limit] bumped ${bucket} limit ${old} -> ${def} after token config change`);
  }
  return s;
}

function prune(state: BucketState, now: number): void {
  const cutoff = now - WINDOW_MS;
  state.timestamps = state.timestamps.filter((t) => t > cutoff);
}

// --- header parsing (IETF draft) ---------------------------------------------
//
// Examples from docs / live observations:
//   RateLimit: "api";r=42;t=123
//   RateLimit: "resolvers";r=2990;t=123
//   RateLimit: "\"api|pages|resolvers\";r=42;t=123"  (grouped form observed)
//   RateLimit-Policy: "fixed window";"api";q=1000;w=300
//
// We do tolerant parsing: try bucket-specific form first, then fall back to
// the grouped / generic forms. Headers are case-insensitive (fetch normalizes).

function extractHeader(headers: Headers, name: string): string | null {
  // fetch normalizes to lower-case, but we try both casings for robustness.
  return headers.get(name) ?? headers.get(name.toLowerCase()) ?? headers.get(name.toUpperCase());
}

function parseRateLimitHeaders(bucket: RateLimitBucket, headers: Headers): { remaining?: number; resetSec?: number; quota?: number; windowSec?: number } {
  const out: { remaining?: number; resetSec?: number; quota?: number; windowSec?: number } = {};
  const rl = extractHeader(headers, "ratelimit") ?? extractHeader(headers, "RateLimit");
  const policy = extractHeader(headers, "ratelimit-policy") ?? extractHeader(headers, "RateLimit-Policy");

  const combined = `${rl ?? ""} ${policy ?? ""}`;

  // If header is grouped as "api|pages|resolvers", it applies to api bucket too.
  // We check both the direct bucket mention and the grouped mention.
  const mentionsBucket = (s: string | null) =>
    !!s && (s.includes(`"${bucket}"`) || s.includes(bucket) || (bucket === "api" && s.includes("api|pages|resolvers")) || (bucket === "resolvers" && s.includes("api|pages|resolvers")));

  if (rl) {
    // remaining: r=123 -- only apply if header actually mentions this bucket
    // (otherwise e.g. an "api" r= would incorrectly bleed into "resolvers").
    let rMatch: RegExpMatchArray | null = null;
    let tMatch: RegExpMatchArray | null = null;

    if (mentionsBucket(rl)) {
      // Prefer a group whose textual chunk mentions the bucket. Split by comma into groups first.
      const groups = rl.split(",");
      for (const g of groups) {
        const has = g.includes(`"${bucket}"`) || g.includes(bucket) || g.includes("api|pages|resolvers");
        if (!has) continue;
        const rg = g.match(/r\s*=\s*(\d+)/i);
        const tg = g.match(/t\s*=\s*(\d+)/i);
        if (rg) rMatch = rg;
        if (tg) tMatch = tg;
        if (rg || tg) break;
      }
      // Fallback: global search if group split didn't match (single-group header)
      if (!rMatch) rMatch = rl.match(/r\s*=\s*(\d+)/i);
      if (!tMatch) tMatch = rl.match(/t\s*=\s*(\d+)/i);
      if (rMatch) out.remaining = Number(rMatch[1]);
      if (tMatch) out.resetSec = Number(tMatch[1]);
    } else {
      // Header for a different bucket -- ignore for this bucket.
      // e.g. resolver response shouldn't update api's remaining.
    }
  }

  if (policy) {
    // policy is authoritative for the quota (q) and window (w)
    if (mentionsBucket(policy)) {
      const groups = policy.split(",");
      for (const g of groups) {
        const has = g.includes(`"${bucket}"`) || g.includes(bucket) || g.includes("api|pages|resolvers");
        if (!has) continue;
        const qg = g.match(/q\s*=\s*(\d+)/i);
        const wg = g.match(/w\s*=\s*(\d+)/i);
        if (qg) out.quota = Number(qg[1]);
        if (wg) out.windowSec = Number(wg[1]);
        if (qg || wg) break;
      }
      if (out.quota == null) {
        const qg = policy.match(/q\s*=\s*(\d+)/i);
        if (qg) out.quota = Number(qg[1]);
      }
      if (out.windowSec == null) {
        const wg = policy.match(/w\s*=\s*(\d+)/i);
        if (wg) out.windowSec = Number(wg[1]);
      }
    }
  }

  // Sanity: window should be 300; ignore absurd values
  if (out.windowSec != null && (out.windowSec < 10 || out.windowSec > 3600)) {
    delete out.windowSec;
  }
  if (out.remaining != null && out.remaining < 0) delete out.remaining;
  if (out.quota != null && out.quota <= 0) delete out.quota;
  // Only warn "unparseable" when a header actually claimed to describe THIS
  // bucket -- a header that simply belongs to the other bucket (e.g. a
  // resolvers-only response while parsing for "api") is not a parse
  // failure, it's just not relevant here, and shouldn't be logged as one.
  const relevantToBucket = mentionsBucket(rl) || mentionsBucket(policy);
  if (combined && relevantToBucket && out.remaining == null && out.resetSec == null && out.quota == null) {
    // Header present but unparseable (e.g. future format change) -- log at debug
    log.debug(`[hf-rate-limit] unparseable headers for ${bucket}: rl=${JSON.stringify(rl)} policy=${JSON.stringify(policy)}`);
  }
  return out;
}

// Public: called after every fetch (200 or 429) to learn remaining/reset.
// Quota (q=) is honored only for platform-health decreases or env overrides;
// otherwise we stay at the Free/Anonymous ceiling per user request.
export function recordRateLimit(bucket: RateLimitBucket, headers: Headers): void {
  const state = ensureBucket(bucket);
  const parsed = parseRateLimitHeaders(bucket, headers);
  let changed = false;
  if (parsed.quota != null) {
    const effectiveQuota = capToConfiguredTier(bucket, parsed.quota);
    if (effectiveQuota !== state.limit) {
      const old = state.limit;
      state.limit = effectiveQuota;
      state.learned = true;
      changed = true;
      log.info(`[hf-rate-limit] learned ${bucket} quota q=${parsed.quota}→${effectiveQuota} (was ${old})${parsed.windowSec ? ` w=${parsed.windowSec}s` : ""}`);
    } else {
      state.learned = true;
    }
  }
  if (parsed.remaining != null) {
    state.remaining = parsed.remaining;
  }
  if (parsed.resetSec != null) {
    state.resetAtMs = Date.now() + parsed.resetSec * 1000;
  }
  // Early warning when remaining is low ( <10% or < SAFETY_MARGIN )
  if (state.remaining != null && (state.remaining < SAFETY_MARGIN || state.remaining < state.limit * 0.1)) {
    log.warn(`[hf-rate-limit] ${bucket} low remaining: r=${state.remaining}/${state.limit} t=${parsed.resetSec ?? "?"}s`);
  }
  if (changed) {
    prune(state, Date.now());
  }
}

// How long (seconds) until this bucket's current 5-min window resets according to the last header?
export function getResetSeconds(bucket: RateLimitBucket): number | undefined {
  const s = buckets.get(bucket);
  if (!s?.resetAtMs) return undefined;
  const secs = Math.ceil((s.resetAtMs - Date.now()) / 1000);
  return secs > 0 ? secs : undefined;
}

// --- sliding-window throttling ----------------------------------------------

function jitterMs(): number {
  return 80 + Math.floor(Math.random() * 220); // 80-300ms
}

export async function acquireRateLimitSlot(bucket: RateLimitBucket): Promise<void> {
  const state = ensureBucket(bucket);
  while (true) {
    const now = Date.now();
    prune(state, now);

    // A learned remaining/resetAt describes one specific HF fixed window.
    // Once that window's reset time has passed, HF's own counter has
    // rolled over to a fresh quota -- forget the stale values instead of
    // staying wedged on a "remaining=0" that can never be refreshed
    // (nothing can ever issue the request that would deliver a new header,
    // since that request is exactly what this function is gating).
    if (state.resetAtMs != null && now >= state.resetAtMs) {
      state.remaining = undefined;
      state.resetAtMs = undefined;
    }

    // Respect a recent 429's resetAt before admitting anything.
    if (state.resetAtMs && now < state.resetAtMs && (state.remaining === 0 || state.timestamps.length >= state.limit - SAFETY_MARGIN)) {
      const waitMs = state.resetAtMs - now + jitterMs();
      log.warn(`[hf-rate-limit] ${bucket} in 429 backoff -- sleeping ${Math.ceil(waitMs / 1000)}s until window reset`);
      await sleep(waitMs);
      continue;
    }

    // If server told us remaining===0 (but no t), treat as at limit. No
    // resetAtMs to expire against here (handled above when one is known),
    // so bound the wait and then forget remaining -- otherwise a header
    // that reports r=0 without a t= would wedge this bucket forever, same
    // failure mode as the resetAtMs case above.
    const effectiveRemaining = state.remaining;
    if (effectiveRemaining === 0) {
      const waitMs = state.resetAtMs ? Math.max(0, state.resetAtMs - now) + jitterMs() : 5000 + jitterMs();
      log.warn(`[hf-rate-limit] ${bucket} remaining=0 -- sleeping ${Math.ceil(waitMs / 1000)}s`);
      await sleep(waitMs);
      if (!state.resetAtMs) state.remaining = undefined;
      continue;
    }

    const headroom = state.limit - state.timestamps.length;
    if (headroom > SAFETY_MARGIN || headroom > state.limit * 0.05) {
      // Room available -- admit.
      state.timestamps.push(now);
      return;
    }

    // At limit (sliding window) -- wait for the oldest entry to age out of the 300s window.
    const oldest = state.timestamps[0];
    if (oldest == null) {
      state.timestamps.push(now);
      return;
    }
    const waitMs = oldest + WINDOW_MS - now + jitterMs();
    const waitSec = Math.ceil(waitMs / 1000);
    // Don't block longer than the window itself; if t tells us the real fixed-window reset is sooner, use that.
    // Our sliding wait is otherwise the right bound for a bursting caller.
    log.info(`[hf-rate-limit] ${bucket} window full (${state.timestamps.length}/${state.limit} in 5m) -- throttling ${waitSec}s`);
    await sleep(Math.max(0, waitMs));
    // loop, prune again
  }
}

export function getRateLimitStatus(): Record<RateLimitBucket, { limit: number; remaining?: number; resetInSec?: number; usedInWindow: number }> {
  const now = Date.now();
  const out: Record<string, { limit: number; remaining?: number; resetInSec?: number; usedInWindow: number }> = {};
  for (const b of ["api", "resolvers"] as RateLimitBucket[]) {
    const s = ensureBucket(b);
    prune(s, now);
    out[b] = {
      limit: s.limit,
      remaining: s.remaining,
      resetInSec: s.resetAtMs ? Math.max(0, Math.ceil((s.resetAtMs - now) / 1000)) : undefined,
      usedInWindow: s.timestamps.length,
    };
  }
  return out as Record<RateLimitBucket, { limit: number; remaining?: number; resetInSec?: number; usedInWindow: number }>;
}

// Exported for tests: reset internal state (not for production use).
export function resetRateLimitState(): void {
  buckets.clear();
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// --- convenience wrapper -----------------------------------------------------

// Parse RateLimit headers + apply exponential fallback for 429 sleeps when
// header is absent/malformed. Returns seconds to sleep (at least 5s).
export function parse429WaitSeconds(headers: Headers | null, fallbackSec = 30): number {
  if (!headers) return fallbackSec;
  // Try both buckets, prefer api if ambiguous. The caller should pass the correct bucket,
  // but this fallback handles grouped headers too.
  for (const b of ["api", "resolvers"] as RateLimitBucket[]) {
    const { resetSec } = parseRateLimitHeaders(b, headers);
    if (resetSec != null && resetSec > 0 && resetSec <= 3600) return resetSec;
  }
  // Raw t= extraction as last resort (in case bucket filtering failed but header still has t)
  const raw = (headers.get("ratelimit") ?? headers.get("RateLimit") ?? "") + "";
  const m = raw.match(/t\s*=\s*(\d+)/i);
  if (m) {
    const n = Number(m[1]);
    if (Number.isFinite(n) && n > 0 && n <= 3600) return n;
  }
  const retryAfter = headers.get("retry-after");
  if (retryAfter) {
    const n = Number(retryAfter);
    if (Number.isFinite(n) && n > 0 && n <= 3600) return n;
  }
  return fallbackSec;
}
