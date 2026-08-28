// Shared Hugging Face lookups -- used both by the /api/hf/* routes (client's
// model-download picker) and by the AI assistant's tool calls (server/src/routes/ai.ts),
// so the assistant grounds model suggestions in live HF data instead of its own
// training data, which goes stale as new model generations ship.
//
// All outbound HF fetches go through hfFetch() so HF_TOKEN auth and the
// 5-minute RateLimit throttling (see ./hf-rate-limit.ts) are applied
// consistently -- no caller should call bare fetch() for huggingface.co.

import {
  acquireRateLimitSlot,
  getHfAuthHeaders,
  parse429WaitSeconds,
  recordRateLimit,
} from "./hf-rate-limit.js";
import { log } from "./log.js";

export interface HfSearchResult {
  id: string;
  downloads: number;
  likes: number;
  // Repo creation date HF's own search API reports (ISO string) -- powers
  // the "Newest" sort option. null if the API response omitted it.
  created_at: string | null;
  // Last-modified timestamp HF reports (ISO string) -- powers the
  // incremental lastModified sweep (server/src/hf-index.ts). null if
  // the API omitted it (older repos, non-standard responses).
  last_modified: string | null;
}

export interface HfFileEntry {
  path: string;
  size_bytes: number;
  quant: string | null;
  // SHA-256 (LFS oid), already present on each entry of the recursive tree
  // response -- no extra per-file request needed. null for non-LFS files.
  sha256: string | null;
}

export const HF_REPO_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;
// Group 1 is unsloth's "Dynamic" quant prefix (UD-Q4_K_XL, UD-IQ2_M, ...) --
// optional; group 2 is the actual quant code. Mirrored in
// client/src/modelGrouping.ts's own QUANT_PATTERN.
const QUANT_PATTERN = /(?:^|[-_.])(UD[-_])?((?:IQ|Q)[0-9]+(?:_[A-Z0-9]+)*|F16|F32|BF16)(?=[-_.]|$)/i;

export function parseQuant(filename: string): string | null {
  const base = filename.split("/").pop() ?? filename;
  const m = base.match(QUANT_PATTERN);
  if (!m) return null;
  return (m[1] ? "UD-" : "") + m[2].toUpperCase();
}

// HF's own accepted values for its search API's `sort` param -- confirmed
// live against https://huggingface.co/api/models (no vendored llama.cpp/HF
// source or docs in this repo to check against instead). Deliberately a
// subset of what the client's old sort dropdown offered: "Name (A-Z)" and
// "Parameters" have no server-side equivalent, and would be misleading as a
// client-only sort now that results are real paginated pages (a client-side
// re-sort only ever sees the current page's 15 items, not the true global
// order).
export type HfSortField = "downloads" | "likes" | "createdAt" | "lastModified";

export interface HfSearchOptions {
  sort?: HfSortField;
  // -1 = descending (HF's own convention), 1 = ascending. Omitted entirely
  // (along with sort) for the default "Relevance" ordering.
  direction?: -1 | 1;
  // Opaque cursor from a previous page's nextCursor -- HF's search API
  // paginates via a `Link: <...&cursor=...>; rel="next"` response header
  // (cursor-based, forward-only), not an offset/page-number param. Confirmed
  // live: the API returns no total-match count in any form either, hence no
  // "N models match" UI on the client side.
  cursor?: string;
  limit?: number;
}

export interface HfSearchPage {
  items: HfSearchResult[];
  nextCursor: string | null;
}

const HF_DEFAULT_PAGE_SIZE = 15;

// Extracts the full next-page URL from a `Link: <url>; rel="next"` response
// header -- returns null when absent (last page) or unparseable. Exported
// for its own unit test (see hf.test.ts) and used by listHfGgufFiles's tree
// pagination, which re-fetches the whole next URL (the tree endpoint's cursor
// is opaque and not necessarily the same `cursor` query param the search API
// uses -- following the link wholesale is what huggingface_hub's paginate()
// does too).
export function parseNextLink(linkHeader: string | null): string | null {
  if (!linkHeader) return null;
  const match = linkHeader.match(/<([^>]+)>\s*;\s*rel="next"/);
  if (!match) return null;
  try {
    new URL(match[1]); // validate -- throw on malformed
    return match[1];
  } catch {
    return null;
  }
}

// Extracts the cursor query param from the next-page Link header. Exported
// for its own unit test (see hf.test.ts) -- this is the one bit of new
// pagination logic worth testing in isolation, since the rest of
// searchHfGgufModels just builds a URL and makes a real network call.
export function parseNextCursor(linkHeader: string | null): string | null {
  const nextUrl = parseNextLink(linkHeader);
  if (!nextUrl) return null;
  try {
    return new URL(nextUrl).searchParams.get("cursor");
  } catch {
    return null;
  }
}

// --- centralized fetch with auth + 5-minute window throttling ----------------

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function mergeAbortSignals(a: AbortSignal | undefined, b: AbortSignal | undefined): AbortSignal | undefined {
  if (!a) return b;
  if (!b) return a;
  // Node 20+: AbortSignal.any composes signals. Fall back to manual if absent.
  if (typeof (AbortSignal as unknown as { any?: (s: AbortSignal[]) => AbortSignal }).any === "function") {
    return (AbortSignal as unknown as { any: (s: AbortSignal[]) => AbortSignal }).any([a, b]);
  }
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort((a.reason ?? b.reason) as Error);
  if (a.aborted || b.aborted) onAbort();
  else {
    a.addEventListener("abort", onAbort, { once: true });
    b.addEventListener("abort", onAbort, { once: true });
  }
  return ctrl.signal;
}

/**
 * Authenticated, rate-limit-aware fetch for huggingface.co.
 * - Injects HF_TOKEN (Bearer) when configured.
 * - Throttles via 5-minute sliding window (api bucket by default).
 * - Records RateLimit/RateLimit-Policy headers to learn real quotas.
 * - On 429, waits the exact `t=` from the RateLimit header then retries once.
 * Exported so hf-index.ts and routes can share the same wrapper.
 *
 * `opts.reason` is a short caller tag (e.g. "backlog-walk", "search-ui",
 * "on-demand-verify") attributing this specific call -- required, not
 * optional, so a new call site can't silently go untagged. It costs nothing
 * at request time and is what lets hf-rate-limit.ts's low-remaining/
 * window-full log lines say *what* has been consuming the shared budget
 * instead of just reporting a bare number (see server/src/hf-rate-limit.ts).
 */
export async function hfFetch(
  url: string,
  init: RequestInit | undefined,
  opts: { bucket?: "api" | "resolvers"; timeoutMs: number; retries?: number; reason: string }
): Promise<Response> {
  const bucket = opts.bucket ?? "api";
  const retries = opts.retries ?? 1;
  // Acquire a slot in the 5-minute window before we even hit the network.
  await acquireRateLimitSlot(bucket, opts.reason);

  const timeoutSignal = AbortSignal.timeout(opts.timeoutMs);
  const signal = mergeAbortSignals(init?.signal as AbortSignal | undefined, timeoutSignal);
  const headers = new Headers(init?.headers);
  for (const [k, v] of Object.entries(getHfAuthHeaders())) {
    if (!headers.has(k)) headers.set(k, v);
  }
  // Identify ourselves -- helps HF support distinguish this app's traffic.
  if (!headers.has("user-agent")) headers.set("user-agent", "llamatoaster-server");

  const res = await fetch(url, { ...init, headers, signal });

  // Learn real quota/remaining from response headers even on success.
  try {
    recordRateLimit(bucket, res.headers as unknown as Headers);
  } catch {
    // non-fatal: header parsing never throws, but be defensive
  }

  if (res.status === 429 && retries > 0) {
    const waitSec = parse429WaitSeconds(res.headers as unknown as Headers, 30);
    log.warn(`[hf] 429 rate-limited on ${bucket} ${url} -- retrying after ${waitSec}s (t from RateLimit header)`);
    await sleep(waitSec * 1000);
    // Recurse with one fewer retry; acquire will re-check the window.
    return hfFetch(url, init, { bucket, timeoutMs: opts.timeoutMs, retries: retries - 1, reason: opts.reason });
  }

  return res;
}

// Re-export token helpers so callers don't need to import hf-rate-limit directly.
export { getHfToken, getHfAuthHeaders, isHfTokenConfigured } from "./hf-rate-limit.js";

// query may be empty -- an empty query + sort=downloads is exactly the
// Models page's default "top 15" listing shown before the user searches.
// `reason` attributes this call for hf-rate-limit.ts's logging -- see
// hfFetch's doc comment.
export async function searchHfGgufModels(
  query: string,
  timeoutMs: number,
  reason: string,
  options: HfSearchOptions = {}
): Promise<HfSearchPage> {
  const q = query.trim();
  const params = new URLSearchParams({ filter: "gguf", limit: String(options.limit ?? HF_DEFAULT_PAGE_SIZE) });
  if (q) params.set("search", q);
  if (options.sort) params.set("sort", options.sort);
  if (options.direction) params.set("direction", String(options.direction));
  if (options.cursor) params.set("cursor", options.cursor);
  const url = `https://huggingface.co/api/models?${params.toString()}`;
  const res = await hfFetch(url, undefined, { bucket: "api", timeoutMs, reason });
  if (!res.ok) {
    // Enrich 429 with the precise reset (t=) so callers (e.g. hf-index's
    // scanHfRepo) can honor the server's rate-limit window instead of a
    // blind 30s fallback.
    if (res.status === 429) {
      const waitSec = parse429WaitSeconds(res.headers as unknown as Headers, 30);
      throw new Error(`HF search failed: 429 t=${waitSec}`);
    }
    throw new Error(`HF search failed: ${res.status}`);
  }
  const data = (await res.json()) as {
    id: string;
    downloads?: number;
    likes?: number;
    createdAt?: string;
    lastModified?: string;
  }[];
  const items = data.map((m) => ({
    id: m.id,
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    created_at: typeof m.createdAt === "string" ? m.createdAt : null,
    last_modified: typeof m.lastModified === "string" ? m.lastModified : null,
  }));
  return { items, nextCursor: parseNextCursor(res.headers.get("link")) };
}

export async function listHfGgufFiles(
  repoId: string,
  timeoutMs: number,
  reason: string,
  revision = "main"
): Promise<HfFileEntry[]> {
  const encRepo = repoId.split("/").map(encodeURIComponent).join("/");
  let url: string | null = `https://huggingface.co/api/models/${encRepo}/tree/${encodeURIComponent(revision)}?recursive=true`;
  const files: HfFileEntry[] = [];
  // The tree endpoint paginates via Link header (default ~1000 entries/page,
  // like huggingface_hub's paginate()). A truncated tree would make the
  // index look like files were deleted, so we must walk every page.
  let pages = 0;
  while (url) {
    const res = await hfFetch(url, undefined, { bucket: "api", timeoutMs, reason });
    if (!res.ok) {
      if (res.status === 429) {
        const waitSec = parse429WaitSeconds(res.headers as unknown as Headers, 30);
        throw new Error(`HF tree fetch failed: 429 t=${waitSec}`);
      }
      throw new Error(`HF tree fetch failed: ${res.status}`);
    }
    const data = (await res.json()) as {
      type: string;
      path: string;
      size?: number;
      lfs?: { size?: number; oid?: string };
    }[];
    for (const e of data) {
      if (e.type === "file" && e.path.toLowerCase().endsWith(".gguf")) {
        files.push({
          path: e.path,
          size_bytes: e.lfs?.size ?? e.size ?? 0,
          quant: parseQuant(e.path),
          sha256: e.lfs?.oid ?? null,
        });
      }
    }
    url = parseNextLink(res.headers.get("link"));
    pages++;
    if (pages > 500) {
      // Defensive cap: never loop forever even if HF keeps sending a next
      // link. 500 pages × ~1000 entries is far beyond any real repo.
      throw new Error("HF tree fetch failed: pagination limit exceeded");
    }
  }
  return files;
}

// Resolve a repo's current default revision (the `sha` field HF reports on
// the model-info endpoint) -- used by scanHfRepo to distinguish a repo that
// simply has a non-"main" default branch (tree/main 404s but tree/<sha> is
// valid, so its GGUF files should be indexed, not deleted) from a repo that
// genuinely no longer exists. Returns null when the repo info can't be
// fetched (transient/network) OR the repo is truly gone -- the caller
// decides between those two via the model-info status.
export async function getHfRepoDefaultRevision(
  repoId: string,
  timeoutMs: number,
  reason: string
): Promise<{ revision: string | null; status: number }> {
  const url = `https://huggingface.co/api/models/${repoId}`;
  const res = await hfFetch(url, undefined, { bucket: "api", timeoutMs, reason });
  if (!res.ok) return { revision: null, status: res.status };
  const data = (await res.json()) as { sha?: string };
  const sha = data.sha;
  return { revision: typeof sha === "string" && sha ? sha : null, status: 200 };
}

export interface HfGgufMeta {
  // Real total parameter count, computed by HF itself from the repo's GGUF
  // tensor shapes (repo-wide, not per-quant-file -- quantization changes
  // bytes-per-parameter, not the element count). null if the repo has no
  // "gguf" section yet (non-GGUF repo, or HF hasn't parsed it) or the fetch
  // failed -- fail-soft, same posture as worker/src/gguf.ts's layer-count
  // reader: callers treat this as "unknown", never block on it.
  param_count: number | null;
  // Repo-level "last modified" timestamp (ISO string) HF's own API reports --
  // the same value HF's own UI surfaces as a repo card's "Updated X ago".
  // Not a per-file commit date (the file-tree API doesn't expose that
  // without one extra request per file); good enough for "is this repo newer
  // than what I downloaded" purposes. null on fetch failure.
  last_modified: string | null;
}

export async function getHfGgufMeta(repoId: string, timeoutMs: number, reason: string): Promise<HfGgufMeta> {
  try {
    const url = `https://huggingface.co/api/models/${repoId}`;
    const res = await hfFetch(url, undefined, { bucket: "api", timeoutMs, reason });
    if (!res.ok) return { param_count: null, last_modified: null };
    const data = (await res.json()) as { gguf?: { total?: number }; lastModified?: string };
    const total = data.gguf?.total;
    return {
      param_count: typeof total === "number" ? total : null,
      last_modified: typeof data.lastModified === "string" ? data.lastModified : null,
    };
  } catch {
    return { param_count: null, last_modified: null };
  }
}

/**
 * Optional resolver-based helper: HEAD a file's /resolve URL to probe
 * existence/size without burning an API quota.  Returns null on 404/other
 * failure.  Not used for GGUF discovery (which needs directory listing, only
 * the API provides), but useful as a fallback/verification path where the
 * API would otherwise be called per-file.
 *
 * Uses the `resolvers` bucket (rate limits 6x higher than API).
 */
export async function headHfResolveFile(
  repoId: string,
  filePath: string,
  timeoutMs: number,
  reason: string
): Promise<{ exists: boolean; size?: number; etag?: string } | null> {
  const encRepo = repoId.split("/").map(encodeURIComponent).join("/");
  const encFile = filePath.split("/").map(encodeURIComponent).join("/");
  const url = `https://huggingface.co/${encRepo}/resolve/main/${encFile}`;
  try {
    const res = await hfFetch(url, { method: "HEAD" }, { bucket: "resolvers", timeoutMs, reason });
    if (res.status === 404) return { exists: false };
    if (!res.ok) return null;
    const len = res.headers.get("content-length");
    const etag = res.headers.get("etag") ?? res.headers.get("x-linked-etag") ?? undefined;
    return { exists: true, size: len ? Number(len) : undefined, etag: etag ?? undefined };
  } catch {
    return null;
  }
}
