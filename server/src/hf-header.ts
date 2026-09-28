// Reads GGUF headers straight from Hugging Face so the shared model catalog's
// technical facts (layer count, per-layer bytes, KV geometry, ...) come from
// the server itself, never from a user's worker -- step 2 of
// docs/PLAN_MODEL_METADATA_TRUST.md.
//
// Per file, exactly ONE request counts against the HF budget: a GET on the
// resolve/ URL with redirects NOT followed. That 302 already carries
//   X-Linked-ETag  -- the file's sha256 (checked against the model id; a
//                     mismatch means these bytes aren't that file, so nothing
//                     is stored for it),
//   X-Linked-Size  -- the file size (for the last tensor's byte size),
//   Location       -- a signed CDN URL.
// The header itself is then read from the CDN in large Range chunks, which
// isn't HF API traffic. Content-addressed, so each file is read once, ever.

import { parseGgufInfo } from "../../shared/gguf.js";
import type { ModelMetadata } from "../../shared/types.js";
import { hfFetch } from "./hf.js";
import { getRateLimitStatus } from "./hf-rate-limit.js";
import { repo } from "./db/repo.js";
import { log } from "./log.js";
import { ggufMetadataPatch, withHeaderFields } from "./model-header.js";

const HF_BASE = "https://huggingface.co";
const RESOLVE_TIMEOUT_MS = 15_000;
const RANGE_TIMEOUT_MS = 30_000;
// Network read-ahead: headers are typically 2-15 MB (tokenizer vocab), so a
// handful of round trips per file instead of hundreds at the local 64 KB.
const RANGE_CHUNK_BYTES = 4 * 1024 * 1024;
const TICK_INTERVAL_MS = 60_000;
const FILES_PER_TICK = 2;
// Leave the shared resolvers budget to the index crawler and user-facing
// lookups: skip a tick once more than this share of the window is used.
const MAX_RESOLVERS_WINDOW_SHARE = 0.5;
const MAX_SAME_SITE_REDIRECTS = 3;

export type HfHeaderOutcome =
  | { status: "ok"; fields: Partial<ModelMetadata> }
  // Permanent for this file: gated/private/missing, not an LFS file, or the
  // bytes HF serves aren't this sha256.
  | { status: "unavailable"; reason: string }
  // Worth retrying later: network error, 5xx, 429, truncated read.
  | { status: "failed"; reason: string };

function unquote(etag: string | null): string | null {
  if (!etag) return null;
  return etag.replace(/^W\//, "").replace(/^"|"$/g, "").toLowerCase();
}

export async function fetchHfHeader(
  repoId: string,
  filename: string,
  revision: string,
  expectedSha256: string,
  baseUrl = HF_BASE
): Promise<HfHeaderOutcome> {
  const enc = (p: string) => p.split("/").map(encodeURIComponent).join("/");
  const url = `${baseUrl}/${enc(repoId)}/resolve/${encodeURIComponent(revision)}/${enc(filename)}`;

  // A renamed/moved repo answers with a same-site redirect to its new
  // resolve/ URL that doesn't carry the file headers yet -- follow those (each
  // hop is one more budgeted request, capped), but never a cross-site one
  // without X-Linked-ETag: that's not HF vouching for the bytes.
  let res: Response;
  let current = url;
  for (let hop = 0; ; hop++) {
    try {
      res = await hfFetch(current, { redirect: "manual" }, { bucket: "resolvers", timeoutMs: RESOLVE_TIMEOUT_MS, reason: "hf-header" });
    } catch (err) {
      return { status: "failed", reason: `resolve: ${err instanceof Error ? err.message : String(err)}` };
    }
    // Nothing to read from this response body; don't leave it hanging.
    await res.body?.cancel().catch(() => {});
    const loc = res.headers.get("location");
    const isRedirect = res.status >= 300 && res.status < 400;
    if (!isRedirect || res.headers.get("x-linked-etag") || !loc) break;
    const next = new URL(loc, current);
    if (next.origin !== new URL(url).origin || hop >= MAX_SAME_SITE_REDIRECTS) {
      return { status: "unavailable", reason: `redirect without X-Linked-ETag to ${next.origin}` };
    }
    current = next.toString();
  }
  if (res.status === 401 || res.status === 403 || res.status === 404 || res.status === 410) {
    return { status: "unavailable", reason: `resolve HTTP ${res.status}` };
  }
  if (res.status < 300 || res.status >= 400) {
    // 2xx = served inline (not an LFS/Xet file -- no hash to check against);
    // 5xx/429-after-retry = transient.
    return res.status < 300
      ? { status: "unavailable", reason: "not an LFS file (no redirect)" }
      : { status: "failed", reason: `resolve HTTP ${res.status}` };
  }

  const etag = unquote(res.headers.get("x-linked-etag"));
  if (!etag) return { status: "unavailable", reason: "no X-Linked-ETag" };
  if (etag !== expectedSha256.toLowerCase()) {
    return { status: "unavailable", reason: `sha mismatch: HF serves ${etag}` };
  }
  const size = Number(res.headers.get("x-linked-size"));
  const location = res.headers.get("location");
  if (!location || !Number.isFinite(size) || size <= 0) {
    return { status: "failed", reason: "redirect missing Location or X-Linked-Size" };
  }
  const cdnUrl = new URL(location, current).toString();

  let pos = 0;
  let readError = null as string | null;
  const info = await parseGgufInfo(
    {
      size: async () => size,
      read: async (into, length) => {
        if (pos >= size) return 0;
        const end = Math.min(pos + length, size) - 1;
        let r: Response;
        try {
          r = await fetch(cdnUrl, {
            headers: { range: `bytes=${pos}-${end}` },
            signal: AbortSignal.timeout(RANGE_TIMEOUT_MS),
          });
        } catch (err) {
          readError = `range read: ${err instanceof Error ? err.message : String(err)}`;
          throw err;
        }
        if (r.status !== 206) {
          await r.body?.cancel().catch(() => {});
          readError = `range read HTTP ${r.status}`;
          throw new Error(readError);
        }
        const bytes = new Uint8Array(await r.arrayBuffer());
        const n = Math.min(bytes.length, length);
        into.set(bytes.subarray(0, n), 0);
        pos += n;
        return n;
      },
    },
    RANGE_CHUNK_BYTES
  );

  // Any network failure mid-read is transient and means the parse may be
  // incomplete (the tensor byte sizes are read last and fail soft) -- never
  // store a partial header as final. A header that fully read but carries no
  // layer count (or exceeded the read cap) won't improve on retry.
  if (readError) return { status: "failed", reason: readError };
  if (info.n_layer === null) return { status: "unavailable", reason: info.debugReason ?? "header unparseable" };
  return { status: "ok", fields: ggufMetadataPatch(info) };
}

// One pass: fetch up to FILES_PER_TICK catalog files that have no HF header
// yet (or a failed one due for retry), store the outcome, and apply an ok
// header to the shared catalog row.
export async function runHfHeaderTick(baseUrl = HF_BASE): Promise<number> {
  const rl = getRateLimitStatus().resolvers;
  if (rl && rl.limit > 0 && rl.usedInWindow > rl.limit * MAX_RESOLVERS_WINDOW_SHARE) return 0;

  const due = repo.listModelsNeedingHfHeader(FILES_PER_TICK, Date.now());
  for (const m of due) {
    const outcome = await fetchHfHeader(m.hf_repo, m.hf_file, m.revision ?? "main", m.id, baseUrl);
    repo.recordHfHeader(m.id, outcome);
    if (outcome.status === "ok") {
      const current = repo.getModel(m.id);
      if (current) repo.setModelMetadata(m.id, withHeaderFields(current.metadata, repo.resolveHeaderFields(m.id)));
    } else {
      log.info(`[hf-header] ${m.hf_repo}/${m.hf_file}: ${outcome.status} (${outcome.reason})`);
    }
  }
  return due.length;
}

let timer: NodeJS.Timeout | null = null;
let running = false;

export function startHfHeaderService(): void {
  if (timer) return;
  timer = setInterval(() => {
    if (running) return;
    running = true;
    runHfHeaderTick()
      .catch((err) => log.warn(`[hf-header] tick failed: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => {
        running = false;
      });
  }, TICK_INTERVAL_MS);
  timer.unref();
}

export function stopHfHeaderService(): void {
  if (timer) clearInterval(timer);
  timer = null;
}
