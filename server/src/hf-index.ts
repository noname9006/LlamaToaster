import { getDb } from "./db/migrate.js";
import { getHfRepoDefaultRevision, listHfGgufFiles, type HfFileEntry } from "./hf.js";
import type { HfGgufIndexEntry } from "../../shared/types.js";
import { log } from "./log.js";
import { isHfTokenConfigured, getRateLimitStatus } from "./hf-rate-limit.js";

// Hugging Face GGUF Index Service
//
// Background service that periodically scans Hugging Face repositories and
// builds a database of available GGUF files. One row per (sha256, repo_id,
// filename) combination in the hf_gguf_index table -- the composite primary
// key (schema.sql) keeps mirrored content across repos distinct; revision is
// stored per row but isn't part of the key.
//
// Each tick does up to four things:
//   1) Backlog walk (one-time), sorted by createdAt (immutable -- unlike
//      lastModified/downloads, a repo's position in a createdAt-sorted
//      listing never shifts after upload, so HF's cursor-based pagination
//      can't skip/duplicate items mid-walk): newest-created -> oldest,
//      covering the whole existing catalog. Runs every tick until it wraps
//      once, then never again (hf_index_backlog_complete flips permanently).
//      Newest-first because recency matters more than lifetime popularity
//      for this tool -- a hot model released yesterday has ~0 downloads and
//      would sit at the bottom of a downloads-sorted crawl for weeks
//      otherwise.
//   2) Recent top-up (perpetual), also createdAt-sorted: stop as soon as we
//      reach the bookmark (max createdAt already covered). Seeded from the
//      backlog walk's very first page (not just at wrap), so this starts
//      covering new uploads from tick 1 instead of waiting for the
//      multi-week backlog walk to finish. Catches brand-new repos; see (3)
//      for content changes to repos already in the index.
//   3) Last-modified sweep (perpetual): sorted by lastModified descending,
//      stop as soon as we reach a bookmark (max lastModified already
//      covered this way). Unlike createdAt, lastModified is NOT immutable --
//      any touch (new file, retag, README edit) bumps a repo back to the top
//      of this sort. In the normal case (steady stream of changes, nothing
//      backed up) this reaches the bookmark within the first page, same as
//      restarting from the top every tick. Only when a burst pushes more
//      changed repos above the bookmark than one tick's batch can cover
//      does it fall back to resuming a short-lived cursor across ticks
//      (hf_index_lastmodified_sweep_resume_cursor) instead of re-covering
//      the same top slice forever -- safe across a ~1-minute gap in a way a
//      stored cursor spanning the multi-day backlog walk in (1) would not
//      be (see scanLastModifiedRepos's doc comment). On a fresh database
//      with no bookmark yet, it seeds one immediately from whatever it sees
//      on tick 1 instead of trying to walk the entire catalog to "complete"
//      -- (1) already owns discovering every existing repo; this sweep only
//      needs to track changes from here forward. This is what lets content
//      changes on repos already in the index get picked up cheaply (a
//      handful of search requests) instead of relying on (4) blindly
//      re-fetching every repo's full file tree on a timer.
//   4) Staleness refresh (mode-gated -- see getStaleRefreshMode): repos whose
//      last_seen is older than HF_INDEX_REFRESH_INTERVAL_MS get re-checked
//      (excludes rows already soft-deleted -- see below, no point
//      background-polling a confirmed-gone repo). With (3) now covering
//      content changes, this is mostly a backstop for the one case neither
//      (3) nor on-demand verification (see below) can see: a repo that's
//      indexed, gets deleted, and that no worker currently references and HF
//      never resurfaces as "modified". Unlike (1)-(3), which page through
//      HF's search API (one request covers up to 50 repos), this sweep costs
//      one full tree-fetch API call PER repo -- by far the most expensive of
//      the four per repo scanned. HF_INDEX_STALENESS_REFRESH_MODE=on-demand
//      disables this background sweep entirely and relies solely on
//      routes/models.ts's hash-lookup already re-verifying a matched-but-
//      stale row the moment a worker actually reports that file's hash (see
//      verifyRepoInBackground) -- i.e. only repos someone's local model
//      directory currently references get re-checked, and only when that
//      happens. Tradeoff: a repo nobody's hash-lookup ever touches again
//      keeps its last known last_seen (and deleted_at) forever.
//
// Deletion is handled lazily, not by a background reconciliation crawl: rows
// are soft-deleted (deleted_at set, never hard-removed) by scanHfRepo when it
// directly observes a repo/file is gone, and separately, POST
// /api/models/hash-lookup triggers a one-off verifyRepoInBackground() for any
// matched-but-stale row -- verification is driven by actual demand (a worker
// depending on that row) instead of continuously re-walking ~200k repos hoping
// to notice the rare deletion. Tradeoff: a repo nobody looks up can stay
// wrongly-live in the index indefinitely, and a soft-deleted repo that gets
// republished on HF isn't automatically rediscovered (accepted for v1 --
// the backlog walk already passed it once and won't revisit).
//
// The index is used by POST /api/models/hash-lookup: workers send SHA-256
// hashes of local files, and the server returns the matching HF metadata
// (repo_id, filename, revision, deleted_at) so the client can display "this is
// bartowski/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf" (or a "removed
// from Hugging Face" note once deleted_at is set).
//
// Rate limits -- see https://huggingface.co/docs/hub/rate-limits :
//   HF enforces per-bucket 5-minute FIXED windows (RateLimit / RateLimit-Policy
//   headers with r=remaining, t=secondsUntilReset, q=quota, w=300). This module
//   tracks usage via server/src/hf-rate-limit.ts's sliding-window limiter and
//   honors 429's t= via parse429WaitSeconds, so callers never blind-sleep 30s.
//   Set HF_TOKEN (or HF_API_KEY / HUGGING_FACE_HUB_TOKEN) for higher quotas:
//     Anonymous: API 500,  Resolvers 3000 /5min
//     Free user: API 1000, Resolvers 5000
//     PRO:       API 2500, Resolvers 12000  (+Team/Enterprise higher)
//   When a token is configured this service automatically scales its per-tick
//   backlog batch size and uses hfFetch()'s limiter to stay within the current
//   5-minute window. See getBacklogBatchSize/getRecentBatchSize/
//   getLastModifiedBatchSize/getMaxReposPerTick.

export const HF_INDEX_REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24 hours
export const HF_INDEX_TIMEOUT_MS = 30_000; // per-repo timeout

// Per-tick caps -- kept as exported constants for tests that import them,
// but the live tick reads getMaxReposPerTick()/getBacklogBatchSize()/
// getRecentBatchSize() so HF_TOKEN + 5-min window state can scale throughput
// without code changes.
export const HF_INDEX_MAX_REPOS_PER_TICK = 125; // anon fallback; token → 250
export const HF_INDEX_BACKLOG_BATCH_SIZE = 250; // anon fallback; token → 500
export const HF_INDEX_RECENT_BATCH_SIZE = 50; // fixed -- real upload volume/5min is tiny
// Higher default than HF_INDEX_RECENT_BATCH_SIZE: any touch to any
// gguf-tagged repo (new quant, retag, README edit) bumps lastModified, so
// this sweep's real per-tick volume is expected to run well above pure
// new-repo creation. Env-overridable like the others if that assumption is
// wrong for the real catalog's activity level.
export const HF_INDEX_LASTMODIFIED_BATCH_SIZE = 100;

export function getMaxReposPerTick(): number {
  const env = process.env.HF_INDEX_MAX_REPOS_PER_TICK?.trim();
  if (env) {
    const n = Number(env);
    if (Number.isFinite(n) && n > 0) return Math.floor(n);
  }
  return isHfTokenConfigured() ? 250 : 125;
}

export function getBacklogBatchSize(): number {
  const env = process.env.HF_INDEX_BACKLOG_BATCH_SIZE?.trim();
  if (env) {
    const n = Number(env);
    if (Number.isFinite(n) && n > 0) return Math.floor(n);
  }
  return isHfTokenConfigured() ? 500 : 250;
}

export function getRecentBatchSize(): number {
  const env = process.env.HF_INDEX_RECENT_BATCH_SIZE?.trim();
  if (env) {
    const n = Number(env);
    if (Number.isFinite(n) && n > 0) return Math.floor(n);
  }
  return HF_INDEX_RECENT_BATCH_SIZE;
}

export function getLastModifiedBatchSize(): number {
  const env = process.env.HF_INDEX_LASTMODIFIED_BATCH_SIZE?.trim();
  if (env) {
    const n = Number(env);
    if (Number.isFinite(n) && n > 0) return Math.floor(n);
  }
  return HF_INDEX_LASTMODIFIED_BATCH_SIZE;
}

// "background" (default): runIndexTick's staleness sweep below queries
// findStaleRepos() every tick and re-scans whatever it returns, same as
// always -- every indexed repo gets a full tree-fetch API call roughly once
// per HF_INDEX_REFRESH_INTERVAL_MS regardless of whether anyone still cares
// about it.
// "on-demand": that background sweep is skipped entirely. Staleness is left
// to routes/models.ts's hash-lookup route, which already calls
// verifyRepoInBackground on a matched-but-stale row the moment a worker
// reports that file's hash -- i.e. a repo only gets re-checked when its
// model is actually in use somewhere, not on a blind timer. See the module
// doc comment's point (4) for the full tradeoff.
export function getStaleRefreshMode(): "background" | "on-demand" {
  const env = process.env.HF_INDEX_STALENESS_REFRESH_MODE?.trim().toLowerCase();
  return env === "on-demand" ? "on-demand" : "background";
}

// How often a tick runs. HF's rate limit is a true sliding window tracked in
// hf-rate-limit.ts's module state (persists across ticks, not reset by tick
// boundaries), so a 5-minute cadence just leaves the window's throughput
// idle between ticks once a tick finishes early (which it now does --
// runIndexTick's per-repo loop no longer has an artificial delay, only the
// real rate-limit throttling). Default 1 minute (the floor this env also
// enforces) to keep feeding that window instead of bursting-then-idling.
// Overridable via HF_INDEX_TICK_INTERVAL_MS env (raw ms).
export const HF_INDEX_TICK_INTERVAL_MS = (() => {
  const env = process.env.HF_INDEX_TICK_INTERVAL_MS?.trim();
  if (env) {
    const n = Number(env);
    if (Number.isFinite(n) && n >= 60_000) return n;
  }
  return 60 * 1000; // 1 minute (floor) -- see comment above
})();

// Fresh key names, deliberately NOT reusing the pre-redesign keys
// (hf_index_discovery_cursor / hf_index_initial_complete / hf_index_last_
// modified_bookmark) -- those described progress through a downloads-sorted
// crawl. Feeding a downloads-sort cursor into a createdAt-sort query is
// meaningless (HF's opaque cursor tokens are query-specific), and an instance
// that had already finished its old downloads-sorted bootstrap would read
// "backlog complete" as true from day one under the new code, permanently
// skipping the createdAt backlog walk and getting stuck re-scanning the same
// top-N newest repos every tick with no bookmark ever seeded. Fresh keys mean
// every deployment -- fresh or upgraded -- cleanly (re)starts the backlog
// walk under the new ordering; the old keys are simply orphaned in the meta
// table.
const BACKLOG_CURSOR_KEY = "hf_index_backlog_cursor";
const RECENT_BOOKMARK_KEY = "hf_index_recent_bookmark";
const BACKLOG_COMPLETE_KEY = "hf_index_backlog_complete";
// Deliberately distinct from the orphaned pre-redesign key
// "hf_index_last_modified_bookmark" mentioned above (same reasoning: that key
// tracked a downloads-sorted crawl's cursor, a different semantic from this
// sweep's "max lastModified already covered" bookmark).
const LASTMODIFIED_BOOKMARK_KEY = "hf_index_lastmodified_sweep_bookmark";
// Short-lived state for an in-progress (not-yet-complete) lastModified
// sweep: when a burst pushes more changed repos above the bookmark than one
// tick's batch can cover, these persist the HF pagination cursor and the
// running max lastModified seen so far, so the *next* tick continues
// forward from there instead of restarting at the top and re-covering the
// same slice. Both are cleared the moment a sweep completes. See
// scanLastModifiedRepos's doc comment for why resuming a cursor here (a
// single-tick gap) is safe in a way the multi-day backlog cursor above is
// not.
const LASTMODIFIED_RESUME_CURSOR_KEY = "hf_index_lastmodified_sweep_resume_cursor";
const LASTMODIFIED_RESUME_NEWEST_KEY = "hf_index_lastmodified_sweep_resume_newest";

// Empty string in the meta row (rather than deleting it) means "wrapped back
// to the start" -- same as never having a cursor at all.
function getBacklogCursor(): string | undefined {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(BACKLOG_CURSOR_KEY) as
    | { value: string }
    | undefined;
  return row?.value || undefined;
}

function setBacklogCursor(cursor: string | null): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(BACKLOG_CURSOR_KEY, cursor ?? "");
}

function getRecentBookmark(): number | undefined {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(RECENT_BOOKMARK_KEY) as
    | { value: string }
    | undefined;
  if (!row?.value) return undefined;
  const n = Number(row.value);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function setRecentBookmark(ms: number): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(RECENT_BOOKMARK_KEY, String(ms));
}

function getLastModifiedBookmark(): number | undefined {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(LASTMODIFIED_BOOKMARK_KEY) as
    | { value: string }
    | undefined;
  if (!row?.value) return undefined;
  const n = Number(row.value);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function setLastModifiedBookmark(ms: number): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(LASTMODIFIED_BOOKMARK_KEY, String(ms));
}

// Empty string means "no resume cursor pending" -- same convention as
// getBacklogCursor. Only meaningful while a lastModified sweep is mid-burst
// (see LASTMODIFIED_RESUME_CURSOR_KEY above); cleared once that sweep
// completes.
function getLastModifiedResumeCursor(): string | undefined {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(LASTMODIFIED_RESUME_CURSOR_KEY) as
    | { value: string }
    | undefined;
  return row?.value || undefined;
}

function setLastModifiedResumeCursor(cursor: string | null): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(LASTMODIFIED_RESUME_CURSOR_KEY, cursor ?? "");
}

function getLastModifiedResumeNewest(): number | undefined {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(LASTMODIFIED_RESUME_NEWEST_KEY) as
    | { value: string }
    | undefined;
  if (!row?.value) return undefined;
  const n = Number(row.value);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function setLastModifiedResumeNewest(ms: number | null): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(LASTMODIFIED_RESUME_NEWEST_KEY, ms == null ? "" : String(ms));
}

export function isBacklogComplete(): boolean {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(BACKLOG_COMPLETE_KEY) as
    | { value: string }
    | undefined;
  return row?.value === "1";
}

function setBacklogComplete(v: boolean): void {
  getDb()
    .prepare(
      `INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value`
    )
    .run(BACKLOG_COMPLETE_KEY, v ? "1" : "0");
}

interface HfIndexRow {
  sha256: string;
  repo_id: string;
  filename: string;
  revision: string;
  file_size: number;
  last_seen: number;
  deleted_at: number | null;
}

// Upsert a single entry into the hf_gguf_index table.
// Composite PK (sha256, repo_id, filename) so mirrored content keeps distinct rows.
// deleted_at is always cleared on upsert -- content that reappears (a false
// 404, or a genuinely republished repo/file) automatically heals out of any
// prior soft-delete.
export function upsertHfGgufEntry(entry: HfGgufIndexEntry): void {
  getDb()
    .prepare(
      `INSERT INTO hf_gguf_index (sha256, repo_id, filename, revision, file_size, last_seen, deleted_at)
       VALUES (?, ?, ?, ?, ?, ?, NULL)
       ON CONFLICT(sha256, repo_id, filename) DO UPDATE SET
          revision = excluded.revision,
          file_size = excluded.file_size,
          last_seen = excluded.last_seen,
          deleted_at = NULL`
    )
    .run(entry.sha256, entry.repo_id, entry.filename, entry.revision, entry.file_size, entry.last_seen);
}

// Look up one or more SHA-256 hashes in the index. Returns entries for
// hashes that were found; missing hashes are simply absent from the result.
// Deliberately NOT filtered by deleted_at -- a soft-deleted row must still be
// returned so the caller (routes/models.ts) can tell "matched, but since
// removed from HF" apart from "never matched at all", and surface that to the
// client instead of the match silently vanishing.
export function lookupHfGgufHashes(hashes: string[]): HfGgufIndexEntry[] {
  if (hashes.length === 0) return [];
  const placeholders = hashes.map(() => "?").join(",");
  const rows = getDb()
    .prepare(`SELECT sha256, repo_id, filename, revision, file_size, last_seen, deleted_at
               FROM hf_gguf_index
               WHERE sha256 IN (${placeholders})`)
    .all(...hashes) as HfIndexRow[];
  return rows.map(mapHfIndexRow);
}

// Count total entries in the index (for admin/health display). Includes
// soft-deleted rows -- this is a raw row count, not a "live entries" count.
export function countHfGgufEntries(): number {
  const row = getDb().prepare("SELECT COUNT(*) AS c FROM hf_gguf_index").get() as { c: number };
  return row.c;
}

// Find repos whose last_seen is older than the given timestamp (i.e. due for
// an incremental refresh). Excludes repos already soft-deleted -- there's no
// value in a background loop repeatedly re-polling a confirmed-gone repo;
// re-discovery of a repo that reappears after deletion is handled lazily (or
// not at all -- see the module doc comment) rather than by this loop.
export function findStaleRepos(cutoffTs: number, limit: number): string[] {
  const rows = getDb()
    .prepare(
      `SELECT DISTINCT repo_id FROM hf_gguf_index
       WHERE last_seen < ? AND deleted_at IS NULL
       ORDER BY last_seen ASC
       LIMIT ?`
    )
    .all(cutoffTs, limit) as { repo_id: string }[];
  return rows.map((r) => r.repo_id);
}

// Soft-delete entries for a repo that no longer exists or has no GGUF files.
// Called after a repo's tree fetch returns empty or a confirmed 404. Rows are
// marked deleted_at, never hard-removed -- upsertHfGgufEntry clears it again
// if the content reappears, and the client can show a "removed from Hugging
// Face" note instead of the match just disappearing. Returns the number of
// rows newly marked (for the caller's `removed` count).
export function markRepoDeleted(repoId: string, now: number): number {
  return getDb()
    .prepare("UPDATE hf_gguf_index SET deleted_at = ? WHERE repo_id = ? AND deleted_at IS NULL")
    .run(now, repoId).changes;
}

// Soft-delete rows for a repo whose filename is no longer present in the
// repo's current tree (a file was renamed, replaced, or removed since the
// last scan). keepFilenames is the complete set of .gguf paths returned by
// listHfGgufFiles for this repo -- rows whose filename isn't in that set are
// stale and get marked deleted_at. Returns the number of rows newly marked.
//
// Stages keepFilenames in a temp table rather than binding them as `NOT IN
// (?,?,...)` query params: that list is capped by SQLite's ~999
// bound-parameter limit per statement, and a repo with more GGUF files than
// one chunk needs splitting across multiple statements -- but each chunk
// would only see ITS OWN filenames, wrongly marking rows deleted that are
// legitimately kept by a *different* chunk. A temp table + subquery evaluates
// the full keep-set in one comparison, with no size limit on how many
// filenames it can hold.
export function pruneRepoEntries(repoId: string, keepFilenames: string[]): number {
  if (keepFilenames.length === 0) {
    return markRepoDeleted(repoId, Date.now());
  }
  const db = getDb();
  db.exec(`CREATE TEMP TABLE IF NOT EXISTS _hf_index_keep_filenames (filename TEXT PRIMARY KEY)`);
  const tx = db.transaction((filenames: string[], now: number) => {
    db.exec(`DELETE FROM _hf_index_keep_filenames`);
    const insert = db.prepare(`INSERT OR IGNORE INTO _hf_index_keep_filenames (filename) VALUES (?)`);
    for (const f of filenames) insert.run(f);
    const result = db
      .prepare(
        `UPDATE hf_gguf_index SET deleted_at = ?
         WHERE repo_id = ? AND deleted_at IS NULL
           AND filename NOT IN (SELECT filename FROM _hf_index_keep_filenames)`
      )
      .run(now, repoId);
    db.exec(`DELETE FROM _hf_index_keep_filenames`);
    return result.changes;
  });
  return tx(keepFilenames, Date.now());
}

// Update last_seen for all entries belonging to a repo (called after a
// successful refresh, even if no files changed -- the repo itself was
// confirmed to still exist and be GGUF-capable).
export function touchRepoEntries(repoId: string, now: number): void {
  getDb().prepare("UPDATE hf_gguf_index SET last_seen = ? WHERE repo_id = ?").run(now, repoId);
}

function mapHfIndexRow(r: HfIndexRow): HfGgufIndexEntry {
  return {
    sha256: r.sha256,
    repo_id: r.repo_id,
    filename: r.filename,
    revision: r.revision,
    file_size: r.file_size,
    last_seen: r.last_seen,
    deleted_at: r.deleted_at,
  };
}

// In-flight guard for verifyRepoInBackground -- a burst of hash-lookup hits
// against the same stale repo (many workers, or one worker's whole model dir)
// would otherwise trigger duplicate concurrent HF calls for it.
const verifyingRepos = new Set<string>();

// Fire-and-forget verification of a single repo, triggered by
// POST /api/models/hash-lookup for a matched-but-stale row (see
// routes/models.ts) instead of a background reconciliation crawl. Never
// throws or blocks the caller -- the HTTP response has already been sent by
// the time this settles; any change (soft-delete, updated files) lands in the
// DB for the *next* lookup to see.
export function verifyRepoInBackground(repoId: string): void {
  if (verifyingRepos.has(repoId)) return;
  verifyingRepos.add(repoId);
  scanHfRepo(repoId, HF_INDEX_TIMEOUT_MS, "on-demand-verify")
    .catch((err) => {
      log.warn(`[hf-index] on-demand verify failed for ${repoId}: ${err instanceof Error ? err.message : String(err)}`);
    })
    .finally(() => {
      verifyingRepos.delete(repoId);
    });
}

// Scan a single repo: fetch its GGUF file tree (each entry already carries
// its SHA-256 -- the LFS oid -- from HF's recursive tree response) and
// upsert into the index. Returns the number of files indexed and rows
// soft-deleted.
// Only soft-deletes on a *confirmed* not-found of the repo itself (model-info
// endpoint agrees) or an empty tree; a tree/main not-found where the repo
// still exists but its default branch isn't named "main" (e.g. "master")
// gets resolved to the real default revision and indexed instead of deleted.
// Transient errors (timeout, 5xx, unresolved not-found) leave existing rows
// intact and just skip this call so the next refresh can retry.
//
// "Not found" means HTTP 404 OR 401: live-verified against real HF traffic
// (2026-08-28, unauthenticated) that both the tree and model-info endpoints
// return 401 "Invalid username or password" -- never 404 -- for a repo id
// that plain doesn't exist, for both a made-up namespace and a made-up repo
// under a real namespace. Treating only 404 as "gone" (the original
// assumption here) meant a deleted repo could never actually be detected
// without an HF_TOKEN: every check would fall through to the generic
// "transient failure" branch and leave the row live forever. This can't
// distinguish "genuinely deleted" from "made private/gated since indexing"
// (same 401 either way, anonymously) -- accepted, since either way an
// anonymous worker can no longer fetch the file from that repo, which is the
// thing hash-lookup actually promises callers.
export async function scanHfRepo(
  repoId: string,
  timeoutMs: number,
  reason: string
): Promise<{ indexed: number; removed: number }> {
  let files: HfFileEntry[];
  let revision = "main";
  try {
    files = await listHfGgufFiles(repoId, timeoutMs, reason);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const isNotFound = /\b(404|401)\b/.test(msg);
    const isRateLimited = /429/.test(msg);
    if (isNotFound) {
      // tree/main not-found: either the repo is gone (or inaccessible) or its
      // default branch isn't "main". Resolve the repo's real default
      // revision before deleting anything -- a GGUF repo on a non-main
      // default branch is valid and must be indexed, not removed.
      try {
        const { revision: defaultRev, status } = await getHfRepoDefaultRevision(repoId, timeoutMs, reason);
        if (status === 404 || status === 401) {
          // Repo genuinely gone (or no longer accessible) -- soft-delete its rows.
          const removed = markRepoDeleted(repoId, Date.now());
          return { indexed: 0, removed };
        }
        if (defaultRev && defaultRev !== "main") {
          // Default branch isn't main -- retry the tree fetch on it.
          files = await listHfGgufFiles(repoId, timeoutMs, reason, defaultRev);
          revision = defaultRev;
        } else {
          // Repo exists and defaults to main, yet tree/main was not-found --
          // a transient oddity; leave rows intact for the next tick.
          log.warn(`[hf-index] scanHfRepo not-found on tree/main but repo ${repoId} exists - leaving rows`);
          return { indexed: 0, removed: 0 };
        }
      } catch (err2) {
        // The default-revision probe itself failed (transient/network) -- we
        // can't confirm the repo is gone, so leave rows intact.
        const msg2 = err2 instanceof Error ? err2.message : String(err2);
        log.warn(`[hf-index] scanHfRepo not-found + default-revision probe failed for ${repoId}: ${msg2} - leaving rows`);
        return { indexed: 0, removed: 0 };
      }
    } else if (isRateLimited) {
      // 429 is already retried once inside hfFetch via the RateLimit t= header;
      // this catch is the second 429 (or a non-header case) -- honor precise t
      // (now embedded in the error message by hf.ts's listHfGgufFiles).
      const tMatch = msg.match(/t\s*=\s*(\d+)/i);
      const waitSec = tMatch ? Number(tMatch[1]) : 30;
      const waitMs = Number.isFinite(waitSec) && waitSec > 0 && waitSec <= 300 ? waitSec * 1000 : 30000;
      log.warn(`[hf-index] scanHfRepo rate-limited for ${repoId}: ${msg} - backing off ${Math.ceil(waitMs / 1000)}s (from RateLimit t=)`);
      await sleep(waitMs);
      return { indexed: 0, removed: 0 };
    } else {
      log.warn(`[hf-index] scanHfRepo transient failure for ${repoId}: ${msg}`);
      return { indexed: 0, removed: 0 };
    }
  }

  if (files.length === 0) {
    const removed = markRepoDeleted(repoId, Date.now());
    return { indexed: 0, removed };
  }

  const now = Date.now();
  let indexed = 0;
  const keepFilenames: string[] = [];
  for (const f of files) {
    keepFilenames.push(f.path);
    // sha256 (LFS oid) comes straight off the recursive tree response --
    // no per-file request needed (that used to cost one extra HF API call
    // per GGUF file and was the main driver of 429 rate-limiting).
    if (!f.sha256) continue; // non-LFS or unhashable, skip
    upsertHfGgufEntry({
      sha256: f.sha256,
      repo_id: repoId,
      filename: f.path,
      revision,
      file_size: f.size_bytes,
      last_seen: now,
      deleted_at: null,
    });
    indexed++;
  }

  // Soft-delete rows whose filename is no longer in the repo's current tree
  // (renamed, replaced, or removed since the last scan). The tree fetch is
  // authoritative and complete (paginated inside listHfGgufFiles), so a file
  // absent from it is genuinely gone -- leaving its row live would hand stale
  // metadata to hash-lookup forever.
  const removed = pruneRepoEntries(repoId, keepFilenames);

  // Touch all entries for this repo (even ones we didn't re-index this
  // pass -- the repo itself was confirmed to exist and have GGUF files).
  // This prevents the repo from showing up as stale on the next tick
  // just because some of its files weren't re-hashed.
  touchRepoEntries(repoId, now);

  return { indexed, removed };
}

// Discover up to `limit` repo ids via HF's search API, continuing from
// `startCursor` (undefined = start of the listing). Sorted by createdAt
// descending -- newest-created first, since recency matters more than
// lifetime downloads for this tool (see module doc comment), and createdAt
// never changes after upload so HF's cursor pagination stays stable across a
// walk that can span days. Returns the cursor to resume from next time; null
// means the listing was exhausted (caller should wrap back to start).
// `complete` is true only when the listing was exhausted cleanly (a null
// nextCursor after a successful page) -- false on an error or a limit-filled
// page, so callers know a null nextCursor isn't a fake "wrapped" signal.
// `newestCreatedAtMs` is the max createdAt seen this call (null if none) --
// used to seed the recent-sweep bookmark from the backlog walk's first page.
export async function scanNewRepos(
  limit: number,
  startCursor?: string
): Promise<{ repoIds: string[]; nextCursor: string | null; complete: boolean; newestCreatedAtMs: number | null }> {
  const { searchHfGgufModels } = await import("./hf.js");
  const found: string[] = [];
  let cursor: string | null = startCursor ?? null;
  let scanned = 0;
  let complete = false;
  let newestCreatedAtMs: number | null = null;

  while (scanned < limit) {
    try {
      const page = await searchHfGgufModels("", HF_INDEX_TIMEOUT_MS, "backlog-walk", {
        sort: "createdAt",
        direction: -1,
        limit: Math.min(50, limit - scanned),
        cursor: cursor ?? undefined,
      });
      if (page.items.length === 0) {
        // Empty page: nothing more to scan. Stop regardless of nextCursor to
        // avoid an infinite loop if HF ever returns a non-null cursor with an
        // empty page. complete only if the listing truly ended.
        complete = page.nextCursor == null;
        break;
      }
      for (const item of page.items) {
        found.push(item.id);
        if (item.created_at) {
          const ts = Date.parse(item.created_at);
          if (Number.isFinite(ts) && (newestCreatedAtMs == null || ts > newestCreatedAtMs)) {
            newestCreatedAtMs = ts;
          }
        }
      }
      scanned += page.items.length;
      cursor = page.nextCursor;
      if (!cursor) {
        complete = true;
        break;
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.warn(`[hf-index] scanNewRepos discovery error: ${msg}`);
      break;
    }
  }

  return { repoIds: found, nextCursor: cursor, complete, newestCreatedAtMs };
}

// Incremental "what's new" sweep: sort by createdAt descending, keep a
// bookmark of the newest timestamp already indexed, and stop paging as soon
// as we hit a repo older than that bookmark. That's a handful of requests per
// tick instead of walking the whole catalog. `complete` is true only when the
// sweep actually reached the bookmark (or exhausted the listing) -- an error
// or a limit-filled page returns complete=false, and the caller must NOT
// advance the bookmark in that case (advancing past an unreached boundary
// would skip every repo between the last page and the old bookmark, since the
// new bookmark would already be past them).
export async function scanRecentRepos(
  limit: number,
  bookmarkMs?: number
): Promise<{ repoIds: string[]; newestSeenMs: number | null; complete: boolean }> {
  const { searchHfGgufModels } = await import("./hf.js");
  const found: string[] = [];
  let cursor: string | null = null;
  let newestSeenMs: number | null = null;
  let scanned = 0;
  let complete = false;

  while (scanned < limit) {
    let page;
    try {
      page = await searchHfGgufModels("", HF_INDEX_TIMEOUT_MS, "recent-sweep", {
        sort: "createdAt",
        direction: -1,
        limit: Math.min(50, limit - scanned),
        cursor: cursor ?? undefined,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.warn(`[hf-index] scanRecentRepos sweep error: ${msg}`);
      break;
    }

    if (page.items.length === 0) {
      // Empty page: nothing more to scan. Stop regardless of nextCursor to
      // avoid an infinite loop if HF ever returns a non-null cursor with an
      // empty page. complete only if the listing truly ended.
      complete = page.nextCursor == null;
      break;
    }

    let hitBookmark = false;
    for (const item of page.items) {
      const ts = item.created_at ? Date.parse(item.created_at) : 0;
      if (Number.isFinite(ts) && ts > 0) {
        if (newestSeenMs == null || ts > newestSeenMs) newestSeenMs = ts;
      }
      // Stop when we reach a repo older than bookmark; keep == as overlap
      // so a same-ms update isn't missed (idempotent re-scan is fine).
      if (bookmarkMs != null && ts !== 0 && ts < bookmarkMs) {
        hitBookmark = true;
        break;
      }
      found.push(item.id);
      scanned++;
      if (scanned >= limit) break;
    }

    if (hitBookmark) {
      complete = true;
      break;
    }
    cursor = page.nextCursor;
    if (!cursor) {
      complete = true;
      break;
    }
    // bookmarkMs set and page had no overlap yet → continue paging
    // If bookmarkMs undefined (first run) we just fill up to limit.
  }

  return { repoIds: found, newestSeenMs, complete };
}

// Incremental "what changed" sweep: sort by lastModified descending, keep a
// bookmark of the newest lastModified already covered, and stop paging as
// soon as we hit a repo older than that bookmark. Structurally close to
// scanRecentRepos above, but on the mutable lastModified field instead of
// the immutable createdAt one -- this is what catches content changes
// (new/updated GGUF files, or a file removed) on repos already in the
// index, cheaply, instead of relying on the 24h-timer staleness refresh to
// eventually stumble onto them.
//
// In the steady-state case (bookmark from last tick is recent), this
// reaches the bookmark within the first page and `startCursor` is unused --
// there's no need to resume anything when the whole gap fits in one call.
// The resume cursor exists for the *burst* case: more repos changed since
// the bookmark than `limit` allows in one tick. Without it, every tick
// would restart at position #1 of the lastModified-desc listing and, since
// that ranking is roughly stable minus new arrivals, keep re-covering
// roughly the same top slice instead of making progress toward the older
// (but still-unindexed) end of the burst -- the bookmark would never
// advance until the burst rate dropped below `limit`/tick. Passing the
// cursor this call stopped at back in as `startCursor` next time fixes
// that: each tick reaches further into the burst instead of restarting.
//
// Resuming a cursor across this short a gap (one tick, ~60s, only while
// actively mid-burst) is safe in a way the multi-day backlog cursor in
// scanNewRepos is not: the only failure mode is a repo that gets touched
// again in that ~60s window and jumps back ahead of the stored cursor
// position -- not a real loss, since it'll simply be re-covered whenever a
// sweep finally completes and the bookmark advances past it. A multi-day
// resume under the same mutable sort key would see far more reshuffling
// and could plausibly skip things for good, which is why that walk uses
// immutable createdAt instead.
//
// `complete` is true only when the sweep actually reached the bookmark (or
// exhausted the listing) -- same contract as scanRecentRepos: the caller
// must NOT advance the bookmark on an incomplete sweep (error or
// limit-filled page), since that would skip every repo between the last
// page examined and the true old bookmark. `resumeCursor` is null exactly
// when complete is true (nothing left to resume); otherwise it's where this
// call stopped, for the caller to pass back in as `startCursor` next tick.
//
// Note: with `bookmarkMs` undefined (no bookmark set yet -- a fresh
// database), `complete` will practically never become true on its own,
// since there's no bookmark to hit and this walk would otherwise have to
// reach the literal end of the entire catalog first. That's fine -- it's
// the caller's job (runIndexTick) to special-case "no bookmark yet" as a
// one-time seed from whatever this first call sees, rather than waiting for
// `complete`. This function itself has no opinion on that; it just reports
// what it found.
export async function scanLastModifiedRepos(
  limit: number,
  bookmarkMs?: number,
  startCursor?: string
): Promise<{ repoIds: string[]; newestSeenMs: number | null; complete: boolean; resumeCursor: string | null }> {
  const { searchHfGgufModels } = await import("./hf.js");
  const found: string[] = [];
  let cursor: string | null = startCursor ?? null;
  let newestSeenMs: number | null = null;
  let scanned = 0;
  let complete = false;

  while (scanned < limit) {
    let page;
    try {
      page = await searchHfGgufModels("", HF_INDEX_TIMEOUT_MS, "lastmodified-sweep", {
        sort: "lastModified",
        direction: -1,
        limit: Math.min(50, limit - scanned),
        cursor: cursor ?? undefined,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.warn(`[hf-index] scanLastModifiedRepos sweep error: ${msg}`);
      break;
    }

    if (page.items.length === 0) {
      // Empty page: nothing more to scan. Stop regardless of nextCursor to
      // avoid an infinite loop if HF ever returns a non-null cursor with an
      // empty page. complete only if the listing truly ended.
      complete = page.nextCursor == null;
      break;
    }

    let hitBookmark = false;
    for (const item of page.items) {
      const ts = item.last_modified ? Date.parse(item.last_modified) : 0;
      if (Number.isFinite(ts) && ts > 0) {
        if (newestSeenMs == null || ts > newestSeenMs) newestSeenMs = ts;
      }
      // Stop when we reach a repo older than bookmark; keep == as overlap
      // so a same-ms update isn't missed (idempotent re-scan is fine).
      if (bookmarkMs != null && ts !== 0 && ts < bookmarkMs) {
        hitBookmark = true;
        break;
      }
      found.push(item.id);
      scanned++;
      if (scanned >= limit) break;
    }

    if (hitBookmark) {
      complete = true;
      break;
    }
    cursor = page.nextCursor;
    if (!cursor) {
      complete = true;
      break;
    }
    // bookmarkMs set and page had no overlap yet → continue paging
    // If bookmarkMs undefined (first run) we just fill up to limit.
  }

  return { repoIds: found, newestSeenMs, complete, resumeCursor: complete ? null : cursor };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// Background service state.
let refreshTimer: ReturnType<typeof setTimeout> | null = null;
let serviceStarted = false;
let isRunning = false;
let lastTickAt: number | null = null;
let totalIndexed = 0;

export function getHfIndexStatus(): {
  isRunning: boolean;
  lastTickAt: number | null;
  totalEntries: number;
  totalIndexed: number;
  hfTokenConfigured: boolean;
  backlogBatchSize: number;
  recentBatchSize: number;
  lastModifiedBatchSize: number;
  maxReposPerTick: number;
  rateLimits: ReturnType<typeof getRateLimitStatus>;
  backlogComplete: boolean;
  recentBookmark: number | null;
  lastModifiedBookmark: number | null;
  // True while the lastModified sweep is mid-burst (more repos changed since
  // the bookmark than one tick's batch could cover) and resuming forward
  // across ticks instead of having caught up. Healthy operation is this
  // being false almost always; frequently true means lastModifiedBatchSize
  // is undersized for real HF activity.
  lastModifiedSweepCatchingUp: boolean;
  staleRefreshMode: "background" | "on-demand";
} {
  return {
    isRunning,
    lastTickAt,
    totalEntries: countHfGgufEntries(),
    totalIndexed,
    hfTokenConfigured: isHfTokenConfigured(),
    backlogBatchSize: getBacklogBatchSize(),
    recentBatchSize: getRecentBatchSize(),
    lastModifiedBatchSize: getLastModifiedBatchSize(),
    maxReposPerTick: getMaxReposPerTick(),
    rateLimits: getRateLimitStatus(),
    backlogComplete: isBacklogComplete(),
    recentBookmark: getRecentBookmark() ?? null,
    lastModifiedBookmark: getLastModifiedBookmark() ?? null,
    lastModifiedSweepCatchingUp: getLastModifiedResumeCursor() != null,
    staleRefreshMode: getStaleRefreshMode(),
  };
}

// Run one index tick: a one-time backlog walk (until it wraps once) plus a
// perpetual recent top-up and last-modified sweep, plus staleness refresh
// (unless HF_INDEX_STALENESS_REFRESH_MODE=on-demand, see getStaleRefreshMode).
// See the module doc comment for the full shape and why deletion isn't
// handled by a background crawl here.
export async function runIndexTick(): Promise<{ indexed: number; repos: number }> {
  if (isRunning) return { indexed: 0, repos: 0 };
  isRunning = true;
  let indexedTotal = 0;
  const now = Date.now();
  const backlogBatch = getBacklogBatchSize();
  const recentBatch = getRecentBatchSize();
  const lastModifiedBatch = getLastModifiedBatchSize();
  const maxStale = getMaxReposPerTick();

  if (process.env.LOG_LEVEL === "debug") {
    const rl = getRateLimitStatus();
    log.debug(
      `[hf-index] tick start: token=${isHfTokenConfigured() ? "yes" : "no"} backlogBatch=${backlogBatch} recentBatch=${recentBatch} lastModifiedBatch=${lastModifiedBatch} staleCap=${maxStale} staleRefreshMode=${getStaleRefreshMode()} backlogComplete=${isBacklogComplete()} recentBookmark=${getRecentBookmark() ?? "none"} lastModifiedBookmark=${getLastModifiedBookmark() ?? "none"} lastModifiedResume=${getLastModifiedResumeCursor() != null ? "mid-burst" : "none"} window api ${rl.api.usedInWindow}/${rl.api.limit} resolvers ${rl.resolvers.usedInWindow}/${rl.resolvers.limit}`
    );
  }

  // Collect every repo we intend to scan this tick into a Set so the same
  // repo can't be re-scanned by the backlog walk, the recent sweep, and
  // staleness refresh in one tick (wastes API quota; the index is idempotent
  // so a duplicate scan has no other effect).
  const toScan = new Set<string>();

  try {
    if (!isBacklogComplete()) {
      const startCursor = getBacklogCursor();
      const { repoIds, nextCursor, complete, newestCreatedAtMs } = await scanNewRepos(backlogBatch, startCursor);
      for (const repoId of repoIds) toScan.add(repoId);
      setBacklogCursor(nextCursor);
      // Seed the top-up bookmark from the very first page, not just at wrap
      // -- lets the recency sweep start covering new uploads from tick 1
      // instead of waiting for the (multi-week) backlog walk to finish.
      if (getRecentBookmark() == null && newestCreatedAtMs != null) {
        setRecentBookmark(newestCreatedAtMs);
      }
      if (complete && nextCursor === null) {
        setBacklogComplete(true);
        log.info("[hf-index] backlog walk complete -- newest→oldest catalog fully covered, switching to recent-only top-up");
      }
    }

    // Runs every tick, from tick 1, independent of backlog progress.
    const bookmarkMs = getRecentBookmark();
    const { repoIds: recentIds, newestSeenMs, complete: recentComplete } = await scanRecentRepos(recentBatch, bookmarkMs);
    for (const repoId of recentIds) toScan.add(repoId);
    // Only advance the bookmark when the sweep actually reached it (or
    // exhausted the listing). Advancing after an error- or limit-truncated
    // sweep would jump the bookmark past repos that were never returned,
    // permanently skipping them.
    if (recentComplete && newestSeenMs != null && (bookmarkMs == null || newestSeenMs > bookmarkMs)) {
      setRecentBookmark(newestSeenMs);
    }
    if (recentIds.length > 0) {
      log.info(`[hf-index] recent sweep: ${recentIds.length} repos (bookmark ${bookmarkMs ? new Date(bookmarkMs).toISOString() : "none"} → ${newestSeenMs ? new Date(newestSeenMs).toISOString() : "none"})`);
    }

    // Runs every tick, independent of backlog progress -- catches content
    // changes (new/updated/removed GGUF files) on repos already in the
    // index far more cheaply than the staleness refresh below, since it
    // only pages through what HF itself reports as recently touched instead
    // of blind-refetching every repo's file tree on a timer.
    const lastModifiedBookmarkMs = getLastModifiedBookmark();
    // Only set while a prior tick's sweep hit its batch cap before reaching
    // the bookmark (a burst) -- lets this tick continue forward from where
    // that one stopped instead of restarting at the top and re-covering the
    // same slice. See scanLastModifiedRepos's doc comment.
    const resumeCursor = getLastModifiedResumeCursor();
    const resumeNewestMs = getLastModifiedResumeNewest();
    const {
      repoIds: changedIds,
      newestSeenMs: newestModifiedMsThisCall,
      complete: lastModifiedComplete,
      resumeCursor: nextResumeCursor,
    } = await scanLastModifiedRepos(lastModifiedBatch, lastModifiedBookmarkMs, resumeCursor);
    for (const repoId of changedIds) toScan.add(repoId);
    // Combine with whatever a prior incomplete call already saw, so a
    // completing tick advances the bookmark to the true newest seen across
    // the whole multi-tick burst, not just this last leg of it.
    const newestModifiedMs =
      resumeNewestMs != null
        ? newestModifiedMsThisCall != null
          ? Math.max(resumeNewestMs, newestModifiedMsThisCall)
          : resumeNewestMs
        : newestModifiedMsThisCall;
    let lastModifiedOutcome: "seeded" | "complete" | "resuming";
    if (lastModifiedBookmarkMs == null) {
      // First run ever, no bookmark yet: don't try to walk backward through
      // the whole catalog to "complete" a sweep that has nothing real to
      // catch up on -- the createdAt backlog walk above is already
      // responsible for visiting every existing repo at least once. Just
      // take whatever this one call saw as the starting line and watch for
      // changes from here on -- the same shortcut scanRecentRepos gets for
      // free via its bookmark being seeded off the backlog walk's first
      // page. Without this, a fresh database would make this sweep crawl
      // the entire lastModified-sorted catalog once (its own full pass,
      // redundant with the backlog walk already doing that in a different
      // order) before it could ever settle into its normal cheap mode.
      lastModifiedOutcome = "seeded";
      setLastModifiedResumeCursor(null);
      setLastModifiedResumeNewest(null);
      if (newestModifiedMs != null) {
        setLastModifiedBookmark(newestModifiedMs);
      }
    } else if (lastModifiedComplete) {
      // Caught up -- clear burst-resume state and advance the bookmark.
      // Same "only advance on a completed sweep" rule as the recent
      // bookmark above -- an error- or limit-truncated sweep must not jump
      // the bookmark past repos it never actually returned.
      lastModifiedOutcome = "complete";
      setLastModifiedResumeCursor(null);
      setLastModifiedResumeNewest(null);
      if (newestModifiedMs != null && newestModifiedMs > lastModifiedBookmarkMs) {
        setLastModifiedBookmark(newestModifiedMs);
      }
    } else {
      // Still mid-burst: persist where this call stopped so the next tick
      // resumes forward instead of restarting at the top.
      lastModifiedOutcome = "resuming";
      setLastModifiedResumeCursor(nextResumeCursor);
      if (newestModifiedMs != null) setLastModifiedResumeNewest(newestModifiedMs);
    }
    if (changedIds.length > 0 || lastModifiedOutcome === "seeded") {
      const suffix =
        lastModifiedOutcome === "seeded"
          ? " (first run -- bookmark seeded, not walking full history)"
          : lastModifiedOutcome === "resuming"
            ? " (mid-burst, resuming next tick)"
            : "";
      log.info(
        `[hf-index] lastModified sweep: ${changedIds.length} repos changed${suffix} (bookmark ${lastModifiedBookmarkMs ? new Date(lastModifiedBookmarkMs).toISOString() : "none"} → ${newestModifiedMs ? new Date(newestModifiedMs).toISOString() : "none"})`
      );
    }

    const staleRepos =
      getStaleRefreshMode() === "background" ? findStaleRepos(now - HF_INDEX_REFRESH_INTERVAL_MS, maxStale) : [];
    for (const repoId of staleRepos) toScan.add(repoId);

    // No fixed inter-repo delay here: acquireRateLimitSlot() (invoked by
    // every hfFetch() call inside scanHfRepo) already throttles precisely
    // against HF's real published quota via the RateLimit/RateLimit-Policy
    // headers. A flat sleep on top of that just wastes wall-clock time --
    // with hundreds of repos queued per tick it could push a tick well past
    // HF_INDEX_TICK_INTERVAL_MS. That used to mean the next scheduled tick
    // was silently skipped entirely (see startHfIndexService, which now
    // self-reschedules from actual completion time instead of a fixed
    // setInterval clock specifically to avoid that).
    for (const repoId of toScan) {
      // Tagged "index-scan" regardless of which of the sweeps above
      // originally surfaced this repoId -- toScan already merged and deduped
      // them, so by this point there's no cheap way to attribute a given
      // repo back to exactly one source (and a repo found by more than one
      // sweep in the same tick couldn't be attributed to just one anyway).
      // "index-scan" vs "on-demand-verify" (verifyRepoInBackground) is the
      // operationally useful distinction: background bulk indexing vs a real
      // worker's hash-lookup needing a specific repo re-checked right now.
      const result = await scanHfRepo(repoId, HF_INDEX_TIMEOUT_MS, "index-scan");
      indexedTotal += result.indexed;
    }

    lastTickAt = now;
  } finally {
    isRunning = false;
  }

  return { indexed: indexedTotal, repos: toScan.size };
}

// Start the background refresh service. Runs a tick immediately
// (non-blocking), then self-reschedules the next tick HF_INDEX_TICK_INTERVAL_MS
// after each one *completes* (not on a fixed setInterval clock -- see below).
export function startHfIndexService(): void {
  if (serviceStarted) return; // already started -- guards against a second
  // concurrent scheduling chain; set synchronously so a second call made
  // before the first tick even finishes is still caught (unlike checking
  // refreshTimer, which isn't assigned until a tick completes below).
  serviceStarted = true;

  // Logged at info level (not gated behind LOG_LEVEL=debug) so a restart
  // makes it immediately visible whether HF_TOKEN (or its accepted aliases,
  // see hf-rate-limit.ts's getHfToken) was actually picked up by this
  // process's env -- the common failure mode is editing orchestrator.env /
  // .env without restarting the service, or editing the wrong file for how
  // this process was launched (see .env.example / orchestrator.env.example).
  const staleRefreshMode = getStaleRefreshMode();
  const staleDesc =
    staleRefreshMode === "background"
      ? `stale cap ${getMaxReposPerTick()}`
      : `stale refresh on-demand (HF_INDEX_STALENESS_REFRESH_MODE=on-demand -- background sweep disabled)`;
  if (isHfTokenConfigured()) {
    log.info(`[hf-index] HF_TOKEN loaded -- using authenticated quotas (backlog batch ${getBacklogBatchSize()}, ${staleDesc})`);
  } else {
    log.info(`[hf-index] HF_TOKEN not loaded -- using anonymous quotas (backlog batch ${getBacklogBatchSize()}, ${staleDesc})`);
  }

  // Previously a plain setInterval(runTick, HF_INDEX_TICK_INTERVAL_MS): once
  // a tick queued more repos than fit in one interval's worth of real HF
  // rate-limit budget (routine during the initial backlog-crawl period, now
  // more so with the lastModified sweep also queuing work every tick) and
  // ran long, the *next* scheduled setInterval firing hit runIndexTick's
  // isRunning guard and was silently thrown away -- a wasted invocation that
  // did nothing, working against this module's own stated goal of keeping
  // the rate-limit window continuously fed (see HF_INDEX_TICK_INTERVAL_MS's
  // comment). Self-rescheduling from each tick's actual completion time
  // instead means a long tick is simply followed immediately by the next
  // one -- still bounded by the real rate limiter, but no scheduled slot is
  // ever thrown away. A tick that finishes early still waits out the rest
  // of the interval, so steady-state cadence is unchanged.
  const scheduleNext = (delayMs: number) => {
    if (!serviceStarted) return; // stopHfIndexService() ran while we were mid-tick
    refreshTimer = setTimeout(runTick, delayMs);
    refreshTimer.unref();
  };

  const runTick = () => {
    const startedAt = Date.now();
    runIndexTick()
      .then((result) => {
        if (result.repos > 0) {
          log.info(`[hf-index] tick complete: ${result.indexed} files indexed from ${result.repos} repos`);
        }
      })
      .catch((err) => {
        log.error("[hf-index] tick failed:", err);
      })
      .finally(() => {
        scheduleNext(Math.max(0, HF_INDEX_TICK_INTERVAL_MS - (Date.now() - startedAt)));
      });
  };

  runTick(); // fire immediately, don't block startup; self-reschedules from here
}

// Stop the background refresh service (called on app shutdown).
export function stopHfIndexService(): void {
  serviceStarted = false;
  if (refreshTimer) {
    clearTimeout(refreshTimer);
    refreshTimer = null;
  }
}
