import { statSync, existsSync, readdirSync } from "node:fs";
import { join, relative, sep, resolve } from "node:path";
import { createHash, type Hash } from "node:crypto";
import { createReadStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { log } from "./log.js";
import { LocalModelCache, type LocalCacheEntry } from "./local-cache.js";
import { readGgufInfo } from "./gguf.js";
import type { ModelDirFile, LocalModelState } from "../../shared/types.js";

// How often an already-matched entry gets re-verified against the server
// (matches the server's own HF_INDEX_REFRESH_INTERVAL_MS staleness window).
// Without this, a match found once was cached forever and never re-checked,
// so a server-side soft-delete (see server/src/hf-index.ts) would never
// reach an already-matched worker.
const HF_MATCH_RECHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;

function isHfCheckStale(checkedAt: number | undefined): boolean {
  return checkedAt == null || Date.now() - checkedAt > HF_MATCH_RECHECK_INTERVAL_MS;
}

// Same staleness window as HF matching (a GGUF header read is a one-time
// cost per file; only a file whose header came back empty/unreadable is worth
// re-attempting, and not on every single reconciliation).
const GGUF_RECHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;

function isGgufCheckStale(checkedAt: number | undefined): boolean {
  return checkedAt == null || Date.now() - checkedAt > GGUF_RECHECK_INTERVAL_MS;
}

interface ScannedFile {
  path: string;
  size: number;
  mtime: number;
  cachedEntry?: LocalCacheEntry;
}

interface ScanResult {
  newFiles: ScannedFile[];
  modifiedFiles: ScannedFile[];
  unchangedFiles: ScannedFile[];
  missingFiles: string[]; // paths that were in cache but no longer exist
  allFiles: ScannedFile[];
}

/**
 * Scans the model directory and compares with the local cache
 * Returns a breakdown of file states
 */
export async function scanModelDirectory(
  modelDir: string,
  cache: LocalModelCache,
  rawJsonDir?: string
): Promise<ScanResult> {
  const root = resolve(modelDir);
  const rawDir = rawJsonDir ? resolve(rawJsonDir) : resolve(root, "raw");

  // Get all cached entries
  const cachedEntries = await cache.getAll();
  const cachedByPath = new Map(cachedEntries.map((e) => [e.path, e]));

  // Walk the filesystem
  const scannedFiles: ScannedFile[] = [];
  const scannedPaths = new Set<string>();

  function walk(dir: string): void {
    if (resolve(dir) === rawDir) return;
    // Skip worker's own cache directory (model_dir/.cache) - contains DB, not models
    if (resolve(dir) === resolve(root, ".cache")) return;
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
        } else if (entry.isFile() || entry.isSymbolicLink()) {
          if (entry.name.endsWith(".part")) continue;
          try {
            const stat = statSync(full);
            // Defensive for test mocks that return plain object without isFile
            if (typeof (stat as any).isFile === "function" && !(stat as any).isFile()) continue;
            if (typeof (stat as any).isDirectory === "function" && (stat as any).isDirectory()) continue;
            if (entry.name.startsWith(".")) {
              // Skip hidden cache dirs that might be symlinked
              if (entry.name === ".cache" && typeof (stat as any).isDirectory === "function" && (stat as any).isDirectory()) continue;
            }
            const rel = relative(root, full).split(sep).join("/");
            scannedFiles.push({
              path: rel,
              size: (stat as any).size,
              mtime: (stat as any).mtimeMs ?? (stat as any).mtime?.getTime?.() ?? Date.now(),
              cachedEntry: cachedByPath.get(rel),
            });
            scannedPaths.add(rel);
          } catch (e) {
            // statSync failed (broken symlink, file deleted between readdir and stat, permission)
            log.warn(`Skipping ${full}: ${e instanceof Error ? e.message : String(e)}`);
          }
        }
      }
    } catch (err) {
      log.warn(`Failed to scan directory ${dir}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (existsSync(root)) {
    walk(root);
  }

  // Categorize files
  const newFiles: ScannedFile[] = [];
  const modifiedFiles: ScannedFile[] = [];
  const unchangedFiles: ScannedFile[] = [];
  const allFiles: ScannedFile[] = [];

  for (const file of scannedFiles) {
    allFiles.push(file);
    const cached = file.cachedEntry;

    if (!cached) {
      newFiles.push(file);
    } else if (cached.size !== file.size || cached.mtime !== file.mtime) {
      modifiedFiles.push(file);
    } else if (!cached.sha256) {
      // Cached entry exists and size/mtime matches, but no SHA yet (e.g. prior
      // run crashed before hashing completed). Treat as needing hash per spec
      // "Files missing cached hashes" must be hashed, not skipped.
      modifiedFiles.push(file);
    } else {
      unchangedFiles.push(file);
    }
  }

  // Find missing files (in cache but not on disk)
  const missingFiles: string[] = [];
  for (const [cachedPath] of cachedByPath) {
    if (!scannedPaths.has(cachedPath)) {
      missingFiles.push(cachedPath);
    }
  }

  return { newFiles, modifiedFiles, unchangedFiles, missingFiles, allFiles };
}

/**
 * Hashes a file and returns the SHA-256 digest
 * Streaming implementation for large files
 */
export function hashFile(path: string): Promise<{ sha256: string; byteLength: number }> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    let byteLength = 0;
    const rs = createReadStream(path);
    rs.on("data", (chunk: Buffer) => {
      hash.update(chunk);
      byteLength += chunk.length;
    });
    rs.on("end", () => resolve({ sha256: hash.digest("hex"), byteLength }));
    rs.on("error", reject);
  });
}

/**
 * Background hashing queue - processes files sequentially without blocking startup
 */
export class HashingQueue {
  private queue: Array<{
    file: ScannedFile;
    modelDir: string;
    cache: LocalModelCache;
    resolve: (entry: LocalCacheEntry) => void;
    reject: (err: Error) => void;
  }> = [];
  private processing = false;
  private readonly cache: LocalModelCache;
  private readonly modelDir: string;

  constructor(modelDir: string, cache: LocalModelCache) {
    this.modelDir = modelDir;
    this.cache = cache;
  }

  /**
   * Add a file to the hashing queue
   */
  enqueue(file: ScannedFile): Promise<LocalCacheEntry> {
    return new Promise((resolve, reject) => {
      this.queue.push({ file, modelDir: this.modelDir, cache: this.cache, resolve, reject });
      if (!this.processing) {
        this.process();
      }
    });
  }

  /**
   * Process the queue sequentially
   */
  private async process(): Promise<void> {
    if (this.processing || this.queue.length === 0) return;
    this.processing = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift()!;
      try {
        const fullPath = resolve(this.modelDir, item.file.path);
        const { sha256, byteLength } = await hashFile(fullPath);

        // Re-stat after hashing to capture final mtime/size (file may have been modified during hashing)
        // If stat fails (file deleted), treat as missing - will be handled on next scan
        let finalSize = item.file.size;
        let finalMtime = item.file.mtime;
        try {
          const finalStat = statSync(fullPath) as any;
          // Defensive for test mocks returning plain object
          if (finalStat && typeof finalStat.size === "number") finalSize = finalStat.size;
          if (finalStat && typeof finalStat.mtimeMs === "number") finalMtime = finalStat.mtimeMs;
          else if (finalStat && finalStat.mtime instanceof Date) finalMtime = finalStat.mtime.getTime();
          // If size changed during hash, the hash is stale - log and use fresh values
          if (finalSize !== byteLength) {
            log.warn(`File ${item.file.path} size changed during hashing (${byteLength} -> ${finalSize}), hash may be stale`);
          }
        } catch {
          // File deleted during hashing - propagate as missing
          throw Object.assign(new Error(`File disappeared during hashing: ${item.file.path}`), { code: "ENOENT" });
        }

        // Single upsert: hashing interim state; HF resolution (verified/unknown)
        // is done later by resolveHfMetadata, not here.
        const updatedEntry: LocalCacheEntry = {
          path: item.file.path,
          size: finalSize,
          mtime: finalMtime,
          sha256,
          hf_model_id: item.file.cachedEntry?.hf_model_id,
          hf_checked_at: item.file.cachedEntry?.hf_checked_at,
          hf_deleted_at: item.file.cachedEntry?.hf_deleted_at,
          last_verified: Date.now(),
          state: "hashing",
        };
        await this.cache.upsert(updatedEntry);

        // Return the upserted entry
        const finalEntry = await this.cache.get(item.file.path);
        if (finalEntry) {
          item.resolve(finalEntry);
        } else {
          item.reject(new Error(`Failed to retrieve cached entry after hashing ${item.file.path}`));
        }
      } catch (err) {
        item.reject(err instanceof Error ? err : new Error(String(err)));
      }
    }

    this.processing = false;
  }

  /**
   * Wait for all queued items to complete
   */
  async waitForCompletion(): Promise<void> {
    while (this.queue.length > 0 || this.processing) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  getQueueLength(): number {
    return this.queue.length;
  }

  isProcessing(): boolean {
    return this.processing;
  }
}

/**
 * Look up hashes against the server's HF index
 * Returns a map of sha256 -> HF metadata.
 * Throws on batch failure so caller does not incorrectly mark hashes as "unknown"
 * when the API was simply unavailable (stale cache scenario). Caller should
 * catch and leave states unchanged for retry on next reconciliation.
 * Partial success (some batches succeeded) still returns those results; only
 * hard failures with no successful batches will throw and cause caller to skip
 * unknown marking. For partial failures, successful batches are returned and
 * failed batches are left for retry (not marked unknown).
 */
export async function lookupHashes(
  apiBaseUrl: string,
  authToken: string,
  hashes: string[]
): Promise<Map<string, { repo_id: string; filename: string; revision: string; deleted_at: number | null }>> {
  const results = new Map<string, { repo_id: string; filename: string; revision: string; deleted_at: number | null }>();

  if (hashes.length === 0) return results;

  // Process in batches of 100 to avoid large payloads
  const BATCH_SIZE = 100;
  let anySuccess = false;
  let anyFailure = false;
  const failedHashes = new Set<string>();
  for (let i = 0; i < hashes.length; i += BATCH_SIZE) {
    const batch = hashes.slice(i, i + BATCH_SIZE);
    try {
      const res = await fetch(`${apiBaseUrl}/api/models/hash-lookup`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({ hashes: batch }),
      });

      if (!res.ok) {
        log.warn(`Hash lookup failed: ${res.status} ${res.statusText}`);
        anyFailure = true;
        for (const h of batch) failedHashes.add(h);
        continue;
      }

      anySuccess = true;
      const data = (await res.json()) as {
        results: Array<{ sha256: string; repo_id: string; filename: string; revision: string; deleted_at: number | null }>;
      };
      for (const entry of data.results) {
        results.set(entry.sha256, {
          repo_id: entry.repo_id,
          filename: entry.filename,
          revision: entry.revision,
          deleted_at: entry.deleted_at,
        });
      }
    } catch (err) {
      log.warn(`Hash lookup error: ${err instanceof Error ? err.message : String(err)}`);
      anyFailure = true;
      for (const h of batch) failedHashes.add(h);
    }
  }

  // Attach failed set to results for caller inspection so it can avoid marking those as unknown
  // If all batches failed, just return empty map with all hashes marked failed - caller will skip unknown marking and retry later
  (results as any).__failedHashes = failedHashes;
  return results;
}

/**
 * Resolve HF metadata for all cached entries that have hashes but no HF match
 * Leaves entries unchanged if lookup fails (API unavailable) so they will be
 * retried on next reconciliation instead of being incorrectly marked "unknown".
 */
export async function resolveHfMetadata(
  cache: LocalModelCache,
  apiBaseUrl: string,
  authToken: string
): Promise<void> {
  const entries = await cache.getAll();
  const hashesToLookup: string[] = [];

  // Includes already-matched entries whose hf_checked_at has gone stale, not
  // just never-matched ones -- otherwise a match found once was cached
  // forever and a server-side soft-delete would never reach this worker
  // (see HF_MATCH_RECHECK_INTERVAL_MS above).
  //
  // Gated on sha256 presence, NOT on state: HashingQueue persists each
  // digest with state "hashing" and relies on THIS pass to transition those
  // entries to verified/unknown afterward -- excluding "hashing" here
  // deadlocked freshly-hashed files in that state forever (they'd never be
  // rehashed as unchanged on later scans, so nothing ever re-visited them).
  const needsLookup = (entry: LocalCacheEntry) =>
    entry.sha256 != null && (!entry.hf_model_id || isHfCheckStale(entry.hf_checked_at));

  for (const entry of entries) {
    if (needsLookup(entry)) {
      hashesToLookup.push(entry.sha256!);
    }
  }

  if (hashesToLookup.length === 0) return;

  let matches: Map<string, { repo_id: string; filename: string; revision: string; deleted_at: number | null }>;
  try {
    matches = await lookupHashes(apiBaseUrl, authToken, hashesToLookup);
  } catch (err) {
    log.warn(`Hash lookup aborted: ${err instanceof Error ? err.message : String(err)} - will retry on next scan`);
    return;
  }

  const failedHashes = (matches as any).__failedHashes as Set<string> | undefined;

  for (const entry of entries) {
    if (needsLookup(entry)) {
      // If this specific hash's batch failed, skip (retry later) rather than marking unknown
      if (failedHashes?.has(entry.sha256!)) {
        log.warn(`Skipping HF resolution for ${entry.path} - lookup batch failed, will retry`);
        continue;
      }
      const match = matches.get(entry.sha256!);
      if (match) {
        await cache.updateHfMatch(entry.path, `${match.repo_id}/${match.filename}`, match.deleted_at);
      } else {
        await cache.updateState(entry.path, "unknown");
      }
    }
  }
}

/**
 * One-time per-file GGUF header read, backfilling n_layer/mtp_layers/
 * quant/param_count onto each cache entry (see
 * worker/src/gguf.ts's readGgufInfo). Gated on a staleness window so a
 * header that came back empty (unreadable/corrupt file) isn't re-read on
 * every reconciliation, and so a real model's header gets a fresh look at
 * most once a day -- not because it's likely to change, but so a file
 * replaced in place (new bytes, same path, e.g. a re-download) doesn't keep
 * stale metadata forever.
 */
export async function backfillGgufMetadata(modelDir: string, cache: LocalModelCache): Promise<void> {
  const entries = await cache.getAll();
  for (const entry of entries) {
    if (!isGgufCheckStale(entry.gguf_checked_at)) continue;
    const fullPath = resolve(modelDir, entry.path);
    if (!existsSync(fullPath)) continue;
    const info = await readGgufInfo(fullPath);
    await cache.upsert({
      ...entry,
      n_layer: info.n_layer ?? entry.n_layer,
      mtp_layers: info.mtp_layers ?? entry.mtp_layers,
      quant: info.quant ?? entry.quant,
      param_count: info.param_count ?? entry.param_count,
      trained_ctx: info.trained_ctx ?? entry.trained_ctx,
      n_head_kv: info.n_head_kv ?? entry.n_head_kv,
      head_dim_k: info.head_dim_k ?? entry.head_dim_k,
      head_dim_v: info.head_dim_v ?? entry.head_dim_v,
      n_embd: info.n_embd ?? entry.n_embd,
      n_head: info.n_head ?? entry.n_head,
      sliding_window: info.sliding_window ?? entry.sliding_window,
      gguf_checked_at: Date.now(),
      last_verified: entry.last_verified,
    });
    if (info.debugReason) {
      log.warn(`[model-scan] GGUF metadata for ${entry.path} incomplete: ${info.debugReason}`);
    }
  }
}

/**
 * Full startup reconciliation flow
 * Scans, validates cache, hashes new/modified files, resolves HF metadata
 */
export async function runStartupReconciliation(
  modelDir: string,
  cache: LocalModelCache,
  apiBaseUrl: string,
  authToken: string,
  rawJsonDir?: string
): Promise<{ scanned: number; hashed: number; resolved: number; missing: number }> {
  log.info("[model-scan] Starting model directory reconciliation...");

  // Step 1: Scan
  const scanResult = await scanModelDirectory(modelDir, cache, rawJsonDir);
  log.info(
    `[model-scan] Found ${scanResult.allFiles.length} files: ` +
    `${scanResult.newFiles.length} new, ${scanResult.modifiedFiles.length} modified, ` +
    `${scanResult.unchangedFiles.length} unchanged, ${scanResult.missingFiles.length} missing`
  );

  // Step 2: Update cache for unchanged files (just update last_verified)
  for (const file of scanResult.unchangedFiles) {
    if (file.cachedEntry) {
      await cache.upsert({
        ...file.cachedEntry,
        last_verified: Date.now(),
      });
    }
  }

  // Step 3: Mark missing files (keep rows as 'missing' for UI; purge after TTL in step 3b)
  for (const path of scanResult.missingFiles) {
    await cache.updateState(path, "missing");
  }

  // Purge stale 'missing' entries that have been missing for >7 days (avoids unbounded growth)
  const entries = await cache.getAll();
  const MISSING_TTL_MS = 7 * 24 * 60 * 60 * 1000;
  const now = Date.now();
  for (const e of entries) {
    if (e.state === "missing" && now - e.last_verified > MISSING_TTL_MS) {
      await cache.delete(e.path);
    }
  }

  // Step 3c: Persist 'detected' for new files so UI shows them immediately (hashing is async)
  for (const file of scanResult.newFiles) {
    await cache.upsert({
      path: file.path,
      size: file.size,
      mtime: file.mtime,
      sha256: undefined,
      hf_model_id: undefined,
      last_verified: Date.now(),
      state: "detected",
    });
  }
  // For modified files, mark as 'modified' immediately (will become hashing/verified after rehash)
  for (const file of scanResult.modifiedFiles) {
    // Preserve existing sha/hf if any until rehashed, but flip state
    const existing = file.cachedEntry;
    await cache.upsert({
      path: file.path,
      size: file.size,
      mtime: file.mtime,
      sha256: existing?.sha256,
      hf_model_id: existing?.hf_model_id,
      hf_checked_at: existing?.hf_checked_at,
      hf_deleted_at: existing?.hf_deleted_at,
      last_verified: Date.now(),
      state: "modified",
    });
  }

  // Step 4: Hash new and modified files (in background)
  const hashingQueue = new HashingQueue(modelDir, cache);
  const toHash = [...scanResult.newFiles, ...scanResult.modifiedFiles];

  for (const file of toHash) {
    // Don't await - let them queue up
    hashingQueue.enqueue(file).catch((err) => {
      const isNotFound = (err as any)?.code === "ENOENT" || /ENOENT|disappeared/.test(err.message);
      if (isNotFound) {
        log.warn(`File disappeared before hashing completed: ${file.path} - will be marked missing on next scan`);
        // Don't mark corrupted for vanished files; let next scan's missing detection handle it
        return;
      }
      log.error(`Failed to hash ${file.path}: ${err.message}`);
      // Use upsert so new files (no prior row) also get a persistent corrupted entry
      // (e.g., permission denied, read error, zero-byte truncated file)
      cache
        .upsert({
          path: file.path,
          size: file.size,
          mtime: file.mtime,
          sha256: undefined,
          hf_model_id: file.cachedEntry?.hf_model_id,
          hf_checked_at: file.cachedEntry?.hf_checked_at,
          hf_deleted_at: file.cachedEntry?.hf_deleted_at,
          last_verified: Date.now(),
          state: "corrupted",
        })
        .catch(() => {});
    });
  }

  // Wait for all hashing to complete
  await hashingQueue.waitForCompletion();

  // Step 5: Resolve HF metadata for all hashed files
  await resolveHfMetadata(cache, apiBaseUrl, authToken);

  // Step 5b: Read each file's GGUF header once (n_layer/quant/param_count/
  // ...) into the cache, so a file dropped into model_dir by hand gets the
  // same per-file metadata a download reports to the server's catalog --
  // without this its Models-page badge shows "?" for quant and a filename-
  // guess (or HF's repo-level, possibly-wrong value) for params.
  await backfillGgufMetadata(modelDir, cache);

  // Count results
  const finalEntries = await cache.getAll();
  const hashed = finalEntries.filter((e) => e.sha256 && (scanResult.newFiles.some((f) => f.path === e.path) || scanResult.modifiedFiles.some((f) => f.path === e.path))).length;
  const resolved = finalEntries.filter((e) => e.hf_model_id && e.state === "verified").length;

  log.info(
    `[model-scan] Reconciliation complete: ${scanResult.allFiles.length} scanned, ` +
    `${hashed} hashed, ${resolved} resolved, ${scanResult.missingFiles.length} missing`
  );

  return {
    scanned: scanResult.allFiles.length,
    hashed,
    resolved,
    missing: scanResult.missingFiles.length,
  };
}

/**
 * Refresh models - can be called manually from the UI
 * Forces a full reconciliation
 */
export async function refreshModels(
  modelDir: string,
  cache: LocalModelCache,
  apiBaseUrl: string,
  authToken: string,
  rawJsonDir?: string
): Promise<{ scanned: number; hashed: number; resolved: number; missing: number }> {
  log.info("[model-scan] Manual refresh triggered");
  return runStartupReconciliation(modelDir, cache, apiBaseUrl, authToken, rawJsonDir);
}

// Reconstructs the reported hf_match shape from a cache entry's compact
// "repo_id/filename" hf_model_id string. Shared by every call site below.
//
// HF repo ids are always exactly "namespace/name" (two segments) -- so the
// first two segments are the repo id and everything after is the filename.
// Splitting on just the first "/" (the old behavior) truncated repo_id down
// to the bare namespace and dumped the real repo name into the filename,
// producing broken /blob/main/ links.
function buildHfMatch(entry: LocalCacheEntry): ModelDirFile["hf_match"] {
  if (!entry.hf_model_id) return null;
  const parts = entry.hf_model_id.split("/");
  return {
    repo_id: parts.slice(0, 2).join("/"),
    filename: parts.slice(2).join("/"),
    revision: "main",
    deleted: Boolean(entry.hf_deleted_at),
  };
}

/**
 * Get model files with state for heartbeat reporting
 */
export async function getModelFilesWithState(
  cache: LocalModelCache,
  modelDir: string
): Promise<ModelDirFile[]> {
  const entries = await cache.getAll();
  const results: ModelDirFile[] = [];

  for (const entry of entries) {
    const fullPath = resolve(modelDir, entry.path);
    const exists = existsSync(fullPath);

    if (!exists) {
      // File is missing - report with missing state
      results.push({
        path: entry.path,
        size_bytes: entry.size,
        sha256: entry.sha256,
        state: "missing",
        hf_match: buildHfMatch(entry),
      });
      continue;
    }

    let stat: ReturnType<typeof statSync>;
    try {
      stat = statSync(fullPath);
    } catch {
      // Deleted between existsSync and statSync (TOCTOU) - treat as missing
      results.push({
        path: entry.path,
        size_bytes: entry.size,
        sha256: entry.sha256,
        state: "missing",
        hf_match: buildHfMatch(entry),
      });
      continue;
    }
    const state = entry.state;

    // Check if file was modified since last verification
    let currentState = state;
    if (stat.mtimeMs !== entry.mtime || stat.size !== entry.size) {
      currentState = "modified";
    }

    // GGUF header metadata rides along so the server can adopt it for a
    // hand-dropped (never-downloaded) file -- see backfillGgufMetadata. Only
    // fields with a real value are sent; absent fields stay absent so the
    // server can distinguish "checked, was null" from "not checked at all".
    const ggufFields: Pick<
      ModelDirFile,
      | "n_layer"
      | "mtp_layers"
      | "quant"
      | "param_count"
      | "trained_ctx"
      | "n_head_kv"
      | "head_dim_k"
      | "head_dim_v"
      | "n_embd"
      | "n_head"
      | "sliding_window"
    > = {};
    if (entry.n_layer != null) ggufFields.n_layer = entry.n_layer;
    if (entry.mtp_layers != null) ggufFields.mtp_layers = entry.mtp_layers;
    if (entry.quant != null) ggufFields.quant = entry.quant;
    if (entry.param_count != null) ggufFields.param_count = entry.param_count;
    if (entry.trained_ctx != null) ggufFields.trained_ctx = entry.trained_ctx;
    if (entry.n_head_kv != null) ggufFields.n_head_kv = entry.n_head_kv;
    if (entry.head_dim_k != null) ggufFields.head_dim_k = entry.head_dim_k;
    if (entry.head_dim_v != null) ggufFields.head_dim_v = entry.head_dim_v;
    if (entry.n_embd != null) ggufFields.n_embd = entry.n_embd;
    if (entry.n_head != null) ggufFields.n_head = entry.n_head;
    if (entry.sliding_window != null) ggufFields.sliding_window = entry.sliding_window;

    results.push({
      path: entry.path,
      size_bytes: stat.size,
      sha256: entry.sha256,
      state: currentState,
      hf_match: buildHfMatch(entry),
      ...ggufFields,
    });
  }

  return results;
}