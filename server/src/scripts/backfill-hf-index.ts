// One-time (but safely re-runnable) catch-up for hf_gguf_index repos that
// were discovered by a past runIndexTick but never actually scanned --
// diagnosed 2026-09-02: runIndexTick used to persist its backlog
// cursor/recent-bookmark/lastModified-bookmark to the DB the instant a batch
// of repo ids was *discovered*, before that batch was actually scanned. Any
// server restart in between (deploy, crash, OOM, pm2 autorestart) left the
// marker already pointing past the batch, silently and permanently dropping
// it -- see hf-index.ts's runIndexTick for the fix (deferred pendingWrites)
// that stops this from happening going forward.
//
// This script only closes the gap for repos ALREADY missing as of past
// restarts: it re-enumerates HF's real `filter=gguf` listing (the exact
// endpoint the crawler itself uses), diffs it against the DB's current
// repo_id set, and runs scanHfRepo() -- the same function the live service
// uses -- against just the missing ones. Not intended to run on a schedule;
// the runIndexTick fix is what prevents the gap from reopening. Re-run this
// manually if you ever want to double check parity again -- it's cheap
// (~200 requests to list, one tree-fetch per missing repo) and idempotent
// (upsertHfGgufEntry is ON CONFLICT DO UPDATE, so scanning an already-indexed
// repo a second time is a harmless no-op).
//
// Usage: npx tsx server/src/scripts/backfill-hf-index.ts
//
// Reads HF_TOKEN and DB_PATH from deploy/orchestrator.env (same file the
// real pm2-launched server reads via ecosystem.server.config.cjs) so this
// runs against the real production DB and quota by default -- override
// either by exporting the env var yourself before running, which takes
// precedence over the file.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

function loadOrchestratorEnv(): void {
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const envPath = join(__dirname, "..", "..", "..", "deploy", "orchestrator.env");
  if (!existsSync(envPath)) return;
  for (const line of readFileSync(envPath, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq);
    const value = trimmed.slice(eq + 1);
    // Shell/CLI-provided env wins over the file, same precedence
    // ecosystem.server.config.cjs's own loader implicitly gets (pm2 sets
    // `env:` wholesale, but here we're layering on top of an already-live
    // process env, so an explicit override must not be clobbered).
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

loadOrchestratorEnv();

// Imported only after loadOrchestratorEnv() runs -- getDb() (migrate.ts)
// reads process.env.DB_PATH the first time it's called (module-level
// singleton), and hf-rate-limit's getHfToken() reads process.env.HF_TOKEN
// per-call, so both need the file's values in place before first use, not
// necessarily before import. Sequencing the import after anyway keeps the
// dependency obvious to a reader.
const { getDb } = await import("../db/migrate.js");
const { scanHfRepo, HF_INDEX_TIMEOUT_MS } = await import("../hf-index.js");
const { searchHfGgufModels } = await import("../hf.js");
const { isHfTokenConfigured } = await import("../hf-rate-limit.js");

async function fetchAllHfRepoIds(): Promise<Set<string>> {
  const ids = new Set<string>();
  let cursor: string | undefined;
  let page = 0;
  while (true) {
    const result = await searchHfGgufModels("", HF_INDEX_TIMEOUT_MS, "backfill-enumerate", {
      limit: 1000,
      cursor,
    });
    for (const item of result.items) ids.add(item.id);
    page++;
    if (page % 20 === 0) console.log(`[backfill] enumerate: page ${page}, ${ids.size} ids so far`);
    if (!result.nextCursor) break;
    cursor = result.nextCursor;
  }
  console.log(`[backfill] enumerate done: ${ids.size} total ids across ${page} pages`);
  return ids;
}

function readDbRepoIds(): Set<string> {
  const rows = getDb()
    .prepare("SELECT DISTINCT repo_id FROM hf_gguf_index")
    .all() as { repo_id: string }[];
  return new Set(rows.map((r) => r.repo_id));
}

async function main(): Promise<void> {
  console.log(`[backfill] DB_PATH: ${process.env.DB_PATH ?? "(default -- likely wrong, check orchestrator.env)"}`);
  console.log(`[backfill] HF_TOKEN configured: ${isHfTokenConfigured() ? "yes" : "no"}`);

  console.log("[backfill] enumerating HF's real filter=gguf listing (a couple minutes)...");
  const hfIds = await fetchAllHfRepoIds();

  const dbIdsBefore = readDbRepoIds();
  console.log(`[backfill] DB has ${dbIdsBefore.size} distinct repo_id before backfill`);

  const missing = [...hfIds].filter((id) => !dbIdsBefore.has(id));
  console.log(`[backfill] missing from DB: ${missing.length} repos (${((missing.length / hfIds.size) * 100).toFixed(2)}%) -- scanning each now`);

  let indexed = 0;
  let scanned = 0;
  let errors = 0;
  for (const repoId of missing) {
    try {
      const result = await scanHfRepo(repoId, HF_INDEX_TIMEOUT_MS, "backfill");
      indexed += result.indexed;
    } catch (err) {
      errors++;
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[backfill] scanHfRepo threw for ${repoId}: ${msg}`);
    }
    scanned++;
    if (scanned % 100 === 0) {
      console.log(`[backfill] progress: ${scanned}/${missing.length} scanned, ${indexed} files indexed so far, ${errors} errors`);
    }
  }

  const dbIdsAfter = readDbRepoIds();
  const stillMissing = missing.filter((id) => !dbIdsAfter.has(id));

  console.log(`[backfill] done: ${scanned} repos scanned, ${indexed} files indexed, ${errors} thrown errors`);
  console.log(`[backfill] DB has ${dbIdsAfter.size} distinct repo_id after backfill (was ${dbIdsBefore.size})`);
  console.log(`[backfill] still missing after this pass: ${stillMissing.length}`);
  if (stillMissing.length > 0 && stillMissing.length <= 50) {
    // A repo can legitimately still be "missing" here without indicating a
    // bug: no .gguf files with a real LFS sha256 (non-LFS/small files,
    // skipped by scanHfRepo -- see its own comment), an empty tree on the
    // default branch, or a repo that genuinely disappeared between the list
    // enumeration above and this scan reaching it. Worth a manual look only
    // if this number is large relative to `missing`.
    console.log(`[backfill] still-missing ids: ${stillMissing.join(", ")}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("[backfill] FATAL:", err);
    process.exit(1);
  });
