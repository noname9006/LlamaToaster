// One-time (but safely re-runnable) catch-up for hf_repo_duplicate_fact:
// resolves the fact for every repo_id that ALREADY shares a live hash with
// another repo_id in hf_gguf_index as of whenever this is run, instead of
// waiting for each collision to surface organically through a worker's
// hash-lookup (lookupHfGgufHashes only triggers resolveDuplicateFactInBackground
// the moment some real hash-lookup call actually hits that collision -- see
// hf-index.ts). Without this, an existing mislabeled model (e.g. a file
// attributed to a fork/mirror instead of the real source repo -- see
// hf-index.ts's isBetterMatch doc comment for the
// "mingxianderen/Qwen3.8-27B-GGUF" incident this fix addresses) would sit
// wrong until something happens to re-request its hash.
//
// Usage: npx tsx server/src/scripts/backfill-hf-duplicate-facts.ts
//
// Reads HF_TOKEN and DB_PATH from deploy/orchestrator.env, same as
// backfill-hf-index.ts -- see that script's own comment for the precedence
// rules (shell/CLI env wins over the file).
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
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

loadOrchestratorEnv();

// Imported only after loadOrchestratorEnv() runs -- see backfill-hf-index.ts's
// own comment for why (getDb()/getHfToken() read process.env at use time).
const { getDb } = await import("../db/migrate.js");
const { resolveHfRepoDuplicateFact } = await import("../hf.js");
const { upsertRepoDuplicateFact, HF_INDEX_TIMEOUT_MS } = await import("../hf-index.js");
const { isHfTokenConfigured } = await import("../hf-rate-limit.js");

// Same collision definition as hf-index.ts's repoIdsWithLiveHashCollision:
// only repos sharing a *live* hash with another live repo ever reach the
// duplicate-fact tiebreak (a soft-deleted row loses to a live one outright,
// no fact needed) -- so that's the only set worth resolving here.
function findCollisionRepoIds(): string[] {
  const rows = getDb()
    .prepare(
      `SELECT DISTINCT repo_id FROM hf_gguf_index a
       WHERE deleted_at IS NULL
         AND EXISTS (
           SELECT 1 FROM hf_gguf_index b
           WHERE b.sha256 = a.sha256 AND b.deleted_at IS NULL AND b.repo_id != a.repo_id
         )`
    )
    .all() as { repo_id: string }[];
  return rows.map((r) => r.repo_id);
}

function alreadyResolved(repoId: string): boolean {
  const row = getDb().prepare(`SELECT 1 FROM hf_repo_duplicate_fact WHERE repo_id = ?`).get(repoId);
  return row != null;
}

function repoRevision(repoId: string): string {
  const row = getDb()
    .prepare(`SELECT revision FROM hf_gguf_index WHERE repo_id = ? AND revision IS NOT NULL LIMIT 1`)
    .get(repoId) as { revision: string } | undefined;
  return row?.revision || "main";
}

async function main(): Promise<void> {
  console.log(`[backfill-duplicate-facts] DB_PATH: ${process.env.DB_PATH ?? "(default -- likely wrong, check orchestrator.env)"}`);
  console.log(`[backfill-duplicate-facts] HF_TOKEN configured: ${isHfTokenConfigured() ? "yes" : "no"}`);

  const collisionIds = findCollisionRepoIds();
  const pending = collisionIds.filter((id) => !alreadyResolved(id));
  console.log(
    `[backfill-duplicate-facts] ${collisionIds.length} repos share a live hash with another repo; ${pending.length} not yet resolved -- resolving now`
  );

  let processed = 0;
  let duplicates = 0;
  let errors = 0;
  let skipped = 0;
  for (const repoId of pending) {
    try {
      const revision = repoRevision(repoId);
      const fact = await resolveHfRepoDuplicateFact(repoId, revision, HF_INDEX_TIMEOUT_MS, "backfill-duplicate-facts");
      // Same posture as hf-index.ts's resolveDuplicateFactInBackground: both
      // fields null is overwhelmingly a failed round trip (every real repo
      // has a createdAt), not a genuine "checked, found nothing" -- don't
      // cache it as resolved (this table has no re-check policy), just leave
      // it for a real retry (re-run this script, or an organic hash-lookup).
      if (fact.duplicated_from == null && fact.created_at == null) {
        skipped++;
      } else {
        upsertRepoDuplicateFact({
          repo_id: repoId,
          duplicated_from: fact.duplicated_from,
          created_at: fact.created_at ? Date.parse(fact.created_at) || null : null,
          checked_at: Date.now(),
        });
        if (fact.duplicated_from) {
          duplicates++;
          console.log(`[backfill-duplicate-facts] ${repoId} -- duplicated from ${fact.duplicated_from}`);
        }
      }
    } catch (err) {
      errors++;
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[backfill-duplicate-facts] resolve failed for ${repoId}: ${msg}`);
    }
    // Incremented unconditionally (success, skip, or error) so progress
    // logging always advances -- an early `continue` on the skip branch
    // would silently stall this counter mid-loop.
    processed++;
    if (processed % 50 === 0) {
      console.log(
        `[backfill-duplicate-facts] progress: ${processed}/${pending.length} processed, ${duplicates} duplicates found, ${skipped} skipped (failed round trip, retry later), ${errors} thrown errors`
      );
    }
  }

  console.log(
    `[backfill-duplicate-facts] done: ${processed} processed, ${duplicates} duplicates found, ${skipped} skipped (failed round trip -- re-run this script to retry), ${errors} thrown errors`
  );
  console.log(
    "[backfill-duplicate-facts] any model previously mislabeled to one of the found duplicate repos self-heals on its next worker heartbeat (queue.ts's registerHashVerifiedModelFiles) -- no separate models-table fix needed."
  );
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("[backfill-duplicate-facts] FATAL:", err);
    process.exit(1);
  });
