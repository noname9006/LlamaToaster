// One-off repair for catalog rows squatted through POST /api/models before
// the audit 2026-10-11 N3 fix -- see server/src/db/catalogRepair.ts for what
// counts as a squat and what the repair changes.
//
// Dry run by default: prints what it would change and exits. Back the DB up
// first (scripts/backup-db.sh), then re-run with --apply.
//
// Usage (from the repo root, with the server stopped):
//   DB_PATH=/path/to/llamatoaster.db npx tsx scripts/repair-catalog-n3.ts
//   DB_PATH=/path/to/llamatoaster.db npx tsx scripts/repair-catalog-n3.ts --apply

import Database from "better-sqlite3";
import { join } from "node:path";
import { findN3Repairs, applyN3Repairs } from "../server/src/db/catalogRepair.js";

const dbPath = process.env.DB_PATH ?? join(process.cwd(), "llamatoaster.db");
const apply = process.argv.includes("--apply");
const db = new Database(dbPath, { fileMustExist: true });

const { candidates, review } = findN3Repairs(db);
console.log(`${dbPath}: ${candidates.length} squatted row(s), ${review.length} locked row(s) to review`);
for (const c of candidates) {
  console.log(
    `  release ${c.id}  owner=${c.created_by}  file=${c.filename}` +
      `  hf=${c.hf_repo ?? "-"}/${c.hf_file ?? "-"}${c.locked ? "  (lock claimed)" : ""}` +
      (c.clearIdentity ? "  -> HF identity cleared (not in index)" : "")
  );
}
for (const r of review) {
  console.log(`  review  ${r.id}  locked to ${r.hf_repo ?? "-"}/${r.hf_file ?? "-"}, which the HF index doesn't list`);
}

if (apply && candidates.length > 0) {
  applyN3Repairs(db, candidates);
  console.log(`applied: ${candidates.length} row(s) released to the shared catalog`);
} else if (candidates.length > 0) {
  console.log("dry run -- nothing changed. Back up the DB, then re-run with --apply.");
}
db.close();
