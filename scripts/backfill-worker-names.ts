// One-off fix for runs created before the "vps-cpu"/"local-6600xt" ->
// "Remote"/"Local" worker rename: those rows still have the old name baked
// in -- runs.worker_name is a plain snapshot taken at trigger time (see
// server/src/db/schema.sql), not derived live from the current worker
// config, so renaming the config alone never touches historical rows.
//
// Safe to run more than once -- it only touches rows still holding an old
// name, and prints exactly what it changed.
//
// Usage (from the repo root, against the real DB -- edit RENAMES first if
// your old names differed from the ones below):
//   DB_PATH=/path/to/llamatoaster.db npx tsx scripts/backfill-worker-names.ts

import Database from "better-sqlite3";
import { join } from "node:path";

const RENAMES: Record<string, string> = {
  "vps-cpu": "Remote",
  "local-6600xt": "Local",
};

const dbPath = process.env.DB_PATH ?? join(process.cwd(), "llamatoaster.db");
const db = new Database(dbPath);

console.log(`backfilling runs.worker_name in ${dbPath}`);
for (const [oldName, newName] of Object.entries(RENAMES)) {
  const result = db.prepare("UPDATE runs SET worker_name = ? WHERE worker_name = ?").run(newName, oldName);
  console.log(`  "${oldName}" -> "${newName}": ${result.changes} row(s) updated`);
}
db.close();
