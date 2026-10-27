import type Database from "better-sqlite3";
import { deriveModelId } from "./repo.js";
import type { ModelSource } from "../../../shared/types.js";

// One-off repair for audit 2026-10-11 N3: before the fix, POST /api/models
// upserted the request body, so any signed-in user could create a catalog row
// under a file's sha256 (becoming its owner, which then blocked the real
// download from correcting it) and claim the download lock on it.
//
// A row that is a squat candidate: a 64-hex id with an owner that is NOT the
// id POST /api/models derives when no id is given (those are ordinary,
// legitimately owned rows -- sha256 of "hf:repo/file" or "local:name:size",
// never a file's hash). Neither the heartbeat nor the download callback ever
// set created_by, and imports use "imported:" ids, so every candidate came
// from a client-supplied id on that route.
//
// Repair hands each candidate back to the shared catalog: no owner, no lock,
// no client-written metadata (header fields are re-merged from
// model_header_reports on the next heartbeat), and its HF identity is cleared
// unless it is a live hf_gguf_index pair -- the heartbeat re-fills it from the
// index. Download-locked shared rows whose identity isn't in the index are
// only listed: a real download of a repo the index hasn't crawled looks the
// same as a pre-fix client overwrite, so they need a human.

export interface N3Candidate {
  id: string;
  created_by: string;
  filename: string;
  hf_repo: string | null;
  hf_file: string | null;
  locked: boolean;
  clearIdentity: boolean;
}

export interface N3Review {
  id: string;
  hf_repo: string | null;
  hf_file: string | null;
}

interface Row {
  id: string;
  source: ModelSource;
  filename: string;
  size_bytes: number;
  hf_repo: string | null;
  hf_file: string | null;
  created_by: string | null;
  hf_identity_locked: number;
}

const HEX64 = /^[0-9a-f]{64}$/;

function liveIdentity(db: Database.Database, row: Pick<Row, "id" | "hf_repo" | "hf_file">): boolean {
  if (!row.hf_repo || !row.hf_file) return false;
  return !!db
    .prepare(`SELECT 1 FROM hf_gguf_index WHERE sha256 = ? AND repo_id = ? AND filename = ? AND deleted_at IS NULL`)
    .get(row.id, row.hf_repo, row.hf_file);
}

export function findN3Repairs(db: Database.Database): { candidates: N3Candidate[]; review: N3Review[] } {
  const rows = db
    .prepare(`SELECT id, source, filename, size_bytes, hf_repo, hf_file, created_by, hf_identity_locked FROM models`)
    .all() as Row[];
  const candidates: N3Candidate[] = [];
  const review: N3Review[] = [];
  for (const row of rows) {
    if (!HEX64.test(row.id)) continue;
    if (row.created_by !== null) {
      const derived = deriveModelId({
        source: row.source,
        filename: row.filename,
        size_bytes: row.size_bytes,
        hf_repo: row.hf_repo ?? undefined,
        hf_file: row.hf_file ?? undefined,
      });
      if (derived === row.id) continue;
      candidates.push({
        id: row.id,
        created_by: row.created_by,
        filename: row.filename,
        hf_repo: row.hf_repo,
        hf_file: row.hf_file,
        locked: row.hf_identity_locked === 1,
        clearIdentity: (row.hf_repo !== null || row.hf_file !== null) && !liveIdentity(db, row),
      });
    } else if (row.hf_identity_locked === 1 && !liveIdentity(db, row)) {
      review.push({ id: row.id, hf_repo: row.hf_repo, hf_file: row.hf_file });
    }
  }
  return { candidates, review };
}

export function applyN3Repairs(db: Database.Database, candidates: N3Candidate[]): void {
  const release = db.prepare(`UPDATE models SET created_by = NULL, hf_identity_locked = 0, metadata = '{}' WHERE id = ?`);
  const clear = db.prepare(`UPDATE models SET source = 'local', hf_repo = NULL, hf_file = NULL WHERE id = ?`);
  db.transaction(() => {
    for (const c of candidates) {
      release.run(c.id);
      if (c.clearIdentity) clear.run(c.id);
    }
  })();
}
