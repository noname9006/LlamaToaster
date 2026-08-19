#!/usr/bin/env bash
# Pre-migration safety check for the Multi-User Transformation Plan (§0.6).
#
# Run this against the REAL VPS DB, by hand, before deploying any build that
# contains Stage 1+ schema changes (new workers/worker_jobs tables, the
# idx_results_item unique index, runs.worker_id). It is read-only except for
# the .backup call, which only ever writes a new file.
#
# Usage (on the VPS, as whichever user owns the DB file):
#   DB_PATH=/data/llamatoaster.db BACKUP_DIR=/backups ./scripts/pre-multiuser-backup.sh
#
# Exits non-zero and prints a clear reason if anything looks unsafe to
# proceed on -- do not deploy the migration until this passes clean.

set -euo pipefail

DB_PATH="${DB_PATH:-/data/llamatoaster.db}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
DATE_TAG="$(date +%F)"
BACKUP_PATH="${BACKUP_DIR}/pre-multiuser-${DATE_TAG}.db"

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 CLI not found -- install it before running this script" >&2
  exit 1
fi
if [ ! -f "$DB_PATH" ]; then
  echo "no DB file at $DB_PATH -- set DB_PATH to the real one" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

echo "==> 1/5  Backing up $DB_PATH -> $BACKUP_PATH"
sqlite3 "$DB_PATH" ".backup '${BACKUP_PATH}'"

echo "==> 2/5  Foreign-key check on the live DB (must be empty)"
FK_ISSUES="$(sqlite3 "$DB_PATH" "PRAGMA foreign_key_check;")"
if [ -n "$FK_ISSUES" ]; then
  echo "FOREIGN KEY VIOLATIONS FOUND -- fix these before migrating:" >&2
  echo "$FK_ISSUES" >&2
  exit 1
fi
echo "    ok -- no foreign key violations"

echo "==> 3/5  Integrity check on the backup file (must say 'ok')"
INTEGRITY="$(sqlite3 "$BACKUP_PATH" "PRAGMA integrity_check;")"
if [ "$INTEGRITY" != "ok" ]; then
  echo "BACKUP FILE FAILED INTEGRITY CHECK: $INTEGRITY" >&2
  exit 1
fi
echo "    ok -- backup integrity check passed"

echo "==> 4/5  Inspecting worker_name values actually in use (do not assume 'Local'/'Remote')"
sqlite3 -header -column "$DB_PATH" \
  "SELECT worker_name, COUNT(*) AS runs FROM runs GROUP BY worker_name;"

echo "==> 5/5  Checking for duplicate (run_id, idx, test_type) result rows"
echo "         (Stage 1's idx_results_item UNIQUE INDEX will fail to create if any exist)"
DUPES="$(sqlite3 "$DB_PATH" \
  "SELECT run_id, idx, test_type, COUNT(*) c FROM results GROUP BY 1,2,3 HAVING c > 1;")"
if [ -n "$DUPES" ]; then
  echo "DUPLICATE RESULT ROWS FOUND -- dedupe (keep the earliest created_at) before" >&2
  echo "creating idx_results_item, or the migration will crash the server on boot:" >&2
  echo "$DUPES" >&2
  exit 1
fi
echo "    ok -- no duplicate result rows"

echo
echo "All checks passed. Backup at: $BACKUP_PATH"
echo "Safe to deploy the Stage 1 schema migration."
