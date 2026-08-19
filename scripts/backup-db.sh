#!/usr/bin/env bash
# Recurring daily backup for the live LlamaToaster DB (MULTIUSER_PLAN.md
# §6.2: "the WAL-aware sqlite3 .backup daily cron ... plus the pre-migration
# backup in §0.6"). This is the ongoing cron job; scripts/pre-multiuser-backup.sh
# is a separate one-shot pre-migration safety gate (FK/dupe checks specific
# to the Stage 1 schema change) -- run this one on a schedule, not that one.
#
# sqlite3's own `.backup` command is what makes this safe against a live,
# actively-written WAL-mode DB (server/src/db/migrate.ts sets
# `journal_mode = WAL`): unlike a plain file copy, it's a proper online
# backup API that takes a consistent snapshot without needing to stop the
# server or interfering with concurrent writers.
#
# Usage (wire into cron, e.g. `0 3 * * * DB_PATH=... BACKUP_DIR=... /path/to/backup-db.sh`):
#   DB_PATH=/home/ubuntu/LlamaToaster/data/llamatoaster.db \
#   BACKUP_DIR=/home/ubuntu/LlamaToaster/backups \
#   ./scripts/backup-db.sh
#
# RETENTION_DAYS (default 14) -- backups older than this are deleted after a
# successful run. A failed run never deletes anything, so old backups keep
# accumulating (and cron's own failure-mail-on-nonzero-exit fires) until
# whatever's wrong is fixed, rather than quietly losing history.

set -euo pipefail

DB_PATH="${DB_PATH:-/home/ubuntu/LlamaToaster/data/llamatoaster.db}"
BACKUP_DIR="${BACKUP_DIR:-/home/ubuntu/LlamaToaster/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
DATE_TAG="$(date +%F)"
BACKUP_PATH="${BACKUP_DIR}/llamatoaster-${DATE_TAG}.db"

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 CLI not found -- install it before running this script" >&2
  exit 1
fi
if [ ! -f "$DB_PATH" ]; then
  echo "no DB file at $DB_PATH -- set DB_PATH to the real one" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

echo "==> Backing up $DB_PATH -> $BACKUP_PATH"
sqlite3 "$DB_PATH" ".backup '${BACKUP_PATH}'"

echo "==> Integrity check on the backup file (must say 'ok')"
INTEGRITY="$(sqlite3 "$BACKUP_PATH" "PRAGMA integrity_check;")"
if [ "$INTEGRITY" != "ok" ]; then
  echo "BACKUP FILE FAILED INTEGRITY CHECK: $INTEGRITY -- leaving it in place for inspection, not deleting old backups this run." >&2
  exit 1
fi
echo "    ok"

echo "==> Pruning backups older than ${RETENTION_DAYS} days"
find "$BACKUP_DIR" -maxdepth 1 -name 'llamatoaster-*.db' -mtime "+${RETENTION_DAYS}" -print -delete

echo "Done. Backup at: $BACKUP_PATH"
