import { existsSync, readdirSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";

// Where worker-pushed run logs live (MULTIUSER_PLAN.md §1.10) and how long they
// stay. Every run owner's disk use goes through these files, and they outlive
// the database rows they belong to unless something removes them (audit
// 2026-09-27 M3).

const DEFAULT_RETENTION_DAYS = 90;

export function runLogDir(): string {
  return process.env.LOG_DIR ?? join(process.cwd(), "data", "run-logs");
}

export function runLogPath(runId: string): string {
  // Test ids are server-generated UUIDs (see createTest's uuid()), never
  // user-controlled path segments -- no traversal risk from this value, but
  // the param still comes off the URL, so it's validated as a run id (an
  // existing `runs` row) before ever reaching this function regardless.
  return join(runLogDir(), `run-${runId}.log.gz`);
}

export function deleteRunLog(runId: string): void {
  rmSync(runLogPath(runId), { force: true });
}

// RUN_LOG_RETENTION_DAYS=0 keeps them forever; unset or unparseable is the default.
function retentionDays(): number {
  const raw = process.env.RUN_LOG_RETENTION_DAYS;
  if (raw == null || raw.trim() === "") return DEFAULT_RETENTION_DAYS;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : DEFAULT_RETENTION_DAYS;
}

// Removes logs past the retention window, by age alone: judging a file by
// whether its run exists would ask whichever database this process opened,
// which on a dev machine needn't be the one that wrote it. Account deletion
// removes its own logs directly. Returns how many went.
export function pruneRunLogs(now = Date.now()): number {
  const dir = runLogDir();
  if (!existsSync(dir)) return 0;
  const days = retentionDays();
  if (days === 0) return 0;
  let removed = 0;
  for (const name of readdirSync(dir)) {
    if (!/^run-.+\.log\.gz$/.test(name)) continue;
    const path = join(dir, name);
    try {
      if (now - statSync(path).mtimeMs > days * 24 * 3600 * 1000) {
        rmSync(path, { force: true });
        removed++;
      }
    } catch {
      // Vanished mid-scan, or locked on Windows: the next sweep gets it.
    }
  }
  return removed;
}
