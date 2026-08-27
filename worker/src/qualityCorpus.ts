import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { DEFAULT_QUALITY_DATASET_HASH } from "../../shared/types.js";

export interface CorpusLogger {
  info: (...parts: unknown[]) => void;
  warn: (...parts: unknown[]) => void;
}

function sha256Of(path: string): string {
  return `sha256:${createHash("sha256").update(readFileSync(path)).digest("hex")}`;
}

// N4's "bundled default corpus with a pinned hash". Provisioned once at
// worker startup (idempotent: re-copies only when the on-disk file doesn't
// already match the expected hash, e.g. after an update ships a revised
// corpus) so a quality run never needs a manually placed dataset for the
// common case. The worker's own resolveQualityDatasetPath discovers it the
// same way as any other corpus file -- by hashing everything in corpora/,
// never by name -- so this only needs to get the bytes onto disk correctly.
export function ensureDefaultQualityCorpus(corporaDir: string, bundledPath: string, log: CorpusLogger): void {
  try {
    mkdirSync(corporaDir, { recursive: true });
    const destPath = join(corporaDir, "default-corpus.txt");
    if (existsSync(destPath) && sha256Of(destPath) === DEFAULT_QUALITY_DATASET_HASH) {
      return;
    }
    if (!existsSync(bundledPath)) {
      log.warn(
        `default quality corpus not found at ${bundledPath} -- quality runs will need a manually placed dataset`
      );
      return;
    }
    writeFileSync(destPath, readFileSync(bundledPath));
    log.info(`provisioned default quality corpus at ${destPath}`);
  } catch (err) {
    log.warn(
      `could not provision the default quality corpus (non-fatal -- quality runs will need a manually placed dataset): ${
        err instanceof Error ? err.message : String(err)
      }`
    );
  }
}
