import { describe, expect, it, afterEach } from "vitest";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureDefaultQualityCorpus } from "./qualityCorpus.js";
import { DEFAULT_QUALITY_DATASET_HASH } from "../../shared/types.js";
import { createHash } from "node:crypto";

function sha256Of(path: string): string {
  return `sha256:${createHash("sha256").update(readFileSync(path)).digest("hex")}`;
}

const dirs: string[] = [];
function tmp(): string {
  const dir = mkdtempSync(join(tmpdir(), "llamatoaster-quality-corpus-test-"));
  dirs.push(dir);
  return dir;
}

afterEach(() => {
  while (dirs.length) {
    const dir = dirs.pop()!;
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      /* Windows temp-dir handle */
    }
  }
});

const silentLog = { info: () => {}, warn: () => {} };

describe("ensureDefaultQualityCorpus (N4's bundled default corpus)", () => {
  it("copies the bundled corpus into corpora/ when nothing is there yet, matching the pinned hash", () => {
    const root = tmp();
    const corporaDir = join(root, "corpora");

    // Use the REAL bundled asset so the hash actually matches the shared
    // constant -- a synthetic fixture here would never match
    // DEFAULT_QUALITY_DATASET_HASH, which is the exact thing this test needs
    // to prove.
    const realBundledPath = join(__dirname, "..", "assets", "default-corpus.txt");
    ensureDefaultQualityCorpus(corporaDir, realBundledPath, silentLog);

    const destPath = join(corporaDir, "default-corpus.txt");
    expect(existsSync(destPath)).toBe(true);
    expect(sha256Of(destPath)).toBe(DEFAULT_QUALITY_DATASET_HASH);
  });

  it("is idempotent -- leaves an already-correct file alone rather than re-writing it", () => {
    const root = tmp();
    const realBundledPath = join(__dirname, "..", "assets", "default-corpus.txt");
    const corporaDir = join(root, "corpora");
    mkdirSync(corporaDir, { recursive: true });
    const destPath = join(corporaDir, "default-corpus.txt");
    writeFileSync(destPath, readFileSync(realBundledPath));
    const before = readFileSync(destPath);

    ensureDefaultQualityCorpus(corporaDir, realBundledPath, silentLog);

    expect(readFileSync(destPath).equals(before)).toBe(true);
  });

  it("overwrites a stale on-disk copy that no longer matches the pinned hash", () => {
    const root = tmp();
    const realBundledPath = join(__dirname, "..", "assets", "default-corpus.txt");
    const corporaDir = join(root, "corpora");
    mkdirSync(corporaDir, { recursive: true });
    const destPath = join(corporaDir, "default-corpus.txt");
    writeFileSync(destPath, "an old, stale corpus that no longer matches the bundled asset");

    ensureDefaultQualityCorpus(corporaDir, realBundledPath, silentLog);

    expect(sha256Of(destPath)).toBe(DEFAULT_QUALITY_DATASET_HASH);
  });

  it("never throws when the bundled asset itself is missing -- warns instead", () => {
    const root = tmp();
    const corporaDir = join(root, "corpora");
    const missingBundledPath = join(root, "does-not-exist.txt");
    const warnings: string[] = [];
    expect(() =>
      ensureDefaultQualityCorpus(corporaDir, missingBundledPath, {
        info: () => {},
        warn: (...parts) => warnings.push(String(parts[0])),
      })
    ).not.toThrow();
    expect(warnings.length).toBeGreaterThan(0);
    expect(existsSync(join(corporaDir, "default-corpus.txt"))).toBe(false);
  });
});
