import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type * as HfIndex from "./hf-index.js";

// getDb() (migrate.ts) is a module-level singleton keyed on process.env.DB_PATH
// at first call -- set it before importing hf-index.js so this test file gets
// its own throwaway DB, isolated from any other test file or the real dev DB.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-hf-index-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let hfIndex: typeof HfIndex;

beforeAll(async () => {
  hfIndex = await import("./hf-index.js");
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open on Windows -- fine, it's in the OS temp dir */
  }
});

describe("pruneRepoEntries", () => {
  it("keeps every file live when the repo has more entries than one prune chunk", () => {
    // Chunking the keep-set across multiple independent statements previously
    // marked rows deleted that were legitimately kept by a *different*
    // chunk, once a repo had more than 900 GGUF files. This repo exceeds
    // that boundary.
    const repoId = "test/many-files-repo";
    const now = Date.now();
    const filenames = Array.from({ length: 1000 }, (_, i) => `model-${i}.gguf`);
    for (const f of filenames) {
      hfIndex.upsertHfGgufEntry({
        sha256: `sha-${f}`,
        repo_id: repoId,
        filename: f,
        revision: "main",
        file_size: 1,
        last_seen: now,
        deleted_at: null,
      });
    }

    const removed = hfIndex.pruneRepoEntries(repoId, filenames);
    expect(removed).toBe(0);

    const found = hfIndex.lookupHfGgufHashes(filenames.map((f) => `sha-${f}`));
    expect(found).toHaveLength(1000);
    expect(found.every((e) => e.deleted_at === null)).toBe(true);
  });

  it("soft-deletes only rows whose filename actually dropped out of the tree", () => {
    const repoId = "test/dropped-file-repo";
    const now = Date.now();
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-a",
      repo_id: repoId,
      filename: "a.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    });
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-b",
      repo_id: repoId,
      filename: "b.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    });

    const removed = hfIndex.pruneRepoEntries(repoId, ["a.gguf"]);
    expect(removed).toBe(1);

    const [a] = hfIndex.lookupHfGgufHashes(["sha-a"]);
    expect(a.deleted_at).toBeNull();

    // Soft-deleted, not hard-removed -- lookupHfGgufHashes still returns it,
    // with deleted_at set, so the caller can tell "matched, but since
    // removed" apart from "never matched".
    const [b] = hfIndex.lookupHfGgufHashes(["sha-b"]);
    expect(b).toBeDefined();
    expect(b.deleted_at).not.toBeNull();
  });
});

describe("markRepoDeleted", () => {
  it("soft-deletes every live row for a repo and is idempotent", () => {
    const repoId = "test/gone-repo";
    const now = Date.now();
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-gone",
      repo_id: repoId,
      filename: "gone.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    });

    expect(hfIndex.markRepoDeleted(repoId, now)).toBe(1);
    const [row] = hfIndex.lookupHfGgufHashes(["sha-gone"]);
    expect(row.deleted_at).toBe(now);

    // Already deleted -- a second call finds nothing new to mark.
    expect(hfIndex.markRepoDeleted(repoId, now + 1)).toBe(0);
  });
});

describe("upsertHfGgufEntry", () => {
  it("clears a prior soft-delete when the same file reappears", () => {
    const repoId = "test/revived-repo";
    const now = Date.now();
    const entry = {
      sha256: "sha-revived",
      repo_id: repoId,
      filename: "revived.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    };
    hfIndex.upsertHfGgufEntry(entry);
    hfIndex.markRepoDeleted(repoId, now);
    expect(hfIndex.lookupHfGgufHashes(["sha-revived"])[0].deleted_at).toBe(now);

    hfIndex.upsertHfGgufEntry({ ...entry, last_seen: now + 1000 });
    expect(hfIndex.lookupHfGgufHashes(["sha-revived"])[0].deleted_at).toBeNull();
  });
});

describe("findStaleRepos", () => {
  it("excludes repos already soft-deleted", () => {
    const staleTs = Date.now() - 1000;
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-stale-live",
      repo_id: "test/stale-live-repo",
      filename: "a.gguf",
      revision: "main",
      file_size: 1,
      last_seen: staleTs,
      deleted_at: null,
    });
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-stale-deleted",
      repo_id: "test/stale-deleted-repo",
      filename: "a.gguf",
      revision: "main",
      file_size: 1,
      last_seen: staleTs,
      deleted_at: null,
    });
    hfIndex.markRepoDeleted("test/stale-deleted-repo", Date.now());

    const stale = hfIndex.findStaleRepos(Date.now(), 100);
    expect(stale).toContain("test/stale-live-repo");
    expect(stale).not.toContain("test/stale-deleted-repo");
  });
});
