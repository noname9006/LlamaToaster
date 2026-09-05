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

describe("lookupHfGgufHashes", () => {
  it("prefers a live match over a soft-deleted one for the same hash", () => {
    // Reproduces the real "pleasen/model" incident (2026-09-05): the same
    // content byte-for-byte in two repos, one of which has since been
    // deleted from HF -- the live one must win regardless of row order.
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-collision-a",
      repo_id: "test/dead-mirror",
      filename: "model.gguf",
      revision: "main",
      file_size: 1,
      last_seen: Date.now(),
      deleted_at: null,
    });
    hfIndex.markRepoDeleted("test/dead-mirror", Date.now());
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-collision-a",
      repo_id: "test/real-repo",
      filename: "real.gguf",
      revision: "main",
      file_size: 1,
      last_seen: Date.now(),
      deleted_at: null,
    });

    const found = hfIndex.lookupHfGgufHashes(["sha-collision-a"]);
    expect(found).toHaveLength(1);
    expect(found[0].repo_id).toBe("test/real-repo");
    expect(found[0].deleted_at).toBeNull();
  });

  it("still returns a deleted match when every row for the hash is deleted", () => {
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-all-gone",
      repo_id: "test/all-gone-a",
      filename: "a.gguf",
      revision: "main",
      file_size: 1,
      last_seen: Date.now(),
      deleted_at: null,
    });
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-all-gone",
      repo_id: "test/all-gone-b",
      filename: "b.gguf",
      revision: "main",
      file_size: 1,
      last_seen: Date.now(),
      deleted_at: null,
    });
    hfIndex.markRepoDeleted("test/all-gone-a", Date.now());
    hfIndex.markRepoDeleted("test/all-gone-b", Date.now());

    const found = hfIndex.lookupHfGgufHashes(["sha-all-gone"]);
    expect(found).toHaveLength(1);
    expect(found[0].deleted_at).not.toBeNull();
  });

  it("among equally-live matches, prefers the most recently seen", () => {
    const older = Date.now() - 100_000;
    const newer = Date.now();
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-recency",
      repo_id: "test/stale-source",
      filename: "old.gguf",
      revision: "main",
      file_size: 1,
      last_seen: older,
      deleted_at: null,
    });
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-recency",
      repo_id: "test/fresh-source",
      filename: "new.gguf",
      revision: "main",
      file_size: 1,
      last_seen: newer,
      deleted_at: null,
    });

    const found = hfIndex.lookupHfGgufHashes(["sha-recency"]);
    expect(found).toHaveLength(1);
    expect(found[0].repo_id).toBe("test/fresh-source");
  });

  it("breaks an exact last_seen tie deterministically by repo_id", () => {
    const now = Date.now();
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-tie",
      repo_id: "test/z-repo",
      filename: "z.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    });
    hfIndex.upsertHfGgufEntry({
      sha256: "sha-tie",
      repo_id: "test/a-repo",
      filename: "a.gguf",
      revision: "main",
      file_size: 1,
      last_seen: now,
      deleted_at: null,
    });

    // Run both orderings of the underlying scan to prove the result doesn't
    // depend on SQLite's (unspecified) row order for this query.
    const found1 = hfIndex.lookupHfGgufHashes(["sha-tie"]);
    const found2 = hfIndex.lookupHfGgufHashes(["sha-tie"]);
    expect(found1[0].repo_id).toBe("test/a-repo");
    expect(found2[0].repo_id).toBe("test/a-repo");
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

describe("getStaleRefreshMode", () => {
  const ENV_KEY = "HF_INDEX_STALENESS_REFRESH_MODE";
  const original = process.env[ENV_KEY];

  afterAll(() => {
    if (original === undefined) delete process.env[ENV_KEY];
    else process.env[ENV_KEY] = original;
  });

  it("defaults to on-demand when unset", () => {
    delete process.env[ENV_KEY];
    expect(hfIndex.getStaleRefreshMode()).toBe("on-demand");
  });

  it("switches to background only on an exact (case-insensitive) match", () => {
    process.env[ENV_KEY] = "background";
    expect(hfIndex.getStaleRefreshMode()).toBe("background");

    process.env[ENV_KEY] = "BACKGROUND";
    expect(hfIndex.getStaleRefreshMode()).toBe("background");

    // Anything else (typo, unrelated value) falls back to the safe default
    // rather than silently enabling the expensive background sweep.
    process.env[ENV_KEY] = "backgroundish";
    expect(hfIndex.getStaleRefreshMode()).toBe("on-demand");
  });
});
