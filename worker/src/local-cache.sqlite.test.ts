import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import Database from "better-sqlite3";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LocalModelCache, type LocalCacheEntry } from "./local-cache.js";

// local-cache.test.ts mocks better-sqlite3, so none of its SQL is ever
// prepared -- a column/placeholder mismatch in upsert passed there while
// every real write failed. These run the same class against a real file.

vi.mock("./log.js", () => ({
  log: { info: vi.fn(), error: vi.fn(), warn: vi.fn(), debug: vi.fn() },
}));

let modelDir: string;
let cache: LocalModelCache | null;

beforeEach(() => {
  modelDir = mkdtempSync(join(tmpdir(), "llamatoaster-local-cache-test-"));
  cache = null;
});

afterEach(async () => {
  await cache?.close();
  rmSync(modelDir, { recursive: true, force: true });
});

const fullEntry: LocalCacheEntry = {
  path: "model.gguf",
  size: 1000,
  mtime: 1_700_000_000_000,
  sha256: "abc123",
  hf_model_id: "user/repo/model.gguf",
  hf_checked_at: 1_700_000_000_001,
  hf_deleted_at: 1_700_000_000_002,
  hf_replaced_by_sha256: "def456",
  n_layer: 32,
  mtp_layers: 1,
  quant: "Q4_K_M",
  param_count: 7_000_000_000,
  trained_ctx: 32768,
  n_head_kv: 8,
  head_dim_k: 128,
  head_dim_v: 128,
  n_embd: 4096,
  n_head: 32,
  sliding_window: 4096,
  sliding_window_pattern: 6,
  shared_kv_layers: 2,
  tensor_layer_bytes: { dense: [100, 100], moe: [0, 0], embed: 50, output: 60, other: 1 },
  kv_layout: { full_k: 1024, full_v: 1024, swa_k: 512, swa_v: 512, window: 4096, state_bytes: 0, vocab: 151936 },
  gguf_checked_at: 1_700_000_000_003,
  last_verified: 1_700_000_000_004,
  state: "verified",
};

describe("LocalModelCache against real SQLite", () => {
  it("round-trips an entry with every field, including kv_layout", async () => {
    cache = new LocalModelCache(modelDir);
    await cache.init();

    await cache.upsert(fullEntry);

    expect(await cache.get("model.gguf")).toEqual(fullEntry);
    expect(await cache.getBySha256("abc123")).toEqual(fullEntry);
  });

  it("round-trips a minimal entry without kv_layout", async () => {
    cache = new LocalModelCache(modelDir);
    await cache.init();

    const minimal: LocalCacheEntry = {
      path: "bare.gguf",
      size: 10,
      mtime: 1,
      last_verified: 2,
      state: "detected",
    };
    await cache.upsert(minimal);

    const got = await cache.get("bare.gguf");
    expect(got).toMatchObject(minimal);
    expect(got?.kv_layout).toBeUndefined();
    expect(got?.tensor_layer_bytes).toBeUndefined();
  });

  it("upsert on an existing path overwrites every column, clearing kv_layout when absent", async () => {
    cache = new LocalModelCache(modelDir);
    await cache.init();

    await cache.upsert(fullEntry);
    await cache.upsert({ ...fullEntry, kv_layout: undefined, quant: "Q8_0" });

    const got = await cache.get("model.gguf");
    expect(got?.quant).toBe("Q8_0");
    expect(got?.kv_layout).toBeUndefined();
    expect(await cache.getAll()).toHaveLength(1);
  });

  it("runs every other write statement against the real schema", async () => {
    cache = new LocalModelCache(modelDir);
    await cache.init();

    await cache.upsert({ ...fullEntry, path: "a.gguf", sha256: undefined, state: "detected" });
    await cache.upsert({ ...fullEntry, path: "b.gguf", sha256: "bsha" });
    await cache.upsert({ ...fullEntry, path: "c.gguf", sha256: "csha" });

    await cache.updateHash("a.gguf", "asha");
    expect(await cache.get("a.gguf")).toMatchObject({ sha256: "asha", state: "hashing" });

    await cache.updateState("a.gguf", "unknown");
    expect((await cache.get("a.gguf"))?.state).toBe("unknown");

    // A live re-check clears the deleted/replaced flags fullEntry carries.
    await cache.updateHfMatch("a.gguf", "u/r/a.gguf");
    expect(await cache.get("a.gguf")).toMatchObject({ hf_model_id: "u/r/a.gguf", state: "verified" });
    expect((await cache.get("a.gguf"))?.hf_deleted_at).toBeUndefined();
    expect((await cache.get("a.gguf"))?.hf_replaced_by_sha256).toBeUndefined();

    await cache.delete("c.gguf");
    expect(await cache.get("c.gguf")).toBeNull();

    expect(await cache.deleteMissing(["a.gguf"])).toBe(1);
    expect((await cache.getAll()).map((e) => e.path)).toEqual(["a.gguf"]);
    expect(await cache.deleteMissing([])).toBe(1);
    expect(await cache.getAll()).toEqual([]);
  });

  it("migrates a v3 cache (no kv_layout column) to v4 and can write to it", async () => {
    // Build the on-disk state a worker running the previous release left behind.
    const cacheDir = join(modelDir, ".cache");
    mkdirSync(cacheDir);
    const old = new Database(join(cacheDir, "local-model-cache.sqlite"));
    old.exec(`
      CREATE TABLE local_model_cache (
        path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
        sha256 TEXT, hf_model_id TEXT, hf_checked_at INTEGER, hf_deleted_at INTEGER,
        last_verified INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'detected',
        hf_replaced_by_sha256 TEXT, n_layer INTEGER, mtp_layers INTEGER, quant TEXT,
        param_count INTEGER, gguf_checked_at INTEGER, trained_ctx INTEGER,
        n_head_kv INTEGER, head_dim_k INTEGER, head_dim_v INTEGER, n_embd INTEGER,
        n_head INTEGER, sliding_window INTEGER, sliding_window_pattern INTEGER,
        shared_kv_layers INTEGER, tensor_layer_bytes TEXT
      );
      INSERT INTO local_model_cache (path, size, mtime, sha256, last_verified, state, quant, gguf_checked_at)
        VALUES ('old.gguf', 5, 6, 'oldsha', 7, 'verified', 'Q4_0', 123);
    `);
    old.pragma("user_version = 3");
    old.close();

    cache = new LocalModelCache(modelDir);
    await cache.init();

    const migrated = await cache.get("old.gguf");
    expect(migrated).toMatchObject({ path: "old.gguf", sha256: "oldsha", quant: "Q4_0", state: "verified" });
    // Header re-read gate cleared so the scanner refills kv_layout.
    expect(migrated?.gguf_checked_at).toBeUndefined();
    expect(migrated?.kv_layout).toBeUndefined();

    // What the scanner's backfill does next: write the re-read header back.
    await cache.upsert({ ...migrated!, kv_layout: fullEntry.kv_layout, gguf_checked_at: 999 });
    const rescanned = await cache.get("old.gguf");
    expect(rescanned?.kv_layout).toEqual(fullEntry.kv_layout);
    expect(rescanned?.gguf_checked_at).toBe(999);

    await cache.close();
    cache = null;
    const check = new Database(join(cacheDir, "local-model-cache.sqlite"));
    expect(check.pragma("user_version", { simple: true })).toBe(4);
    check.close();
  });
});
