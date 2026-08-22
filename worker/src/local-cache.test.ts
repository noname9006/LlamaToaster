import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { LocalModelCache, createLocalModelCache } from "./local-cache.js";
import type { LocalModelState } from "../../../shared/types.js";

vi.mock("sqlite", () => ({
  open: vi.fn(),
}));

vi.mock("sqlite3", () => ({
  default: {
    Database: vi.fn(),
  },
}));

vi.mock("node:fs", () => ({
  existsSync: vi.fn().mockReturnValue(true),
  mkdirSync: vi.fn(),
}));

vi.mock("./log.js", () => ({
  log: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  },
}));

describe("LocalModelCache", () => {
  let mockDb: {
    exec: ReturnType<typeof vi.fn>;
    run: ReturnType<typeof vi.fn>;
    get: ReturnType<typeof vi.fn>;
    all: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
  };

  beforeEach(async () => {
    vi.clearAllMocks();
    
    mockDb = {
      exec: vi.fn().mockResolvedValue(undefined),
      run: vi.fn().mockResolvedValue({ changes: 1 }),
      get: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue([]),
      close: vi.fn().mockResolvedValue(undefined),
    };

    const { open } = await import("sqlite");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue(mockDb);
  });

  afterEach(async () => {
    // Clean up
  });

  it("should create tables on init", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    expect(mockDb.exec).toHaveBeenCalledWith(
      expect.stringContaining("CREATE TABLE IF NOT EXISTS local_model_cache")
    );
    expect(mockDb.exec).toHaveBeenCalledWith(
      expect.stringContaining("CREATE INDEX IF NOT EXISTS idx_local_model_cache_sha256")
    );
    
    await cache.close();
  });

  it("should upsert a cache entry", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    const entry = {
      path: "model.gguf",
      size: 1000,
      mtime: Date.now(),
      sha256: "abc123",
      hf_model_id: "user/repo/model.gguf",
      last_verified: Date.now(),
      state: "verified" as LocalModelState,
    };

    await cache.upsert(entry);

    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("INSERT INTO local_model_cache"),
      entry.path,
      entry.size,
      entry.mtime,
      entry.sha256,
      entry.hf_model_id,
      null, // hf_checked_at
      null, // hf_deleted_at
      entry.last_verified,
      entry.state
    );
    
    await cache.close();
  });

  it("should update state", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    await cache.updateState("model.gguf", "hashing");
    
    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("UPDATE local_model_cache SET state = ?"),
      "hashing",
      expect.any(Number),
      "model.gguf"
    );
    
    await cache.close();
  });

  it("should update hash", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    await cache.updateHash("model.gguf", "abc123");
    
    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("UPDATE local_model_cache SET sha256 = ?"),
      "abc123",
      "hashing",
      expect.any(Number),
      "model.gguf"
    );
    
    await cache.close();
  });

  it("should update HF match", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    await cache.updateHfMatch("model.gguf", "user/repo/model.gguf");
    
    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("UPDATE local_model_cache SET hf_model_id = ?"),
      "user/repo/model.gguf",
      expect.any(Number), // hf_checked_at
      null, // hf_deleted_at
      "verified",
      expect.any(Number), // last_verified
      "model.gguf"
    );

    await cache.close();
  });

  it("should update HF match with a deletedAt value", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    await cache.updateHfMatch("model.gguf", "user/repo/model.gguf", 1700000000000);

    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("UPDATE local_model_cache SET hf_model_id = ?"),
      "user/repo/model.gguf",
      expect.any(Number),
      1700000000000,
      "verified",
      expect.any(Number),
      "model.gguf"
    );

    await cache.close();
  });

  it("should delete an entry", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    await cache.delete("model.gguf");
    
    expect(mockDb.run).toHaveBeenCalledWith(
      "DELETE FROM local_model_cache WHERE path = ?",
      "model.gguf"
    );
    
    await cache.close();
  });

  it("should delete missing entries", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    mockDb.run.mockResolvedValueOnce({ changes: 2 });
    
    const deleted = await cache.deleteMissing(["model1.gguf", "model2.gguf"]);
    
    expect(deleted).toBe(2);
    expect(mockDb.run).toHaveBeenCalledWith(
      expect.stringContaining("DELETE FROM local_model_cache WHERE path NOT IN"),
      "model1.gguf",
      "model2.gguf"
    );
    
    await cache.close();
  });

  it("should delete all entries when no existing paths", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    mockDb.run.mockResolvedValueOnce({ changes: 5 });
    
    const deleted = await cache.deleteMissing([]);
    
    expect(deleted).toBe(5);
    expect(mockDb.run).toHaveBeenCalledWith("DELETE FROM local_model_cache");
    
    await cache.close();
  });

  it("should get an entry by path", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    const mockEntry = {
      path: "model.gguf",
      size: 1000,
      mtime: Date.now(),
      sha256: "abc123",
      hf_model_id: "user/repo/model.gguf",
      last_verified: Date.now(),
      state: "verified",
    };
    mockDb.get.mockResolvedValueOnce(mockEntry);
    
    const result = await cache.get("model.gguf");
    
    expect(result).toEqual(mockEntry);
    
    await cache.close();
  });

  it("should return null for non-existent entry", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    mockDb.get.mockResolvedValueOnce(null);
    
    const result = await cache.get("nonexistent.gguf");
    
    expect(result).toBeNull();
    
    await cache.close();
  });

  it("should get all entries", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    const mockEntries = [
      { path: "model1.gguf", size: 1000, mtime: 1, sha256: "abc", hf_model_id: null, last_verified: 1, state: "detected" },
      { path: "model2.gguf", size: 2000, mtime: 2, sha256: "def", hf_model_id: "user/repo/model2.gguf", last_verified: 2, state: "verified" },
    ];
    mockDb.all.mockResolvedValueOnce(mockEntries);
    
    const result = await cache.getAll();
    
    expect(result).toHaveLength(2);
    expect(result[0].path).toBe("model1.gguf");
    expect(result[1].path).toBe("model2.gguf");
    
    await cache.close();
  });

  it("should get entry by sha256", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    const mockEntry = {
      path: "model.gguf",
      size: 1000,
      mtime: Date.now(),
      sha256: "abc123",
      hf_model_id: "user/repo/model.gguf",
      last_verified: Date.now(),
      state: "verified",
    };
    mockDb.get.mockResolvedValueOnce(mockEntry);
    
    const result = await cache.getBySha256("abc123");
    
    expect(result).toEqual(mockEntry);
    
    await cache.close();
  });

  it("should close database", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();
    
    await cache.close();
    
    expect(mockDb.close).toHaveBeenCalled();
  });
});

describe("createLocalModelCache", () => {
  it("should create and initialize cache", async () => {
    const mockDb = {
      exec: vi.fn().mockResolvedValue(undefined),
      run: vi.fn().mockResolvedValue({ changes: 1 }),
      get: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue([]),
      close: vi.fn().mockResolvedValue(undefined),
    };

    const { open } = await import("sqlite");
    (open as ReturnType<typeof vi.fn>).mockResolvedValue(mockDb);
    
    const cache = await createLocalModelCache("/test/model/dir");
    
    expect(cache).toBeInstanceOf(LocalModelCache);
    await cache.close();
  });
});