import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { LocalModelCache, createLocalModelCache } from "./local-cache.js";
import type { LocalModelState } from "../../../shared/types.js";

// better-sqlite3's constructor returns a Database whose statement objects
// (from .prepare()) carry the actual .get()/.all()/.run() methods -- mock
// that two-step shape, not a flat exec/get/all/run object like the old
// async `sqlite` wrapper had.
let mockStatement: {
  get: ReturnType<typeof vi.fn>;
  all: ReturnType<typeof vi.fn>;
  run: ReturnType<typeof vi.fn>;
};
let mockDb: {
  exec: ReturnType<typeof vi.fn>;
  pragma: ReturnType<typeof vi.fn>;
  prepare: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
};

vi.mock("better-sqlite3", () => ({
  // A plain function, not an arrow function -- `new Database(...)` invokes
  // the mock's implementation via `new`, and arrow functions can't be
  // constructors ("X is not a constructor").
  default: vi.fn(function () {
    return mockDb;
  }),
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
  beforeEach(() => {
    vi.clearAllMocks();

    mockStatement = {
      get: vi.fn().mockReturnValue(null),
      all: vi.fn().mockReturnValue([]),
      run: vi.fn().mockReturnValue({ changes: 1 }),
    };
    mockDb = {
      exec: vi.fn(),
      pragma: vi.fn(),
      prepare: vi.fn().mockReturnValue(mockStatement),
      close: vi.fn(),
    };
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

    expect(mockStatement.run).toHaveBeenCalledWith(
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

    expect(mockStatement.run).toHaveBeenCalledWith("hashing", expect.any(Number), "model.gguf");

    await cache.close();
  });

  it("should update hash", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    await cache.updateHash("model.gguf", "abc123");

    expect(mockStatement.run).toHaveBeenCalledWith("abc123", "hashing", expect.any(Number), "model.gguf");

    await cache.close();
  });

  it("should update HF match", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    await cache.updateHfMatch("model.gguf", "user/repo/model.gguf");

    expect(mockStatement.run).toHaveBeenCalledWith(
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

    expect(mockStatement.run).toHaveBeenCalledWith(
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

    expect(mockStatement.run).toHaveBeenCalledWith("model.gguf");

    await cache.close();
  });

  it("should delete missing entries", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    mockStatement.run.mockReturnValueOnce({ changes: 2 });

    const deleted = await cache.deleteMissing(["model1.gguf", "model2.gguf"]);

    expect(deleted).toBe(2);
    expect(mockStatement.run).toHaveBeenCalledWith("model1.gguf", "model2.gguf");

    await cache.close();
  });

  it("should delete all entries when no existing paths", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    mockStatement.run.mockReturnValueOnce({ changes: 5 });

    const deleted = await cache.deleteMissing([]);

    expect(deleted).toBe(5);
    expect(mockStatement.run).toHaveBeenCalledWith();

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
    mockStatement.get.mockReturnValueOnce(mockEntry);

    const result = await cache.get("model.gguf");

    expect(result).toEqual(mockEntry);

    await cache.close();
  });

  it("should return null for non-existent entry", async () => {
    const cache = new LocalModelCache("/test/model/dir");
    await cache.init();

    mockStatement.get.mockReturnValueOnce(null);

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
    mockStatement.all.mockReturnValueOnce(mockEntries);

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
    mockStatement.get.mockReturnValueOnce(mockEntry);

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
    const cache = await createLocalModelCache("/test/model/dir");

    expect(cache).toBeInstanceOf(LocalModelCache);
    await cache.close();
  });
});
