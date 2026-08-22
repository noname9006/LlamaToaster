import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { scanModelDirectory, hashFile, HashingQueue, lookupHashes, resolveHfMetadata, runStartupReconciliation } from "./model-scanner.js";
import { LocalModelCache } from "./local-cache.js";
import type { LocalModelState } from "../../../shared/types.js";

// Use vi.hoisted to define mocks at the top level
const mockFs = vi.hoisted(() => ({
  statSync: vi.fn(),
  existsSync: vi.fn(),
  readdirSync: vi.fn(),
  createReadStream: vi.fn(),
}));

const mockCreateHash = vi.hoisted(() => vi.fn(() => ({
  update: vi.fn().mockReturnThis(),
  digest: vi.fn().mockReturnValue("abc123"),
})));

const mockLocalCache = vi.hoisted(() => ({
  getAll: vi.fn().mockResolvedValue([]),
  upsert: vi.fn().mockResolvedValue(undefined),
  updateState: vi.fn().mockResolvedValue(undefined),
  updateHash: vi.fn().mockResolvedValue(undefined),
  updateHfMatch: vi.fn().mockResolvedValue(undefined),
}));

// Mock modules
vi.mock("node:fs", () => mockFs);
vi.mock("node:crypto", () => ({
  createHash: mockCreateHash,
}));
vi.mock("node:stream/promises", () => ({
  pipeline: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("node:stream", () => ({
  Readable: { fromWeb: vi.fn() },
  Transform: vi.fn(),
}));
vi.mock("./local-cache.js", () => ({
  LocalModelCache: vi.fn().mockImplementation(() => mockLocalCache),
}));
vi.mock("./log.js", () => ({
  log: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

describe("model-scanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFs.statSync.mockReset();
    mockFs.existsSync.mockReset();
    mockFs.readdirSync.mockReset();
    mockFs.createReadStream.mockReset();
    mockCreateHash.mockReset();
    mockLocalCache.getAll.mockReset();
    mockLocalCache.upsert.mockReset();
    mockLocalCache.updateState.mockReset();
    mockLocalCache.updateHash.mockReset();
    mockLocalCache.updateHfMatch.mockReset();
    // Default createHash returns an object with update/digest
    mockCreateHash.mockReturnValue({ update: vi.fn().mockReturnThis(), digest: vi.fn().mockReturnValue("abc123") } as any);
    // Default createReadStream returns a stream that immediately emits data+end
    mockFs.createReadStream.mockImplementation(() => ({
      on: vi.fn((event: string, cb: any) => {
        if (event === "data") cb(Buffer.from("test data"));
        if (event === "end") cb();
        return { on: vi.fn() } as any;
      }),
    } as any));
    // Default statSync returns a valid file stat (plain object, no isFile method - handled defensively)
    mockFs.statSync.mockImplementation(() => ({ size: 1000, mtimeMs: 1000, isFile: () => true, isDirectory: () => false } as any));
    // Ensure existsSync true by default for hashing re-stat
    mockFs.existsSync.mockReturnValue(true);
  });

  describe("scanModelDirectory", () => {
    it("should return empty results when model dir does not exist", async () => {
      mockFs.existsSync.mockReturnValue(false);
      mockLocalCache.getAll.mockResolvedValue([]);

      const result = await scanModelDirectory("/nonexistent", mockLocalCache as unknown as LocalModelCache);

      expect(result.newFiles).toHaveLength(0);
      expect(result.modifiedFiles).toHaveLength(0);
      expect(result.unchangedFiles).toHaveLength(0);
      expect(result.missingFiles).toHaveLength(0);
      expect(result.allFiles).toHaveLength(0);
    });

    it("should detect new files not in cache", async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([
        { name: "model1.gguf", isFile: () => true, isDirectory: () => false },
        { name: "model2.gguf", isFile: () => true, isDirectory: () => false },
      ]);
      mockFs.statSync
        .mockReturnValueOnce({ size: 1000, mtimeMs: 1000 })
        .mockReturnValueOnce({ size: 2000, mtimeMs: 2000 });

      mockLocalCache.getAll.mockResolvedValue([]);

      const result = await scanModelDirectory("/models", mockLocalCache as unknown as LocalModelCache);

      expect(result.newFiles).toHaveLength(2);
      expect(result.newFiles[0].path).toBe("model1.gguf");
      expect(result.newFiles[1].path).toBe("model2.gguf");
    });

    it("should detect unchanged files", async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([
        { name: "model.gguf", isFile: () => true, isDirectory: () => false },
      ]);
      mockFs.statSync.mockReturnValue({ size: 1000, mtimeMs: 1000 });

      const cachedEntry = { path: "model.gguf", size: 1000, mtime: 1000, sha256: "abc123" };
      mockLocalCache.getAll.mockResolvedValue([cachedEntry]);

      const result = await scanModelDirectory("/models", mockLocalCache as unknown as LocalModelCache);

      expect(result.unchangedFiles).toHaveLength(1);
      expect(result.unchangedFiles[0].path).toBe("model.gguf");
    });

    it("should detect modified files (size or mtime changed)", async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([
        { name: "model.gguf", isFile: () => true, isDirectory: () => false },
      ]);
      mockFs.statSync.mockReturnValue({ size: 1500, mtimeMs: 2000 });

      const cachedEntry = { path: "model.gguf", size: 1000, mtime: 1000, sha256: "abc123" };
      mockLocalCache.getAll.mockResolvedValue([cachedEntry]);

      const result = await scanModelDirectory("/models", mockLocalCache as unknown as LocalModelCache);

      expect(result.modifiedFiles).toHaveLength(1);
      expect(result.modifiedFiles[0].path).toBe("model.gguf");
    });

    it("should detect missing files (in cache but not on disk)", async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([]);

      const cachedEntry = { path: "missing.gguf", size: 1000, mtime: 1000 };
      mockLocalCache.getAll.mockResolvedValue([cachedEntry]);

      const result = await scanModelDirectory("/models", mockLocalCache as unknown as LocalModelCache);

      expect(result.missingFiles).toHaveLength(1);
      expect(result.missingFiles[0]).toBe("missing.gguf");
    });

    it("should skip .part files", async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([
        { name: "model.gguf.part", isFile: () => true, isDirectory: () => false },
        { name: "model.gguf", isFile: () => true, isDirectory: () => false },
      ]);
      mockFs.statSync.mockReturnValue({ size: 1000, mtimeMs: 1000 });

      mockLocalCache.getAll.mockResolvedValue([]);

      const result = await scanModelDirectory("/models", mockLocalCache as unknown as LocalModelCache);

      expect(result.allFiles).toHaveLength(1);
      expect(result.allFiles[0].path).toBe("model.gguf");
    });
  });

  describe("hashFile", () => {
    it("should hash a file and return sha256 and byteLength", async () => {
      const mockStream = {
        on: vi.fn((event, cb) => {
          if (event === "data") cb(Buffer.from("test data"));
          if (event === "end") cb();
        }),
      };
      mockFs.createReadStream.mockReturnValue(mockStream);

      const result = await hashFile("/test/model.gguf");

      expect(result).toEqual({
        sha256: "abc123",
        byteLength: 9,
      });
    });
  });

  describe("HashingQueue", () => {
    it("should process files sequentially", async () => {
      const mockCache = {
        upsert: vi.fn().mockResolvedValue(undefined),
        updateHash: vi.fn().mockResolvedValue(undefined),
        get: vi.fn().mockResolvedValue({ path: "model.gguf", sha256: "abc123" }),
        updateState: vi.fn().mockResolvedValue(undefined),
        updateHfMatch: vi.fn().mockResolvedValue(undefined),
        getAll: vi.fn().mockResolvedValue([]),
        delete: vi.fn().mockResolvedValue(undefined),
        deleteMissing: vi.fn().mockResolvedValue(0),
      } as unknown as LocalModelCache;

      const queue = new HashingQueue("/models", mockCache);

      const file1 = { path: "model1.gguf", size: 1000, mtime: 1000, cachedEntry: undefined };
      const file2 = { path: "model2.gguf", size: 2000, mtime: 2000, cachedEntry: undefined };

      const p1 = queue.enqueue(file1);
      const p2 = queue.enqueue(file2);

      await Promise.all([p1, p2]);
      await queue.waitForCompletion();

      // After fix, HashingQueue does single upsert per file (no second updateHash)
      expect(mockCache.upsert).toHaveBeenCalledTimes(2);
    });

    it("should handle hash errors", async () => {
      const mockCache = {
        upsert: vi.fn().mockRejectedValue(new Error("Hash failed")),
        updateHash: vi.fn().mockResolvedValue(undefined),
        get: vi.fn().mockResolvedValue(null),
        updateState: vi.fn().mockResolvedValue(undefined),
      } as unknown as LocalModelCache;

      const queue = new HashingQueue("/models", mockCache);
      const file = { path: "model.gguf", size: 1000, mtime: 1000, cachedEntry: undefined };

      await expect(queue.enqueue(file)).rejects.toThrow("Hash failed");
    });
  });

  describe("lookupHashes", () => {
    it("should return empty map for empty hashes array", async () => {
      const result = await lookupHashes("http://localhost", "token", []);
      expect(result.size).toBe(0);
    });

    it("should batch hashes in groups of 100", async () => {
      const fetch = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: vi.fn().mockResolvedValue({
            results: [{ sha256: "hash1", repo_id: "repo1", filename: "model1.gguf", revision: "main" }],
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: vi.fn().mockResolvedValue({
            results: [{ sha256: "hash2", repo_id: "repo2", filename: "model2.gguf", revision: "main" }],
          }),
        });

      global.fetch = fetch;

      const hashes = Array.from({ length: 150 }, (_, i) => `hash${i}`);
      const result = await lookupHashes("http://localhost", "token", hashes);

      expect(fetch).toHaveBeenCalledTimes(2);
      expect(result.size).toBe(2);
    });

    it("should handle API errors gracefully", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      });

      const result = await lookupHashes("http://localhost", "token", ["hash1"]);

      expect(result.size).toBe(0);
    });
  });

  describe("resolveHfMetadata", () => {
    it("should resolve HF metadata for entries with hashes but no match", async () => {
      const mockCache = {
        getAll: vi.fn().mockResolvedValue([
          { path: "model1.gguf", sha256: "hash1", hf_model_id: undefined, state: "detected" },
          // Matched AND recently re-checked -- should be skipped, not re-looked-up.
          {
            path: "model2.gguf",
            sha256: "hash2",
            hf_model_id: "existing/repo/model2.gguf",
            hf_checked_at: Date.now(),
            state: "verified",
          },
          { path: "model3.gguf", sha256: "hash3", hf_model_id: undefined, state: "detected" },
        ]),
        updateHfMatch: vi.fn().mockResolvedValue(undefined),
        updateState: vi.fn().mockResolvedValue(undefined),
      } as unknown as LocalModelCache;

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          results: [{ sha256: "hash1", repo_id: "repo1", filename: "model1.gguf", revision: "main", deleted_at: null }],
        }),
      });

      await resolveHfMetadata(mockCache, "http://localhost", "token");

      expect(mockCache.updateHfMatch).toHaveBeenCalledWith("model1.gguf", "repo1/model1.gguf", null);
      // model2 matched and freshly checked -> not touched; model3 has hash but no match -> unknown
      expect(mockCache.updateState).toHaveBeenCalledWith("model3.gguf", "unknown");
      expect(mockCache.updateHfMatch).not.toHaveBeenCalledWith("model2.gguf", expect.anything(), expect.anything());
      expect(mockCache.updateState).not.toHaveBeenCalledWith("model2.gguf", expect.anything());
    });

    it("should re-verify an already-matched entry once its check has gone stale", async () => {
      const staleCheckedAt = Date.now() - 25 * 60 * 60 * 1000; // >24h ago
      const mockCache = {
        getAll: vi.fn().mockResolvedValue([
          {
            path: "model1.gguf",
            sha256: "hash1",
            hf_model_id: "old/repo/model1.gguf",
            hf_checked_at: staleCheckedAt,
            state: "verified",
          },
        ]),
        updateHfMatch: vi.fn().mockResolvedValue(undefined),
        updateState: vi.fn().mockResolvedValue(undefined),
      } as unknown as LocalModelCache;

      const deletedAt = 1700000000000;
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          results: [{ sha256: "hash1", repo_id: "old", filename: "repo/model1.gguf", revision: "main", deleted_at: deletedAt }],
        }),
      });

      await resolveHfMetadata(mockCache, "http://localhost", "token");

      // Re-checked despite already having a match, because hf_checked_at is stale --
      // and the server's deleted_at is carried through to updateHfMatch.
      expect(mockCache.updateHfMatch).toHaveBeenCalledWith("model1.gguf", "old/repo/model1.gguf", deletedAt);
    });
  });

  describe("runStartupReconciliation", () => {
    it("should run full reconciliation flow", async () => {
      const mockCache = {
        getAll: vi.fn().mockResolvedValue([]),
        get: vi.fn().mockResolvedValue({ path: "model.gguf", sha256: "abc123", state: "hashing" }),
        upsert: vi.fn().mockResolvedValue(undefined),
        updateState: vi.fn().mockResolvedValue(undefined),
        delete: vi.fn().mockResolvedValue(undefined),
      } as unknown as LocalModelCache;

      mockFs.existsSync.mockReturnValue(true);
      mockFs.readdirSync.mockReturnValue([
        { name: "model.gguf", isFile: () => true, isDirectory: () => false },
      ]);
      mockFs.statSync.mockReturnValue({ size: 1000, mtimeMs: 1000 });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({ results: [] }),
      });

      const result = await runStartupReconciliation("/models", mockCache, "http://localhost", "token");

      expect(result).toHaveProperty("scanned");
      expect(result).toHaveProperty("hashed");
      expect(result).toHaveProperty("resolved");
      expect(result).toHaveProperty("missing");
    });
  });
});