import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.6): buildContextSnapshot's own
// isolation coverage, exercised directly (no HTTP/SSE mocking needed --
// ai.test.ts already covers the streaming/budget machinery, which is
// unrelated to this). "The boundary is enforced by SQL, not by instruction"
// is the whole point of §4.6 -- this is what actually proves it: a
// fabricated/ignored system prompt can't leak another user's hardware or
// results if the DATA itself was never fetched for them in the first place.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-ai-context-isolation-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof import("../db/repo.js")["repo"];
let buildContextSnapshot: (userId: string | undefined) => string;
let getDb: typeof import("../db/migrate.js")["getDb"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb } = await import("../db/migrate.js"));
  ({ buildContextSnapshot } = await import("./ai.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

function claimWorker(machineId: string, userId: string, hostname: string): void {
  const worker = repo.workerRepo.getOrCreateByMachineId(machineId, hostname);
  getDb().prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(userId, worker.id);
}

describe("buildContextSnapshot cross-user isolation (§4.6)", () => {
  it("a user's context includes only their own machine, never another user's", () => {
    const ownerA = repo.userRepo.upsertByIdentity("github", { providerUserId: "ai-ctx-owner-a", login: "a", avatarUrl: null });
    claimWorker("ai-ctx-machine-a", ownerA.id, "alices-gpu-tower");
    const ownerB = repo.userRepo.upsertByIdentity("github", { providerUserId: "ai-ctx-owner-b", login: "b", avatarUrl: null });
    claimWorker("ai-ctx-machine-b", ownerB.id, "bobs-secret-box");

    const snapshotA = buildContextSnapshot(ownerA.id);
    expect(snapshotA).toContain("alices-gpu-tower");
    expect(snapshotA).not.toContain("bobs-secret-box");

    const snapshotB = buildContextSnapshot(ownerB.id);
    expect(snapshotB).toContain("bobs-secret-box");
    expect(snapshotB).not.toContain("alices-gpu-tower");
  });

  it("single-tenant mode (userId undefined) includes every machine", () => {
    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "ai-ctx-anon-owner", login: "c", avatarUrl: null });
    claimWorker("ai-ctx-machine-anon", owner.id, "anyones-visible-box");

    const snapshot = buildContextSnapshot(undefined);
    expect(snapshot).toContain("anyones-visible-box");
  });

  it("a user's context includes only their own results, never another user's", () => {
    repo.registerModel({ id: "ai-ctx-model", filename: "ai-ctx-model.gguf", size_bytes: 1, source: "local", metadata: {} });

    const ownerA = repo.userRepo.upsertByIdentity("github", { providerUserId: "ai-ctx-results-a", login: "ra", avatarUrl: null });
    const ownerB = repo.userRepo.upsertByIdentity("github", { providerUserId: "ai-ctx-results-b", login: "rb", avatarUrl: null });

    function createResult(runId: string, owner: { id: string }, workerName: string) {
      repo.createRun(owner.id, {
        id: runId,
        worker_name: workerName,
        llama_cpp_build: "b1",
        llama_cpp_backend: "cpu",
        model_id: "ai-ctx-model",
        config: { model_id: "ai-ctx-model" } as never,
        status: "running",
        started_at: Date.now(),
      });
      repo.createRunItems(owner.id, runId, [
        {
          idx: 0,
          n_prompt: 512,
          n_gen: 0,
          threads: 8,
          n_gpu_layers: 0,
          batch_size: 2048,
          ubatch_size: 512,
          cache_type_k: "f16",
          cache_type_v: "f16",
          flash_attn: "on",
          mtp: "off",
          n_gpu_layers_draft: 0,
        } as never,
      ]);
      repo.recordRunItemTerminal(runId, 0, {
        status: "done",
        ram_peak_mib: 100,
        vram_peak_mib: null,
        ram_avg_mib: 100,
        vram_avg_mib: null,
        results: [
          {
            test_type: "pp",
            n_prompt: 512,
            n_gen: 0,
            n_threads: 8,
            n_gpu_layers: 0,
            batch_size: 2048,
            ubatch_size: 512,
            cache_type_k: "f16",
            cache_type_v: "f16",
            flash_attn: "on",
            mtp: "off",
            n_gpu_layers_draft: 0,
            avg_tps: 100,
            stddev_tps: 1,
            ram_peak_mib: 100,
            vram_peak_mib: null,
            ram_free_before_mib: null,
            vram_free_before_mib: null,
            system_memory_total_mb: null,
            gpu_memory_total_mb: null,
            gpu_memory_total_accuracy: "unavailable",
            gpu_memory_total_source: null,
            gpu_memory_free_start_accuracy: "unavailable",
            gpu_memory_free_start_source: null,
            gpu_memory_model_avg_accuracy: "unavailable",
            gpu_memory_model_avg_source: null,
            gpu_memory_model_peak_accuracy: "unavailable",
            gpu_memory_model_peak_source: null,
            gpu_layers_loaded: null,
            total_model_layers: null,
          },
        ],
      } as never);
    }

    createResult("ai-ctx-run-a", ownerA, "alices-worker-name");
    createResult("ai-ctx-run-b", ownerB, "bobs-worker-name");

    const snapshotA = buildContextSnapshot(ownerA.id);
    expect(snapshotA).toContain("alices-worker-name");
    expect(snapshotA).not.toContain("bobs-worker-name");
  });
});
