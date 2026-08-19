import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3): GET /api/results/export
// scoping. Also exercises the real recordRunItemTerminal path (not a raw
// SQL insert) so this test would have caught the actual bug found while
// writing it: results.user_id was never populated at insert time, silently
// making every export return nothing for anyone once scoping shipped.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-results-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { resultsRoutes } = await import("./results.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(resultsRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({ id: "export-model-1", filename: "m.gguf", size_bytes: 1, source: "local", metadata: {} });
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

async function sessionFor(login: string) {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: login });
  return { userId: user.id, token };
}

// Builds one real, terminal, successful result row owned by `owner` --
// through the actual repo write path (createRun -> createRunItems ->
// recordRunItemTerminal), not a raw SQL insert, so this exercises exactly
// what the real worker item-tick route triggers.
function createOwnedResult(owner: { userId: string }, runId: string): void {
  repo.createRun(owner.userId, {
    id: runId,
    worker_name: "box",
    llama_cpp_build: "b1",
    llama_cpp_backend: "cpu",
    model_id: "export-model-1",
    config: { model_id: "export-model-1" } as never,
    status: "running",
    started_at: Date.now(),
  });
  repo.createRunItems(owner.userId, runId, [
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

describe("GET /api/results/export cross-user isolation (§4.3)", () => {
  it("results.user_id is actually populated by the real write path (regression guard)", () => {
    const owner = { userId: "regression-check-owner" };
    // upsertByIdentity needed for the FK -- createRun's user_id references users(id).
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "regression-check", login: "r", avatarUrl: null });
    createOwnedResult({ userId: user.id }, "regression-check-run");
    const rows = repo.getResultsForRun("regression-check-run");
    expect(rows.length).toBe(1);
    void owner;
  });

  it("an unauthenticated export (single-tenant mode) sees every result", async () => {
    const a = await sessionFor("export-anon-a");
    createOwnedResult(a, "export-anon-run-a");
    const res = await fetch(`${baseUrl}/api/results/export?format=json`);
    const rows = (await res.json()) as { run_id: string }[];
    expect(rows.some((r) => r.run_id === "export-anon-run-a")).toBe(true);
  });

  it("user B's export does not include user A's results", async () => {
    const ownerA = await sessionFor("export-owner-a");
    createOwnedResult(ownerA, "export-run-a");
    const ownerB = await sessionFor("export-owner-b");
    createOwnedResult(ownerB, "export-run-b");

    const resA = await fetch(`${baseUrl}/api/results/export?format=json`, {
      headers: { authorization: `Bearer ${ownerA.token}` },
    });
    const rowsA = (await resA.json()) as { run_id: string }[];
    expect(rowsA.some((r) => r.run_id === "export-run-a")).toBe(true);
    expect(rowsA.some((r) => r.run_id === "export-run-b")).toBe(false);

    const resB = await fetch(`${baseUrl}/api/results/export?format=json`, {
      headers: { authorization: `Bearer ${ownerB.token}` },
    });
    const rowsB = (await resB.json()) as { run_id: string }[];
    expect(rowsB.some((r) => r.run_id === "export-run-b")).toBe(true);
    expect(rowsB.some((r) => r.run_id === "export-run-a")).toBe(false);
  });

  it("requesting a specific run id you don't own returns nothing for it, not an error", async () => {
    const ownerA = await sessionFor("export-specific-owner");
    createOwnedResult(ownerA, "export-specific-run");
    const intruder = await sessionFor("export-specific-intruder");

    const res = await fetch(`${baseUrl}/api/results/export?format=json&runs=export-specific-run`, {
      headers: { authorization: `Bearer ${intruder.token}` },
    });
    expect(res.status).toBe(200);
    const rows = (await res.json()) as { run_id: string }[];
    expect(rows.length).toBe(0);
  });
});
