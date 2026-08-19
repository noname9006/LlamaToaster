import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): workers.ts ownership --
// GET /api/workers scoped to the caller's own machines, and every per-worker
// action route (install/activate/delete build, download/delete a model file,
// available-builds) gated by assertOwnsWorker via requireOnlineWorker.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-workers-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "workers-test-secret";

vi.mock("../github-releases.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../github-releases.js")>();
  return {
    ...actual,
    getReleases: vi.fn(async () => [
      {
        tag: "b9999",
        published_at: "2026-01-01T00:00:00Z",
        assets: [
          { name: "llama-b9999-bin-ubuntu-x64.zip", download_url: "http://example.invalid/x.zip", size_bytes: 123 },
        ],
      },
    ]),
  };
});

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let getDb: typeof import("../db/migrate.js")["getDb"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb } = await import("../db/migrate.js"));
  const { workersRoutes } = await import("./workers.js");
  const { queueRoutes } = await import("./queue.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(workersRoutes);
  await app.register(queueRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless, OS temp dir */
  }
});

function authed(token: string): Record<string, string> {
  return { authorization: `Bearer ${token}` };
}

async function sessionFor(login: string): Promise<{ userId: string; token: string }> {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: login });
  return { userId: user.id, token };
}

async function heartbeatWorker(machineId: string): Promise<string> {
  const res = await fetch(`${baseUrl}/api/worker/heartbeat`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer workers-test-secret" },
    body: JSON.stringify({
      machine_id: machineId,
      hostname: machineId,
      backend: "cpu",
      hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
      installed_builds: [],
      model_files: [],
      status: "idle",
    }),
  });
  expect(res.status).toBe(200);
  return repo.workerRepo.getByMachineId(machineId)!.id;
}

// Simulates a Stage 3-approved worker without going through the real device
// flow -- direct SQL, same technique used elsewhere in this test suite for
// setting up ownership state cheaply.
function claimWorker(workerId: string, userId: string): void {
  getDb().prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(userId, workerId);
}

describe("GET /api/workers scoping", () => {
  it("unauthenticated (single-tenant mode) sees every machine", async () => {
    const idA = await heartbeatWorker("list-a");
    const idB = await heartbeatWorker("list-b");
    const { userId } = await sessionFor("list-owner");
    claimWorker(idB, userId);

    const res = await fetch(`${baseUrl}/api/workers`);
    const body = (await res.json()) as { workers: { id: string }[] };
    const ids = body.workers.map((w) => w.id);
    expect(ids).toContain(idA);
    expect(ids).toContain(idB);
  });

  it("authenticated sees only its own claimed machines", async () => {
    const idOwned = await heartbeatWorker("list-owned-only");
    const idOthers = await heartbeatWorker("list-others-only");
    const { userId, token } = await sessionFor("list-scoped-owner");
    claimWorker(idOwned, userId);
    const { userId: otherUserId } = await sessionFor("list-scoped-other");
    claimWorker(idOthers, otherUserId);

    const res = await fetch(`${baseUrl}/api/workers`, { headers: authed(token) });
    const body = (await res.json()) as { workers: { id: string }[] };
    const ids = body.workers.map((w) => w.id);
    expect(ids).toContain(idOwned);
    expect(ids).not.toContain(idOthers);
  });
});

describe("per-worker action routes ownership (assertOwnsWorker via requireOnlineWorker)", () => {
  it("the owner can act on their own machine", async () => {
    const id = await heartbeatWorker("action-owner-ok");
    const { userId, token } = await sessionFor("action-owner");
    claimWorker(id, userId);

    const res = await fetch(`${baseUrl}/api/workers/${id}/llama-cpp/install`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip" }),
    });
    expect(res.status).toBe(202);
  });

  it("a different authenticated user cannot act on someone else's claimed machine", async () => {
    const id = await heartbeatWorker("action-intruder");
    const { userId } = await sessionFor("action-real-owner");
    claimWorker(id, userId);
    const { token: intruderToken } = await sessionFor("action-intruder-user");

    const res = await fetch(`${baseUrl}/api/workers/${id}/llama-cpp/install`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(intruderToken) },
      body: JSON.stringify({ tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip" }),
    });
    expect(res.status).toBe(403);
  });

  it("an authenticated user cannot act on an UNCLAIMED machine either -- ownerId null is still a mismatch", async () => {
    const id = await heartbeatWorker("action-unclaimed");
    const { token } = await sessionFor("action-unclaimed-caller");

    const res = await fetch(`${baseUrl}/api/workers/${id}/llama-cpp/install`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip" }),
    });
    expect(res.status).toBe(403);
  });

  it("an unauthenticated caller (single-tenant mode) can act on any machine regardless of ownership", async () => {
    const id = await heartbeatWorker("action-single-tenant");
    const { userId } = await sessionFor("action-single-tenant-owner");
    claimWorker(id, userId);

    const res = await fetch(`${baseUrl}/api/workers/${id}/llama-cpp/install`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tag: "b9999", asset_name: "llama-b9999-bin-ubuntu-x64.zip" }),
    });
    expect(res.status).toBe(202);
  });

  it("GET .../available-builds is ownership-gated the same way", async () => {
    const id = await heartbeatWorker("action-available-builds");
    const { userId } = await sessionFor("action-builds-owner");
    claimWorker(id, userId);
    const { token: intruderToken } = await sessionFor("action-builds-intruder");

    const res = await fetch(`${baseUrl}/api/workers/${id}/available-builds`, { headers: authed(intruderToken) });
    expect(res.status).toBe(403);
  });

  it("DELETE .../models (delete a model file) is ownership-gated the same way", async () => {
    const id = await heartbeatWorker("action-delete-file");
    const { userId } = await sessionFor("action-delete-owner");
    claimWorker(id, userId);
    const { token: intruderToken } = await sessionFor("action-delete-intruder");

    const res = await fetch(`${baseUrl}/api/workers/${id}/models?file=some.gguf`, {
      method: "DELETE",
      headers: authed(intruderToken),
    });
    expect(res.status).toBe(403);
  });
});

describe("DELETE /api/workers/:id (permanently remove a machine)", () => {
  it("the owner can remove their own idle machine", async () => {
    const id = await heartbeatWorker("remove-owner-ok");
    const { userId, token } = await sessionFor("remove-owner");
    claimWorker(id, userId);

    const res = await fetch(`${baseUrl}/api/workers/${id}`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(200);
    expect(repo.workerRepo.getWorker(id)).toBeUndefined();
  });

  it("a different authenticated user cannot remove someone else's machine", async () => {
    const id = await heartbeatWorker("remove-intruder");
    const { userId } = await sessionFor("remove-real-owner");
    claimWorker(id, userId);
    const { token: intruderToken } = await sessionFor("remove-intruder-user");

    const res = await fetch(`${baseUrl}/api/workers/${id}`, { method: "DELETE", headers: authed(intruderToken) });
    expect(res.status).toBe(403);
    expect(repo.workerRepo.getWorker(id)).toBeDefined();
  });

  it("404s an unknown machine id", async () => {
    const res = await fetch(`${baseUrl}/api/workers/not-a-real-id`, { method: "DELETE" });
    expect(res.status).toBe(404);
  });

  it("409s a busy machine instead of silently orphaning its in-flight job", async () => {
    const id = await heartbeatWorker("remove-busy");
    getDb().prepare(`UPDATE workers SET active_job_id = 'fake-job-id' WHERE id = ?`).run(id);

    const res = await fetch(`${baseUrl}/api/workers/${id}`, { method: "DELETE" });
    expect(res.status).toBe(409);
    expect(repo.workerRepo.getWorker(id)).toBeDefined();
  });

  it("unauthenticated (single-tenant mode) can remove any machine regardless of claimed ownership", async () => {
    const id = await heartbeatWorker("remove-single-tenant");
    const { userId } = await sessionFor("remove-single-tenant-owner");
    claimWorker(id, userId);

    const res = await fetch(`${baseUrl}/api/workers/${id}`, { method: "DELETE" });
    expect(res.status).toBe(200);
    expect(repo.workerRepo.getWorker(id)).toBeUndefined();
  });

  it("removing a machine detaches (not deletes) its past runs -- worker_id goes null, the worker_name snapshot survives", async () => {
    const id = await heartbeatWorker("remove-with-history");
    getDb()
      .prepare(`INSERT INTO runs (id, worker_id, worker_name, status, started_at) VALUES (?, ?, ?, ?, ?)`)
      .run("run-remove-with-history", id, "remove-with-history", "done", Date.now());

    const res = await fetch(`${baseUrl}/api/workers/${id}`, { method: "DELETE" });
    expect(res.status).toBe(200);

    const run = getDb().prepare(`SELECT worker_id, worker_name FROM runs WHERE id = ?`).get("run-remove-with-history") as {
      worker_id: string | null;
      worker_name: string | null;
    };
    expect(run.worker_id).toBeNull();
    expect(run.worker_name).toBe("remove-with-history");
  });
});
