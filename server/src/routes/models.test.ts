import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): models catalog ownership --
// created_by set on first insert, only the creator may overwrite identity
// fields on conflict, DELETE requires ownership, and the old fan-out-to-
// every-worker file deletion is gone. resolveAuthUser reads the Authorization
// header directly (see auth-middleware.ts), so this test app doesn't need
// the authMiddleware hook registered at all -- every route here already
// works with or without a session, exactly like in production with
// AUTH_ENABLED off vs on.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-models-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { modelsRoutes } = await import("./models.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(modelsRoutes);
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
    /* file still open -- fine, it's in the OS temp dir */
  }
});

function authed(token: string): Record<string, string> {
  return { authorization: `Bearer ${token}` };
}

async function sessionFor(login: string): Promise<string> {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: login });
  return token;
}

async function registerModel(body: unknown, headers: Record<string, string> = {}) {
  return fetch(`${baseUrl}/api/models`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

describe("POST /api/models ownership", () => {
  it("registers a model unauthenticated (single-tenant mode) with no creator", async () => {
    const res = await registerModel({ id: "m-anon", filename: "anon.gguf", size_bytes: 1, source: "local" });
    expect(res.status).toBe(201);
    expect(repo.getModelCreatedBy("m-anon")).toBeNull();
  });

  it("stamps created_by on first insert when authenticated", async () => {
    const token = await sessionFor("model-creator-1");
    const res = await registerModel(
      { id: "m-owned", filename: "owned.gguf", size_bytes: 1, source: "local" },
      authed(token)
    );
    expect(res.status).toBe(201);
    const model = repo.getModel("m-owned")!;
    expect(model.filename).toBe("owned.gguf");
  });

  it("the SAME creator re-registering updates identity fields normally", async () => {
    const token = await sessionFor("model-creator-2");
    await registerModel({ id: "m-reregister", filename: "v1.gguf", size_bytes: 1, source: "local" }, authed(token));
    const res = await registerModel(
      { id: "m-reregister", filename: "v2.gguf", size_bytes: 2, source: "local" },
      authed(token)
    );
    expect(res.status).toBe(201);
    expect(repo.getModel("m-reregister")!.filename).toBe("v2.gguf");
  });

  it("a DIFFERENT authenticated caller re-registering silently does not overwrite identity fields", async () => {
    const creatorToken = await sessionFor("model-creator-3");
    await registerModel({ id: "m-guarded", filename: "original.gguf", size_bytes: 1, source: "local" }, authed(creatorToken));

    const intruderToken = await sessionFor("model-intruder-1");
    const res = await registerModel(
      { id: "m-guarded", filename: "overwritten.gguf", size_bytes: 999, source: "local" },
      authed(intruderToken)
    );
    // No error surfaced -- registerModel's upsert just no-ops the conflict
    // branch (see its own doc comment) -- but the filename must NOT change.
    expect(res.status).toBe(201);
    expect(repo.getModel("m-guarded")!.filename).toBe("original.gguf");
  });

  it("an unclaimed model (registered anonymously) stays freely updatable by anyone", async () => {
    await registerModel({ id: "m-unclaimed", filename: "first.gguf", size_bytes: 1, source: "local" });
    const token = await sessionFor("model-claimer-1");
    const res = await registerModel(
      { id: "m-unclaimed", filename: "updated.gguf", size_bytes: 2, source: "local" },
      authed(token)
    );
    expect(res.status).toBe(201);
    expect(repo.getModel("m-unclaimed")!.filename).toBe("updated.gguf");
    // Re-registering never "adopts" an unclaimed model -- created_by is only
    // ever set on the very first insert, not on a later conflict update.
    expect(repo.getModelCreatedBy("m-unclaimed")).toBeNull();
  });
});

describe("DELETE /api/models/:id ownership", () => {
  it("the creator can delete their own model", async () => {
    const token = await sessionFor("model-deleter-1");
    await registerModel({ id: "m-del-owned", filename: "x.gguf", size_bytes: 1, source: "local" }, authed(token));
    const res = await fetch(`${baseUrl}/api/models/m-del-owned`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    // No more file_deletion_queued_on -- deleting the catalog row no longer
    // fans out to every worker (§4.4).
    expect(body).toEqual({ ok: true });
    expect(repo.getModel("m-del-owned")).toBeUndefined();
  });

  it("a non-creator cannot delete someone else's model", async () => {
    const ownerToken = await sessionFor("model-deleter-owner");
    await registerModel({ id: "m-del-protected", filename: "x.gguf", size_bytes: 1, source: "local" }, authed(ownerToken));

    const intruderToken = await sessionFor("model-deleter-intruder");
    const res = await fetch(`${baseUrl}/api/models/m-del-protected`, {
      method: "DELETE",
      headers: authed(intruderToken),
    });
    expect(res.status).toBe(403);
    expect(repo.getModel("m-del-protected")).toBeDefined();
  });

  it("an unclaimed model can be deleted by any authenticated caller", async () => {
    await registerModel({ id: "m-del-unclaimed", filename: "x.gguf", size_bytes: 1, source: "local" });
    const token = await sessionFor("model-deleter-anyone");
    const res = await fetch(`${baseUrl}/api/models/m-del-unclaimed`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(200);
  });

  it("an unauthenticated caller (single-tenant mode) can delete any model", async () => {
    const token = await sessionFor("model-deleter-single-tenant-owner");
    await registerModel({ id: "m-del-single-tenant", filename: "x.gguf", size_bytes: 1, source: "local" }, authed(token));
    const res = await fetch(`${baseUrl}/api/models/m-del-single-tenant`, { method: "DELETE" });
    expect(res.status).toBe(200);
  });
});

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.5): both routes below read a
// worker's cached model_files_json -- must only ever see the CALLER's own
// machines, never another user's, even though the model catalog itself is
// global.
describe("worker-derived model routes cross-user isolation (§4.5)", () => {
  async function sessionWithId(login: string) {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: login });
    return { userId: user.id, token };
  }

  // Creates a worker whose last heartbeat reported the given filename with
  // n_layer/mtp_layers metadata, claimed by `owner`.
  async function createWorkerWithModelFile(machineId: string, owner: { userId: string }, filename: string) {
    const worker = repo.workerRepo.getOrCreateByMachineId(machineId, machineId);
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET user_id = ?, model_files_json = ? WHERE id = ?`).run(
      owner.userId,
      JSON.stringify([{ path: filename, size_bytes: 123, n_layer: 42, mtp_layers: null }]),
      worker.id
    );
    return worker;
  }

  it("GET /api/models/locations only shows the caller's own machine, not another user's", async () => {
    const ownerA = await sessionWithId("locations-owner-a");
    const workerA = await createWorkerWithModelFile("locations-machine-a", ownerA, "shared-model.gguf");
    const ownerB = await sessionWithId("locations-owner-b");
    await createWorkerWithModelFile("locations-machine-b", ownerB, "shared-model.gguf");

    await registerModel({ id: "locations-model", filename: "shared-model.gguf", size_bytes: 1, source: "local" });

    const res = await fetch(`${baseUrl}/api/models/locations`, { headers: authed(ownerA.token) });
    const body = (await res.json()) as { locations: Record<string, string[]> };
    expect(body.locations["locations-model"]).toEqual([workerA.id]);
  });

  it("unauthenticated (single-tenant mode) sees every machine's copy", async () => {
    const ownerA = await sessionWithId("locations-anon-owner-a");
    const workerA = await createWorkerWithModelFile("locations-anon-machine-a", ownerA, "anon-shared.gguf");
    const ownerB = await sessionWithId("locations-anon-owner-b");
    const workerB = await createWorkerWithModelFile("locations-anon-machine-b", ownerB, "anon-shared.gguf");
    await registerModel({ id: "locations-anon-model", filename: "anon-shared.gguf", size_bytes: 1, source: "local" });

    const res = await fetch(`${baseUrl}/api/models/locations`);
    const body = (await res.json()) as { locations: Record<string, string[]> };
    expect(body.locations["locations-anon-model"]?.sort()).toEqual([workerA.id, workerB.id].sort());
  });

  it("locations also match hash-keyed models by the worker-reported sha256 when paths differ", async () => {
    // A manually-dropped copy whose local filename differs from the canonical
    // HF one must still be located via its digest -- model ids ARE sha256s for
    // every download/hash-identified registration.
    const ownerA = await sessionWithId("locations-sha-owner-a");
    const sha = "b".repeat(64);
    const workerA = await createWorkerWithModelFile("locations-sha-machine-a", ownerA, "renamed-copy.gguf");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(
      JSON.stringify([{ path: "renamed-copy.gguf", size_bytes: 123, sha256: sha, state: "verified" }]),
      workerA.id
    );

    await registerModel({
      id: sha,
      filename: "canonical-name.gguf",
      size_bytes: 123,
      source: "huggingface",
      hf_repo: "org/repo",
      hf_file: "canonical-name.gguf",
    });

    const res = await fetch(`${baseUrl}/api/models/locations`, { headers: authed(ownerA.token) });
    const body = (await res.json()) as { locations: Record<string, string[]> };
    expect(body.locations[sha]).toEqual([workerA.id]);
  });

  it("locations ignores files the worker reports as missing (deleted locally)", async () => {
    // The worker's cache keeps deleted files around (state "missing") for
    // re-download bookkeeping -- that must not keep locating the model on the
    // machine it was just deleted from.
    const ownerA = await sessionWithId("locations-missing-owner-a");
    const sha = "e".repeat(64);
    const workerA = await createWorkerWithModelFile("locations-missing-machine-a", ownerA, "gone.gguf");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(
      JSON.stringify([
        { path: "gone.gguf", size_bytes: 123, sha256: sha, state: "missing" },
        { path: "live.gguf", size_bytes: 5, sha256: "f".repeat(64), state: "verified" },
      ]),
      workerA.id
    );

    await registerModel({
      id: sha,
      filename: "gone.gguf",
      size_bytes: 123,
      source: "huggingface",
      hf_repo: "org/repo",
      hf_file: "gone.gguf",
    });

    const res = await fetch(`${baseUrl}/api/models/locations`, { headers: authed(ownerA.token) });
    const body = (await res.json()) as { locations: Record<string, string[]> };
    expect(body.locations[sha]).toBeUndefined();
  });

  it("backfill-layer-count only reads n_layer from the CALLER's own machine's cache", async () => {
    const ownerA = await sessionWithId("backfill-owner-a");
    await createWorkerWithModelFile("backfill-machine-a", ownerA, "backfill-target.gguf");
    const ownerB = await sessionWithId("backfill-owner-b");

    await registerModel({ id: "backfill-model", filename: "backfill-target.gguf", size_bytes: 1, source: "local" });

    // Owner A can see it (n_layer resolves from A's own machine).
    const resA = await fetch(`${baseUrl}/api/models/backfill-model/backfill-layer-count`, {
      method: "POST",
      headers: authed(ownerA.token),
    });
    const bodyA = (await resA.json()) as { n_layer: number | null };
    expect(bodyA.n_layer).toBe(42);

    // Re-register so metadata is reset (registerModel doesn't touch
    // metadata.n_layer, but start from a clean model id instead to avoid
    // any ordering dependency between these two assertions).
    await registerModel({ id: "backfill-model-b", filename: "backfill-target.gguf", size_bytes: 1, source: "local" });
    const resB = await fetch(`${baseUrl}/api/models/backfill-model-b/backfill-layer-count`, {
      method: "POST",
      headers: authed(ownerB.token),
    });
    const bodyB = (await resB.json()) as { n_layer: number | null };
    expect(bodyB.n_layer).toBeNull();
  });

  it("backfill-layer-count does not read n_layer from a missing (deleted) file's stale cache", async () => {
    const ownerA = await sessionWithId("backfill-missing-owner-a");
    const workerA = await createWorkerWithModelFile("backfill-missing-machine-a", ownerA, "backfill-gone.gguf");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(
      JSON.stringify([{ path: "backfill-gone.gguf", size_bytes: 123, n_layer: 42, mtp_layers: null, state: "missing" }]),
      workerA.id
    );

    await registerModel({ id: "backfill-missing-model", filename: "backfill-gone.gguf", size_bytes: 1, source: "local" });

    const res = await fetch(`${baseUrl}/api/models/backfill-missing-model/backfill-layer-count`, {
      method: "POST",
      headers: authed(ownerA.token),
    });
    const body = (await res.json()) as { n_layer: number | null };
    expect(body.n_layer).toBeNull();
  });
});
