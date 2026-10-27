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
process.env.SUPERADMIN_IDENTITIES = "github:model-deleter-admin";

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

  it("never changes an existing row: same creator, another user and anonymous all get 409", async () => {
    const creatorToken = await sessionFor("model-creator-2");
    await registerModel({ id: "m-guarded", filename: "original.gguf", size_bytes: 1, source: "local" }, authed(creatorToken));
    const intruderToken = await sessionFor("model-intruder-1");
    for (const headers of [authed(creatorToken), authed(intruderToken), {}]) {
      const res = await registerModel({ id: "m-guarded", filename: "overwritten.gguf", size_bytes: 999, source: "local" }, headers);
      expect(res.status).toBe(409);
    }
    expect(repo.getModel("m-guarded")!.filename).toBe("original.gguf");
  });
});

// Audit 2026-10-11 N3: the route used to spread the body into the catalog
// upsert, so a client could assert the download lock and rewrite or squat
// shared rows.
describe("POST /api/models cannot write shared catalog rows (N3)", () => {
  const downloaded = (sha: string) =>
    repo.registerModel(
      { id: sha, filename: "real.gguf", size_bytes: 1000, source: "huggingface", hf_repo: "good/repo", hf_file: "real.gguf", metadata: { n_layer: 40 } },
      { verifiedBy: "download" }
    );

  it("can't overwrite a download-locked row, even claiming the lock and an owner", async () => {
    const sha = "a1".repeat(32);
    downloaded(sha);
    const token = await sessionFor("n3-overwriter");
    const res = await registerModel(
      { id: sha, source: "local", filename: "real.gguf", lock_hf_identity: true, created_by: "someone" },
      authed(token)
    );
    expect(res.status).toBe(409);
    expect(repo.getModel(sha)).toMatchObject({ hf_repo: "good/repo", hf_identity_locked: true, metadata: { n_layer: 40 } });
    expect(repo.getModelCreatedBy(sha)).toBeNull();
  });

  it("ignores lock_hf_identity and created_by in the body of a new row", async () => {
    const token = await sessionFor("n3-claimer");
    const res = await registerModel(
      { id: "m-n3-claims", source: "local", filename: "x.gguf", lock_hf_identity: true, created_by: "someone-else" },
      authed(token)
    );
    expect(res.status).toBe(201);
    expect(repo.getModel("m-n3-claims")!.hf_identity_locked).toBe(false);
    expect(repo.getModelCreatedBy("m-n3-claims")).not.toBe("someone-else");
    expect(repo.getModelCreatedBy("m-n3-claims")).not.toBeNull();
  });

  it("a sha256 id is local-only: no HF identity or metadata, and it's lower-cased", async () => {
    const token = await sessionFor("n3-sha-rules");
    const sha = "B2".repeat(32);
    for (const extra of [
      { source: "huggingface", hf_repo: "evil/fake-GGUF", hf_file: "x.gguf" },
      { source: "local", hf_repo: "evil/fake-GGUF" },
      { source: "local", metadata: { gguf_header_adopted: true } },
    ]) {
      const res = await registerModel({ id: sha, filename: "x.gguf", ...extra }, authed(token));
      expect(res.status, JSON.stringify(extra)).toBe(400);
    }
    const ok = await registerModel({ id: sha, filename: "x.gguf", source: "local", metadata: {} }, authed(token));
    expect(ok.status).toBe(201);
    expect(repo.getModel(sha.toLowerCase())).toBeDefined();
  });

  it("a squatted sha256 is taken over by the real download, which makes it shared again", async () => {
    const sha = "c3".repeat(32);
    const token = await sessionFor("n3-squatter");
    expect((await registerModel({ id: sha, filename: "squat.gguf", source: "local" }, authed(token))).status).toBe(201);
    expect(repo.getModelCreatedBy(sha)).not.toBeNull();

    downloaded(sha);
    expect(repo.getModel(sha)).toMatchObject({ filename: "real.gguf", hf_repo: "good/repo", hf_identity_locked: true });
    // Shared now: the squatter can no longer delete it out from under others.
    expect(repo.getModelCreatedBy(sha)).toBeNull();
    const del = await fetch(`${baseUrl}/api/models/${sha}`, { method: "DELETE", headers: authed(token) });
    expect(del.status).toBe(403);
  });

  it("an index-verified heartbeat write also takes a squatted row over, without locking it", async () => {
    const sha = "d4".repeat(32);
    const token = await sessionFor("n3-squatter-2");
    await registerModel({ id: sha, filename: "squat.gguf", source: "local" }, authed(token));
    repo.registerModel(
      { id: sha, filename: "real.gguf", size_bytes: 1, source: "huggingface", hf_repo: "good/repo", hf_file: "real.gguf" },
      { verifiedBy: "index" }
    );
    expect(repo.getModel(sha)).toMatchObject({ hf_repo: "good/repo", hf_identity_locked: false });
    expect(repo.getModelCreatedBy(sha)).toBeNull();
  });

  it("an unverified write (e.g. an import stand-in) still can't touch another user's row", async () => {
    const owner = await sessionFor("n3-owner");
    await registerModel({ id: "m-n3-owned", filename: "mine.gguf", source: "local" }, authed(owner));
    const other = repo.userRepo.upsertByIdentity("github", { providerUserId: "n3-other", login: "n3-other", avatarUrl: null });
    repo.registerModel({ id: "m-n3-owned", filename: "theirs.gguf", source: "local", created_by: other.id });
    expect(repo.getModel("m-n3-owned")!.filename).toBe("mine.gguf");
  });

  it("the Add-model form's own payloads still register", async () => {
    const token = await sessionFor("n3-form-user");
    const hf = await registerModel(
      { source: "huggingface", hf_repo: "user/repo", hf_file: "model-q4_k_m.gguf", size_bytes: 0, metadata: { quant: "q4_k_m" } },
      authed(token)
    );
    expect(hf.status).toBe(201);
    const local = await registerModel({ source: "local", id: "my-model", filename: "model.gguf", size_bytes: 123, metadata: {} }, authed(token));
    expect(local.status).toBe(201);
  });

  it("rejects malformed fields", async () => {
    for (const body of [
      { source: "s3" },
      { source: "local", size_bytes: -1 },
      { source: "local", filename: 42 },
      { source: "local", metadata: [] },
      { source: "huggingface", hf_repo: "only/repo" },
    ]) {
      expect((await registerModel(body)).status, JSON.stringify(body)).toBe(400);
    }
  });
});

describe("GET /api/models visibility", () => {
  it("shows a signed-in user only their own models; single-tenant sees everything", async () => {
    const tokenA = await sessionFor("vis-owner-a");
    const tokenB = await sessionFor("vis-owner-b");
    await registerModel({ id: "m-vis-a", filename: "a.gguf", size_bytes: 1, source: "local" }, authed(tokenA));
    // Unowned shared row that user B has on one of their machines.
    repo.registerModel({ id: "b".repeat(64), filename: "b.gguf", size_bytes: 1, source: "local" });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "vis-owner-b", login: "vis-owner-b", avatarUrl: null });
    const w = repo.workerRepo.getOrCreateByMachineId("vis-machine-b", "vis-machine-b");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET user_id = ?, model_files_json = ? WHERE id = ?`).run(
      userB.id, JSON.stringify([{ path: "b.gguf", size_bytes: 1, sha256: "b".repeat(64) }]), w.id);

    const ids = async (headers: Record<string, string>) =>
      ((await (await fetch(`${baseUrl}/api/models`, { headers })).json()) as { models: { id: string }[] }).models.map((m) => m.id);
    const a = await ids(authed(tokenA));
    expect(a).toContain("m-vis-a");
    expect(a).not.toContain("b".repeat(64));
    const b = await ids(authed(tokenB));
    expect(b).toContain("b".repeat(64));
    expect(b).not.toContain("m-vis-a");
    const all = await ids({});
    expect(all).toEqual(expect.arrayContaining(["m-vis-a", "b".repeat(64)]));
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

  it("an unclaimed (shared catalog) model can't be deleted by an ordinary authenticated caller", async () => {
    await registerModel({ id: "m-del-unclaimed", filename: "x.gguf", size_bytes: 1, source: "local" });
    const token = await sessionFor("model-deleter-anyone");
    const res = await fetch(`${baseUrl}/api/models/m-del-unclaimed`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(403);
    expect(repo.getModel("m-del-unclaimed")).toBeDefined();
  });

  it("a superadmin can delete an unclaimed model", async () => {
    await registerModel({ id: "m-del-unclaimed-admin", filename: "x.gguf", size_bytes: 1, source: "local" });
    const token = await sessionFor("model-deleter-admin");
    const res = await fetch(`${baseUrl}/api/models/m-del-unclaimed-admin`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(200);
  });

  // Audit 2026-10-18 A4: only 'running' used to block the delete. A scheduled
  // run's job already carries the model; once claimed, its result insert hit
  // the results.model_id foreign key and the run never finished.
  it("refuses while a run against the model is scheduled or running", async () => {
    const token = await sessionFor("model-deleter-a4");
    await registerModel({ id: "m-del-scheduled", filename: "x.gguf", size_bytes: 1, source: "local" }, authed(token));
    repo.createTest(undefined, {
      id: "run-del-scheduled", root_run_id: "run-del-scheduled", kind: null, comparison_id: null, worker_name: "w", worker_id: null,
      llama_cpp_build: "b1", llama_cpp_backend: "cpu", model_id: "m-del-scheduled", config: {} as never, status: "scheduled", started_at: Date.now(),
    } as never);
    const res = await fetch(`${baseUrl}/api/models/m-del-scheduled`, { method: "DELETE", headers: authed(token) });
    expect(res.status).toBe(409);
    expect(repo.getModel("m-del-scheduled")).toBeDefined();
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

    await registerModel({ id: "backfill-model", filename: "backfill-target.gguf", size_bytes: 1, source: "local" }, authed(ownerA.token));

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
    await registerModel({ id: "backfill-model-b", filename: "backfill-target.gguf", size_bytes: 1, source: "local" }, authed(ownerB.token));
    const resB = await fetch(`${baseUrl}/api/models/backfill-model-b/backfill-layer-count`, {
      method: "POST",
      headers: authed(ownerB.token),
    });
    const bodyB = (await resB.json()) as { n_layer: number | null };
    expect(bodyB.n_layer).toBeNull();
  });

  // Audit 2026-10-18 A3: this route used to copy the caller's machine's
  // reading of ANY same-named file into ANY catalog row -- one user rewrote a
  // shared row's n_layer 32 -> 999 for everyone.
  it("backfill-layer-count never rewrites a shared row, and 404s for a model the caller can't see", async () => {
    const ownerA = await sessionWithId("a3-owner-a");
    const ownerB = await sessionWithId("a3-owner-b");
    await createWorkerWithModelFile("a3-machine-b", ownerB, "a3-shared.gguf"); // B's file reads n_layer 42
    const { model } = (await (
      await registerModel({ source: "huggingface", hf_repo: "org/a3", hf_file: "a3-shared.gguf" }, authed(ownerA.token))
    ).json()) as { model: { id: string } };
    repo.updateModelMetadata(model.id, { n_layer: 32 });

    // B has no relation to the row: not even told it exists.
    const unrelated = await fetch(`${baseUrl}/api/models/${model.id}/backfill-layer-count`, { method: "POST", headers: authed(ownerB.token) });
    expect(unrelated.status).toBe(404);
    expect(repo.getModel(model.id)?.metadata.n_layer).toBe(32);

    // B adds the same model (joins the shared row): B gets B's machine's
    // reading back, the shared row stays as it was.
    await registerModel({ source: "huggingface", hf_repo: "org/a3", hf_file: "a3-shared.gguf" }, authed(ownerB.token));
    const joined = await fetch(`${baseUrl}/api/models/${model.id}/backfill-layer-count`, { method: "POST", headers: authed(ownerB.token) });
    expect(((await joined.json()) as { n_layer: number }).n_layer).toBe(42);
    expect(repo.getModel(model.id)?.metadata.n_layer).toBe(32);
  });

  it("backfill-layer-count ignores a stored reading with an implausible layer count", async () => {
    const owner = await sessionWithId("a3-implausible-owner");
    const worker = await createWorkerWithModelFile("a3-implausible-machine", owner, "a3-huge.gguf");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(
      JSON.stringify([{ path: "a3-huge.gguf", size_bytes: 1, n_layer: 1_000_000_000 }]),
      worker.id
    );
    await registerModel({ id: "a3-huge-model", filename: "a3-huge.gguf", size_bytes: 1, source: "local" });
    // Single-tenant mode (no session) is the one path that still writes.
    const res = await fetch(`${baseUrl}/api/models/a3-huge-model/backfill-layer-count`, { method: "POST" });
    expect(((await res.json()) as { n_layer: number | null }).n_layer).toBeNull();
    expect(repo.getModel("a3-huge-model")?.metadata.n_layer).toBeUndefined();
  });

  it("backfill-param-count 404s for a model the caller can't see", async () => {
    const ownerA = await sessionWithId("a3-param-owner-a");
    const ownerB = await sessionWithId("a3-param-owner-b");
    await registerModel({ id: "a3-param-model", filename: "p.gguf", size_bytes: 1, source: "local" }, authed(ownerA.token));
    const res = await fetch(`${baseUrl}/api/models/a3-param-model/backfill-param-count`, { method: "POST", headers: authed(ownerB.token) });
    expect(res.status).toBe(404);
  });

  it("backfill-layer-count does not read n_layer from a missing (deleted) file's stale cache", async () => {
    const ownerA = await sessionWithId("backfill-missing-owner-a");
    const workerA = await createWorkerWithModelFile("backfill-missing-machine-a", ownerA, "backfill-gone.gguf");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET model_files_json = ? WHERE id = ?`).run(
      JSON.stringify([{ path: "backfill-gone.gguf", size_bytes: 123, n_layer: 42, mtp_layers: null, state: "missing" }]),
      workerA.id
    );

    await registerModel({ id: "backfill-missing-model", filename: "backfill-gone.gguf", size_bytes: 1, source: "local" }, authed(ownerA.token));

    const res = await fetch(`${baseUrl}/api/models/backfill-missing-model/backfill-layer-count`, {
      method: "POST",
      headers: authed(ownerA.token),
    });
    const body = (await res.json()) as { n_layer: number | null };
    expect(body.n_layer).toBeNull();
  });
});

// Re-audit 2026-10-13 R1: the form sends no id, so the id comes from the
// repo/file and is the same for everyone. A second user adding the same file
// used to get a 409 and still couldn't see the model.
describe("POST /api/models without an id joins an existing row", () => {
  const hfBody = { source: "huggingface", hf_repo: "shared/repo", hf_file: "shared-Q4.gguf", filename: "shared-Q4.gguf" };
  const visible = async (token: string) =>
    ((await (await fetch(`${baseUrl}/api/models`, { headers: authed(token) })).json()) as { models: { id: string }[] }).models.map((m) => m.id);

  it("a second user gets 200 with the same row, unchanged, and now sees it", async () => {
    const a = await sessionFor("r1-first");
    const b = await sessionFor("r1-second");
    const first = await registerModel({ ...hfBody, metadata: { n_layer: 32 } }, authed(a));
    expect(first.status).toBe(201);
    const { model } = (await first.json()) as { model: { id: string } };
    expect(await visible(b)).not.toContain(model.id);

    const second = await registerModel({ ...hfBody, filename: "renamed.gguf", metadata: { n_layer: 99 } }, authed(b));
    expect(second.status).toBe(200);
    expect(((await second.json()) as { model: { id: string } }).model.id).toBe(model.id);
    expect(repo.getModel(model.id)).toMatchObject({ filename: "shared-Q4.gguf", metadata: { n_layer: 32 } });
    expect(repo.getModelCreatedBy(model.id)).not.toBeNull();
    expect(await visible(b)).toContain(model.id);
  });

  it("the creator adding it again gets 200, not 409", async () => {
    const a = await sessionFor("r1-repeat");
    const body = { ...hfBody, hf_file: "repeat.gguf" };
    expect((await registerModel(body, authed(a))).status).toBe(201);
    expect((await registerModel(body, authed(a))).status).toBe(200);
  });

  it("deleting a shared added model only takes it off that user's list", async () => {
    const a = await sessionFor("r1-del-a");
    const b = await sessionFor("r1-del-b");
    const body = { ...hfBody, hf_file: "del.gguf" };
    const { model } = (await (await registerModel(body, authed(a))).json()) as { model: { id: string } };
    await registerModel(body, authed(b));

    // The creator removes it: the row stays for B and goes to the shared catalog.
    expect((await fetch(`${baseUrl}/api/models/${model.id}`, { method: "DELETE", headers: authed(a) })).status).toBe(200);
    expect(repo.getModel(model.id)).toBeDefined();
    expect(repo.getModelCreatedBy(model.id)).toBeNull();
    expect(await visible(a)).not.toContain(model.id);
    expect(await visible(b)).toContain(model.id);

    // B removes it too: off B's list; the shared row stays (admin-only delete).
    expect((await fetch(`${baseUrl}/api/models/${model.id}`, { method: "DELETE", headers: authed(b) })).status).toBe(200);
    expect(await visible(b)).not.toContain(model.id);
    expect(repo.getModel(model.id)).toBeDefined();
  });

  it("a sole adder still deletes the row outright", async () => {
    const a = await sessionFor("r1-sole");
    const { model } = (await (await registerModel({ ...hfBody, hf_file: "sole.gguf" }, authed(a))).json()) as { model: { id: string } };
    expect((await fetch(`${baseUrl}/api/models/${model.id}`, { method: "DELETE", headers: authed(a) })).status).toBe(200);
    expect(repo.getModel(model.id)).toBeUndefined();
  });
});
