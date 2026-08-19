import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FastifyRequest } from "fastify";
import { UnauthorizedError, BadRequestError } from "./errors.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-worker-auth-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "test-shared-secret";

let authenticateWorker: typeof import("./worker-auth.js")["authenticateWorker"];
let repo: typeof import("./db/repo.js")["repo"];

beforeAll(async () => {
  ({ authenticateWorker } = await import("./worker-auth.js"));
  ({ repo } = await import("./db/repo.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless, OS temp dir */
  }
});

function fakeRequest(opts: { authHeader?: string; body?: unknown }): FastifyRequest {
  return {
    headers: opts.authHeader ? { authorization: opts.authHeader } : {},
    body: opts.body,
  } as unknown as FastifyRequest;
}

describe("authenticateWorker", () => {
  it("rejects a missing Authorization header", async () => {
    await expect(authenticateWorker(fakeRequest({ body: { machine_id: "m1" } }))).rejects.toBeInstanceOf(
      UnauthorizedError
    );
  });

  it("rejects a wrong bearer token", async () => {
    await expect(
      authenticateWorker(fakeRequest({ authHeader: "Bearer wrong-token", body: { machine_id: "m1" } }))
    ).rejects.toBeInstanceOf(UnauthorizedError);
  });

  it("rejects a correct token with no machine_id in the body", async () => {
    await expect(
      authenticateWorker(fakeRequest({ authHeader: "Bearer test-shared-secret", body: {} }))
    ).rejects.toBeInstanceOf(BadRequestError);
  });

  it("resolves (and creates) a worker row for a valid token + machine_id", async () => {
    const worker = await authenticateWorker(
      fakeRequest({ authHeader: "Bearer test-shared-secret", body: { machine_id: "m-auth-1", hostname: "box-1" } })
    );
    expect(worker.machineId).toBe("m-auth-1");
    expect(worker.displayName).toBe("box-1");
  });

  it("resolves the SAME worker row on a repeat call with the same machine_id", async () => {
    const first = await authenticateWorker(
      fakeRequest({ authHeader: "Bearer test-shared-secret", body: { machine_id: "m-auth-2", hostname: "box-2" } })
    );
    const second = await authenticateWorker(
      fakeRequest({ authHeader: "Bearer test-shared-secret", body: { machine_id: "m-auth-2", hostname: "box-2" } })
    );
    expect(second.id).toBe(first.id);
  });
});

describe("authenticateWorker, Stage 3 session path", () => {
  it("resolves a worker via a valid worker session token, with no machine_id/body needed at all", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("m-session-1", "session-box");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "worker-session-owner", login: "owner", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: worker.id, label: worker.displayName });

    const resolved = await authenticateWorker(fakeRequest({ authHeader: `Bearer ${token}`, body: undefined }));
    expect(resolved.id).toBe(worker.id);
  });

  it("session auth is tried BEFORE the shared-token fallback -- a session token isn't also checked against WORKER_SHARED_TOKEN", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("m-session-2", "session-box-2");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "worker-session-owner-2", login: "owner2", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: worker.id });

    // No machine_id in the body -- if this fell through to the Stage 1 path
    // it would 400, not succeed.
    const resolved = await authenticateWorker(fakeRequest({ authHeader: `Bearer ${token}`, body: {} }));
    expect(resolved.id).toBe(worker.id);
  });

  it("rejects a non-worker (browser) session token even though it's a valid session", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "browser-session-owner", login: "browser-owner", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "browser tab" }); // isWorker defaults to false

    await expect(
      authenticateWorker(fakeRequest({ authHeader: `Bearer ${token}`, body: { machine_id: "m1" } }))
    ).rejects.toBeInstanceOf(UnauthorizedError);
  });

  it("rejects an expired worker session (falls through to the shared-token path, which also rejects it)", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("m-session-3", "session-box-3");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "worker-session-owner-3", login: "owner3", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: worker.id });
    const session = repo.sessionRepo.getByTokenHash((await import("./session.js")).hashToken(token))!;
    (await import("./db/migrate.js")).getDb().prepare(`UPDATE sessions SET expires_at = ? WHERE id = ?`).run(Date.now() - 1000, session.id);

    await expect(
      authenticateWorker(fakeRequest({ authHeader: `Bearer ${token}`, body: { machine_id: "m1" } }))
    ).rejects.toBeInstanceOf(UnauthorizedError);
  });
});

describe("authenticateWorker with no WORKER_SHARED_TOKEN configured", () => {
  beforeEach(() => {
    delete process.env.WORKER_SHARED_TOKEN;
  });

  it("rejects every request, even with a token presented", async () => {
    await expect(
      authenticateWorker(fakeRequest({ authHeader: "Bearer anything", body: { machine_id: "m1" } }))
    ).rejects.toBeInstanceOf(UnauthorizedError);
    process.env.WORKER_SHARED_TOKEN = "test-shared-secret"; // restore for any later test file ordering
  });
});
