import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-device-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let hashToken: (t: string) => string;

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ hashToken } = await import("../session.js"));
  const { deviceRoutes, deviceApprovalRoutes } = await import("./device.js");
  const { authMiddleware } = await import("../auth-middleware.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  app.addHook("preHandler", authMiddleware); // deviceApprovalRoutes needs req.user; deviceRoutes is exempt via PUBLIC_PATHS
  await app.register(deviceRoutes);
  await app.register(deviceApprovalRoutes);
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

async function postJson(path: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

function authed(token: string): Record<string, string> {
  return { authorization: `Bearer ${token}` };
}

async function startDevice(machineId: string, overrides: Partial<{ hostname: string; platform: string; arch: string }> = {}) {
  const res = await postJson("/api/device/start", {
    machine_id: machineId,
    hostname: overrides.hostname ?? "gpu-tower",
    platform: overrides.platform ?? "linux",
    arch: overrides.arch ?? "x64",
  });
  expect(res.status).toBe(200);
  return (await res.json()) as { device_code: string; user_code: string; verification_uri: string; interval: number; expires_in: number };
}

describe("POST /api/device/start", () => {
  it("returns a device_code + user_code + RFC 8628-shaped envelope for a brand-new machine", async () => {
    const body = await startDevice("start-machine-1");
    expect(body.device_code).toHaveLength(43); // base64url(32 bytes), see session.ts's generateEnrolmentCode
    expect(body.user_code).toMatch(/^[A-Z0-9]{4}-[A-Z0-9]{4}$/);
    expect(body.verification_uri).toBe("/device");
    expect(body.interval).toBe(5);
    expect(body.expires_in).toBe(900);
  });

  it("re-calling with the SAME machine_id reuses the row and revokes any prior worker session", async () => {
    const first = await startDevice("start-machine-reenroll");
    // Simulate the machine having completed enrolment once already.
    const worker = repo.workerRepo.getByEnrolmentCodeHash(hashToken(first.device_code))!;
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "reenroll-owner", login: "owner", avatarUrl: null });
    repo.workerRepo.approve(worker.id, user.id);
    const oldSession = repo.sessionRepo.create(user.id, { isWorker: true, workerId: worker.id });

    const second = await startDevice("start-machine-reenroll");
    expect(second.device_code).not.toBe(first.device_code);

    // Old worker session is gone.
    expect(repo.sessionRepo.getByTokenHash(hashToken(oldSession.token))).toBeUndefined();
    // Ownership is untouched -- re-enrolling an owned machine needs no re-approval.
    const reissued = repo.workerRepo.getByEnrolmentCodeHash(hashToken(second.device_code))!;
    expect(reissued.id).toBe(worker.id);
    expect(reissued.userId).toBe(user.id);
    expect(reissued.approvedAt).not.toBeNull();
  });

  it("400s on a malformed body (missing required fields)", async () => {
    const res = await postJson("/api/device/start", { machine_id: "only-this" });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/device/token", () => {
  it("returns authorization_pending before approval", async () => {
    const { device_code } = await startDevice("token-pending-machine");
    const res = await postJson("/api/device/token", { device_code });
    expect(res.status).toBe(400);
    expect((await res.json()) as { error: string }).toEqual({ error: "authorization_pending" });
  });

  it("returns expired_token for an unknown/garbage code", async () => {
    const res = await postJson("/api/device/token", { device_code: "not-a-real-code" });
    expect(res.status).toBe(400);
    expect((await res.json()) as { error: string }).toEqual({ error: "expired_token" });
  });

  it("issues a worker session once approved, and the code is single-use (a second poll expires)", async () => {
    const { device_code } = await startDevice("token-approve-machine");
    const worker = repo.workerRepo.getByEnrolmentCodeHash(hashToken(device_code))!;
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "token-approver", login: "approver", avatarUrl: null });
    repo.workerRepo.approve(worker.id, user.id);

    const firstPoll = await postJson("/api/device/token", { device_code });
    expect(firstPoll.status).toBe(200);
    const body = (await firstPoll.json()) as { session_token: string; refresh_token: string };
    expect(body.session_token).toBeDefined();
    expect(body.refresh_token).toBeDefined();

    // The issued session actually resolves to a real worker session.
    const session = repo.sessionRepo.getByTokenHash(hashToken(body.session_token));
    expect(session?.isWorker).toBe(true);
    expect(session?.workerId).toBe(worker.id);

    // Replaying the same device_code must NOT issue a second session.
    const secondPoll = await postJson("/api/device/token", { device_code });
    expect(secondPoll.status).toBe(400);
    expect((await secondPoll.json()) as { error: string }).toEqual({ error: "expired_token" });
  });
});

describe("GET /api/device/status", () => {
  it("requires authentication", async () => {
    const res = await fetch(`${baseUrl}/api/device/status?user_code=ABCD-EFGH`);
    expect(res.status).toBe(401);
  });

  it("reports not_found for an unknown code", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "status-viewer-1", login: "viewer", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "viewer" });
    const res = await fetch(`${baseUrl}/api/device/status?user_code=NOPE-NOPE`, { headers: authed(token) });
    expect((await res.json()) as { state: string }).toEqual({ state: "not_found" });
  });

  it("reports pending with machine details, then approved after approval", async () => {
    const { user_code } = await startDevice("status-flow-machine", { hostname: "status-box", platform: "win32", arch: "x64" });
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "status-viewer-2", login: "viewer2", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "viewer2" });

    const pendingRes = await fetch(`${baseUrl}/api/device/status?user_code=${user_code}`, { headers: authed(token) });
    const pendingBody = (await pendingRes.json()) as { state: string; machine?: { hostname: string; platform: string; arch: string } };
    expect(pendingBody.state).toBe("pending");
    expect(pendingBody.machine).toEqual({ hostname: "status-box", platform: "win32", arch: "x64", gpu: null });

    const worker = repo.workerRepo.getByUserCode(user_code)!;
    repo.workerRepo.approve(worker.id, user.id);

    const approvedRes = await fetch(`${baseUrl}/api/device/status?user_code=${user_code}`, { headers: authed(token) });
    expect((await approvedRes.json()) as { state: string }).toEqual({ state: "approved" });
  });
});

describe("POST /api/device/approve", () => {
  it("requires authentication", async () => {
    const res = await postJson("/api/device/approve", { user_code: "ABCD-EFGH" });
    expect(res.status).toBe(401);
  });

  it("404s an unknown code", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "approve-1", login: "approve-user", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "approver" });
    const res = await postJson("/api/device/approve", { user_code: "NOPE-NOPE" }, authed(token));
    expect(res.status).toBe(404);
  });

  it("approves a pending code, setting the CALLER as the owner -- this is the only place user_id is ever set", async () => {
    const { user_code } = await startDevice("approve-flow-machine");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "approve-2", login: "approve-user-2", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "approver2" });

    const res = await postJson("/api/device/approve", { user_code }, authed(token));
    expect(res.status).toBe(200);
    expect((await res.json()) as { ok: boolean; machine: { hostname: string } }).toMatchObject({ ok: true });

    const worker = repo.workerRepo.getByUserCode(user_code)!;
    expect(worker.userId).toBe(user.id);
    expect(worker.approvedAt).not.toBeNull();
  });

  it("409s a double-approve", async () => {
    const { user_code } = await startDevice("approve-twice-machine");
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "approve-3a", login: "a", avatarUrl: null });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "approve-3b", login: "b", avatarUrl: null });
    const sessionA = repo.sessionRepo.create(userA.id, { label: "a" });
    const sessionB = repo.sessionRepo.create(userB.id, { label: "b" });

    const firstRes = await postJson("/api/device/approve", { user_code }, authed(sessionA.token));
    expect(firstRes.status).toBe(200);

    // A second approver (even a different account) hitting the SAME code gets 409.
    const secondRes = await postJson("/api/device/approve", { user_code }, authed(sessionB.token));
    expect(secondRes.status).toBe(409);

    // Ownership must still belong to whoever approved FIRST.
    const worker = repo.workerRepo.getByUserCode(user_code)!;
    expect(worker.userId).toBe(userA.id);
  });
});

describe("full enrolment flow, start to finish", () => {
  it("start -> status(pending) -> approve -> status(approved) -> token(success) -> token(consumed)", async () => {
    const { device_code, user_code } = await startDevice("full-flow-machine", { hostname: "full-flow-box" });
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "full-flow-user", login: "full-flow", avatarUrl: null });
    const { token: browserToken } = repo.sessionRepo.create(user.id, { label: "browser" });

    const status1 = await fetch(`${baseUrl}/api/device/status?user_code=${user_code}`, { headers: authed(browserToken) });
    expect((await status1.json()) as { state: string }).toMatchObject({ state: "pending" });

    const approveRes = await postJson("/api/device/approve", { user_code }, authed(browserToken));
    expect(approveRes.status).toBe(200);

    const status2 = await fetch(`${baseUrl}/api/device/status?user_code=${user_code}`, { headers: authed(browserToken) });
    expect((await status2.json()) as { state: string }).toEqual({ state: "approved" });

    const tokenRes = await postJson("/api/device/token", { device_code });
    expect(tokenRes.status).toBe(200);
    const { session_token } = (await tokenRes.json()) as { session_token: string };
    expect(repo.sessionRepo.getByTokenHash(hashToken(session_token))?.workerId).toBeDefined();

    const secondTokenRes = await postJson("/api/device/token", { device_code });
    expect(secondTokenRes.status).toBe(400);
  });
});
