// Security finding C1: re-enrolling an OWNED machine. Knowing its machine_id
// must never be enough -- the caller either proves it holds the machine's
// Ed25519 key (a signed device challenge) or the owner approves it.
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { sign, createPrivateKey } from "node:crypto";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";
import { generateMachineKey, signReenrol, type MachineKeyPair } from "../../../worker/src/machine-key.js";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-device-reenrol-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let hashToken: (t: string) => string;
let getDb: typeof import("../db/migrate.js")["getDb"];
let MAX_PENDING_REENROLMENTS: number;

beforeAll(async () => {
  ({ repo, MAX_PENDING_REENROLMENTS } = await import("../db/repo.js"));
  ({ hashToken } = await import("../session.js"));
  ({ getDb } = await import("../db/migrate.js"));
  const { deviceRoutes, deviceApprovalRoutes } = await import("./device.js");
  const { sessionRoutes } = await import("./sessions.js");
  const { authMiddleware } = await import("../auth-middleware.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  app.addHook("preHandler", authMiddleware);
  await app.register(deviceRoutes);
  await app.register(deviceApprovalRoutes);
  await app.register(sessionRoutes);
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

type StartResponse = { device_code: string; user_code: string; approved?: boolean };

async function start(machineId: string, extra: Record<string, unknown> = {}): Promise<StartResponse> {
  const res = await postJson("/api/device/start", {
    machine_id: machineId,
    hostname: "gpu-tower",
    platform: "linux",
    arch: "x64",
    ...extra,
  });
  expect(res.status).toBe(200);
  return (await res.json()) as StartResponse;
}

async function challenge(machineId: string): Promise<string> {
  const res = await postJson("/api/device/challenge", { machine_id: machineId });
  expect(res.status).toBe(200);
  return ((await res.json()) as { nonce: string }).nonce;
}

async function signedStart(machineId: string, key: MachineKeyPair): Promise<StartResponse> {
  const nonce = await challenge(machineId);
  return start(machineId, { public_key: key.publicKeyPem, nonce, signature: signReenrol(key.privateKeyPem, nonce, machineId) });
}

async function redeem(deviceCode: string): Promise<{ status: number; body: { session_token?: string; error?: string } }> {
  const res = await postJson("/api/device/token", { device_code: deviceCode });
  return { status: res.status, body: (await res.json()) as { session_token?: string; error?: string } };
}

let userSeq = 0;
function newUser(): { id: string; token: string } {
  userSeq += 1;
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: `reenrol-${userSeq}`, login: `u${userSeq}`, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: "browser" });
  return { id: user.id, token };
}

// A machine enrolled the normal way, with a key: start -> owner approves ->
// redeem. Returns its live worker session token.
async function enrolWithKey(machineId: string, owner: { token: string }, key: MachineKeyPair) {
  const first = await start(machineId, { public_key: key.publicKeyPem });
  expect((await postJson("/api/device/approve", { user_code: first.user_code }, authed(owner.token))).status).toBe(200);
  const redeemed = await redeem(first.device_code);
  expect(redeemed.status).toBe(200);
  const workerId = repo.workerRepo.getByMachineId(machineId)!.id;
  return { workerId, workerToken: redeemed.body.session_token! };
}

function workerRow(workerId: string): unknown {
  return getDb().prepare(`SELECT * FROM workers WHERE id = ?`).get(workerId);
}

function sessionAlive(token: string): boolean {
  return repo.sessionRepo.getByTokenHash(hashToken(token)) !== undefined;
}

describe("C1 regression: machine_id alone gets nothing", () => {
  it("an attacker who only knows machine_id never gets a session, and the live machine is untouched", async () => {
    const owner = newUser();
    const { workerId, workerToken } = await enrolWithKey("c1-victim", owner, generateMachineKey());
    const before = workerRow(workerId);

    const attempt = await start("c1-victim");
    expect(attempt.approved).toBeUndefined();
    const polled = await redeem(attempt.device_code);
    expect(polled.status).toBe(400);
    expect(polled.body.error).toBe("authorization_pending");

    // The real worker keeps working, and its row didn't change at all.
    expect(sessionAlive(workerToken)).toBe(true);
    expect(workerRow(workerId)).toEqual(before);
  });

  it("100 anonymous /start calls leave the workers row byte-for-byte unchanged and the queue capped", async () => {
    const owner = newUser();
    const { workerId, workerToken } = await enrolWithKey("c1-flood", owner, generateMachineKey());
    const before = workerRow(workerId);
    for (let i = 0; i < 100; i++) await start("c1-flood", { hostname: `evil-${i}` });
    expect(workerRow(workerId)).toEqual(before);
    expect(sessionAlive(workerToken)).toBe(true);
    const queued = getDb().prepare(`SELECT COUNT(*) AS n FROM worker_reenrolments WHERE worker_id = ?`).get(workerId) as { n: number };
    expect(queued.n).toBe(MAX_PENDING_REENROLMENTS);
  });

  it("a flood of attempts doesn't push out the owner's own newest attempt", async () => {
    const owner = newUser();
    const { workerId } = await enrolWithKey("c1-flood-owner", owner, generateMachineKey());
    for (let i = 0; i < 20; i++) await start("c1-flood-owner");
    const mine = await start("c1-flood-owner", { hostname: "real-one" });
    const res = await postJson("/api/device/approve", { user_code: mine.user_code }, authed(owner.token));
    expect(res.status).toBe(200);
    expect((await redeem(mine.device_code)).status).toBe(200);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.hostname).toBe("real-one");
  });
});

describe("owner-approved reconnect", () => {
  it("the owner approves; redemption ends the old sessions and adopts the new install's key", async () => {
    const owner = newUser();
    const oldKey = generateMachineKey();
    const { workerId, workerToken } = await enrolWithKey("owner-reconnect", owner, oldKey);

    const newKey = generateMachineKey();
    const attempt = await start("owner-reconnect", { public_key: newKey.publicKeyPem });

    const status = await fetch(`${baseUrl}/api/device/status?user_code=${attempt.user_code}`, { headers: authed(owner.token) });
    const statusBody = (await status.json()) as { state: string; reconnectOf?: { id: string } };
    expect(statusBody.state).toBe("pending");
    expect(statusBody.reconnectOf?.id).toBe(workerId);

    expect((await postJson("/api/device/approve", { user_code: attempt.user_code }, authed(owner.token))).status).toBe(200);
    // Approval alone doesn't cut the old install off -- redemption does.
    expect(sessionAlive(workerToken)).toBe(true);

    const redeemed = await redeem(attempt.device_code);
    expect(redeemed.status).toBe(200);
    const session = repo.sessionRepo.getByTokenHash(hashToken(redeemed.body.session_token!));
    expect(session?.workerId).toBe(workerId);
    expect(session?.userId).toBe(owner.id);
    expect(sessionAlive(workerToken)).toBe(false);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe(newKey.publicKeyPem);
    // Single use.
    expect((await redeem(attempt.device_code)).status).toBe(400);
  });

  it("another user can't approve it (403) and can't even see it (not_found)", async () => {
    const owner = newUser();
    const stranger = newUser();
    await enrolWithKey("owner-only", owner, generateMachineKey());
    const attempt = await start("owner-only");

    const status = await fetch(`${baseUrl}/api/device/status?user_code=${attempt.user_code}`, { headers: authed(stranger.token) });
    expect((await status.json()) as { state: string }).toEqual({ state: "not_found" });
    const res = await postJson("/api/device/approve", { user_code: attempt.user_code }, authed(stranger.token));
    expect(res.status).toBe(403);
    expect((await redeem(attempt.device_code)).body.error).toBe("authorization_pending");
  });

  it("a worker session can't approve anything -- it isn't a user session", async () => {
    const owner = newUser();
    const { workerToken } = await enrolWithKey("worker-cant-approve", owner, generateMachineKey());
    const attempt = await start("worker-cant-approve");
    const res = await postJson("/api/device/approve", { user_code: attempt.user_code }, authed(workerToken));
    expect(res.status).toBe(401);
  });
});

describe("signed reconnect (proof of the machine key)", () => {
  it("a valid signature is approved at once; old sessions end only on redemption", async () => {
    const owner = newUser();
    const key = generateMachineKey();
    const { workerId, workerToken } = await enrolWithKey("signed-ok", owner, key);

    const attempt = await signedStart("signed-ok", key);
    expect(attempt.approved).toBe(true);
    expect(sessionAlive(workerToken)).toBe(true);

    const redeemed = await redeem(attempt.device_code);
    expect(redeemed.status).toBe(200);
    expect(repo.sessionRepo.getByTokenHash(hashToken(redeemed.body.session_token!))?.workerId).toBe(workerId);
    expect(sessionAlive(workerToken)).toBe(false);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe(key.publicKeyPem);
  });

  it("a nonce can't be used twice", async () => {
    const key = generateMachineKey();
    await enrolWithKey("signed-replay", newUser(), key);
    const nonce = await challenge("signed-replay");
    const body = { public_key: key.publicKeyPem, nonce, signature: signReenrol(key.privateKeyPem, nonce, "signed-replay") };
    expect((await start("signed-replay", body)).approved).toBe(true);
    expect((await start("signed-replay", body)).approved).toBeUndefined();
  });

  it("a nonce issued for another machine_id doesn't count", async () => {
    const key = generateMachineKey();
    await enrolWithKey("signed-other", newUser(), key);
    const nonce = await challenge("some-other-machine");
    const attempt = await start("signed-other", { nonce, signature: signReenrol(key.privateKeyPem, nonce, "signed-other") });
    expect(attempt.approved).toBeUndefined();
  });

  it("an expired nonce doesn't count", async () => {
    const key = generateMachineKey();
    await enrolWithKey("signed-expired", newUser(), key);
    const nonce = await challenge("signed-expired");
    getDb().prepare(`UPDATE device_challenges SET expires_at = ? WHERE nonce_hash = ?`).run(Date.now() - 1, hashToken(nonce));
    const attempt = await start("signed-expired", { nonce, signature: signReenrol(key.privateKeyPem, nonce, "signed-expired") });
    expect(attempt.approved).toBeUndefined();
  });

  it("a signature from the wrong key doesn't count", async () => {
    await enrolWithKey("signed-wrong-key", newUser(), generateMachineKey());
    const attempt = await signedStart("signed-wrong-key", generateMachineKey());
    expect(attempt.approved).toBeUndefined();
  });

  it("a signature without the domain prefix doesn't count", async () => {
    const key = generateMachineKey();
    await enrolWithKey("signed-no-domain", newUser(), key);
    const nonce = await challenge("signed-no-domain");
    const bare = sign(null, Buffer.from(`${nonce}\nsigned-no-domain`), createPrivateKey(key.privateKeyPem)).toString("base64");
    expect((await start("signed-no-domain", { nonce, signature: bare })).approved).toBeUndefined();
  });

  it("asking for more nonces never invalidates an earlier one", async () => {
    const key = generateMachineKey();
    await enrolWithKey("signed-many-nonces", newUser(), key);
    const mine = await challenge("signed-many-nonces");
    for (let i = 0; i < 20; i++) await challenge("signed-many-nonces");
    const attempt = await start("signed-many-nonces", {
      nonce: mine,
      signature: signReenrol(key.privateKeyPem, mine, "signed-many-nonces"),
    });
    expect(attempt.approved).toBe(true);
  });

  it("the challenge answers the same for a machine the server has never seen", async () => {
    const res = await postJson("/api/device/challenge", { machine_id: "never-seen-anywhere" });
    expect(res.status).toBe(200);
    expect(((await res.json()) as { nonce: string }).nonce).toBeTruthy();
  });
});

describe("revocation needs the owner again", () => {
  it("revoking the worker's session in Settings forgets its key, so a signed reconnect needs approval", async () => {
    const owner = newUser();
    const key = generateMachineKey();
    const { workerId, workerToken } = await enrolWithKey("revoke-one", owner, key);
    const session = repo.sessionRepo.getByTokenHash(hashToken(workerToken))!;

    const res = await fetch(`${baseUrl}/api/sessions/${session.id}`, { method: "DELETE", headers: authed(owner.token) });
    expect(res.status).toBe(200);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBeNull();
    expect((await signedStart("revoke-one", key)).approved).toBeUndefined();
  });

  it("sign-out-everywhere forgets the keys of every machine it signs out", async () => {
    const owner = newUser();
    const key = generateMachineKey();
    const { workerId } = await enrolWithKey("revoke-all", owner, key);
    expect((await postJson("/api/sessions/revoke-all", {}, authed(owner.token))).status).toBe(200);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBeNull();
    expect((await signedStart("revoke-all", key)).approved).toBeUndefined();
  });

  it("a replayed refresh token forgets the machine key (its config.json may have been copied)", async () => {
    const owner = newUser();
    const key = generateMachineKey();
    const first = await start("revoke-replay", { public_key: key.publicKeyPem });
    await postJson("/api/device/approve", { user_code: first.user_code }, authed(owner.token));
    const firstRes = await postJson("/api/device/token", { device_code: first.device_code });
    const { refresh_token } = (await firstRes.json()) as { refresh_token: string };
    const workerId = repo.workerRepo.getByMachineId("revoke-replay")!.id;

    expect((await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(refresh_token) })).status).toBe(200);
    // The same (now previous) refresh token again = replay.
    expect((await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(refresh_token) })).status).toBe(401);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBeNull();
    expect((await signedStart("revoke-replay", key)).approved).toBeUndefined();
  });

  it("a worker token can't sign its owner out everywhere (C1 impact)", async () => {
    const owner = newUser();
    const { workerToken } = await enrolWithKey("revoke-by-worker", owner, generateMachineKey());
    expect((await postJson("/api/sessions/revoke-all", {}, authed(workerToken))).status).toBe(401);
    expect((await fetch(`${baseUrl}/api/sessions`, { headers: authed(workerToken) })).status).toBe(401);
    expect((await fetch(`${baseUrl}/api/auth/identities`, { headers: authed(workerToken) })).status).toBe(401);
    expect(sessionAlive(owner.token)).toBe(true);
  });
});

describe("POST /api/worker/register-key (machines enrolled before keys)", () => {
  async function legacyEnrol(machineId: string) {
    const owner = newUser();
    const first = await start(machineId); // pre-key worker: no public_key
    await postJson("/api/device/approve", { user_code: first.user_code }, authed(owner.token));
    const { body } = await redeem(first.device_code);
    return { owner, workerId: repo.workerRepo.getByMachineId(machineId)!.id, workerToken: body.session_token! };
  }

  it("registers once over the worker's own session; the same key again is a no-op; a different key is 409", async () => {
    const { workerId, workerToken } = await legacyEnrol("tofu-1");
    const key = generateMachineKey();
    const first = await postJson("/api/worker/register-key", { public_key: key.publicKeyPem }, authed(workerToken));
    expect(first.status).toBe(200);
    expect(((await first.json()) as { registered: boolean }).registered).toBe(true);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe(key.publicKeyPem);

    const again = await postJson("/api/worker/register-key", { public_key: key.publicKeyPem }, authed(workerToken));
    expect(((await again.json()) as { registered: boolean }).registered).toBe(false);

    const other = await postJson("/api/worker/register-key", { public_key: generateMachineKey().publicKeyPem }, authed(workerToken));
    expect(other.status).toBe(409);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe(key.publicKeyPem);

    // And now the machine can reconnect on its own.
    expect((await signedStart("tofu-1", key)).approved).toBe(true);
  });

  it("refuses a user session and a missing token", async () => {
    const { owner } = await legacyEnrol("tofu-2");
    const key = generateMachineKey();
    expect((await postJson("/api/worker/register-key", { public_key: key.publicKeyPem }, authed(owner.token))).status).toBe(401);
    expect((await postJson("/api/worker/register-key", { public_key: key.publicKeyPem })).status).toBe(401);
  });

  it("rejects something that isn't an Ed25519 public key", async () => {
    const { workerToken } = await legacyEnrol("tofu-3");
    const res = await postJson("/api/worker/register-key", { public_key: "not a key" }, authed(workerToken));
    expect(res.status).toBe(400);
  });
});

describe("merge carries the key", () => {
  it("merging a rebuilt install into the old machine replaces the old key with the new one", async () => {
    const owner = newUser();
    const oldKey = generateMachineKey();
    const { workerId, workerToken } = await enrolWithKey("merge-old", owner, oldKey);
    // Same hostname as the original, new machine_id: the "reinstalled from scratch" case.
    const newKey = generateMachineKey();
    const second = await start("merge-new", { public_key: newKey.publicKeyPem });
    const merged = await postJson(
      "/api/device/approve",
      { user_code: second.user_code, merge_into: workerId },
      authed(owner.token)
    );
    expect(merged.status).toBe(200);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe(newKey.publicKeyPem);
    expect(sessionAlive(workerToken)).toBe(false);
    expect((await redeem(second.device_code)).status).toBe(200);
    // The old install's key no longer reconnects anything.
    expect((await signedStart("merge-new", oldKey)).approved).toBeUndefined();
    expect((await signedStart("merge-new", newKey)).approved).toBe(true);
  });
});
