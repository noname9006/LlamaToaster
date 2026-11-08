import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";
import fastifyRateLimit from "@fastify/rate-limit";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-device-bruteforce-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let generateUserCode: () => string;

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ generateUserCode } = await import("../session.js"));
  const { deviceRoutes, deviceApprovalRoutes } = await import("./device.js");
  const { authMiddleware } = await import("../auth-middleware.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  // Same registration as index.ts: opt-in per route. Without this the
  // routes' `config.rateLimit` is silently inert (the other device tests
  // never register the plugin, so they cannot see the limit).
  await app.register(fastifyRateLimit, { global: false });
  app.addHook("preHandler", authMiddleware);
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
    /* fine, it's in the OS temp dir */
  }
});

const post = (path: string, body: unknown, token: string) =>
  fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });

const status = (code: string, token: string) =>
  fetch(`${baseUrl}/api/device/status?user_code=${code}`, { headers: { authorization: `Bearer ${token}` } });

function login(id: string): string {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: id, login: id, avatarUrl: null });
  return repo.sessionRepo.create(user.id, { label: id }).token;
}

describe("brute-forcing the enrolment user_code", () => {
  it("generated codes are well-formed, unbiased in alphabet, and don't repeat", () => {
    const seen = new Set<string>();
    const chars = new Set<string>();
    for (let i = 0; i < 20_000; i++) {
      const c = generateUserCode();
      expect(c).toMatch(/^[ABCDEFGHJKMNPQRSTUVWXYZ23456789]{4}-[ABCDEFGHJKMNPQRSTUVWXYZ23456789]{4}$/);
      seen.add(c);
      for (const ch of c.replace("-", "")) chars.add(ch);
    }
    expect(seen.size).toBe(20_000); // 31^8 ~ 8.5e11: a collision in 20k draws is ~2e-4
    expect(chars.size).toBe(31);
  });

  it("approve: an attacker guessing random codes is throttled to 20/min and never hits a real code", async () => {
    // Victim has a pending enrolment.
    const start = await fetch(`${baseUrl}/api/device/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ machine_id: "victim-machine", hostname: "victim-pc", platform: "win32", arch: "x64" }),
    });
    expect(start.status).toBe(200);
    const { user_code: victimCode } = (await start.json()) as { user_code: string };

    const attacker = login("attacker-1");
    const codes: number[] = [];
    for (let i = 0; i < 200; i++) {
      const guess = generateUserCode();
      if (guess === victimCode) continue; // astronomically unlikely; keep the test deterministic
      codes.push((await post("/api/device/approve", { user_code: guess }, attacker)).status);
    }
    const notFound = codes.filter((s) => s === 404).length;
    const limited = codes.filter((s) => s === 429).length;
    expect(notFound).toBe(20);
    expect(limited).toBe(codes.length - 20);
    expect(codes.some((s) => s === 200)).toBe(false);

    // Even the real code is refused while the attacker's window is exhausted.
    expect((await post("/api/device/approve", { user_code: victimCode }, attacker)).status).toBe(429);
    // ...and the victim's enrolment was never claimed.
    expect(repo.workerRepo.getByUserCode(victimCode)!.approvedAt).toBeFalsy();
  });

  it("status: is the lookup oracle rate-limited too?", async () => {
    const attacker = login("attacker-2");
    const codes: number[] = [];
    for (let i = 0; i < 200; i++) codes.push((await status(generateUserCode(), attacker)).status);
    expect(codes.filter((s) => s === 429).length).toBeGreaterThan(0);
  });
});
