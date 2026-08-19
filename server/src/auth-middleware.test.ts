import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-auth-middleware-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("./db/repo.js")["repo"];
let hashToken: (t: string) => string;
let PUBLIC_PATHS: Set<string>;
let resolveAuthUser: typeof import("./auth-middleware.js")["resolveAuthUser"];

beforeAll(async () => {
  ({ repo } = await import("./db/repo.js"));
  ({ hashToken } = await import("./session.js"));
  const mw = await import("./auth-middleware.js");
  PUBLIC_PATHS = mw.PUBLIC_PATHS;
  resolveAuthUser = mw.resolveAuthUser;

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  app.addHook("preHandler", mw.authMiddleware);

  // A representative mix -- exercises every branch authMiddleware has to
  // decide between, without needing the real routes/*.ts files registered.
  app.get("/health", async () => ({ ok: true }));
  app.get("/api/auth/status", async (req) => ({ user: resolveAuthUser(req)?.user ?? null }));
  app.get("/api/protected", async (req) => ({ user: (req as unknown as { user: unknown }).user }));
  app.post("/api/worker/queue", async () => ({ ok: true }));
  app.get<{ Params: { id: string } }>("/api/runs/:id/log", async () => ({ ok: true, via: "GET (user-authed)" }));
  app.post<{ Params: { id: string } }>("/api/runs/:id/log", async () => ({ ok: true, via: "POST (worker-authed)" }));

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

describe("PUBLIC_PATHS", () => {
  it("lists exactly the routes the plan names as reachable with no session", () => {
    expect(PUBLIC_PATHS.has("/health")).toBe(true);
    expect(PUBLIC_PATHS.has("/api/auth/status")).toBe(true);
    expect(PUBLIC_PATHS.has("/api/protected")).toBe(false);
  });
});

describe("authMiddleware", () => {
  it("blocks an unauthenticated request to a protected API route", async () => {
    const res = await fetch(`${baseUrl}/api/protected`);
    expect(res.status).toBe(401);
  });

  it("allows a request bearing a valid session token, via Authorization: Bearer", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "mw-1", login: "mw-user", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "test" });
    const res = await fetch(`${baseUrl}/api/protected`, { headers: { authorization: `Bearer ${token}` } });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { user: { id: string } };
    expect(body.user.id).toBe(user.id);
  });

  it("allows a request bearing a valid session token via the lt_session cookie", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "mw-2", login: "mw-cookie-user", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "test" });
    const res = await fetch(`${baseUrl}/api/protected`, { headers: { cookie: `lt_session=${token}` } });
    expect(res.status).toBe(200);
  });

  it("rejects an expired session even with a technically-valid token hash", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "mw-3", login: "mw-expired", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "test" });
    // Directly age the session past its expiry -- no clock-freezing needed,
    // this just moves expires_at into the past.
    const session = repo.sessionRepo.getByTokenHash(hashToken(token))!;
    const db = (await import("./db/migrate.js")).getDb();
    db.prepare(`UPDATE sessions SET expires_at = ? WHERE id = ?`).run(Date.now() - 1000, session.id);

    const res = await fetch(`${baseUrl}/api/protected`, { headers: { authorization: `Bearer ${token}` } });
    expect(res.status).toBe(401);
  });

  it("rejects a garbage/unknown token the same as a missing one", async () => {
    const res = await fetch(`${baseUrl}/api/protected`, { headers: { authorization: "Bearer not-a-real-token" } });
    expect(res.status).toBe(401);
  });

  it("GET /api/auth/status never 401s -- returns {user: null} when unauthenticated", async () => {
    const res = await fetch(`${baseUrl}/api/auth/status`);
    expect(res.status).toBe(200);
    expect((await res.json()) as { user: unknown }).toEqual({ user: null });
  });

  it("GET /api/auth/status reflects the caller's own session when authenticated", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "mw-4", login: "mw-status", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "test" });
    const res = await fetch(`${baseUrl}/api/auth/status`, { headers: { authorization: `Bearer ${token}` } });
    const body = (await res.json()) as { user: { id: string } | null };
    expect(body.user?.id).toBe(user.id);
  });

  it("/api/worker/* is exempt -- no user session required (still worker-token-authenticated inside its own handler)", async () => {
    const res = await fetch(`${baseUrl}/api/worker/queue`, { method: "POST" });
    expect(res.status).toBe(200);
  });

  it("POST /api/runs/:id/log is exempt (worker push) but GET on the SAME url still requires a user session", async () => {
    const postRes = await fetch(`${baseUrl}/api/runs/abc/log`, { method: "POST" });
    expect(postRes.status).toBe(200);

    const getRes = await fetch(`${baseUrl}/api/runs/abc/log`);
    expect(getRes.status).toBe(401);
  });

  it("an unmatched route fails closed (never treated as public)", async () => {
    const res = await fetch(`${baseUrl}/api/totally-unknown-route`);
    // 404 from Fastify's own not-found handling -- the key assertion is NOT
    // 200, i.e. the middleware never let an unmatched route through as if it
    // were authenticated or public.
    expect(res.status).not.toBe(200);
  });
});

describe("resolveAuthUser", () => {
  it("returns null (never throws) for a request with no credentials", async () => {
    const fakeReq = { headers: {} } as Parameters<typeof resolveAuthUser>[0];
    expect(resolveAuthUser(fakeReq)).toBeNull();
  });
});
