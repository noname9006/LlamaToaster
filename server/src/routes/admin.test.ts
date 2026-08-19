import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the admin surface's own two
// gates -- hostname (404 on mismatch, checked first) and isSuperadmin (403).
// SUPERADMIN_IDENTITIES must be set BEFORE superadmin.ts is ever imported --
// it's read into a module-level Set at import time, not per-request (see
// that file's own doc comment) -- same reasoning as DB_PATH needing to be
// set before repo.js's first import.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-admin-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.SUPERADMIN_IDENTITIES = "github:admin-test-id";
process.env.ADMIN_PUBLIC_URL = "http://supervise.test.local";

const ADMIN_HOST = "supervise.test.local";
const MAIN_HOST = "app.test.local";

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { adminRoutes } = await import("./admin.js");
  const { authMiddleware } = await import("../auth-middleware.js");

  // trustProxy so req.hostname reads X-Forwarded-Host -- plain fetch() can't
  // override the real Host header (it's on the Fetch spec's forbidden-header
  // list; undici silently ignores an attempt), so tests simulate hitting a
  // second hostname this way instead of via raw sockets. Test-scoped only --
  // the real server/src/index.ts app doesn't set this.
  app = Fastify({ logger: false, trustProxy: true });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  // Registered exactly like index.ts's real AUTH_ENABLED=true setup -- the
  // ordering test below depends on this actually being present, not just
  // adminRoutes' own hook in isolation.
  app.addHook("preHandler", authMiddleware);
  await app.register(adminRoutes);
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

function withHost(host: string, extra: Record<string, string> = {}): Record<string, string> {
  return { "x-forwarded-host": host, ...extra };
}

async function superadminSession(): Promise<string> {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "admin-test-id", login: "admin", avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: "admin session" });
  return token;
}

async function normalSession(login: string): Promise<string> {
  const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
  const { token } = repo.sessionRepo.create(user.id, { label: login });
  return token;
}

describe("admin hostname gate", () => {
  it("404s on the main hostname even with a valid superadmin session -- never reveals the surface exists", async () => {
    const token = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/stats`, {
      headers: withHost(MAIN_HOST, { authorization: `Bearer ${token}` }),
    });
    expect(res.status).toBe(404);
  });

  it("404s on the main hostname with NO session at all (proves the hostname check runs before authMiddleware's 401)", async () => {
    // The critical ordering assertion: authMiddleware (registered on the
    // parent, preHandler) would 401 an unauthenticated request to a route
    // it doesn't recognize as public -- if that ran BEFORE adminRoutes' own
    // onRequest hook, this would come back 401, not 404, leaking that an
    // auth-gated surface exists at this path on the wrong hostname.
    const res = await fetch(`${baseUrl}/api/admin/stats`, { headers: withHost(MAIN_HOST) });
    expect(res.status).toBe(404);
  });

  it("403s on the admin hostname with no session", async () => {
    const res = await fetch(`${baseUrl}/api/admin/stats`, { headers: withHost(ADMIN_HOST) });
    expect(res.status).toBe(403);
  });

  it("403s on the admin hostname for a real session that ISN'T superadmin-listed", async () => {
    const token = await normalSession("regular-user-1");
    const res = await fetch(`${baseUrl}/api/admin/stats`, {
      headers: withHost(ADMIN_HOST, { authorization: `Bearer ${token}` }),
    });
    expect(res.status).toBe(403);
  });

  it("200s on the admin hostname for a real superadmin session", async () => {
    const token = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/stats`, {
      headers: withHost(ADMIN_HOST, { authorization: `Bearer ${token}` }),
    });
    expect(res.status).toBe(200);
  });
});

describe("admin routes, cross-tenant by design", () => {
  it("GET /api/admin/stats reports global counts, not scoped to the caller", async () => {
    const token = await superadminSession();
    await normalSession("stats-user-a");
    await normalSession("stats-user-b");
    const res = await fetch(`${baseUrl}/api/admin/stats`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${token}` }) });
    const body = (await res.json()) as { users: number };
    // At least the superadmin + the two normal users just created --
    // exact count depends on test execution order across this file, so this
    // just asserts it's NOT scoped down to "1" (the caller alone).
    expect(body.users).toBeGreaterThanOrEqual(3);
  });

  it("GET /api/admin/runs sees a run belonging to a DIFFERENT user (cross-tenant, by design)", async () => {
    const adminToken = await superadminSession();
    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "admin-runs-owner", login: "owner", avatarUrl: null });
    repo.registerModel({ id: "admin-runs-model", filename: "m.gguf", size_bytes: 1, source: "local", metadata: {} });
    repo.createRun(owner.id, {
      id: "admin-visible-run",
      worker_name: "someones-box",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: "admin-runs-model",
      config: { model_id: "admin-runs-model" } as never,
      status: "done",
      started_at: Date.now(),
    });

    const res = await fetch(`${baseUrl}/api/admin/runs`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    const body = (await res.json()) as { runs: { id: string; userDisplayName: string | null }[] };
    const found = body.runs.find((r) => r.id === "admin-visible-run");
    expect(found).toBeDefined();
    expect(found?.userDisplayName).toBe("owner");
  });

  it("GET /api/admin/runs filters by status", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/runs?status=done`, {
      headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }),
    });
    const body = (await res.json()) as { runs: { status: string }[] };
    expect(body.runs.every((r) => r.status === "done")).toBe(true);
  });

  it("GET /api/admin/results/export returns unscoped rows, 404s on the main hostname", async () => {
    const adminToken = await superadminSession();
    const okRes = await fetch(`${baseUrl}/api/admin/results/export?format=json`, {
      headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }),
    });
    expect(okRes.status).toBe(200);

    const blockedRes = await fetch(`${baseUrl}/api/admin/results/export?format=json`, {
      headers: withHost(MAIN_HOST, { authorization: `Bearer ${adminToken}` }),
    });
    expect(blockedRes.status).toBe(404);
  });

  it("GET /api/admin/users lists every account", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/users`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    const body = (await res.json()) as { users: { id: string }[] };
    expect(body.users.length).toBeGreaterThanOrEqual(1);
  });
});
