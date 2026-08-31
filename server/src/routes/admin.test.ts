import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";
import type { AdminStats } from "../../../shared/types.js";

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

  it("GET /api/admin/stats counts only TESTED models/quants and TERMINAL tests, never raw table rows", async () => {
    const adminToken = await superadminSession();
    const headers = withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` });
    const fetchStats = async (): Promise<AdminStats> =>
      (await (await fetch(`${baseUrl}/api/admin/stats`, { headers })).json()) as AdminStats;
    // Delta-based assertions -- this suite shares one DB across its tests, so
    // absolute counts depend on execution order; the DELTA each fixture below
    // contributes must be exact regardless.
    const before = await fetchStats();

    // Registered but never referenced by any run -- must NOT count toward
    // modelsTested (and its Q4_K_M filename must NOT count toward quants).
    repo.registerModel({ id: "admin-stats-never-run", filename: "unused-Q4_K_M.gguf", size_bytes: 1, source: "local", metadata: {} });
    // Tested model whose quant is only parseable from the filename.
    repo.registerModel({ id: "admin-stats-model-a", filename: "Llama-3-8B-Q4_K_M.gguf", size_bytes: 1, source: "local", metadata: {} });
    // Tested model with an unparseable filename -- falls back to metadata.quant.
    repo.registerModel({ id: "admin-stats-model-b", filename: "opaque-name.gguf", size_bytes: 1, source: "local", metadata: { quant: "Q8_0" } });

    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "admin-stats-owner", login: "stats-owner", avatarUrl: null });
    const baseRun = {
      worker_name: "stats-worker",
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      status: "running" as const,
      started_at: Date.now(),
    };
    repo.createRun(owner.id, { ...baseRun, id: "admin-stats-run-a", model_id: "admin-stats-model-a", config: { model_id: "admin-stats-model-a" } as never });
    // Second run so model-b counts as tested too -- its quant can only come
    // from metadata.quant, since "opaque-name.gguf" parses to nothing.
    repo.createRun(owner.id, { ...baseRun, id: "admin-stats-run-b", model_id: "admin-stats-model-b", config: { model_id: "admin-stats-model-b" } as never });
    const sweepItem = (idx: number) => ({
      idx,
      n_prompt: 512,
      n_gen: 128,
      n_depth: 0,
      concurrency: 1,
      threads: 4,
      n_gpu_layers: 99,
      batch_size: 512,
      ubatch_size: 512,
      cache_type_k: "f16",
      cache_type_v: "f16",
      flash_attn: "off",
      mtp: "off",
      n_gpu_layers_draft: 0,
      n_cpu_moe: 0,
    });
    repo.createRunItems(undefined, "admin-stats-run-a", [sweepItem(0), sweepItem(1), sweepItem(2)]);
    repo.recordRunItemTerminal("admin-stats-run-a", 0, { status: "done" });
    repo.recordRunItemTerminal("admin-stats-run-a", 1, { status: "failed_oom" });
    // idx 2 stays 'queued' -- planned but never performed, so tests must not
    // count it.

    const after = await fetchStats();
    expect(after.runs - before.runs).toBe(2);
    expect(after.modelsTested - before.modelsTested).toBe(2);
    expect(after.quants - before.quants).toBe(2);
    expect(after.tests - before.tests).toBe(2);
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

// The supervise dashboard's own two platform-wide toggles (shared/types.ts's
// AppSettings) -- item 4 of the Settings rework: community sharing starts
// greyed out (default false) until an operator turns it on here, account
// deletion starts allowed (default true, matching what already shipped).
describe("admin settings (AppSettings toggles)", () => {
  it("GET /api/admin/settings reports the documented defaults on a fresh DB", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { communitySharingAllowed: boolean; accountDeletionAllowed: boolean };
    expect(body.communitySharingAllowed).toBe(false);
    expect(body.accountDeletionAllowed).toBe(true);
  });

  it("POST /api/admin/settings flips one flag without disturbing the other, and persists", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const res1 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ communitySharingAllowed: true }),
    });
    expect(res1.status).toBe(200);
    expect((await res1.json()) as { communitySharingAllowed: boolean }).toMatchObject({ communitySharingAllowed: true });

    const res2 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ accountDeletionAllowed: false }),
    });
    expect(res2.status).toBe(200);
    const body2 = (await res2.json()) as { communitySharingAllowed: boolean; accountDeletionAllowed: boolean };
    // communitySharingAllowed from the previous call must still be true --
    // this call only touched accountDeletionAllowed.
    expect(body2.communitySharingAllowed).toBe(true);
    expect(body2.accountDeletionAllowed).toBe(false);

    // Reset back to the defaults other tests in this file/process rely on.
    await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ communitySharingAllowed: false, accountDeletionAllowed: true }),
    });
  });

  it("400s a non-boolean value instead of silently coercing it", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers: { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) },
      body: JSON.stringify({ communitySharingAllowed: "yes" }),
    });
    expect(res.status).toBe(400);
  });

  it("workerVramDiscrepancyPolicy defaults to warn and persists an accepted value", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const res1 = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res1.status).toBe(200);
    // Fresh DB -- the documented default is the shipped v1 behavior.
    expect(((await res1.json()) as { workerVramDiscrepancyPolicy?: string }).workerVramDiscrepancyPolicy ?? "warn").toBe("warn");

    const res2 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ workerVramDiscrepancyPolicy: "retry_once_then_fail" }),
    });
    expect(res2.status).toBe(200);
    const body2 = (await res2.json()) as { workerVramDiscrepancyPolicy: string };
    expect(body2.workerVramDiscrepancyPolicy).toBe("retry_once_then_fail");

    // Restore the default for later tests in this file/process.
    await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ workerVramDiscrepancyPolicy: "warn" }),
    });
  });

  it("400s a policy value outside the whitelist instead of storing it", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers: { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) },
      body: JSON.stringify({ workerVramDiscrepancyPolicy: "yolo" }),
    });
    expect(res.status).toBe(400);
  });

  it("probeMaxLoads defaults to 24 and persists an accepted override", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const res1 = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res1.status).toBe(200);
    // Fresh DB -- the documented default.
    expect(((await res1.json()) as { probeMaxLoads?: number }).probeMaxLoads ?? 24).toBe(24);

    const res2 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ probeMaxLoads: 40 }),
    });
    expect(res2.status).toBe(200);
    const body2 = (await res2.json()) as { probeMaxLoads: number };
    expect(body2.probeMaxLoads).toBe(40);

    // Restore the default for later tests in this file/process.
    await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: 24 }) });
  });

  it("400s a probeMaxLoads value outside [1, 200] instead of storing it", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const tooLow = await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: 0 }) });
    expect(tooLow.status).toBe(400);

    const tooHigh = await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: 201 }) });
    expect(tooHigh.status).toBe(400);

    const notAnInt = await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: 12.5 }) });
    expect(notAnInt.status).toBe(400);

    const notANumber = await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: "24" }) });
    expect(notANumber.status).toBe(400);
  });

  it("404s on the main hostname, same as every other admin route", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(MAIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res.status).toBe(404);
  });
});
