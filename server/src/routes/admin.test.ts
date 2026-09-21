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
  // The main site's own read routes, registered beside the admin ones exactly
  // as index.ts does -- the "per-test observation" suite compares the two
  // surfaces' payloads and checks the main site is still tenant-scoped.
  const { testsRoutes } = await import("./tests.js");
  const { curveRoutes } = await import("./curves.js");
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
  await app.register(testsRoutes);
  await app.register(curveRoutes);
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
    // sweep.repeats: 2 so the runs assertion below actually exercises the
    // items x repeats multiplication, not just item counting.
    repo.createTest(owner.id, {
      ...baseRun,
      id: "admin-stats-run-a",
      model_id: "admin-stats-model-a",
      config: { model_id: "admin-stats-model-a", sweep: { repeats: 2 } } as never,
    });
    // Second run so model-b counts as tested too -- its quant can only come
    // from metadata.quant, since "opaque-name.gguf" parses to nothing. No
    // items created for this one, so it contributes 0 to `runs` regardless
    // of repeats.
    repo.createTest(owner.id, { ...baseRun, id: "admin-stats-run-b", model_id: "admin-stats-model-b", config: { model_id: "admin-stats-model-b" } as never });
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
    repo.createTestItems(undefined, "admin-stats-run-a", [sweepItem(0), sweepItem(1), sweepItem(2)]);
    repo.recordTestItemTerminal("admin-stats-run-a", 0, { status: "done" });
    repo.recordTestItemTerminal("admin-stats-run-a", 1, { status: "failed_oom" });
    // idx 2 stays 'queued' -- planned but never performed, so tests must not
    // count it.

    const after = await fetchStats();
    // Individual physical executions (items x repeats), not run rows: run-a
    // has 3 items at repeats=2 (=6), run-b has 0 items (=0) -- NOT the 2 run
    // rows a raw COUNT(*) FROM runs would give.
    expect(after.runs - before.runs).toBe(6);
    expect(after.modelsTested - before.modelsTested).toBe(2);
    expect(after.quants - before.quants).toBe(2);
    expect(after.tests - before.tests).toBe(2);
  });

  it("GET /api/admin/stats.machines excludes enrolments that never heartbeated", async () => {
    const adminToken = await superadminSession();
    const headers = withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` });
    const fetchStats = async (): Promise<AdminStats> =>
      (await (await fetch(`${baseUrl}/api/admin/stats`, { headers })).json()) as AdminStats;
    const before = await fetchStats();

    // Enrolled but abandoned -- never sent a single heartbeat. Must not
    // count as a machine (it never actually connected).
    repo.workerRepo.getOrCreateByMachineId("admin-stats-machine-abandoned", "abandoned-box");
    // Enrolled and heartbeated at least once -- must count.
    const connected = repo.workerRepo.getOrCreateByMachineId("admin-stats-machine-connected", "connected-box");
    repo.workerRepo.recordHeartbeat(
      connected.id,
      {
        machine_id: "admin-stats-machine-connected",
        capabilities: [],
        hostname: "connected-box",
        backend: "cpu",
        hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
        installed_builds: [],
        model_files: [],
        status: "idle",
      },
      null
    );

    const after = await fetchStats();
    expect(after.machines - before.machines).toBe(1);
  });

  it("GET /api/admin/runs sees a run belonging to a DIFFERENT user (cross-tenant, by design)", async () => {
    const adminToken = await superadminSession();
    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "admin-runs-owner", login: "owner", avatarUrl: null });
    repo.registerModel({ id: "admin-runs-model", filename: "m.gguf", size_bytes: 1, source: "local", metadata: {} });
    repo.createTest(owner.id, {
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

// The supervise dashboard's own platform-wide toggles (shared/types.ts's
// AppSettings) -- item 4 of the Settings rework: community sharing starts
// disabled entirely (default false) until an operator turns it on here,
// letting users individually decide starts allowed (default true, matching
// each user's own already-default-on consent flag), account deletion starts
// allowed (default true, matching what already shipped).
describe("admin settings (AppSettings toggles)", () => {
  it("GET /api/admin/settings reports the documented defaults on a fresh DB", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      communitySharingAllowed: boolean;
      communityUserChoiceAllowed: boolean;
      accountDeletionAllowed: boolean;
    };
    expect(body.communitySharingAllowed).toBe(false);
    expect(body.communityUserChoiceAllowed).toBe(true);
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

  it("communityUserChoiceAllowed defaults to true and persists a toggle-off, independent of the other flags", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const before = (await (
      await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) })
    ).json()) as { accountDeletionAllowed: boolean };

    const res1 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ communityUserChoiceAllowed: false }),
    });
    expect(res1.status).toBe(200);
    const body1 = (await res1.json()) as { communityUserChoiceAllowed: boolean; accountDeletionAllowed: boolean };
    expect(body1.communityUserChoiceAllowed).toBe(false);
    // Untouched by this POST -- whatever it was before stays as-is.
    expect(body1.accountDeletionAllowed).toBe(before.accountDeletionAllowed);

    // Restore the default for later tests in this file/process.
    await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ communityUserChoiceAllowed: true }),
    });
  });

  it("400s a non-boolean communityUserChoiceAllowed value instead of silently coercing it", async () => {
    const adminToken = await superadminSession();
    const res = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers: { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) },
      body: JSON.stringify({ communityUserChoiceAllowed: "yes" }),
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

  it("probeMaxLoads defaults to 40 and persists an accepted override", async () => {
    const adminToken = await superadminSession();
    const headers = { "content-type": "application/json", ...withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) };

    const res1 = await fetch(`${baseUrl}/api/admin/settings`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${adminToken}` }) });
    expect(res1.status).toBe(200);
    // Fresh DB -- the documented default.
    expect(((await res1.json()) as { probeMaxLoads?: number }).probeMaxLoads ?? 40).toBe(40);

    const res2 = await fetch(`${baseUrl}/api/admin/settings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ probeMaxLoads: 32 }),
    });
    expect(res2.status).toBe(200);
    const body2 = (await res2.json()) as { probeMaxLoads: number };
    expect(body2.probeMaxLoads).toBe(32);

    // Restore the default for later tests in this file/process.
    await fetch(`${baseUrl}/api/admin/settings`, { method: "POST", headers, body: JSON.stringify({ probeMaxLoads: 40 }) });
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

// The console renders any user's test with the main site's own TestDetail
// page, which reads a fixed set of endpoints. Each has a read-only mirror
// under /api/admin/tests|models|workers/... built from the SAME handler as the
// main route (a `(scope) => handler` factory), so what an operator sees is what
// the owner sees. These pin the three properties that make that safe:
// cross-tenant on the admin origin, still tenant-scoped on the main site, and
// unable to change anything.
describe("admin per-test observation (mirrors of TestDetail's reads)", () => {
  const RESULT = {
    test_type: "tg",
    n_prompt: 512,
    n_gen: 128,
    n_threads: 8,
    n_gpu_layers: 0,
    batch_size: 2048,
    ubatch_size: 512,
    avg_tps: 50,
    stddev_tps: 0,
    ram_peak_mib: 2000,
    ram_avg_mib: 1800,
    vram_peak_mib: null,
    vram_avg_mib: null,
    ram_free_before_mib: null,
    vram_free_before_mib: null,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    system_memory_total_mb: null,
    gpu_memory_total_mb: null,
    gpu_layers_loaded: null,
    total_model_layers: null,
    gpu_memory_total_accuracy: "unavailable",
    gpu_memory_free_start_accuracy: "unavailable",
    gpu_memory_model_avg_accuracy: "unavailable",
    gpu_memory_model_peak_accuracy: "unavailable",
    gpu_memory_total_source: null,
    gpu_memory_free_start_source: null,
    gpu_memory_model_avg_source: null,
    gpu_memory_model_peak_source: null,
  };
  const sweepItem = (idx: number, n_prompt: number) => ({
    idx,
    n_prompt,
    n_gen: 128,
    n_depth: 0,
    concurrency: 1,
    threads: 8,
    n_gpu_layers: 0,
    batch_size: 2048,
    ubatch_size: 512,
    cache_type_k: "f16",
    cache_type_v: "f16",
    flash_attn: "on",
    mtp: "off",
    n_gpu_layers_draft: 0,
    n_cpu_moe: 0,
  });

  let ownerA: { id: string; token: string };
  let ownerB: { id: string; token: string };
  let workerId: string;
  const MODEL = "observe-model";

  // One finished test per tenant on the SAME model, with different contexts so
  // a curve can be traced back to whose rows it was built from.
  function seedTest(userId: string, id: string, nPrompt: number, workerIdForTest: string | null): void {
    repo.createTest(userId, {
      id,
      worker_name: "observe-box",
      worker_id: workerIdForTest,
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: MODEL,
      config: { model_id: MODEL, sweep: { repeats: 1 } } as never,
      status: "done",
      started_at: Date.now(),
    } as never);
    repo.createTestItems(undefined, id, [sweepItem(0, nPrompt)]);
    repo.recordTestItemTerminal(id, 0, { status: "done", results: [{ ...RESULT, n_prompt: nPrompt }] as never });
  }

  beforeAll(async () => {
    repo.registerModel({ id: MODEL, filename: "observe.gguf", size_bytes: 1, source: "local", metadata: {} });
    const a = repo.userRepo.upsertByIdentity("github", { providerUserId: "observe-a", login: "observe-a", avatarUrl: null });
    const b = repo.userRepo.upsertByIdentity("github", { providerUserId: "observe-b", login: "observe-b", avatarUrl: null });
    ownerA = { id: a.id, token: repo.sessionRepo.create(a.id, { label: "a" }).token };
    ownerB = { id: b.id, token: repo.sessionRepo.create(b.id, { label: "b" }).token };
    workerId = repo.workerRepo.getOrCreateByMachineId("observe-machine", "observe-box").id;
    seedTest(a.id, "observe-a-test", 512, workerId);
    seedTest(b.id, "observe-b-test", 1024, null);
  });

  const adminGet = async (path: string) =>
    fetch(`${baseUrl}${path}`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${await superadminSession()}` }) });

  const MIRRORS = [
    "/api/admin/tests/observe-a-test",
    "/api/admin/tests/observe-a-test/batch-members",
    "/api/admin/tests/observe-a-test/probe-attempts",
    "/api/admin/tests/observe-a-test/profiles",
    "/api/admin/tests/observe-a-test/knee",
    "/api/admin/tests/observe-a-test/log",
    "/api/admin/tests/observe-a-test/export",
    "/api/admin/tests/observe-a-test/summary",
    `/api/admin/models/${MODEL}/curve`,
    "/api/admin/workers",
    "/api/admin/models",
  ];

  it("returns another user's test to the superadmin -- identical to the owner's own view", async () => {
    const viaAdmin = await adminGet("/api/admin/tests/observe-a-test");
    expect(viaAdmin.status).toBe(200);
    const adminBody = (await viaAdmin.json()) as { run: { id: string }; results: unknown[]; items: unknown[] };
    expect(adminBody.run.id).toBe("observe-a-test");
    expect(adminBody.results).toHaveLength(1);
    expect(adminBody.items).toHaveLength(1);

    const viaOwner = await fetch(`${baseUrl}/api/tests/observe-a-test`, { headers: { authorization: `Bearer ${ownerA.token}` } });
    expect(viaOwner.status).toBe(200);
    expect(adminBody).toEqual(await viaOwner.json());
  });

  it("leaves the main site tenant-scoped: another user -- even the superadmin -- still can't read it there", async () => {
    const asOtherUser = await fetch(`${baseUrl}/api/tests/observe-a-test`, { headers: { authorization: `Bearer ${ownerB.token}` } });
    expect(asOtherUser.status).toBe(404);
    const asSuperadmin = await fetch(`${baseUrl}/api/tests/observe-a-test`, {
      headers: { authorization: `Bearer ${await superadminSession()}` },
    });
    expect(asSuperadmin.status).toBe(404);
  });

  it("404s every mirror on the main hostname, even for a valid superadmin session", async () => {
    const headers = withHost(MAIN_HOST, { authorization: `Bearer ${await superadminSession()}` });
    for (const path of MIRRORS) {
      const res = await fetch(`${baseUrl}${path}`, { headers });
      expect(res.status, path).toBe(404);
    }
  });

  it("403s every mirror for a signed-in user who isn't a superadmin, and for no session", async () => {
    for (const path of MIRRORS) {
      const asUser = await fetch(`${baseUrl}${path}`, { headers: withHost(ADMIN_HOST, { authorization: `Bearer ${ownerA.token}` }) });
      expect(asUser.status, `${path} as owner`).toBe(403);
      const anon = await fetch(`${baseUrl}${path}`, { headers: withHost(ADMIN_HOST) });
      expect(anon.status, `${path} anonymous`).toBe(403);
    }
  });

  it("answers every mirror for a real test on the admin origin", async () => {
    for (const path of MIRRORS) {
      const res = await adminGet(path);
      // The log endpoint 404s with its own message once it HAS resolved the test
      // (no log was ever pushed) -- distinct from "run not found", which would
      // mean the scope wrongly hid it.
      if (path.endsWith("/log")) {
        expect(res.status, path).toBe(404);
        expect(((await res.json()) as { error: string }).error).toBe("no log file for this run");
      } else {
        expect(res.status, path).toBe(200);
      }
    }
  });

  it("404s an unknown test id with the same message the main site uses", async () => {
    const res = await adminGet("/api/admin/tests/no-such-test");
    expect(res.status).toBe(404);
    expect(((await res.json()) as { error: string }).error).toBe("run not found");
    expect((await adminGet("/api/admin/tests/no-such-test/summary")).status).toBe(404);
  });

  it("summary names the owning user and the machine", async () => {
    const res = await adminGet("/api/admin/tests/observe-a-test/summary");
    const body = (await res.json()) as { id: string; userId: string; userDisplayName: string; workerId: string; workerDisplayName: string };
    expect(body.id).toBe("observe-a-test");
    expect(body.userId).toBe(ownerA.id);
    expect(body.userDisplayName).toBe("observe-a");
    expect(body.workerId).toBe(workerId);
    expect(body.workerDisplayName).toBe("observe-box");
  });

  it("probe-attempts follows the main route: a test with no machine attached is a 404, not an empty ladder", async () => {
    expect((await adminGet("/api/admin/tests/observe-b-test/probe-attempts")).status).toBe(404);
    const withMachine = await adminGet("/api/admin/tests/observe-a-test/probe-attempts");
    expect(withMachine.status).toBe(200);
    expect(((await withMachine.json()) as { attempts: unknown[] }).attempts).toEqual([]);
  });

  it("lists every tenant's machines and the model registry", async () => {
    const workers = (await (await adminGet("/api/admin/workers")).json()) as { workers: { id: string }[] };
    expect(workers.workers.some((w) => w.id === workerId)).toBe(true);
    const models = (await (await adminGet("/api/admin/models")).json()) as { models: { id: string }[] };
    expect(models.models.some((m) => m.id === MODEL)).toBe(true);
  });

  it("scopes the curve to the test's owner when ?user= is given, and spans tenants when it isn't", async () => {
    const testIds = async (query: string): Promise<string[]> => {
      const res = await adminGet(`/api/admin/models/${MODEL}/curve${query}`);
      expect(res.status).toBe(200);
      const body = (await res.json()) as { points: { testId: string }[] };
      return [...new Set(body.points.map((p) => p.testId))].sort();
    };
    expect(await testIds(`?user=${ownerA.id}`)).toEqual(["observe-a-test"]);
    expect(await testIds(`?user=${ownerB.id}`)).toEqual(["observe-b-test"]);
    expect(await testIds("")).toEqual(["observe-a-test", "observe-b-test"]);

    // ...and it matches what that owner's own request returns on the main site.
    const viaAdmin = (await (await adminGet(`/api/admin/models/${MODEL}/curve?user=${ownerA.id}`)).json()) as { points: unknown[] };
    const viaOwner = (await (
      await fetch(`${baseUrl}/api/models/${MODEL}/curve`, { headers: { authorization: `Bearer ${ownerA.token}` } })
    ).json()) as { points: unknown[] };
    expect(viaAdmin.points).toEqual(viaOwner.points);
  });

  it("is read-only: nothing under /api/admin can stop, pause, resume, trigger or delete a test", async () => {
    const token = await superadminSession();
    const bare = withHost(ADMIN_HOST, { authorization: `Bearer ${token}` });
    for (const [method, path] of [
      ["POST", "/api/admin/tests/observe-a-test/stop"],
      ["POST", "/api/admin/tests/observe-a-test/pause"],
      ["POST", "/api/admin/tests/observe-a-test/resume"],
      ["POST", "/api/admin/tests/trigger"],
      ["DELETE", "/api/admin/tests/observe-a-test"],
      ["POST", "/api/admin/tests/observe-a-test"],
    ] as const) {
      // A JSON content-type only where there's a body: Fastify rejects an empty
      // JSON body with a 400 before routing, which would mask a real route.
      const res =
        method === "POST"
          ? await fetch(`${baseUrl}${path}`, { method, headers: { ...bare, "content-type": "application/json" }, body: "{}" })
          : await fetch(`${baseUrl}${path}`, { method, headers: bare });
      expect(res.status, `${method} ${path}`).toBe(404);
    }
    // ...and the test is untouched.
    const still = (await (await adminGet("/api/admin/tests/observe-a-test")).json()) as { run: { status: string } };
    expect(still.run.status).toBe("done");
  });
});
