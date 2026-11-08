// Audit 2026-11-01: small route and queue defects that have no better home.
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";
import fastifyRateLimit from "@fastify/rate-limit";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-routefix-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.LOG_DIR = join(tmpDir, "logs");
process.env.AUTH_ENABLED = "true";
process.env.PUBLIC_URL = "https://main.test";
// The burst below is about this app's per-user limit, not the instance-wide HF one.
process.env.HF_API_RATE_LIMIT = "100000";

let app: FastifyInstance;
let repo: (typeof import("../db/repo.js"))["repo"];
let getDb: (typeof import("../db/migrate.js"))["getDb"];
let closeDb: (typeof import("../db/migrate.js"))["closeDb"];
let hashToken: (typeof import("../session.js"))["hashToken"];

let seq = 0;
function user(name: string) {
  const label = `${name}-${++seq}`;
  const u = repo.userRepo.upsertByIdentity("github", { providerUserId: label, login: label, avatarUrl: null });
  const { token } = repo.sessionRepo.create(u.id, { label });
  return { id: u.id, h: { authorization: `Bearer ${token}` } };
}
function ownedWorker(userId: string) {
  const machineId = `box-${++seq}`;
  const worker = repo.workerRepo.getOrCreateByMachineId(machineId, machineId);
  getDb().prepare(`UPDATE workers SET user_id = ?, approved_at = ?, public_key = 'key' WHERE id = ?`).run(userId, Date.now(), worker.id);
  return worker.id;
}
function workerSession(userId: string, workerId: string) {
  return repo.sessionRepo.create(userId, { isWorker: true, workerId, label: "w" });
}
const refresh = (token: string) => app.inject({ method: "POST", url: "/api/auth/refresh", headers: { authorization: `Bearer ${token}` } });

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ getDb, closeDb } = await import("../db/migrate.js"));
  ({ hashToken } = await import("../session.js"));
  const { authRoutes } = await import("./auth.js");
  const { sessionRoutes } = await import("./sessions.js");
  const { deviceRoutes, deviceApprovalRoutes } = await import("./device.js");
  const { optimizeRoutes } = await import("./optimize.js");
  const { workersRoutes } = await import("./workers.js");
  const { modelsRoutes } = await import("./models.js");
  const { authMiddleware } = await import("../auth-middleware.js");
  const { appErrorHandler } = await import("../error-handler.js");

  app = Fastify();
  await app.register(fastifyCookie);
  await app.register(fastifyRateLimit, { global: false });
  app.addHook("preHandler", authMiddleware);
  for (const plugin of [authRoutes, sessionRoutes, deviceRoutes, deviceApprovalRoutes, optimizeRoutes, workersRoutes, modelsRoutes]) {
    await app.register(plugin);
  }
  app.setErrorHandler(appErrorHandler);
  await app.ready();
});

afterAll(async () => {
  await app.close();
  closeDb();
});

describe("refresh tokens (M6, N16)", () => {
  it("a retry right after a lost response rotates again, leaving the machine trusted", async () => {
    const u = user("grace");
    const workerId = ownedWorker(u.id);
    const s = workerSession(u.id, workerId);
    expect((await refresh(s.refresh!)).statusCode).toBe(200);
    const retry = await refresh(s.refresh!);
    expect(retry.statusCode).toBe(200);
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBe("key");
    expect((await refresh(retry.json().refresh_token)).statusCode).toBe(200);
  });

  it("the same token past the window is theft: the session and the machine's key go", async () => {
    const u = user("replay");
    const workerId = ownedWorker(u.id);
    const s = workerSession(u.id, workerId);
    const first = await refresh(s.refresh!);
    expect(first.statusCode).toBe(200);
    getDb()
      .prepare(`UPDATE sessions SET rotated_at = ? WHERE refresh_hash = ?`)
      .run(Date.now() - 120_000, hashToken(first.json().refresh_token));
    expect((await refresh(s.refresh!)).statusCode).toBe(401);
    expect(repo.sessionRepo.getByRefreshHash(hashToken(first.json().refresh_token))).toBeUndefined();
    expect(repo.workerRepo.getEnrolmentById(workerId)?.publicKey).toBeNull();
  });

  it("a row rotated before the column existed has no grace", async () => {
    const u = user("legacy");
    const workerId = ownedWorker(u.id);
    const s = workerSession(u.id, workerId);
    const first = await refresh(s.refresh!);
    getDb().prepare(`UPDATE sessions SET rotated_at = NULL WHERE refresh_hash = ?`).run(hashToken(first.json().refresh_token));
    expect((await refresh(s.refresh!)).statusCode).toBe(401);
  });

  it("an expired refresh token stays expired on the second try", async () => {
    const u = user("expired");
    const s = workerSession(u.id, ownedWorker(u.id));
    getDb()
      .prepare(`UPDATE sessions SET expires_at = ?, last_seen_at = ? WHERE refresh_hash = ?`)
      .run(Date.now() - 1000, Date.now() - 1e10, hashToken(s.refresh!));
    expect((await refresh(s.refresh!)).statusCode).toBe(401);
    expect((await refresh(s.refresh!)).statusCode).toBe(401);
  });
});

describe("public routes (D6, D7, L6, L8)", () => {
  it("device routes enforce their own body limits", async () => {
    const big = JSON.stringify({ machine_id: "m", pad: "x".repeat(200_000) });
    const headers = { "content-type": "application/json" };
    for (const url of ["/api/device/challenge", "/api/device/token", "/api/device/start"]) {
      expect((await app.inject({ method: "POST", url, headers, payload: big })).statusCode).toBe(413);
    }
    // The largest machine the hardware validator accepts still fits.
    const cores = Array.from({ length: 1024 }, (_, id) => ({ id, logical: [id, id + 1024], l3: 0, die: 0, effClass: 0 }));
    const hardware = {
      platform: "linux",
      arch: "x64",
      cpu: { manufacturer: "x", brand: "x", flags: Array(50).fill("avx512_vnni"), cores: 2048 },
      cpu_topology: { source: "linux", cores, logicalCount: 2048, masksSupported: true },
      gpu: [],
    };
    const huge = await app.inject({
      method: "POST",
      url: "/api/device/start",
      headers,
      payload: JSON.stringify({ machine_id: "huge-box", hostname: "h", platform: "linux", arch: "x64", hardware }),
    });
    expect(huge.statusCode).toBe(200);
    expect((await app.inject({ method: "POST", url: "/api/device/challenge", headers, payload: JSON.stringify({ machine_id: "m" }) })).statusCode).toBe(200);
  });

  it("a real provider name is the only thing /auth/:provider recognises", async () => {
    for (const name of ["constructor", "__proto__", "toString", "nope"]) {
      expect((await app.inject({ method: "GET", url: `/auth/${name}` })).statusCode).toBe(404);
      expect((await app.inject({ method: "GET", url: `/auth/${name}/callback` })).statusCode).toBe(404);
    }
    expect((await app.inject({ method: "GET", url: "/auth/github" })).statusCode).toBe(302);
  });

  it("approving with no body is a 400", async () => {
    const u = user("approve");
    expect((await app.inject({ method: "POST", url: "/api/device/approve", headers: u.h })).statusCode).toBe(400);
  });

  it("logging out takes a POST", async () => {
    const u = user("logout");
    expect((await app.inject({ method: "GET", url: "/auth/logout", headers: u.h })).statusCode).toBe(404);
    expect((await app.inject({ method: "POST", url: "/auth/logout", headers: u.h })).statusCode).toBe(200);
    expect((await app.inject({ method: "GET", url: "/api/sessions", headers: u.h })).statusCode).toBe(401);
  });
});

describe("Hugging Face proxy limits (N12)", () => {
  it("one user's burst stops at the limit, and another user is unaffected", async () => {
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => new Response("[]", { status: 200 })) as typeof fetch;
    try {
      const heavy = user("hf-heavy");
      const search = (h: Record<string, string>) => app.inject({ method: "GET", url: "/api/hf/search?q=x", headers: h });
      for (let i = 0; i < 600; i++) expect((await search(heavy.h)).statusCode).toBe(200);
      expect((await search(heavy.h)).statusCode).toBe(429);
      expect((await search(user("hf-other").h)).statusCode).toBe(200);

      const tree = await app.inject({ method: "GET", url: "/api/hf/repo/a/b", headers: heavy.h });
      expect(tree.statusCode).toBe(200);
      const updater = user("hf-updates");
      for (let i = 0; i < 240; i++) {
        expect((await app.inject({ method: "GET", url: "/api/models/hf-updates", headers: updater.h })).statusCode).toBe(200);
      }
      expect((await app.inject({ method: "GET", url: "/api/models/hf-updates", headers: updater.h })).statusCode).toBe(429);
    } finally {
      globalThis.fetch = realFetch;
    }
  }, 60_000);
});

describe("queue lifecycle (A9, R2, R5, A7, B6, D10)", () => {
  function runningJob(workerId: string) {
    const runId = `run-${++seq}`;
    getDb()
      .prepare(`INSERT INTO runs (id, root_run_id, worker_id, model_id, config, status, started_at) VALUES (?, ?, ?, 'm', '{}', 'scheduled', 1)`)
      .run(runId, runId, workerId);
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "benchmark", payload: {}, runId });
    return { runId, jobId };
  }

  it("a requeued job keeps the user's pause, a new run still starts unpaused", () => {
    const u = user("pause");
    const workerId = ownedWorker(u.id);
    const { jobId } = runningJob(workerId);
    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(jobId);
    repo.workerRepo.setPauseRequested(workerId, true);
    repo.queueRepo.returnToPending(jobId);
    expect(repo.queueRepo.claimNextJob(workerId)?.job_id).toBe(jobId);
    expect(repo.workerRepo.getPauseRequested(workerId)).toBe(true);

    repo.queueRepo.markJobCompleted(jobId);
    runningJob(workerId);
    repo.queueRepo.claimNextJob(workerId);
    expect(repo.workerRepo.getPauseRequested(workerId)).toBe(false);
  });

  it("an undone claim puts the run back to scheduled, and nothing else", () => {
    const u = user("unclaim");
    const workerId = ownedWorker(u.id);
    const { runId, jobId } = runningJob(workerId);
    repo.queueRepo.claimNextJob(workerId);
    expect(repo.getTest(undefined, runId)?.status).toBe("running");
    repo.queueRepo.unclaimJob(jobId);
    expect(repo.getTest(undefined, runId)?.status).toBe("scheduled");
    expect(repo.queueRepo.getJob(jobId)?.status).toBe("pending");

    // A second undo of a job that moved on changes nothing.
    repo.queueRepo.claimNextJob(workerId);
    repo.queueRepo.markJobCompleted(jobId);
    getDb().prepare(`UPDATE runs SET status = 'done' WHERE id = ?`).run(runId);
    repo.queueRepo.unclaimJob(jobId);
    expect(repo.getTest(undefined, runId)?.status).toBe("done");
  });

  it("the stall clock also covers a job with no run", () => {
    const u = user("stall");
    const workerId = ownedWorker(u.id);
    const jobId = repo.queueRepo.enqueueJob(workerId, { type: "fit_check", payload: { models: [] } });
    repo.queueRepo.claimNextJob(workerId);
    expect(repo.queueRepo.listStalledJobs(Date.now() - 1000).some((j) => j.id === jobId)).toBe(false);
    getDb().prepare(`UPDATE worker_jobs SET progress_at = ? WHERE id = ?`).run(Date.now() - 5000, jobId);
    expect(repo.queueRepo.listStalledJobs(Date.now() - 1000).some((j) => j.id === jobId)).toBe(true);
  });

  it("failed_unsupported counts as a failed item", () => {
    const u = user("unsupported");
    const workerId = ownedWorker(u.id);
    const { runId } = runningJob(workerId);
    getDb().prepare(`INSERT INTO run_items (id, run_id, idx, status) VALUES (?, ?, 0, 'failed_unsupported')`).run(`i-${seq}`, runId);
    expect(repo.getTest(undefined, runId)?.items_failed).toBe(1);
    expect(repo.listComparisonMembers("none")).toEqual([]);
  });

  it("run-scoped job lookups use the run index", () => {
    const plan = getDb().prepare(`EXPLAIN QUERY PLAN SELECT id FROM worker_jobs WHERE run_id = ? AND status = 'claimed'`).all("x") as { detail: string }[];
    expect(plan.map((p) => p.detail).join(" ")).toContain("idx_worker_jobs_run");
  });

  it("a Will it fit job doesn't read as the New test page's own", async () => {
    const u = user("fit-kinds");
    const workerId = ownedWorker(u.id);
    repo.queueRepo.enqueueJob(workerId, { type: "fit_check", payload: { target_keys: ["hf:a/b.gguf"], models: [] } });
    const list = await app.inject({ method: "GET", url: `/api/workers/${workerId}/fit-checks?ctx=4096`, headers: u.h });
    expect(list.json().pending).toBe(false);
    repo.queueRepo.enqueueJob(workerId, { type: "fit_check", payload: { models: [] } });
    const after = await app.inject({ method: "GET", url: `/api/workers/${workerId}/fit-checks?ctx=4096`, headers: u.h });
    expect(after.json().pending).toBe(true);
  });
});

// Audit 2026-11-01 E1: the public start route used to leave a permanent row behind
// every time, so anyone could grow this table without an account.
describe("abandoned enrolments (E1)", () => {
  const start = (machineId: string) =>
    app.inject({
      method: "POST",
      url: "/api/device/start",
      headers: { "content-type": "application/json" },
      payload: JSON.stringify({ machine_id: machineId, hostname: "h", platform: "linux", arch: "x64" }),
    });
  const row = (machineId: string) => getDb().prepare(`SELECT id, user_code FROM workers WHERE machine_id = ?`).get(machineId) as { id: string; user_code: string | null } | undefined;
  const expire = (machineId: string) => getDb().prepare(`UPDATE workers SET enrolment_expires_at = ? WHERE machine_id = ?`).run(Date.now() - 1000, machineId);

  it("removes one nobody approved, and a row an earlier sweep already emptied", async () => {
    expect((await start("e1-gone")).statusCode).toBe(200);
    expect((await start("e1-emptied")).statusCode).toBe(200);
    expire("e1-gone");
    getDb().prepare(`UPDATE workers SET user_code = NULL, enrolment_code_hash = NULL, enrolment_expires_at = NULL WHERE machine_id = ?`).run("e1-emptied");
    repo.workerRepo.pruneExpiredEnrolments();
    expect(row("e1-gone")).toBeUndefined();
    expect(row("e1-emptied")).toBeUndefined();
  });

  it("leaves one still in its window", async () => {
    await start("e1-live");
    repo.workerRepo.pruneExpiredEnrolments();
    expect(row("e1-live")?.user_code).toBeTruthy();
  });

  it("a machine with history keeps its row when an upgrade attempt lapses", async () => {
    // Stage 1 machines: one that has reported in, one that has runs.
    for (const machineId of ["e1-heartbeat", "e1-runs"]) repo.workerRepo.getOrCreateByMachineId(machineId, machineId);
    getDb().prepare(`UPDATE workers SET last_heartbeat_at = ? WHERE machine_id = 'e1-heartbeat'`).run(Date.now());
    getDb().prepare(`INSERT INTO runs (id, worker_id, status, started_at) VALUES ('e1-run', ?, 'done', ?)`).run(row("e1-runs")!.id, Date.now());
    for (const machineId of ["e1-heartbeat", "e1-runs"]) {
      expect((await start(machineId)).statusCode).toBe(200);
      expire(machineId);
    }
    repo.workerRepo.pruneExpiredEnrolments();
    for (const machineId of ["e1-heartbeat", "e1-runs"]) {
      expect(row(machineId)).toBeDefined();
      expect(row(machineId)?.user_code).toBeNull();
    }
  });

  it("caps how many can wait at once, without stopping one already waiting", async () => {
    const { MAX_PENDING_ENROLMENTS } = await import("./device.js");
    expect((await start("e1-waiting")).statusCode).toBe(200);
    const insert = getDb().prepare(
      `INSERT INTO workers (id, machine_id, display_name, hostname, enrolment_code_hash, enrolment_expires_at, created_at, updated_at)
       VALUES (?, ?, 'x', 'x', 'x', ?, 0, 0)`
    );
    const fill = getDb().transaction(() => {
      for (let i = 0; i < MAX_PENDING_ENROLMENTS; i++) insert.run(`e1-fill-${i}`, `e1-fill-${i}`, Date.now() + 60_000);
    });
    fill();
    try {
      expect((await start("e1-one-too-many")).statusCode).toBe(429);
      expect(row("e1-one-too-many")).toBeUndefined();
      expect((await start("e1-waiting")).statusCode).toBe(200);
    } finally {
      getDb().prepare(`DELETE FROM workers WHERE id LIKE 'e1-fill-%'`).run();
    }
    expect((await start("e1-one-too-many")).statusCode).toBe(200);
  });
});
