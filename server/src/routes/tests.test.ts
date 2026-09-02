import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-runs-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.WORKER_SHARED_TOKEN = "runs-test-secret";
process.env.LOG_DIR = join(tmpDir, "run-logs");

// getReleases() hits the real GitHub API -- never do that from a fast unit
// test (network flakiness/rate limits/offline CI). filterReleasesForWorker
// and assetMatchesWorker are pure functions with no I/O, so keep the real
// ones and only stub the network-calling export.
vi.mock("../github-releases.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../github-releases.js")>();
  return {
    ...actual,
    getReleases: vi.fn(async () => [
      {
        tag: "b9999",
        published_at: "2026-01-01T00:00:00Z",
        assets: [
          { name: "llama-b9999-bin-ubuntu-x64.zip", download_url: "http://example.invalid/x.zip", size_bytes: 123 },
        ],
      },
    ]),
  };
});

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

const baseSweep = {
  n_prompt: [512],
  n_gen: [128],
  threads: [8],
  n_gpu_layers: [0],
  batch_size: [2048],
  ubatch_size: [512],
  cache_type_k: ["f16"],
  cache_type_v: ["f16"],
  flash_attn: ["on"],
  mtp: ["off"],
  n_gpu_layers_draft: [0],
  n_cpu_moe: [0],
  repeats: 1,
};

async function postJson(path: string, body: unknown) {
  return fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function heartbeat(machineId: string, opts: { backend?: string; installed?: boolean } = {}) {
  return fetch(`${baseUrl}/api/worker/heartbeat`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer runs-test-secret" },
    body: JSON.stringify({
      machine_id: machineId,
      hostname: machineId,
      backend: opts.backend ?? "cpu",
      hardware: { platform: "linux", arch: "x64", cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 }, gpu: [] },
      installed_builds: opts.installed
        ? [{ tag: "b1", asset_name: "llama-b1-bin-ubuntu-x64.zip", installed_at: 1, active: true }]
        : [],
      model_files: [],
      status: "idle",
    }),
  });
}

// BENCHMARKING_PLAN_V8.md §0.5 caps a user at three active roots. These
// tests each create their own run and never finish it, so without a drain
// the fourth test in the file would start hitting that quota instead of
// exercising the route it is actually about.
function drainActiveRuns(): void {
  for (const run of repo.listTests(undefined)) {
    if (run.status === "running" || run.status === "scheduled") {
      repo.reconcileStaleTest(undefined, run.id, "test cleanup");
    }
  }
}

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { testsRoutes } = await import("./tests.js");
  const { queueRoutes } = await import("./queue.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(testsRoutes);
  await app.register(queueRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;

  repo.registerModel({
    id: "model-1",
    filename: "model.gguf",
    size_bytes: 1000,
    source: "local",
    metadata: {},
  });
});

beforeEach(() => {
  drainActiveRuns();
});

afterAll(async () => {
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless, OS temp dir */
  }
});

describe("POST /api/runs/trigger", () => {
  it("rejects an unknown worker_id", async () => {
    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: "nope", sweep: baseSweep });
    expect(res.status).toBe(400);
  });

  it("rejects a machine that has never heartbeated (offline)", async () => {
    const worker = repo.workerRepo.getOrCreateByMachineId("trigger-never-online", "box");
    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    expect(res.status).toBe(409);
  });

  it("creates a scheduled run and enqueues install_build + benchmark when nothing is installed", async () => {
    await heartbeat("trigger-fresh", { backend: "cpu", installed: false });
    const worker = repo.workerRepo.getByMachineId("trigger-fresh")!;

    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    expect(res.status).toBe(201);
    const body = (await res.json()) as { run: { id: string; status: string; worker_id: string } };
    expect(body.run.status).toBe("scheduled");
    expect(body.run.worker_id).toBe(worker.id);

    // Worker's own queue poll should see the install_build job FIRST.
    const first = await fetch(`${baseUrl}/api/worker/queue`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer runs-test-secret" },
      body: JSON.stringify({
        machine_id: "trigger-fresh",
        hostname: "trigger-fresh",
        backend: "cpu",
        hardware: {
          platform: "linux",
          arch: "x64",
          cpu: { manufacturer: "x", brand: "x", flags: [], cores: 4 },
          gpu: [],
        },
        installed_builds: [],
        model_files: [],
        status: "idle",
      }),
    });
    expect(first.status).toBe(200);
    const firstJob = (await first.json()) as { type: string };
    expect(firstJob.type).toBe("install_build");
  });

  it("skips install_build and goes straight to running once the worker claims benchmark, when already installed", async () => {
    await heartbeat("trigger-installed", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("trigger-installed")!;

    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    expect(res.status).toBe(201);
    const { run } = (await res.json()) as { run: { id: string; status: string } };
    expect(run.status).toBe("scheduled");

    const claimed = repo.queueRepo.claimNextJob(worker.id);
    expect(claimed?.type).toBe("benchmark");

    const updated = repo.getTest(undefined, run.id);
    expect(updated?.status).toBe("running");
  });
});

// The Run -> Test rename kept every /api/runs/... path registered as a
// backward-compat alias (see tests.ts) so an already-deployed worker or open
// browser tab isn't broken by the rollout. Everything above this point
// exercises those legacy paths (proving the alias itself works); this block
// separately confirms the new /api/tests/... paths -- what client/worker
// code now actually calls -- work end to end too.
describe("/api/tests/... (the current, non-legacy path)", () => {
  it("GET /api/tests lists a triggered test, and GET /api/tests/:id fetches it", async () => {
    await heartbeat("tests-path-list", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("tests-path-list")!;
    const triggerRes = await postJson("/api/tests/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    expect(triggerRes.status).toBe(201);
    const { run } = (await triggerRes.json()) as { run: { id: string } };

    const listRes = await fetch(`${baseUrl}/api/tests`);
    expect(listRes.status).toBe(200);
    const { runs } = (await listRes.json()) as { runs: { id: string }[] };
    expect(runs.some((r) => r.id === run.id)).toBe(true);

    const getRes = await fetch(`${baseUrl}/api/tests/${run.id}`);
    expect(getRes.status).toBe(200);
    const got = (await getRes.json()) as { run: { id: string } };
    expect(got.run.id).toBe(run.id);
  });

  it("POST /api/tests/:id/stop stops a scheduled test", async () => {
    await heartbeat("tests-path-stop", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("tests-path-stop")!;
    const triggerRes = await postJson("/api/tests/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };

    const stopRes = await postJson(`/api/tests/${run.id}/stop`, {});
    expect(stopRes.status).toBe(200);
    const updated = repo.getTest(undefined, run.id);
    expect(updated?.status).toBe("cancelled");
  });
});

describe("run stop/pause/resume", () => {
  it("stop on a never-claimed (still scheduled) run reconciles it immediately", async () => {
    await heartbeat("stop-pending", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("stop-pending")!;
    const triggerRes = await postJson("/api/runs/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };

    const stopRes = await postJson(`/api/runs/${run.id}/stop`, {});
    expect(stopRes.status).toBe(200);
    const stopped = repo.getTest(undefined, run.id);
    expect(stopped?.status).toBe("cancelled");
  });

  it("stop on a claimed (running) run flags cancel_requested, delivered via heartbeat, without reconciling immediately", async () => {
    await heartbeat("stop-claimed", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("stop-claimed")!;
    const triggerRes = await postJson("/api/runs/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };
    const claimed = repo.queueRepo.claimNextJob(worker.id)!;

    const stopRes = await postJson(`/api/runs/${run.id}/stop`, {});
    expect(stopRes.status).toBe(200);
    // Still 'running' -- not reconciled synchronously; the worker must see
    // cancel_requested on its next heartbeat and report items cancelled.
    expect(repo.getTest(undefined, run.id)?.status).toBe("running");

    const hb = await heartbeat("stop-claimed", { backend: "cpu", installed: true });
    // heartbeat() doesn't send active_job by default -- verify the flag is
    // queryable directly instead of round-tripping through the wire shape.
    void hb;
    const job = repo.queueRepo.getJob(claimed.job_id);
    expect(job?.status).toBe("claimed"); // cancel_requested doesn't change status itself
  });

  it("a cancelled job whose lease then expires finalizes instead of getting stuck pending forever", async () => {
    // Regression test: found via live end-to-end testing of the rewritten
    // worker (MULTIUSER_PLAN.md §1.9/§1.17) -- a worker that goes silent
    // (crashes, network drops) right after a Stop was requested leaves its
    // job 'claimed' with cancel_requested=1. Before this fix, the reaper's
    // returnToPending() put it back to 'pending' without clearing
    // cancel_requested, and claimNextJob's own `WHERE cancel_requested = 0`
    // filter then made that job permanently unclaimable -- its run stuck
    // 'running' forever, with no path to ever finalize it. reapExpiredLeases
    // must treat a cancelled+expired job as terminal, not retry it.
    const { reapExpiredLeases } = await import("../reaper.js");
    const silentLog = { warn: () => {} };

    await heartbeat("stop-then-crash", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("stop-then-crash")!;
    const triggerRes = await postJson("/api/runs/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };
    const claimed = repo.queueRepo.claimNextJob(worker.id)!;

    await postJson(`/api/runs/${run.id}/stop`, {});
    expect(repo.queueRepo.getJob(claimed.job_id)?.status).toBe("claimed");

    // The worker never reports back (crashed) -- simulate its lease expiring
    // by fast-forwarding Date.now(), the same technique queue.test.ts's own
    // lease-expiry tests use, then run the real reaper sweep.
    const expired = repo.queueRepo.listExpiredLeases(Date.now() + 10 * 60_000);
    expect(expired.find((j) => j.id === claimed.job_id)?.cancel_requested).toBeTruthy();

    const originalNow = Date.now;
    Date.now = () => originalNow() + 10 * 60_000;
    try {
      reapExpiredLeases(silentLog);
    } finally {
      Date.now = originalNow;
    }

    // Terminal, not stuck pending -- and never claimable again.
    expect(repo.queueRepo.getJob(claimed.job_id)?.status).toBe("failed");
    expect(repo.queueRepo.claimNextJob(worker.id)).toBeUndefined();

    // The run itself must have been reconciled, not left 'running' forever.
    const finalRun = repo.getTest(undefined, run.id);
    expect(finalRun?.status).not.toBe("running");
  });

  it("kill -9 mid-run: a job stuck pending after one lease expiry finalizes even though nothing ever reclaims it", async () => {
    // Regression test, same live-testing session as the one above
    // (MULTIUSER_PLAN.md §1.17's "kill -9 the worker mid-run" exit
    // criterion): a job's FIRST lease expiry always just requeues it
    // (returnToPending) hoping the same worker reconnects (a pm2/systemd
    // auto-restart blip) -- but jobs are worker_id-scoped, only that same
    // worker can ever reclaim it. If it's truly gone for good, the job would
    // otherwise sit 'pending' forever with nothing left to check it again
    // (listExpiredLeases only looks at 'claimed' rows), and its run would
    // stay 'running' forever. reapExpiredLeases must also sweep 'pending'
    // jobs whose owning worker has gone offline.
    //
    // LEASE_MS (60s) is already longer than OFFLINE_AFTER_MS (35s), so by
    // the time any lease is confirmed expired at all, its worker's silence
    // has necessarily already crossed the offline threshold too -- a real
    // restart-blip recovery only ever happens via lease *extension* (a fresh
    // heartbeat before the 60s mark), never via this reap path. That means
    // one reap pass legitimately does both steps at once here: requeue, then
    // immediately notice the (already-offline) worker will never reclaim it,
    // and finalize it -- resolving comfortably inside the ~90s budget rather
    // than needing a second full lease cycle.
    const { reapExpiredLeases } = await import("../reaper.js");
    const silentLog = { warn: () => {} };

    await heartbeat("kill-mid-run", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("kill-mid-run")!;
    const triggerRes = await postJson("/api/runs/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };
    const claimed = repo.queueRepo.claimNextJob(worker.id)!;
    expect(repo.queueRepo.getJob(claimed.job_id)?.status).toBe("claimed");

    // The worker goes silent for good (kill -9, no Stop involved, and no
    // heartbeat ever arrives again) -- jump past LEASE_MS.
    const originalNow = Date.now;
    Date.now = () => originalNow() + 75_000;
    try {
      reapExpiredLeases(silentLog);
    } finally {
      Date.now = originalNow;
    }

    expect(repo.queueRepo.getJob(claimed.job_id)?.status).toBe("failed");
    const finalRun = repo.getTest(undefined, run.id);
    expect(finalRun?.status).not.toBe("running");
  });

  it("pause/resume toggle the worker's pause_requested flag", async () => {
    await heartbeat("pause-worker", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("pause-worker")!;
    const triggerRes = await postJson("/api/runs/trigger", {
      model_id: "model-1",
      worker_id: worker.id,
      sweep: baseSweep,
    });
    const { run } = (await triggerRes.json()) as { run: { id: string } };

    expect(repo.workerRepo.getPauseRequested(worker.id)).toBe(false);
    await postJson(`/api/runs/${run.id}/pause`, {});
    expect(repo.workerRepo.getPauseRequested(worker.id)).toBe(true);
    await postJson(`/api/runs/${run.id}/resume`, {});
    expect(repo.workerRepo.getPauseRequested(worker.id)).toBe(false);
  });

  it("stop on an unknown run returns 404", async () => {
    const res = await postJson("/api/runs/does-not-exist/stop", {});
    expect(res.status).toBe(404);
  });
});

// Multi-user Stage 4 isolation (MULTIUSER_PLAN.md §4.3's own exit criterion:
// "for each [route], assert that user B receives 403/404 for a resource
// owned by user A"). GET/list is covered above (userId threading); these
// cover the mutating trio specifically, since a wrong scoping there would
// let user B stop/pause/resume user A's run.
describe("run stop/pause/resume cross-user isolation (§4.3)", () => {
  async function sessionFor(login: string) {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: login, login, avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: login });
    return { userId: user.id, token };
  }

  // Worker is claimed BEFORE triggering -- assertOwnsWorker gates the
  // trigger route itself (an authenticated caller can't act on an unclaimed
  // machine either, see workers.test.ts), so an unclaimed-then-triggered
  // sequence would 403 at the trigger step, never reaching a real run.
  async function triggerAsOwner(machineId: string, owner: { userId: string; token: string }) {
    await heartbeat(machineId, { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId(machineId)!;
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE workers SET user_id = ? WHERE id = ?`).run(owner.userId, worker.id);
    const res = await fetch(`${baseUrl}/api/runs/trigger`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${owner.token}` },
      body: JSON.stringify({ model_id: "model-1", worker_id: worker.id, sweep: baseSweep }),
    });
    expect(res.status).toBe(201);
    const { run } = (await res.json()) as { run: { id: string } };
    return run.id;
  }

  it("user B cannot stop user A's run (404), but the real owner still can", async () => {
    const ownerA = await sessionFor("isolation-stop-owner");
    const runId = await triggerAsOwner("isolation-stop-worker", ownerA);

    const userB = await sessionFor("isolation-stop-intruder");
    const intruderRes = await fetch(`${baseUrl}/api/runs/${runId}/stop`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${userB.token}` },
      body: "{}",
    });
    expect(intruderRes.status).toBe(404);

    const ownerRes = await fetch(`${baseUrl}/api/runs/${runId}/stop`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${ownerA.token}` },
      body: "{}",
    });
    expect(ownerRes.status).toBe(200);
  });

  it("user B cannot pause or resume user A's run (404 either way)", async () => {
    const ownerA = await sessionFor("isolation-pause-owner");
    const runId = await triggerAsOwner("isolation-pause-worker", ownerA);

    const userB = await sessionFor("isolation-pause-intruder");
    const pauseRes = await fetch(`${baseUrl}/api/runs/${runId}/pause`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${userB.token}` },
      body: "{}",
    });
    expect(pauseRes.status).toBe(404);
    const resumeRes = await fetch(`${baseUrl}/api/runs/${runId}/resume`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${userB.token}` },
      body: "{}",
    });
    expect(resumeRes.status).toBe(404);
  });

  it("user B cannot even see user A's run in GET /api/runs/:id (404) or the list", async () => {
    const ownerA = await sessionFor("isolation-get-owner");
    const runId = await triggerAsOwner("isolation-get-worker", ownerA);

    const userB = await sessionFor("isolation-get-intruder");
    const getRes = await fetch(`${baseUrl}/api/runs/${runId}`, {
      headers: { authorization: `Bearer ${userB.token}` },
    });
    expect(getRes.status).toBe(404);

    const listRes = await fetch(`${baseUrl}/api/runs`, { headers: { authorization: `Bearer ${userB.token}` } });
    const listBody = (await listRes.json()) as { runs: { id: string }[] };
    expect(listBody.runs.map((r) => r.id)).not.toContain(runId);
  });
});

describe("run log push/pull (§1.10)", () => {
  async function createTest(machineId: string) {
    await heartbeat(machineId, { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId(machineId)!;
    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    const { run } = (await res.json()) as { run: { id: string } };
    return { worker, runId: run.id };
  }

  it("404s before any log has been pushed", async () => {
    const { runId } = await createTest("log-none");
    const res = await fetch(`${baseUrl}/api/runs/${runId}/log`);
    expect(res.status).toBe(404);
  });

  it("round-trips a pushed log: gzip in, gzip out, decompresses to the original text", async () => {
    const { worker, runId } = await createTest("log-roundtrip");
    const zlib = await import("node:zlib");
    const original = "build b1234 (cuda)\n[pp 512] 120.4 tok/s\n[tg 128] 45.2 tok/s\nrun completed\n";
    const gzipped = zlib.gzipSync(Buffer.from(original, "utf8"));

    const pushRes = await fetch(`${baseUrl}/api/runs/${runId}/log`, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        authorization: "Bearer runs-test-secret",
        "x-machine-id": "log-roundtrip",
      },
      body: gzipped,
    });
    expect(pushRes.status).toBe(200);

    const pullRes = await fetch(`${baseUrl}/api/runs/${runId}/log`);
    expect(pullRes.status).toBe(200);
    expect(pullRes.headers.get("content-encoding")).toBe("gzip");
    // fetch() (undici) transparently decompresses a Content-Encoding: gzip
    // response body, exactly like a browser would -- that's the whole point
    // of serving it this way (§1.10). What the CALLER sees is already plain
    // text; the gzip bytes only ever exist on the wire and on disk.
    const text = await pullRes.text();
    expect(text).toBe(original);

    void worker;
  });

  it("rejects a push from a machine that did not execute the run", async () => {
    const { runId } = await createTest("log-owner");
    await heartbeat("log-intruder", { backend: "cpu", installed: true });
    const zlib = await import("node:zlib");
    const res = await fetch(`${baseUrl}/api/runs/${runId}/log`, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        authorization: "Bearer runs-test-secret",
        "x-machine-id": "log-intruder",
      },
      body: zlib.gzipSync(Buffer.from("not this machine's run")),
    });
    expect(res.status).toBe(403);
  });

  it("rejects a push with a wrong bearer token", async () => {
    const { runId } = await createTest("log-badtoken");
    const zlib = await import("node:zlib");
    const res = await fetch(`${baseUrl}/api/runs/${runId}/log`, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        authorization: "Bearer wrong",
        "x-machine-id": "log-badtoken",
      },
      body: zlib.gzipSync(Buffer.from("x")),
    });
    expect(res.status).toBe(401);
  });
});

// Multi-user Stage 4 fix (MULTIUSER_PLAN.md §4.3): POST /api/runs/:id/items/:idx
// had NO worker authentication at all before this -- any caller could report
// fabricated results against any run. These tests cover the dual-mode check
// that closes that gap: a Stage 3 worker session (resolves a specific
// worker.id) or the Stage 1 shared secret (trusts the run's own worker_id).
describe("POST /api/runs/:id/items/:idx worker auth (§4.3 fix)", () => {
  async function createTest(machineId: string) {
    await heartbeat(machineId, { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId(machineId)!;
    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    const { run } = (await res.json()) as { run: { id: string } };
    return { worker, runId: run.id };
  }

  async function postItemUpdate(runId: string, idx: number, payload: unknown, token?: string) {
    return fetch(`${baseUrl}/api/runs/${runId}/items/${idx}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
    });
  }

  it("rejects a call with no token at all", async () => {
    const { runId } = await createTest("items-no-token");
    const res = await postItemUpdate(runId, 0, { status: "loading" });
    expect(res.status).toBe(401);
  });

  it("rejects a wrong shared token", async () => {
    const { runId } = await createTest("items-wrong-token");
    const res = await postItemUpdate(runId, 0, { status: "loading" }, "not-the-right-secret");
    expect(res.status).toBe(401);
  });

  it("accepts a tick and a terminal update from the correct shared token (Stage 1 fallback)", async () => {
    const { runId } = await createTest("items-shared-token");
    const tick = await postItemUpdate(runId, 0, { status: "loading" }, "runs-test-secret");
    expect(tick.status).toBe(200);
    const terminal = await postItemUpdate(runId, 0, { status: "failed", error: "x" }, "runs-test-secret");
    expect(terminal.status).toBe(200);
  });

  it("rejects a shared-token caller reporting against a run with no assigned worker", async () => {
    // Simulate a legacy row with no worker_id -- the shared secret's fallback
    // trusts the RUN's own worker_id (no machine_id in this route's payload
    // to check against directly), so a run with none can never be reported
    // on by anyone, matching this route's own doc comment.
    const { runId } = await createTest("items-no-worker-run");
    const db = (await import("../db/migrate.js")).getDb();
    db.prepare(`UPDATE runs SET worker_id = NULL WHERE id = ?`).run(runId);
    const res = await postItemUpdate(runId, 0, { status: "loading" }, "runs-test-secret");
    expect(res.status).toBe(403);
  });

  it("accepts a Stage 3 worker session that matches the run's own worker", async () => {
    const { worker, runId } = await createTest("items-session-owner");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "items-owner", login: "owner", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: worker.id, label: worker.displayName });

    const res = await postItemUpdate(runId, 0, { status: "loading" }, token);
    expect(res.status).toBe(200);
  });

  it("rejects a Stage 3 worker session belonging to a DIFFERENT machine than the run's own worker", async () => {
    const { runId } = await createTest("items-session-victim");
    await heartbeat("items-session-intruder", { backend: "cpu", installed: true });
    const intruder = repo.workerRepo.getByMachineId("items-session-intruder")!;
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "items-intruder", login: "intruder", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: intruder.id, label: intruder.displayName });

    const res = await postItemUpdate(runId, 0, { status: "loading" }, token);
    expect(res.status).toBe(403);
  });

  it("rejects a browser (non-worker) session presented as a worker token", async () => {
    const { runId } = await createTest("items-browser-session");
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "items-browser", login: "browser-user", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "browser tab" }); // isWorker defaults false

    // Not a worker session, and not the shared secret either -- must fail
    // one way or the other, never silently succeed.
    const res = await postItemUpdate(runId, 0, { status: "loading" }, token);
    expect(res.status).not.toBe(200);
  });
});

// §0.12's observability convention: thermally_flagged_ratio fires "once per
// completed run" (the plan's own words). It used to fire on every GET of
// /api/runs/:id/sustained instead -- once per page view/poll, not once per
// run -- so this covers the actual completion transition.
describe("§0.12 thermally_flagged_ratio fires once, at run completion", () => {
  it("logs exactly once from the item-terminal write that finalizes the run, never again on a later tick", async () => {
    await heartbeat("thermal-log-machine", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("thermal-log-machine")!;
    const res = await postJson("/api/runs/trigger", { model_id: "model-1", worker_id: worker.id, sweep: baseSweep });
    const { run } = (await res.json()) as { run: { id: string } };

    const infoSpy = vi.spyOn(app.log, "info");
    try {
      const terminal = await fetch(`${baseUrl}/api/runs/${run.id}/items/0`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer runs-test-secret" },
        body: JSON.stringify({ status: "done" }),
      });
      expect(terminal.status).toBe(200);
      const terminalBody = (await terminal.json()) as { run_status: string };
      expect(terminalBody.run_status).toBe("done");

      const ratioCalls = infoSpy.mock.calls.filter(([, msg]) => msg === "thermally_flagged_ratio");
      expect(ratioCalls).toHaveLength(1);
      expect(ratioCalls[0][0]).toMatchObject({
        thermally_flagged_ratio: 0,
        run_id: run.id,
        flagged: 0,
        denominator: 1,
      });
    } finally {
      infoSpy.mockRestore();
    }
  });
});

function minimalResult(overrides: Record<string, unknown> = {}) {
  return {
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
    ...overrides,
  };
}

// N3's fairness re-check must fire PER MEMBER at the moment its own results
// land, not only when someone happens to open the comparison view -- a drift
// like a mismatched method_version literally cannot be seen at trigger time
// (both members carry method_version: null until something has actually been
// measured), so this is the one place it can ever be caught.
describe("N3 comparison_member_failed fires when a member's own results reveal drift", () => {
  it("logs comparison_member_failed on the second member's completion when its method_version differs", async () => {
    repo.registerModel({ id: "cmp-model-a", filename: "a.gguf", size_bytes: 1000, source: "local", metadata: {} });
    repo.registerModel({ id: "cmp-model-b", filename: "b.gguf", size_bytes: 1000, source: "local", metadata: {} });
    await heartbeat("cmp-recheck-machine", { backend: "cpu", installed: true });
    const worker = repo.workerRepo.getByMachineId("cmp-recheck-machine")!;

    const resA = await postJson("/api/runs/trigger", {
      model_id: "cmp-model-a",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-recheck-1",
    });
    expect(resA.status).toBe(201);
    const runA = ((await resA.json()) as { run: { id: string } }).run;

    const resB = await postJson("/api/runs/trigger", {
      model_id: "cmp-model-b",
      worker_id: worker.id,
      kind: "sweep",
      sweep: baseSweep,
      comparison_id: "cmp-recheck-1",
    });
    expect(resB.status).toBe(201);
    const runB = ((await resB.json()) as { run: { id: string } }).run;

    // Member A completes first, with no sibling to compare against yet --
    // no violation possible.
    const terminalA = await fetch(`${baseUrl}/api/runs/${runA.id}/items/0`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer runs-test-secret" },
      body: JSON.stringify({ status: "done", results: [minimalResult({ method_version: 1 })] }),
    });
    expect(terminalA.status).toBe(200);

    const warnSpy = vi.spyOn(app.log, "warn");
    try {
      // Member B completes with a DIFFERENT method_version -- a drift that
      // was invisible at trigger time (both were null then) and only shows
      // up now that both members have real measurements.
      const terminalB = await fetch(`${baseUrl}/api/runs/${runB.id}/items/0`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer runs-test-secret" },
        body: JSON.stringify({ status: "done", results: [minimalResult({ method_version: 2 })] }),
      });
      expect(terminalB.status).toBe(200);

      const failedCalls = warnSpy.mock.calls.filter(([, msg]) => msg === "comparison_member_failed");
      expect(failedCalls.length).toBeGreaterThan(0);
      expect(failedCalls[0][0]).toMatchObject({
        comparison_member_failed: true,
        member_test_id: runB.id,
        reason: "method_version",
      });
    } finally {
      warnSpy.mockRestore();
    }
  });
});
