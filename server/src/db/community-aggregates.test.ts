import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";
import type { getDb as GetDbType } from "./migrate.js";

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- repo.communityRepo's own
// isolated test file. getDb() is a module-level singleton keyed on
// process.env.DB_PATH at first call, so this needs its own throwaway DB, set
// before repo.js is ever imported.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-community-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let getDb: typeof GetDbType;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
  ({ getDb } = await import("./migrate.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

// Raw SQL throughout, same posture as legacy-history.test.ts -- this test
// needs precise control over which user/worker/run/result combination lands
// in which group cell, which going through the normal createRun/
// recordRunItemTerminal pipeline (designed around one worker executing one
// sweep at a time) would make far more verbose for no benefit.
function insertUser(id: string, shareBenchmarks: boolean): void {
  getDb()
    .prepare(`INSERT INTO users (id, display_name, share_benchmarks, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`)
    .run(id, `display-${id}`, shareBenchmarks ? 1 : 0, Date.now(), Date.now());
}

function insertWorker(id: string, platform: string, gpuModel: string): void {
  getDb()
    .prepare(
      `INSERT INTO workers (id, machine_id, display_name, hardware_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`
    )
    .run(id, id, id, JSON.stringify({ platform, gpu: [{ model: gpuModel }] }), Date.now(), Date.now());
}

function insertRun(
  id: string,
  userId: string,
  workerId: string,
  modelId: string,
  backend: string,
  build: string
): void {
  getDb()
    .prepare(
      `INSERT INTO runs (id, worker_id, user_id, model_id, llama_cpp_backend, llama_cpp_build, status, started_at)
       VALUES (?, ?, ?, ?, ?, ?, 'done', ?)`
    )
    .run(id, workerId, userId, modelId, backend, build, Date.now());
}

function insertResult(
  runId: string,
  userId: string,
  modelId: string,
  testType: string,
  avgTps: number,
  ramPeakMib: number,
  vramPeakMib: number
): void {
  getDb()
    .prepare(
      `INSERT INTO results (id, run_id, user_id, model_id, test_type, avg_tps, ram_peak_mib, vram_peak_mib, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(`result-${runId}-${testType}`, runId, userId, modelId, testType, avgTps, ramPeakMib, vramPeakMib, Date.now());
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

describe("repo.communityRepo (§5.4 k-anonymity)", () => {
  const modelId = "community-model-1";

  beforeAll(() => {
    repo.registerModel({ id: modelId, filename: "community-model.gguf", size_bytes: 1, source: "local", metadata: {} });

    // Group X: RTX 4090 / cuda / linux / model-1 -- 6 opted-in contributors
    // (u1..u6), each on a DIFFERENT llama_cpp_build string, proving build is
    // genuinely dropped from the group key rather than accidentally
    // fragmenting the group into six singletons. Both a tg and a pp result
    // per contributor, at very different magnitudes, proving test_type is
    // NOT averaged together.
    for (let i = 1; i <= 6; i++) {
      const userId = `u${i}`;
      insertUser(userId, true);
      insertWorker(`w${i}`, "linux", "RTX 4090");
      insertRun(`run${i}`, userId, `w${i}`, modelId, "cuda", `b${1000 + i}`);
      insertResult(`run${i}`, userId, modelId, "tg", 90 + i, 8000, 20000);
      insertResult(`run${i}`, userId, modelId, "pp", 2000 + i, 8000, 20000);
    }

    // An 8th contributor sharing group X's exact dimensions but opted OUT --
    // must never be counted, even though its row otherwise matches perfectly.
    insertUser("u8-optout", false);
    insertWorker("w8", "linux", "RTX 4090");
    insertRun("run8", "u8-optout", "w8", modelId, "cuda", "b9999");
    insertResult("run8", "u8-optout", modelId, "tg", 999, 8000, 20000);

    // Group Y: a single opted-in contributor on a rare GPU -- the "synthetic
    // single-contributor group" that must be completely absent from every
    // query below, at k=4 short of the k=5 threshold.
    insertUser("u7-lonely", true);
    insertWorker("w7", "linux", "RTX 3060");
    insertRun("run7", "u7-lonely", "w7", modelId, "cuda", "b2000");
    insertResult("run7", "u7-lonely", modelId, "tg", 40, 6000, 10000);
  });

  it("suppresses a group below k=5 entirely, even called by an unrelated caller", () => {
    const rows = repo.communityRepo.listAggregates("some-other-caller");
    const rtx3060 = rows.find((r) => r.gpuModel === "RTX 3060");
    expect(rtx3060).toBeUndefined();
  });

  it("excludes the caller's own contribution from both the group and its average, and drops build from the key", () => {
    // Called as u1: 6 opted-in contributors minus u1 = 5, exactly clearing
    // k=5. If `build` were still part of the group key, this group would
    // fragment into six size-1 cells and vanish entirely from the output.
    const rows = repo.communityRepo.listAggregates("u1");
    const tgRow = rows.find((r) => r.gpuModel === "RTX 4090" && r.testType === "tg");
    expect(tgRow).toBeDefined();
    expect(tgRow!.contributorCount).toBe(5); // u2..u6 -- u1 excluded, u8-optout excluded
    expect(tgRow!.runCount).toBe(5);
    // avg of 92,93,94,95,96 (u2..u6's tg avg_tps) = 94
    expect(tgRow!.avgTps).toBeCloseTo(94, 1);
  });

  it("keeps pp and tg as separate rows instead of averaging them together", () => {
    const rows = repo.communityRepo.listAggregates("u1");
    const ppRow = rows.find((r) => r.gpuModel === "RTX 4090" && r.testType === "pp");
    const tgRow = rows.find((r) => r.gpuModel === "RTX 4090" && r.testType === "tg");
    expect(ppRow).toBeDefined();
    expect(tgRow).toBeDefined();
    expect(ppRow!.avgTps).toBeGreaterThan(1000);
    expect(tgRow!.avgTps).toBeLessThan(200);
  });

  it("respects share_benchmarks consent even for a caller with no stake in the group", () => {
    const rows = repo.communityRepo.listAggregates("some-other-caller");
    const tgRow = rows.find((r) => r.gpuModel === "RTX 4090" && r.testType === "tg");
    expect(tgRow).toBeDefined();
    // All 6 opted-in contributors count (nobody excluded this time) --
    // u8-optout must still not be among them.
    expect(tgRow!.contributorCount).toBe(6);
  });

  it("every returned row clears k=5 and carries no UUID- or username-shaped value", () => {
    const rows = repo.communityRepo.listAggregates("some-other-caller");
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.contributorCount).toBeGreaterThanOrEqual(5);
      // CommunityAggregateRow's own type has no userId/username field at
      // all -- this walks every string value actually present to also catch
      // a UUID or a synthetic user id leaking through an unexpected field.
      const values = [row.modelId, row.modelFilename, row.backend, row.testType, row.platform, row.gpuModel];
      for (const v of values) {
        if (v === null) continue;
        expect(UUID_RE.test(v)).toBe(false);
        expect(v).not.toMatch(/^u\d+$/);
        expect(v).not.toContain("u8-optout");
        expect(v).not.toContain("u7-lonely");
      }
    }
  });

  it("listFacets suppresses a facet value below k=5 and surfaces one that clears it", () => {
    const facets = repo.communityRepo.listFacets("u1");
    const gpu4090 = facets.gpuModels.find((g) => g.value === "RTX 4090");
    expect(gpu4090).toBeDefined();
    expect(gpu4090!.contributorCount).toBe(5); // u1 excluded, u8-optout excluded by consent

    const gpu3060 = facets.gpuModels.find((g) => g.value === "RTX 3060");
    expect(gpu3060).toBeUndefined(); // only 1 contributor, well under k=5

    // model facet counts every contributor on that model regardless of GPU
    // -- u2..u6 (RTX 4090) plus u7-lonely (RTX 3060) = 6, u1 excluded as
    // caller, u8-optout excluded by consent.
    const model = facets.models.find((m) => m.modelId === modelId);
    expect(model).toBeDefined();
    expect(model!.contributorCount).toBe(6);
  });
});
