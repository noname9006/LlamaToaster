import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-catalog-repair-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof import("./repo.js")["repo"];
let getDb: typeof import("./migrate.js")["getDb"];
let findN3Repairs: typeof import("./catalogRepair.js")["findN3Repairs"];
let applyN3Repairs: typeof import("./catalogRepair.js")["applyN3Repairs"];

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
  ({ getDb } = await import("./migrate.js"));
  ({ findN3Repairs, applyN3Repairs } = await import("./catalogRepair.js"));
});

afterAll(() => {
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

// Rows in the shape the pre-fix POST /api/models could leave behind, written
// with plain SQL since the fixed code paths can no longer produce them.
function insertRow(row: {
  id: string;
  created_by: string | null;
  source?: string;
  filename?: string;
  hf_repo?: string | null;
  hf_file?: string | null;
  locked?: boolean;
  metadata?: object;
}) {
  getDb()
    .prepare(
      `INSERT INTO models (id, filename, size_bytes, source, hf_repo, hf_file, metadata, created_by, created_at, hf_identity_locked)
       VALUES (?, ?, 0, ?, ?, ?, ?, ?, 0, ?)`
    )
    .run(
      row.id,
      row.filename ?? "f.gguf",
      row.source ?? "huggingface",
      row.hf_repo ?? null,
      row.hf_file ?? null,
      JSON.stringify(row.metadata ?? {}),
      row.created_by,
      row.locked ? 1 : 0
    );
}

function index(sha: string, repoId: string, filename: string) {
  getDb()
    .prepare(`INSERT INTO hf_gguf_index (sha256, repo_id, filename, last_seen) VALUES (?, ?, ?, 0)`)
    .run(sha, repoId, filename);
}

describe("N3 catalog repair", () => {
  it("releases squats, keeps legitimate owned rows, and lists unverifiable locks", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "squatter", login: "squatter", avatarUrl: null });

    const squatFake = "1a".repeat(32);
    insertRow({ id: squatFake, created_by: user.id, hf_repo: "evil/fake", hf_file: "x.gguf", locked: true, metadata: { gguf_header_adopted: true } });

    const squatLive = "2b".repeat(32);
    index(squatLive, "good/repo", "real.gguf");
    insertRow({ id: squatLive, created_by: user.id, hf_repo: "good/repo", hf_file: "real.gguf" });

    // The route's own derived id: a 64-hex id, but not a file hash.
    const derived = repo.insertModelIfAbsent({ source: "huggingface", hf_repo: "u/r", hf_file: "m.gguf", created_by: user.id })!;
    insertRow({ id: "my-model", created_by: user.id, source: "local" });

    const overwritten = "3c".repeat(32);
    insertRow({ id: overwritten, created_by: null, hf_repo: "evil/fake", hf_file: "y.gguf", locked: true });
    const realLocked = "4d".repeat(32);
    index(realLocked, "good/repo", "z.gguf");
    insertRow({ id: realLocked, created_by: null, hf_repo: "good/repo", hf_file: "z.gguf", locked: true });

    const { candidates, review } = findN3Repairs(getDb());
    expect(candidates.map((c) => c.id).sort()).toEqual([squatFake, squatLive].sort());
    expect(review.map((r) => r.id)).toEqual([overwritten]);

    applyN3Repairs(getDb(), candidates);

    expect(repo.getModel(squatFake)).toMatchObject({ source: "local", hf_repo: undefined, hf_identity_locked: false, metadata: {} });
    expect(repo.getModelCreatedBy(squatFake)).toBeNull();
    expect(repo.getModel(squatLive)).toMatchObject({ hf_repo: "good/repo", hf_file: "real.gguf" });
    expect(repo.getModelCreatedBy(squatLive)).toBeNull();
    expect(repo.getModelCreatedBy(derived.id)).toBe(user.id);
    expect(repo.getModelCreatedBy("my-model")).toBe(user.id);
    expect(repo.getModel(overwritten)!.hf_repo).toBe("evil/fake");

    // Idempotent: nothing left to release on a second pass.
    expect(findN3Repairs(getDb()).candidates).toEqual([]);
  });
});
