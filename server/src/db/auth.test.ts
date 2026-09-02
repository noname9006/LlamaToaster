import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { repo as RepoType } from "./repo.js";
import type { getDb as GetDbType } from "./migrate.js";
import type { TestConfig } from "../../../shared/types.js";

// getDb() (migrate.ts) is a module-level singleton keyed on process.env.DB_PATH
// at first call -- set it before importing repo.js so this test file gets its
// own throwaway DB, isolated from any other test file or the real dev DB.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-auth-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let repo: typeof RepoType;
let hashToken: (token: string) => string;
let getDb: typeof GetDbType;

beforeAll(async () => {
  ({ repo } = await import("./repo.js"));
  ({ hashToken } = await import("../session.js"));
  ({ getDb } = await import("./migrate.js"));
});

afterAll(() => {
  // better-sqlite3 keeps the DB (and its WAL/SHM sidecars) open for the
  // process lifetime -- best-effort cleanup only, same posture as the other
  // DB-layer test files in this directory.
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

describe("userRepo.upsertByIdentity", () => {
  it("creates a new user + identity on first login, and is idempotent on repeat", () => {
    const first = repo.userRepo.upsertByIdentity("github", {
      providerUserId: "gh-1",
      login: "octocat",
      avatarUrl: "https://example.invalid/a.png",
    });
    expect(first.displayName).toBe("octocat");
    expect(first.avatarUrl).toBe("https://example.invalid/a.png");
    expect(first.isSuperadmin).toBe(false);

    const second = repo.userRepo.upsertByIdentity("github", {
      providerUserId: "gh-1",
      login: "octocat",
      avatarUrl: "https://example.invalid/a.png",
    });
    expect(second.id).toBe(first.id); // same account, not a duplicate
  });

  it("refreshes provider_login on a later login without creating a new account", () => {
    const first = repo.userRepo.upsertByIdentity("github", { providerUserId: "gh-2", login: "old-name", avatarUrl: null });
    const renamed = repo.userRepo.upsertByIdentity("github", { providerUserId: "gh-2", login: "new-name", avatarUrl: null });
    expect(renamed.id).toBe(first.id);
    const identities = repo.userRepo.getIdentities(first.id);
    expect(identities).toHaveLength(1);
    expect(identities[0]!.providerLogin).toBe("new-name");
  });

  it("two different providers with the same provider_user_id string are different accounts", () => {
    // Confirms the lookup is keyed on (provider, provider_user_id) together,
    // not provider_user_id alone -- GitHub's numeric ids and Google's `sub`
    // claims live in entirely different id spaces and could collide as bare
    // strings.
    const gh = repo.userRepo.upsertByIdentity("github", { providerUserId: "42", login: "gh-42", avatarUrl: null });
    const gg = repo.userRepo.upsertByIdentity("google", { providerUserId: "42", login: "google-42", avatarUrl: null });
    expect(gh.id).not.toBe(gg.id);
  });
});

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- Settings' own consent
// toggle, default-on (schema.sql's own DEFAULT 1).
describe("userRepo.setShareBenchmarks", () => {
  it("defaults to true and can be flipped in either direction", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-1", login: "share-user", avatarUrl: null });
    expect(user.shareBenchmarks).toBe(true);

    const off = repo.userRepo.setShareBenchmarks(user.id, false);
    expect(off.shareBenchmarks).toBe(false);
    expect(repo.userRepo.getUser(user.id)!.shareBenchmarks).toBe(false);

    const backOn = repo.userRepo.setShareBenchmarks(user.id, true);
    expect(backOn.shareBenchmarks).toBe(true);
  });
});

// Security checklist's own "Account deletion ships in v1" item.
describe("userRepo.deleteAccount", () => {
  it("cascades every owned row, reassigns (not deletes) a created model, and never touches another user's data", () => {
    const db = getDb();

    const owner = repo.userRepo.upsertByIdentity("github", { providerUserId: "delete-owner", login: "owner", avatarUrl: null });
    repo.userRepo.linkIdentity(owner.id, "google", { providerUserId: "delete-owner-google", login: "owner-g", avatarUrl: null });
    const session = repo.sessionRepo.create(owner.id, { label: "to-be-deleted" });

    const worker = repo.workerRepo.getOrCreateByMachineId("delete-owner-machine", "delete-owner-machine");
    repo.workerRepo.approve(worker.id, owner.id);

    const model = repo.registerModel({
      id: "delete-owner-model",
      filename: "owner-model.gguf",
      size_bytes: 1,
      source: "local",
      metadata: {},
      created_by: owner.id,
    });
    expect(repo.getModelCreatedBy(model.id)).toBe(owner.id);

    repo.createTest(owner.id, {
      id: "delete-owner-run",
      worker_name: "delete-owner-machine",
      worker_id: worker.id,
      llama_cpp_build: "b1",
      llama_cpp_backend: "cpu",
      model_id: model.id,
      config: { model_id: model.id } as TestConfig,
      status: "done",
      started_at: Date.now(),
    });
    db.prepare(`INSERT INTO results (id, run_id, user_id, model_id, test_type, avg_tps, created_at) VALUES (?, ?, ?, ?, 'tg', 10, ?)`).run(
      "delete-owner-result",
      "delete-owner-run",
      owner.id,
      model.id,
      Date.now()
    );
    db.prepare(`INSERT INTO run_items (id, run_id, user_id, idx, status) VALUES (?, ?, ?, 0, 'done')`).run(
      "delete-owner-item",
      "delete-owner-run",
      owner.id
    );

    // A completely unrelated user + their own resources -- must survive
    // untouched, proving this isn't a blunt "clear everything" operation.
    const other = repo.userRepo.upsertByIdentity("github", { providerUserId: "delete-bystander", login: "bystander", avatarUrl: null });
    const otherSession = repo.sessionRepo.create(other.id, { label: "bystander-session" });
    const otherWorker = repo.workerRepo.getOrCreateByMachineId("bystander-machine", "bystander-machine");
    repo.workerRepo.approve(otherWorker.id, other.id);

    // sessionRepo.create returns only the raw token (never the row id, see
    // its own doc comment) -- look sessions back up by token_hash, the same
    // way every real caller (resolveAuthUser) does.
    const sessionTokenHash = hashToken(session.token);
    const otherSessionTokenHash = hashToken(otherSession.token);

    repo.userRepo.deleteAccount(owner.id);

    expect(repo.userRepo.getUser(owner.id)).toBeUndefined();
    expect(db.prepare(`SELECT id FROM sessions WHERE token_hash = ?`).get(sessionTokenHash)).toBeUndefined();
    expect(db.prepare(`SELECT * FROM identities WHERE user_id = ?`).all(owner.id)).toEqual([]);
    expect(db.prepare(`SELECT id FROM workers WHERE id = ?`).get(worker.id)).toBeUndefined();
    expect(db.prepare(`SELECT id FROM runs WHERE id = ?`).get("delete-owner-run")).toBeUndefined();
    expect(db.prepare(`SELECT id FROM results WHERE id = ?`).get("delete-owner-result")).toBeUndefined();
    expect(db.prepare(`SELECT id FROM run_items WHERE id = ?`).get("delete-owner-item")).toBeUndefined();

    // The model itself survives (it's a global, shared catalog row) --
    // only its creator attribution is cleared, not the row.
    expect(repo.getModel(model.id)).toBeDefined();
    expect(repo.getModelCreatedBy(model.id)).toBeNull();

    // The bystander is completely unaffected.
    expect(repo.userRepo.getUser(other.id)).toBeDefined();
    expect(db.prepare(`SELECT id FROM sessions WHERE token_hash = ?`).get(otherSessionTokenHash)).toBeDefined();
    expect(db.prepare(`SELECT id FROM workers WHERE id = ?`).get(otherWorker.id)).toBeDefined();
  });
});

describe("userRepo.linkIdentity / findByIdentity", () => {
  it("attaches a new provider to an existing account without creating a second one", () => {
    const primary = repo.userRepo.upsertByIdentity("github", { providerUserId: "link-1", login: "primary", avatarUrl: null });
    const linked = repo.userRepo.linkIdentity(primary.id, "google", {
      providerUserId: "google-link-1",
      login: "primary-on-google",
      avatarUrl: null,
    });
    expect(linked.id).toBe(primary.id);
    const identities = repo.userRepo.getIdentities(primary.id);
    expect(identities.map((i) => i.provider).sort()).toEqual(["github", "google"]);
  });

  it("findByIdentity is what a caller uses to detect an identity already belongs to someone else, before linking", () => {
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "owner-a", login: "a", avatarUrl: null });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "owner-b", login: "b", avatarUrl: null });
    repo.userRepo.linkIdentity(userA.id, "google", { providerUserId: "shared-google-id", login: "a-google", avatarUrl: null });

    const owner = repo.userRepo.findByIdentity("google", "shared-google-id");
    expect(owner?.id).toBe(userA.id);
    expect(owner?.id).not.toBe(userB.id); // this is the check a route uses to reject linking it to userB
  });

  it("findByIdentity returns undefined for an identity nobody has linked", () => {
    expect(repo.userRepo.findByIdentity("github", "never-seen")).toBeUndefined();
  });
});

describe("sessionRepo", () => {
  it("create() returns the raw token/refresh once -- only their hashes are persisted", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-1", login: "sess-user", avatarUrl: null });
    const { token, refresh, expiresAt } = repo.sessionRepo.create(user.id, { label: "test browser" });
    expect(token).toHaveLength(43); // base64url(32 bytes)
    expect(refresh).toBeNull(); // non-worker session
    expect(expiresAt).toBeGreaterThan(Date.now());

    const found = repo.sessionRepo.getByTokenHash(hashToken(token));
    expect(found?.userId).toBe(user.id);
    expect(found?.isWorker).toBe(false);

    // The raw token is never queryable directly -- only its hash is stored.
    expect(repo.sessionRepo.getByTokenHash(token)).toBeUndefined();
  });

  it("a worker session gets a refresh token too", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-2", login: "sess-worker-owner", avatarUrl: null });
    const { refresh } = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "worker-abc", label: "GPU box" });
    expect(refresh).not.toBeNull();
    const found = repo.sessionRepo.getByRefreshHash(hashToken(refresh!));
    expect(found?.isWorker).toBe(true);
    expect(found?.workerId).toBe("worker-abc");
  });

  it("rotate() issues new token+refresh, keeps the session id stable, and remembers the old refresh for replay detection", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-3", login: "sess-rotate", avatarUrl: null });
    const created = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "worker-rot" });
    const before = repo.sessionRepo.getByRefreshHash(hashToken(created.refresh!))!;

    const rotated = repo.sessionRepo.rotate(before.id);
    expect(rotated.token).not.toBe(created.token);
    expect(rotated.refresh).not.toBe(created.refresh);

    // Same session row (sessions.id stable across rotation -- MULTIUSER_PLAN.md
    // §2.5's own gotcha about not rotating the primary key).
    const afterRotate = repo.sessionRepo.getByRefreshHash(hashToken(rotated.refresh));
    expect(afterRotate?.id).toBe(before.id);

    // Old token no longer resolves.
    expect(repo.sessionRepo.getByTokenHash(hashToken(created.token))).toBeUndefined();

    // Old refresh is now the PREVIOUS refresh -- presenting it again should
    // be detectable as a replay.
    const replay = repo.sessionRepo.getByPrevRefreshHash(hashToken(created.refresh!));
    expect(replay?.id).toBe(before.id);
  });

  it("revokeById removes the session; revokeAllExcept keeps only the named one", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-4", login: "sess-revoke", avatarUrl: null });
    const a = repo.sessionRepo.create(user.id, { label: "a" });
    const b = repo.sessionRepo.create(user.id, { label: "b" });
    const c = repo.sessionRepo.create(user.id, { label: "c" });
    const aId = repo.sessionRepo.getByTokenHash(hashToken(a.token))!.id;
    const bId = repo.sessionRepo.getByTokenHash(hashToken(b.token))!.id;

    repo.sessionRepo.revokeById(aId);
    expect(repo.sessionRepo.getByTokenHash(hashToken(a.token))).toBeUndefined();

    repo.sessionRepo.revokeAllExcept(user.id, bId);
    const remaining = repo.sessionRepo.listForUser(user.id);
    expect(remaining.map((s) => s.id)).toEqual([bId]);
    void c;
  });

  it("listForUser lists newest first and never includes another user's sessions", () => {
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-5a", login: "list-a", avatarUrl: null });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-5b", login: "list-b", avatarUrl: null });
    repo.sessionRepo.create(userA.id, { label: "a1" });
    repo.sessionRepo.create(userB.id, { label: "b1" });
    const sessionsA = repo.sessionRepo.listForUser(userA.id);
    expect(sessionsA.every((s) => s.userId === userA.id)).toBe(true);
    expect(sessionsA.some((s) => s.label === "b1")).toBe(false);
  });

  it("touch() is throttled to one write per hour per session", () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "sess-6", login: "sess-touch", avatarUrl: null });
    const created = repo.sessionRepo.create(user.id, { label: "touch-test" });
    // create() itself stamps last_seen_at = now (same INSERT as created_at) --
    // touch() is only about extending an ALREADY-live session, not first-set.
    const session = repo.sessionRepo.getByTokenHash(hashToken(created.token))!;
    expect(session.lastSeenAt).not.toBeNull();

    // Touching again immediately (well within the 1h throttle window) must
    // not move lastSeenAt or expiresAt.
    repo.sessionRepo.touch(session);
    const afterTouch = repo.sessionRepo.getByTokenHash(hashToken(created.token))!;
    expect(afterTouch.lastSeenAt).toBe(session.lastSeenAt);
    expect(afterTouch.expiresAt).toBe(session.expiresAt);

    // A session last seen over an hour ago DOES get extended.
    const stale = { ...afterTouch, lastSeenAt: Date.now() - 2 * 3600_000 };
    repo.sessionRepo.touch(stale);
    const afterStaleTouch = repo.sessionRepo.getByTokenHash(hashToken(created.token))!;
    expect(afterStaleTouch.lastSeenAt).toBeGreaterThan(stale.lastSeenAt);
  });
});
