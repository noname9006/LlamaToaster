import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-sessions-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { sessionRoutes } = await import("./sessions.js");
  const { authMiddleware } = await import("../auth-middleware.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(fastifyCookie);
  app.addHook("preHandler", authMiddleware); // sessionRoutes needs req.user/req.session populated
  await app.register(sessionRoutes);
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

function authed(token: string): Record<string, string> {
  return { authorization: `Bearer ${token}` };
}

describe("POST /api/auth/refresh", () => {
  it("rejects a request with no bearer token", async () => {
    const res = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST" });
    expect(res.status).toBe(401);
  });

  it("rejects a non-worker session's access token (it has no refresh_hash to match)", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "refresh-1", login: "browser-user", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "browser" });
    const res = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(token) });
    expect(res.status).toBe(401);
  });

  it("rotates a worker session's token+refresh and keeps the session usable", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "refresh-2", login: "worker-owner", avatarUrl: null });
    const created = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "w-1", label: "GPU box" });

    const res = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(created.refresh!) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { session_token: string; refresh_token: string; expires_at: number };
    expect(body.session_token).not.toBe(created.token);
    expect(body.refresh_token).not.toBe(created.refresh);

    // The new session_token actually works for an authenticated call.
    const sessionsRes = await fetch(`${baseUrl}/api/sessions`, { headers: authed(body.session_token) });
    expect(sessionsRes.status).toBe(200);
  });

  it("detects refresh-token replay (presenting the OLD refresh after rotation) and revokes the session", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "refresh-3", login: "worker-replay", avatarUrl: null });
    const created = repo.sessionRepo.create(user.id, { isWorker: true, workerId: "w-2" });

    const rotateRes = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(created.refresh!) });
    const rotated = (await rotateRes.json()) as { session_token: string };
    expect(rotateRes.status).toBe(200);

    // Present the ORIGINAL (now-stale) refresh token again -- a captured/replayed token.
    const replayRes = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed(created.refresh!) });
    expect(replayRes.status).toBe(401);

    // The whole session must now be dead, including the token issued by the
    // legitimate rotation moments before -- replay detection kills the
    // session outright rather than trusting either party further.
    const sessionsRes = await fetch(`${baseUrl}/api/sessions`, { headers: authed(rotated.session_token) });
    expect(sessionsRes.status).toBe(401);
  });

  it("rejects an unknown/garbage refresh token", async () => {
    const res = await fetch(`${baseUrl}/api/auth/refresh`, { method: "POST", headers: authed("not-a-real-token") });
    expect(res.status).toBe(401);
  });
});

describe("GET /api/sessions", () => {
  it("requires authentication", async () => {
    const res = await fetch(`${baseUrl}/api/sessions`);
    expect(res.status).toBe(401);
  });

  it("lists only the caller's own sessions, flags the current one", async () => {
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "list-a", login: "list-a-user", avatarUrl: null });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "list-b", login: "list-b-user", avatarUrl: null });
    const sessionA1 = repo.sessionRepo.create(userA.id, { label: "a1" });
    repo.sessionRepo.create(userA.id, { label: "a2" });
    repo.sessionRepo.create(userB.id, { label: "b1" });

    const res = await fetch(`${baseUrl}/api/sessions`, { headers: authed(sessionA1.token) });
    const body = (await res.json()) as { sessions: { label: string | null; current: boolean }[] };
    expect(body.sessions.map((s) => s.label).sort()).toEqual(["a1", "a2"]);
    expect(body.sessions.find((s) => s.label === "a1")?.current).toBe(true);
    expect(body.sessions.find((s) => s.label === "a2")?.current).toBe(false);
  });
});

describe("DELETE /api/sessions/:id", () => {
  it("revokes the caller's own session", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "del-1", login: "del-user", avatarUrl: null });
    const primary = repo.sessionRepo.create(user.id, { label: "primary" });
    const other = repo.sessionRepo.create(user.id, { label: "other" });
    const otherId = (
      (await (await fetch(`${baseUrl}/api/sessions`, { headers: authed(primary.token) })).json()) as {
        sessions: { id: string; label: string | null }[];
      }
    ).sessions.find((s) => s.label === "other")!.id;

    const res = await fetch(`${baseUrl}/api/sessions/${otherId}`, { method: "DELETE", headers: authed(primary.token) });
    expect(res.status).toBe(200);

    const afterRes = await fetch(`${baseUrl}/api/sessions`, { headers: authed(primary.token) });
    const after = (await afterRes.json()) as { sessions: { label: string | null }[] };
    expect(after.sessions.map((s) => s.label)).toEqual(["primary"]);
    void other;
  });

  it("refuses to revoke another user's session (404, not a cross-tenant 200)", async () => {
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "del-a", login: "del-a-user", avatarUrl: null });
    const userB = repo.userRepo.upsertByIdentity("github", { providerUserId: "del-b", login: "del-b-user", avatarUrl: null });
    const sessionA = repo.sessionRepo.create(userA.id, { label: "a" });
    const sessionB = repo.sessionRepo.create(userB.id, { label: "b" });
    const sessionBId = repo.sessionRepo.listForUser(userB.id)[0]!.id;

    const res = await fetch(`${baseUrl}/api/sessions/${sessionBId}`, { method: "DELETE", headers: authed(sessionA.token) });
    expect(res.status).toBe(404);

    // B's session must still be alive -- the delete must not have gone through.
    const stillThere = await fetch(`${baseUrl}/api/sessions`, { headers: authed(sessionB.token) });
    expect(stillThere.status).toBe(200);
  });
});

describe("POST /api/sessions/revoke-all", () => {
  it("revokes every OTHER session but keeps the caller's own alive", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "revoke-all-1", login: "revoke-all-user", avatarUrl: null });
    const current = repo.sessionRepo.create(user.id, { label: "current" });
    repo.sessionRepo.create(user.id, { label: "elsewhere-1" });
    repo.sessionRepo.create(user.id, { label: "elsewhere-2" });

    const res = await fetch(`${baseUrl}/api/sessions/revoke-all`, { method: "POST", headers: authed(current.token) });
    expect(res.status).toBe(200);

    const afterRes = await fetch(`${baseUrl}/api/sessions`, { headers: authed(current.token) });
    const after = (await afterRes.json()) as { sessions: { label: string | null }[] };
    expect(after.sessions.map((s) => s.label)).toEqual(["current"]);
  });
});

describe("GET /api/auth/identities", () => {
  it("requires authentication", async () => {
    const res = await fetch(`${baseUrl}/api/auth/identities`);
    expect(res.status).toBe(401);
  });

  it("lists only the caller's own linked identities, never another user's", async () => {
    const userA = repo.userRepo.upsertByIdentity("github", { providerUserId: "ident-a", login: "ident-a-login", avatarUrl: null });
    repo.userRepo.linkIdentity(userA.id, "google", { providerUserId: "ident-a-google", login: "ident-a-google-login", avatarUrl: null });
    repo.userRepo.upsertByIdentity("github", { providerUserId: "ident-b", login: "ident-b-login", avatarUrl: null });
    const { token } = repo.sessionRepo.create(userA.id, { label: "ident-test" });

    const res = await fetch(`${baseUrl}/api/auth/identities`, { headers: authed(token) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { identities: { provider: string; providerLogin: string | null }[] };
    expect(body.identities.map((i) => i.provider).sort()).toEqual(["github", "google"]);
    expect(body.identities.some((i) => i.providerLogin === "ident-b-login")).toBe(false);
  });
});

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- Settings' own consent toggle.
describe("POST /api/auth/share-benchmarks", () => {
  it("requires authentication", async () => {
    const res = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ enabled: false }),
    });
    expect(res.status).toBe(401);
  });

  it("flips the caller's own flag and persists it", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-1", login: "share-route-user", avatarUrl: null });
    expect(repo.userRepo.getUser(user.id)!.shareBenchmarks).toBe(true); // default-on

    const { token } = repo.sessionRepo.create(user.id, { label: "share" });
    const res = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: false }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { shareBenchmarks: boolean };
    expect(body.shareBenchmarks).toBe(false);
    expect(repo.userRepo.getUser(user.id)!.shareBenchmarks).toBe(false);
  });

  it("400s a non-boolean body instead of silently coercing it", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-2", login: "share-route-user-2", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "share-bad" });
    const res = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: "nope" }),
    });
    expect(res.status).toBe(400);
  });

  it("403s turning it ON while the operator has communitySharingAllowed off (the default) -- turning it OFF is always allowed", async () => {
    expect(repo.appSettingsRepo.get().communitySharingAllowed).toBe(false); // documented default
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-3", login: "share-route-user-3", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "share-disallowed" });

    const onRes = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: true }),
    });
    expect(onRes.status).toBe(403);
    expect(repo.userRepo.getUser(user.id)!.shareBenchmarks).toBe(true); // default-on, unchanged by the rejected call

    const offRes = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: false }),
    });
    expect(offRes.status).toBe(200);
  });

  it("allows turning it ON once the operator sets communitySharingAllowed", async () => {
    repo.appSettingsRepo.setCommunitySharingAllowed(true);
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-4", login: "share-route-user-4", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "share-allowed" });

    const res = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: true }),
    });
    expect(res.status).toBe(200);
    repo.appSettingsRepo.setCommunitySharingAllowed(false); // restore the default for later tests
  });

  it("403s turning it ON while communitySharingAllowed is on but communityUserChoiceAllowed is off -- turning it OFF is still always allowed", async () => {
    repo.appSettingsRepo.setCommunitySharingAllowed(true);
    repo.appSettingsRepo.setCommunityUserChoiceAllowed(false);
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-5", login: "share-route-user-5", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "share-no-user-choice" });

    const onRes = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: true }),
    });
    expect(onRes.status).toBe(403);
    expect(repo.userRepo.getUser(user.id)!.shareBenchmarks).toBe(true); // default-on, unchanged by the rejected call

    const offRes = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: false }),
    });
    expect(offRes.status).toBe(200);

    // Restore the defaults for later tests.
    repo.appSettingsRepo.setCommunitySharingAllowed(false);
    repo.appSettingsRepo.setCommunityUserChoiceAllowed(true);
  });

  it("allows turning it ON once both communitySharingAllowed and communityUserChoiceAllowed are set", async () => {
    repo.appSettingsRepo.setCommunitySharingAllowed(true);
    repo.appSettingsRepo.setCommunityUserChoiceAllowed(true);
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "share-route-6", login: "share-route-user-6", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "share-both-allowed" });

    const res = await fetch(`${baseUrl}/api/auth/share-benchmarks`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ enabled: true }),
    });
    expect(res.status).toBe(200);
    repo.appSettingsRepo.setCommunitySharingAllowed(false); // restore the default for later tests
  });
});

// Security checklist's own "Account deletion ships in v1" item.
describe("DELETE /api/auth/account", () => {
  it("requires authentication", async () => {
    const res = await fetch(`${baseUrl}/api/auth/account`, {
      method: "DELETE",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ confirm: true }),
    });
    expect(res.status).toBe(401);
  });

  it("400s without an explicit confirm:true -- a bare DELETE must never delete anything", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "delete-route-noconfirm", login: "no-confirm", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "no-confirm" });

    const bareRes = await fetch(`${baseUrl}/api/auth/account`, { method: "DELETE", headers: authed(token) });
    expect(bareRes.status).toBe(400);

    const falseRes = await fetch(`${baseUrl}/api/auth/account`, {
      method: "DELETE",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ confirm: false }),
    });
    expect(falseRes.status).toBe(400);

    // Still there -- neither request should have deleted anything.
    expect(repo.userRepo.getUser(user.id)).toBeDefined();
  });

  it("deletes the caller's own account, clears the session cookie, and the old token stops working", async () => {
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "delete-route-confirm", login: "confirmed", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "confirmed" });

    const res = await fetch(`${baseUrl}/api/auth/account`, {
      method: "DELETE",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ confirm: true }),
    });
    expect(res.status).toBe(200);
    expect((await res.json()) as { ok: boolean }).toEqual({ ok: true });
    expect(res.headers.get("set-cookie")).toMatch(/lt_session=;/); // cleared

    expect(repo.userRepo.getUser(user.id)).toBeUndefined();

    const afterRes = await fetch(`${baseUrl}/api/sessions`, { headers: authed(token) });
    expect(afterRes.status).toBe(401); // the session this token names no longer exists
  });

  it("403s the whole request (before even checking confirm) when the operator has disabled account deletion", async () => {
    repo.appSettingsRepo.setAccountDeletionAllowed(false);
    const user = repo.userRepo.upsertByIdentity("github", { providerUserId: "delete-route-disabled", login: "disabled", avatarUrl: null });
    const { token } = repo.sessionRepo.create(user.id, { label: "delete-disabled" });

    const res = await fetch(`${baseUrl}/api/auth/account`, {
      method: "DELETE",
      headers: { "content-type": "application/json", ...authed(token) },
      body: JSON.stringify({ confirm: true }),
    });
    expect(res.status).toBe(403);
    expect(repo.userRepo.getUser(user.id)).toBeDefined(); // untouched

    repo.appSettingsRepo.setAccountDeletionAllowed(true); // restore the default for later tests
  });
});
