import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyCookie from "@fastify/cookie";

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-auth-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.PUBLIC_URL = "http://localhost:9999";
process.env.GITHUB_CLIENT_ID = "test-client-id";
process.env.GITHUB_CLIENT_SECRET = "test-client-secret";
// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): must be set before superadmin.js
// and admin.js are ever imported -- both read their env vars into
// module-level state at import time, not per-request (see each file's own
// doc comment). Harmless for every OTHER describe block in this file: none
// of them send an X-Forwarded-Host that matches this, so req.hostname never
// equals ADMIN_HOSTNAME for them regardless of these being set.
process.env.ADMIN_PUBLIC_URL = "http://supervise.test.local";
process.env.SUPERADMIN_IDENTITIES = "github:9100";
const ADMIN_HOST = "supervise.test.local";

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];
let hashToken: (t: string) => string;

// GitHub's real token/profile endpoints are never reached from a test --
// selectively intercepted here (anything NOT targeting the local test
// server passes through unmodified, though nothing else is expected to be
// called). Mutated per-test to control the exchange's outcome.
const originalFetch = global.fetch;
let mockTokenBody: unknown = { access_token: "fake-access-token" };
let mockTokenOk = true;
let mockProfileBody: unknown = { id: 555, login: "octocat", avatar_url: "https://example.invalid/a.png" };
let mockProfileOk = true;

beforeEach(() => {
  mockTokenBody = { access_token: "fake-access-token" };
  mockTokenOk = true;
  mockProfileBody = { id: 555, login: "octocat", avatar_url: "https://example.invalid/a.png" };
  mockProfileOk = true;
});

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  ({ hashToken } = await import("../session.js"));
  const { authRoutes } = await import("./auth.js");

  global.fetch = (async (url: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const urlStr = typeof url === "string" ? url : url instanceof URL ? url.toString() : (url as Request).url;
    if (urlStr.startsWith("https://github.com/login/oauth/access_token")) {
      return new Response(JSON.stringify(mockTokenBody), { status: mockTokenOk ? 200 : 500 });
    }
    if (urlStr.startsWith("https://api.github.com/user")) {
      return new Response(JSON.stringify(mockProfileBody), { status: mockProfileOk ? 200 : 500 });
    }
    return originalFetch(url as any, init);
  }) as typeof fetch;

  // trustProxy so req.hostname reads X-Forwarded-Host, letting the admin-
  // hostname tests below simulate a second origin without a real DNS entry
  // -- see admin.test.ts's own doc comment for why (Host itself can't be
  // overridden via fetch()).
  app = Fastify({ logger: false, trustProxy: true });
  await app.register(fastifyCookie);
  await app.register(authRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  global.fetch = originalFetch;
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

afterEach(() => {
  mockTokenBody = { access_token: "fake-access-token" };
  mockTokenOk = true;
  mockProfileBody = { id: 555, login: "octocat", avatar_url: "https://example.invalid/a.png" };
  mockProfileOk = true;
});

function cookieValue(res: Response, name: string): string | undefined {
  const found = res.headers.getSetCookie().find((c) => c.startsWith(`${name}=`));
  return found?.split(";")[0]?.split("=").slice(1).join("=");
}

async function startLogin(
  extraHeaders: Record<string, string> = {},
  query = ""
): Promise<{ state: string; stateCookie: string; linkCookie?: string }> {
  const res = await fetch(`${baseUrl}/auth/github${query}`, { redirect: "manual", headers: extraHeaders });
  expect(res.status).toBe(302);
  const location = res.headers.get("location")!;
  const state = new URL(location).searchParams.get("state")!;
  const stateCookie = cookieValue(res, "oauth_state")!;
  const linkCookie = cookieValue(res, "oauth_link_user");
  return { state, stateCookie, linkCookie };
}

async function callback(s: { state: string; stateCookie: string }, sessionToken?: string): Promise<Response> {
  const cookie = `oauth_state=${s.stateCookie}` + (sessionToken ? `; lt_session=${sessionToken}` : "");
  return fetch(`${baseUrl}/auth/github/callback?code=c&state=${s.state}`, { redirect: "manual", headers: { cookie } });
}

async function loginAs(id: number, login: string): Promise<{ token: string; userId: string }> {
  mockProfileBody = { id, login, avatar_url: null };
  const res = await callback(await startLogin());
  const token = cookieValue(res, "lt_session")!;
  return { token, userId: repo.sessionRepo.getByTokenHash(hashToken(token))!.userId };
}

describe("GET /api/auth/status", () => {
  it("returns {user: null} with no session", async () => {
    const res = await fetch(`${baseUrl}/api/auth/status`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { user: unknown };
    expect(body.user).toBeNull();
  });
});

describe("GET /auth/:provider", () => {
  it("404s for an unknown provider", async () => {
    const res = await fetch(`${baseUrl}/auth/not-a-real-provider`, { redirect: "manual" });
    expect(res.status).toBe(404);
  });

  it("redirects to GitHub's authorize URL with a fresh state, and sets the oauth_state cookie", async () => {
    const { state, stateCookie } = await startLogin();
    expect(state).toHaveLength(43); // base64url(32 bytes), see session.ts's generateState
    expect(stateCookie).toBe(state);
  });

  it("never sets the legacy oauth_link_user cookie, even with ?link=1", async () => {
    const res = await fetch(`${baseUrl}/auth/github?link=1`, { redirect: "manual" });
    expect(cookieValue(res, "oauth_link_user")).toBeUndefined();
  });
});

describe("GET /auth/:provider/callback", () => {
  it("redirects with oauth_state error on a missing/mismatched state (CSRF guard)", async () => {
    const { stateCookie } = await startLogin();
    const res = await fetch(`${baseUrl}/auth/github/callback?code=abc&state=WRONG`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${stateCookie}` },
    });
    expect(res.headers.get("location")).toBe("/login?error=oauth_state");
  });

  it("redirects with oauth_exchange error when the token endpoint returns no access_token (e.g. a replayed code)", async () => {
    const { state, stateCookie } = await startLogin();
    mockTokenBody = { error: "bad_verification_code" }; // real GitHub behavior -- HTTP 200, no access_token
    const res = await fetch(`${baseUrl}/auth/github/callback?code=replayed&state=${state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${stateCookie}` },
    });
    expect(res.headers.get("location")).toBe("/login?error=oauth_exchange");
  });

  it("redirects with oauth_profile error when the profile fetch fails", async () => {
    const { state, stateCookie } = await startLogin();
    mockProfileOk = false;
    const res = await fetch(`${baseUrl}/auth/github/callback?code=abc&state=${state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${stateCookie}` },
    });
    expect(res.headers.get("location")).toBe("/login?error=oauth_profile");
  });

  it("a successful callback creates a user + session and redirects home", async () => {
    const { state, stateCookie } = await startLogin();
    mockProfileBody = { id: 9001, login: "new-user", avatar_url: null };
    const res = await fetch(`${baseUrl}/auth/github/callback?code=abc&state=${state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${stateCookie}` },
    });
    expect(res.headers.get("location")).toBe("/");
    const sessionToken = cookieValue(res, "lt_session");
    expect(sessionToken).toBeDefined();

    const session = repo.sessionRepo.getByTokenHash(hashToken(sessionToken!));
    expect(session).toBeDefined();
    const user = repo.userRepo.getUser(session!.userId);
    expect(user?.displayName).toBe("new-user");
  });

  it("a returning user's callback resolves to the SAME account, not a new one", async () => {
    mockProfileBody = { id: 9002, login: "returning-user", avatar_url: null };
    const first = await startLogin();
    const firstRes = await fetch(`${baseUrl}/auth/github/callback?code=a&state=${first.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${first.stateCookie}` },
    });
    const firstUserId = repo.userRepo.getUser(
      repo.sessionRepo.getByTokenHash(hashToken(cookieValue(firstRes, "lt_session")!))!.userId
    )!.id;

    const second = await startLogin();
    const secondRes = await fetch(`${baseUrl}/auth/github/callback?code=b&state=${second.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${second.stateCookie}` },
    });
    const secondUserId = repo.userRepo.getUser(
      repo.sessionRepo.getByTokenHash(hashToken(cookieValue(secondRes, "lt_session")!))!.userId
    )!.id;

    expect(secondUserId).toBe(firstUserId);
  });

  it("account linking: an authenticated user connecting a second provider attaches it to their own account", async () => {
    const primary = await loginAs(9003, "link-primary");

    // Start the link flow WHILE authenticated -- the intent lives server-side,
    // never in a cookie.
    const linkStart = await startLogin({ cookie: `lt_session=${primary.token}` }, "?link=1");
    expect(linkStart.linkCookie).toBeUndefined();

    mockProfileBody = { id: 9004, login: "link-secondary", avatar_url: null };
    const linkRes = await callback(linkStart, primary.token);
    expect(linkRes.headers.get("location")).toBe("/settings?linked=1");
    // The session that asked to link is kept -- no second session minted.
    expect(cookieValue(linkRes, "lt_session")).toBeUndefined();

    const identities = repo.userRepo.getIdentities(primary.userId);
    expect(identities.some((i) => i.providerUserId === "9004")).toBe(true);
    expect(identities).toHaveLength(1 + 1); // primary (9003) + newly linked (9004)
  });

  it("account linking rejects an identity that's already someone else's account", async () => {
    await loginAs(9005, "owner-a");
    const b = await loginAs(9006, "owner-b");

    const linkStart = await startLogin({ cookie: `lt_session=${b.token}` }, "?link=1");
    mockProfileBody = { id: 9005, login: "owner-a-again", avatar_url: null }; // same id as account A
    const linkRes = await callback(linkStart, b.token);
    expect(linkRes.headers.get("location")).toBe("/settings?error=already_linked");

    const owner = repo.userRepo.findByIdentity("github", "9005");
    expect(owner?.id).not.toBe(b.userId);
  });

  // The reported takeover: an unsigned oauth_link_user cookie named the
  // account to link to, so an attacker could attach their own identity to
  // any user id and then log in as that user.
  it("a forged oauth_link_user cookie (no session) is ignored -- plain login, victim untouched", async () => {
    const victim = await loginAs(9010, "victim-1");
    const s = await startLogin();
    mockProfileBody = { id: 9011, login: "attacker-1", avatar_url: null };
    const res = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${s.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${s.stateCookie}; oauth_link_user=${victim.userId}` },
    });
    expect(res.headers.get("location")).toBe("/");
    expect(repo.userRepo.findByIdentity("github", "9011")?.id).not.toBe(victim.userId);
    expect(repo.userRepo.getIdentities(victim.userId)).toHaveLength(1);
  });

  it("a forged oauth_link_user cookie alongside the attacker's own session links nothing to the victim", async () => {
    const victim = await loginAs(9012, "victim-2");
    const attacker = await loginAs(9013, "attacker-2");
    const s = await startLogin({ cookie: `lt_session=${attacker.token}` }, "?link=1");
    mockProfileBody = { id: 9014, login: "attacker-2-alt", avatar_url: null };
    const res = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${s.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${s.stateCookie}; oauth_link_user=${victim.userId}; lt_session=${attacker.token}` },
    });
    expect(res.headers.get("location")).toBe("/settings?linked=1");
    expect(repo.userRepo.findByIdentity("github", "9014")?.id).toBe(attacker.userId);
    expect(repo.userRepo.getIdentities(victim.userId)).toHaveLength(1);
  });

  it("a link state completed in a DIFFERENT session (e.g. planted in a victim's browser) is rejected", async () => {
    const attacker = await loginAs(9015, "attacker-3");
    const victim = await loginAs(9016, "victim-3");
    const s = await startLogin({ cookie: `lt_session=${attacker.token}` }, "?link=1");
    mockProfileBody = { id: 9017, login: "attacker-3-alt", avatar_url: null };
    const res = await callback(s, victim.token);
    expect(res.headers.get("location")).toBe("/settings?error=link_session");
    expect(repo.userRepo.findByIdentity("github", "9017")).toBeFalsy();
    expect(repo.userRepo.getIdentities(victim.userId)).toHaveLength(1);
  });

  it("a link state completed with NO session fails closed -- never falls back to a plain login", async () => {
    const owner = await loginAs(9018, "owner-4");
    const s = await startLogin({ cookie: `lt_session=${owner.token}` }, "?link=1");
    mockProfileBody = { id: 9019, login: "someone", avatar_url: null };
    const res = await callback(s);
    expect(res.headers.get("location")).toBe("/settings?error=link_session");
    expect(cookieValue(res, "lt_session")).toBeUndefined();
    expect(repo.userRepo.findByIdentity("github", "9019")).toBeFalsy();
  });

  it("a link state whose session was revoked in between is rejected", async () => {
    const owner = await loginAs(9020, "owner-5");
    const s = await startLogin({ cookie: `lt_session=${owner.token}` }, "?link=1");
    await fetch(`${baseUrl}/auth/logout`, { redirect: "manual", headers: { cookie: `lt_session=${owner.token}` } });
    mockProfileBody = { id: 9021, login: "late", avatar_url: null };
    const res = await callback(s, owner.token);
    // Revoking the session cascades the intent away, so this is just a login
    // of identity 9021 as its own account -- never a link to owner's.
    expect(res.headers.get("location")).toBe("/");
    expect(repo.userRepo.findByIdentity("github", "9021")?.id).not.toBe(owner.userId);
    expect(repo.userRepo.getIdentities(owner.userId)).toHaveLength(1);
  });

  it("a link state is single-use -- replaying it does not link a second identity", async () => {
    const owner = await loginAs(9022, "owner-6");
    const s = await startLogin({ cookie: `lt_session=${owner.token}` }, "?link=1");
    mockProfileBody = { id: 9023, login: "first", avatar_url: null };
    expect((await callback(s, owner.token)).headers.get("location")).toBe("/settings?linked=1");

    mockProfileBody = { id: 9024, login: "replay", avatar_url: null };
    const replay = await callback(s, owner.token);
    // No intent left, so this is an ordinary login of a brand-new identity.
    expect(replay.headers.get("location")).toBe("/");
    expect(repo.userRepo.findByIdentity("github", "9024")?.id).not.toBe(owner.userId);
    expect(repo.userRepo.getIdentities(owner.userId)).toHaveLength(2);
  });

  it("an expired link intent is ignored and pruned", async () => {
    const owner = await loginAs(9025, "owner-7");
    const s = await startLogin({ cookie: `lt_session=${owner.token}` }, "?link=1");
    // Nothing else can age the row -- expire it directly.
    const { getDb } = await import("../db/migrate.js");
    getDb().prepare(`UPDATE oauth_link_intents SET expires_at = ?`).run(Date.now() - 1);
    expect(repo.sessionRepo.pruneExpiredLinkIntents()).toBeGreaterThanOrEqual(1);
    mockProfileBody = { id: 9026, login: "stale", avatar_url: null };
    const res = await callback(s, owner.token);
    expect(res.headers.get("location")).toBe("/");
    expect(repo.userRepo.getIdentities(owner.userId)).toHaveLength(1);
  });
});

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the admin origin's own OAuth
// branch -- lookup-only (never mints a new account), superadmin-gated, and
// a host-only session cookie. Uses X-Forwarded-Host (see beforeAll's own
// trustProxy comment) to simulate the second hostname.
describe("admin origin OAuth branch (§5.1)", () => {
  async function startAdminLogin(extraHeaders: Record<string, string> = {}) {
    return startLogin({ "x-forwarded-host": ADMIN_HOST, ...extraHeaders });
  }

  it("the start handler builds a redirect_uri pointing at the ADMIN origin's own callback", async () => {
    const res = await fetch(`${baseUrl}/auth/github`, {
      redirect: "manual",
      headers: { "x-forwarded-host": ADMIN_HOST },
    });
    const location = res.headers.get("location")!;
    const redirectUri = new URL(location).searchParams.get("redirect_uri")!;
    expect(redirectUri).toBe("http://supervise.test.local/auth/github/callback");
  });

  it("never sets oauth_link_user on the admin origin, even with ?link=1 and an admin-origin session already present", async () => {
    // Get a real admin-origin session first (superadmin identity 9100).
    mockProfileBody = { id: 9100, login: "the-admin", avatar_url: null };
    // This identity has no prior MAIN-site account -- findByIdentity would
    // return undefined, so bootstrap one directly via the repo (equivalent
    // to "the operator already has an account from using the main app").
    repo.userRepo.upsertByIdentity("github", { providerUserId: "9100", login: "the-admin", avatarUrl: null });
    const login = await startAdminLogin();
    const callbackRes = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${login.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${login.stateCookie}`, "x-forwarded-host": ADMIN_HOST },
    });
    const adminSessionToken = cookieValue(callbackRes, "lt_session")!;
    expect(adminSessionToken).toBeDefined();

    const res = await fetch(`${baseUrl}/auth/github?link=1`, {
      redirect: "manual",
      headers: { cookie: `lt_session=${adminSessionToken}`, "x-forwarded-host": ADMIN_HOST },
    });
    expect(cookieValue(res, "oauth_link_user")).toBeUndefined();
    const { getDb } = await import("../db/migrate.js");
    const state = new URL(res.headers.get("location")!).searchParams.get("state")!;
    expect(getDb().prepare(`SELECT 1 FROM oauth_link_intents WHERE state_hash = ?`).get(hashToken(state))).toBeUndefined();
  });

  it("a superadmin-listed identity WITH an existing account gets a session and lands on the admin SPA's own root", async () => {
    mockProfileBody = { id: 9100, login: "the-admin", avatar_url: null };
    repo.userRepo.upsertByIdentity("github", { providerUserId: "9100", login: "the-admin", avatarUrl: null });

    const login = await startAdminLogin();
    const res = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${login.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${login.stateCookie}`, "x-forwarded-host": ADMIN_HOST },
    });
    expect(res.headers.get("location")).toBe("/");
    const token = cookieValue(res, "lt_session");
    expect(token).toBeDefined();
    expect(repo.sessionRepo.getByTokenHash(hashToken(token!))).toBeDefined();

    // Host-only cookie -- no Domain attribute anywhere in the raw Set-Cookie
    // line (see routes/auth.ts's own comment on why this IS the isolation
    // mechanism).
    const rawSetCookie = res.headers.getSetCookie().find((c) => c.startsWith("lt_session="))!;
    expect(rawSetCookie.toLowerCase()).not.toContain("domain=");
  });

  it("a superadmin-listed identity with NO existing account is rejected -- lookup-only, never upserts", async () => {
    mockProfileBody = { id: 9101, login: "never-logged-in-before", avatar_url: null };
    // Deliberately no repo.userRepo.upsertByIdentity call -- this identity
    // has never touched the main site.
    process.env.SUPERADMIN_IDENTITIES = "github:9100,github:9101";

    const login = await startAdminLogin();
    const res = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${login.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${login.stateCookie}`, "x-forwarded-host": ADMIN_HOST },
    });
    // Bounces to the MAIN site's absolute /login, not a relative one --
    // this origin's own admin SPA has no login page.
    expect(res.headers.get("location")).toBe("http://localhost:9999/login?error=not_authorized");
    expect(cookieValue(res, "lt_session")).toBeUndefined();
    expect(repo.userRepo.findByIdentity("github", "9101")).toBeUndefined();

    process.env.SUPERADMIN_IDENTITIES = "github:9100"; // restore for any later test ordering
  });

  it("an identity with a real account but NOT on the superadmin list is rejected", async () => {
    mockProfileBody = { id: 9102, login: "regular-account", avatar_url: null };
    repo.userRepo.upsertByIdentity("github", { providerUserId: "9102", login: "regular-account", avatarUrl: null });

    const login = await startAdminLogin();
    const res = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${login.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${login.stateCookie}`, "x-forwarded-host": ADMIN_HOST },
    });
    expect(res.headers.get("location")).toBe("http://localhost:9999/login?error=not_authorized");
    expect(cookieValue(res, "lt_session")).toBeUndefined();
  });
});

describe("GET /auth/logout", () => {
  it("revokes the current session and clears the cookie", async () => {
    mockProfileBody = { id: 9007, login: "logout-user", avatar_url: null };
    const login = await startLogin();
    const loginRes = await fetch(`${baseUrl}/auth/github/callback?code=x&state=${login.state}`, {
      redirect: "manual",
      headers: { cookie: `oauth_state=${login.stateCookie}` },
    });
    const token = cookieValue(loginRes, "lt_session")!;
    expect(repo.sessionRepo.getByTokenHash(hashToken(token))).toBeDefined();

    const logoutRes = await fetch(`${baseUrl}/auth/logout`, {
      redirect: "manual",
      headers: { cookie: `lt_session=${token}` },
    });
    expect(logoutRes.headers.get("location")).toBe("/login");
    expect(repo.sessionRepo.getByTokenHash(hashToken(token))).toBeUndefined();
  });

  it("is a harmless no-op when nobody is logged in", async () => {
    const res = await fetch(`${baseUrl}/auth/logout`, { redirect: "manual" });
    expect(res.headers.get("location")).toBe("/login");
  });
});
