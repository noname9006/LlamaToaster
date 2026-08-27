import type { FastifyInstance, FastifyRequest } from "fastify";
import { repo } from "../db/repo.js";
import { generateState, safeEqual } from "../session.js";
import { resolveAuthUser } from "../auth-middleware.js";
import { isSuperadminIdentity } from "../superadmin.js";
import { ADMIN_HOSTNAME } from "./admin.js";
import type { AuthStatus, AuthUser } from "../../../shared/types.js";

// Same-origin decision (MULTIUSER_PLAN.md §0.5): secure is DERIVED, never
// hardcoded, so login still works on plain http://localhost in dev --
// v3.3's hardcoded `secure: true` made that impossible.
function cookiesSecure(): boolean {
  const publicUrl = process.env.PUBLIC_URL;
  if (!publicUrl) return false;
  try {
    return new URL(publicUrl).protocol === "https:";
  } catch {
    return false;
  }
}

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): which base URL a given
// request's OAuth round trip belongs to -- ADMIN_PUBLIC_URL when the request
// came in on the admin hostname, PUBLIC_URL otherwise. Used identically by
// the start handler (building the redirect_uri sent to the provider) and the
// callback handler (rebuilding that SAME redirect_uri for the token
// exchange, which the provider rejects on a mismatch) -- if these two ever
// disagreed for one request, login would break on whichever origin picked
// the "wrong" one. This is what lets an operator start login FROM the admin
// origin at all: without it, the start handler would always build a
// main-site redirect_uri regardless of which hostname initiated the flow,
// so the provider would send them back to the main site's callback instead
// of the admin one, and the hostname branch below would never even see the
// request.
function originForRequest(req: FastifyRequest): string | undefined {
  if (ADMIN_HOSTNAME && req.hostname === ADMIN_HOSTNAME) return process.env.ADMIN_PUBLIC_URL;
  return process.env.PUBLIC_URL;
}

// One small object per provider (§2.4) -- the route handlers below never
// change when a provider is added; only mapProfile and this entry do.
interface OAuthProvider {
  id: string;
  authorizeUrl: string;
  scope: string; // profile only, never email (§2.1)
  tokenUrl: string;
  profileUrl: string;
  clientId: string;
  clientSecret: string;
  mapProfile(json: unknown): { providerUserId: string; login: string; avatarUrl: string | null } | null;
}

const PROVIDERS: Record<string, OAuthProvider> = {
  github: {
    id: "github",
    authorizeUrl: "https://github.com/login/oauth/authorize",
    scope: "read:user", // NOT user:email -- see §2.1
    tokenUrl: "https://github.com/login/oauth/access_token",
    profileUrl: "https://api.github.com/user",
    clientId: process.env.GITHUB_CLIENT_ID ?? "",
    clientSecret: process.env.GITHUB_CLIENT_SECRET ?? "",
    mapProfile: (j) => {
      const p = j as { id?: number; login?: string; avatar_url?: string };
      return typeof p.id === "number" && p.login
        ? { providerUserId: String(p.id), login: p.login, avatarUrl: p.avatar_url ?? null }
        : null;
    },
  },
  // google: { ... } -- same shape, different URLs/scope ("openid profile"),
  // different mapProfile (Google's stable id is the `sub` claim, not `id`).
  // Nothing else in this file changes when it's added (§2.4).
};

// Rough "Browser on OS" label for the Settings page's session list -- not a
// full UA-parsing library (this project avoids adding dependencies for a
// cosmetic label), just enough to distinguish a handful of sessions from
// each other. Falls back to a truncated raw UA string for anything
// unrecognized rather than a useless "Unknown".
function describeUserAgent(req: FastifyRequest): string {
  const ua = req.headers["user-agent"];
  if (!ua) return "Unknown device";
  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /OPR\//.test(ua)
      ? "Opera"
      : /Chrome\//.test(ua)
        ? "Chrome"
        : /Firefox\//.test(ua)
          ? "Firefox"
          : /Safari\//.test(ua)
            ? "Safari"
            : null;
  const os = /Windows/.test(ua)
    ? "Windows"
    : /Mac OS X/.test(ua)
      ? "macOS"
      : /Android/.test(ua)
        ? "Android"
        : /iPhone|iPad/.test(ua)
          ? "iOS"
          : /Linux/.test(ua)
            ? "Linux"
            : null;
  if (browser && os) return `${browser} on ${os}`;
  if (browser) return browser;
  return ua.slice(0, 80);
}

export async function authRoutes(app: FastifyInstance): Promise<void> {
  // Always registered, whether or not AUTH_ENABLED -- this is the SPA's own
  // boot check, and it needs to answer "is there even a login gate right
  // now" as much as "who is logged in", since AUTH_ENABLED can be off with
  // these routes still present (MULTIUSER_PLAN.md §2.3's own
  // independently-deployable Stage 1/Stage 2 split).
  app.get("/api/auth/status", async (req): Promise<AuthStatus> => {
    const authEnabled = process.env.AUTH_ENABLED === "true";
    const appSettings = repo.appSettingsRepo.get();
    if (!authEnabled) return { user: null, authEnabled, appSettings };
    return { user: resolveAuthUser(req)?.user ?? null, authEnabled, appSettings };
  });

  app.get<{ Params: { provider: string }; Querystring: { link?: string } }>(
    "/auth/:provider",
    // §2.6: 10/min per IP -- unauthenticated by nature (this is how login
    // itself starts), so IP is the only key available; the default
    // keyGenerator (request.ip) is exactly right here.
    { config: { rateLimit: { max: 10, timeWindow: "1 minute" } } },
    async (req, reply) => {
      const provider = PROVIDERS[req.params.provider];
      if (!provider) return reply.code(404).send({ error: "unknown provider" });

      const publicUrl = originForRequest(req);
      if (!publicUrl) {
        return reply.code(503).send({ error: "PUBLIC_URL is not configured on this server" });
      }

      const state = generateState();
      const secure = cookiesSecure();
      reply.setCookie("oauth_state", state, { httpOnly: true, secure, sameSite: "lax", path: "/", maxAge: 600 });

      // Settings -> "Connect another account" (§2.4) -- carries the "attach
      // to my existing, currently-authenticated account" intent through the
      // redirect round trip. /auth/:provider is itself a PUBLIC_PATH (an
      // unauthenticated visitor must be able to reach it to log in at all),
      // so req.user is never populated by the middleware here -- resolved
      // directly instead. Never on the admin origin (§5.1: "no account
      // linking here either, linking is a main-site concept") -- guarded
      // explicitly rather than relying on the admin cookie jar's isolation
      // alone, since an admin-origin session visiting this URL with ?link=1
      // would otherwise still have a real `authed` to carry through.
      const authed = ADMIN_HOSTNAME && req.hostname === ADMIN_HOSTNAME ? null : resolveAuthUser(req);
      if (req.query.link === "1" && authed) {
        reply.setCookie("oauth_link_user", authed.user.id, {
          httpOnly: true,
          secure,
          sameSite: "lax",
          path: "/",
          maxAge: 600,
        });
      }

      const params = new URLSearchParams({
        client_id: provider.clientId,
        scope: provider.scope,
        state,
        redirect_uri: `${publicUrl}/auth/${provider.id}/callback`,
      });
      return reply.redirect(`${provider.authorizeUrl}?${params}`);
    }
  );

  app.get<{ Params: { provider: string }; Querystring: { code?: string; state?: string } }>(
    "/auth/:provider/callback",
    { config: { rateLimit: { max: 10, timeWindow: "1 minute" } } },
    async (req, reply) => {
      const provider = PROVIDERS[req.params.provider];
      if (!provider) return reply.code(404).send({ error: "unknown provider" });

      // MUST match whatever the start handler used to build the redirect_uri
      // it sent the provider (see originForRequest's own doc comment) -- a
      // mismatch here isn't a local bug, the provider itself rejects the
      // token exchange over it.
      const publicUrl = originForRequest(req);
      if (!publicUrl) {
        return reply.code(503).send({ error: "PUBLIC_URL is not configured on this server" });
      }

      const { code, state } = req.query;
      const cookieState = req.cookies?.oauth_state;
      const linkUserId = req.cookies?.oauth_link_user ?? null;
      reply.clearCookie("oauth_state", { path: "/" });
      reply.clearCookie("oauth_link_user", { path: "/" });

      if (!code || !state || !cookieState || !safeEqual(state, cookieState)) {
        return reply.redirect("/login?error=oauth_state");
      }

      let tokenJson: { access_token?: string; error?: string };
      try {
        const tokenRes = await fetch(provider.tokenUrl, {
          method: "POST",
          headers: { "content-type": "application/json", accept: "application/json" },
          body: JSON.stringify({
            client_id: provider.clientId,
            client_secret: provider.clientSecret,
            code,
            redirect_uri: `${publicUrl}/auth/${provider.id}/callback`,
          }),
        });
        tokenJson = (await tokenRes.json()) as { access_token?: string; error?: string };
      } catch (err) {
        req.log.warn({ provider: provider.id, error: err instanceof Error ? err.message : String(err) }, "oauth token exchange failed");
        return reply.redirect("/login?error=oauth_exchange");
      }
      // GitHub returns HTTP 200 with {error: "bad_verification_code"} on a
      // replayed/expired code -- naively destructuring access_token here
      // would turn that into a 500 on a NOT NULL violation three calls
      // later, instead of a clean redirect.
      if (!tokenJson.access_token) return reply.redirect("/login?error=oauth_exchange");

      let mapped: { providerUserId: string; login: string; avatarUrl: string | null } | null;
      try {
        const profileRes = await fetch(provider.profileUrl, {
          headers: { authorization: `Bearer ${tokenJson.access_token}`, accept: "application/json" },
        });
        if (!profileRes.ok) return reply.redirect("/login?error=oauth_profile");
        mapped = provider.mapProfile(await profileRes.json());
      } catch (err) {
        req.log.warn({ provider: provider.id, error: err instanceof Error ? err.message : String(err) }, "oauth profile fetch failed");
        return reply.redirect("/login?error=oauth_profile");
      }
      if (!mapped) return reply.redirect("/login?error=oauth_profile");

      // Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the admin origin's own
      // branch, checked before the normal login/link path below and always
      // returning here regardless of outcome -- signing in on this hostname
      // must never fall through to upsertByIdentity (which would silently
      // mint a normal-user account for someone who isn't already
      // allowlisted) or to the linking flow (a main-site-only concept).
      if (ADMIN_HOSTNAME && req.hostname === ADMIN_HOSTNAME) {
        // Lookup-only -- findByIdentity returns undefined for anyone who
        // has never logged into the MAIN site with this identity before,
        // which is the common case for "not actually an admin" and is
        // indistinguishable here from "wrong account": either way, no
        // session gets created.
        const account = repo.userRepo.findByIdentity(provider.id, mapped.providerUserId);
        if (!account || !isSuperadminIdentity(provider.id, mapped.providerUserId)) {
          // Bounces to the MAIN site's own /login -- this origin's small
          // admin SPA has no login page of its own to redirect back to.
          const mainUrl = process.env.PUBLIC_URL ?? "/";
          return reply.redirect(`${mainUrl}/login?error=not_authorized`);
        }
        // One independent session per device/browser: reusing a single active
        // session row and rotating its token on every login (the previous
        // behavior) invalidated every OTHER browser the same account was
        // already logged into -- logging in on a laptop cut off the desktop.
        // Each callback mints its own 30-day session row instead; sessions
        // remain individually revocable from Settings' session list.
        const token = repo.sessionRepo.create(account.id, { label: describeUserAgent(req) }).token;
        // HOST-ONLY cookie -- no `domain` option set, same as every other
        // setCookie in this file. That absence is the entire isolation
        // mechanism: a Domain=llamatoaster.com attribute here would make
        // this cookie valid on the main site too and silently undo the
        // whole point of a separate origin (§5.1).
        reply.setCookie("lt_session", token, {
          httpOnly: true,
          secure: cookiesSecure(),
          sameSite: "lax",
          path: "/",
          maxAge: 30 * 24 * 3600,
        });
        return reply.redirect("/");
      }

      let user: AuthUser;
      if (linkUserId) {
        // Reject linking an identity that's already someone ELSE's account
        // (MULTIUSER_PLAN.md §2.4's own warning: silently re-pointing an
        // existing identity would be an account-takeover primitive). Linking
        // it to the SAME account it's already on, or linking it fresh, are
        // both fine -- userRepo.linkIdentity handles those two cases.
        const existingOwner = repo.userRepo.findByIdentity(provider.id, mapped.providerUserId);
        if (existingOwner && existingOwner.id !== linkUserId) {
          return reply.redirect("/settings?error=already_linked");
        }
        const linked = repo.userRepo.linkIdentity(linkUserId, provider.id, mapped);
        user = {
          id: linked.id,
          displayName: linked.displayName,
          avatarUrl: linked.avatarUrl,
          isSuperadmin: false,
          shareBenchmarks: linked.shareBenchmarks,
        };
      } else {
        const upserted = repo.userRepo.upsertByIdentity(provider.id, mapped);
        user = {
          id: upserted.id,
          displayName: upserted.displayName,
          avatarUrl: upserted.avatarUrl,
          isSuperadmin: false,
          shareBenchmarks: upserted.shareBenchmarks,
        };
      }

      // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.2): one-shot claim of every
      // pre-auth (or pre-this-user) run/result/run_item/worker row, so the
      // operator's own history doesn't become invisible in the normal UI the
      // moment they log in as a brand-new user. Checked on EVERY login of a
      // superadmin-listed identity (covers both a fresh login and a later
      // "link another account" that happens to be superadmin-listed) --
      // repo.claimLegacyHistory itself is the actual guard (the meta flag),
      // so this is a cheap no-op on every call after the first.
      if (isSuperadminIdentity(provider.id, mapped.providerUserId) && repo.claimLegacyHistory(user.id)) {
        req.log.info({ user_id: user.id, provider: provider.id }, "legacy pre-auth history claimed");
      }

      // One independent session per device/browser -- see the admin-origin
      // branch above for why the previous single-session-reuse behavior was
      // removed (it logged out every other machine on each new login).
      const token = repo.sessionRepo.create(user.id, { label: describeUserAgent(req) }).token;
      reply.setCookie("lt_session", token, {
        httpOnly: true,
        secure: cookiesSecure(),
        sameSite: "lax",
        path: "/",
        maxAge: 30 * 24 * 3600,
      });
      return reply.redirect(linkUserId ? "/settings?linked=1" : "/");
    }
  );

  app.get("/auth/logout", async (req, reply) => {
    const resolved = resolveAuthUser(req);
    if (resolved) repo.sessionRepo.revokeById(resolved.session.id);
    reply.clearCookie("lt_session", { path: "/" });
    return reply.redirect("/login");
  });
}
