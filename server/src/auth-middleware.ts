import type { FastifyRequest } from "fastify";
import { UnauthorizedError, ForbiddenError } from "./errors.js";
import { hashToken } from "./session.js";
import { repo } from "./db/repo.js";
import type { SessionRecord } from "./db/repo.js";
import { isSuperadminIdentity } from "./superadmin.js";
import type { AuthUser } from "../../shared/types.js";

// Routes reachable with no session at all -- exact Fastify ROUTE PATTERNS
// (req.routeOptions.url, e.g. "/api/runs/:id"), never the raw request URL,
// which carries the query string and would make "/api/auth/status?x=1" miss
// this set -- an allowlist a client can influence by adding a query string is
// the wrong shape (MULTIUSER_PLAN.md §2.3). /auth/logout is added once its
// route handler exists (see routes/auth.ts).
export const PUBLIC_PATHS = new Set([
  "/health",
  "/api/auth/status",
  // Verifies its own bearer REFRESH token directly (routes/sessions.ts) --
  // a different token space than a normal session's access token, so the
  // generic session check below would always reject it first.
  "/api/auth/refresh",
  "/auth/:provider",
  "/auth/:provider/callback",
  "/auth/logout",
  // Multi-user Stage 3 (MULTIUSER_PLAN.md §3.1/§2.3): device-initiated,
  // before any session exists at all -- the worker calls these itself to
  // obtain and then redeem an enrolment code. GET /api/device/status and
  // POST /api/device/approve are deliberately NOT here -- those are
  // browser-facing and require a real logged-in session (routes/device.ts's
  // deviceApprovalRoutes).
  "/api/device/start",
  "/api/device/token",
]);

// Routes outside the /api/worker/ prefix that are STILL worker-authenticated
// (a Stage 3 session or the Stage 1 WORKER_SHARED_TOKEN, not a browser user
// session) rather than /api/worker/*-prefixed -- both check their own token
// inline instead of calling authenticateWorker() directly, since neither has
// a JSON req.body with a machine_id field to read the Stage 1 fallback's
// identity from (the run-log push's body is raw gzip bytes; the item-tick
// body is a RunItemUpdateInput, which never carried machine_id -- see
// routes/runs.ts's own comments on each for how each resolves the caller
// instead). Keyed on (method, url) together, unlike the /api/worker/ prefix
// check, because both URL patterns are ALSO used by a GET/browser-session
// caller (log download, run detail polling) -- only the POST leg of each is
// worker-to-server.
const WORKER_AUTHENTICATED_ROUTES = new Set([
  "POST /api/runs/:id/log",
  "POST /api/runs/:id/items/:idx",
  // Worker -> server download-completion callback (routes/workers.ts) -- was
  // missed when auth was bolted on, so every download's callback 401'd with
  // "no session" under AUTH_ENABLED (the download itself succeeded, but the
  // model never got registered since the callback never got through).
  "POST /api/models/download-callback",
  // BENCHMARKING_PLAN_V8.md N2/N4 -- worker-authed result ingestion. Both
  // handlers enforce a STRICTER rule than authenticateWorker's dual mode:
  // an enrolled worker session only, never the shared deployment secret,
  // because these rows feed verified claims other tenants see.
  "POST /api/runs/:id/probe-result",
  "POST /api/runs/:id/quality-result",
]);

// Cast target for req.user/req.session once authMiddleware (or a route that
// calls resolveAuthUser directly, e.g. GET /api/auth/status) has populated
// them -- matches MULTIUSER_PLAN.md §2.3's own `(req as AuthenticatedRequest)`
// pattern rather than a global Fastify type augmentation.
export interface AuthenticatedRequest extends FastifyRequest {
  user: AuthUser;
  session: SessionRecord;
}

// Authorization: Bearer first (workers, API clients), then the lt_session
// cookie (browser) -- @fastify/cookie must be registered for req.cookies to
// exist at all; server/src/index.ts only does so when AUTH_ENABLED.
export function extractToken(req: FastifyRequest): string | undefined {
  const auth = req.headers.authorization;
  if (auth?.startsWith("Bearer ")) return auth.slice("Bearer ".length);
  return (req as FastifyRequest & { cookies?: Record<string, string> }).cookies?.lt_session;
}

// The actual token -> session -> user -> isSuperadmin resolution, shared by
// authMiddleware below and by GET /api/auth/status (routes/auth.ts): status
// is itself a PUBLIC_PATH (an unauthenticated caller must get `{user: null}`
// back, not a 401, since the SPA's boot check calls it before any login has
// happened), which means the middleware never runs for it and it has to do
// this resolution itself rather than reading req.user off a hook that was
// skipped. Returns null (never throws) on any failure -- "not logged in" and
// "session invalid" look identical to a caller of this function; only
// authMiddleware itself turns a null into a 401.
export function resolveAuthUser(req: FastifyRequest): { user: AuthUser; session: SessionRecord } | null {
  const token = extractToken(req);
  if (!token) return null;

  const session = repo.sessionRepo.getByTokenHash(hashToken(token));
  if (!session || session.expiresAt < Date.now()) return null;

  const userRecord = repo.userRepo.getUser(session.userId);
  if (!userRecord) return null;

  // Evaluated against EVERY identity linked to this account (§5.1) -- a
  // superadmin's account is superadmin on all its linked providers or none,
  // never partially.
  const identities = repo.userRepo.getIdentities(userRecord.id);
  const isSuperadmin = identities.some((i) => isSuperadminIdentity(i.provider, i.providerUserId));

  const user: AuthUser = {
    id: userRecord.id,
    displayName: userRecord.displayName,
    avatarUrl: userRecord.avatarUrl,
    isSuperadmin,
    shareBenchmarks: userRecord.shareBenchmarks,
  };
  return { user, session };
}

// Rate-limit keyGenerator for a route the plan (§2.6) says is "keyed on
// user" -- @fastify/rate-limit's own hook runs in onRequest, BEFORE
// authMiddleware's preHandler ever populates req.user, so a keyGenerator
// can't just read req.user; it does its own resolveAuthUser lookup instead.
// Falls back to IP when there's no session at all (AUTH_ENABLED off, or an
// unauthenticated caller hitting a route this app doesn't actually gate) --
// a real per-user limit is meaningless with no user to attribute it to.
export function userOrIpKeyGenerator(req: FastifyRequest): string {
  return resolveAuthUser(req)?.user.id ?? req.ip;
}

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3/§4.4): shared ownership gate for
// every route that acts on a specific machine (trigger a run, install/
// activate/delete a build, download/delete a model file) -- undefined userId
// (AUTH_ENABLED off) always passes, matching every other single-tenant-mode
// fallback in this codebase. A worker with no owner yet (a Stage 1 shared-
// token machine, or a Stage 3 enrolment nobody's approved) can't be acted on
// by an authenticated caller either -- ownerId null is still a mismatch, not
// an implicit claim. Uses workerRepo.getEnrolmentById (userId, unlike the
// public Worker type -- see WorkerEnrolment's own doc comment) rather than
// widening what GET /api/workers exposes.
export function assertOwnsWorker(userId: string | undefined, workerId: string): void {
  if (!userId) return;
  const enrolment = repo.workerRepo.getEnrolmentById(workerId);
  if (!enrolment || enrolment.userId !== userId) {
    throw new ForbiddenError("you don't own this machine");
  }
}

export async function authMiddleware(req: FastifyRequest): Promise<void> {
  const routeUrl = req.routeOptions?.url;
  // FAIL CLOSED. An unmatched route (routeUrl undefined) is never treated as
  // public -- it falls through to the 404 handler with no session attached,
  // same posture v3.3 got wrong by falling back to the raw (client-supplied,
  // query-string-carrying) req.url instead.
  if (routeUrl === undefined) return;
  if (PUBLIC_PATHS.has(routeUrl)) return;
  if (req.method === "GET" && !routeUrl.startsWith("/api/")) return; // SPA static assets
  // Workers keep their own auth (worker-auth.ts's authenticateWorker,
  // called explicitly inside each handler), never this middleware's user
  // session (MULTIUSER_PLAN.md §1.15: "the protocol... stay identical; only
  // authenticateWorker() changes") -- as of Stage 3 that function itself is
  // dual-mode (a real worker session first, WORKER_SHARED_TOKEN as a
  // fallback for not-yet-migrated Stage 1 workers), but either way it's a
  // worker credential, not this middleware's browser/user one. §2.3's own
  // PUBLIC_PATHS snippet predates these exact route paths and doesn't list
  // them, but gating them behind a USER session here would 401 every worker
  // heartbeat/queue-poll before authenticateWorker ever runs, breaking
  // worker communication outright the moment AUTH_ENABLED flips on.
  if (routeUrl.startsWith("/api/worker/")) return;
  if (WORKER_AUTHENTICATED_ROUTES.has(`${req.method} ${routeUrl}`)) return;

  const resolved = resolveAuthUser(req);
  if (!resolved) throw new UnauthorizedError("no session");

  (req as AuthenticatedRequest).user = resolved.user;
  (req as AuthenticatedRequest).session = resolved.session;

  // Sliding expiry (§2.5), throttled to at most one write per hour per
  // session inside touch() itself -- safe to call unconditionally here.
  repo.sessionRepo.touch(resolved.session);
}
