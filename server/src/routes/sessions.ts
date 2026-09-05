import type { FastifyInstance, FastifyRequest } from "fastify";
import { repo } from "../db/repo.js";
import { hashToken } from "../session.js";
import { UnauthorizedError, NotFoundError } from "../errors.js";
import type { AuthenticatedRequest } from "../auth-middleware.js";
import type { SessionInfo, IdentityInfo } from "../../../shared/types.js";

// Dedicated Authorization-header read, deliberately NOT auth-middleware's
// extractToken -- a refresh token is never sent as the lt_session cookie
// (browser sessions don't use refresh tokens at all, §2.5), so falling back
// to a cookie here would silently accept the wrong token kind.
function bearerToken(req: FastifyRequest): string | undefined {
  const auth = req.headers.authorization;
  return auth?.startsWith("Bearer ") ? auth.slice("Bearer ".length) : undefined;
}

// Only registered when AUTH_ENABLED (server/src/index.ts, same gate as the
// auth middleware itself) -- every route here reads req.user/req.session,
// which only the middleware ever populates; registering these with the
// middleware off would crash on the first request. §2.5 wire responses use
// the plan's own snake_case (session_token/refresh_token/expires_at) since
// this is a worker-facing endpoint mirroring the existing worker wire
// convention (e.g. HeartbeatResponse), not a browser/client.ts-facing one.
export async function sessionRoutes(app: FastifyInstance): Promise<void> {
  // Worker refresh-token rotation (§2.5). Not gated by the generic auth
  // middleware in the normal sense -- it verifies its own presented bearer
  // token directly (a REFRESH token, a different token space than a normal
  // session's access token) via /api/auth/refresh's own PUBLIC_PATHS entry,
  // same reasoning as /api/worker/* having its own inline auth.
  app.post("/api/auth/refresh", async (req) => {
    const presented = bearerToken(req);
    if (!presented) throw new UnauthorizedError("no refresh token");
    const hash = hashToken(presented);

    // Replay detection: presenting the PREVIOUS refresh token means it was
    // captured and reused (a legitimate client always presents its CURRENT
    // one). Kill the whole session rather than issue new tokens to whoever
    // is holding the stale one.
    const replayed = repo.sessionRepo.getByPrevRefreshHash(hash);
    if (replayed) {
      repo.sessionRepo.revokeById(replayed.id);
      req.log.warn({ session: replayed.id, worker: replayed.workerId }, "refresh token replay -- session revoked");
      throw new UnauthorizedError("refresh token reuse detected");
    }

    const session = repo.sessionRepo.getByRefreshHash(hash);
    if (!session) throw new UnauthorizedError("refresh token expired");
    // Sliding expiry for worker refresh tokens, same as every other
    // authenticated route (auth-middleware.ts:170, worker-auth.ts:31): a
    // worker that's actively heartbeating/polling keeps its refresh token
    // alive indefinitely. Without this, a 90-day refresh token on a worker
    // that never sleeps would expire on its own and force a re-enrolment
    // every ~3 months -- the exact UX gap this worker hit.
    repo.sessionRepo.touch(session);
    if (session.expiresAt < Date.now()) throw new UnauthorizedError("refresh token expired");

    // sessions.id is STABLE across rotation -- a caller that stored it
    // earlier (Stage 3's workers.session_id) is never orphaned by a refresh.
    const rotated = repo.sessionRepo.rotate(session.id);
    return { session_token: rotated.token, refresh_token: rotated.refresh, expires_at: rotated.expiresAt };
  });

  app.get("/api/sessions", async (req): Promise<{ sessions: SessionInfo[] }> => {
    const { user, session } = req as AuthenticatedRequest;
    const sessions = repo.sessionRepo.listForUser(user.id);
    return {
      sessions: sessions.map((s) => ({
        id: s.id,
        label: s.label,
        isWorker: s.isWorker,
        createdAt: s.createdAt,
        lastSeenAt: s.lastSeenAt,
        current: s.id === session.id,
      })),
    };
  });

  app.delete<{ Params: { id: string } }>("/api/sessions/:id", async (req) => {
    const { user } = req as AuthenticatedRequest;
    // Ownership check via listForUser rather than a bare revokeById(params.id)
    // -- revokeById itself has no user scoping, so skipping this would let
    // any authenticated caller revoke ANY session by guessing its uuid.
    const owned = repo.sessionRepo.listForUser(user.id).find((s) => s.id === req.params.id);
    if (!owned) throw new NotFoundError("session not found");
    repo.sessionRepo.revokeById(owned.id);
    return { ok: true };
  });

  app.post("/api/sessions/revoke-all", async (req) => {
    const { user, session } = req as AuthenticatedRequest;
    repo.sessionRepo.revokeAllExcept(user.id, session.id);
    return { ok: true };
  });

  // Settings -> "Connected accounts" (§2.4/§2.8) -- every provider identity
  // linked to the caller's own account.
  app.get("/api/auth/identities", async (req): Promise<{ identities: IdentityInfo[] }> => {
    const { user } = req as AuthenticatedRequest;
    const identities = repo.userRepo.getIdentities(user.id);
    return {
      identities: identities.map((i) => ({ provider: i.provider, providerLogin: i.providerLogin, createdAt: i.createdAt })),
    };
  });

  // Settings' own toggle (§5.4) -- the consent flag community aggregates
  // (routes/ai.ts's community tools) are gated on. Returns the updated
  // AuthUser shape so the client can update its local state from the
  // response rather than re-fetching /api/auth/status. Turning it ON also
  // requires both of the operator's own gates, communitySharingAllowed and
  // communityUserChoiceAllowed (Settings hides the toggle client-side for
  // the same reason, but this is the enforcement a direct API call can't
  // bypass); turning it OFF is always allowed regardless of either gate,
  // since that can only ever narrow what's shared.
  app.post<{ Body: { enabled?: boolean } }>("/api/auth/share-benchmarks", async (req, reply) => {
    const { user } = req as AuthenticatedRequest;
    if (typeof req.body?.enabled !== "boolean") {
      return reply.code(400).send({ error: "enabled must be a boolean" });
    }
    if (req.body.enabled) {
      const settings = repo.appSettingsRepo.get();
      if (!settings.communitySharingAllowed || !settings.communityUserChoiceAllowed) {
        return reply.code(403).send({ error: "community benchmark sharing is currently disabled by the operator" });
      }
    }
    const updated = repo.userRepo.setShareBenchmarks(user.id, req.body.enabled);
    return { shareBenchmarks: updated.shareBenchmarks };
  });

  // Security checklist's own "Account deletion ships in v1" item -- self
  // service, irreversible, so it requires an explicit `{confirm: true}`
  // body rather than accepting a bare DELETE (a stray/scripted call with no
  // body must never delete anything). repo.userRepo.deleteAccount cascades
  // every owned row (sessions/identities/workers/runs/results/run_items);
  // the caller's OWN session is one of those, so its cookie is cleared here
  // too rather than leaving the browser holding a token for a row that no
  // longer exists.
  app.delete<{ Body: { confirm?: boolean } }>("/api/auth/account", async (req, reply) => {
    const { user } = req as AuthenticatedRequest;
    if (!repo.appSettingsRepo.get().accountDeletionAllowed) {
      return reply.code(403).send({ error: "account deletion is currently disabled by the operator" });
    }
    if (req.body?.confirm !== true) {
      return reply.code(400).send({ error: "confirm: true is required to delete an account" });
    }
    repo.userRepo.deleteAccount(user.id);
    reply.clearCookie("lt_session", { path: "/" });
    return { ok: true };
  });
}
