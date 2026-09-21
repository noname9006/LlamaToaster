import type { FastifyInstance, FastifyRequest } from "fastify";
import { repo } from "../db/repo.js";
import { resolveAuthUser, allUsersScope } from "../auth-middleware.js";
import type { AuthenticatedRequest, UserScope } from "../auth-middleware.js";
import { NotFoundError, ForbiddenError } from "../errors.js";
import { loadExportRows, formatResultsExport } from "./results.js";
import { getTestByIdHandler, getTestLogHandler, getBatchMembersHandler } from "./tests.js";
import { getProfilesHandler } from "./profiles.js";
import { getCurveHandler, getKneeHandler } from "./curves.js";
import { getProbeAttemptsHandler } from "./measurements.js";
import { exportTestHandler } from "./exchange.js";
import {
  isVramDiscrepancyPolicy,
  isValidProbeMaxLoads,
  type AdminTestFilters,
  type AppSettings,
} from "../../../shared/types.js";

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the ONLY cross-tenant read
// surface in this app, reachable exclusively from its own origin --
// ADMIN_PUBLIC_URL, e.g. "https://supervise.llamatoaster.com" -- never as a
// route inside the main SPA. Unset (the common case for a single-tenant or
// not-yet-admin-configured deployment) means this whole plugin 404s
// unconditionally; there is no way to reach it without deliberately
// configuring a second hostname. Exported for routes/auth.ts's OAuth
// callback, which branches on the same hostname (§5.1's own split).
export const ADMIN_HOSTNAME = process.env.ADMIN_PUBLIC_URL ? new URL(process.env.ADMIN_PUBLIC_URL).hostname : undefined;

export async function adminRoutes(app: FastifyInstance): Promise<void> {
  // Registered as onRequest, not preHandler, deliberately -- Fastify runs
  // every onRequest hook (regardless of which plugin registered it) before
  // any preHandler hook (regardless of which plugin registered IT), so this
  // is guaranteed to run before index.ts's global authMiddleware even though
  // that hook is registered on the PARENT app instance. Without that
  // ordering, a request to /api/admin/* on the MAIN hostname with no session
  // would hit authMiddleware first and 401 -- confirming an auth-gated
  // surface exists at this path before the hostname check ever ran, exactly
  // the probe this plugin exists to prevent (§5.1: "never confirm an admin
  // surface exists to probe further").
  //
  // Self-contained (resolveAuthUser directly, not a cast assuming
  // authMiddleware already populated req.user) for the same reason: this
  // hook must not depend on another plugin's hook having run first.
  app.addHook("onRequest", async (req) => {
    if (!ADMIN_HOSTNAME || req.hostname !== ADMIN_HOSTNAME) throw new NotFoundError();
    const authed = resolveAuthUser(req);
    if (!authed || !authed.user.isSuperadmin) throw new ForbiddenError("superadmin required");
    (req as AuthenticatedRequest).user = authed.user;
    (req as AuthenticatedRequest).session = authed.session;
  });

  app.get("/api/admin/stats", async () => repo.adminRepo.stats());

  const listTestsAdminHandler = async (req: FastifyRequest<{ Querystring: AdminTestFilters }>) => {
    return { runs: repo.adminRepo.listTests(req.query) };
  };
  app.get("/api/admin/tests", listTestsAdminHandler);
  app.get("/api/admin/runs", listTestsAdminHandler);

  // Same CSV/MD/JSON shapes as the main site's own GET /api/results/export
  // (see formatResultsExport's own doc comment) but unscoped -- every
  // tenant's results, by design, not just the caller's own.
  app.get<{ Querystring: { format?: string; tests?: string } }>(
    "/api/admin/results/export",
    async (req, reply) => {
      const format = (req.query.format ?? "json").toLowerCase();
      const runIds = req.query.tests ? req.query.tests.split(",").map((s) => s.trim()).filter(Boolean) : undefined;
      const rows = loadExportRows(undefined, runIds);
      const { contentType, filename, body } = formatResultsExport(rows, format);
      reply.header("content-type", contentType);
      reply.header("content-disposition", `attachment; filename="${filename}"`);
      return body;
    }
  );

  app.get("/api/admin/users", async () => ({ users: repo.adminRepo.listUsers() }));

  // --- Per-test observation ---------------------------------------------------
  // Read-only mirrors of every endpoint the main site's TestDetail page reads
  // (client/src/pages/TestDetail.tsx and the components it embeds), so the
  // console can render ANY user's test with that very page. Each is the main
  // route's own handler (a `(scope) => handler` factory, see UserScope in
  // auth-middleware.ts) built with allUsersScope, so the payloads are
  // identical to what the owner's browser receives -- there is no second
  // implementation to drift. Deliberately GET-only: nothing here can pause,
  // stop, trigger or delete another tenant's work.
  app.get("/api/admin/tests/:id", { logLevel: "silent" }, getTestByIdHandler(allUsersScope));
  app.get("/api/admin/tests/:id/batch-members", getBatchMembersHandler(allUsersScope));
  app.get("/api/admin/tests/:id/probe-attempts", getProbeAttemptsHandler(allUsersScope));
  app.get("/api/admin/tests/:id/profiles", getProfilesHandler(allUsersScope));
  app.get("/api/admin/tests/:id/knee", getKneeHandler(allUsersScope));
  app.get("/api/admin/tests/:id/log", getTestLogHandler(allUsersScope));
  app.get("/api/admin/tests/:id/export", exportTestHandler(allUsersScope));

  // Who owns the test and which machine ran it -- the page header's
  // "supervising X's test" line. Same row shape as the table's.
  app.get<{ Params: { id: string } }>("/api/admin/tests/:id/summary", async (req, reply) => {
    const summary = repo.adminRepo.getTestSummary(req.params.id);
    if (!summary) return reply.code(404).send({ error: "run not found" });
    return summary;
  });

  // The curve is keyed by model, not by test, so "all users" would fold every
  // tenant's measurements of that model together -- not what the test's owner
  // sees, and a real divergence for a test whose machine was later removed
  // (runs.worker_id goes NULL, so the panel can't narrow by worker). The
  // console passes the test's owner as ?user=, which scopes the curve exactly
  // as that user's own request would be; omitted (an owner-less legacy test)
  // it falls back to every user, the single-tenant behaviour.
  const curveOwnerScope: UserScope = (req) => {
    const user = (req.query as { user?: unknown }).user;
    return typeof user === "string" && user ? user : undefined;
  };
  app.get("/api/admin/models/:id/curve", getCurveHandler(curveOwnerScope));

  // The two lists TestDetail resolves labels from (the machine's hardware line
  // by run.worker_id; the MTP draft model's filename by id). The main site's
  // own GET /api/workers is caller-scoped; the console needs every tenant's.
  app.get("/api/admin/workers", async () => ({ workers: repo.workerRepo.listWorkers() }));
  app.get("/api/admin/models", async () => ({ models: repo.listModels() }));

  // The supervise dashboard's own platform-wide toggles (see shared/
  // types.ts's AppSettings doc comment) -- community benchmark sharing (and
  // whether users get their own say over it), self-service account
  // deletion, and the worker VRAM/probe knobs below. Same {}-partial-body
  // shape for all of them so the admin SPA can flip any one without
  // re-sending the others' values.
  app.get("/api/admin/settings", async (): Promise<AppSettings> => repo.appSettingsRepo.get());

  app.post<{ Body: Partial<AppSettings> }>("/api/admin/settings", async (req, reply) => {
    const body = req.body ?? {};
    if (body.communitySharingAllowed !== undefined && typeof body.communitySharingAllowed !== "boolean") {
      return reply.code(400).send({ error: "communitySharingAllowed must be a boolean" });
    }
    if (body.communityUserChoiceAllowed !== undefined && typeof body.communityUserChoiceAllowed !== "boolean") {
      return reply.code(400).send({ error: "communityUserChoiceAllowed must be a boolean" });
    }
    if (body.accountDeletionAllowed !== undefined && typeof body.accountDeletionAllowed !== "boolean") {
      return reply.code(400).send({ error: "accountDeletionAllowed must be a boolean" });
    }
    if (body.workerVramDiscrepancyPolicy !== undefined && !isVramDiscrepancyPolicy(body.workerVramDiscrepancyPolicy)) {
      return reply.code(400).send({ error: "workerVramDiscrepancyPolicy must be one of: warn, retry_once_then_fail, fail" });
    }
    if (body.probeMaxLoads !== undefined && !isValidProbeMaxLoads(body.probeMaxLoads)) {
      return reply.code(400).send({ error: "probeMaxLoads must be an integer in [1, 200]" });
    }
    if (body.communitySharingAllowed !== undefined) {
      repo.appSettingsRepo.setCommunitySharingAllowed(body.communitySharingAllowed);
    }
    if (body.communityUserChoiceAllowed !== undefined) {
      repo.appSettingsRepo.setCommunityUserChoiceAllowed(body.communityUserChoiceAllowed);
    }
    if (body.accountDeletionAllowed !== undefined) {
      repo.appSettingsRepo.setAccountDeletionAllowed(body.accountDeletionAllowed);
    }
    if (body.workerVramDiscrepancyPolicy !== undefined) {
      repo.appSettingsRepo.setWorkerVramDiscrepancyPolicy(body.workerVramDiscrepancyPolicy);
    }
    if (body.probeMaxLoads !== undefined) {
      repo.appSettingsRepo.setProbeMaxLoads(body.probeMaxLoads);
    }
    return repo.appSettingsRepo.get();
  });
}
