import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import type { AuthenticatedRequest } from "../auth-middleware.js";
import { NotFoundError, ForbiddenError } from "../errors.js";
import { loadExportRows, formatResultsExport } from "./results.js";
import { isVramDiscrepancyPolicy, type AdminRunFilters, type AppSettings } from "../../../shared/types.js";

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

  app.get<{ Querystring: AdminRunFilters }>("/api/admin/runs", async (req) => {
    return { runs: repo.adminRepo.listRuns(req.query) };
  });

  // Same CSV/MD/JSON shapes as the main site's own GET /api/results/export
  // (see formatResultsExport's own doc comment) but unscoped -- every
  // tenant's results, by design, not just the caller's own.
  app.get<{ Querystring: { format?: string; runs?: string } }>(
    "/api/admin/results/export",
    async (req, reply) => {
      const format = (req.query.format ?? "json").toLowerCase();
      const runIds = req.query.runs ? req.query.runs.split(",").map((s) => s.trim()).filter(Boolean) : undefined;
      const rows = loadExportRows(undefined, runIds);
      const { contentType, filename, body } = formatResultsExport(rows, format);
      reply.header("content-type", contentType);
      reply.header("content-disposition", `attachment; filename="${filename}"`);
      return body;
    }
  );

  app.get("/api/admin/users", async () => ({ users: repo.adminRepo.listUsers() }));

  // The supervise dashboard's own two platform-wide toggles (see shared/
  // types.ts's AppSettings doc comment) -- community benchmark sharing and
  // self-service account deletion. Same {}-partial-body shape for both so
  // the admin SPA can flip either without re-sending the other's value.
  app.get("/api/admin/settings", async (): Promise<AppSettings> => repo.appSettingsRepo.get());

  app.post<{ Body: Partial<AppSettings> }>("/api/admin/settings", async (req, reply) => {
    const body = req.body ?? {};
    if (body.communitySharingAllowed !== undefined && typeof body.communitySharingAllowed !== "boolean") {
      return reply.code(400).send({ error: "communitySharingAllowed must be a boolean" });
    }
    if (body.accountDeletionAllowed !== undefined && typeof body.accountDeletionAllowed !== "boolean") {
      return reply.code(400).send({ error: "accountDeletionAllowed must be a boolean" });
    }
    if (body.workerVramDiscrepancyPolicy !== undefined && !isVramDiscrepancyPolicy(body.workerVramDiscrepancyPolicy)) {
      return reply.code(400).send({ error: "workerVramDiscrepancyPolicy must be one of: warn, retry_once_then_fail, fail" });
    }
    if (body.communitySharingAllowed !== undefined) {
      repo.appSettingsRepo.setCommunitySharingAllowed(body.communitySharingAllowed);
    }
    if (body.accountDeletionAllowed !== undefined) {
      repo.appSettingsRepo.setAccountDeletionAllowed(body.accountDeletionAllowed);
    }
    if (body.workerVramDiscrepancyPolicy !== undefined) {
      repo.appSettingsRepo.setWorkerVramDiscrepancyPolicy(body.workerVramDiscrepancyPolicy);
    }
    return repo.appSettingsRepo.get();
  });
}
