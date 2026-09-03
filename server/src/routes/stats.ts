import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";

// The intentionally cross-tenant numbers available on the main site
// (outside the admin-only surface in routes/admin.ts) -- see client/src/
// pages/Dashboard.tsx's own header comment for why its whole stat-card row
// is platform-wide rather than scoped to the caller's own account. Reuses
// adminRepo.stats() itself (same query as GET /api/admin/stats) rather than
// a separate implementation -- these are the same numbers, just also
// readable off the main origin. No auth gate of its own beyond whatever the
// global authMiddleware already applies when AUTH_ENABLED -- it's all
// aggregate counts, no per-user breakdown, so exposing it here carries the
// same "just a count" reasoning that already justified the users total.
export async function statsRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/stats", async () => repo.adminRepo.stats());
}
