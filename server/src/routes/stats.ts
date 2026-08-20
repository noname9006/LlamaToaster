import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";

// The one intentionally cross-tenant number available on the main site
// (outside the admin-only surface in routes/admin.ts) -- see client/src/
// pages/Dashboard.tsx's own header comment for why a total user count is
// the deliberate exception to that page's otherwise per-account scoping.
// No auth gate of its own beyond whatever the global authMiddleware already
// applies when AUTH_ENABLED -- it's just a count, no per-user data.
export async function statsRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/stats", async () => ({ users: repo.statsRepo.userCount() }));
}
