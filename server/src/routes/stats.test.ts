import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import type { AdminStats } from "../../../shared/types.js";

// Operator request (following on from Multi-user Stage 5, MULTIUSER_PLAN.md
// §5.2): GET /api/stats moved from a single unscoped user count to the full
// cross-tenant AdminStats shape -- see this route's own file and client/src/
// pages/Dashboard.tsx's header comment. No auth hook registered here, same
// reasoning as models.test.ts: the route has no gate of its own, so this
// exercises exactly what AUTH_ENABLED=off (or a public path) sees in
// production.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-stats-route-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { statsRoutes } = await import("./stats.js");

  app = Fastify({ logger: false });
  app.setErrorHandler((error: { statusCode?: number; message: string }, _req, reply) => {
    reply.code(error.statusCode ?? 500).send({ error: error.message });
  });
  await app.register(statsRoutes);
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

async function fetchStats(): Promise<AdminStats> {
  return (await (await fetch(`${baseUrl}/api/stats`)).json()) as AdminStats;
}

describe("GET /api/stats", () => {
  it("returns the full AdminStats shape, matching repo.adminRepo.stats() directly", async () => {
    const body = await fetchStats();
    expect(body).toEqual(repo.adminRepo.stats());
  });

  it("counts users across every account, not just one caller's own", async () => {
    const before = await fetchStats();

    repo.userRepo.upsertByIdentity("github", { providerUserId: "stats-route-user-a", login: "a", avatarUrl: null });
    repo.userRepo.upsertByIdentity("github", { providerUserId: "stats-route-user-b", login: "b", avatarUrl: null });

    const after = await fetchStats();
    expect(after.users).toBe(before.users + 2);
  });
});
