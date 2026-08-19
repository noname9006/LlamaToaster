import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";
import fastifyRateLimit from "@fastify/rate-limit";

// Regression coverage for the §2.6 wiring itself (a route declaring
// `config: { rateLimit: {...} }` with @fastify/rate-limit registered
// `{ global: false }`, MULTIUSER_PLAN.md §2.6) -- a throwaway app with a low
// max so the test doesn't need to fire 30+ requests, and a keyGenerator that
// mirrors userOrIpKeyGenerator's shape (id-or-ip) without needing a real
// session/DB round trip for this narrow check.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-rate-limit-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");

let app: FastifyInstance;
let baseUrl: string;

beforeAll(async () => {
  app = Fastify({ logger: false });
  await app.register(fastifyRateLimit, { global: false });

  app.get(
    "/ip-limited",
    { config: { rateLimit: { max: 2, timeWindow: "1 minute" } } },
    async () => ({ ok: true })
  );

  app.get<{ Querystring: { as?: string } }>(
    "/key-limited",
    {
      config: {
        rateLimit: { max: 2, timeWindow: "1 minute", keyGenerator: (req) => (req.query as { as?: string }).as ?? req.ip },
      },
    },
    async () => ({ ok: true })
  );

  app.get("/unlimited", async () => ({ ok: true }));

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

describe("rate limiting (§2.6 wiring)", () => {
  it("allows requests under the limit and 429s once it's exceeded", async () => {
    expect((await fetch(`${baseUrl}/ip-limited`)).status).toBe(200);
    expect((await fetch(`${baseUrl}/ip-limited`)).status).toBe(200);
    const third = await fetch(`${baseUrl}/ip-limited`);
    expect(third.status).toBe(429);
  });

  it("tracks separate keys (e.g. different users) independently, not lumped into one bucket", async () => {
    // "user-a" gets its own 2-request budget...
    expect((await fetch(`${baseUrl}/key-limited?as=user-a`)).status).toBe(200);
    expect((await fetch(`${baseUrl}/key-limited?as=user-a`)).status).toBe(200);
    expect((await fetch(`${baseUrl}/key-limited?as=user-a`)).status).toBe(429);

    // ...that exhausting it has no effect on "user-b"'s own budget.
    expect((await fetch(`${baseUrl}/key-limited?as=user-b`)).status).toBe(200);
  });

  it("a route with no rateLimit config is unaffected (global: false means opt-in only)", async () => {
    for (let i = 0; i < 5; i++) {
      expect((await fetch(`${baseUrl}/unlimited`)).status).toBe(200);
    }
  });
});
