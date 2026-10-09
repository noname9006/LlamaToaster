import { describe, expect, it } from "vitest";
import Fastify from "fastify";
import { NotFoundError, UnauthorizedError } from "./errors.js";
import { appErrorHandler } from "./error-handler.js";

// Builds an app with the production error handler and a log stream that
// captures every line, so each test can assert on the level and stack.
function buildApp() {
  const lines: Array<Record<string, unknown>> = [];
  const stream = { write: (chunk: string) => void lines.push(JSON.parse(chunk)) };
  const app = Fastify({ logger: { level: "info", stream } });
  app.setErrorHandler(appErrorHandler);
  app.get("/api/workers", async () => {
    throw new UnauthorizedError("no session");
  });
  app.get("/api/missing", async () => {
    throw new NotFoundError();
  });
  app.get("/api/boom", async () => {
    throw new Error("secret detail: /var/lib/db.sqlite");
  });
  return { app, lines };
}

describe("appErrorHandler", () => {
  it("answers an unauthenticated request with 401 and the caller-facing message", async () => {
    const { app } = buildApp();
    const res = await app.inject({ method: "GET", url: "/api/workers" });
    expect(res.statusCode).toBe(401);
    expect(res.json()).toEqual({ error: "no session" });
    await app.close();
  });

  it("logs an expected 401 at info with no stack trace and no error-level line", async () => {
    const { app, lines } = buildApp();
    await app.inject({ method: "GET", url: "/api/workers" });
    await app.close();

    const rejected = lines.find((l) => l.msg === "no session");
    expect(rejected).toBeDefined();
    expect(rejected?.level).toBe(30); // pino info
    expect(rejected?.statusCode).toBe(401);
    expect(JSON.stringify(lines)).not.toContain('"stack"');
    expect(lines.some((l) => (l.level as number) >= 50)).toBe(false);
  });

  it("logs an expected 404 at info, not error", async () => {
    const { app, lines } = buildApp();
    const res = await app.inject({ method: "GET", url: "/api/missing" });
    await app.close();

    expect(res.statusCode).toBe(404);
    expect(lines.some((l) => (l.level as number) >= 50)).toBe(false);
  });

  it("answers an unexpected 500 generically and does not leak the internal message", async () => {
    const { app, lines } = buildApp();
    const res = await app.inject({ method: "GET", url: "/api/boom" });
    await app.close();

    expect(res.statusCode).toBe(500);
    expect(res.json()).toEqual({ error: "internal server error" });
    expect(res.body).not.toContain("secret detail");
    // A real 5xx still gets the full error at error level, stack included.
    const failure = lines.find((l) => (l.level as number) >= 50);
    expect(failure).toBeDefined();
    expect(JSON.stringify(failure)).toContain('"stack"');
  });
});
