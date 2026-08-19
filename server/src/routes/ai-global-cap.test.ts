import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify, { type FastifyInstance } from "fastify";

// Isolated from ai.test.ts's own suite deliberately: this file needs a LOW
// AI_GLOBAL_DAILY_CAP to be testable without seeding thousands of rows, but
// ai.test.ts's per-user tests seed usage rows that also count toward the
// same day's GLOBAL total (same ai_usage table) -- sharing one low cap
// across both files' tests would make them collide. Own temp DB, own cap.
const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-ai-global-cap-test-"));
process.env.DB_PATH = join(tmpDir, "test.db");
process.env.AI_API_KEY = "test-key";
process.env.AI_BASE_URL = "https://provider.invalid/v1";
process.env.AI_MODEL = "test-model";
process.env.AI_GLOBAL_DAILY_CAP = "5"; // read once at ai.js's module load, below

let app: FastifyInstance;
let baseUrl: string;
let repo: typeof import("../db/repo.js")["repo"];

const originalFetch = global.fetch;

beforeAll(async () => {
  ({ repo } = await import("../db/repo.js"));
  const { aiRoutes } = await import("./ai.js");

  global.fetch = (async (url: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const urlStr = typeof url === "string" ? url : url instanceof URL ? url.toString() : (url as Request).url;
    if (urlStr.startsWith("https://provider.invalid/")) {
      const body = `data: ${JSON.stringify({ choices: [{ delta: { content: "hi" } }] })}\n\ndata: [DONE]\n\n`;
      return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
    }
    // MUST forward init -- also how the test's own calls to the local test
    // server reach it, since global.fetch is mocked process-wide.
    return originalFetch(url as any, init);
  }) as typeof fetch;

  app = Fastify({ logger: false });
  await app.register(aiRoutes);
  await app.listen({ port: 0, host: "127.0.0.1" });
  const address = app.server.address();
  if (address === null || typeof address === "string") throw new Error("expected a bound TCP address");
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  global.fetch = originalFetch;
  await app.close();
  try {
    if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* file still open -- fine, it's in the OS temp dir */
  }
});

async function chat(headers: Record<string, string> = {}) {
  return fetch(`${baseUrl}/api/ai/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
  });
}

describe("POST /api/ai/chat -- global circuit breaker (§2.6)", () => {
  it("succeeds while under the cap, then 503s every caller once it's reached -- authenticated or not", async () => {
    const day = new Date().toISOString().slice(0, 10);
    expect(repo.aiUsageRepo.getGlobalDayCount(day)).toBe(0); // fresh DB, nothing seeded yet

    // 5 successive AUTHENTICATED calls (each records usage) exactly reach
    // the cap -- a fresh user each time so no per-user limit interferes.
    for (let i = 0; i < 5; i++) {
      const user = repo.userRepo.upsertByIdentity("github", { providerUserId: `cap-${i}`, login: `cap-${i}`, avatarUrl: null });
      const { token } = repo.sessionRepo.create(user.id, { label: `cap-${i}` });
      const res = await chat({ authorization: `Bearer ${token}` });
      expect(res.status).toBe(200);
    }
    expect(repo.aiUsageRepo.getGlobalDayCount(day)).toBe(5);

    // The cap is now exhausted -- every NEW caller is blocked, regardless of
    // whether they have a session at all.
    const anonRes = await chat();
    expect(anonRes.status).toBe(503);
    expect((await anonRes.json()) as { error: string }).toMatchObject({ error: expect.stringContaining("daily budget") });

    const freshUser = repo.userRepo.upsertByIdentity("github", { providerUserId: "cap-victim", login: "victim", avatarUrl: null });
    const { token } = repo.sessionRepo.create(freshUser.id, { label: "victim" });
    const authedRes = await chat({ authorization: `Bearer ${token}` });
    expect(authedRes.status).toBe(503);

    // The blocked calls must not have been recorded -- still exactly 5.
    expect(repo.aiUsageRepo.getGlobalDayCount(day)).toBe(5);
  });
});
