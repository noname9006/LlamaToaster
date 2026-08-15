import Fastify, { type FastifyError } from "fastify";
import fastifyStatic from "@fastify/static";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import { runsRoutes } from "./routes/runs.js";
import { modelsRoutes } from "./routes/models.js";
import { resultsRoutes } from "./routes/results.js";
import { workersRoutes } from "./routes/workers.js";
import { aiRoutes } from "./routes/ai.js";
import { getDb } from "./db/migrate.js";

// Backs the AI assistant's server-side config only (AI_API_KEY/AI_BASE_URL/
// AI_MODEL, see routes/ai.ts) -- every other env var here (PORT,
// WORKER_PORT, ...) keeps coming from the shell/systemd as before, so a
// missing .env (the common case -- it's optional and gitignored) is not an
// error. process.loadEnvFile is a built-in Node 20.12+/22 API, no dotenv
// dependency needed. Reading env vars happens inside route handlers/config.ts
// at request time, not at module-load time, so it doesn't matter that this
// runs after this file's own imports have already evaluated.
try {
  process.loadEnvFile();
} catch {
  /* no .env file at the repo root -- fine, AI_* vars just won't be set */
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const clientDist = join(__dirname, "..", "..", "client", "dist");

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.BIND_HOST ?? "127.0.0.1";

// Fastify's default logger (pino) already timestamps every line, but its
// default level is "info" -- LOG_LEVEL=debug surfaces the app.log.debug()
// calls added throughout routes/ for run/item/worker-action lifecycles,
// which otherwise leave no trace in the process's own log output.
const app = Fastify({
  logger: {
    level: process.env.LOG_LEVEL ?? "info",
    timestamp: () => `,"time":"${new Date().toISOString()}"`,
  },
});

app.register(fastifyStatic, {
  root: clientDist,
  prefix: "/",
});

app.register(runsRoutes);
app.register(modelsRoutes);
app.register(resultsRoutes);
app.register(workersRoutes);
app.register(aiRoutes);

// GET /api/runs/:id and the tick leg of POST /api/runs/:id/items/:idx are
// polled every ~1-2s by every open Runs/RunDetail tab and by the worker
// itself while a run is active -- both routes set logLevel: "silent" (see
// routes/runs.ts) so they no longer emit Fastify's default per-request log
// pair. This hook counts them instead and flushes one summary line per
// minute, so the *volume* of polling traffic is still visible without
// drowning out every other log line during a run.
const POLLING_ROUTES = [
  { method: "GET", url: "/api/runs/:id" },
  { method: "POST", url: "/api/runs/:id/items/:idx" },
] as const;

const pollingCounts = new Map<string, number>();

app.addHook("onResponse", async (request, reply) => {
  const { method, url } = request.routeOptions;
  const isPolling = POLLING_ROUTES.some((r) => r.method === method && r.url === url);
  if (!isPolling) return;
  const key = `${method} ${url} -> ${reply.statusCode}`;
  pollingCounts.set(key, (pollingCounts.get(key) ?? 0) + 1);
});

const pollingSummaryInterval = setInterval(() => {
  if (pollingCounts.size === 0) return;
  const total = [...pollingCounts.values()].reduce((sum, n) => sum + n, 0);
  app.log.info(
    { counts: Object.fromEntries(pollingCounts), total },
    "polling requests (last 60s)"
  );
  pollingCounts.clear();
}, 60_000);
pollingSummaryInterval.unref();

app.addHook("onClose", async () => {
  clearInterval(pollingSummaryInterval);
});

app.get("/health", async (_req, reply) => {
  try {
    getDb().prepare("SELECT 1").get();
    return { ok: true, db: true };
  } catch (err) {
    reply.code(500);
    return {
      ok: false,
      db: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
});

// The SPA (client/) owns all client-side routing (/, /models, /runs/:id, ...)
// -- any GET that isn't a static asset or an /api/* route falls through here
// and gets the same index.html, so the router can take over path matching in
// the browser. Non-GET or /api/* misses still get a real JSON 404.
//
// index.html is re-read from disk on every fallback request rather than
// cached at startup: a client-only deploy (new client/dist/ copied in without
// restarting this process) would otherwise leave this route serving an
// index.html whose hashed asset filenames no longer exist on disk, which
// breaks every path except "/" (served fresh by fastifyStatic) -- the app
// would 200-and-serve-HTML for the missing JS/CSS instead of loading them.
const indexHtmlPath = join(clientDist, "index.html");

app.setNotFoundHandler((request, reply) => {
  if (request.method === "GET" && !request.url.startsWith("/api/")) {
    try {
      const indexHtml = readFileSync(indexHtmlPath, "utf8");
      return reply.type("text/html").send(indexHtml);
    } catch {
      app.log.warn(
        `${indexHtmlPath} not found -- run "npm run build" to build the frontend before serving it`
      );
    }
  }
  reply.code(404);
  return { error: "not found" };
});

app.setErrorHandler((error: FastifyError, _req, reply) => {
  app.log.error(error);
  reply.code(error.statusCode ?? 500).send({ error: error.message });
});

async function start(): Promise<void> {
  try {
    await app.listen({ port: PORT, host: HOST });
    app.log.info(`VPS bound to ${HOST}:${PORT} (tailnet-only per plan)`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

const db = process.env.DB_PATH;
if (db) app.log.info(`Using DB at ${db}`);

start();
