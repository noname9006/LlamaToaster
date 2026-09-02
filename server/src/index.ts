import Fastify, { type FastifyError } from "fastify";
import fastifyStatic from "@fastify/static";
import fastifyCookie from "@fastify/cookie";
import fastifyRateLimit from "@fastify/rate-limit";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync, existsSync } from "node:fs";
import { testsRoutes } from "./routes/tests.js";
import { profilesRoutes } from "./routes/profiles.js";
import { measurementRoutes } from "./routes/measurements.js";
import { curveRoutes } from "./routes/curves.js";
import { comparisonRoutes } from "./routes/comparisons.js";
import { exchangeRoutes } from "./routes/exchange.js";
import { modelsRoutes } from "./routes/models.js";
import { resultsRoutes } from "./routes/results.js";
import { workersRoutes } from "./routes/workers.js";
import { aiRoutes } from "./routes/ai.js";
import { queueRoutes } from "./routes/queue.js";
import { authRoutes } from "./routes/auth.js";
import { sessionRoutes } from "./routes/sessions.js";
import { deviceRoutes, deviceApprovalRoutes } from "./routes/device.js";
import { adminRoutes } from "./routes/admin.js";
import { statsRoutes } from "./routes/stats.js";
import { getDb } from "./db/migrate.js";
import { runMaintenanceSweep, REAP_INTERVAL_MS } from "./reaper.js";
import { authMiddleware } from "./auth-middleware.js";
import { startHfIndexService, stopHfIndexService } from "./hf-index.js";

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
const adminDist = join(__dirname, "..", "..", "admin", "dist");

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1) -- same derivation as
// routes/admin.ts's own ADMIN_HOSTNAME (re-declared here rather than
// imported: this is purely about which STATIC ROOT to serve, an unrelated
// concern from that file's auth/hostname gating, and importing across just
// for one constant isn't worth the coupling).
const ADMIN_HOSTNAME = process.env.ADMIN_PUBLIC_URL ? new URL(process.env.ADMIN_PUBLIC_URL).hostname : undefined;

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
  // MULTIUSER_PLAN.md §6.1: once nginx sits in front of this process, req.ip
  // would otherwise be nginx's own loopback address for every request,
  // silently collapsing §2.6's per-IP rate limits (auth, device-flow) into
  // one shared global limit. Scoped to loopback specifically -- not a blanket
  // `true` -- so a direct connection from anywhere else (e.g. another
  // tailnet peer, in the still-supported Tailscale-only deployment mode)
  // can't spoof X-Forwarded-For to fake a different client identity; only a
  // request that already came from 127.0.0.1/::1 (nginx, by construction --
  // see BIND_HOST) gets its forwarded headers trusted at all.
  trustProxy: ["127.0.0.1", "::1"],
});

app.register(fastifyStatic, {
  root: clientDist,
  prefix: "/",
});

// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): a SECOND static root, active
// only for requests whose Host matches the admin hostname -- Fastify (via
// find-my-way's built-in host-constraint strategy) tries constrained routes
// before the unconstrained one above, so admin. gets admin/dist and every
// other hostname keeps getting client/dist, with no explicit branching
// logic needed here. decorateReply: false avoids a "reply.sendFile already
// exists" clash with the registration above -- nothing here calls it
// directly. Skipped entirely (with a warning, not a crash) if
// ADMIN_PUBLIC_URL is set but `npm run build --prefix admin` hasn't been run
// yet -- registering @fastify/static against a directory that doesn't exist
// throws at startup otherwise, and this shouldn't take down the main site.
if (ADMIN_HOSTNAME) {
  if (existsSync(adminDist)) {
    app.register(fastifyStatic, {
      root: adminDist,
      prefix: "/",
      decorateReply: false,
      constraints: { host: ADMIN_HOSTNAME },
    });
  } else {
    app.log.warn(
      `ADMIN_PUBLIC_URL is set but ${adminDist} doesn't exist -- run "npm run build --prefix admin" first`
    );
  }
}

// Harmless to register unconditionally -- it only parses the Cookie header
// into req.cookies, which future auth cookies (lt_session, oauth_state) need
// regardless of whether AUTH_ENABLED is on. secret is only required for
// *signed* cookies, which nothing here uses.
app.register(fastifyCookie);

// global:false -- opt in per route via `config: { rateLimit: {...} }`
// (MULTIUSER_PLAN.md §2.6) rather than a blanket limit on every route,
// most of which (polled GETs, worker heartbeats) see far higher legitimate
// traffic than any of the specifically-budgeted routes below.
app.register(fastifyRateLimit, { global: false });

app.register(testsRoutes);
// BENCHMARKING_PLAN_V8.md M3 -- scored profile cards, pure post-processing
// over stored results (changing goals never re-measures).
app.register(profilesRoutes);
// N2/N4 -- worker-authed, idempotent probe and quality ingestion.
app.register(measurementRoutes);
// N1/N5 -- context curves and the concurrency knee, both derived on read.
app.register(curveRoutes);
// N3 -- model-vs-model comparison view with its blocking fairness checks.
app.register(comparisonRoutes);
// N7 -- export bundles that carry their own methods section, and per-row
// validated import.
app.register(exchangeRoutes);
app.register(modelsRoutes);
app.register(resultsRoutes);
app.register(workersRoutes);
app.register(aiRoutes);
app.register(queueRoutes);
app.register(statsRoutes);
// Registered unconditionally (not gated on AUTH_ENABLED) -- GET
// /api/auth/status is the SPA's own boot check and needs to answer even when
// auth is off (MULTIUSER_PLAN.md §2.3's independent-deploy split), and the
// /auth/:provider* routes are harmless without it (nothing requires being
// logged in yet, so completing a login just creates an unused session).
app.register(authRoutes);
// Same reasoning as authRoutes above -- POST /api/device/start|token are
// device-initiated and harmless without AUTH_ENABLED (a Stage-1-only
// deployment's workers never call these at all, they use the old
// WORKER_SHARED_TOKEN path instead; see routes/device.ts's own header comment).
app.register(deviceRoutes);
// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): also registered
// unconditionally -- adminRoutes is entirely self-gated (its own onRequest
// hook checks ADMIN_HOSTNAME + isSuperadmin via resolveAuthUser directly,
// never assuming the AUTH_ENABLED-gated authMiddleware below has run) and
// 404s everything if ADMIN_PUBLIC_URL isn't configured, same "harmless
// without AUTH_ENABLED" posture as authRoutes/deviceRoutes above.
app.register(adminRoutes);

// Multi-user Stage 2 (MULTIUSER_PLAN.md §2.3): registered only when
// AUTH_ENABLED=true, so Stage 1 (still shared-secret worker auth, no user
// auth at all) and Stage 2 can be deployed independently -- flipping this on
// without a configured OAuth provider/PUBLIC_URL would lock every route
// behind a login flow nothing can complete yet.
const AUTH_ENABLED = process.env.AUTH_ENABLED === "true";
if (AUTH_ENABLED) {
  app.addHook("preHandler", authMiddleware);
  // sessionRoutes (GET/DELETE /api/sessions, POST /api/sessions/revoke-all)
  // and deviceApprovalRoutes (GET /api/device/status, POST /api/device/approve)
  // all read req.user, which only authMiddleware ever populates -- registering
  // them with auth off would crash on the first request.
  app.register(sessionRoutes);
  app.register(deviceApprovalRoutes);
  app.log.info("AUTH_ENABLED=true -- user auth middleware active");
}

// GET /api/runs/:id and the tick leg of POST /api/runs/:id/items/:idx are
// polled every ~1-2s by every open Tests/TestDetail tab and by the worker
// itself while a run is active -- both routes set logLevel: "silent" (see
// routes/tests.ts) so they no longer emit Fastify's default per-request log
// pair. This hook counts them instead and flushes one summary line per
// minute, so the *volume* of polling traffic is still visible without
// drowning out every other log line during a run.
// Each route is registered under both its current path and the legacy
// /api/runs/... one it replaces (see routes/tests.ts) -- both patterns are
// listed here so polling volume is suppressed/counted the same either way
// during the rollout. Drop the /api/runs/... entries once every worker and
// browser tab is confirmed off the legacy path.
const POLLING_ROUTES = [
  { method: "GET", url: "/api/tests/:id" },
  { method: "GET", url: "/api/runs/:id" },
  { method: "POST", url: "/api/tests/:id/items/:idx" },
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

const reapInterval = setInterval(() => runMaintenanceSweep(app.log), REAP_INTERVAL_MS);
reapInterval.unref();

// Hugging Face GGUF index service -- see server/src/hf-index.ts for the
// full indexing flow. Started after the server is listening so a slow
// initial scan doesn't delay the first request; stopped on shutdown
// alongside the other background intervals.
startHfIndexService();

app.addHook("onClose", async () => {
  clearInterval(reapInterval);
  stopHfIndexService();
});

// Public (MULTIUSER_PLAN.md §6.2): a bare {ok, db} only -- this used to also
// echo err.message straight into the response, which is a real information
// leak once the app is Internet-facing (raw DB error text, e.g. a file path,
// visible to anyone). The detail still needs to reach someone -- it's logged
// server-side instead.
app.get("/health", async (_req, reply) => {
  try {
    getDb().prepare("SELECT 1").get();
    return { ok: true, db: true };
  } catch (err) {
    app.log.error({ err }, "/health: DB check failed");
    reply.code(500);
    return { ok: false, db: false };
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
// Multi-user Stage 5 (MULTIUSER_PLAN.md §5.1): the admin SPA's own
// client-side routes (a stats page, the runs table, ...) need the exact same
// index.html fallback treatment, just from admin/dist instead.
const adminIndexHtmlPath = join(adminDist, "index.html");

app.setNotFoundHandler((request, reply) => {
  if (request.method === "GET" && !request.url.startsWith("/api/")) {
    const onAdminHost = ADMIN_HOSTNAME !== undefined && request.hostname === ADMIN_HOSTNAME;
    const path = onAdminHost ? adminIndexHtmlPath : indexHtmlPath;
    try {
      const indexHtml = readFileSync(path, "utf8");
      return reply.type("text/html").send(indexHtml);
    } catch {
      app.log.warn(`${path} not found -- run "npm run build" to build the frontend before serving it`);
    }
  }
  reply.code(404);
  return { error: "not found" };
});

app.setErrorHandler((error: FastifyError, _req, reply) => {
  app.log.error(error);
  const statusCode = error.statusCode ?? 500;
  // Never leak an internal error's message to the client -- it can contain
  // file paths, SQL, or other implementation detail. Only errors we
  // deliberately threw as a 4xx (see errors.ts) have a message meant to be
  // shown to the caller.
  const message = statusCode >= 500 ? "internal server error" : error.message;
  reply.code(statusCode).send({ error: message });
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
