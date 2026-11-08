// The OpenAI-compatible endpoint external agents talk to. llama-server itself sits
// on a private port behind this, because exposing it directly would mean no
// per-request timings (they exist only in the response), no strict model names
// (it answers `gpt-4o` with whatever is loaded), CORS * by default, and /slots,
// /props and /lora-adapters on the public port. See docs/plans/MODEL_DAEMON_PLAN.md.
//
// node:http only: a worker install has no web framework and should not gain one.

import { createServer, request as httpRequest, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import type { GatewaySettings, SamplingEndpoint, SamplingSettings } from "../../../shared/daemonSettings.js";
import { applySampling, isLoopbackHost } from "../../../shared/daemonSettings.js";
import type { DaemonEndpoint, DaemonUsageRow } from "../../../shared/daemonUsage.js";
import { findRemoteMedia } from "./mediaUrls.js";
import { UsageTap } from "./usageTap.js";

export const MAX_BODY_BYTES = 64 * 1024 * 1024;
export const MAX_IN_FLIGHT = 128;
// Request bodies held in memory at once. A single large one still fits; many at the
// maximum would otherwise run a worker machine out of memory.
export const MAX_BUFFERED_BYTES = 256 * 1024 * 1024;

interface Route {
  endpoint: DaemonEndpoint | null;
  /** Which request fields sampling maps onto. Null: no sampling applies. */
  sampling: SamplingEndpoint | null;
  /** Worth a usage row. */
  record: boolean;
}

const ROUTES: Record<string, Route> = {
  "/v1/chat/completions": { endpoint: "chat", sampling: "chat", record: true },
  "/v1/completions": { endpoint: "completions", sampling: "completions", record: true },
  "/v1/responses": { endpoint: "responses", sampling: "responses", record: true },
  "/v1/messages": { endpoint: "messages", sampling: "messages", record: true },
  "/v1/messages/count_tokens": { endpoint: null, sampling: null, record: false },
  "/v1/embeddings": { endpoint: "embeddings", sampling: null, record: true },
  "/v1/rerank": { endpoint: "rerank", sampling: null, record: true },
};

export interface GatewayUpstream {
  port: number;
  key: string;
  alias: string;
  modelId: string;
  loadHash: string;
  createdAt: number;
}

export interface GatewayDeps {
  upstream(): GatewayUpstream | null;
  /** Why there is no upstream, and when to try again. */
  unavailable(): { message: string; retryAfterSeconds: number };
  sampling(): SamplingSettings | null;
  keyHashes(): string[];
  watched(): boolean;
  onUsage(row: DaemonUsageRow): void;
  log: { info(msg: string): void; warn(msg: string): void };
  /** Overrides MAX_BUFFERED_BYTES. */
  maxBufferedBytes?: number;
}

export function hashApiKey(key: string): string {
  return createHash("sha256").update(key).digest("hex");
}

function keyMatches(presented: string, hashes: string[]): boolean {
  const candidate = Buffer.from(hashApiKey(presented), "hex");
  let matched = false;
  for (const hash of hashes) {
    const stored = Buffer.from(hash, "hex");
    if (stored.length === candidate.length && timingSafeEqual(stored, candidate)) matched = true;
  }
  return matched;
}

function presentedKey(req: IncomingMessage): string | null {
  const auth = req.headers.authorization;
  if (auth?.startsWith("Bearer ")) return auth.slice("Bearer ".length).trim();
  // Anthropic clients send this instead.
  const anthropic = req.headers["x-api-key"];
  return typeof anthropic === "string" ? anthropic : null;
}

function hostnameOf(hostHeader: string | undefined): string {
  if (!hostHeader) return "";
  if (hostHeader.startsWith("[")) return hostHeader.slice(0, hostHeader.indexOf("]") + 1);
  return hostHeader.split(":")[0].toLowerCase();
}

const LOOPBACK_NAMES = new Set(["127.0.0.1", "localhost", "[::1]"]);

// The OpenAI SDKs read error.message and error.type from this shape.
function sendError(res: ServerResponse, status: number, type: string, message: string, headers: Record<string, string> = {}): void {
  if (res.headersSent) {
    res.destroy();
    return;
  }
  const body = JSON.stringify({ error: { message, type, code: status } });
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body), ...headers });
  res.end(body);
}

// "too_large": past `limit`. "busy": `take` refused, so memory is what's short.
function readBody(req: IncomingMessage, limit: number, take: (bytes: number) => boolean): Promise<Buffer | "too_large" | "busy"> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let done = false;
    req.on("data", (chunk: Buffer) => {
      if (done) return;
      size += chunk.length;
      if (size > limit) {
        done = true;
        resolve("too_large");
        return;
      }
      if (!take(chunk.length)) {
        done = true;
        resolve("busy");
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (done) return;
      done = true;
      resolve(Buffer.concat(chunks));
    });
    // Also how a client that goes away mid-upload gives up its place and its memory.
    req.on("error", reject);
  });
}

export class DaemonGateway {
  private server: Server | null = null;
  private settings: GatewaySettings | null = null;
  private lastError: string | null = null;
  private inFlightCount = 0;
  private readingCount = 0;
  private bufferedBytes = 0;
  private totalRequests = 0;
  private readonly active = new Set<() => void>();
  // One per forwarded request, for the most that ever ran alongside it.
  private readonly overlaps = new Set<{ peak: number }>();
  private idleWaiters: (() => void)[] = [];
  // apply and stop run one at a time. Two binds at once made the loser clear
  // this.server while the winner listened on, with nothing left to close it.
  private listenerQueue: Promise<unknown> = Promise.resolve();

  constructor(private readonly deps: GatewayDeps) {}

  private inListenerQueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.listenerQueue.then(fn, fn);
    this.listenerQueue = run.catch(() => undefined);
    return run;
  }

  get inFlight(): number {
    return this.inFlightCount;
  }

  /** Bytes of request bodies currently held. */
  get bufferedBodyBytes(): number {
    return this.bufferedBytes;
  }

  get requestsTotal(): number {
    return this.totalRequests;
  }

  status(): { listening: boolean; host: string; port: number; error: string | null } {
    return {
      listening: this.server?.listening === true,
      host: this.settings?.host ?? "",
      port: this.settings?.port ?? 0,
      error: this.lastError,
    };
  }

  /** Listens per `settings`, rebinding only when the address changed. Refuses a
   * non-loopback bind with no keys: the server checks this too, so this is the
   * backstop for a stale config or a key revoked after the fact. */
  apply(settings: GatewaySettings): Promise<void> {
    return this.inListenerQueue(() => this.applyNow(settings));
  }

  private async applyNow(settings: GatewaySettings): Promise<void> {
    const sameAddress = this.settings?.host === settings.host && this.settings?.port === settings.port;
    this.settings = settings;
    if (!isLoopbackHost(settings.host) && this.deps.keyHashes().length === 0) {
      this.lastError = "Refusing to listen beyond this machine without an API key.";
      this.deps.log.warn(`daemon gateway: ${this.lastError}`);
      await this.closeListener();
      return;
    }
    if (sameAddress && this.server?.listening) {
      this.lastError = null;
      return;
    }
    await this.closeListener();
    const server = createServer((req, res) => void this.handle(req, res));
    // Node's default requestTimeout stays: it limits receiving a request, never how
    // long a generation runs, and it is what stops a body trickling in forever.
    server.keepAliveTimeout = 65_000;
    await new Promise<void>((resolve) => {
      const onError = (err: NodeJS.ErrnoException) => {
        this.lastError =
          err.code === "EADDRINUSE"
            ? `Port ${settings.port} is already in use.`
            : err.code === "EADDRNOTAVAIL"
              ? `${settings.host} is not an address on this machine.`
              : `Could not listen: ${err.message}`;
        this.deps.log.warn(`daemon gateway: ${this.lastError}`);
        this.server = null;
        resolve();
      };
      server.once("error", onError);
      server.listen(settings.port, settings.host, () => {
        server.off("error", onError);
        server.on("error", (err) => this.deps.log.warn(`daemon gateway: ${err.message}`));
        this.server = server;
        this.lastError = null;
        this.deps.log.info(`daemon gateway listening on http://${settings.host}:${settings.port}/v1`);
        resolve();
      });
    });
  }

  stop(): Promise<void> {
    return this.inListenerQueue(() => this.closeListener());
  }

  private async closeListener(): Promise<void> {
    const server = this.server;
    if (!server) return;
    this.server = null;
    // Stops accepting at once; requests already running finish on their sockets.
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeIdleConnections();
      setTimeout(resolve, 500).unref();
    });
  }

  /** Resolves once nothing is in flight, or after `ms`, when the rest are cut off. */
  async drain(ms: number): Promise<void> {
    if (this.inFlightCount > 0) {
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, ms);
        timer.unref();
        this.idleWaiters.push(() => {
          clearTimeout(timer);
          resolve();
        });
      });
    }
    for (const abort of [...this.active]) abort();
  }

  /** Same-origin requests carry no Origin. A browser page on any other site does. */
  private checkBrowser(req: IncomingMessage, res: ServerResponse): boolean {
    const origin = req.headers.origin;
    if (!origin) return true;
    const allowed = this.settings?.cors_origins ?? [];
    if (!allowed.includes(origin)) {
      sendError(res, 403, "forbidden", "This origin is not on the endpoint's allowed list.");
      return false;
    }
    res.setHeader("access-control-allow-origin", origin);
    res.setHeader("vary", "Origin");
    return true;
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    try {
      const path = (req.url ?? "/").split("?")[0];
      // A bind to loopback answers only to loopback names, which is what stops a
      // public hostname being pointed here by DNS rebinding.
      if (this.settings && isLoopbackHost(this.settings.host) && !LOOPBACK_NAMES.has(hostnameOf(req.headers.host))) {
        sendError(res, 403, "forbidden", "This endpoint only answers to localhost.");
        return;
      }
      if (!this.checkBrowser(req, res)) return;
      if (req.method === "OPTIONS") {
        res.setHeader("access-control-allow-methods", "GET, POST, OPTIONS");
        res.setHeader("access-control-allow-headers", "authorization, content-type, x-api-key, anthropic-version, anthropic-beta");
        res.writeHead(204);
        res.end();
        return;
      }
      if (req.method === "GET" && (path === "/health" || path === "/v1/health")) {
        const up = this.deps.upstream();
        res.writeHead(up ? 200 : 503, { "content-type": "application/json" });
        res.end(JSON.stringify({ status: up ? "ok" : "unavailable" }));
        return;
      }

      const keys = this.deps.keyHashes();
      if (keys.length > 0) {
        const key = presentedKey(req);
        if (!key || !keyMatches(key, keys)) {
          sendError(res, 401, "authentication_error", "Missing or invalid API key.", { "www-authenticate": "Bearer" });
          return;
        }
      }

      if (req.method === "GET" && (path === "/v1/models" || path.startsWith("/v1/models/"))) {
        this.models(res, path);
        return;
      }
      const route = Object.hasOwn(ROUTES, path) ? ROUTES[path] : undefined;
      if (!route) {
        sendError(res, 404, "not_found_error", `No such endpoint: ${path.slice(0, 200)}`);
        return;
      }
      if (req.method !== "POST") {
        sendError(res, 405, "invalid_request_error", "Use POST.", { allow: "POST" });
        return;
      }
      // A cross-site form post cannot set this, so it cannot run the model either.
      if (!(req.headers["content-type"] ?? "").toLowerCase().includes("application/json")) {
        sendError(res, 415, "invalid_request_error", "Content-Type must be application/json.");
        return;
      }
      await this.inference(req, res, path, route);
    } catch (err) {
      this.deps.log.warn(`daemon gateway: ${err instanceof Error ? err.message : String(err)}`);
      sendError(res, 500, "server_error", "The gateway hit an unexpected error.");
    }
  }

  private models(res: ServerResponse, path: string): void {
    // Answered here, not by llama-server, so a listing can never wake a sleeping model.
    const up = this.deps.upstream();
    const entry = up
      ? { id: up.alias, object: "model", created: Math.floor(up.createdAt / 1000), owned_by: "llamatoaster" }
      : null;
    if (path !== "/v1/models") {
      let asked: string;
      try {
        asked = decodeURIComponent(path.slice("/v1/models/".length));
      } catch {
        sendError(res, 404, "model_not_found", "No loaded model has that name.");
        return;
      }
      if (!entry || asked !== entry.id) {
        sendError(res, 404, "model_not_found", `No loaded model named ${asked.slice(0, 200)}.`);
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(entry));
      return;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: entry ? [entry] : [] }));
  }

  private async inference(req: IncomingMessage, res: ServerResponse, path: string, route: Route): Promise<void> {
    const up = this.deps.upstream();
    if (!up) {
      const why = this.deps.unavailable();
      req.resume();
      sendError(res, 503, "model_not_loaded", why.message, { "retry-after": String(why.retryAfterSeconds) });
      return;
    }
    // Counts requests still uploading too: they are the ones holding memory, and each
    // waits here unseen by inFlightCount until its body has arrived.
    if (this.inFlightCount + this.readingCount >= MAX_IN_FLIGHT) {
      req.resume();
      sendError(res, 503, "overloaded", "Too many requests in flight.", { "retry-after": "5" });
      return;
    }
    const maxBuffered = this.deps.maxBufferedBytes ?? MAX_BUFFERED_BYTES;
    let held = 0;
    const release = () => {
      this.bufferedBytes -= held;
      held = 0;
    };
    let handedOff = false;
    this.readingCount++;
    try {
      let raw: Buffer | "too_large" | "busy";
      try {
        raw = await readBody(req, MAX_BODY_BYTES, (bytes) => {
          if (this.bufferedBytes + bytes > maxBuffered) return false;
          this.bufferedBytes += bytes;
          held += bytes;
          return true;
        });
      } finally {
        this.readingCount--;
      }
      if (raw === "too_large") {
        sendError(res, 413, "invalid_request_error", `Request body is over ${MAX_BODY_BYTES / (1024 * 1024)} MiB.`, { connection: "close" });
        return;
      }
      if (raw === "busy") {
        sendError(res, 503, "overloaded", "Too many large requests at once.", { "retry-after": "5", connection: "close" });
        return;
      }
      let parsed: Record<string, unknown>;
      try {
        const value: unknown = JSON.parse(raw.toString("utf8"));
        if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("not an object");
        parsed = value as Record<string, unknown>;
      } catch {
        sendError(res, 400, "invalid_request_error", "The request body is not a JSON object.");
        return;
      }
      // An upload can outlast the model: it may have been unloaded for a test since
      // this request started, and the old port must not be used.
      const current = this.deps.upstream();
      if (!current || current.port !== up.port) {
        const why = this.deps.unavailable();
        sendError(res, 503, "model_not_loaded", why.message, { "retry-after": String(why.retryAfterSeconds) });
        return;
      }
      // llama-server would answer any name with whatever is loaded, which tells an
      // agent asking for one model that it got it.
      if (typeof parsed.model === "string" && parsed.model !== "" && parsed.model !== up.alias) {
        sendError(res, 404, "model_not_found", `Model ${parsed.model.slice(0, 200)} is not loaded. The loaded model is ${up.alias}.`);
        return;
      }
      if (findRemoteMedia(parsed)) {
        sendError(res, 400, "invalid_request_error", "Remote media URLs are not accepted. Send the file inline as a data: URL.");
        return;
      }
      const sampling = route.sampling ? this.deps.sampling() : null;
      const applied = route.sampling && sampling ? applySampling(parsed, route.sampling, sampling) : parsed;
      const body = applied === parsed ? raw : Buffer.from(JSON.stringify(applied));
      handedOff = true;
      this.forward(req, res, path, route, up, body, parsed.stream === true, release);
    } finally {
      if (!handedOff) release();
    }
  }

  private forward(
    req: IncomingMessage,
    res: ServerResponse,
    path: string,
    route: Route,
    up: GatewayUpstream,
    body: Buffer,
    stream: boolean,
    releaseBody: () => void
  ): void {
    const startedAt = Date.now();
    this.inFlightCount++;
    this.totalRequests++;
    // A request that started alone and was then joined shared its slots from there on,
    // so what counts is the most that ran with it, not how many were running at the start.
    const overlap = { peak: this.inFlightCount };
    for (const other of this.overlaps) other.peak = Math.max(other.peak, this.inFlightCount);
    this.overlaps.add(overlap);
    const watched = this.deps.watched();

    let finished = false;
    const finish = (status: number, usage: ReturnType<UsageTap["finish"]> | null, cancelled: boolean) => {
      if (finished) return;
      finished = true;
      releaseBody();
      this.overlaps.delete(overlap);
      this.active.delete(abort);
      this.inFlightCount--;
      if (this.inFlightCount === 0) {
        for (const waiter of this.idleWaiters.splice(0)) waiter();
      }
      if (!route.record || !route.endpoint) return;
      this.deps.onUsage({
        id: randomUUID(),
        at: startedAt,
        model_id: up.modelId,
        load_hash: up.loadHash,
        endpoint: route.endpoint,
        stream,
        status,
        cancelled,
        processed_tokens: usage?.processed_tokens ?? null,
        cached_tokens: usage?.cached_tokens ?? null,
        generated_tokens: usage?.generated_tokens ?? null,
        prompt_ms: usage?.prompt_ms ?? null,
        predicted_ms: usage?.predicted_ms ?? null,
        total_ms: Date.now() - startedAt,
        concurrency: overlap.peak,
        watched,
      });
    };

    const upstream = httpRequest({
      host: "127.0.0.1",
      port: up.port,
      path,
      method: "POST",
      headers: {
        "content-type": "application/json",
        "content-length": body.length,
        authorization: `Bearer ${up.key}`,
        // A compressed response could not be read for its timings.
        "accept-encoding": "identity",
        ...(req.headers.accept ? { accept: req.headers.accept } : {}),
        ...(typeof req.headers["anthropic-version"] === "string" ? { "anthropic-version": req.headers["anthropic-version"] } : {}),
        ...(typeof req.headers["anthropic-beta"] === "string" ? { "anthropic-beta": req.headers["anthropic-beta"] } : {}),
      },
    });

    const abort = () => {
      upstream.destroy();
      if (!res.writableEnded) res.destroy();
      // 499: the conventional status for a request ended before any response.
      finish(res.statusCode === 200 ? 499 : res.statusCode, null, true);
    };
    this.active.add(abort);

    // Closing upstream is what releases the slot: llama-server cancels the task.
    res.on("close", () => {
      if (!res.writableFinished) {
        upstream.destroy();
        finish(res.statusCode, tap?.finish() ?? null, true);
      }
    });

    let tap: UsageTap | null = null;
    upstream.on("response", (upRes) => {
      tap = new UsageTap(upRes.headers["content-type"]);
      const headers: Record<string, string | string[]> = {};
      for (const [name, value] of Object.entries(upRes.headers)) {
        const lower = name.toLowerCase();
        // CORS is this gateway's decision. llama-server sends * by default.
        if (value === undefined || lower.startsWith("access-control-") || lower === "connection" || lower === "keep-alive" || lower === "transfer-encoding" || lower === "server") continue;
        headers[name] = value;
      }
      res.writeHead(upRes.statusCode ?? 502, headers);
      upRes.on("data", (chunk: Buffer) => {
        tap?.feed(chunk);
        if (!res.write(chunk)) {
          upRes.pause();
          res.once("drain", () => upRes.resume());
        }
      });
      upRes.on("end", () => {
        res.end();
        finish(upRes.statusCode ?? 502, tap?.finish() ?? null, false);
      });
      upRes.on("error", () => {
        res.destroy();
        finish(upRes.statusCode ?? 502, tap?.finish() ?? null, true);
      });
    });
    upstream.on("error", (err) => {
      if (finished) return;
      this.deps.log.warn(`daemon gateway: upstream: ${err.message}`);
      sendError(res, 503, "model_not_loaded", "The model stopped responding.", { "retry-after": "10" });
      finish(503, null, false);
    });
    upstream.end(body);
  }
}
