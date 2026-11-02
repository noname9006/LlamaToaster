import { afterEach, describe, expect, it, vi } from "vitest";
import { createServer, type Server, request as httpRequest } from "node:http";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ChildProcess } from "node:child_process";
import type { DaemonUsageRow } from "../../../shared/daemonUsage.js";
import type { DaemonLoadSpec } from "../../../shared/daemonTypes.js";
import { DaemonGateway, hashApiKey, MAX_IN_FLIGHT, type GatewayDeps } from "./gateway.js";
import { DaemonLog } from "./logParse.js";
import { extractUsage, UsageTap } from "./usageTap.js";
import { buildMemorySnapshot, type MemoryInputs } from "./memory.js";
import { DaemonStore, MAX_SPILL_BASELINES, MAX_SPOOLED_USAGE } from "./store.js";
import { DaemonManager, loadHash, type DaemonHost } from "./manager.js";

const quietLog = { info: () => undefined, warn: () => undefined, error: () => undefined };

describe("usage tap", () => {
  // Shapes copied from real b11609 responses.
  it("reads a chat response's timings over its usage", () => {
    const u = extractUsage({
      usage: { completion_tokens: 24, prompt_tokens: 21, prompt_tokens_details: { cached_tokens: 0 } },
      timings: { cache_n: 16, prompt_n: 9, prompt_ms: 37.8, predicted_n: 24, predicted_ms: 135.4 },
    });
    expect(u).toEqual({ processed_tokens: 9, cached_tokens: 16, generated_tokens: 24, prompt_ms: 37.8, predicted_ms: 135.4 });
  });

  it("reads a Responses body, where the cache is inside input_tokens", () => {
    const u = extractUsage({ usage: { input_tokens: 20, output_tokens: 8, input_tokens_details: { cached_tokens: 14 } } });
    expect(u).toEqual({ processed_tokens: 6, cached_tokens: 14, generated_tokens: 8, prompt_ms: null, predicted_ms: null });
  });

  it("reads an Anthropic body, where it is not", () => {
    const u = extractUsage({ usage: { cache_read_input_tokens: 14, input_tokens: 6, output_tokens: 8 } });
    expect(u).toEqual({ processed_tokens: 6, cached_tokens: 14, generated_tokens: 8, prompt_ms: null, predicted_ms: null });
  });

  it("finds the final chunk of a stream split across packets", () => {
    const tap = new UsageTap("text/event-stream");
    const final = 'data: {"choices":[],"timings":{"cache_n":24,"prompt_n":1,"prompt_ms":5.9,"predicted_n":24,"predicted_ms":133}}\n\ndata: [DONE]\n\n';
    tap.feed(Buffer.from('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'));
    tap.feed(Buffer.from(final.slice(0, 30)));
    tap.feed(Buffer.from(final.slice(30)));
    expect(tap.finish()).toMatchObject({ processed_tokens: 1, cached_tokens: 24, generated_tokens: 24, predicted_ms: 133 });
  });

  it("merges Anthropic stream events and a Responses completed event", () => {
    const anthropic = new UsageTap("text/event-stream");
    anthropic.feed(Buffer.from('event: message_start\ndata: {"type":"message_start","message":{"usage":{"cache_read_input_tokens":3,"input_tokens":4,"output_tokens":0}}}\n\n'));
    anthropic.feed(Buffer.from('event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":8}}\n\n'));
    expect(anthropic.finish()).toMatchObject({ processed_tokens: 4, cached_tokens: 3, generated_tokens: 8 });

    const responses = new UsageTap("text/event-stream");
    responses.feed(Buffer.from('data: {"type":"response.completed","response":{"usage":{"input_tokens":20,"output_tokens":8}},"timings":{"cache_n":14,"prompt_n":6,"prompt_ms":13,"predicted_n":8,"predicted_ms":43}}\n\n'));
    expect(responses.finish()).toEqual({ processed_tokens: 6, cached_tokens: 14, generated_tokens: 8, prompt_ms: 13, predicted_ms: 43 });
  });

  it("survives a body that is not JSON and one that is too large to keep", () => {
    const bad = new UsageTap("application/json");
    bad.feed(Buffer.from("<html>"));
    expect(bad.finish().processed_tokens).toBeNull();
    const big = new UsageTap("application/json");
    big.feed(Buffer.alloc(9 << 20, 32));
    big.feed(Buffer.from('{"usage":{"prompt_tokens":1}}'));
    expect(big.finish().processed_tokens).toBeNull();
  });
});

describe("log parsing", () => {
  // Real --verbosity 4 lines from b11609.
  const LOAD = [
    "0.00.789.863 I load_tensors: offloaded 49/49 layers to GPU",
    "0.00.789.869 I load_tensors:   CPU_Mapped model buffer size =   137.94 MiB",
    "0.00.789.870 I load_tensors:      Vulkan0 model buffer size =   863.64 MiB",
    "0.01.040.281 I llama_context: Vulkan_Host  output buffer size =     2.32 MiB",
    "0.01.075.816 I llama_kv_cache:    Vulkan0 KV buffer size =   192.00 MiB",
    "0.01.117.833 I sched_reserve:    Vulkan0 compute buffer size =    43.51 MiB",
    "0.01.117.835 I sched_reserve: Vulkan_Host compute buffer size =    15.02 MiB",
    "0.02.135.097 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 8192, kv_unified = 'true'",
  ].join("\n");

  it("splits the buffers between GPU and host", () => {
    const log = new DaemonLog();
    log.feed(`${LOAD}\n`);
    expect(log.report()).toMatchObject({
      weightsGpuMib: 863.64,
      weightsHostMib: 137.94,
      kvGpuMib: 192,
      computeGpuMib: 43.51,
      computeHostMib: expect.closeTo(17.34),
      gpuLayers: 49,
      totalLayers: 49,
      nSlots: 4,
      nCtxSlot: 8192,
    });
  });

  it("handles lines split across chunks, and the latest prompt cache size", () => {
    const log = new DaemonLog();
    const line = "0.07.332.520 I srv        update:  - cache state: 1 prompts, 20.027 MiB (limits: 8192.000 MiB, 8192 tokens, 349335 est)\n";
    log.feed(line.slice(0, 40));
    log.feed(line.slice(40));
    log.feed("0.12.568.498 I srv        update:  - cache state: 2 prompts, 48.448 MiB (limits: 8192.000 MiB)\n");
    expect(log.report().promptCacheMib).toBe(48.448);
  });

  it("starts over when the model reloads after sleep", () => {
    const log = new DaemonLog();
    log.feed(`${LOAD}\n0.16.9 I que    start_loop: exiting sleeping state\n`);
    expect(log.report().weightsGpuMib).toBeNull();
    log.feed(`${LOAD}\n`);
    expect(log.report().weightsGpuMib).toBe(863.64);
  });

  it("forgets the buffers when the model goes to sleep, which frees them", () => {
    const log = new DaemonLog();
    log.feed(`${LOAD}\n0.16.940 I srv  handle_sleep: server is entering sleeping state\n`);
    const report = log.report();
    expect(report.weightsGpuMib).toBeNull();
    expect(report.kvGpuMib).toBeNull();
    expect(report.bufferLines).toEqual([]);
  });

  it("replaces the KV cache on a later reservation rather than adding to it", () => {
    const log = new DaemonLog();
    log.feed("a: Vulkan0 KV buffer size = 100.00 MiB\nb: Vulkan0 KV buffer size = 20.00 MiB\nc: Vulkan0 compute buffer size = 5.00 MiB\nd: Vulkan0 KV buffer size = 300.00 MiB\n");
    expect(log.report().kvGpuMib).toBe(300);
  });

  it("collects load failures for the message", () => {
    const log = new DaemonLog();
    log.feed("0.0 E llama_model_load: error loading model: failed to open file\n0.1 I fine\n");
    expect(log.loadErrorSummary()).toContain("failed to open file");
    expect(log.loadErrorSummary()).not.toContain("fine");
  });
});

describe("memory arithmetic", () => {
  const base: MemoryInputs = {
    at: 1,
    unified: false,
    gpuName: "RX 6600 XT",
    ramTotalMib: 49077,
    ramAvailableMib: 36321,
    childRssMib: 1048,
    childRunning: true,
    workerRssMib: 70,
    gpuTotalMib: 8176,
    gpuUsedMib: 1901,
    gpuProcessMib: 1110,
    report: null,
    spill: { measured: true, inSystemRamMib: -11, spilled: false },
  };

  it("matches a real reading", () => {
    const m = buildMemorySnapshot(base);
    expect(m.vram).toEqual({ total: 8176, app: 1110, other: 791, free: 6275 });
    expect(m.ram).toEqual({ total: 49077, app: 1118, other: 11638, free: 36321 });
  });

  it("says unknown, not small, with no per-process reading", () => {
    const m = buildMemorySnapshot({ ...base, gpuProcessMib: null, childRssMib: null });
    expect(m.vram).toEqual({ total: 8176, app: null, other: null, free: 6275 });
    expect(m.ram.app).toBeNull();
    expect(m.ram.other).toBeNull();
  });

  it("reads a true zero with nothing loaded", () => {
    const m = buildMemorySnapshot({ ...base, childRunning: false, childRssMib: null, gpuProcessMib: null, gpuUsedMib: 791 });
    expect(m.vram).toEqual({ total: 8176, app: 0, other: 791, free: 7385 });
    expect(m.ram.app).toBe(70);
  });

  it("shows one bar on unified memory", () => {
    const m = buildMemorySnapshot({ ...base, unified: true });
    expect(m.vram).toBeNull();
    expect(m.unified).toBe(true);
  });

  it("never reports other apps below zero", () => {
    const m = buildMemorySnapshot({ ...base, gpuProcessMib: 5000, gpuUsedMib: 1901 });
    expect(m.vram?.other).toBe(0);
  });
});

// --- Gateway, against a real HTTP upstream that plays llama-server ---

interface Fake {
  server: Server;
  port: number;
  requests: { path: string; body: Record<string, unknown>; headers: Record<string, unknown> }[];
  respond: (req: { path: string; body: Record<string, unknown> }, res: import("node:http").ServerResponse) => void;
  closed: () => number;
}

async function fakeUpstream(): Promise<Fake> {
  let closedCount = 0;
  const fake: Fake = {
    server: null as unknown as Server,
    port: 0,
    requests: [],
    closed: () => closedCount,
    respond: (_req, res) => {
      res.writeHead(200, { "content-type": "application/json", "access-control-allow-origin": "*" });
      res.end(JSON.stringify({ usage: { prompt_tokens: 5, completion_tokens: 2 }, timings: { cache_n: 1, prompt_n: 4, prompt_ms: 10, predicted_n: 2, predicted_ms: 20 } }));
    },
  };
  fake.server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString() || "{}") as Record<string, unknown>;
      fake.requests.push({ path: req.url ?? "", body, headers: req.headers });
      res.on("close", () => closedCount++);
      fake.respond({ path: req.url ?? "", body }, res);
    });
  });
  await new Promise<void>((r) => fake.server.listen(0, "127.0.0.1", r));
  fake.port = (fake.server.address() as { port: number }).port;
  return fake;
}

interface Harness {
  gateway: DaemonGateway;
  port: number;
  fake: Fake;
  usage: DaemonUsageRow[];
  state: { up: boolean; hashes: string[]; sampling: GatewayDeps extends { sampling(): infer S } ? S : never };
}

const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const fn of cleanups.splice(0)) await fn();
});

async function harness(opts: { host?: string; cors?: string[]; maxBufferedBytes?: number } = {}): Promise<Harness> {
  const fake = await fakeUpstream();
  const usage: DaemonUsageRow[] = [];
  const state: Harness["state"] = { up: true, hashes: [], sampling: null };
  const gateway = new DaemonGateway({
    upstream: () => (state.up ? { port: fake.port, key: "internal", alias: "model-a", modelId: "sha", loadHash: "lh", createdAt: 1000 } : null),
    unavailable: () => ({ message: "no model", retryAfterSeconds: 30 }),
    sampling: () => state.sampling,
    keyHashes: () => state.hashes,
    watched: () => false,
    onUsage: (row) => usage.push(row),
    log: quietLog,
    maxBufferedBytes: opts.maxBufferedBytes,
  });
  // Port 0 is refused by validation, not by the gateway; find one by binding.
  const probe = createServer();
  await new Promise<void>((r) => probe.listen(0, "127.0.0.1", r));
  const port = (probe.address() as { port: number }).port;
  await new Promise<void>((r) => probe.close(() => r()));
  await gateway.apply({ host: opts.host ?? "127.0.0.1", port, cors_origins: opts.cors ?? [], autostart: false });
  cleanups.push(async () => {
    await gateway.stop();
    await new Promise<void>((r) => fake.server.close(() => r()));
    fake.server.closeAllConnections();
  });
  return { gateway, port, fake, usage, state };
}

function post(h: Harness, path: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(`http://127.0.0.1:${h.port}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

describe("gateway", () => {
  it("proxies a request and records a usage row with no text in it", async () => {
    const h = await harness();
    const res = await post(h, "/v1/chat/completions", { model: "model-a", messages: [{ role: "user", content: "SECRET" }] });
    expect(res.status).toBe(200);
    await res.json();
    await vi.waitFor(() => expect(h.usage).toHaveLength(1));
    expect(h.usage[0]).toMatchObject({ endpoint: "chat", processed_tokens: 4, cached_tokens: 1, generated_tokens: 2, prompt_ms: 10, predicted_ms: 20, status: 200, cancelled: false, concurrency: 1, model_id: "sha", load_hash: "lh" });
    expect(JSON.stringify(h.usage[0])).not.toContain("SECRET");
  });

  it("uses its own key upstream and never forwards the caller's", async () => {
    const h = await harness();
    h.state.hashes = [hashApiKey("client-key")];
    await (await post(h, "/v1/chat/completions", {}, { authorization: "Bearer client-key" })).text();
    expect(h.fake.requests[0].headers.authorization).toBe("Bearer internal");
    expect(h.fake.requests[0].headers["x-api-key"]).toBeUndefined();
  });

  it("strips llama-server's CORS header", async () => {
    const h = await harness();
    const res = await post(h, "/v1/chat/completions", {});
    expect(res.headers.get("access-control-allow-origin")).toBeNull();
  });

  it("requires a valid key, from either header", async () => {
    const h = await harness();
    h.state.hashes = [hashApiKey("good")];
    expect((await post(h, "/v1/chat/completions", {})).status).toBe(401);
    expect((await post(h, "/v1/chat/completions", {}, { authorization: "Bearer bad" })).status).toBe(401);
    expect((await fetch(`http://127.0.0.1:${h.port}/v1/models`)).status).toBe(401);
    expect((await post(h, "/v1/chat/completions", {}, { authorization: "Bearer good" })).status).toBe(200);
    expect((await post(h, "/v1/messages", {}, { "x-api-key": "good" })).status).toBe(200);
  });

  it("keeps /health open, without saying anything about the model", async () => {
    const h = await harness();
    h.state.hashes = [hashApiKey("good")];
    const res = await fetch(`http://127.0.0.1:${h.port}/health`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ok" });
  });

  it("refuses a model name other than the loaded one", async () => {
    const h = await harness();
    const res = await post(h, "/v1/chat/completions", { model: "gpt-4o" });
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: { type: string; message: string } };
    expect(body.error.type).toBe("model_not_found");
    expect(body.error.message).toContain("model-a");
    expect(h.fake.requests).toHaveLength(0);
  });

  it("lists models itself, so a listing cannot wake a sleeping server", async () => {
    const h = await harness();
    const list = (await (await fetch(`http://127.0.0.1:${h.port}/v1/models`)).json()) as { data: { id: string }[] };
    expect(list.data.map((m) => m.id)).toEqual(["model-a"]);
    expect((await fetch(`http://127.0.0.1:${h.port}/v1/models/model-a`)).status).toBe(200);
    expect((await fetch(`http://127.0.0.1:${h.port}/v1/models/other`)).status).toBe(404);
    expect(h.fake.requests).toHaveLength(0);
  });

  it("does not expose llama-server's admin routes", async () => {
    const h = await harness();
    for (const path of ["/slots", "/props", "/metrics", "/lora-adapters", "/tokenize", "/", "/v1/health/../slots", "/__proto__"]) {
      expect((await fetch(`http://127.0.0.1:${h.port}${path}`)).status === 404 || path.includes("health")).toBe(true);
      expect((await post(h, path, {})).status).toBe(404);
    }
    expect(h.fake.requests).toHaveLength(0);
  });

  it("answers 503 with Retry-After when nothing is loaded", async () => {
    const h = await harness();
    h.state.up = false;
    const res = await post(h, "/v1/chat/completions", {});
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("30");
    expect(((await (await fetch(`http://127.0.0.1:${h.port}/v1/models`)).json()) as { data: unknown[] }).data).toEqual([]);
    expect((await fetch(`http://127.0.0.1:${h.port}/health`)).status).toBe(503);
  });

  it("blocks browser pages from other sites, and a form post", async () => {
    const h = await harness();
    expect((await post(h, "/v1/chat/completions", {}, { origin: "https://evil.example" })).status).toBe(403);
    expect(h.fake.requests).toHaveLength(0);
    const form = await fetch(`http://127.0.0.1:${h.port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "text/plain" }, body: "{}" });
    expect(form.status).toBe(415);
  });

  it("allows an origin on the list, with a preflight", async () => {
    const h = await harness({ cors: ["http://localhost:3000"] });
    const pre = await fetch(`http://127.0.0.1:${h.port}/v1/chat/completions`, { method: "OPTIONS", headers: { origin: "http://localhost:3000" } });
    expect(pre.status).toBe(204);
    expect(pre.headers.get("access-control-allow-origin")).toBe("http://localhost:3000");
    const res = await post(h, "/v1/chat/completions", {}, { origin: "http://localhost:3000" });
    expect(res.status).toBe(200);
    expect(res.headers.get("access-control-allow-origin")).toBe("http://localhost:3000");
  });

  it("refuses a rebound hostname on a loopback bind", async () => {
    const h = await harness();
    const status = await new Promise<number>((resolve) => {
      const req = httpRequest({ host: "127.0.0.1", port: h.port, path: "/v1/models", headers: { host: "evil.example" } }, (res) => {
        res.resume();
        resolve(res.statusCode ?? 0);
      });
      req.end();
    });
    expect(status).toBe(403);
  });

  it("refuses to listen beyond this machine without a key", async () => {
    const h = await harness();
    await h.gateway.apply({ host: "0.0.0.0", port: h.port, cors_origins: [], autostart: false });
    expect(h.gateway.status().listening).toBe(false);
    expect(h.gateway.status().error).toContain("API key");
  });

  it("reports a port already in use", async () => {
    const h = await harness();
    const other = new DaemonGateway({ upstream: () => null, unavailable: () => ({ message: "", retryAfterSeconds: 1 }), sampling: () => null, keyHashes: () => [], watched: () => false, onUsage: () => undefined, log: quietLog });
    await other.apply({ host: "127.0.0.1", port: h.port, cors_origins: [], autostart: false });
    expect(other.status().listening).toBe(false);
    expect(other.status().error).toContain("in use");
  });

  it("applies sampling and forwards the changed body", async () => {
    const h = await harness();
    h.state.sampling = { values: { top_k: 7, temperature: 0.5 } };
    await (await post(h, "/v1/chat/completions", { messages: [], temperature: 0.1 })).text();
    expect(h.fake.requests[0].body).toMatchObject({ top_k: 7, temperature: 0.1 });
  });

  it("rejects a body that is not a JSON object", async () => {
    const h = await harness();
    const res = await fetch(`http://127.0.0.1:${h.port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: "[1,2]" });
    expect(res.status).toBe(400);
  });

  it("streams through without buffering and reads the final chunk", async () => {
    const h = await harness();
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    h.fake.respond = (_req, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write('data: {"choices":[{"delta":{"content":"a"}}]}\n\n');
      void gate.then(() => {
        res.write('data: {"choices":[],"timings":{"cache_n":2,"prompt_n":40,"prompt_ms":10,"predicted_n":9,"predicted_ms":90}}\n\ndata: [DONE]\n\n');
        res.end();
      });
    };
    const res = await post(h, "/v1/chat/completions", { stream: true });
    const reader = res.body!.getReader();
    const first = await reader.read();
    expect(Buffer.from(first.value!).toString()).toContain('"a"');
    expect(h.gateway.inFlight).toBe(1);
    release();
    while (!(await reader.read()).done);
    await vi.waitFor(() => expect(h.usage).toHaveLength(1));
    expect(h.usage[0]).toMatchObject({ stream: true, processed_tokens: 40, cached_tokens: 2, generated_tokens: 9, cancelled: false });
    expect(h.gateway.inFlight).toBe(0);
  });

  it("releases the upstream when the client goes away", async () => {
    const h = await harness();
    h.fake.respond = (_req, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write('data: {"choices":[]}\n\n');
    };
    const ac = new AbortController();
    const res = await fetch(`http://127.0.0.1:${h.port}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ stream: true }),
      signal: ac.signal,
    });
    await res.body!.getReader().read();
    ac.abort();
    await vi.waitFor(() => expect(h.fake.closed()).toBe(1));
    await vi.waitFor(() => expect(h.usage).toHaveLength(1));
    expect(h.usage[0].cancelled).toBe(true);
    expect(h.gateway.inFlight).toBe(0);
  });

  it("drain waits for running requests, then cuts off the rest", async () => {
    const h = await harness();
    h.fake.respond = (_req, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write('data: {"choices":[]}\n\n');
    };
    const res = await post(h, "/v1/chat/completions", { stream: true });
    const reader = res.body!.getReader();
    await reader.read();
    expect(h.gateway.inFlight).toBe(1);
    await h.gateway.drain(50);
    await vi.waitFor(() => expect(h.gateway.inFlight).toBe(0));
    await vi.waitFor(() => expect(h.usage).toHaveLength(1));
    expect(h.usage[0].cancelled).toBe(true);
    await vi.waitFor(() => expect(h.fake.closed()).toBe(1));
  });
});

// Audit 2026-11-01 E5, E9.
describe("gateway under hostile clients", () => {
  // Sends part of a body, then leaves the upload open.
  function startUpload(h: Harness, declared: number, sent: number) {
    const req = httpRequest({
      host: "127.0.0.1",
      port: h.port,
      path: "/v1/chat/completions",
      method: "POST",
      headers: { "content-type": "application/json", "content-length": declared },
    });
    req.on("error", () => undefined);
    req.write("x".repeat(sent));
    return req;
  }
  async function until(check: () => boolean | Promise<boolean>) {
    const deadline = Date.now() + 5_000;
    while (!(await check())) {
      if (Date.now() > deadline) throw new Error("timed out");
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it("counts uploads in progress against the in-flight limit, and lets go when they drop", async () => {
    const h = await harness();
    const uploads = Array.from({ length: MAX_IN_FLIGHT }, () => startUpload(h, 1_000_000, 10));
    await until(async () => (await post(h, "/v1/chat/completions", { messages: [] })).status === 503);
    const refused = await post(h, "/v1/chat/completions", { messages: [] });
    expect(refused.status).toBe(503);
    expect(((await refused.json()) as { error: { type: string } }).error.type).toBe("overloaded");
    for (const u of uploads) u.destroy();
    await until(async () => (await post(h, "/v1/chat/completions", { messages: [] })).status === 200);
    await until(() => h.gateway.bufferedBodyBytes === 0);
  });

  it("refuses a body once the bodies held would pass the memory budget", async () => {
    const h = await harness({ maxBufferedBytes: 1000 });
    const first = startUpload(h, 100_000, 900);
    await until(() => h.gateway.bufferedBodyBytes === 900);
    const res = await post(h, "/v1/chat/completions", { messages: [], pad: "y".repeat(400) });
    expect(res.status).toBe(503);
    expect(h.gateway.bufferedBodyBytes).toBe(900);
    first.destroy();
    await until(() => h.gateway.bufferedBodyBytes === 0);
    expect((await post(h, "/v1/chat/completions", { messages: [], pad: "y".repeat(400) })).status).toBe(200);
  });

  it("holds a body's bytes until the request ends, then returns them on every exit", async () => {
    const h = await harness();
    let during = -1;
    h.fake.respond = (_req, res) => {
      during = h.gateway.bufferedBodyBytes;
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    };
    expect((await post(h, "/v1/chat/completions", { messages: [] })).status).toBe(200);
    expect(during).toBeGreaterThan(0);
    await until(() => h.gateway.bufferedBodyBytes === 0);
    expect((await post(h, "/v1/chat/completions", { model: "other", messages: [] })).status).toBe(404);
    const invalid = await fetch(`http://127.0.0.1:${h.port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: "not json" });
    expect(invalid.status).toBe(400);
    expect(h.gateway.bufferedBodyBytes).toBe(0);
  });

  it("answers a malformed model name with a 404, not a 500", async () => {
    const h = await harness();
    const res = await fetch(`http://127.0.0.1:${h.port}/v1/models/%E0%A4%A`);
    expect(res.status).toBe(404);
    expect(((await res.json()) as { error: { type: string } }).error.type).toBe("model_not_found");
  });
});

describe("store", () => {
  function temp() {
    const dir = mkdtempSync(join(tmpdir(), "llamatoaster-daemon-store-"));
    cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
    return join(dir, "state.json");
  }
  const row = (id: string) => ({ id } as DaemonUsageRow);

  it("survives a restart", () => {
    const path = temp();
    const a = new DaemonStore(path, () => undefined);
    a.addUsage(row("1"));
    a.setConfig({ version: 3, gateway: { host: "127.0.0.1", port: 1, cors_origins: [], autostart: true }, key_hashes: ["h"], sampling_model_id: null, sampling: null });
    const b = new DaemonStore(path, () => undefined);
    expect(b.config?.version).toBe(3);
    expect(b.usageBatch().map((r) => r.id)).toEqual(["1"]);
    b.ackUsage(["1"]);
    expect(b.usageBatch()).toEqual([]);
  });

  it("drops the oldest rows past its bound, and counts them", () => {
    const store = new DaemonStore(temp(), () => undefined);
    for (let i = 0; i < MAX_SPOOLED_USAGE + 3; i++) store.addUsage(row(String(i)));
    expect(store.usageDropped).toBe(3);
    expect(store.usageBatch()[0].id).toBe("3");
  });

  it("starts fresh from a corrupt file", () => {
    const path = temp();
    writeFileSync(path, "{nope");
    const warnings: string[] = [];
    const store = new DaemonStore(path, (m) => warnings.push(m));
    expect(store.config).toBeNull();
    expect(warnings).toHaveLength(1);
  });

  it("keeps a spill baseline across a restart", () => {
    const path = temp();
    const a = new DaemonStore(path, () => undefined);
    expect(a.spillBaseline("m|cuda|gpu|b1")).toBeNull();
    a.setSpillBaseline("m|cuda|gpu|b1", { mib: 676, toleranceMib: 3 });
    const b = new DaemonStore(path, () => undefined);
    expect(b.spillBaseline("m|cuda|gpu|b1")).toEqual({ mib: 676, toleranceMib: 3 });
    expect(b.spillBaseline("m|cuda|gpu|b2")).toBeNull();
    b.setSpillBaseline("m|cuda|gpu|b1", { mib: 700, toleranceMib: 1 });
    expect(b.spillBaseline("m|cuda|gpu|b1")).toEqual({ mib: 700, toleranceMib: 1 });
  });

  it("drops the oldest spill baselines past its bound", () => {
    const store = new DaemonStore(temp(), () => undefined);
    for (let i = 0; i < MAX_SPILL_BASELINES + 2; i++) store.setSpillBaseline(`k${i}`, { mib: i, toleranceMib: 0 });
    expect(store.spillBaseline("k0")).toBeNull();
    expect(store.spillBaseline("k1")).toBeNull();
    expect(store.spillBaseline("k2")).toEqual({ mib: 2, toleranceMib: 0 });
    // A re-measured key counts as the newest, so it outlives the next eviction.
    store.setSpillBaseline("k2", { mib: 9, toleranceMib: 0 });
    store.setSpillBaseline("fresh", { mib: 1, toleranceMib: 0 });
    expect(store.spillBaseline("k2")).toEqual({ mib: 9, toleranceMib: 0 });
    expect(store.spillBaseline("k3")).toBeNull();
  });

  it("starts with no baselines from a state file written before they existed", () => {
    const path = temp();
    writeFileSync(path, JSON.stringify({ config: null, last_spec: null, usage: [], usage_dropped: 0 }));
    expect(new DaemonStore(path, () => undefined).spillBaseline("anything")).toBeNull();
  });

  it("holds no key", () => {
    const path = temp();
    const store = new DaemonStore(path, () => undefined);
    store.setConfig({ version: 1, gateway: { host: "127.0.0.1", port: 1, cors_origins: [], autostart: false }, key_hashes: [hashApiKey("the-key")], sampling_model_id: null, sampling: null });
    expect(readFileSync(path, "utf8")).not.toContain("the-key");
  });
});

// --- Manager, with a real HTTP "llama-server" behind a fake process ---

class FakeProc extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  exitCode: number | null = null;
  signalCode: string | null = null;
  pid = 4242;
  spawnfile = "";
  killed = false;
  server: Server | null = null;
  kill() {
    this.killed = true;
    this.server?.close();
    this.server?.closeAllConnections();
    this.exitCode = 0;
    setImmediate(() => this.emit("exit", 0, null));
    return true;
  }
  crash(code = 1) {
    this.server?.close();
    this.server?.closeAllConnections();
    this.exitCode = code;
    this.emit("exit", code, null);
  }
}

const spec = (over: Partial<DaemonLoadSpec> = {}): DaemonLoadSpec => ({
  model_id: "sha-a",
  model_name: "a.gguf",
  model: { id: "sha-a", hf_file: "a.gguf", filename: "a.gguf", source: "huggingface" } as unknown as DaemonLoadSpec["model"],
  alias: "model-a",
  settings: { ctx_size: 4096 },
  sampling: { values: {} },
  ...over,
});

function managerHarness(opts: { failStart?: boolean; help?: string } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "llamatoaster-daemon-mgr-"));
  cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
  const procs: FakeProc[] = [];
  const spawns: { args: string[]; env: NodeJS.ProcessEnv }[] = [];
  // The next starts stay "loading" for as long as this is set.
  const control = { neverReady: false, failStart: opts.failStart ?? false };
  const host: DaemonHost = {
    backend: "cpu",
    hardware: () => ({}),
    activeServer: () => ({ tag: "b1", path: "/fake/llama-server" }),
    helpText: async () => opts.help ?? "--ctx-size\n--offline\n--no-webui\n--threads\n",
    resolveModelPath: async (m) => `/models/${m.hf_file}`,
    resolveModelDirFile: (f) => `/models/${f}`,
    log: quietLog,
    spawnServer: (_path, args, env) => {
      spawns.push({ args, env });
      const proc = new FakeProc();
      proc.spawnfile = "/fake/llama-server";
      procs.push(proc);
      if (control.failStart) {
        setTimeout(() => {
          proc.stderr.emit("data", Buffer.from("0.1 E llama_model_load: error loading model: failed to open file\n"));
          proc.crash(1);
        }, 5);
        return proc as unknown as ChildProcess;
      }
      if (control.neverReady) return proc as unknown as ChildProcess;
      const port = Number(args[args.indexOf("--port") + 1]);
      proc.server = createServer((req, res) => {
        res.setHeader("content-type", "application/json");
        if (req.url === "/props") res.end(JSON.stringify({ is_sleeping: false, total_slots: 2, default_generation_settings: { n_ctx: 4096, params: { temperature: 0.8, top_k: 40 } } }));
        else res.end(JSON.stringify({ status: "ok" }));
      }).listen(port, "127.0.0.1");
      return proc as unknown as ChildProcess;
    },
  };
  const store = new DaemonStore(join(dir, "state.json"), () => undefined);
  const manager = new DaemonManager(host, store);
  cleanups.push(async () => {
    await manager.shutdown();
  });
  return { manager, store, procs, spawns, control };
}

// Audit 2026-11-01 E3: restoring a model used to block the worker's whole job loop
// for as long as the load took, up to the load timeout.
describe("manager: a reload nobody asked to wait for", () => {
  async function untilSpawned(spawns: unknown[], n: number) {
    const deadline = Date.now() + 5_000;
    while (spawns.length < n) {
      if (Date.now() > deadline) throw new Error("no spawn");
      await new Promise((r) => setTimeout(r, 5));
    }
  }

  it("gives way to a test, and the model still comes back afterwards", async () => {
    const { manager, spawns, procs, control } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("tg_test");
    control.neverReady = true;
    const restoring = manager.restoreIfPending();
    await untilSpawned(spawns, 2);
    await manager.suspendForJob("tg_test");
    await restoring;
    expect(procs[1].killed).toBe(true);
    expect((await manager.status(false)).phase).toBe("suspended");

    control.neverReady = false;
    await manager.restoreIfPending();
    expect((await manager.status(false)).phase).toBe("ready");
    expect(spawns).toHaveLength(3);
  });

  it("gives way to an unload, and doesn't come back", async () => {
    const { manager, spawns, control } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("tg_test");
    control.neverReady = true;
    const restoring = manager.restoreIfPending();
    await untilSpawned(spawns, 2);
    await manager.unload();
    await restoring;
    expect((await manager.status(false)).phase).toBe("idle");
    control.neverReady = false;
    await manager.restoreIfPending();
    expect(spawns).toHaveLength(2);
  });

  it("gives way to a load of another model, which is the one that stays", async () => {
    const { manager, spawns, control } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("tg_test");
    control.neverReady = true;
    const restoring = manager.restoreIfPending();
    await untilSpawned(spawns, 2);
    control.neverReady = false;
    await manager.load(spec({ alias: "model-b" }));
    await restoring;
    const s = await manager.status(false);
    expect(s.phase).toBe("ready");
    expect(s.alias).toBe("model-b");
    await manager.restoreIfPending();
    expect(spawns).toHaveLength(3);
  });

  it("isn't started at all if a test arrives first", async () => {
    const { manager, spawns } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("tg_test");
    const restoring = manager.restoreIfPending();
    const suspending = manager.suspendForJob("tg_test");
    await Promise.all([restoring, suspending]);
    expect(spawns).toHaveLength(1);
    expect((await manager.status(false)).phase).toBe("suspended");
  });

  it("a crash restart gives way too, and the test doesn't lose the model", async () => {
    const { manager, spawns, procs, control } = managerHarness();
    await manager.load(spec());
    control.neverReady = true;
    procs[0].crash();
    await untilSpawned(spawns, 2);
    await manager.suspendForJob("tg_test");
    expect((await manager.status(false)).phase).toBe("suspended");
    control.neverReady = false;
    await manager.restoreIfPending();
    expect((await manager.status(false)).phase).toBe("ready");
  });

  it("a test arriving right after a crash keeps the model owed", async () => {
    const { manager, spawns, procs } = managerHarness();
    await manager.load(spec());
    procs[0].crash();
    await manager.suspendForJob("tg_test");
    expect(spawns).toHaveLength(1);
    expect((await manager.status(false)).phase).toBe("suspended");
    await manager.restoreIfPending();
    expect((await manager.status(false)).phase).toBe("ready");
  });

  it("a failed reload shows as failed, without an unhandled rejection or a retry", async () => {
    const { manager, spawns, control } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("tg_test");
    control.failStart = true;
    await manager.restoreIfPending();
    expect((await manager.status(false)).phase).toBe("failed");
    await manager.restoreIfPending();
    expect(spawns).toHaveLength(2);
  });
});

// Audit 2026-11-01 E4: a queued load job carries the config from when it was queued.
describe("manager: config from a queued job", () => {
  const config = (version: number) =>
    ({ version, gateway: { host: "127.0.0.1", port: 48123, cors_origins: [], autostart: false }, api_keys: [], sampling: null, sampling_model_id: null }) as unknown as Parameters<DaemonManager["applyConfig"]>[0];

  it("is ignored when something newer has arrived", async () => {
    const { manager, store } = managerHarness();
    await manager.applyConfig(config(5));
    await manager.applyConfig(config(3), { onlyIfNewer: true });
    expect(store.config?.version).toBe(5);
    await manager.applyConfig(config(6), { onlyIfNewer: true });
    expect(store.config?.version).toBe(6);
  });

  it("telemetry's answer always wins, as a re-added machine starts counting again", async () => {
    const { manager, store } = managerHarness();
    await manager.applyConfig(config(5));
    await manager.applyConfig(config(1));
    expect(store.config?.version).toBe(1);
  });
});

describe("manager", () => {
  it("loads to ready, with the key in the environment and not on the command line", async () => {
    const { manager, spawns } = managerHarness();
    await manager.load(spec());
    const s = await manager.status(false);
    expect(s.phase).toBe("ready");
    expect(s.actual).toMatchObject({ n_ctx: 4096, n_slots: 2 });
    expect(s.generation_defaults).toEqual({ temperature: 0.8, top_k: 40 });
    const { args, env } = spawns[0];
    expect(args).toContain("--offline");
    expect(args).toContain("--no-webui");
    expect(args.join(" ")).toContain("--verbosity 4");
    expect(args.join(" ")).toContain("--ctx-size 4096");
    expect(args).toContain("127.0.0.1");
    expect(args.join(" ")).not.toContain(String(env.LLAMA_API_KEY));
    expect(env.LLAMA_API_KEY).toMatch(/^[0-9a-f]{64}$/);
  });

  it("fails clearly when the process dies during load, and does not retry", async () => {
    const { manager, procs } = managerHarness({ failStart: true });
    await expect(manager.load(spec())).rejects.toThrow(/failed to open file/);
    const s = await manager.status(false);
    expect(s.phase).toBe("failed");
    expect(s.message).toContain("failed to open file");
    expect(procs).toHaveLength(1);
  });

  it("refuses a setting the build lacks instead of dropping it", async () => {
    const { manager, spawns } = managerHarness({ help: "--threads\n" });
    await expect(manager.load(spec())).rejects.toThrow(/does not support: Context length/);
    expect(spawns).toHaveLength(0);
  });

  it("cancels a load", async () => {
    const { manager } = managerHarness({ failStart: false });
    let cancelled = false;
    const pending = manager.load(spec(), { isCancelled: () => cancelled, onSpawn: () => (cancelled = true) });
    await expect(pending).rejects.toThrow("cancelled");
    expect((await manager.status(false)).phase).toBe("idle");
  });

  it("replaces the running model on a new load", async () => {
    const { manager, procs } = managerHarness();
    await manager.load(spec());
    await manager.load(spec({ model_id: "sha-b", alias: "model-b" }));
    expect(procs[0].killed).toBe(true);
    expect((await manager.status(false)).alias).toBe("model-b");
  });

  it("unloads", async () => {
    const { manager, procs, store } = managerHarness();
    await manager.load(spec());
    await manager.unload();
    expect(procs[0].killed).toBe(true);
    expect((await manager.status(false)).phase).toBe("idle");
    expect(store.lastSpec).toBeNull();
  });

  it("gives the GPU to a test and restores only when asked", async () => {
    const { manager, procs } = managerHarness();
    await manager.load(spec());
    expect(manager.holdsGpu()).toBe(true);
    await manager.suspendForJob("benchmark");
    expect(procs[0].killed).toBe(true);
    expect(manager.holdsGpu()).toBe(false);
    const during = await manager.status(false);
    expect(during.phase).toBe("suspended");
    expect(during.alias).toBe("model-a");
    // A second test in the batch: nothing to do, and no reload in between.
    await manager.suspendForJob("run_probe");
    expect(procs).toHaveLength(1);
    await manager.restoreIfPending();
    expect(procs).toHaveLength(2);
    expect((await manager.status(false)).phase).toBe("ready");
    await manager.restoreIfPending();
    expect(procs).toHaveLength(2);
  });

  it("does nothing for a test with no model loaded", async () => {
    const { manager, procs } = managerHarness();
    await manager.suspendForJob("benchmark");
    await manager.restoreIfPending();
    expect(procs).toHaveLength(0);
    expect((await manager.status(false)).phase).toBe("idle");
  });

  it("an unload during a test cancels the restore", async () => {
    const { manager, procs } = managerHarness();
    await manager.load(spec());
    await manager.suspendForJob("benchmark");
    await manager.unload();
    await manager.restoreIfPending();
    expect(procs).toHaveLength(1);
    expect((await manager.status(false)).phase).toBe("idle");
  });

  it("restarts after a crash, up to a bound, then stays down", async () => {
    const { manager, procs } = managerHarness();
    await manager.load(spec());
    procs[0].crash(3);
    await vi.waitFor(() => expect(manager.holdsGpu() && procs.length === 2).toBe(true));
    await vi.waitFor(async () => expect((await manager.status(false)).phase).toBe("ready"));
    procs[1].crash(3);
    await vi.waitFor(() => expect(procs.length).toBe(3));
    await vi.waitFor(async () => expect((await manager.status(false)).phase).toBe("ready"));
    procs[2].crash(3);
    await vi.waitFor(async () => expect((await manager.status(false)).phase).toBe("crashed"));
    await new Promise((r) => setTimeout(r, 50));
    expect(procs).toHaveLength(3);
    const s = await manager.status(false);
    expect(s.message).toContain("left down");
    expect(s.message).toContain("exit code 3");
  });

  it("does not report a deliberate stop as a crash", async () => {
    const { manager, procs } = managerHarness();
    await manager.load(spec());
    await manager.unload();
    procs[0].emit("exit", 0, null);
    expect((await manager.status(false)).phase).toBe("idle");
    expect(procs).toHaveLength(1);
  });

  it("knows what it is serving, so neither can be deleted under it", async () => {
    const { manager } = managerHarness();
    await manager.load(spec());
    expect(manager.isServing("/models/a.gguf", null)).toBe(true);
    expect(manager.isServing("/models/b.gguf", null)).toBe(false);
    expect(manager.isServing(null, "/fake/llama-server")).toBe(true);
    expect(manager.isServing(null, "/other/llama-server")).toBe(false);
  });

  it("sets up the autostart only when asked to", async () => {
    const { manager, store, procs } = managerHarness();
    store.setLastSpec(spec());
    store.setConfig({ version: 1, gateway: { host: "127.0.0.1", port: 0, cors_origins: [], autostart: false }, key_hashes: [], sampling_model_id: null, sampling: null });
    await manager.start();
    await manager.restoreIfPending();
    expect(procs).toHaveLength(0);
  });

  it("uses the same hash for the same load, and another for a different one", () => {
    expect(loadHash(spec(), "b1")).toBe(loadHash(spec({ settings: { ctx_size: 4096 } }), "b1"));
    expect(loadHash(spec(), "b1")).not.toBe(loadHash(spec(), "b2"));
    expect(loadHash(spec(), "b1")).not.toBe(loadHash(spec({ settings: { ctx_size: 8192 } }), "b1"));
  });
});
