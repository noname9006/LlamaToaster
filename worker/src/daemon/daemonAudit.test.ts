// Audit 2026-11-01 (docs/AUDIT_REPORT_SERVE_2026-11-01.md). Each of these failed against
// the code before the audit's fixes.
import { afterEach, describe, expect, it, vi } from "vitest";
import { createServer, type Server, type ServerResponse } from "node:http";
import { connect } from "node:net";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ChildProcess } from "node:child_process";
import type { DaemonUsageRow } from "../../../shared/daemonUsage.js";
import type { DaemonLoadSpec } from "../../../shared/daemonTypes.js";
import { DaemonGateway } from "./gateway.js";
import { DaemonManager, type DaemonHost } from "./manager.js";
import { DaemonStore } from "./store.js";
import { DaemonLog, safeForRing } from "./logParse.js";
import { findRemoteMedia } from "./mediaUrls.js";
import { pickGpu, spillFromReadings } from "./memory.js";

const quiet = { info: () => undefined, warn: () => undefined, error: () => undefined };
const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const fn of cleanups.splice(0)) await fn();
});

const freePort = async () => {
  const probe = createServer();
  await new Promise<void>((r) => probe.listen(0, "127.0.0.1", r));
  const { port } = probe.address() as { port: number };
  await new Promise<void>((r) => probe.close(() => r()));
  return port;
};

describe("remote media", () => {
  const chat = (part: unknown) => ({ messages: [{ role: "user", content: [part] }] });

  it.each([
    ["chat, object form", chat({ type: "image_url", image_url: { url: "http://router.local/x.png" } })],
    ["chat, string form", chat({ type: "image_url", image_url: "https://example.com/x.png" })],
    ["a part with no type", chat({ image_url: { url: "http://10.0.0.1/" } })],
    ["video", chat({ type: "video_url", video_url: { url: "http://10.0.0.1/v.mp4" } })],
    ["audio", chat({ type: "audio_url", audio_url: { url: "http://10.0.0.1/a.wav" } })],
    ["responses, nested", { input: [{ role: "user", content: [{ type: "input_image", image_url: "http://10.0.0.1/x.png" }] }] }],
    ["responses, direct", { input: [{ type: "input_image", image_url: "http://10.0.0.1/x.png" }] }],
    ["anthropic", chat({ type: "image", source: { type: "url", url: "http://10.0.0.1/x.png" } })],
    ["inside a tool result", chat({ type: "tool_result", content: [{ type: "image", source: { type: "url", url: "http://10.0.0.1/x.png" } }] })],
    ["file scheme", chat({ type: "image_url", image_url: { url: "file:///etc/passwd" } })],
    ["other case", chat({ type: "image_url", image_url: { url: "HTTP://10.0.0.1/" } })],
  ])("finds %s", (_name, body) => {
    expect(findRemoteMedia(body as Record<string, unknown>)).toBeTruthy();
  });

  it.each([
    ["inline image", chat({ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } })],
    ["inline, upper case", chat({ type: "image_url", image_url: { url: "DATA:image/png;base64,AAAA" } })],
    ["anthropic inline", chat({ type: "image", source: { type: "base64", data: "AAAA" } })],
    ["a URL in plain text", { messages: [{ role: "user", content: "read http://example.com" }] }],
    ["a text part naming a URL", chat({ type: "text", text: "http://example.com/x.png" })],
    // An agent's web_fetch tool: refusing these would break ordinary agents.
    ["an anthropic tool call's arguments", chat({ type: "tool_use", name: "web_fetch", input: { url: "http://example.com", image_url: "http://example.com/x.png" } })],
    ["an openai tool call's arguments", { messages: [{ role: "assistant", tool_calls: [{ function: { name: "f", arguments: '{"image_url":"http://x"}' } }] }] }],
    ["a tool schema", { messages: [], tools: [{ function: { parameters: { properties: { image_url: { type: "string" } } } } }] }],
    ["no messages", { prompt: "hello" }],
    ["a string input", { input: "hello" }],
  ])("leaves %s alone", (_name, body) => {
    expect(findRemoteMedia(body as Record<string, unknown>)).toBeNull();
  });
});

function gatewayHarness(respond: (res: ServerResponse, body: Record<string, unknown>) => void) {
  const state = { up: true };
  const usage: DaemonUsageRow[] = [];
  let upstreamRequests = 0;
  const upstream: Server = createServer((req, res) => {
    upstreamRequests++;
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => respond(res, JSON.parse(Buffer.concat(chunks).toString() || "{}") as Record<string, unknown>));
  });
  const gateway = new DaemonGateway({
    upstream: () => (state.up ? { port: (upstream.address() as { port: number }).port, key: "k", alias: "m", modelId: "sha", loadHash: "h", createdAt: 1 } : null),
    unavailable: () => ({ message: "no model", retryAfterSeconds: 30 }),
    sampling: () => null,
    keyHashes: () => [],
    watched: () => false,
    onUsage: (row) => usage.push(row),
    log: quiet,
  });
  cleanups.push(async () => {
    await gateway.stop();
    upstream.closeAllConnections();
    await new Promise<void>((r) => upstream.close(() => r()));
  });
  return {
    state,
    usage,
    gateway,
    requests: () => upstreamRequests,
    start: async () => {
      await new Promise<void>((r) => upstream.listen(0, "127.0.0.1", r));
      const port = await freePort();
      await gateway.apply({ host: "127.0.0.1", port, cors_origins: [], autostart: false });
      return port;
    },
  };
}

const okBody = JSON.stringify({ timings: { cache_n: 0, prompt_n: 40, prompt_ms: 10, predicted_n: 9, predicted_ms: 20 } });
const send = (port: number, body: unknown) =>
  fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

describe("gateway", () => {
  it("refuses a remote image before llama-server can fetch it", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    const port = await h.start();
    const res = await send(port, { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "http://10.0.0.1/x.png" } }] }] });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: { message: string } }).error.message).toContain("data: URL");
    expect(h.requests()).toBe(0);
  });

  it("keeps a limit on how long a request takes to arrive", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    await h.start();
    expect((h.gateway as unknown as { server: Server }).server.requestTimeout).toBeGreaterThan(0);
  });

  it("does not send an upload to a model unloaded while it arrived", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    const port = await h.start();
    const reply = await new Promise<string>((resolve) => {
      const body = "{}";
      const s = connect(port, "127.0.0.1", () => {
        s.write(`POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: ${body.length}\r\n\r\n`);
        setTimeout(() => {
          h.state.up = false;
          s.write(body);
        }, 50);
      });
      s.on("data", (d) => {
        resolve(String(d));
        s.destroy();
      });
    });
    expect(reply).toContain("503");
    expect(h.requests()).toBe(0);
  });

  it("does not orphan a listener when two configs arrive together", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    await new Promise((r) => setImmediate(r));
    const port = await freePort();
    const settings = { host: "127.0.0.1", port, cors_origins: [], autostart: false };
    await Promise.all([h.gateway.apply(settings), h.gateway.apply(settings), h.gateway.apply(settings)]);
    expect(h.gateway.status()).toMatchObject({ listening: true, error: null });
    await h.gateway.stop();
    await expect(fetch(`http://127.0.0.1:${port}/health`)).rejects.toThrow();
  });

  it("lets a stop and a rebind take turns", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    const a = await freePort();
    const b = await freePort();
    await h.gateway.apply({ host: "127.0.0.1", port: a, cors_origins: [], autostart: false });
    await Promise.all([h.gateway.apply({ host: "127.0.0.1", port: b, cors_origins: [], autostart: false }), h.gateway.stop(), h.gateway.apply({ host: "127.0.0.1", port: b, cors_origins: [], autostart: false })]);
    expect(h.gateway.status()).toMatchObject({ listening: true, port: b });
    await expect(fetch(`http://127.0.0.1:${a}/health`)).rejects.toThrow();
  });

  it("records how many ran alongside a request at any point, not only at its start", async () => {
    let releaseFirst: () => void = () => undefined;
    const first = new Promise<void>((r) => (releaseFirst = r));
    const h = gatewayHarness((res, body) => {
      void (body.tag === "A" ? first : Promise.resolve()).then(() => res.end(okBody));
    });
    const port = await h.start();
    const a = send(port, { tag: "A" }).then((r) => r.text());
    await vi.waitFor(() => expect(h.gateway.inFlight).toBe(1));
    await (await send(port, { tag: "B" })).text();
    releaseFirst();
    await a;
    await vi.waitFor(() => expect(h.usage).toHaveLength(2));
    expect(h.usage.map((r) => r.concurrency)).toEqual([2, 2]);
  });

  it("still records 1 for a request that ran alone", async () => {
    const h = gatewayHarness((res) => res.end(okBody));
    const port = await h.start();
    await (await send(port, {})).text();
    await (await send(port, {})).text();
    await vi.waitFor(() => expect(h.usage).toHaveLength(2));
    expect(h.usage.map((r) => r.concurrency)).toEqual([1, 1]);
  });
});

class FakeProc extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  exitCode: number | null = null;
  signalCode: string | null = null;
  pid = 4242;
  spawnfile = "/fake/llama-server";
  server: Server | null = null;
  holdExit: Promise<void> | null = null;
  kill() {
    this.closeHttp();
    void (this.holdExit ?? Promise.resolve()).then(() => {
      this.exitCode = 0;
      this.emit("exit", 0, null);
    });
    return true;
  }
  closeHttp() {
    this.server?.close();
    this.server?.closeAllConnections();
  }
  die(code: number) {
    this.closeHttp();
    this.exitCode = code;
    this.emit("exit", code, null);
  }
}

const spec = (over: Partial<DaemonLoadSpec> = {}): DaemonLoadSpec => ({
  model_id: "a",
  model_name: "a.gguf",
  model: { id: "a", hf_file: "a.gguf", filename: "a.gguf", source: "huggingface" } as unknown as DaemonLoadSpec["model"],
  alias: "a",
  settings: {},
  sampling: { values: {} },
  ...over,
});

function managerHarness(onSpawn: (proc: FakeProc, n: number, port: number) => void, resolved = "/models/a.gguf") {
  const dir = mkdtempSync(join(tmpdir(), "llamatoaster-daemon-audit-"));
  const procs: FakeProc[] = [];
  const host: DaemonHost = {
    backend: "cpu",
    hardware: () => ({}),
    activeServer: () => ({ tag: "b1", path: "/fake/llama-server" }),
    helpText: async () => "",
    resolveModelPath: async () => resolved,
    resolveModelDirFile: (f) => `/models/${f}`,
    log: quiet,
    spawnServer: (_path, args) => {
      const proc = new FakeProc();
      procs.push(proc);
      onSpawn(proc, procs.length, Number(args[args.indexOf("--port") + 1]));
      return proc as unknown as ChildProcess;
    },
  };
  const manager = new DaemonManager(host, new DaemonStore(join(dir, "state.json"), () => undefined));
  cleanups.push(async () => {
    await manager.shutdown();
    for (const p of procs) p.closeHttp();
    rmSync(dir, { recursive: true, force: true });
  });
  return { manager, procs };
}

function healthy(proc: FakeProc, port: number) {
  proc.server = createServer((_req, res) => res.end("{}")).listen(port, "127.0.0.1");
}

describe("manager", () => {
  it("notices a process that dies during its first /props read, and restarts it", async () => {
    const { manager, procs } = managerHarness((proc, n, port) => {
      if (n > 1) return healthy(proc, port);
      proc.server = createServer((req, res) => {
        if (req.url === "/props") proc.die(139);
        else res.end("{}");
      }).listen(port, "127.0.0.1");
    });
    await manager.load(spec());
    await vi.waitFor(() => expect(procs).toHaveLength(2));
    await vi.waitFor(async () => expect((await manager.status(false)).phase).toBe("ready"));
    expect((await manager.status(false)).restarts).toBe(1);
  });

  it("protects a file by where it is on disk, whatever the catalog calls it", async () => {
    const { manager } = managerHarness((proc, _n, port) => healthy(proc, port), "/models/sub/renamed-on-disk.gguf");
    await manager.load(spec());
    expect(manager.isServing("/models/sub/renamed-on-disk.gguf", null)).toBe(true);
    expect(manager.isServing("/models/a.gguf", null)).toBe(false);
  });

  it("also protects the projector and the draft model", async () => {
    const { manager } = managerHarness((proc, _n, port) => healthy(proc, port));
    await manager.load(spec({ mmproj_filename: "mmproj.gguf" }));
    expect(manager.isServing("/models/mmproj.gguf", null)).toBe(true);
  });

  it("tries another port when the chosen one was taken in the meantime", async () => {
    const { manager, procs } = managerHarness((proc, n, port) => {
      if (n > 1) return healthy(proc, port);
      setTimeout(() => {
        proc.stderr.emit("data", Buffer.from("0.1 E srv    start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 1\n"));
        proc.die(1);
      }, 5);
    });
    await manager.load(spec());
    expect(procs).toHaveLength(2);
    expect((await manager.status(false)).phase).toBe("ready");
  });

  it("gives up after that one retry", async () => {
    const { manager, procs } = managerHarness((proc) => {
      setTimeout(() => {
        proc.stderr.emit("data", Buffer.from("0.1 E srv    start: couldn't bind HTTP server socket\n"));
        proc.die(1);
      }, 5);
    });
    await expect(manager.load(spec())).rejects.toThrow(/exited during load/);
    expect(procs).toHaveLength(2);
    expect((await manager.status(false)).phase).toBe("failed");
  });

  it("says it is stopping and still holds the GPU until the process has gone", async () => {
    let release: () => void = () => undefined;
    const { manager, procs } = managerHarness((proc, _n, port) => {
      healthy(proc, port);
      proc.holdExit = new Promise((r) => (release = r));
    });
    await manager.load(spec());
    const unloading = manager.unload();
    await vi.waitFor(() => expect(procs[0].server?.listening).toBe(false));
    const during = await manager.status(false);
    expect(during.phase).toBe("stopping");
    expect(during.model_id).toBe("a");
    expect(manager.holdsGpu()).toBe(true);
    expect(manager.isServing("/models/a.gguf", null)).toBe(true);
    release();
    await unloading;
    expect(manager.holdsGpu()).toBe(false);
    expect((await manager.status(false)).phase).toBe("idle");
  });

  it("reports the projector and draft model, so the page can offer a reload", async () => {
    const { manager } = managerHarness((proc, _n, port) => healthy(proc, port));
    await manager.load(spec({ mmproj_filename: "mmproj.gguf", draft_model: { id: "d", hf_file: "d.gguf" } as unknown as DaemonLoadSpec["model"] }));
    expect(await manager.status(false)).toMatchObject({ mmproj_filename: "mmproj.gguf", draft_model_id: "d" });
  });
});

describe("choosing the GPU", () => {
  const intel = { model: "Intel(R) UHD Graphics 770", vendor: "Intel" };
  const rtx = { model: "NVIDIA GeForce RTX 4070", vendor: "NVIDIA" };

  it("is not fooled by an integrated GPU listed first", () => {
    expect(pickGpu({ gpu: [intel, rtx] }, "vulkan")).toEqual({ name: rtx.model, count: 2, unified: false });
  });

  it("reads unified with only an integrated GPU", () => {
    expect(pickGpu({ gpu: [intel] }, "vulkan")).toEqual({ name: intel.model, count: 1, unified: true });
  });

  it("trusts a machine that reports unified memory", () => {
    expect(pickGpu({ gpu: [{ model: "Apple M3" }], unified_memory: true }, "metal").unified).toBe(true);
  });

  it("has no unified memory to speak of on the CPU backend or with no GPU", () => {
    expect(pickGpu({ gpu: [intel] }, "cpu").unified).toBe(false);
    expect(pickGpu({ gpu: [] }, "vulkan")).toEqual({ name: null, count: 0, unified: false });
  });
});

describe("the spill check", () => {
  const buffers = { deviceMib: 1098, contextMib: 235, byDeviceMib: {}, hostPinnedMib: 0 };

  it("claims nothing from a single reading", () => {
    expect(spillFromReadings(buffers, [900]).measured).toBe(false);
  });

  it("is not set off by rounding", () => {
    expect(spillFromReadings(buffers, [1097, 1097]).spilled).toBe(false);
  });

  it("takes its tolerance from the counter's own movement", () => {
    expect(spillFromReadings(buffers, [1000, 1090]).spilled).toBe(false);
    expect(spillFromReadings(buffers, [1050, 1050]).spilled).toBe(true);
    expect(spillFromReadings(buffers, [1050, 1050]).inSystemRamMib).toBe(48);
  });

  it("reads clean when VRAM holds more than the buffers", () => {
    expect(spillFromReadings(buffers, [1110, 1110]).spilled).toBe(false);
  });

  // Without a baseline VRAM can hold memory outside the buffers that hides a spill
  // of its own size: a clean reading says nothing, a spill reading still stands.
  it("calls a clean reading unmeasured when no probe has measured a baseline", () => {
    expect(spillFromReadings(buffers, [1110, 1110])).toMatchObject({ measured: false, spilled: false });
    expect(spillFromReadings(buffers, [1110, 1110], null).measured).toBe(false);
    expect(spillFromReadings(buffers, [1050, 1050])).toMatchObject({ measured: true, spilled: true });
  });

  // GeForce MX150, CUDA, 7.5B gemma4 file at 40 layers: the buffers claim 1923MiB,
  // dedicated VRAM holds 1954MiB, and 634MiB of weights sit in system RAM.
  describe("with the baseline a probe measured (676MiB, +/-3)", () => {
    const baseline = { mib: 676, toleranceMib: 3 };
    const cuda40 = { deviceMib: 1923, contextMib: 55, byDeviceMib: {}, hostPinnedMib: 0 };
    const cuda24 = { deviceMib: 1232, contextMib: 26, byDeviceMib: {}, hostPinnedMib: 0 };

    it("finds the spill the buffers alone hide", () => {
      expect(spillFromReadings(cuda40, [1954, 1954]).measured).toBe(false);
      expect(spillFromReadings(cuda40, [1954, 1954], baseline)).toMatchObject({ measured: true, spilled: true, inSystemRamMib: 645 });
    });

    it("measures a load that did not spill as clean, not unknown", () => {
      expect(spillFromReadings(cuda24, [1905, 1905], baseline)).toMatchObject({ measured: true, spilled: false });
    });

    it("takes the baseline's own tolerance on top of the counter's movement", () => {
      expect(spillFromReadings(cuda24, [1905, 1905], { mib: 676, toleranceMib: 0 }).spilled).toBe(true);
      expect(spillFromReadings(cuda24, [1905, 1905], { mib: 676, toleranceMib: 3 }).spilled).toBe(false);
    });

    it("still claims nothing from a single reading", () => {
      expect(spillFromReadings(cuda40, [1954], baseline).measured).toBe(false);
    });
  });
});

describe("what stays in the log ring", () => {
  it.each([
    "parse: error parsing grammar: Undefined rule identifier 'SECRET'",
    '0.04.485.330 W srv    operator(): got exception: {"error":{"code":400,"message":"JSON schema SECRET"}}',
    "0.1.2.3 E srv    send_error: task id = 1, error: SECRET",
    "root ::= SECRET",
    "\trepeat_last_n = 64",
  ])("drops %s", (line) => {
    expect(safeForRing(line)).toBe(false);
    const log = new DaemonLog();
    log.feed(`${line}\n`);
    expect(log.tail()).toBe("");
  });

  it.each([
    "0.02.861.231 I slot print_timing: id  3 | task 1 | prompt eval time =     321.91 ms /    21 tokens",
    "0.01.117.833 I sched_reserve:    Vulkan0 compute buffer size =    43.51 MiB",
    "F:/build/ggml.c:123: GGML_ASSERT(x != nullptr) failed",
    "terminate called after throwing an instance of 'std::bad_alloc'",
    "vk::Queue::submit: ErrorDeviceLost",
  ])("keeps %s", (line) => {
    expect(safeForRing(line)).toBe(true);
  });

  it("stops collecting load errors once requests can arrive", () => {
    const log = new DaemonLog();
    log.feed("0.1 E llama_model_load: error loading model\n");
    expect(log.loadErrorSummary()).toContain("error loading model");
    log.markReady();
    log.feed("0.2 E srv: error from a request\n");
    expect(log.loadErrorSummary()).toBeNull();
  });

  it("notices a failed bind", () => {
    const log = new DaemonLog();
    expect(log.bindFailed).toBe(false);
    log.feed("0.1 E srv    start: couldn't bind HTTP server socket\n");
    expect(log.bindFailed).toBe(true);
  });
});
