// Owns the served llama-server: start, stop, suspend for a test, restart after a
// crash. A daemon_load job ends when the model is ready; the process then lives here,
// outside the worker's serial job loop. Every lifecycle change takes the same lock,
// so a crash restart can never race a test taking the GPU.

import { spawn, type ChildProcess } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { createServer } from "node:net";
import { networkInterfaces, platform } from "node:os";
import { resolve as resolvePath } from "node:path";
import { buildLoadArgs, EMPTY_SAMPLING, LOAD_SETTINGS, SAMPLING_SETTINGS, unsupportedSettings, validateLoadSettings, ALIAS_PATTERN, type SamplingSettings } from "../../../shared/daemonSettings.js";
import type { DaemonActualLoad, DaemonConfig, DaemonInterface, DaemonLoadSpec, DaemonMemorySnapshot, DaemonPhase, DaemonStatus } from "../../../shared/daemonTypes.js";
import type { DaemonUsageRow } from "../../../shared/daemonUsage.js";
import type { Backend, Model } from "../../../shared/types.js";
import { helpHasFlag } from "../binary-probe.js";
import { DaemonGateway, type GatewayUpstream } from "./gateway.js";
import { DaemonLog } from "./logParse.js";
import { readDaemonMemory, type HardwareLike } from "./memory.js";
import { DaemonStore } from "./store.js";

export const LOAD_TIMEOUT_MS = 15 * 60 * 1000;
const READY_POLL_MS = 250;
const STOP_GRACE_MS = 5_000;
// How long running requests get to finish before a test or unload takes the GPU.
export const DRAIN_MS = 30_000;
export const MAX_CRASH_RESTARTS = 2;
export const CRASH_WINDOW_MS = 10 * 60 * 1000;
// A queued load older than this is dropped rather than claiming the GPU hours later.
export const MAX_LOAD_JOB_AGE_MS = 30 * 60 * 1000;

export interface DaemonHost {
  backend: Backend;
  hardware(): HardwareLike;
  /** Null when the machine has no llama-server build active. */
  activeServer(): { tag: string; path: string } | null;
  /** Binary's --help text, memoized by the caller. */
  helpText(path: string): Promise<string>;
  resolveModelPath(model: Model): Promise<string>;
  /** Absolute path of a file in model_dir. Throws outside it or when missing. */
  resolveModelDirFile(filename: string): string;
  log: { info(msg: string): void; warn(msg: string): void; error(msg: string): void };
  /** The phase changed: telemetry should not wait for its next tick. */
  onChange?: () => void;
  spawnServer?: (path: string, args: string[], env: NodeJS.ProcessEnv) => ChildProcess;
}

interface Running {
  spec: DaemonLoadSpec;
  proc: ChildProcess;
  port: number;
  key: string;
  loadHash: string;
  buildTag: string;
  /** Every file this process has open, as resolved on disk. */
  files: string[];
  log: DaemonLog;
  loadedAt: number;
  crashWatch: boolean;
  nCtx: number | null;
  nSlots: number | null;
  sleeping: boolean;
}

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address ? address.port : 0;
      probe.close(() => (port ? resolve(port) : reject(new Error("no free port"))));
    });
  });
}

function samePath(a: string, b: string): boolean {
  const x = resolvePath(a);
  const y = resolvePath(b);
  return platform() === "win32" ? x.toLowerCase() === y.toLowerCase() : x === y;
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : 1));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableJson(v)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Usage rows from different placements must not be plotted as one curve. */
export function loadHash(spec: DaemonLoadSpec, buildTag: string): string {
  return createHash("sha256")
    .update(stableJson({ model: spec.model_id, build: buildTag, settings: spec.settings, mmproj: spec.mmproj_filename ?? null, draft: spec.draft_model?.id ?? null }))
    .digest("hex")
    .slice(0, 16);
}

export class DaemonLoadCancelled extends Error {
  constructor() {
    super("cancelled");
  }
}

export class DaemonManager {
  readonly gateway: DaemonGateway;
  private phase: DaemonPhase = "idle";
  private message: string | null = null;
  private running: Running | null = null;
  // Out of service but still alive, while its last requests finish.
  private stopping: Running | null = null;
  // Set while a test holds the GPU or after one, until the queue drains.
  private suspendedSpec: DaemonLoadSpec | null = null;
  private pendingRestore = false;
  private restarts: number[] = [];
  private generationDefaults: Record<string, number> | null = null;
  private watched = false;
  private lock: Promise<unknown> = Promise.resolve();
  // Bumped by anything that wants the lock for a decision of its own. A reload this
  // manager started by itself gives way to it instead of holding the lock for as
  // long as a model takes to load.
  private interrupts = 0;
  private supportedCache: { path: string; keys: string[] } | null = null;
  private lastMemory: DaemonMemorySnapshot | null = null;
  private memoryReading: Promise<void> | null = null;
  private memoryGeneration = 0;
  // This process's dedicated VRAM since it last settled, for the spill check.
  private dedicatedReadings: number[] = [];

  constructor(private readonly host: DaemonHost, private readonly store: DaemonStore) {
    this.gateway = new DaemonGateway({
      upstream: () => this.upstream(),
      unavailable: () => this.unavailable(),
      sampling: () => this.liveSampling(),
      keyHashes: () => this.store.config?.key_hashes ?? [],
      watched: () => this.watched,
      onUsage: (row: DaemonUsageRow) => this.store.addUsage(row),
      log: host.log,
    });
  }

  private withLock<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.lock.then(fn, fn);
    this.lock = run.catch(() => undefined);
    return run;
  }

  /** Called at startup: listens if a config exists, and arranges the autostart. */
  async start(): Promise<void> {
    const config = this.store.config;
    if (!config) return;
    await this.gateway.apply(config.gateway);
    if (config.gateway.autostart && this.store.lastSpec) {
      this.suspendedSpec = this.store.lastSpec;
      this.pendingRestore = true;
      this.phase = "suspended";
      this.message = "Loading when the worker's queue is clear.";
    }
  }

  /** True while memory reads would either be wrong or disturb inference. */
  holdsGpu(): boolean {
    return this.running != null || this.stopping != null;
  }

  /** By the paths on disk, not catalog names: a file found by its hash can be named
   * anything. Both arguments are absolute. */
  isServing(filePath: string | null, buildPath: string | null): boolean {
    return [this.running, this.stopping].some((r) => {
      if (!r) return false;
      if (filePath != null && r.files.some((f) => samePath(f, filePath))) return true;
      return buildPath != null && samePath(r.proc.spawnfile, buildPath);
    });
  }

  setWatched(watched: boolean): void {
    this.watched = watched;
  }

  private upstream(): GatewayUpstream | null {
    const r = this.running;
    if (!r || (this.phase !== "ready" && this.phase !== "sleeping")) return null;
    return { port: r.port, key: r.key, alias: r.spec.alias, modelId: r.spec.model_id, loadHash: r.loadHash, createdAt: r.loadedAt };
  }

  private unavailable(): { message: string; retryAfterSeconds: number } {
    switch (this.phase) {
      case "loading":
        return { message: "The model is loading.", retryAfterSeconds: 5 };
      case "suspended":
        return { message: "The model is unloaded while a test runs on this machine. It comes back afterwards.", retryAfterSeconds: 60 };
      case "stopping":
        return { message: "The model is being unloaded.", retryAfterSeconds: 30 };
      default:
        return { message: "No model is loaded on this machine.", retryAfterSeconds: 30 };
    }
  }

  private liveSampling(): SamplingSettings | null {
    const spec = this.running?.spec;
    if (!spec) return null;
    const config = this.store.config;
    if (config && config.sampling_model_id === spec.model_id && config.sampling) return config.sampling;
    return spec.sampling ?? EMPTY_SAMPLING;
  }

  /** onlyIfNewer: for a config that sat in a queue, which may predate a key revoked or
   * an endpoint closed since. Telemetry doesn't ask for it: the server's answer is
   * current by definition, including when a re-added machine restarts at version 1. */
  async applyConfig(config: DaemonConfig, opts: { onlyIfNewer?: boolean } = {}): Promise<void> {
    const stored = this.store.config;
    if (opts.onlyIfNewer && stored && config.version < stored.version) {
      await this.gateway.apply(stored.gateway);
      return;
    }
    this.store.setConfig(config);
    await this.gateway.apply(config.gateway);
  }

  private async buildArgs(spec: DaemonLoadSpec, serverPath: string, port: number): Promise<{ args: string[]; files: string[] }> {
    const { values, errors } = validateLoadSettings(spec.settings);
    if (errors.length > 0) throw new Error(errors.join("; "));
    if (!ALIAS_PATTERN.test(spec.alias)) throw new Error("the API identifier has invalid characters");
    const help = await this.host.helpText(serverPath);
    const supported = new Set(this.supportedKeysFromHelp(help));
    const missing = unsupportedSettings(values, supported);
    if (missing.length > 0) {
      const labels = missing.map((k) => LOAD_SETTINGS.find((d) => d.key === k)?.label ?? k);
      throw new Error(`This llama.cpp build does not support: ${labels.join(", ")}`);
    }
    const modelPath = await this.host.resolveModelPath(spec.model);
    if (!this.host.spawnServer && !existsSync(modelPath)) {
      throw new Error(`${spec.model_name} is not on this machine. Download it on the Models page first.`);
    }
    const args = [
      "--model", modelPath,
      "--host", "127.0.0.1",
      "--port", String(port),
      "--alias", spec.alias,
      // 5 prints request bodies. See logParse.ts.
      "--verbosity", "4",
    ];
    // Stops downloads, which a model already on disk has no need of. It does not stop
    // llama-server fetching an image URL from a request: see mediaUrls.ts for that.
    if (helpHasFlag(help, "--offline")) args.push("--offline");
    if (helpHasFlag(help, "--no-webui")) args.push("--no-webui");
    const files = [modelPath];
    if (spec.mmproj_filename) {
      const mmproj = this.host.resolveModelDirFile(spec.mmproj_filename);
      files.push(mmproj);
      args.push("--mmproj", mmproj);
    }
    if (spec.draft_model) {
      const draft = await this.host.resolveModelPath(spec.draft_model);
      files.push(draft);
      args.push("--model-draft", draft);
    }
    args.push(...buildLoadArgs(values));
    return { args, files };
  }

  private supportedKeysFromHelp(help: string): string[] {
    return LOAD_SETTINGS.filter((d) => helpHasFlag(help, d.flag)).map((d) => d.key);
  }

  private async supportedSettings(): Promise<string[]> {
    const server = this.host.activeServer();
    if (!server) return [];
    if (this.supportedCache?.path === server.path) return this.supportedCache.keys;
    const keys = this.supportedKeysFromHelp(await this.host.helpText(server.path).catch(() => ""));
    this.supportedCache = { path: server.path, keys };
    return keys;
  }

  /** Runs the load to ready. A running model goes first: one model at a time. */
  load(spec: DaemonLoadSpec, opts: { onSpawn?: (proc: ChildProcess) => void; isCancelled?: () => boolean } = {}): Promise<void> {
    this.interrupts++;
    return this.withLock(async () => {
      await this.stopRunning();
      this.suspendedSpec = null;
      this.pendingRestore = false;
      this.restarts = [];
      this.store.setLastSpec(spec);
      await this.startProcess(spec, opts);
    });
  }

  unload(): Promise<void> {
    this.interrupts++;
    return this.withLock(async () => {
      this.suspendedSpec = null;
      this.pendingRestore = false;
      this.restarts = [];
      this.store.setLastSpec(null);
      await this.stopRunning();
      this.setPhase("idle", null);
    });
  }

  /** A test needs the GPU to itself. Waits for running requests, up to DRAIN_MS. */
  suspendForJob(jobType: string): Promise<void> {
    this.interrupts++;
    return this.withLock(async () => {
      const spec = this.running?.spec ?? this.suspendedSpec;
      if (!spec) return;
      this.suspendedSpec = spec;
      this.pendingRestore = true;
      this.setPhase("suspended", `Unloaded for a ${jobType.replace(/_/g, " ")} job. Comes back when the queue is clear.`);
      if (this.running) {
        await this.stopRunning();
        this.host.log.info(`daemon: unloaded ${spec.alias} for a ${jobType} job`);
      }
    });
  }

  /** Called when the queue has run dry, without waiting: a load can take minutes, and
   * the worker has to go on claiming jobs meanwhile. The next load, unload or test
   * cancels this reload rather than waiting for it (audit 2026-11-01 E3). */
  restoreIfPending(): Promise<void> {
    if (!this.pendingRestore) return Promise.resolve();
    const token = this.interrupts;
    return this.withLock(async () => {
      if (this.interrupts !== token) return;
      const spec = this.suspendedSpec;
      this.pendingRestore = false;
      this.suspendedSpec = null;
      if (!spec || this.running) return;
      this.host.log.info(`daemon: reloading ${spec.alias}`);
      await this.startInBackground(spec, token, "reload");
    });
  }

  // A start nobody is waiting on. Given way to, the model is still owed: whatever
  // interrupted decides next, and a test suspends it again from here.
  private async startInBackground(spec: DaemonLoadSpec, token: number, what: string): Promise<void> {
    try {
      await this.startProcess(spec, { isCancelled: () => this.interrupts !== token });
    } catch (err) {
      if (err instanceof DaemonLoadCancelled) {
        this.suspendedSpec = spec;
        this.pendingRestore = true;
        return;
      }
      this.host.log.error(`daemon: ${what} failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private setPhase(phase: DaemonPhase, message: string | null): void {
    const changed = phase !== this.phase || message !== this.message;
    this.phase = phase;
    this.message = message;
    if (!changed) return;
    // A read still in flight describes the old state. The last finished one stays on
    // screen until the next arrives, a few seconds at most, rather than the panel
    // blanking on every change.
    this.memoryGeneration++;
    this.dedicatedReadings = [];
    this.host.onChange?.();
  }

  private async startProcess(spec: DaemonLoadSpec, opts: { onSpawn?: (proc: ChildProcess) => void; isCancelled?: () => boolean }, attempt = 0): Promise<void> {
    const server = this.host.activeServer();
    if (!server) {
      this.setPhase("failed", "This machine has no llama-server build. Install one on the Machines page.");
      throw new Error(this.message as string);
    }
    this.setPhase("loading", null);
    let built: { args: string[]; files: string[] };
    const port = await findFreePort();
    try {
      built = await this.buildArgs(spec, server.path, port);
    } catch (err) {
      this.setPhase("failed", err instanceof Error ? err.message : String(err));
      throw err;
    }
    const key = randomBytes(32).toString("hex");
    // By env rather than argv, where any local process could read it. Spreading
    // keeps Windows' existing Path key intact; a second PATH key would be ignored.
    const env = { ...process.env, LLAMA_API_KEY: key };
    const proc = this.host.spawnServer
      ? this.host.spawnServer(server.path, built.args, env)
      : spawn(server.path, built.args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true, env });
    const log = new DaemonLog();
    proc.stdout?.on("data", (c: Buffer) => log.feed(c));
    proc.stderr?.on("data", (c: Buffer) => log.feed(c));
    opts.onSpawn?.(proc);
    const running: Running = {
      spec,
      proc,
      port,
      key,
      loadHash: loadHash(spec, server.tag),
      buildTag: server.tag,
      files: built.files,
      log,
      loadedAt: Date.now(),
      crashWatch: false,
      nCtx: null,
      nSlots: null,
      sleeping: false,
    };
    this.running = running;
    proc.once("exit", (code, signal) => this.onExit(running, code, signal));
    proc.once("error", (err) => this.host.log.warn(`daemon: llama-server: ${err.message}`));

    try {
      await this.waitReady(running, opts.isCancelled);
    } catch (err) {
      this.running = null;
      if (proc.exitCode == null && proc.signalCode == null) await this.kill(proc);
      // Another process took the port between choosing it and binding it.
      if (!(err instanceof DaemonLoadCancelled) && log.bindFailed && attempt === 0) {
        return this.startProcess(spec, opts, attempt + 1);
      }
      if (err instanceof DaemonLoadCancelled) this.setPhase("idle", null);
      else this.setPhase("failed", err instanceof Error ? err.message : String(err));
      throw err;
    }
    // Watching starts before anything else is awaited: a process that died during the
    // read below would otherwise go unnoticed and read as ready for good.
    running.crashWatch = true;
    log.markReady();
    await this.refreshProps(running);
    if (this.running !== running) return;
    running.nCtx = running.nCtx ?? running.log.report().nCtxSlot;
    running.nSlots = running.nSlots ?? running.log.report().nSlots;
    running.loadedAt = Date.now();
    this.setPhase("ready", null);
    this.host.log.info(`daemon: ${spec.alias} ready on the ${server.tag} build`);
  }

  private async waitReady(r: Running, isCancelled?: () => boolean): Promise<void> {
    const deadline = Date.now() + LOAD_TIMEOUT_MS;
    let exited: string | null = null;
    r.proc.once("exit", (code, signal) => {
      exited = signal ? `signal ${signal}` : `code ${code}`;
    });
    r.proc.once("error", (err) => {
      exited = err.message;
    });
    while (Date.now() < deadline) {
      if (isCancelled?.()) throw new DaemonLoadCancelled();
      if (exited) {
        const reason = r.log.loadErrorSummary();
        throw new Error(`llama-server exited during load (${exited})${reason ? `:\n${reason}` : "."}`);
      }
      try {
        const res = await fetch(`http://127.0.0.1:${r.port}/health`, { signal: AbortSignal.timeout(2_000) });
        if (res.status === 200) return;
      } catch {
        // Not listening yet.
      }
      await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
    }
    throw new Error(`The model did not finish loading within ${LOAD_TIMEOUT_MS / 60_000} minutes.`);
  }

  private onExit(r: Running, code: number | null, signal: NodeJS.Signals | null): void {
    if (this.running !== r || !r.crashWatch) return;
    // After the first request, stderr could in principle hold anything, so this
    // machine's own log gets the tail and the server gets only the exit status.
    this.host.log.error(`daemon: llama-server exited (${signal ?? code}). Last output:\n${r.log.tail()}`);
    this.running = null;
    const now = Date.now();
    this.restarts = this.restarts.filter((at) => now - at < CRASH_WINDOW_MS);
    const reason = `llama-server stopped unexpectedly (${signal ? `signal ${signal}` : `exit code ${code}`}).`;
    if (this.restarts.length >= MAX_CRASH_RESTARTS) {
      this.setPhase("crashed", `${reason} It crashed ${MAX_CRASH_RESTARTS + 1} times in ${CRASH_WINDOW_MS / 60_000} minutes, so it was left down.`);
      return;
    }
    this.restarts.push(now);
    this.setPhase("crashed", `${reason} Restarting.`);
    // Held here so a test arriving before the restart still brings the model back after.
    this.suspendedSpec = r.spec;
    const token = this.interrupts;
    void this.withLock(async () => {
      // A test or an unload may have taken over while this waited for the lock,
      // or be next in line for it.
      if (this.running || this.phase !== "crashed" || this.interrupts !== token) return;
      this.suspendedSpec = null;
      await this.startInBackground(r.spec, token, "restart");
    });
  }

  private async stopRunning(): Promise<void> {
    const r = this.running;
    if (!r) return;
    if (this.phase === "ready" || this.phase === "sleeping") this.setPhase("stopping", "Letting running requests finish.");
    // New requests see no upstream from here on. Ones already forwarded hold their
    // own socket and get DRAIN_MS to finish.
    this.running = null;
    this.stopping = r;
    r.crashWatch = false;
    try {
      await this.gateway.drain(DRAIN_MS);
      await this.kill(r.proc);
    } finally {
      this.stopping = null;
    }
  }

  private kill(proc: ChildProcess): Promise<void> {
    return new Promise((resolve) => {
      if (proc.exitCode != null || proc.signalCode != null) return resolve();
      const timer = setTimeout(() => {
        proc.kill("SIGKILL");
        resolve();
      }, STOP_GRACE_MS);
      proc.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
      proc.kill();
    });
  }

  /** /props is one of the endpoints that does not wake a sleeping server. /slots
   * does, measured, so it is never polled here. */
  private async refreshProps(r: Running): Promise<void> {
    try {
      const res = await fetch(`http://127.0.0.1:${r.port}/props`, {
        headers: { authorization: `Bearer ${r.key}` },
        signal: AbortSignal.timeout(2_000),
      });
      if (!res.ok) return;
      const props = (await res.json()) as {
        is_sleeping?: boolean;
        total_slots?: number;
        default_generation_settings?: { n_ctx?: number; params?: Record<string, unknown> };
      };
      r.sleeping = props.is_sleeping === true;
      if (typeof props.total_slots === "number") r.nSlots = props.total_slots;
      const settings = props.default_generation_settings;
      if (typeof settings?.n_ctx === "number") r.nCtx = settings.n_ctx;
      if (settings?.params) {
        const defaults: Record<string, number> = {};
        for (const def of SAMPLING_SETTINGS) {
          const value = settings.params[def.defaultField];
          if (typeof value === "number" && Number.isFinite(value)) defaults[def.key] = value;
        }
        this.generationDefaults = defaults;
      }
    } catch {
      // Busy or restarting; the next tick asks again.
    }
  }

  private interfaces(): DaemonInterface[] {
    const out: DaemonInterface[] = [];
    for (const [name, addresses] of Object.entries(networkInterfaces())) {
      for (const a of addresses ?? []) {
        if (a.family === "IPv4" && !a.internal) out.push({ name, address: a.address });
      }
    }
    return out.slice(0, 16);
  }

  // In the background: a GPU read takes seconds on Windows, and waiting for it made
  // the page show "Loading" long after the model was ready. A reading started before
  // the process changed describes a process that is gone, so it is dropped.
  private refreshMemory(r: Running | null, report: ReturnType<DaemonLog["report"]> | null): void {
    if (this.memoryReading) return;
    const generation = this.memoryGeneration;
    this.memoryReading = readDaemonMemory({
      backend: this.host.backend,
      hardware: this.host.hardware(),
      pid: r?.proc.pid ?? null,
      report,
      // Only a settled model: during a load or a wake its VRAM is still filling, and
      // the buffers it has reported would read as spilled.
      dedicatedReadings: r && this.phase === "ready" ? this.dedicatedReadings : undefined,
    })
      .then((snapshot) => {
        if (generation === this.memoryGeneration) this.lastMemory = snapshot;
      })
      .catch((err) => {
        this.host.log.warn(`daemon: memory read failed: ${err instanceof Error ? err.message : String(err)}`);
      })
      .finally(() => {
        this.memoryReading = null;
      });
  }

  /** The report telemetry sends. `readMemory`: a viewer is watching and no test is running. */
  async status(readMemory: boolean): Promise<DaemonStatus> {
    const r = this.running;
    if (r && (this.phase === "ready" || this.phase === "sleeping")) {
      await this.refreshProps(r);
      if (this.running === r) this.setPhase(r.sleeping ? "sleeping" : "ready", null);
    }
    const report = r?.log.report() ?? null;
    const actual: DaemonActualLoad | null = r
      ? {
          gpu_layers: report?.gpuLayers ?? null,
          total_layers: report?.totalLayers ?? null,
          n_ctx: r.nCtx ?? report?.nCtxSlot ?? null,
          n_slots: r.nSlots ?? report?.nSlots ?? null,
        }
      : null;
    if (readMemory) this.refreshMemory(r, report);
    else this.lastMemory = null;
    const memory = this.lastMemory;
    const server = this.host.activeServer();
    const spec = r?.spec ?? this.stopping?.spec ?? this.suspendedSpec;
    const gw = this.gateway.status();
    return {
      phase: this.phase,
      message: this.message,
      model_id: spec?.model_id ?? null,
      model_name: spec?.model_name ?? null,
      alias: spec?.alias ?? null,
      build_tag: r?.buildTag ?? null,
      load_hash: r?.loadHash ?? null,
      settings: spec?.settings ?? null,
      mmproj_filename: spec?.mmproj_filename ?? null,
      draft_model_id: spec?.draft_model?.id ?? null,
      actual,
      loaded_at: r?.loadedAt ?? null,
      in_flight: this.gateway.inFlight,
      requests_total: this.gateway.requestsTotal,
      generation_defaults: this.generationDefaults,
      gateway: gw,
      interfaces: this.interfaces(),
      active_build_tag: server?.tag ?? null,
      supported_settings: await this.supportedSettings(),
      restarts: this.restarts.length,
      usage_dropped: this.store.usageDropped,
      autostart_model_id: this.store.config?.gateway.autostart ? this.store.lastSpec?.model_id ?? null : null,
      memory,
    };
  }

  /** Worker shutdown. Bounded by the caller. */
  async shutdown(): Promise<void> {
    const r = this.running;
    const draining = this.stopping;
    this.running = null;
    if (r) r.crashWatch = false;
    this.store.flushIfDirty();
    await this.gateway.stop();
    await Promise.all([r, draining].map((p) => (p ? this.kill(p.proc) : undefined)));
  }
}
