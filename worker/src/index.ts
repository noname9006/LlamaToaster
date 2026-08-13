import { createServer, type IncomingMessage } from "node:http";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve, relative, isAbsolute, sep } from "node:path";
import {
  readFileSync,
  writeFileSync,
  renameSync,
  mkdirSync,
  createWriteStream,
  unlinkSync,
  readdirSync,
  statSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { runBench, type BenchResult } from "./bench.js";
import { runServerBench } from "./serverBench.js";
import { MemorySampler, captureFreeMemoryBaseline, type SampleStats, type FreeMemoryBaseline } from "./sampler.js";
import { readGpuMemory } from "./vram.js";
import { writeRawJson, postRunItemUpdate, postModelDownloadResult } from "./vps-client.js";
import { log, configureLogging } from "./log.js";
import {
  detectPlatform,
  listInstalledBuilds,
  getInstalledBuild,
  installBuild,
  deleteBuild,
  validateTag,
} from "./llama-builds.js";
import { detectHardware, detectBackend } from "./hardware.js";
import { readGgufInfo } from "./gguf.js";
import type {
  Model,
  SweepConfig,
  IngestResultInput,
  Backend,
  InstalledBuild,
  RunItemTickInput,
  RunItemTerminalInput,
  ModelDownloadCallbackInput,
} from "../../shared/types.js";
import { expandSweep, deriveTestType, type SweepItem } from "../../shared/sweep.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const configPath = process.env.WORKER_CONFIG
  ? join(process.cwd(), process.env.WORKER_CONFIG)
  : join(__dirname, "..", "config.json");

interface WorkerConfig {
  worker_name: string;
  // Optional on purpose -- omit it and the worker auto-detects a sensible
  // value from live hardware at startup (see the detectBackend call below).
  // Set it explicitly only to pin/override that (e.g. force `cpu` on a box
  // that does have a GPU).
  backend?: Backend;
  llama_cpp_build: string;
  llama_bench_path: string;
  model_dir: string;
  vps_url: string;
  bind_host: string;
  port?: number;
  raw_json_dir?: string;
  bench_timeout_ms?: number;
  llama_cpp_builds_dir?: string;
  log_dir?: string;
  // llama-server sibling of llama_bench_path, used only for MTP benchmarking
  // (see worker/src/serverBench.ts). Auto-derived from llama_bench_path's
  // own directory if unset -- only needed here at all to support a manually
  // pinned llama_bench_path that predates the installed-build registry (see
  // deriveServerPath below); a build installed via the Workers page already
  // gets its server_path recorded in that registry directly.
  llama_server_path?: string;
  // Fixed local port the worker binds llama-server to for MTP benchmarking.
  // Only one benchmark ever runs at a time per worker (the existing `busy`
  // flag already enforces that), so a fixed port is simpler than allocating
  // one dynamically and has no collision risk in practice.
  mtp_server_port?: number;
  // How long to keep raw stdout/stderr dumps (see writeRawJson below) before
  // a periodic janitor deletes them -- see pruneRawJson. Split in two since
  // a failed item's raw output is the main reason these files are worth
  // keeping at all (see the app's own results table for successful numbers),
  // so it defaults to a much longer/indefinite retention than a successful
  // item's, which is mostly redundant with what's already in the DB.
  raw_json_retention_days?: number;
  // Unset (the default) means "never auto-delete failed items' raw output."
  raw_json_retention_days_failed?: number;
}

function loadConfig(): WorkerConfig {
  const raw = readFileSync(configPath, "utf8");
  return JSON.parse(raw) as WorkerConfig;
}

const config = loadConfig();
const PORT = config.port ?? 8080;
const buildsDir = resolve(config.llama_cpp_builds_dir ?? join(__dirname, "..", "llama-builds"));

configureLogging(resolve(config.log_dir ?? join(__dirname, "..", "logs")), process.env.LOG_LEVEL);

// Always detected now (previously only ran when config.backend was unset,
// which on a worker with an explicit config.json value -- the common case --
// meant this never ran at all) since Run.backend_device_name's Tier-1
// fallback (see below) needs it regardless of whether backend itself came
// from config.json or auto-detection.
const detectedHardware = await detectHardware();

// backend is optional in config.json -- when unset, detect it live from
// this machine's actual GPU rather than requiring it be hardcoded. An
// explicit value in config.json always wins (e.g. to force `cpu` on a box
// that does have a GPU).
const backend: Backend = config.backend ?? detectBackend(detectedHardware.platform, detectedHardware.gpu);
if (!config.backend) {
  log.info(`no backend set in config.json -- auto-detected "${backend}" from hardware`);
}

// Tier-1 backend_device_name fallback (see shared/types.ts's
// Run.backend_device_name) -- always available, reported via /health below.
// Upgraded opportunistically once the first non-MTP sweep item completes
// (see worker/src/index.ts's runSweepItem, which sends bench.gpu_info --
// llama-bench's own exact device-name string -- as a Tier-2 upgrade).
const backendDeviceNameFallback: string | undefined =
  backend === "cpu"
    ? "CPU"
    : (detectedHardware.gpu[0]?.model || detectedHardware.gpu[0]?.vendor || undefined);

log.info(
  `[worker ${config.worker_name}] starting (backend=${backend}, build=${config.llama_cpp_build}, log level=${
    process.env.LOG_LEVEL ?? "info"
  })`
);

// Best-effort warm-up of the VRAM-reading path (worker/src/vram.ts). On
// Windows the generic backend's free/used reading spawns `powershell.exe`
// running Get-Counter against the "GPU Adapter Memory" category -- confirmed
// live that the *very first* invocation of that pipeline in a freshly
// started worker process pays real PowerShell-startup + performance-counter
// -provider cold-start cost that reliably exceeds vram.ts's own
// EXEC_TIMEOUT_MS (5s), silently returning "unavailable" for that one call
// only (a run's item 0's vram_free_before, specifically -- every later item
// succeeds instantly once the provider is warm). Firing this once here,
// before any real run can plausibly be triggered, means the first genuine
// captureFreeMemoryBaseline call (sampler.ts, used by runSweepItem/
// runSweepItemViaServer below) already hits a warm path instead of paying
// that cost itself. Fire-and-forget and swallow all errors: this must never
// delay /health or /run from becoming available, and a failed warm-up just
// means the first real call pays the same cost this was trying to avoid --
// no worse than before this existed.
void readGpuMemory(backend, undefined).catch(() => {});

// The active build is mutable at runtime (via /llama-cpp/activate) so a
// version switch from the web UI takes effect on the next /run without
// restarting this process. It starts from config.json but is persisted back
// to config.json on every switch so a restart doesn't revert it.
interface ActiveBuild {
  tag: string;
  path: string;
  serverPath?: string;
}

// Only used for the config.json-seeded path below -- a build activated via
// the installed-build registry already carries its own server_path (see
// worker/src/llama-builds.ts's findServerBinary), found the same way at
// install time. This covers a llama_bench_path pinned by hand, predating
// that registry.
function deriveServerPath(benchPath: string): string | null {
  const dir = dirname(benchPath);
  for (const name of ["llama-server.exe", "llama-server"]) {
    const candidate = join(dir, name);
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

let activeBuild: ActiveBuild | null =
  config.llama_bench_path && existsSync(config.llama_bench_path)
    ? {
        tag: config.llama_cpp_build,
        path: config.llama_bench_path,
        serverPath:
          (config.llama_server_path && existsSync(config.llama_server_path)
            ? config.llama_server_path
            : deriveServerPath(config.llama_bench_path)) ?? undefined,
      }
    : null;

if (!activeBuild) {
  log.warn(
    `no active llama-bench build (llama_bench_path missing or not found: ` +
      `${config.llama_bench_path}). Install one via the Workers page before triggering a run.`
  );
}

function persistConfig(patch: Partial<WorkerConfig>): void {
  const merged = { ...config, ...patch };
  const tmp = `${configPath}.tmp`;
  writeFileSync(tmp, JSON.stringify(merged, null, 2), "utf8");
  renameSync(tmp, configPath);
  Object.assign(config, patch);
}

function resolveModelPath(model: Model): string {
  const name = model.source === "local" ? model.filename : model.hf_file ?? model.filename;
  if (!name) {
    throw new Error(`model ${model.id} has no filename to resolve (source=${model.source})`);
  }
  const root = resolve(config.model_dir);
  const resolved = resolve(root, name);
  const rel = relative(root, resolved);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error(`model filename resolves outside model_dir: ${name}`);
  }
  return resolved;
}

// Same containment rule as resolveModelPath, applied to a raw filename
// coming from an HF download request rather than a registered Model.
function resolveDownloadTarget(hfFile: string): string {
  const root = resolve(config.model_dir);
  const resolved = resolve(root, hfFile);
  const rel = relative(root, resolved);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error(`hf_file resolves outside model_dir: ${hfFile}`);
  }
  return resolved;
}

interface ModelDirFile {
  path: string;
  size_bytes: number;
}

// Walks model_dir recursively so the orchestrator can learn "which files does
// this worker actually have" in one call instead of asking about one
// filename at a time (see server/src/routes/models.ts's /api/models/locations)
// -- also how a Gemma-4-style MTP companion sitting in an "MTP/" subfolder
// (see resolveModelPath's `name` handling) is discovered at all. Skips
// raw_json_dir when it happens to be nested inside model_dir (not the case
// in either shipped config, but no reason to list bench output as if it were
// a model file).
function listModelDirFiles(): ModelDirFile[] {
  const root = resolve(config.model_dir);
  if (!existsSync(root)) return [];
  const rawDir = resolve(config.raw_json_dir ?? join(root, "raw"));
  const results: ModelDirFile[] = [];
  function walk(dir: string): void {
    if (resolve(dir) === rawDir) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile()) {
        const rel = relative(root, full).split(sep).join("/");
        results.push({ path: rel, size_bytes: statSync(full).size });
      }
    }
  }
  walk(root);
  return results;
}

// Where writeRawJson (worker/src/vps-client.ts) actually lands a given raw
// dump -- bucketed by outcome so pruneRawJson below can apply a different
// retention window to each without needing to ask the VPS "was this item a
// failure" after the fact (the worker keeps no local run history otherwise).
function rawJsonSubdir(rawJsonDir: string, outcome: "ok" | "failed"): string {
  return join(rawJsonDir, outcome);
}

const DEFAULT_RAW_JSON_RETENTION_DAYS_OK = 14;

// Deletes files older than retentionDays from one raw-json bucket (ok/ or
// failed/) by mtime -- best-effort, a file that vanishes mid-scan (e.g. a
// concurrent write, vanishingly unlikely here since nothing else touches
// this directory) or can't be stat'd is just skipped, not fatal.
function pruneRawJsonDir(dir: string, retentionDays: number): void {
  if (!existsSync(dir)) return;
  const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000;
  let deleted = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const full = join(dir, entry.name);
    try {
      if (statSync(full).mtimeMs < cutoff) {
        unlinkSync(full);
        deleted++;
      }
    } catch {
      // vanished or unreadable between listing and stat/unlink -- skip it
    }
  }
  if (deleted > 0) {
    log.info(`pruned ${deleted} raw JSON file(s) from ${dir} (older than ${retentionDays}d)`);
  }
}

// Successful items default to a bounded retention (their numbers are
// already durably in the DB -- the raw dump is only useful for a short
// window of "let me double-check something"). Failed items default to
// never auto-pruned (raw_json_retention_days_failed unset): they're the
// main reason these files exist at all, and failures are the exception,
// not the rule, so they don't grow unbounded the way successful-run dumps
// would over months of active benchmarking.
function pruneRawJson(): void {
  const rawDir = resolve(config.raw_json_dir ?? join(config.model_dir, "raw"));
  pruneRawJsonDir(rawJsonSubdir(rawDir, "ok"), config.raw_json_retention_days ?? DEFAULT_RAW_JSON_RETENTION_DAYS_OK);
  if (config.raw_json_retention_days_failed != null) {
    pruneRawJsonDir(rawJsonSubdir(rawDir, "failed"), config.raw_json_retention_days_failed);
  }
}

const HF_REPO_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

function validateHfRepo(repo: string): void {
  if (!HF_REPO_PATTERN.test(repo)) throw new Error(`invalid hf_repo: ${repo}`);
}

function validateHfFile(file: string): void {
  if (!file || file.includes("..") || file.startsWith("/") || file.includes("\0")) {
    throw new Error(`invalid hf_file: ${file}`);
  }
}

function hfResolveUrl(repo: string, file: string): string {
  const encRepo = repo.split("/").map(encodeURIComponent).join("/");
  const encFile = file.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${encRepo}/resolve/main/${encFile}`;
}

class HashingPassThrough extends Transform {
  private hash = createHash("sha256");
  private bytes = 0;
  override _transform(
    chunk: Buffer,
    _enc: string,
    cb: (err?: Error | null, data?: unknown) => void
  ): void {
    this.hash.update(chunk);
    this.bytes += chunk.length;
    cb(undefined, chunk);
  }
  digestHex(): string {
    return this.hash.digest("hex");
  }
  get byteLength(): number {
    return this.bytes;
  }
}

const MAX_RUN_BODY_BYTES = 5 * 1024 * 1024;
const MAX_ADMIN_BODY_BYTES = 64 * 1024;
// How often a running item's live progress tick (phase + ram/vram) is sent.
const TICK_INTERVAL_MS = 2000;
// Ticks fire every TICK_INTERVAL_MS but only reach the VPS -- a worker log
// watcher (pm2 logs, the daily log file) otherwise sees nothing between
// "spawning llama-bench" and process-close, which for a long CPU sweep can be
// tens of minutes of apparent silence. Mirror one tick in HEARTBEAT_EVERY_N
// to the worker's own log so it's visible without querying the API.
const HEARTBEAT_EVERY_N = 10; // 10 * TICK_INTERVAL_MS = 20s
// Fallback only: when a build's llama-bench predates --progress (see
// bench.ts's supportsProgressFlag) there's no structured "model loaded, now
// benchmarking" signal at all (stdout is fully buffered until exit with -o
// json), so the loading->benchmarking transition is inferred from stderr
// going quiet for this long instead. Ignored entirely once a real --progress
// marker line has been seen for the item (see runSweepItem's
// usedProgressMarkers flag) -- a real signal always wins over the heuristic.
const QUIET_PERIOD_MS = 1800;
// Local port the worker binds llama-server to for MTP benchmarking (see
// worker/src/serverBench.ts). Only one benchmark ever runs at a time per
// worker (the `busy` flag below enforces that), so a fixed port carries no
// real collision risk -- config.mtp_server_port overrides it if needed.
const DEFAULT_MTP_SERVER_PORT = 8899;

let busy = false;

interface CurrentRun {
  run_id: string;
  model_filename: string;
  started_at: number;
}
let currentRun: CurrentRun | null = null;

// Run-level pause/stop control -- reset at the start of every /run call
// (see the run_id.startsWith check in the pause/resume/stop handlers below,
// which only accept a run_id matching currentRun so a stale request from a
// previous run can't affect a new one). llama-bench runs every repeat (-r)
// of one sweep item inside a single process invocation, so there's no safe
// way to pause mid-item without corrupting that repeat's timing -- pause
// only takes effect between items (see the /run loop below). Stop kills
// the in-flight process immediately via the same SIGKILL path bench.ts's
// own timeout uses.
let pauseRequested = false;
let stopRequested = false;
let activeBenchProc: import("node:child_process").ChildProcess | null = null;

interface DownloadProgress {
  bytes: number;
  total: number | null;
  started_at: number;
}

// Keyed by "hf_repo/hf_file" so a concurrent poll from the browser (a
// separate HTTP connection from the download's own long-lived request) can
// look up the same in-flight transfer. Entries are removed once the
// download settles either way, so a stale/finished key just 404s.
const downloadProgress = new Map<string, DownloadProgress>();

function readBody(req: IncomingMessage, maxBytes: number): Promise<string> {
  return new Promise((resolvePromise, reject) => {
    let body = "";
    let tooLarge = false;
    req.on("data", (c) => {
      if (tooLarge) return;
      body += c;
      if (Buffer.byteLength(body) > maxBytes) {
        tooLarge = true;
        reject(Object.assign(new Error("request body too large"), { statusCode: 413 }));
        req.destroy();
      }
    });
    req.on("error", (err) => reject(err));
    req.on("end", () => {
      if (!tooLarge) resolvePromise(body);
    });
  });
}

function toInstalledBuildList(): InstalledBuild[] {
  return listInstalledBuilds(buildsDir).map((b) => ({
    tag: b.tag,
    asset_name: b.asset_name,
    installed_at: b.installed_at,
    active: activeBuild?.tag === b.tag,
  }));
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://internal");

  if (req.method === "POST" && url.pathname === "/run") {
    if (busy) {
      log.warn("rejected /run: worker is already running a benchmark");
      return send(res, 409, { error: "worker is already running a benchmark" });
    }
    if (!activeBuild) {
      log.warn("rejected /run: no active llama-bench build installed");
      return send(res, 400, {
        error: "no active llama-bench build installed -- install one via the Workers page first",
      });
    }
    busy = true;
    let release: (() => void) | null = () => {
      busy = false;
      release = null;
    };

    let body = "";
    let tooLarge = false;
    req.on("data", (c) => {
      if (tooLarge) return;
      body += c;
      if (Buffer.byteLength(body) > MAX_RUN_BODY_BYTES) {
        tooLarge = true;
        release?.();
        send(res, 413, { error: "request body too large" });
        req.destroy();
      }
    });
    req.on("error", () => {
      release?.();
    });
    req.on("end", async () => {
      if (tooLarge) return;

      let reqBody: {
        run_id: string;
        model_id: string;
        model: Model;
        // Companion --model-draft file for a Gemma-4-style base model whose
        // MTP head isn't baked into `model` itself -- only present when the
        // sweep has an "mtp: on" item and the server resolved one (see
        // server/src/routes/runs.ts). Absent for Qwen-style in-file MTP.
        mtp_model?: Model;
        sweep: Omit<SweepConfig, "model_id">;
        llama_cpp_build: string;
        llama_cpp_backend: Backend;
      };
      let modelPath: string;
      let mtpModelPath: string | undefined;
      try {
        reqBody = JSON.parse(body);
        log.info(
          `run ${reqBody.run_id} received: model=${reqBody.model?.filename ?? reqBody.model_id} ` +
            `build=${reqBody.llama_cpp_build} backend=${reqBody.llama_cpp_backend}`
        );
        log.debug(`run ${reqBody.run_id} sweep: ${JSON.stringify(reqBody.sweep)}`);

        modelPath = resolveModelPath(reqBody.model);
        if (!existsSync(modelPath)) {
          throw new Error(`model file not found at ${modelPath} (source=${reqBody.model.source})`);
        }
        if (reqBody.mtp_model) {
          mtpModelPath = resolveModelPath(reqBody.mtp_model);
          if (!existsSync(mtpModelPath)) {
            throw new Error(`mtp model file not found at ${mtpModelPath} (source=${reqBody.mtp_model.source})`);
          }
        }
      } catch (err) {
        release?.();
        const message = err instanceof Error ? err.message : String(err);
        log.warn(`run rejected: ${message}`);
        return send(res, 400, { error: message });
      }

      // From here on the request is accepted -- ack immediately and work
      // through the sweep in the background, one llama-bench process per
      // parameter combination (see shared/sweep.ts's expandSweep) rather
      // than one CSV-args call for the whole sweep. The VPS learns each
      // item's outcome individually via /api/runs/:id/items/:idx as it
      // happens, not via one final call at the end -- previously this
      // handler held the connection open for the whole bench (up to
      // bench_timeout_ms, tens of minutes), so any timeout or network
      // hiccup on that one long-lived request left the VPS marking the run
      // "failed" while this worker kept working -- with `busy` still
      // (correctly) true and no way to reconcile the two, so the next
      // trigger attempt would 409. Per-item reporting also means a crash on
      // one combination doesn't cost the rest of the sweep: this loop keeps
      // going regardless of how any individual item turns out.
      release = null;
      const items = expandSweep(reqBody.sweep);
      currentRun = { run_id: reqBody.run_id, model_filename: reqBody.model.filename, started_at: Date.now() };
      pauseRequested = false;
      stopRequested = false;
      activeBenchProc = null;
      send(res, 202, { ok: true, run_id: reqBody.run_id, status: "started", items: items.length });

      log.info(`run ${reqBody.run_id}: ${items.length} test(s) to run`);
      const rawDir = config.raw_json_dir ?? join(config.model_dir, "raw");
      try {
        for (const item of items) {
          // Checked before *every* item (not just once) since pausing can
          // hold here for a while -- a stop requested while paused must
          // still take effect immediately rather than waiting out the pause.
          if (stopRequested) {
            log.info(`run ${reqBody.run_id}: stopped, cancelling item ${item.idx} (not started)`);
            await safeItemTerminal(reqBody.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
            continue;
          }
          while (pauseRequested && !stopRequested) {
            await sleep(500);
          }
          if (stopRequested) {
            log.info(`run ${reqBody.run_id}: stopped while paused, cancelling item ${item.idx} (not started)`);
            await safeItemTerminal(reqBody.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
            continue;
          }
          if (item.mtp === "on") {
            if (!activeBuild!.serverPath) {
              log.error(`run ${reqBody.run_id}: item ${item.idx} needs mtp but the active build has no llama-server binary`);
              await safeItemTerminal(reqBody.run_id, item.idx, {
                status: "failed",
                error: "active llama.cpp build has no llama-server binary -- reinstall the build to pick one up",
              });
              continue;
            }
            await runSweepItemViaServer({
              runId: reqBody.run_id,
              item,
              repeats: reqBody.sweep.repeats,
              modelPath,
              mtpModelPath,
              llamaServerPath: activeBuild!.serverPath,
              port: config.mtp_server_port ?? DEFAULT_MTP_SERVER_PORT,
              backend,
              timeoutMs: config.bench_timeout_ms,
              rawJsonDir: rawDir,
            });
          } else {
            await runSweepItem({
              runId: reqBody.run_id,
              item,
              repeats: reqBody.sweep.repeats,
              modelPath,
              llamaBenchPath: activeBuild!.path,
              backend,
              timeoutMs: config.bench_timeout_ms,
              vpsUrl: config.vps_url,
              rawJsonDir: rawDir,
            });
          }
        }
        log.info(`run ${reqBody.run_id} finished all ${items.length} test(s)${stopRequested ? " (stopped early)" : ""}`);
      } finally {
        busy = false;
        currentRun = null;
        pauseRequested = false;
        stopRequested = false;
        activeBenchProc = null;
      }
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/run/pause") {
    if (!busy) return send(res, 409, { error: "worker is not running a benchmark" });
    pauseRequested = true;
    log.info(`run ${currentRun?.run_id ?? "?"}: pause requested -- takes effect before the next sweep item ` +
      `(the in-flight one, including all its repeats, runs to completion first)`);
    return send(res, 200, { ok: true, paused: true });
  }

  if (req.method === "POST" && url.pathname === "/run/resume") {
    if (!busy) return send(res, 409, { error: "worker is not running a benchmark" });
    pauseRequested = false;
    log.info(`run ${currentRun?.run_id ?? "?"}: resumed`);
    return send(res, 200, { ok: true, paused: false });
  }

  if (req.method === "POST" && url.pathname === "/run/stop") {
    if (!busy) return send(res, 409, { error: "worker is not running a benchmark" });
    stopRequested = true;
    pauseRequested = false;
    log.info(`run ${currentRun?.run_id ?? "?"}: stop requested, killing in-flight process if any`);
    activeBenchProc?.kill("SIGKILL");
    return send(res, 200, { ok: true, stopping: true });
  }

  if (req.method === "GET" && url.pathname === "/health") {
    return send(res, 200, {
      ok: true,
      worker: config.worker_name,
      busy,
      paused: pauseRequested,
      current_run: currentRun,
      backend,
      backend_device_name: backendDeviceNameFallback ?? null,
      active_build: activeBuild?.tag ?? null,
    });
  }

  if (req.method === "GET" && url.pathname === "/hardware") {
    try {
      const hw = await detectHardware();
      return send(res, 200, hw);
    } catch (err) {
      return send(res, 500, { error: err instanceof Error ? err.message : String(err) });
    }
  }

  if (req.method === "GET" && url.pathname === "/llama-cpp") {
    return send(res, 200, {
      platform: detectPlatform(),
      arch: process.arch,
      backend,
      installed: toInstalledBuildList(),
      active_tag: activeBuild?.tag ?? null,
    });
  }

  if (req.method === "POST" && url.pathname === "/llama-cpp/install") {
    if (busy) return send(res, 409, { error: "worker is running a benchmark, try again after it finishes" });
    try {
      const body = JSON.parse(await readBody(req, MAX_ADMIN_BODY_BYTES)) as {
        tag?: string;
        asset_name?: string;
        download_url?: string;
      };
      if (!body.tag || !body.asset_name || !body.download_url) {
        return send(res, 400, { error: "tag, asset_name, download_url are required" });
      }
      log.info(`installing llama.cpp build ${body.tag} (${body.asset_name})`);
      const installed = await installBuild({
        buildsDir,
        tag: body.tag,
        assetName: body.asset_name,
        downloadUrl: body.download_url,
      });
      log.info(`installed llama.cpp build ${body.tag} -> ${installed.bench_path}`);
      return send(res, 201, { ok: true, build: installed });
    } catch (err) {
      log.error(`llama-cpp install failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  if (req.method === "POST" && url.pathname === "/llama-cpp/activate") {
    if (busy) return send(res, 409, { error: "worker is running a benchmark, try again after it finishes" });
    try {
      const body = JSON.parse(await readBody(req, MAX_ADMIN_BODY_BYTES)) as { tag?: string };
      if (!body.tag) return send(res, 400, { error: "tag is required" });
      validateTag(body.tag);
      const build = getInstalledBuild(buildsDir, body.tag);
      if (!build) return send(res, 404, { error: `build ${body.tag} is not installed` });
      activeBuild = { tag: build.tag, path: build.bench_path, serverPath: build.server_path };
      persistConfig({
        llama_cpp_build: build.tag,
        llama_bench_path: build.bench_path,
        llama_server_path: build.server_path,
      });
      log.info(`activated llama.cpp build ${activeBuild.tag}`);
      return send(res, 200, { ok: true, active_tag: activeBuild.tag });
    } catch (err) {
      log.error(`llama-cpp activate failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  if (req.method === "DELETE" && url.pathname.startsWith("/llama-cpp/")) {
    if (busy) return send(res, 409, { error: "worker is running a benchmark, try again after it finishes" });
    const tag = decodeURIComponent(url.pathname.slice("/llama-cpp/".length));
    try {
      validateTag(tag);
      if (activeBuild?.tag === tag) {
        return send(res, 400, {
          error: "cannot delete the active build -- activate a different one first",
        });
      }
      deleteBuild(buildsDir, tag);
      log.info(`deleted llama.cpp build ${tag}`);
      return send(res, 200, { ok: true });
    } catch (err) {
      log.error(`llama-cpp delete failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  if (req.method === "POST" && url.pathname === "/models/download") {
    let body: { hf_repo?: string; hf_file?: string };
    try {
      body = JSON.parse(await readBody(req, MAX_ADMIN_BODY_BYTES));
      if (!body.hf_repo || !body.hf_file) {
        return send(res, 400, { error: "hf_repo and hf_file are required" });
      }
      validateHfRepo(body.hf_repo);
      validateHfFile(body.hf_file);
    } catch (err) {
      return sendError(res, err);
    }
    const hfRepo = body.hf_repo!;
    const hfFile = body.hf_file!;
    const progressKey = `${hfRepo}/${hfFile}`;
    if (downloadProgress.has(progressKey)) {
      return send(res, 409, { error: "download already in progress for that file" });
    }

    let target: string;
    try {
      target = resolveDownloadTarget(hfFile);
      mkdirSync(dirname(target), { recursive: true });
    } catch (err) {
      return sendError(res, err);
    }

    // Ack immediately and run the actual transfer in the background -- see
    // the /run handler above for the same fire-and-forget shape and why:
    // this used to block the whole HTTP response on the entire
    // download+hash+gguf-parse, which for a multi-GB file routinely outlived
    // not just this app's own configured MODEL_DOWNLOAD_TIMEOUT_MS but also
    // Node's undici default per-request timeout (5 minutes, independent of
    // any AbortSignal the caller passes) -- the orchestrator would then
    // report this worker as "inaccessible" and never register the model,
    // even though the file kept downloading here the whole time. The
    // outcome is now reported back explicitly via postModelDownloadResult
    // once it's known, success or failure, same pattern /run's per-item
    // terminal updates use.
    const downloadStartedAt = Date.now();
    // Seeded before the ack goes out, not after the upstream HF fetch
    // resolves like the old blocking handler did -- the orchestrator starts
    // polling /models/download/progress as soon as it sees this response,
    // so the entry needs to already exist to avoid a spurious "not found"
    // on the very first poll.
    const progress: DownloadProgress = { bytes: 0, total: null, started_at: downloadStartedAt };
    downloadProgress.set(progressKey, progress);
    log.info(`downloading ${progressKey} -> ${target}`);
    send(res, 202, { ok: true, hf_repo: hfRepo, hf_file: hfFile, status: "started" });

    void runModelDownload(hfRepo, hfFile, target, progressKey, progress, downloadStartedAt);
    return;
  }

  if (req.method === "GET" && url.pathname === "/models/download/progress") {
    const hfRepo = url.searchParams.get("hf_repo");
    const hfFile = url.searchParams.get("hf_file");
    if (!hfRepo || !hfFile) return send(res, 400, { error: "hf_repo and hf_file are required" });
    const progress = downloadProgress.get(`${hfRepo}/${hfFile}`);
    if (!progress) return send(res, 404, { error: "no download in progress for that file" });
    return send(res, 200, progress);
  }

  if (req.method === "GET" && url.pathname === "/models") {
    try {
      return send(res, 200, { files: listModelDirFiles() });
    } catch (err) {
      log.error(`models listing failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  if (req.method === "DELETE" && url.pathname === "/models") {
    const filename = url.searchParams.get("file");
    if (!filename) return send(res, 400, { error: "file is required" });
    try {
      validateHfFile(filename);
      const target = resolveDownloadTarget(filename);
      if (!existsSync(target)) {
        return send(res, 404, { error: `file not found: ${filename}` });
      }
      unlinkSync(target);
      log.info(`deleted model file ${filename}`);
      return send(res, 200, { ok: true });
    } catch (err) {
      log.error(`model file delete failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  // Reads GGUF metadata from a file already sitting in model_dir -- lets the
  // server backfill n_layer/mtp_layers for models that were registered/
  // downloaded before gguf.ts read one or the other, without re-downloading
  // anything.
  if (req.method === "GET" && url.pathname === "/models/gguf-info") {
    const filename = url.searchParams.get("file");
    if (!filename) return send(res, 400, { error: "file is required" });
    try {
      validateHfFile(filename);
      const target = resolveDownloadTarget(filename);
      if (!existsSync(target)) {
        return send(res, 404, { error: `file not found: ${filename}` });
      }
      const { n_layer, mtp_layers } = await readGgufInfo(target);
      return send(res, 200, { ok: true, n_layer, mtp_layers });
    } catch (err) {
      log.error(`gguf-info failed: ${err instanceof Error ? err.message : String(err)}`);
      return sendError(res, err);
    }
  }

  send(res, 404, { error: "not found" });
});

const ITEM_RETRY_DELAYS_MS = [2000, 4000, 8000, 16000, 32000];

function sleep(ms: number): Promise<void> {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

// Best-effort, same honest posture as this file's other diagnostic-only
// signals (see hardware.ts, sampler.ts's VRAM comment): a signal-killed
// process (and it wasn't our own bench_timeout_ms kill, tracked separately
// via BenchResult.timedOut) is what an OS OOM-killer looks like from here,
// and a handful of common allocator error phrases catch the rest. Won't be
// 100% precise -- some non-OOM context-creation failures may still just say
// "failed".
const OOM_STDERR_PATTERN =
  /out of memory|failed to allocate|cudaMalloc|insufficient memory|bad_alloc|not enough (memory|vram)/i;

function classifyFailure(bench: BenchResult): "failed_oom" | "failed" {
  if (!bench.timedOut && (bench.signal === "SIGKILL" || bench.signal === "SIGABRT")) {
    return "failed_oom";
  }
  if (OOM_STDERR_PATTERN.test(bench.stderr)) {
    return "failed_oom";
  }
  return "failed";
}

// The worker has no local record of a run once it's dispatched -- the VPS is
// the only place run/item status lives. If a terminal update can't reach it,
// retry with backoff rather than silently dropping the outcome and leaving
// that item (and potentially the whole run) stuck "running" forever with no
// way to recover it short of manual DB surgery.
async function safeItemTerminal(
  runId: string,
  idx: number,
  payload: RunItemTerminalInput
): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      await postRunItemUpdate(config.vps_url, runId, idx, payload, 10_000);
      log.info(`item update ok for run ${runId} item ${idx} (status=${payload.status}, attempt ${attempt + 1})`);
      return;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (attempt >= ITEM_RETRY_DELAYS_MS.length) {
        log.error(
          `item update failed for run ${runId} item ${idx} after ${attempt + 1} attempts, giving up: ${message}`
        );
        return;
      }
      const delay = ITEM_RETRY_DELAYS_MS[attempt];
      log.warn(
        `item update attempt ${attempt + 1} failed for run ${runId} item ${idx}, retrying in ${delay}ms: ${message}`
      );
      await sleep(delay);
    }
  }
}

// Fire-and-forget: losing one tick is harmless (the next tick or the
// terminal update above catches the VPS up), so this deliberately doesn't
// retry or block the caller on the network round trip.
function sendTick(runId: string, idx: number, tick: RunItemTickInput): void {
  postRunItemUpdate(config.vps_url, runId, idx, tick, 3000).catch((err) => {
    log.debug(
      `tick failed for run ${runId} item ${idx} (non-fatal): ${err instanceof Error ? err.message : String(err)}`
    );
  });
}

// Same retry-with-backoff posture as safeItemTerminal above -- this is the
// only thing that gets a completed download registered as a Model
// server-side, so losing it silently would leave a fully downloaded file on
// disk the app never learns about.
async function safeReportDownloadResult(payload: ModelDownloadCallbackInput): Promise<void> {
  const key = `${payload.hf_repo}/${payload.hf_file}`;
  for (let attempt = 0; ; attempt++) {
    try {
      await postModelDownloadResult(config.vps_url, payload, 10_000);
      log.info(`download callback ok for ${key} (ok=${payload.ok}, attempt ${attempt + 1})`);
      return;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (attempt >= ITEM_RETRY_DELAYS_MS.length) {
        log.error(`download callback failed for ${key} after ${attempt + 1} attempts, giving up: ${message}`);
        return;
      }
      const delay = ITEM_RETRY_DELAYS_MS[attempt];
      log.warn(`download callback attempt ${attempt + 1} failed for ${key}, retrying in ${delay}ms: ${message}`);
      await sleep(delay);
    }
  }
}

// Runs the actual transfer+hash+gguf-parse in the background after
// POST /models/download has already acked -- see that handler for why.
// Deliberately keeps the progress-map entry alive through the gguf parse and
// the callback POST below, not just the byte transfer: the orchestrator's
// client polls this file's progress to know when to stop watching it and
// only registers the model in its DB once the callback arrives, so clearing
// the entry any earlier would let a poll see "no longer downloading" and
// refresh the model list before the model actually exists there. See
// client/src/pages/Models.tsx's finalizeDownload for the other half.
async function runModelDownload(
  hfRepo: string,
  hfFile: string,
  target: string,
  progressKey: string,
  progress: DownloadProgress,
  downloadStartedAt: number
): Promise<void> {
  try {
    const sourceUrl = hfResolveUrl(hfRepo, hfFile);
    const upstream = await fetch(sourceUrl, {
      headers: { "user-agent": "llamatoaster-worker" },
      redirect: "follow",
    });
    if (!upstream.ok || !upstream.body) {
      throw new Error(`download failed: ${upstream.status} ${upstream.statusText}`);
    }

    const contentLength = Number(upstream.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > 0) {
      progress.total = contentLength;
    }
    const tracker = new Transform({
      transform(chunk: Buffer, _enc, cb) {
        progress.bytes += chunk.length;
        cb(null, chunk);
      },
    });

    const hasher = new HashingPassThrough();
    await pipeline(Readable.fromWeb(upstream.body as any), tracker, hasher, createWriteStream(target));
    const elapsedMs = Date.now() - downloadStartedAt;
    log.info(`downloaded ${progressKey}: ${hasher.byteLength}B in ${elapsedMs}ms`);

    const { n_layer, mtp_layers } = await readGgufInfo(target);
    log.info(
      `gguf metadata for ${progressKey}: n_layer=${n_layer ?? "unknown"} mtp_layers=${mtp_layers ?? "unknown"}`
    );

    await safeReportDownloadResult({
      worker: config.worker_name,
      hf_repo: hfRepo,
      hf_file: hfFile,
      ok: true,
      sha256: hasher.digestHex(),
      size_bytes: hasher.byteLength,
      n_layer,
      mtp_layers,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error(`model download failed for ${progressKey}: ${message}`);
    await safeReportDownloadResult({
      worker: config.worker_name,
      hf_repo: hfRepo,
      hf_file: hfFile,
      ok: false,
      error: message,
    });
  } finally {
    downloadProgress.delete(progressKey);
  }
}

interface RunSweepItemInput {
  runId: string;
  item: SweepItem;
  repeats: number;
  modelPath: string;
  llamaBenchPath: string;
  backend: Backend;
  timeoutMs: number | undefined;
  vpsUrl: string;
  rawJsonDir: string;
}

// Marker lines llama-bench prints to stderr when run with --progress (see
// worker/src/bench.ts's supportsProgressFlag) -- e.g.
// "llama-bench: benchmark 1/1: prompt run 2/3". Confirmed against upstream
// llama-bench.cpp; there's no live in-repeat percentage or tokens/sec, only
// these repeat-boundary markers, so live t/s below is derived by timing the
// gap between them rather than read directly.
const PP_RUN_RE = /benchmark\s+\d+\/\d+:\s+prompt run\s+(\d+)\/(\d+)/i;
const TG_RUN_RE = /benchmark\s+\d+\/\d+:\s+generation run\s+(\d+)\/(\d+)/i;
const WARMUP_RE = /benchmark\s+\d+\/\d+:\s+warmup prompt run/i;
const STARTING_RE = /benchmark\s+\d+\/\d+:\s+starting/i;

// llama.cpp's own one-line model-load summary, printed once *per model
// loaded* by either binary once that model's tensor loading finishes --
// confirmed live against the real binaries across -ngl 0 ("0/43"), a partial
// value ("10/43", matching the per-layer "layer N assigned to device" lines
// it also prints at this same verbosity), and full offload ("43/43"). X =
// gpu_layers_loaded, Y = total_model_layers -- Y is the GGUF's real
// transformer layer count + 1 (the output layer), not the same number as
// models.metadata.n_layer, so this is the only correct source for either
// value. Requires -v (llama-bench, see worker/src/bench.ts's
// supportsVerboseFlag) or -lv 4 (llama-server, see worker/src/serverBench.ts's
// buildArgs) -- absent at default verbosity on both binaries, confirmed live.
// `g` flag so parseOffloadLayers below can
// collect every occurrence, not just the first -- an MTP item loads TWO
// models (base + --model-draft companion) via llama-server, each printing
// its own line.
const OFFLOAD_LAYERS_RE = /load_tensors: offloaded (\d+)\/(\d+) layers to GPU/g;

interface OffloadInfo {
  gpu_layers_loaded: number;
  total_model_layers: number;
}

interface OffloadResult {
  // The base/target model's offload -- always the model at `modelPath`, and
  // (for a non-MTP item) the only model loaded at all.
  main: OffloadInfo | null;
  // The MTP/--model-draft companion's own offload, only meaningful when
  // hasMtpDraft is true. Null when hasMtpDraft is false, or when only one
  // "load_tensors: offloaded" line was actually captured (e.g. a build that
  // doesn't log the draft model's own load at this verbosity) -- there's
  // nothing to disambiguate from in that case, so the single match is
  // attributed to main and draft is left unset rather than guessed.
  draft: OffloadInfo | null;
}

// A plain llama-bench item (bench.ts's path) always loads exactly one model,
// so a single "offloaded X/Y" line is unambiguous. An MTP item run through
// llama-server (serverBench.ts's path, hasMtpDraft true) loads two: the base
// model and its --model-draft companion, each printing its own line. Simply
// taking the first match here (the previous behavior) silently reported
// whichever model's line happened to print first as if it were the base
// model's -- confirmed live against real MTP run output that this is
// actually the *draft* model's line, not the base model's, so
// gpu_layers_loaded/total_model_layers on every MTP row previously showed
// the tiny draft head's own e.g. "5/5" instead of the real base model's e.g.
// "43/43". Disambiguated here by total layer count instead of match
// position, which is robust regardless of which model's line prints first:
// an MTP draft/companion model is, by definition and by how much smaller
// speculative-decoding heads are than the base models they're paired with,
// always the one with far fewer transformer layers -- so of at most two
// matches, the larger total_model_layers is always the base model's line and
// the smaller is always the draft's.
function parseOffloadLayers(stderr: string, hasMtpDraft: boolean): OffloadResult {
  const matches: OffloadInfo[] = [...stderr.matchAll(OFFLOAD_LAYERS_RE)].map((m) => ({
    gpu_layers_loaded: Number(m[1]),
    total_model_layers: Number(m[2]),
  }));
  if (matches.length === 0) return { main: null, draft: null };
  if (!hasMtpDraft || matches.length < 2) {
    const main = matches.reduce((a, b) => (b.total_model_layers > a.total_model_layers ? b : a));
    return { main, draft: null };
  }
  const [main, draft] = [...matches].sort((a, b) => b.total_model_layers - a.total_model_layers);
  return { main, draft };
}

// llama.cpp's own diagnostic stderr (device detection, model metadata,
// tensor-offload lines, timing breakdown) was previously only ever saved to
// the per-item raw JSON dump (writeRawJson below) -- nothing mirrored it
// into the worker's own text log, so following along live (pm2 logs, the
// daily log file under logs/) only ever showed this app's own progress
// lines, never llama.cpp's. logDiagnosticOutput mirrors it in, filtered
// defensively: llama-server's MTP path runs at -lv 4 (see serverBench.ts's
// buildArgs) rather than the max level 5 ("debug", which prints a
// per-token/per-draft-candidate trace -- confirmed live to be unnecessary
// noise for anything this app reads), but level 4 ("trace") is still verbose
// enough that a request/response body -- including this app's own synthetic
// filler-token prompts and the model's generated text -- could plausibly
// appear verbatim in a line at that verbosity. Genuine llama.cpp diagnostic
// lines are short, structured,
// human-authored strings; a dumped prompt/response array or generated-text
// blob is not, so any line past a generous length threshold is elided
// rather than printed in full -- this trades away logging an unusually long
// *legitimate* diagnostic line (none observed live against either binary)
// for never printing raw prompt/reply content into the log.
const MAX_LOGGED_LINE_CHARS = 300;

function filterDiagnosticOutput(stderr: string): string {
  return stderr
    .split("\n")
    .map((line) =>
      line.length > MAX_LOGGED_LINE_CHARS
        ? `${line.slice(0, MAX_LOGGED_LINE_CHARS)} …[${
            line.length - MAX_LOGGED_LINE_CHARS
          } more chars elided -- see the raw JSON dump for the full line]`
        : line
    )
    .filter((line) => line.trim().length > 0)
    .join("\n");
}

function logDiagnosticOutput(label: string, processName: string, stderr: string): void {
  const filtered = filterDiagnosticOutput(stderr);
  if (!filtered) return;
  log.info(`${label}: ${processName} output:\n${filtered}`);
}

// Maps a finished BenchResult into a terminal run-item report -- shared by
// both the llama-bench path (runSweepItem) and the llama-server/MTP path
// (runSweepItemViaServer below) since both produce the same BenchResult
// shape and this classification (success vs. OOM vs. plain failure vs.
// user-cancelled, see classifyFailure) doesn't care which process ran.
async function finalizeSweepItemResult(
  runId: string,
  item: SweepItem,
  label: string,
  processName: string,
  bench: BenchResult,
  stats: SampleStats,
  baseline: FreeMemoryBaseline,
  // Read from llama.cpp's own runtime output (see parseOffloadLayers above),
  // both null when the line was never seen at all (item failed before model
  // load finished, or a build too old to support the required verbosity
  // flag).
  offload: OffloadResult
): Promise<void> {
  if (bench.code === 0 && bench.results.length > 0) {
    // A single sweep item with both n_prompt and n_gen set (and no -pg)
    // produces two rows -- a pp-only row and a tg-only row -- both kept here
    // (previously only bench.results[0] was, silently discarding whichever
    // of pp/tg came second). GPU-memory fields inherit whatever
    // captureFreeMemoryBaseline/MemorySampler already resolved -- both
    // already collapse to null/"unavailable" on the cpu backend via
    // vram.ts's own readGpuMemory short-circuit, so no separate check is
    // needed here.
    const results: IngestResultInput[] = bench.results.map((r) => ({
      ...r,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
      ram_free_before_mib: baseline.ram_free_before_mib,
      vram_free_before_mib: baseline.vram_free_before_mib,
      system_memory_total_mb: baseline.system_memory_total_mib,
      gpu_memory_total_mb: baseline.gpu_memory_total_mib,
      gpu_memory_total_accuracy: baseline.gpu_memory_total_accuracy,
      gpu_memory_total_source: baseline.gpu_memory_total_source,
      gpu_memory_free_start_accuracy: baseline.vram_free_before_accuracy,
      gpu_memory_free_start_source: baseline.vram_free_before_source,
      gpu_memory_model_avg_accuracy: stats.vram_avg_accuracy,
      gpu_memory_model_avg_source: stats.vram_avg_source,
      gpu_memory_model_peak_accuracy: stats.vram_peak_accuracy,
      gpu_memory_model_peak_source: stats.vram_peak_source,
      gpu_layers_loaded: offload.main?.gpu_layers_loaded ?? null,
      total_model_layers: offload.main?.total_model_layers ?? null,
      gpu_layers_loaded_draft: offload.draft?.gpu_layers_loaded ?? null,
      total_model_layers_draft: offload.draft?.total_model_layers ?? null,
    }));
    log.info(
      `${label} done: ${results.map((r) => `${r.test_type}=${r.avg_tps.toFixed(2)}tok/s`).join(" ")} ` +
        `ram_peak=${stats.ram_peak_mib}MiB ram_avg=${stats.ram_avg_mib}MiB` +
        (stats.vram_peak_mib != null ? ` vram_peak=${stats.vram_peak_mib}MiB vram_avg=${stats.vram_avg_mib}MiB` : "")
    );
    // bench.warning (only ever set by the llama-server/MTP path) flags a
    // successful item that still had some readings rejected as implausible
    // -- surfaced via the same `error` field a failed item uses, since
    // recordRunItemTerminal stores it unconditionally regardless of status,
    // and RunDetail.tsx already prefers item.error over the normal detail
    // text for any status. Without this, a "done" item with a silently
    // dropped tg reading (the original bug report) looked identical to a
    // clean run.
    if (bench.warning) log.warn(`${label} completed with warnings: ${bench.warning}`);
    await safeItemTerminal(runId, item.idx, {
      status: "done",
      error: bench.warning,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
      results,
      // Tier-2 backend_device_name upgrade (see shared/types.ts's
      // Run.backend_device_name) -- undefined on the llama-server/MTP path,
      // which never sets bench.gpu_info (confirmed live: llama-server prints
      // no device-enumeration line at any verbosity), so this naturally
      // no-ops there without any extra branching.
      backend_device_name: bench.gpu_info,
    });
    return;
  }
  // A user-requested stop kills this exact process (SIGKILL, via /run/stop)
  // -- report it as "cancelled", not a genuine failure, so it doesn't read
  // as something having gone wrong.
  const status: "failed_oom" | "failed" | "cancelled" = stopRequested ? "cancelled" : classifyFailure(bench);
  const errorMessage = stopRequested
    ? "cancelled by user"
    : bench.code === 0
      ? `${processName} exited cleanly but produced no parseable result`
      : bench.stderr || `${processName} exited with code ${bench.code}`;
  log.error(`${label} ${status} (code ${bench.code}, signal ${bench.signal ?? "none"}): ${errorMessage}`);
  await safeItemTerminal(runId, item.idx, {
    status,
    error: errorMessage,
    ram_peak_mib: stats.ram_peak_mib,
    vram_peak_mib: stats.vram_peak_mib,
    ram_avg_mib: stats.ram_avg_mib,
    vram_avg_mib: stats.vram_avg_mib,
  });
}

// Runs exactly one sweep combination as its own llama-bench process, reports
// its progress live, and always resolves (never throws) regardless of
// outcome -- so the caller's loop over every item in the sweep can continue
// unconditionally instead of one bad combination aborting the rest.
async function runSweepItem(input: RunSweepItemInput): Promise<void> {
  const { runId, item, backend } = input;
  const label = `run ${runId} item ${item.idx}`;
  const itemStartedAt = Date.now();

  const testType = deriveTestType(item.n_prompt, item.n_gen);
  const benchmarkingPhase: RunItemTickInput["status"] =
    testType === "pp" ? "processing" : testType === "tg" ? "generating" : "benchmarking";

  const baseline = await captureFreeMemoryBaseline(backend);
  log.info(
    `${label}: free memory before start: ram=${baseline.ram_free_before_mib}MiB` +
      (baseline.vram_free_before_mib != null ? ` vram=${baseline.vram_free_before_mib}MiB` : "")
  );
  sendTick(runId, item.idx, {
    status: "loading",
    ram_free_before_mib: baseline.ram_free_before_mib,
    vram_free_before_mib: baseline.vram_free_before_mib,
  });

  const sampler = new MemorySampler();
  let phase: RunItemTickInput["status"] = "loading";
  let detail: string | undefined;
  let lastStderrAt = Date.now();
  let sawStderr = false;
  let tickTimer: NodeJS.Timeout | null = null;
  // Once a real --progress marker line has been seen, the quiet-period
  // fallback below is disabled for the rest of this item -- a real signal
  // always wins over the heuristic that exists only to cover builds without
  // --progress support.
  let usedProgressMarkers = false;
  let liveTps: number | undefined;
  let lastRepMarker: { kind: "pp" | "tg"; rep: number; at: number } | null = null;

  try {
    const bench = await runBench({
      modelPath: input.modelPath,
      item,
      repeats: input.repeats,
      llamaBenchPath: input.llamaBenchPath,
      backend,
      timeoutMs: input.timeoutMs,
      onSpawn: (proc) => {
        activeBenchProc = proc;
        sampler.start(proc.pid, backend, TICK_INTERVAL_MS);
        let tickCount = 0;
        tickTimer = setInterval(() => {
          if (!usedProgressMarkers && phase === "loading" && sawStderr && Date.now() - lastStderrAt > QUIET_PERIOD_MS) {
            phase = benchmarkingPhase;
            detail = undefined;
            log.debug(
              `${label}: stderr quiet for ${QUIET_PERIOD_MS}ms, moving to ${phase} (no --progress marker seen)`
            );
          }
          const current = sampler.current;
          tickCount++;
          if (tickCount % HEARTBEAT_EVERY_N === 0) {
            const elapsedS = Math.round((Date.now() - itemStartedAt) / 1000);
            log.info(
              `${label}: ${phase}${detail ? ` (${detail})` : ""} ram=${current.ram_mib ?? "?"}MiB ` +
                `elapsed=${elapsedS}s`
            );
          }
          sendTick(runId, item.idx, {
            status: phase,
            detail,
            ram_mib: current.ram_mib,
            vram_mib: current.vram_mib,
            live_tps: liveTps ?? null,
          });
        }, TICK_INTERVAL_MS);
      },
      onStderrLine: (line) => {
        sawStderr = true;
        lastStderrAt = Date.now();

        const ppMatch = PP_RUN_RE.exec(line);
        const tgMatch = TG_RUN_RE.exec(line);
        if (ppMatch || tgMatch) {
          usedProgressMarkers = true;
          const kind: "pp" | "tg" = ppMatch ? "pp" : "tg";
          const match = (ppMatch ?? tgMatch)!;
          const rep = Number(match[1]);
          const reps = Number(match[2]);
          const now = Date.now();
          if (lastRepMarker && lastRepMarker.kind === kind) {
            const elapsedS = (now - lastRepMarker.at) / 1000;
            const tokens = kind === "pp" ? item.n_prompt : item.n_gen;
            if (elapsedS > 0 && tokens > 0) liveTps = tokens / elapsedS;
          } else {
            // Phase just changed (pp -> tg) or this is the very first marker
            // -- without this, the old phase's liveTps value would otherwise
            // keep being reported for a tick or two under the *new* phase's
            // label (a stale pp reading shown as if it were tg) until a
            // same-phase gap is actually measured again.
            liveTps = undefined;
          }
          lastRepMarker = { kind, rep, at: now };
          phase = kind === "pp" ? "processing" : "generating";
          detail = `run ${rep}/${reps}`;
          const tpsSuffix = liveTps != null ? ` (${liveTps.toFixed(1)} t/s measured last repeat)` : "";
          log.info(`${label}: ${phase} run ${rep}/${reps}${tpsSuffix}`);
          return;
        }

        if (WARMUP_RE.test(line) || STARTING_RE.test(line)) {
          usedProgressMarkers = true;
          const isWarmup = WARMUP_RE.test(line);
          phase = "loading";
          detail = isWarmup ? "warmup" : undefined;
          log.info(`${label}: ${isWarmup ? "warmup prompt run" : "starting"}`);
          return;
        }

        // Fallback for builds without --progress support: surface real
        // loading-phase detail (model size, tensor load progress, whatever
        // this build actually prints) verbatim while it's still available,
        // same idea as a model-loading progress readout in an LLM chat
        // client. Only used pre-marker -- once usedProgressMarkers flips on,
        // unrecognized lines are ignored rather than overwriting detail.
        if (phase === "loading" && !usedProgressMarkers) detail = line.slice(0, 200);
      },
      log,
    });
    if (tickTimer) clearInterval(tickTimer);
    activeBenchProc = null;
    const stats = sampler.stop();
    // false: this is the non-MTP path, which only ever loads one model.
    const offload = parseOffloadLayers(bench.stderr, false);
    const outcome = bench.code === 0 && bench.results.length > 0 ? "ok" : "failed";
    writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-bench", bench.stderr);
    await finalizeSweepItemResult(runId, item, label, "llama-bench", bench, stats, baseline, offload);
  } catch (err) {
    if (tickTimer) clearInterval(tickTimer);
    activeBenchProc = null;
    const stats = sampler.stop();
    const message = err instanceof Error ? err.message : String(err);
    log.error(`${label} handler threw: ${message}`);
    await safeItemTerminal(runId, item.idx, {
      status: "failed",
      error: message,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
    });
  }
}

interface RunSweepItemViaServerInput {
  runId: string;
  item: SweepItem;
  repeats: number;
  modelPath: string;
  mtpModelPath: string | undefined;
  llamaServerPath: string;
  port: number;
  backend: Backend;
  timeoutMs: number | undefined;
  rawJsonDir: string;
}

// MTP-benchmarking sibling of runSweepItem above: same outer shape (baseline
// capture, memory sampling, raw output dump, shared finalization, always
// resolves rather than throwing), but drives llama-server over HTTP via
// worker/src/serverBench.ts instead of spawning llama-bench, since
// llama-bench has no MTP/speculative-decoding support at all. Progress
// reporting is simpler here: serverBench.ts drives its own repeat loop
// directly, so ticks come from a plain callback rather than regex-scraped
// subprocess stderr.
async function runSweepItemViaServer(input: RunSweepItemViaServerInput): Promise<void> {
  const { runId, item, backend } = input;
  const label = `run ${runId} item ${item.idx} (mtp)`;

  const baseline = await captureFreeMemoryBaseline(backend);
  log.info(
    `${label}: free memory before start: ram=${baseline.ram_free_before_mib}MiB` +
      (baseline.vram_free_before_mib != null ? ` vram=${baseline.vram_free_before_mib}MiB` : "")
  );
  sendTick(runId, item.idx, {
    status: "loading",
    ram_free_before_mib: baseline.ram_free_before_mib,
    vram_free_before_mib: baseline.vram_free_before_mib,
  });

  const sampler = new MemorySampler();
  try {
    const bench = await runServerBench({
      modelPath: input.modelPath,
      mtpModelPath: input.mtpModelPath,
      item,
      repeats: input.repeats,
      llamaServerPath: input.llamaServerPath,
      port: input.port,
      timeoutMs: input.timeoutMs,
      onSpawn: (proc) => {
        activeBenchProc = proc;
        sampler.start(proc.pid, backend, TICK_INTERVAL_MS);
      },
      onProgress: (phase, detail, liveTps) => {
        log.info(`${label}: ${phase} (${detail})${liveTps != null ? ` (${liveTps.toFixed(1)} t/s)` : ""}`);
        const current = sampler.current;
        sendTick(runId, item.idx, {
          status: phase,
          detail,
          ram_mib: current.ram_mib,
          vram_mib: current.vram_mib,
          live_tps: liveTps ?? null,
        });
      },
      log,
    });
    activeBenchProc = null;
    const stats = sampler.stop();
    // Only true for a Gemma-4-style base model with a separate --model-draft
    // companion file -- a Qwen-style in-file MTP model loads just the one
    // model, same as the non-MTP path (see ServerBenchRunInput.mtpModelPath).
    const offload = parseOffloadLayers(bench.stderr, Boolean(input.mtpModelPath));
    const outcome = bench.code === 0 && bench.results.length > 0 ? "ok" : "failed";
    writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-server", bench.stderr);
    await finalizeSweepItemResult(runId, item, label, "llama-server", bench, stats, baseline, offload);
  } catch (err) {
    activeBenchProc = null;
    const stats = sampler.stop();
    const message = err instanceof Error ? err.message : String(err);
    log.error(`${label} handler threw: ${message}`);
    await safeItemTerminal(runId, item.idx, {
      status: "failed",
      error: message,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
    });
  }
}

function send(res: any, code: number, data: unknown): void {
  res.statusCode = code;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(data));
}

function sendError(res: any, err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  const statusCode = (err as { statusCode?: number })?.statusCode ?? 400;
  send(res, statusCode, { error: message });
}

server.listen(PORT, config.bind_host, () => {
  log.info(`[worker ${config.worker_name}] listening on ${config.bind_host}:${PORT} (tailnet-only)`);
});

// Once at startup (covers a worker that's restarted often enough that a
// periodic timer alone might rarely fire) and then daily -- a long-lived
// pm2-managed process is the common case, but there's no need for this to
// run any more often than that for a "delete stuff older than N days" check.
const RAW_JSON_PRUNE_INTERVAL_MS = 24 * 60 * 60 * 1000;
pruneRawJson();
setInterval(pruneRawJson, RAW_JSON_PRUNE_INTERVAL_MS).unref();

function shutdown(signal: string): void {
  log.info(`received ${signal}, shutting down...`);
  server.close(() => process.exit(0));
  // Force-exit if a benchmark is still in flight and close() hangs.
  setTimeout(() => process.exit(1), 5000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
