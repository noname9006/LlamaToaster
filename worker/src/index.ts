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
import { runBench, matchOffloadLine, type BenchResult, type OffloadResult } from "./bench.js";
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
import {
  backendVisibleGpus,
  type Model,
  type SweepConfig,
  type IngestResultInput,
  type Backend,
  type InstalledBuild,
  type RunItemTickInput,
  type RunItemTerminalInput,
  type ModelDownloadCallbackInput,
  type TerminalRunItemStatus,
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

// Tier-1 backend_device_name (see shared/types.ts's Run.backend_device_name)
// -- reported via /health below, optionally scoped to a specific run's
// --main-gpu selection via that route's ?main_gpu= query param (only ever
// sent by server/src/routes/runs.ts's ensureActiveBuild, right before
// dispatching that exact run; every other /health poll -- e.g. the Workers
// page's generic card fetch -- omits it and gets this backend's first
// visible device instead). Upgraded opportunistically once the first
// non-MTP sweep item completes (see runSweepItem below, which sends
// bench.gpu_info -- llama-bench's own exact device-name string -- as a
// Tier-2 upgrade) -- Tier-2 has the same "lists every visible device, not
// just the selected one" caveat as this Tier-1 estimate on a backend that
// can see multiple GPUs at once (e.g. Vulkan), since that's what llama-bench
// itself reports regardless of -mg.
//
// Filtered through backendVisibleGpus rather than indexing detectedHardware
// .gpu directly: that array is in whatever order systeminformation happens
// to enumerate controllers, which has no relationship to which devices this
// backend can even see (e.g. an Intel iGPU isn't a CUDA device at all) --
// gpu[0] unfiltered could easily be a device this backend can't use.
function backendDeviceName(mainGpu: number | undefined): string | undefined {
  if (backend === "cpu") return "CPU";
  const visible = backendVisibleGpus(detectedHardware.gpu, backend);
  if (visible.length === 0) return undefined;
  const picked = mainGpu != null ? visible[mainGpu] : visible[0];
  return picked?.model || picked?.vendor || undefined;
}

log.info(
  `[worker ${config.worker_name}] starting (backend=${backend}, build=${config.llama_cpp_build}, log level=${
    process.env.LOG_LEVEL ?? "info"
  })`
);
log.info(
  `[worker ${config.worker_name}] OS: ${detectedHardware.platform} (${detectedHardware.arch})`
);
log.info(
  `[worker ${config.worker_name}] CPU: ${
    detectedHardware.cpu.brand || detectedHardware.cpu.manufacturer || "unknown"
  }${detectedHardware.cpu.cores ? ` (${detectedHardware.cpu.cores} threads)` : ""}`
);
log.info(
  `[worker ${config.worker_name}] RAM: ${
    detectedHardware.mem_total_bytes != null
      ? `${Math.round(detectedHardware.mem_total_bytes / (1024 * 1024))}MiB`
      : "unavailable (systeminformation reported no usable total)"
  }`
);
log.info(
  `[worker ${config.worker_name}] GPU: ${
    detectedHardware.gpu.length > 0
      ? detectedHardware.gpu.map((g) => g.model || g.vendor).join(", ")
      : "none detected"
  }`
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

// Cache for listModelDirFiles(), populated once at startup and then on a
// timer (see refreshModelDirFilesCache below) rather than re-walked on every
// /models request -- mirrors the /hardware snapshot above, though unlike
// hardware this data *can* change while the process is up (downloads,
// manual file drops), hence the periodic refresh instead of a one-shot.
let modelDirFilesCache: ModelDirFile[] = [];

function refreshModelDirFilesCache(): void {
  try {
    modelDirFilesCache = listModelDirFiles();
  } catch (err) {
    log.error(`model dir listing refresh failed: ${err instanceof Error ? err.message : String(err)}`);
  }
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
        // See shared/types.ts's RunConfig.main_gpu -- forwarded unchanged
        // into runSweepItem/runSweepItemViaServer below.
        main_gpu?: number;
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
      // Tallied across every item regardless of which path handled it, so
      // the run's own finish line can report a success/fail breakdown
      // instead of just "finished all N test(s)" with no indication of how
      // many of those actually succeeded.
      const statusCounts: Partial<Record<TerminalRunItemStatus, number>> = {};
      const tally = (status: TerminalRunItemStatus): void => {
        statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      };
      try {
        for (const item of items) {
          // Checked before *every* item (not just once) since pausing can
          // hold here for a while -- a stop requested while paused must
          // still take effect immediately rather than waiting out the pause.
          if (stopRequested) {
            log.info(`run ${reqBody.run_id}: stopped, cancelling item ${item.idx} (not started)`);
            await safeItemTerminal(reqBody.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
            tally("cancelled");
            continue;
          }
          while (pauseRequested && !stopRequested) {
            await sleep(500);
          }
          if (stopRequested) {
            log.info(`run ${reqBody.run_id}: stopped while paused, cancelling item ${item.idx} (not started)`);
            await safeItemTerminal(reqBody.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
            tally("cancelled");
            continue;
          }
          if (item.mtp === "on") {
            if (!activeBuild!.serverPath) {
              log.error(`run ${reqBody.run_id}: item ${item.idx} needs mtp but the active build has no llama-server binary`);
              await safeItemTerminal(reqBody.run_id, item.idx, {
                status: "failed",
                error: "active llama.cpp build has no llama-server binary -- reinstall the build to pick one up",
              });
              tally("failed");
              continue;
            }
            tally(
              await runSweepItemViaServer({
                runId: reqBody.run_id,
                item,
                repeats: reqBody.sweep.repeats,
                modelPath,
                mtpModelPath,
                llamaServerPath: activeBuild!.serverPath,
                port: config.mtp_server_port ?? DEFAULT_MTP_SERVER_PORT,
                backend,
                mainGpu: reqBody.main_gpu,
                timeoutMs: config.bench_timeout_ms,
                rawJsonDir: rawDir,
              })
            );
          } else {
            tally(
              await runSweepItem({
                runId: reqBody.run_id,
                item,
                repeats: reqBody.sweep.repeats,
                modelPath,
                llamaBenchPath: activeBuild!.path,
                backend,
                mainGpu: reqBody.main_gpu,
                timeoutMs: config.bench_timeout_ms,
                vpsUrl: config.vps_url,
                rawJsonDir: rawDir,
              })
            );
          }
        }
        const breakdown =
          (["done", "failed", "failed_oom", "cancelled"] as const)
            .filter((s) => statusCounts[s])
            .map((s) => `${s}=${statusCounts[s]}`)
            .join(" ") || "no items processed";
        log.info(
          `run ${reqBody.run_id} finished all ${items.length} test(s)${stopRequested ? " (stopped early)" : ""}: ${breakdown}`
        );
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
    // ?main_gpu=<n> scopes backend_device_name to a specific about-to-run
    // run's --main-gpu selection -- see backendDeviceName above.
    const mainGpuParam = url.searchParams.get("main_gpu");
    const mainGpu =
      mainGpuParam != null && /^\d+$/.test(mainGpuParam) ? Number(mainGpuParam) : undefined;
    return send(res, 200, {
      ok: true,
      worker: config.worker_name,
      busy,
      paused: pauseRequested,
      current_run: currentRun,
      backend,
      backend_device_name: backendDeviceName(mainGpu) ?? null,
      active_build: activeBuild?.tag ?? null,
    });
  }

  if (req.method === "GET" && url.pathname === "/hardware") {
    // Serves the snapshot taken once at process startup (see detectedHardware
    // above) rather than re-probing live -- on some machines (e.g. bachika1980)
    // systeminformation's cpu()/graphics() calls are slow enough that hitting
    // this on every orchestrator poll made the Workers page noticeably laggy
    // for no benefit, since hardware doesn't change while the process is up.
    // Restart the worker to pick up real hardware changes.
    return send(res, 200, detectedHardware);
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
    return send(res, 200, { files: modelDirFilesCache });
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
      refreshModelDirFilesCache();
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

    // Reflect the new file immediately rather than waiting for the periodic
    // refresh -- see refreshModelDirFilesCache above.
    refreshModelDirFilesCache();

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
  // See shared/types.ts's RunConfig.main_gpu -- forwarded to bench.ts's
  // buildArgs unchanged.
  mainGpu?: number;
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

// Offload detail (which model got how many layers on the GPU) is now parsed
// once by whichever module actually ran the process -- bench.ts's runBench
// for the llama-bench path, serverBench.ts's runServerBench for the MTP/
// llama-server path -- and arrives here pre-parsed on BenchResult.offload
// (see bench.ts's parseOffloadLayers/OffloadResult). matchOffloadLine is the
// one piece still used directly in this file: a live, single-line version
// for surfacing offload as soon as it's seen while streaming llama-bench's
// stderr (see runSweepItem's onStderrLine below), well before that item's
// process has actually exited.

// One line per sweep item's actual parameters, used both when a test starts
// (before anything is known about the run yet) and in its end-of-test
// summary -- the same string either way, so a log reader can match the two
// up at a glance. ngld is only meaningful for an "on" mtp item (see
// shared/sweep.ts's SweepItem doc comment), so it's omitted otherwise rather
// than printing a value nobody asked for and nothing used.
function formatItemParams(item: SweepItem): string {
  const base =
    `n_prompt=${item.n_prompt} n_gen=${item.n_gen} threads=${item.threads} ngl=${item.n_gpu_layers} ` +
    `batch=${item.batch_size} ubatch=${item.ubatch_size} ctk=${item.cache_type_k} ctv=${item.cache_type_v} ` +
    `fa=${item.flash_attn} mtp=${item.mtp}`;
  return item.mtp === "on" ? `${base} ngld=${item.n_gpu_layers_draft}` : base;
}

// Logged once per item at test start (see runSweepItem/runSweepItemViaServer
// below) so which physical GPU a run is actually targeting is visible
// immediately, without waiting for llama-bench's own end-of-item JSON (Tier-2
// backend_device_name, only available on the llama-bench path anyway -- see
// bench.ts's gpu_info doc comment) or the client's display, which may itself
// still be showing the Tier-1 estimate. Flags an out-of-range mainGpu
// explicitly (rather than letting -mg fail silently downstream in
// llama-bench/llama-server) since that's exactly the failure mode a raw,
// unfiltered hardware.gpu index used to produce -- see shared/types.ts's
// backendVisibleGpus.
function formatDeviceSelection(mainGpu: number | undefined): string {
  const visible = backendVisibleGpus(detectedHardware.gpu, backend);
  if (mainGpu == null) {
    return visible.length > 0
      ? `auto -- split across all ${backend}-visible GPU(s): ${visible.map((g) => g.model || g.vendor).join(", ")}`
      : `auto (no ${backend}-visible GPU detected)`;
  }
  const picked = visible[mainGpu];
  return picked
    ? `${picked.model || picked.vendor} (main_gpu=${mainGpu} of ${visible.length} ${backend}-visible GPU(s))`
    : `main_gpu=${mainGpu} is OUT OF RANGE -- only ${visible.length} ${backend}-visible GPU(s) detected`;
}

function formatOffloadLine(offload: OffloadResult): string {
  if (!offload.main) return "offload: unknown (no offload line seen in output)";
  const main = `main=${offload.main.gpu_layers_loaded}/${offload.main.total_model_layers}`;
  const draft = offload.draft ? ` draft=${offload.draft.gpu_layers_loaded}/${offload.draft.total_model_layers}` : "";
  return `offload: ${main}${draft}`;
}

// Mirrors the ram/vram columns RunDetail.tsx's results table shows (free,
// avg, max, total, plus the accuracy/source tag behind the little info icon
// next to each vram figure -- see AccuracyIcon in that file) so the same
// numbers a person would see in the UI are visible directly in this log.
// vram's accuracy/source also covers what a unified-memory system (Metal --
// see vram.ts's readDarwinGpuMemory) reports through this same field: there's
// no separate "unified memory" number to show, since on that backend vram
// *is* system RAM up to a dynamic ceiling -- the "unified_memory_estimate"
// source tag makes that explicit instead of silently labeling it as
// dedicated VRAM.
function formatMemoryLines(stats: SampleStats, baseline: FreeMemoryBaseline): string[] {
  const ramParts = [
    `free_before=${baseline.ram_free_before_mib}MiB`,
    `avg=${stats.ram_avg_mib}MiB`,
    `peak=${stats.ram_peak_mib}MiB`,
  ];
  if (baseline.system_memory_total_mib != null) ramParts.push(`total=${baseline.system_memory_total_mib}MiB`);
  const ramLine = `ram: ${ramParts.join(" ")}`;

  const vramParts: string[] = [];
  if (baseline.gpu_memory_total_mib != null) vramParts.push(`total=${baseline.gpu_memory_total_mib}MiB`);
  if (baseline.vram_free_before_mib != null) vramParts.push(`free_before=${baseline.vram_free_before_mib}MiB`);
  if (stats.vram_avg_mib != null) vramParts.push(`avg=${stats.vram_avg_mib}MiB`);
  if (stats.vram_peak_mib != null) vramParts.push(`peak=${stats.vram_peak_mib}MiB`);
  if (vramParts.length === 0) return [ramLine, "vram: unavailable"];
  const accuracy = stats.vram_peak_mib != null ? stats.vram_peak_accuracy : baseline.gpu_memory_total_accuracy;
  const source = stats.vram_peak_source ?? baseline.gpu_memory_total_source;
  const vramLine = `vram: ${vramParts.join(" ")} (${accuracy}${source ? `/${source}` : ""})`;
  return [ramLine, vramLine];
}

// pp/tg avg+stddev, same numbers RunDetail.tsx's "PP tok/s"/"TG tok/s"
// columns show once a test finishes (hover-for-stddev there, always-shown
// here). sample_count/suspect_count are only ever set by the llama-server/
// MTP path (see serverBench.ts's buildPhaseSummary) -- undefined on the
// llama-bench path, so the "(n=...)" suffix is naturally omitted there.
function formatResultsLine(results: IngestResultInput[]): string {
  return results
    .map((r) => {
      const suspect = r.suspect_count ? `, ${r.suspect_count} suspect` : "";
      const n = r.sample_count != null ? ` (n=${r.sample_count}${suspect})` : "";
      return `${r.test_type}=${r.avg_tps.toFixed(2)}tok/s ±${r.stddev_tps.toFixed(2)}${n}`;
    })
    .join("  ");
}

// Same estimate RunDetail.tsx's TTFT column derives (n_prompt ÷ PP speed) --
// not a directly measured request latency, see that column's own tooltip.
function formatTtftLine(results: IngestResultInput[]): string | null {
  const pp = results.find((r) => r.test_type === "pp" || r.test_type === "pg");
  if (!pp || pp.n_prompt <= 0 || pp.avg_tps <= 0) return null;
  const ttftMs = (pp.n_prompt / pp.avg_tps) * 1000;
  return `ttft: ~${ttftMs.toFixed(0)}ms (est. n_prompt÷pp avg)`;
}

// Draft-model acceptance rate for an MTP item's tg row, same figure
// RunDetail.tsx's "mtp" column shows on hover (see serverBench.ts's
// checkSpecDecodeMetrics, which is what actually populates spec_drafted/
// spec_accepted). Absent on every non-MTP row, and on an MTP row whose
// /metrics call never confirmed anything at all (see that function's own
// warning path).
function formatSpecDecodeLine(results: IngestResultInput[]): string | null {
  const tg = results.find((r) => r.test_type === "tg");
  if (!tg || tg.spec_drafted == null || tg.spec_accepted == null) return null;
  const rate = tg.spec_drafted > 0 ? (tg.spec_accepted / tg.spec_drafted) * 100 : 0;
  return `mtp accept: ${tg.spec_accepted}/${tg.spec_drafted} (${rate.toFixed(1)}%)`;
}

// llama.cpp's own diagnostic stderr (device detection, model metadata,
// hundreds of per-tensor "loading tensor blk.N...." lines, timing breakdown)
// is always saved in full to the per-item raw JSON dump (writeRawJson below)
// regardless of log level, so nothing is lost by keeping it out of the
// worker's own info-level text log -- confirmed live that mirroring it there
// unfiltered (the previous behavior) buried the structured
// params/offload/per-run/summary lines this file now logs for every test
// (see formatItemParams and friends above) under tens of thousands of
// characters of tensor-by-tensor noise for a single MTP item alone.
// logDiagnosticOutput below still mirrors it into the log file, just at
// debug level (see LOG_LEVEL) so it's there on demand for troubleshooting
// without being what a normal `pm2 logs`/daily-log read has to scroll past.
// Filtered defensively regardless of level: llama-server's MTP path runs at
// -lv 4 (see serverBench.ts's buildArgs) rather than the max level 5
// ("debug", which prints a per-token/per-draft-candidate trace -- confirmed
// live to be unnecessary noise for anything this app reads), but level 4
// ("trace") is still verbose enough that a request/response body --
// including this app's own synthetic filler-token prompts and the model's
// generated text -- could plausibly appear verbatim in a line at that
// verbosity. Genuine llama.cpp diagnostic lines are short, structured,
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
  log.debug(`${label}: ${processName} raw output:\n${filtered}`);
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
  mainGpu: number | undefined
): Promise<TerminalRunItemStatus> {
  // Read from llama.cpp's own runtime output (see bench.ts's
  // parseOffloadLayers) -- both null when the line was never seen at all
  // (item failed before model load finished, or a build too old to support
  // the required verbosity flag).
  const offload: OffloadResult = bench.offload ?? { main: null, draft: null };
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
    // One structured block per test, covering everything RunDetail.tsx's
    // results table shows for this row (params, actual offload, pp/tg
    // avg+stddev, TTFT estimate, ram/vram free/avg/peak, MTP acceptance rate)
    // plus which engine produced it -- see the format* helpers above. This
    // replaces the previous single-line "done: pp=...tok/s ram_peak=..."
    // summary, which dropped most of that detail on the floor.
    const summaryLines = [`${label}: TEST SUMMARY -- engine=${processName} status=done`, `  params: ${formatItemParams(item)}`, `  ${formatOffloadLine(offload)}`, `  results: ${formatResultsLine(results)}`];
    const ttftLine = formatTtftLine(results);
    if (ttftLine) summaryLines.push(`  ${ttftLine}`);
    for (const line of formatMemoryLines(stats, baseline)) summaryLines.push(`  ${line}`);
    const specLine = formatSpecDecodeLine(results);
    if (specLine) summaryLines.push(`  ${specLine}`);
    log.info(summaryLines.join("\n"));
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
      // no-ops there without any extra branching. Only applied when this run
      // didn't pin a specific mainGpu: llama-bench's own gpu_info lists
      // *every* device this backend can see (confirmed live -- it's built
      // from the backend's full device enumeration, not what -sm none -mg
      // actually restricted compute to), which is the right thing to show
      // for an auto/split-across-all-GPUs run but would silently reintroduce
      // the "picked one GPU, display shows several" bug this run explicitly
      // avoided by pinning one -- the already-correct single-device Tier-1
      // name (see backendDeviceName above) stays authoritative in that case.
      backend_device_name: mainGpu == null ? bench.gpu_info : undefined,
    });
    return "done";
  }
  // A user-requested stop kills this exact process (SIGKILL, via /run/stop)
  // -- report it as "cancelled", not a genuine failure, so it doesn't read
  // as something having gone wrong.
  const status: TerminalRunItemStatus = stopRequested ? "cancelled" : classifyFailure(bench);
  const errorMessage = stopRequested
    ? "cancelled by user"
    : bench.code === 0
      ? `${processName} exited cleanly but produced no parseable result`
      : bench.stderr || `${processName} exited with code ${bench.code}`;
  const failureLines = [
    `${label}: TEST SUMMARY -- engine=${processName} status=${status}`,
    `  params: ${formatItemParams(item)}`,
    `  ${formatOffloadLine(offload)}`,
    ...formatMemoryLines(stats, baseline).map((line) => `  ${line}`),
    `  error (code ${bench.code}, signal ${bench.signal ?? "none"}): ${errorMessage}`,
  ];
  log.error(failureLines.join("\n"));
  await safeItemTerminal(runId, item.idx, {
    status,
    error: errorMessage,
    ram_peak_mib: stats.ram_peak_mib,
    vram_peak_mib: stats.vram_peak_mib,
    ram_avg_mib: stats.ram_avg_mib,
    vram_avg_mib: stats.vram_avg_mib,
  });
  return status;
}

// Runs exactly one sweep combination as its own llama-bench process, reports
// its progress live, and always resolves (never throws) regardless of
// outcome -- so the caller's loop over every item in the sweep can continue
// unconditionally instead of one bad combination aborting the rest.
async function runSweepItem(input: RunSweepItemInput): Promise<TerminalRunItemStatus> {
  const { runId, item, backend } = input;
  const label = `run ${runId} item ${item.idx}`;
  const itemStartedAt = Date.now();

  const testType = deriveTestType(item.n_prompt, item.n_gen);
  const benchmarkingPhase: RunItemTickInput["status"] =
    testType === "pp" ? "processing" : testType === "tg" ? "generating" : "benchmarking";

  log.info(`${label}: starting test -- engine=llama-bench params: ${formatItemParams(item)}`);
  log.info(`${label}: device: ${formatDeviceSelection(input.mainGpu)}`);
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
  // Logged once, live, the first time llama-bench's own "offloaded X/Y
  // layers to GPU" line streams by (see matchOffloadLine) -- right when
  // model loading finishes, well before the first benchmark repeat, so
  // offload is visible without waiting for the item's TEST SUMMARY at the
  // very end.
  let loggedLiveOffload = false;

  try {
    const bench = await runBench({
      modelPath: input.modelPath,
      item,
      repeats: input.repeats,
      llamaBenchPath: input.llamaBenchPath,
      backend,
      mainGpu: input.mainGpu,
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
          // Only mirrored to the info log while still loading -- once
          // repeats start, each one already logs its own speed+memory line
          // below (see the PP_RUN_RE/TG_RUN_RE handling), so a heartbeat on
          // top of that would just repeat the same info every ~20s for the
          // rest of the item. Loading has no repeat markers at all, so this
          // remains the only visibility into a long model load.
          if (tickCount % HEARTBEAT_EVERY_N === 0 && phase === "loading") {
            const elapsedS = Math.round((Date.now() - itemStartedAt) / 1000);
            log.info(
              `${label}: loading${detail ? ` (${detail})` : ""} ram=${current.ram_mib ?? "?"}MiB` +
                `${current.vram_mib != null ? ` vram=${current.vram_mib}MiB` : ""} elapsed=${elapsedS}s`
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

        if (!loggedLiveOffload) {
          const offloadHit = matchOffloadLine(line);
          if (offloadHit) {
            loggedLiveOffload = true;
            log.info(`${label}: offload: main=${offloadHit.gpu_layers_loaded}/${offloadHit.total_model_layers}`);
          }
        }

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
          const current = sampler.current;
          const memSuffix = ` ram=${current.ram_mib}MiB${current.vram_mib != null ? ` vram=${current.vram_mib}MiB` : ""}`;
          log.info(`${label}: ${phase} run ${rep}/${reps}${tpsSuffix}${memSuffix}`);
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
    const outcome = bench.code === 0 && bench.results.length > 0 ? "ok" : "failed";
    writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-bench", bench.stderr);
    return await finalizeSweepItemResult(runId, item, label, "llama-bench", bench, stats, baseline, input.mainGpu);
  } catch (err) {
    if (tickTimer) clearInterval(tickTimer);
    activeBenchProc = null;
    const stats = sampler.stop();
    const message = err instanceof Error ? err.message : String(err);
    log.error(`${label}: TEST SUMMARY -- engine=llama-bench status=failed\n  params: ${formatItemParams(item)}\n  error: ${message}`);
    await safeItemTerminal(runId, item.idx, {
      status: "failed",
      error: message,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
    });
    return "failed";
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
  // See shared/types.ts's RunConfig.main_gpu -- forwarded to serverBench.ts's
  // buildArgs unchanged.
  mainGpu?: number;
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
async function runSweepItemViaServer(input: RunSweepItemViaServerInput): Promise<TerminalRunItemStatus> {
  const { runId, item, backend } = input;
  const label = `run ${runId} item ${item.idx} (mtp)`;

  log.info(`${label}: starting test -- engine=llama-server params: ${formatItemParams(item)}`);
  log.info(`${label}: device: ${formatDeviceSelection(input.mainGpu)}`);
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
      mainGpu: input.mainGpu,
      timeoutMs: input.timeoutMs,
      onSpawn: (proc) => {
        activeBenchProc = proc;
        sampler.start(proc.pid, backend, TICK_INTERVAL_MS);
      },
      onProgress: (phase, detail, liveTps) => {
        const current = sampler.current;
        const memSuffix = ` ram=${current.ram_mib}MiB${current.vram_mib != null ? ` vram=${current.vram_mib}MiB` : ""}`;
        log.info(`${label}: ${phase} (${detail})${liveTps != null ? ` (${liveTps.toFixed(1)} t/s)` : ""}${memSuffix}`);
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
    const outcome = bench.code === 0 && bench.results.length > 0 ? "ok" : "failed";
    writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-server", bench.stderr);
    return await finalizeSweepItemResult(runId, item, label, "llama-server", bench, stats, baseline, input.mainGpu);
  } catch (err) {
    activeBenchProc = null;
    const stats = sampler.stop();
    const message = err instanceof Error ? err.message : String(err);
    log.error(`${label}: TEST SUMMARY -- engine=llama-server status=failed\n  params: ${formatItemParams(item)}\n  error: ${message}`);
    await safeItemTerminal(runId, item.idx, {
      status: "failed",
      error: message,
      ram_peak_mib: stats.ram_peak_mib,
      vram_peak_mib: stats.vram_peak_mib,
      ram_avg_mib: stats.ram_avg_mib,
      vram_avg_mib: stats.vram_avg_mib,
    });
    return "failed";
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

// Populate the /models cache right away so the first request after startup
// doesn't race an empty cache, then keep it fresh on a timer -- see
// refreshModelDirFilesCache above.
const MODEL_LIST_REFRESH_INTERVAL_MS = 2 * 60 * 1000;
refreshModelDirFilesCache();
setInterval(refreshModelDirFilesCache, MODEL_LIST_REFRESH_INTERVAL_MS).unref();

function shutdown(signal: string): void {
  log.info(`received ${signal}, shutting down...`);
  server.close(() => process.exit(0));
  // Force-exit if a benchmark is still in flight and close() hangs.
  setTimeout(() => process.exit(1), 5000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
