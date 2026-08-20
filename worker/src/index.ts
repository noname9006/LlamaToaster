import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve, relative, isAbsolute, sep } from "node:path";
import {
  readFileSync,
  writeFileSync,
  renameSync,
  mkdirSync,
  createWriteStream,
  createReadStream,
  unlinkSync,
  readdirSync,
  statSync,
  chmodSync,
} from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { gzipSync } from "node:zlib";
import { hostname as osHostname } from "node:os";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { runBench, matchOffloadLine, type BenchResult, type OffloadResult } from "./bench.js";
import { runServerBench } from "./serverBench.js";
import { MemorySampler, captureFreeMemoryBaseline, type SampleStats, type FreeMemoryBaseline } from "./sampler.js";
import { readGpuMemory } from "./vram.js";
import { estimateVramNeededMib, isVramDiscrepancy } from "../../shared/vramEstimate.js";
import {
  writeRawJson,
  postRunItemUpdate,
  postModelDownloadResult,
  postHeartbeat,
  pollQueue,
  reportJobResult,
  pushRunLog,
  startDeviceEnrolment,
  pollDeviceToken,
  refreshWorkerSession,
  HttpError,
} from "./vps-client.js";
import { log, configureLogging, setRunLogFile } from "./log.js";
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
  type ModelDirFile,
  type WorkerStatePush,
  type ActiveJobReport,
  type QueueJob,
  type BenchmarkJob,
  type InstallBuildJobPayload,
  type ActivateBuildJobPayload,
  type DeleteBuildJobPayload,
  type DownloadModelJobPayload,
  type WorkerVramInfo,
} from "../../shared/types.js";
import { expandSweep, deriveTestType, type SweepItem } from "../../shared/sweep.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// resolve(), not join(): join() concatenates segments literally, so an
// ABSOLUTE WORKER_CONFIG value (e.g. a device-flow-enrolled config living
// outside this repo checkout entirely, MULTIUSER_PLAN.md §6.0) got silently
// nested under process.cwd() instead of used as-is -- resolve() treats a
// later absolute argument as an override, same as every shell's own path
// semantics, while still resolving a relative value against cwd() exactly
// as before.
const configPath = process.env.WORKER_CONFIG
  ? resolve(process.cwd(), process.env.WORKER_CONFIG)
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
  // Stable id this worker persists locally, generated once on first boot and
  // reused across restarts -- this, not worker_name, is what identifies this
  // machine to the server (MULTIUSER_PLAN.md §1.2). Self-announced: the
  // first heartbeat/queue-poll carrying a machine_id the server has never
  // seen creates its `workers` row.
  machine_id?: string;
  // Stage 1 (legacy, still supported): a single shared secret authenticating
  // every worker, checked as a Bearer token on every pull-queue call. Kept
  // working unchanged for an already-deployed worker whose config.json was
  // never migrated -- only consulted when session_token below is absent (see
  // ensureAuthCredential and server/src/worker-auth.ts's dual-mode
  // authenticateWorker, MULTIUSER_PLAN.md §1.15/§3.5).
  worker_shared_token?: string;
  // Stage 3: this worker's own per-worker session, normally obtained once via
  // device-code enrolment (see ensureAuthCredential below) and persisted back
  // here so a restart doesn't re-enrol. Takes priority over
  // worker_shared_token when both happen to be present.
  session_token?: string;
  // Rotates alongside session_token (see refreshWorkerSession) -- presented
  // only to POST /api/auth/refresh, never to any other route.
  refresh_token?: string;
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
  // Only one job ever runs at a time per worker (the pull loop is strictly
  // serial), so a fixed port is simpler than allocating one dynamically and
  // has no collision risk in practice.
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
  // Optional, off by default -- a local-only diagnostic endpoint
  // (127.0.0.1-bound /health), NOT used for dispatch: the worker needs no
  // inbound HTTP at all under the pull model (MULTIUSER_PLAN.md §1.11).
  debug_port?: number;
  // How many download_model jobs this worker runs concurrently -- everything
  // else (benchmark, install_build, ...) stays strictly serial regardless of
  // this value (see workerMain's downloadJobPool). Defaults to 2 if unset;
  // a local, per-machine resource tuning knob, same posture as backend/
  // llama_bench_path above -- no server-side UI to set it, edit config.json.
  max_concurrent_downloads?: number;
}

function loadConfig(): WorkerConfig {
  const raw = readFileSync(configPath, "utf8");
  return JSON.parse(raw) as WorkerConfig;
}

const config = loadConfig();

// Best-effort -- chmod is a no-op on Windows (systeminformation-detected
// platform, not checked here directly) for the POSIX mode bits used, but the
// call itself doesn't throw for a plain 0o600, so this stays safe to call
// unconditionally on every platform. config.json holds a 90-day bearer
// credential once session_token is populated (MULTIUSER_PLAN.md §3.5), same
// as the shared secret it replaces, so this runs regardless of which auth
// mode is active.
function secureConfigFile(): void {
  try {
    chmodSync(configPath, 0o600);
  } catch (err) {
    log.debug(`could not chmod config.json to 0600: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function persistConfig(patch: Partial<WorkerConfig>): void {
  const merged = { ...config, ...patch };
  const tmp = `${configPath}.tmp`;
  writeFileSync(tmp, JSON.stringify(merged, null, 2), "utf8");
  renameSync(tmp, configPath);
  secureConfigFile();
  Object.assign(config, patch);
}

secureConfigFile(); // lock down permissions even if this run never calls persistConfig

if (!config.machine_id) {
  const generated = randomUUID();
  persistConfig({ machine_id: generated });
  log.info(`generated new machine_id: ${generated}`);
}
// No hard requirement on worker_shared_token/session_token here anymore --
// see ensureAuthCredential below, called once hardware detection has run
// (device enrolment needs platform/arch): neither being set is the normal
// state for a fresh install and triggers enrolment rather than failing.

const buildsDir = resolve(config.llama_cpp_builds_dir ?? join(__dirname, "..", "llama-builds"));

const logDir = resolve(config.log_dir ?? join(__dirname, "..", "logs"));
configureLogging(logDir, process.env.LOG_LEVEL);

// One dedicated, structured log file per run (see setRunLogFile/
// executeBenchmarkJob below), separate from the shared daily worker-*.log
// above -- pushed to the server on completion (pushRunLogIfPresent) and
// served back via GET /api/runs/:id/log, surfaced in the UI next to the CSV
// export whenever a run has a failed test (see RunDetail.tsx).
const runLogsDir = join(logDir, "runs");
try {
  mkdirSync(runLogsDir, { recursive: true });
} catch (err) {
  log.error(
    `could not create run-logs dir ${runLogsDir}: ${err instanceof Error ? err.message : String(err)} ` +
      `(per-run log files will be unavailable)`
  );
}

// run_id is always a uuid (server/src/routes/runs.ts's uuid()), but this is
// also used to build a filesystem path -- validated defensively rather than
// trusted regardless.
const SAFE_RUN_ID_RE = /^[a-zA-Z0-9_-]+$/;

function runLogFilePath(runId: string): string {
  return join(runLogsDir, `${runId}.log`);
}

const detectedHardware = await detectHardware();

// backend is optional in config.json -- when unset, detect it live from
// this machine's actual GPU rather than requiring it be hardcoded. An
// explicit value in config.json always wins (e.g. to force `cpu` on a box
// that does have a GPU).
const backend: Backend = config.backend ?? detectBackend(detectedHardware.platform, detectedHardware.gpu);
if (!config.backend) {
  log.info(`no backend set in config.json -- auto-detected "${backend}" from hardware`);
}

// "model (4.0 GB)" / "model (1024 MB shared)" for a HardwareInfo.gpu entry --
// used only in log output (see the startup GPU: line and formatDeviceSelection
// below), never in backend_device_name itself, which stays just the plain
// name since that's what feeds RunDetail.tsx's "<name> (<backend>)" label.
function formatGpuEntry(g: { model: string; vendor: string; vram_mb?: number | null; vram_dynamic?: boolean }): string {
  const name = g.model || g.vendor || "unknown";
  if (g.vram_mb == null) return name;
  const vram = g.vram_mb >= 1024 ? `${(g.vram_mb / 1024).toFixed(1)} GB` : `${g.vram_mb} MB`;
  return `${name} (${vram}${g.vram_dynamic ? " shared" : ""})`;
}

log.info(
  `[worker ${config.worker_name}] starting (machine_id=${config.machine_id}, backend=${backend}, build=${config.llama_cpp_build}, log level=${
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
      ? detectedHardware.gpu.map((g) => formatGpuEntry(g)).join(", ")
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
// before any real job can plausibly be claimed, means the first genuine
// captureFreeMemoryBaseline call (sampler.ts, used by runSweepItem/
// runSweepItemViaServer below) already hits a warm path instead of paying
// that cost itself. Fire-and-forget and swallow all errors.
void readGpuMemory(backend, undefined).catch(() => {});

// The active build is mutable at runtime (via an activate_build job) so a
// version switch from the web UI takes effect on the next benchmark job
// without restarting this process. It starts from config.json but is
// persisted back to config.json on every switch so a restart doesn't revert
// it.
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
// coming from a download/delete job rather than a registered Model.
function resolveDownloadTarget(hfFile: string): string {
  const root = resolve(config.model_dir);
  const resolved = resolve(root, hfFile);
  const rel = relative(root, resolved);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error(`hf_file resolves outside model_dir: ${hfFile}`);
  }
  return resolved;
}

// Walks model_dir recursively so the server can learn "which files does this
// worker actually have" from the worker's own heartbeat instead of asking
// about one filename at a time -- also how a Gemma-4-style MTP companion
// sitting in an "MTP/" subfolder (see resolveModelPath's `name` handling) is
// discovered at all. Skips raw_json_dir when it happens to be nested inside
// model_dir (not the case in either shipped config, but no reason to list
// bench output as if it were a model file).
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
        // A .part is an in-progress/paused download (executeDownloadModelJob
        // below) -- never a real, usable model file, so it must never be
        // reported as one (it'd otherwise show up in the Models page's Local
        // file listing as if it were complete).
        if (entry.name.endsWith(".part")) continue;
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
// heartbeat -- mirrors the hardware snapshot above, though unlike hardware
// this data *can* change while the process is up (downloads, manual file
// drops), hence the periodic refresh instead of a one-shot. Sent verbatim as
// WorkerStatePush.model_files on every heartbeat/queue poll -- see
// collectState below.
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
// retention window to each without needing to ask the server "was this item
// a failure" after the fact (the worker keeps no local run history otherwise).
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

// Hashes a file already fully on disk, streamed rather than read whole into
// memory (models are often multiple GB). Used by executeDownloadModelJob
// AFTER the download completes, not inline during the write -- a Range-
// resumed download only streams the NEW bytes, so hashing inline would only
// ever cover the tail written in this particular process run, not bytes
// written by an earlier attempt before a pause/restart. One full sequential
// read of the file is a bounded, known cost regardless of how many times a
// download was paused and resumed.
function hashFile(path: string): Promise<{ sha256: string; byteLength: number }> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    let byteLength = 0;
    const rs = createReadStream(path);
    rs.on("data", (chunk: Buffer) => {
      hash.update(chunk);
      byteLength += chunk.length;
    });
    rs.on("end", () => resolve({ sha256: hash.digest("hex"), byteLength }));
    rs.on("error", reject);
  });
}

// How often a running item's live progress tick (phase + ram/vram) is sent.
const TICK_INTERVAL_MS = 2000;
// Ticks fire every TICK_INTERVAL_MS but only reach the server -- a worker log
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
// worker/src/serverBench.ts). Only one job ever runs at a time per worker
// (the pull loop is strictly serial), so a fixed port carries no real
// collision risk -- config.mtp_server_port overrides it if needed.
const DEFAULT_MTP_SERVER_PORT = 8899;

// Whether a job is currently executing -- drives WorkerStatePush.status
// (collectState below). Set/cleared only by workerMain's own loop.
let busy = false;

// Run-level pause/stop control -- reset at the start of every benchmark job
// (see executeBenchmarkJob below). llama-bench runs every repeat (-r) of one
// sweep item inside a single process invocation, so there's no safe way to
// pause mid-item without corrupting that repeat's timing -- pause only takes
// effect between items. Stop kills the in-flight process immediately via the
// same SIGKILL path bench.ts's own timeout uses. Both are now driven by the
// heartbeat's control response (requestStop/the heartbeat loop below)
// instead of an inbound HTTP call -- the machinery reacting to them is
// otherwise unchanged.
let pauseRequested = false;
let stopRequested = false;
let activeBenchProc: import("node:child_process").ChildProcess | null = null;

// What the heartbeat loop reports as ActiveJobReport while a job is running
// -- job_id is set once when a job is claimed (workerMain) and never
// undefined while `busy` is true. Phase/detail/item_idx/bytes are updated by
// whichever job executor is currently running (updateJobReport below).
let currentJobReport: ActiveJobReport | null = null;

function updateJobReport(patch: Partial<Omit<ActiveJobReport, "job_id">>): void {
  if (!currentJobReport) return;
  Object.assign(currentJobReport, patch);
}

// Concurrently-running download_model jobs -- separate from currentJobReport
// above, which stays reserved for the ONE serial (benchmark/build) job slot.
// Keyed by job_id. workerMain's downloadJobPool adds/removes entries as
// downloads start and finish; the heartbeat loop reports Array.from(...) of
// this on every beat (postHeartbeat's activeDownloads param) regardless of
// whether the worker is otherwise idle or busy.
const activeDownloadReports = new Map<string, ActiveJobReport>();
// One AbortController per active download, used by requestDownloadStop
// (delivered via the heartbeat's cancel_job_ids, same channel requestStop
// uses for the serial job) to cancel a specific in-flight fetch -- keyed
// alongside activeDownloadReports rather than merged into it since
// AbortController isn't JSON-serializable state.
const activeDownloadControllers = new Map<string, AbortController>();

function updateDownloadReport(jobId: string, patch: Partial<Omit<ActiveJobReport, "job_id">>): void {
  const cur = activeDownloadReports.get(jobId);
  if (cur) Object.assign(cur, patch);
}

// Delivered on the heartbeat's cancel_job_ids -- the download equivalent of
// requestStop below. No-ops if jobId doesn't name a download this worker is
// currently tracking (e.g. a stale/duplicate signal after it already
// finished on its own).
function requestDownloadStop(jobId: string): void {
  activeDownloadControllers.get(jobId)?.abort();
}

function toInstalledBuildList(): InstalledBuild[] {
  return listInstalledBuilds(buildsDir).map((b) => ({
    tag: b.tag,
    asset_name: b.asset_name,
    installed_at: b.installed_at,
    active: activeBuild?.tag === b.tag,
    bench_path: b.bench_path,
    server_path: b.server_path,
  }));
}

// One self-reported snapshot sent on every heartbeat/queue poll -- replaces
// the old live GET /health + GET /llama-cpp + GET /hardware fan-out
// (MULTIUSER_PLAN.md §1.5). n_layer/mtp_layers are deliberately NOT read per
// file here -- that would mean a GGUF header parse for potentially hundreds
// of files on every 10s heartbeat. The common case (a model downloaded
// through this app) already has n_layer/mtp_layers on the MODEL's own
// registered metadata, set once at download time (see
// executeDownloadModelJob) -- this per-file list is only consulted as a
// fallback for models registered another way, and for those it correctly
// reports "unknown" rather than paying that cost on every heartbeat.
//
// vram is read fresh here too (see shared/types.ts's WorkerVramInfo doc
// comment) -- there's no more server->worker proxy to serve NewRun.tsx's
// pre-flight VRAM-fit banner on demand (MULTIUSER_PLAN.md §1.11), so this
// heartbeat is the only place that reading happens. Skipped entirely while
// busy: not meaningful for pre-flight fit-checking against a machine that's
// already running something, and avoids a second concurrent readGpuMemory
// call alongside MemorySampler's own (see captureFreeMemoryBaseline).
async function collectState(): Promise<WorkerStatePush> {
  const vram = busy
    ? undefined
    : await captureFreeMemoryBaseline(backend)
        .then((baseline): WorkerVramInfo => ({ ok: true, backend, ...baseline }))
        .catch(() => null);
  return {
    machine_id: config.machine_id!,
    capabilities: ["benchmark"],
    hostname: osHostname(),
    backend,
    hardware: detectedHardware,
    installed_builds: toInstalledBuildList(),
    model_files: modelDirFilesCache,
    status: busy ? "busy" : "idle",
    vram,
  };
}

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

// The worker has no local record of a run once it's dispatched -- the server
// is the only place run/item status lives. If a terminal update can't reach
// it, retry with backoff rather than silently dropping the outcome and
// leaving that item (and potentially the whole run) stuck "running" forever
// with no way to recover it short of manual DB surgery.
async function safeItemTerminal(
  runId: string,
  idx: number,
  payload: RunItemTerminalInput
): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      await withAuth((token) => postRunItemUpdate(config.vps_url, token, runId, idx, payload, 10_000));
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
// terminal update above catches the server up), so this deliberately doesn't
// retry or block the caller on the network round trip.
function sendTick(runId: string, idx: number, tick: RunItemTickInput): void {
  withAuth((token) => postRunItemUpdate(config.vps_url, token, runId, idx, tick, 3000)).catch((err) => {
    log.debug(
      `tick failed for run ${runId} item ${idx} (non-fatal): ${err instanceof Error ? err.message : String(err)}`
    );
  });
}

// Same retry-with-backoff posture as safeItemTerminal above -- this is the
// only thing that gets a completed download registered as a Model
// server-side, so losing it silently would leave a fully downloaded file on
// disk the app never learns about. Returns whether delivery ultimately
// succeeded so callers can distinguish that from a silently-swallowed
// failure -- see executeDownloadModelJob's success path, which used to
// report the job as complete even when this returned false.
async function safeReportDownloadResult(payload: ModelDownloadCallbackInput): Promise<boolean> {
  const key = `${payload.hf_repo}/${payload.hf_file}`;
  for (let attempt = 0; ; attempt++) {
    try {
      await withAuth((token) => postModelDownloadResult(config.vps_url, token, payload, 10_000));
      log.info(`download callback ok for ${key} (ok=${payload.ok}, attempt ${attempt + 1})`);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (attempt >= ITEM_RETRY_DELAYS_MS.length) {
        log.error(`download callback failed for ${key} after ${attempt + 1} attempts, giving up: ${message}`);
        return false;
      }
      const delay = ITEM_RETRY_DELAYS_MS[attempt];
      log.warn(`download callback attempt ${attempt + 1} failed for ${key}, retrying in ${delay}ms: ${message}`);
      await sleep(delay);
    }
  }
}

interface RunSweepItemInput {
  runId: string;
  item: SweepItem;
  repeats: number;
  modelPath: string;
  // On-disk byte size of the model at modelPath -- forwarded to
  // finalizeSweepItemResult's VRAM-discrepancy check (see
  // shared/vramEstimate.ts), not used for anything else here.
  modelSizeBytes: number;
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

// One-time banner written to a run's own dedicated log file (see
// setRunLogFile/runLogFilePath) the moment it starts -- everything needed to
// make sense of the rest of that file without cross-referencing this
// worker's shared daily log: exactly which llama.cpp build/backend ran it,
// what hardware/OS it ran on, and how big the sweep is. Mirrors the
// OS/CPU/RAM/GPU lines this file already logs once at process startup (see
// detectedHardware above), just scoped to this one run instead of the
// worker's whole lifetime.
const RUN_LOG_RULE = "=".repeat(70);

function formatRunLogHeader(
  runId: string,
  model: Model,
  mtpModel: Model | undefined,
  build: ActiveBuild,
  runBackend: Backend,
  itemCount: number,
  repeats: number
): string {
  const installed = getInstalledBuild(buildsDir, build.tag);
  const buildLine = installed ? `${build.tag} (${installed.asset_name})` : build.tag;
  const gpuLine =
    detectedHardware.gpu.length > 0
      ? detectedHardware.gpu.map((g) => formatGpuEntry(g)).join(", ")
      : "none detected";
  const ramLine =
    detectedHardware.mem_total_bytes != null
      ? `${Math.round(detectedHardware.mem_total_bytes / (1024 * 1024))}MiB`
      : "unavailable";
  const lines = [
    RUN_LOG_RULE,
    " LlamaToaster run log",
    `  run:         ${runId}`,
    `  worker:      ${config.worker_name}`,
    `  started:     ${new Date().toISOString()}`,
    `  llama.cpp:   ${buildLine} (backend=${runBackend})`,
    `  os:          ${detectedHardware.platform} (${detectedHardware.arch})`,
    `  cpu:         ${detectedHardware.cpu.brand || detectedHardware.cpu.manufacturer || "unknown"}${
      detectedHardware.cpu.cores ? ` (${detectedHardware.cpu.cores} threads)` : ""
    }`,
    `  ram:         ${ramLine}`,
    `  gpu:         ${gpuLine}`,
    `  model:       ${model.filename}`,
    ...(mtpModel ? [`  mtp model:   ${mtpModel.filename}`] : []),
    `  tests:       ${itemCount} sweep combination(s)`,
    `  repeats:     ${repeats} per test`,
    `  total runs:  ${itemCount * repeats} (tests x repeats)`,
    RUN_LOG_RULE,
  ];
  return lines.join("\n");
}

// Closing block for a run's dedicated log file -- "tests" tallies each
// sweep combination's own outcome (the real unit of success/failure: a
// single llama-bench/llama-server process covers all of that test's
// repeats and either produces a usable result or doesn't), "runs" scales
// that up by the configured repeat count for a figure in the vocabulary the
// rest of this app uses for -r (see RunDetail.tsx's own "runs means
// repeats" comment) -- a done test's repeats all completed, a failed one's
// produced nothing usable, and a cancelled one's never started at all.
function formatRunLogFooter(
  runId: string,
  itemCount: number,
  repeats: number,
  statusCounts: Partial<Record<TerminalRunItemStatus, number>>,
  stoppedEarly: boolean
): string {
  const done = statusCounts.done ?? 0;
  const failed = (statusCounts.failed ?? 0) + (statusCounts.failed_oom ?? 0);
  const cancelled = statusCounts.cancelled ?? 0;
  const endReason = stoppedEarly
    ? "stopped early -- cancelled by user before every test ran"
    : `performed all ${itemCount} test(s)`;
  const lines = [
    RUN_LOG_RULE,
    ` RUN SUMMARY -- run ${runId}`,
    `  finished:               ${new Date().toISOString()}`,
    `  tests:                  ${itemCount} total -- done=${done} failed=${failed} cancelled=${cancelled}`,
    `  runs (tests x repeats): completed=${done * repeats} failed=${failed * repeats} not run=${
      cancelled * repeats
    } (of ${itemCount * repeats} total)`,
    `  end reason:             ${endReason}`,
    RUN_LOG_RULE,
  ];
  return lines.join("\n");
}

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
// backend is a parameter (not read from the module-level const) so a
// backend-override job (see executeBenchmarkJob's effectiveBackend) logs its
// actual one-off backend here, not this worker's unrelated persisted
// default.
function formatDeviceSelection(backend: Backend, mainGpu: number | undefined): string {
  const visible = backendVisibleGpus(detectedHardware.gpu, backend);
  if (mainGpu == null) {
    return visible.length > 0
      ? `auto -- split across all ${backend}-visible GPU(s): ${visible.map((g) => formatGpuEntry(g)).join(", ")}`
      : `auto (no ${backend}-visible GPU detected)`;
  }
  const picked = visible[mainGpu];
  return picked
    ? `${formatGpuEntry(picked)} (main_gpu=${mainGpu} of ${visible.length} ${backend}-visible GPU(s))`
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
  mainGpu: number | undefined,
  modelSizeBytes: number
): Promise<TerminalRunItemStatus> {
  // Read from llama.cpp's own runtime output (see bench.ts's
  // parseOffloadLayers) -- both null when the line was never seen at all
  // (item failed before model load finished, or a build too old to support
  // the required verbosity flag).
  const offload: OffloadResult = bench.offload ?? { main: null, draft: null };
  // llama.cpp's own "offloaded X/Y layers to GPU" line only ever reflects
  // buffer *assignment*, never actual VRAM residency (see
  // shared/vramEstimate.ts's top comment) -- so a claimed-full offload with
  // an implausibly low observed vram_peak_mib means the model likely never
  // really left system RAM (Windows CUDA sysmem fallback), regardless of
  // what llama.cpp itself believes happened. Only checked once there's
  // something real to compare: a positive offload claim and an actual VRAM
  // sample (both null/0 on a cpu-backend or -ngl 0 run -- nothing to flag).
  // Also skipped whenever item.n_cpu_moe > 0: a deliberate partial MoE-to-CPU
  // placement *correctly* produces lower VRAM than this estimate expects
  // (that's the whole point of --n-cpu-moe, see shared/sweep.ts's
  // SweepItem.n_cpu_moe) -- without this guard, every legitimate cpu-moe run
  // would false-positive against the exact warning built to catch the
  // opposite problem.
  let vramDiscrepancyWarning: string | undefined;
  if (item.n_cpu_moe === 0 && offload.main && offload.main.gpu_layers_loaded > 0 && stats.vram_peak_mib != null) {
    const estimatedMib = estimateVramNeededMib({
      modelSizeBytes,
      totalModelLayers: offload.main.total_model_layers,
      requestedNgl: offload.main.gpu_layers_loaded,
    });
    if (estimatedMib != null && isVramDiscrepancy(estimatedMib, stats.vram_peak_mib)) {
      vramDiscrepancyWarning =
        `claimed offload ${offload.main.gpu_layers_loaded}/${offload.main.total_model_layers} layers to GPU ` +
        `(~${estimatedMib}MiB expected) but observed VRAM peaked at only ${stats.vram_peak_mib}MiB -- likely ` +
        `silently running from system RAM (Windows CUDA sysmem fallback), not actual GPU offload`;
    }
  }
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
    // bench.warning (previously only ever set by the llama-server/MTP path,
    // e.g. a reading rejected as implausible) and vramDiscrepancyWarning
    // above (either path -- computed once, up front, from data both engines
    // already produce) are combined here into one string -- surfaced via the
    // same `error` field a failed item uses, since recordRunItemTerminal
    // stores it unconditionally regardless of status, and RunDetail.tsx
    // already prefers item.error over the normal detail text for any status.
    // Without this, a "done" item with a silently dropped tg reading (the
    // original MTP bug report) or a silently-not-really-offloaded model (the
    // sysmem-fallback bug report) both looked identical to a clean run.
    const combinedWarning = [bench.warning, vramDiscrepancyWarning].filter(Boolean).join(" | ") || undefined;
    if (combinedWarning) log.warn(`${label} completed with warnings: ${combinedWarning}`);
    await safeItemTerminal(runId, item.idx, {
      status: "done",
      error: combinedWarning,
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
  // A user-requested stop kills this exact process (SIGKILL, via the
  // heartbeat's cancel_job_ids control) -- report it as "cancelled", not a
  // genuine failure, so it doesn't read as something having gone wrong.
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
  log.info(`${label}: device: ${formatDeviceSelection(backend, input.mainGpu)}`);
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
    const rawJsonPath = writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-bench", bench.stderr);
    const result = await finalizeSweepItemResult(
      runId,
      item,
      label,
      "llama-bench",
      bench,
      stats,
      baseline,
      input.mainGpu,
      input.modelSizeBytes
    );
    // Printed last (after the TEST SUMMARY block above, whatever this item's
    // outcome) so anyone reading the log -- including a stopped/cancelled
    // item, which still reaches this same path -- has the exact file with
    // this test's full raw stdout/stderr one line away, instead of having to
    // scroll back through the (now-collapsed, but still not everything)
    // console output above to reconstruct it.
    log.info(`${label}: debug log: ${rawJsonPath}`);
    return result;
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
  // See RunSweepItemInput.modelSizeBytes above -- same purpose.
  modelSizeBytes: number;
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
  log.info(`${label}: device: ${formatDeviceSelection(backend, input.mainGpu)}`);
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
    const rawJsonPath = writeRawJson(rawJsonSubdir(input.rawJsonDir, outcome), `${runId}-${item.idx}`, {
      stdout: bench.stdout,
      stderr: bench.stderr,
    });
    logDiagnosticOutput(label, "llama-server", bench.stderr);
    const result = await finalizeSweepItemResult(
      runId,
      item,
      label,
      "llama-server",
      bench,
      stats,
      baseline,
      input.mainGpu,
      input.modelSizeBytes
    );
    // See runSweepItem's identical line above -- same reasoning, same
    // "always printed last, cancelled included" behavior.
    log.info(`${label}: debug log: ${rawJsonPath}`);
    return result;
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

// Best-effort push of a completed run's log to the server (MULTIUSER_PLAN.md
// §1.10) -- gzipped, identified by X-Machine-Id. Called from
// executeBenchmarkJob's finally block regardless of outcome (stopped/failed
// runs still get a log). A failure here is logged but never thrown -- losing
// the log is unfortunate, not a reason to fail the whole job when every item
// already reported its own outcome individually.
async function pushRunLogIfPresent(runId: string): Promise<void> {
  const path = runLogFilePath(runId);
  if (!existsSync(path)) return;
  try {
    const text = readFileSync(path, "utf8");
    const gzipped = gzipSync(Buffer.from(text, "utf8"));
    await withAuth((token) => pushRunLog(config.vps_url, token, config.machine_id!, runId, gzipped));
    log.info(`run ${runId}: log pushed (${gzipped.length}B gzipped)`);
  } catch (err) {
    log.warn(`run ${runId}: failed to push log (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
  }
}

// Executes a claimed 'benchmark' job -- one job per run, looping every sweep
// combination in-process exactly as before (MULTIUSER_PLAN.md §1.1). Always
// resolves once the whole sweep (or a user stop) has been worked through;
// individual item failures are reported per-item via safeItemTerminal and
// never abort the loop. Throws only for a genuine "can't even start"
// condition (missing model file, requested build not installed) -- caught by
// workerMain, which reports the job itself failed and reconciles the run.
async function executeBenchmarkJob(payload: BenchmarkJob): Promise<void> {
  const modelPath = resolveModelPath(payload.model);
  if (!existsSync(modelPath)) {
    throw new Error(`model file not found at ${modelPath} (source=${payload.model.source})`);
  }
  let mtpModelPath: string | undefined;
  if (payload.mtp_model) {
    mtpModelPath = resolveModelPath(payload.mtp_model);
    if (!existsSync(mtpModelPath)) {
      throw new Error(`mtp model file not found at ${mtpModelPath} (source=${payload.mtp_model.source})`);
    }
  }

  // The server always resolves the exact tag this run should execute
  // against (see server/src/routes/runs.ts's resolveBuildForRun) -- look it
  // up directly in this worker's own registry rather than assuming it
  // matches activeBuild, which may be a different installed tag for the
  // same backend.
  const resolvedBuild = getInstalledBuild(buildsDir, payload.llama_cpp_build);
  if (!resolvedBuild) {
    throw new Error(`build ${payload.llama_cpp_build} is not installed on this worker`);
  }
  const effectiveBuild: ActiveBuild = {
    tag: resolvedBuild.tag,
    path: resolvedBuild.bench_path,
    serverPath: resolvedBuild.server_path,
  };
  const effectiveBackend: Backend = payload.llama_cpp_backend;

  pauseRequested = false;
  stopRequested = false;
  activeBenchProc = null;

  const items = expandSweep(payload.sweep);
  setRunLogFile(runLogFilePath(payload.run_id));
  log.info(
    formatRunLogHeader(
      payload.run_id,
      payload.model,
      payload.mtp_model,
      effectiveBuild,
      effectiveBackend,
      items.length,
      payload.sweep.repeats
    )
  );
  log.info(`run ${payload.run_id}: ${items.length} test(s) to run`);
  const rawDir = config.raw_json_dir ?? join(config.model_dir, "raw");
  // Tallied across every item regardless of which path handled it, so the
  // run's own finish line can report a success/fail breakdown instead of
  // just "finished all N test(s)" with no indication of how many of those
  // actually succeeded.
  const statusCounts: Partial<Record<TerminalRunItemStatus, number>> = {};
  const tally = (status: TerminalRunItemStatus): void => {
    statusCounts[status] = (statusCounts[status] ?? 0) + 1;
  };
  try {
    for (const item of items) {
      updateJobReport({
        phase: "benchmarking",
        item_idx: item.idx,
        items_total: items.length,
        detail: formatItemParams(item),
      });
      // Checked before *every* item (not just once) since pausing can hold
      // here for a while -- a stop requested while paused must still take
      // effect immediately rather than waiting out the pause.
      if (stopRequested) {
        log.info(`run ${payload.run_id}: stopped, cancelling item ${item.idx} (not started)`);
        await safeItemTerminal(payload.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
        tally("cancelled");
        continue;
      }
      while (pauseRequested && !stopRequested) {
        await sleep(500);
      }
      if (stopRequested) {
        log.info(`run ${payload.run_id}: stopped while paused, cancelling item ${item.idx} (not started)`);
        await safeItemTerminal(payload.run_id, item.idx, { status: "cancelled", error: "cancelled by user" });
        tally("cancelled");
        continue;
      }
      if (item.mtp === "on") {
        if (!effectiveBuild.serverPath) {
          log.error(`run ${payload.run_id}: item ${item.idx} needs mtp but the active build has no llama-server binary`);
          await safeItemTerminal(payload.run_id, item.idx, {
            status: "failed",
            error: "active llama.cpp build has no llama-server binary -- reinstall the build to pick one up",
          });
          tally("failed");
          continue;
        }
        tally(
          await runSweepItemViaServer({
            runId: payload.run_id,
            item,
            repeats: payload.sweep.repeats,
            modelPath,
            mtpModelPath,
            llamaServerPath: effectiveBuild.serverPath,
            port: config.mtp_server_port ?? DEFAULT_MTP_SERVER_PORT,
            backend: effectiveBackend,
            mainGpu: payload.main_gpu,
            timeoutMs: config.bench_timeout_ms,
            rawJsonDir: rawDir,
            modelSizeBytes: payload.model.size_bytes,
          })
        );
      } else {
        tally(
          await runSweepItem({
            runId: payload.run_id,
            item,
            repeats: payload.sweep.repeats,
            modelPath,
            llamaBenchPath: effectiveBuild.path,
            backend: effectiveBackend,
            mainGpu: payload.main_gpu,
            timeoutMs: config.bench_timeout_ms,
            vpsUrl: config.vps_url,
            rawJsonDir: rawDir,
            modelSizeBytes: payload.model.size_bytes,
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
      `run ${payload.run_id} finished all ${items.length} test(s)${stopRequested ? " (stopped early)" : ""}: ${breakdown}`
    );
    log.info(formatRunLogFooter(payload.run_id, items.length, payload.sweep.repeats, statusCounts, stopRequested));
  } finally {
    setRunLogFile(null);
    updateJobReport({ phase: "finalizing", detail: "pushing run log" });
    await pushRunLogIfPresent(payload.run_id);
  }
}

async function executeInstallBuildJob(payload: InstallBuildJobPayload): Promise<void> {
  updateJobReport({ phase: "downloading", detail: `${payload.tag} (${payload.asset_name})` });
  log.info(`installing llama.cpp build ${payload.tag} (${payload.asset_name})`);
  const installed = await installBuild({
    buildsDir,
    tag: payload.tag,
    assetName: payload.asset_name,
    downloadUrl: payload.download_url,
  });
  log.info(`installed llama.cpp build ${payload.tag} -> ${installed.bench_path}`);
}

async function executeActivateBuildJob(payload: ActivateBuildJobPayload): Promise<void> {
  validateTag(payload.tag);
  const build = getInstalledBuild(buildsDir, payload.tag);
  if (!build) throw new Error(`build ${payload.tag} is not installed`);
  activeBuild = { tag: build.tag, path: build.bench_path, serverPath: build.server_path };
  persistConfig({
    llama_cpp_build: build.tag,
    llama_bench_path: build.bench_path,
    llama_server_path: build.server_path,
  });
  log.info(`activated llama.cpp build ${activeBuild.tag}`);
}

async function executeDeleteBuildJob(payload: DeleteBuildJobPayload): Promise<void> {
  validateTag(payload.tag);
  if (activeBuild?.tag === payload.tag) {
    throw new Error("cannot delete the active build -- activate a different one first");
  }
  deleteBuild(buildsDir, payload.tag);
  log.info(`deleted llama.cpp build ${payload.tag}`);
}

// jobId identifies this download in activeDownloadReports (workerMain's
// downloadJobPool sets up the initial entry before calling this, and cleans
// it -- and the AbortController this reacts to on pause -- up afterward,
// regardless of outcome). Reporting goes through updateDownloadReport, never
// the serial-job updateJobReport/currentJobReport, since several of these
// can be running at once.
async function executeDownloadModelJob(
  jobId: string,
  payload: DownloadModelJobPayload,
  signal: AbortSignal
): Promise<void> {
  validateHfRepo(payload.hf_repo);
  validateHfFile(payload.hf_file);
  const target = resolveDownloadTarget(payload.hf_file);
  const partPath = `${target}.part`;
  mkdirSync(dirname(target), { recursive: true });
  const progressKey = `${payload.hf_repo}/${payload.hf_file}`;
  const downloadStartedAt = Date.now();
  updateDownloadReport(jobId, { phase: "downloading", bytes: 0, detail: progressKey });

  try {
    // A .part left over from an earlier attempt (paused, crashed, or a
    // transient network drop) is resumed via an HTTP Range request rather
    // than restarted from byte 0 -- see the 206 handling below. Already
    // fully downloaded under a previous job? resolveDownloadTarget's target
    // would exist and this job wouldn't normally have been queued again, but
    // guard cheaply anyway rather than re-fetch a file that's already there.
    let resumeFrom = 0;
    if (existsSync(target)) {
      log.info(`${progressKey} already present at ${target}, skipping re-download`);
    } else {
      try {
        resumeFrom = statSync(partPath).size;
      } catch {
        resumeFrom = 0;
      }
      if (resumeFrom > 0) log.info(`resuming ${progressKey} from byte ${resumeFrom} (${partPath})`);
      else log.info(`downloading ${progressKey} -> ${target}`);

      const sourceUrl = hfResolveUrl(payload.hf_repo, payload.hf_file);
      const headers: Record<string, string> = { "user-agent": "llamatoaster-worker" };
      if (resumeFrom > 0) headers.range = `bytes=${resumeFrom}-`;
      const upstream = await fetch(sourceUrl, { headers, redirect: "follow", signal });
      if (!upstream.body || (!upstream.ok && upstream.status !== 206)) {
        throw new Error(`download failed: ${upstream.status} ${upstream.statusText}`);
      }

      // Some CDN edges/mirrors don't honor Range and just return 200 with
      // the full body -- appending that on top of existing bytes would
      // corrupt the file, so treat it as a fresh download instead of
      // trusting our own resumeFrom.
      const resumed = upstream.status === 206;
      if (resumeFrom > 0 && !resumed) {
        log.warn(`${progressKey}: server ignored the Range request, restarting from byte 0`);
        resumeFrom = 0;
      }

      const contentLength = Number(upstream.headers.get("content-length"));
      const totalBytes = Number.isFinite(contentLength) && contentLength > 0 ? contentLength + resumeFrom : undefined;
      if (totalBytes) updateDownloadReport(jobId, { total_bytes: totalBytes });
      updateDownloadReport(jobId, { bytes: resumeFrom });

      const tracker = new Transform({
        transform(chunk: Buffer, _enc, cb) {
          const cur = activeDownloadReports.get(jobId);
          updateDownloadReport(jobId, { bytes: (cur?.bytes ?? resumeFrom) + chunk.length });
          cb(null, chunk);
        },
      });
      await pipeline(
        Readable.fromWeb(upstream.body as any),
        tracker,
        createWriteStream(partPath, { flags: resumed ? "a" : "w" })
      );
      const elapsedMs = Date.now() - downloadStartedAt;
      log.info(`downloaded ${progressKey} -> ${partPath} in ${elapsedMs}ms`);

      updateDownloadReport(jobId, { phase: "finalizing", detail: "hashing" });
      const { sha256, byteLength } = await hashFile(partPath);
      renameSync(partPath, target);

      updateDownloadReport(jobId, { detail: "reading GGUF metadata" });
      const { n_layer, mtp_layers, expert_count } = await readGgufInfo(target);
      log.info(
        `gguf metadata for ${progressKey}: n_layer=${n_layer ?? "unknown"} mtp_layers=${mtp_layers ?? "unknown"} ` +
          `expert_count=${expert_count ?? "unknown"}`
      );

      // Reflect the new file immediately rather than waiting for the periodic
      // refresh -- see refreshModelDirFilesCache above.
      refreshModelDirFilesCache();

      const reported = await safeReportDownloadResult({
        worker: config.worker_name,
        machine_id: config.machine_id,
        hf_repo: payload.hf_repo,
        hf_file: payload.hf_file,
        ok: true,
        sha256,
        size_bytes: byteLength,
        n_layer,
        mtp_layers,
        expert_count,
      });
      if (!reported) {
        // The file is on disk and hashed, but the server never learned about
        // it -- surface this as a real job failure instead of letting
        // workerMain report {ok: true} for a download the catalog never
        // registered (see server/src/routes/workers.ts's registerModel call).
        throw new Error(`downloaded ${progressKey} but the completion callback never got through`);
      }
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (signal.aborted) {
      // Paused (routes/workers.ts's pause route -> requestDownloadStop) --
      // the .part file is deliberately left in place so a fresh
      // download_model job for the same file resumes instead of restarting.
      // Still reported as a job failure (worker_jobs.status='failed', same
      // posture as a user-stopped benchmark job -- see requestStop's own
      // comment) since nothing further will happen for THIS job.
      const keptBytes = (() => {
        try {
          return statSync(partPath).size;
        } catch {
          return 0;
        }
      })();
      log.info(`model download paused for ${progressKey} (${keptBytes} bytes kept for resume)`);
    } else {
      log.error(`model download failed for ${progressKey}: ${message}`);
    }
    await safeReportDownloadResult({
      worker: config.worker_name,
      machine_id: config.machine_id,
      hf_repo: payload.hf_repo,
      hf_file: payload.hf_file,
      ok: false,
      error: signal.aborted ? "paused by user" : message,
    });
    throw err; // also mark the worker_jobs row itself failed
  }
}

async function executeDeleteModelFileJob(payload: { filename: string }): Promise<void> {
  validateHfFile(payload.filename);
  const target = resolveDownloadTarget(payload.filename);
  if (!existsSync(target)) {
    // Already absent -- the desired end state already holds, so this is
    // treated as a harmless success (idempotent) rather than a failure.
    log.warn(`delete_model_file: file already absent, nothing to do: ${payload.filename}`);
    return;
  }
  unlinkSync(target);
  log.info(`deleted model file ${payload.filename}`);
  refreshModelDirFilesCache();
}

// download_model is deliberately NOT handled here -- workerMain's
// downloadJobPool intercepts and dispatches it before a job ever reaches
// this function (see the loop below), so the type excludes it: a stray call
// with one would be a real bug, not something to silently execute serially.
type SerialQueueJob = Exclude<QueueJob, { type: "download_model" }>;

async function executeJob(job: SerialQueueJob): Promise<void> {
  switch (job.type) {
    case "benchmark":
      return executeBenchmarkJob(job.payload);
    case "install_build":
      return executeInstallBuildJob(job.payload);
    case "activate_build":
      return executeActivateBuildJob(job.payload);
    case "delete_build":
      return executeDeleteBuildJob(job.payload);
    case "delete_model_file":
      return executeDeleteModelFileJob(job.payload);
  }
}

function jobInitialPhase(type: QueueJob["type"]): ActiveJobReport["phase"] {
  return type === "download_model" ? "downloading" : "loading";
}

// Delivered on the heartbeat's control.cancel_job_ids -- ignored unless it
// names the job actually running right now (a stale or mismatched signal,
// e.g. one arriving just after this job already finished on its own,
// naturally no-ops here rather than killing an unrelated process).
function requestStop(jobId: string): void {
  if (currentJobReport?.job_id !== jobId) return;
  stopRequested = true;
  activeBenchProc?.kill("SIGKILL");
  log.info(`job ${jobId}: stop requested via heartbeat control`);
}

// Resolved once by ensureAuthCredential (called near the bottom of this
// file, after hardware detection) and mutated in place by refreshAuth on a
// 401 -- every outbound authenticated call reads these fresh via withAuth
// rather than capturing a value, so a mid-process refresh takes effect on
// the very next call without restarting anything.
let authToken: string;
let refreshToken: string | undefined;
// True once this worker is authenticating via a per-worker session (Stage
// 3) rather than the legacy shared secret (Stage 1) -- only a session can be
// refreshed; a 401 under the shared-secret path just means static
// misconfiguration (wrong/rotated WORKER_SHARED_TOKEN), nothing to retry.
let sessionAuth = false;

// Device-code enrolment (MULTIUSER_PLAN.md §3.1 step 2-3, §3.5) -- called
// only when this worker holds neither a session nor the legacy shared
// secret, i.e. a genuinely fresh install. Blocks until a human approves the
// printed code (or it expires), since there is nothing useful this worker
// can do before it has a credential.
async function enrolDevice(): Promise<void> {
  log.info(`[worker ${config.worker_name}] no credential configured -- starting device enrolment`);
  const start = await startDeviceEnrolment(config.vps_url, {
    machine_id: config.machine_id!,
    hostname: osHostname(),
    platform: detectedHardware.platform,
    arch: detectedHardware.arch,
    hardware: detectedHardware,
  });
  log.info(`[worker ${config.worker_name}] to connect this machine, visit ${config.vps_url}${start.verification_uri}`);
  log.info(`[worker ${config.worker_name}] code: ${start.user_code}  (expires in ${Math.round(start.expires_in / 60)} minutes)`);

  const deadline = Date.now() + start.expires_in * 1000;
  while (Date.now() < deadline) {
    await sleep(start.interval * 1000);
    const poll = await pollDeviceToken(config.vps_url, start.device_code);
    if (poll.state === "approved") {
      persistConfig({ session_token: poll.session_token, refresh_token: poll.refresh_token });
      authToken = poll.session_token;
      refreshToken = poll.refresh_token;
      sessionAuth = true;
      log.info(`[worker ${config.worker_name}] device approved -- connected`);
      return;
    }
    if (poll.state === "expired") {
      throw new Error("device enrolment code expired before it was approved -- restart the worker to try again");
    }
    // pending -- a human hasn't approved it yet, keep polling.
  }
  throw new Error("device enrolment timed out waiting for approval -- restart the worker to try again");
}

// Resolves which credential this worker authenticates with, preferring an
// existing per-worker session, falling back to the legacy shared secret for
// an already-deployed Stage 1 worker, and only running the interactive
// enrolment flow above when neither is configured at all (a fresh install).
async function ensureAuthCredential(): Promise<void> {
  if (config.session_token) {
    authToken = config.session_token;
    refreshToken = config.refresh_token;
    sessionAuth = true;
    return;
  }
  if (config.worker_shared_token) {
    authToken = config.worker_shared_token;
    sessionAuth = false;
    log.info(
      `[worker ${config.worker_name}] using legacy worker_shared_token auth -- remove it from config.json and restart to switch to per-worker device enrolment`
    );
    return;
  }
  await enrolDevice();
}

// Shared by every concurrent refreshAuth call (see below) so the heartbeat
// loop and workerMain's queue loop -- which run concurrently and can both
// hit a 401 around the same moment, e.g. right after a session is revoked
// from Settings -- never each present the refresh token independently. A
// refresh token is single-use (server/src/db/repo.ts's sessionRepo.rotate);
// the second caller to present an already-rotated-away one would trip the
// server's replay detection and get the WHOLE session revoked (server/src/
// routes/sessions.ts) -- worse than either caller just waiting for the
// other's in-flight refresh to finish.
let refreshInFlight: Promise<void> | null = null;

// Rotates this worker's session and swaps authToken/refreshToken in place --
// called reactively on a 401 from any session-authenticated call (see
// withAuth below), never on a timer (see refreshWorkerSession's own doc
// comment for why a fixed schedule isn't needed). Persists the rotated pair
// immediately: the server's refresh-replay detection revokes the whole
// session if a stale refresh token is ever presented again (server/src/
// routes/sessions.ts), so a crash between rotating here and persisting would
// strand this worker on a refresh token the server no longer accepts.
async function refreshAuth(): Promise<void> {
  if (!sessionAuth || !refreshToken) throw new Error("session expired and no refresh token is available");
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = (async () => {
    const rotated = await refreshWorkerSession(config.vps_url, refreshToken!);
    persistConfig({ session_token: rotated.session_token, refresh_token: rotated.refresh_token });
    authToken = rotated.session_token;
    refreshToken = rotated.refresh_token;
    log.info(`[worker ${config.worker_name}] session refreshed`);
  })();
  try {
    await refreshInFlight;
  } finally {
    refreshInFlight = null;
  }
}

// Wraps every session-authenticated outbound call with a single transparent
// retry after refreshing on a 401 -- so a session that went stale while this
// process was asleep/offline (past its sliding-expiry window, see
// refreshWorkerSession's doc comment) self-heals on the very next call
// instead of failing forever. A no-op passthrough under the Stage 1
// shared-token path (sessionAuth false), where a 401 has nothing to refresh.
async function withAuth<T>(fn: (token: string) => Promise<T>): Promise<T> {
  try {
    return await fn(authToken);
  } catch (err) {
    if (sessionAuth && err instanceof HttpError && err.status === 401) {
      await refreshAuth();
      return await fn(authToken);
    }
    throw err;
  }
}

const HEARTBEAT_INTERVAL_MS = 10_000;

// Runs forever, independent of job execution (MULTIUSER_PLAN.md §1.9) --
// this is what lets the server see this worker as "busy" with live progress
// for the whole duration of a job, and what delivers cancel/pause control
// directives with a worst-case ~10s latency.
function startHeartbeatLoop(): void {
  setInterval(async () => {
    try {
      const state = await collectState();
      const res = await withAuth((token) =>
        postHeartbeat(config.vps_url, token, state, currentJobReport, Array.from(activeDownloadReports.values()), 10_000)
      );
      for (const jobId of res.control.cancel_job_ids) {
        requestStop(jobId);
        requestDownloadStop(jobId);
      }
      pauseRequested = res.control.pause;
    } catch (err) {
      // Transient -- the server's lease is 6 heartbeats wide (LEASE_MS), so
      // one or two missed heartbeats in a row is recoverable without any
      // action here.
      log.debug(`heartbeat failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
    }
  }, HEARTBEAT_INTERVAL_MS).unref();
}

// Comfortably above the server's own ~25s long-poll window (LONG_POLL_MS,
// server/src/routes/queue.ts) so this never aborts a poll that was about to
// return 204 on its own.
const QUEUE_POLL_TIMEOUT_MS = 35_000;

function maxConcurrentDownloads(): number {
  const n = config.max_concurrent_downloads;
  return Number.isInteger(n) && (n as number) > 0 ? (n as number) : 2;
}

// One promise per in-flight download, keyed by job_id -- purely a pool-size
// gate for workerMain's loop below (Promise.race lets it wake up as soon as
// any one slot frees, rather than polling on a timer). Distinct from
// activeDownloadReports (progress) and activeDownloadControllers (abort) --
// this one only tracks "is a slot occupied."
const activeDownloadJobs = new Map<string, Promise<void>>();

// Executes one download_model job to completion and reports the outcome --
// the download equivalent of workerMain's serial try/executeJob/
// reportJobResult block below, but scoped to this one job's own state
// (activeDownloadReports/activeDownloadControllers) rather than the shared
// busy/currentJobReport globals, since several of these run concurrently.
async function runDownloadJob(job: { job_id: string; payload: DownloadModelJobPayload }): Promise<void> {
  const controller = new AbortController();
  activeDownloadControllers.set(job.job_id, controller);
  activeDownloadReports.set(job.job_id, { job_id: job.job_id, phase: "downloading" });
  log.info(`claimed job ${job.job_id} (download_model)`);
  try {
    await executeDownloadModelJob(job.job_id, job.payload, controller.signal);
    await withAuth((token) => reportJobResult(config.vps_url, token, config.machine_id!, job.job_id, { ok: true }));
    log.info(`job ${job.job_id} (download_model) completed`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error(`job ${job.job_id} (download_model) failed: ${message}`);
    try {
      await withAuth((token) =>
        reportJobResult(config.vps_url, token, config.machine_id!, job.job_id, { ok: false, error: message })
      );
    } catch (reportErr) {
      log.error(
        `could not report job ${job.job_id} failure to server: ${
          reportErr instanceof Error ? reportErr.message : String(reportErr)
        }`
      );
    }
  } finally {
    activeDownloadReports.delete(job.job_id);
    activeDownloadControllers.delete(job.job_id);
  }
}

// The main loop: pull one job at a time, execute it to completion, report
// the outcome, repeat forever -- EXCEPT download_model jobs, which are
// dispatched into a bounded concurrent pool (activeDownloadJobs,
// maxConcurrentDownloads) instead of being awaited inline, so the loop can
// go straight back to polling rather than blocking for the whole transfer.
// Every other job type stays exactly as serial as before: the loop still
// only polls for a NEW job once nothing else is being awaited here, so e.g.
// a benchmark job claimed while downloads are already running won't itself
// start until any job ahead of it in this loop finishes -- only downloads
// get to run alongside one another. Only ever polls the queue while able to
// act on what it might get back (MULTIUSER_PLAN.md §1.3) -- heartbeating
// (busy or idle, and carrying every active download's own progress) is
// entirely the separate loop above.
async function workerMain(): Promise<void> {
  while (true) {
    try {
      if (activeDownloadJobs.size >= maxConcurrentDownloads()) {
        // Every download slot is full -- wait for one to free rather than
        // claim a job (of any type) we can't act on yet, which would
        // otherwise just sit 'claimed' un-executed until its lease expires.
        await Promise.race(activeDownloadJobs.values());
        continue;
      }

      const state = await collectState();
      const job = await withAuth((token) => pollQueue(config.vps_url, token, state, QUEUE_POLL_TIMEOUT_MS));
      if (!job) continue;

      if (job.type === "download_model") {
        const p = runDownloadJob(job).finally(() => activeDownloadJobs.delete(job.job_id));
        activeDownloadJobs.set(job.job_id, p);
        continue; // not awaited -- poll again immediately, this one runs in the background
      }

      busy = true;
      currentJobReport = { job_id: job.job_id, phase: jobInitialPhase(job.type) };
      pauseRequested = false;
      stopRequested = false;
      activeBenchProc = null;
      log.info(`claimed job ${job.job_id} (${job.type})`);
      try {
        await executeJob(job);
        await withAuth((token) => reportJobResult(config.vps_url, token, config.machine_id!, job.job_id, { ok: true }));
        log.info(`job ${job.job_id} (${job.type}) completed`);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log.error(`job ${job.job_id} (${job.type}) failed: ${message}`);
        try {
          await withAuth((token) =>
            reportJobResult(config.vps_url, token, config.machine_id!, job.job_id, {
              ok: false,
              error: message,
            })
          );
        } catch (reportErr) {
          log.error(
            `could not report job ${job.job_id} failure to server: ${
              reportErr instanceof Error ? reportErr.message : String(reportErr)
            }`
          );
        }
      } finally {
        busy = false;
        currentJobReport = null;
        pauseRequested = false;
        stopRequested = false;
        activeBenchProc = null;
      }
    } catch (err) {
      log.error(`worker loop error: ${err instanceof Error ? err.message : String(err)}`);
      await sleep(10_000);
    }
  }
}

// Optional, off by default (config.debug_port unset) -- a purely local
// diagnostic endpoint, never used for dispatch (MULTIUSER_PLAN.md §1.11).
if (config.debug_port) {
  const debugServer = createServer((req, res) => {
    if (req.url === "/health") {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          ok: true,
          machine_id: config.machine_id,
          busy,
          backend,
          active_build: activeBuild?.tag ?? null,
          active_job: currentJobReport,
        })
      );
      return;
    }
    res.statusCode = 404;
    res.end(JSON.stringify({ error: "not found" }));
  });
  debugServer.listen(config.debug_port, "127.0.0.1", () => {
    log.info(
      `[worker ${config.worker_name}] debug endpoint listening on 127.0.0.1:${config.debug_port} (local troubleshooting only)`
    );
  });
}

// Once at startup (covers a worker that's restarted often enough that a
// periodic timer alone might rarely fire) and then daily -- a long-lived
// pm2-managed process is the common case, but there's no need for this to
// run any more often than that for a "delete stuff older than N days" check.
const RAW_JSON_PRUNE_INTERVAL_MS = 24 * 60 * 60 * 1000;
pruneRawJson();
setInterval(pruneRawJson, RAW_JSON_PRUNE_INTERVAL_MS).unref();

// Populate the model-files cache right away so the first heartbeat after
// startup doesn't report an empty list, then keep it fresh on a timer -- see
// refreshModelDirFilesCache above.
const MODEL_LIST_REFRESH_INTERVAL_MS = 2 * 60 * 1000;
refreshModelDirFilesCache();
setInterval(refreshModelDirFilesCache, MODEL_LIST_REFRESH_INTERVAL_MS).unref();

log.info(`[worker ${config.worker_name}] no inbound HTTP listener -- pull queue only (tailnet not required)`);
await ensureAuthCredential();
startHeartbeatLoop();
void workerMain();

function shutdown(signal: string): void {
  log.info(`received ${signal}, shutting down...`);
  process.exit(0);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
