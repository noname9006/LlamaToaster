import { BadRequestError } from "./errors.js";
import type {
  WorkerStatePush,
  HardwareInfo,
  InstalledBuild,
  ModelDirFile,
  ActiveJobReport,
} from "../../shared/types.js";
import { LOCAL_MODEL_STATES } from "../../shared/types.js";

// A worker is a semi-trusted client, not part of the server -- MULTIUSER_PLAN.md
// §1.8. Everything here is validated and capped before it's ever persisted or
// (Stage 5) fed into an LLM context: reject malformed input outright, never
// silently coerce it, and cap every array/string so one misbehaving or
// oversized worker (a huge model_dir, a hand-edited config) can't wedge
// itself with 413s or feed unbounded data downstream.
const LIMITS = {
  maxModelFiles: 2_000,
  maxBuilds: 100,
  maxGpus: 16,
  maxCapabilities: 50,
  maxString: 256,
} as const;

// Strips C0/DEL control characters (newlines, tabs, escape sequences, ...)
// -- none of these fields are ever meant to contain them, and Stage 5 feeds
// worker-reported strings (gpu model, hostname, ...) into another user's AI
// context, so a control character here is either junk or an attempt at one.
function sanitizeString(value: unknown, field: string, max: number = LIMITS.maxString): string {
  if (typeof value !== "string") throw new BadRequestError(`${field} must be a string`);
  return value.replace(/[\x00-\x1f\x7f]/g, "").slice(0, max);
}

function optionalString(value: unknown, field: string, max: number = LIMITS.maxString): string | undefined {
  if (value === undefined || value === null) return undefined;
  return sanitizeString(value, field, max);
}

function requireNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new BadRequestError(`${field} must be a number`);
  return value;
}

function optionalNumber(value: unknown, field: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  return requireNumber(value, field);
}

function optionalBoolean(value: unknown, field: string): boolean | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "boolean") throw new BadRequestError(`${field} must be a boolean`);
  return value;
}

function requireArray(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new BadRequestError(`${field} must be an array`);
  return value;
}

function parseHardware(value: unknown): HardwareInfo {
  if (typeof value !== "object" || value === null) throw new BadRequestError("hardware is required");
  const h = value as Record<string, unknown>;
  const cpuRaw = h.cpu;
  if (typeof cpuRaw !== "object" || cpuRaw === null) throw new BadRequestError("hardware.cpu is required");
  const cpu = cpuRaw as Record<string, unknown>;
  const flags = requireArray(cpu.flags ?? [], "hardware.cpu.flags")
    .slice(0, LIMITS.maxCapabilities)
    .map((f, i) => sanitizeString(f, `hardware.cpu.flags[${i}]`));

  const gpuArray = requireArray(h.gpu ?? [], "hardware.gpu").slice(0, LIMITS.maxGpus);
  const gpu = gpuArray.map((g, i) => {
    if (typeof g !== "object" || g === null) throw new BadRequestError(`hardware.gpu[${i}] must be an object`);
    const row = g as Record<string, unknown>;
    return {
      vendor: sanitizeString(row.vendor, `hardware.gpu[${i}].vendor`),
      model: sanitizeString(row.model, `hardware.gpu[${i}].model`),
      vram_mb: optionalNumber(row.vram_mb, `hardware.gpu[${i}].vram_mb`) ?? null,
      vram_dynamic: optionalBoolean(row.vram_dynamic, `hardware.gpu[${i}].vram_dynamic`),
    };
  });

  return {
    platform: sanitizeString(h.platform, "hardware.platform"),
    arch: sanitizeString(h.arch, "hardware.arch"),
    cpu: {
      manufacturer: sanitizeString(cpu.manufacturer, "hardware.cpu.manufacturer"),
      brand: sanitizeString(cpu.brand, "hardware.cpu.brand"),
      flags,
      cores: requireNumber(cpu.cores, "hardware.cpu.cores"),
    },
    gpu,
    mem_total_bytes: optionalNumber(h.mem_total_bytes, "hardware.mem_total_bytes"),
  };
}

function parseInstalledBuilds(value: unknown): InstalledBuild[] {
  return requireArray(value, "installed_builds")
    .slice(0, LIMITS.maxBuilds)
    .map((b, i) => {
      if (typeof b !== "object" || b === null) throw new BadRequestError(`installed_builds[${i}] must be an object`);
      const row = b as Record<string, unknown>;
      return {
        tag: sanitizeString(row.tag, `installed_builds[${i}].tag`),
        asset_name: sanitizeString(row.asset_name, `installed_builds[${i}].asset_name`),
        installed_at: requireNumber(row.installed_at, `installed_builds[${i}].installed_at`),
        active: row.active === true,
        bench_path: optionalString(row.bench_path, `installed_builds[${i}].bench_path`),
        server_path: optionalString(row.server_path, `installed_builds[${i}].server_path`),
      };
    });
}

function parseModelFiles(value: unknown): ModelDirFile[] {
  return requireArray(value, "model_files")
    .slice(0, LIMITS.maxModelFiles)
    .map((f, i) => {
      if (typeof f !== "object" || f === null) throw new BadRequestError(`model_files[${i}] must be an object`);
      const row = f as Record<string, unknown>;

      // sha256/state/hf_match carry the worker's hash-based identification of
      // each file (see worker/src/model-scanner.ts's resolveHfMetadata). They
      // used to be silently dropped here, so the server never learned HOW a
      // local file was identified -- hash-lookup results dead-ended at the
      // worker and the Models page could only ever match by filename.
      const rawSha = optionalString(row.sha256, `model_files[${i}].sha256`, 64);
      // Optional, but when present it must actually look like a SHA-256 hex
      // digest -- it keys catalog registration (models.id), not just display.
      if (rawSha != null && !/^[0-9a-f]{64}$/i.test(rawSha)) {
        throw new BadRequestError(`model_files[${i}].sha256 must be a 64-char hex digest`);
      }
      const state = optionalString(row.state, `model_files[${i}].state`, 16);
      if (state != null && !LOCAL_MODEL_STATES.includes(state as (typeof LOCAL_MODEL_STATES)[number])) {
        throw new BadRequestError(
          `model_files[${i}].state must be one of: ${LOCAL_MODEL_STATES.join(", ")}`
        );
      }
      let hf_match: ModelDirFile["hf_match"];
      if (row.hf_match === undefined || row.hf_match === null) {
        hf_match = null;
      } else {
        if (typeof row.hf_match !== "object") throw new BadRequestError(`model_files[${i}].hf_match must be an object`);
        const m = row.hf_match as Record<string, unknown>;
        hf_match = {
          repo_id: sanitizeString(m.repo_id, `model_files[${i}].hf_match.repo_id`),
          filename: sanitizeString(m.filename, `model_files[${i}].hf_match.filename`, 512),
          revision: sanitizeString(m.revision ?? "main", `model_files[${i}].hf_match.revision`),
          deleted: m.deleted === true,
        };
      }

      // Per-file GGUF header metadata (see ModelDirFile's doc comments and
      // worker/src/gguf.ts's readGgufInfo). Each field is included ONLY when
      // the worker actually reported a value -- "checked and got null" is
      // deliberately indistinguishable from "never reported" here, so the
      // catalog-adoption code (routes/queue.ts) can tell "a real value to
      // store" from "nothing to store" without a second presence flag.
      const nLayer = optionalNumber(row.n_layer, `model_files[${i}].n_layer`);
      const mtpLayers = optionalNumber(row.mtp_layers, `model_files[${i}].mtp_layers`);
      const expertCount = optionalNumber(row.expert_count, `model_files[${i}].expert_count`);
      const paramCount = optionalNumber(row.param_count, `model_files[${i}].param_count`);
      const quant = optionalString(row.quant, `model_files[${i}].quant`, 32);

      return {
        path: sanitizeString(row.path, `model_files[${i}].path`),
        size_bytes: requireNumber(row.size_bytes, `model_files[${i}].size_bytes`),
        ...(nLayer != null ? { n_layer: nLayer } : {}),
        ...(mtpLayers != null ? { mtp_layers: mtpLayers } : {}),
        ...(expertCount != null ? { expert_count: expertCount } : {}),
        ...(paramCount != null ? { param_count: paramCount } : {}),
        ...(quant != null ? { quant } : {}),
        ...(rawSha != null ? { sha256: rawSha.toLowerCase() } : {}),
        ...(state != null ? { state: state as ModelDirFile["state"] } : {}),
        hf_match,
      };
    });
}

// Reject, don't coerce -- a malformed field throws BadRequestError (400)
// rather than silently substituting a default, so a broken/malicious worker
// gets a clear error instead of a partially-trusted state getting persisted.
export function parseWorkerState(body: unknown): WorkerStatePush {
  if (typeof body !== "object" || body === null) throw new BadRequestError("request body must be an object");
  const b = body as Record<string, unknown>;

  const status = b.status;
  if (status !== "idle" && status !== "busy") throw new BadRequestError("status must be 'idle' or 'busy'");

  return {
    machine_id: sanitizeString(b.machine_id, "machine_id"),
    capabilities: requireArray(b.capabilities ?? [], "capabilities")
      .slice(0, LIMITS.maxCapabilities)
      .map((c, i) => sanitizeString(c, `capabilities[${i}]`)),
    hostname: sanitizeString(b.hostname, "hostname"),
    backend: sanitizeString(b.backend, "backend"),
    hardware: parseHardware(b.hardware),
    installed_builds: parseInstalledBuilds(b.installed_builds ?? []),
    model_files: parseModelFiles(b.model_files ?? []),
    status,
  };
}

const ACTIVE_JOB_PHASES = ["downloading", "extracting", "loading", "benchmarking", "finalizing"] as const;

function parseActiveJobReportShape(raw: unknown, field: string): ActiveJobReport {
  if (typeof raw !== "object" || raw === null) throw new BadRequestError(`${field} must be an object`);
  const a = raw as Record<string, unknown>;

  if (!ACTIVE_JOB_PHASES.includes(a.phase as (typeof ACTIVE_JOB_PHASES)[number])) {
    throw new BadRequestError(`${field}.phase must be one of ${ACTIVE_JOB_PHASES.join(", ")}`);
  }

  return {
    job_id: sanitizeString(a.job_id, `${field}.job_id`),
    phase: a.phase as ActiveJobReport["phase"],
    bytes: optionalNumber(a.bytes, `${field}.bytes`),
    total_bytes: optionalNumber(a.total_bytes, `${field}.total_bytes`),
    item_idx: optionalNumber(a.item_idx, `${field}.item_idx`),
    items_total: optionalNumber(a.items_total, `${field}.items_total`),
    detail: optionalString(a.detail, `${field}.detail`, 1024), // best-effort human text, e.g. a stderr line
  };
}

// Present only on the heartbeat route, and only while a job is running --
// req.body there is a WorkerStatePush with this one extra field alongside it
// (MULTIUSER_PLAN.md §1.5). Absent/null means idle, not a validation error.
export function parseActiveJobReport(body: unknown): ActiveJobReport | undefined {
  if (typeof body !== "object" || body === null) return undefined;
  const raw = (body as Record<string, unknown>).active_job;
  if (raw === undefined || raw === null) return undefined;
  return parseActiveJobReportShape(raw, "active_job");
}

// Concurrently-running download_model jobs (worker/src/index.ts's
// downloadJobPool) -- reported alongside active_job on every heartbeat, not
// exclusive with it: a worker can be "idle" (no serial job) while several of
// these are in flight, since only the serial job slot drives
// WorkerStatePush.status. Absent/empty means no downloads active, same
// non-error posture as parseActiveJobReport.
const MAX_ACTIVE_DOWNLOADS_REPORTED = 32; // generous upper bound, not a real concurrency cap

export function parseActiveDownloads(body: unknown): ActiveJobReport[] {
  if (typeof body !== "object" || body === null) return [];
  const raw = (body as Record<string, unknown>).active_downloads;
  if (raw === undefined || raw === null) return [];
  if (!Array.isArray(raw)) throw new BadRequestError("active_downloads must be an array");
  return raw
    .slice(0, MAX_ACTIVE_DOWNLOADS_REPORTED)
    .map((r, i) => parseActiveJobReportShape(r, `active_downloads[${i}]`));
}

// Multi-user Stage 3 (MULTIUSER_PLAN.md §3.4): POST /api/device/start's own
// body -- a worker identifying itself before any session exists, same
// "semi-trusted client" posture as parseWorkerState above.
export interface DeviceStartInput {
  machine_id: string;
  hostname: string;
  platform: string;
  arch: string;
  // Optional so a not-yet-updated worker binary (built before this field
  // existed) can still enrol -- see repo.ts's createPending/reissueEnrolment
  // for how a missing value is handled. Reuses parseHardware, the same
  // validator the heartbeat path (parseWorkerState below) already applies to
  // this exact shape, since a worker reports it identically in both places.
  hardware?: HardwareInfo;
}

export function parseDeviceStart(body: unknown): DeviceStartInput {
  if (typeof body !== "object" || body === null) throw new BadRequestError("request body must be an object");
  const b = body as Record<string, unknown>;
  return {
    machine_id: sanitizeString(b.machine_id, "machine_id"),
    hostname: sanitizeString(b.hostname, "hostname"),
    platform: sanitizeString(b.platform, "platform"),
    arch: sanitizeString(b.arch, "arch"),
    hardware: b.hardware !== undefined ? parseHardware(b.hardware) : undefined,
  };
}

// POST /api/device/token's own body -- just the one opaque code.
export function parseDeviceToken(body: unknown): { device_code: string } {
  if (typeof body !== "object" || body === null) throw new BadRequestError("request body must be an object");
  const b = body as Record<string, unknown>;
  return { device_code: sanitizeString(b.device_code, "device_code", 128) };
}
