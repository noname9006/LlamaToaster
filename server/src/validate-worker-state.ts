import { BadRequestError } from "./errors.js";
import type { WorkerStatePush, HardwareInfo, InstalledBuild, ModelDirFile, ActiveJobReport } from "../../shared/types.js";

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

function parseModelFiles(
  value: unknown
): (ModelDirFile & { n_layer?: number | null; mtp_layers?: number | null })[] {
  return requireArray(value, "model_files")
    .slice(0, LIMITS.maxModelFiles)
    .map((f, i) => {
      if (typeof f !== "object" || f === null) throw new BadRequestError(`model_files[${i}] must be an object`);
      const row = f as Record<string, unknown>;
      return {
        path: sanitizeString(row.path, `model_files[${i}].path`),
        size_bytes: requireNumber(row.size_bytes, `model_files[${i}].size_bytes`),
        n_layer: optionalNumber(row.n_layer, `model_files[${i}].n_layer`) ?? null,
        mtp_layers: optionalNumber(row.mtp_layers, `model_files[${i}].mtp_layers`) ?? null,
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

// Present only on the heartbeat route, and only while a job is running --
// req.body there is a WorkerStatePush with this one extra field alongside it
// (MULTIUSER_PLAN.md §1.5). Absent/null means idle, not a validation error.
export function parseActiveJobReport(body: unknown): ActiveJobReport | undefined {
  if (typeof body !== "object" || body === null) return undefined;
  const raw = (body as Record<string, unknown>).active_job;
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "object") throw new BadRequestError("active_job must be an object");
  const a = raw as Record<string, unknown>;

  if (!ACTIVE_JOB_PHASES.includes(a.phase as (typeof ACTIVE_JOB_PHASES)[number])) {
    throw new BadRequestError(`active_job.phase must be one of ${ACTIVE_JOB_PHASES.join(", ")}`);
  }

  return {
    job_id: sanitizeString(a.job_id, "active_job.job_id"),
    phase: a.phase as ActiveJobReport["phase"],
    bytes: optionalNumber(a.bytes, "active_job.bytes"),
    total_bytes: optionalNumber(a.total_bytes, "active_job.total_bytes"),
    item_idx: optionalNumber(a.item_idx, "active_job.item_idx"),
    items_total: optionalNumber(a.items_total, "active_job.items_total"),
    detail: optionalString(a.detail, "active_job.detail", 1024), // best-effort human text, e.g. a stderr line
  };
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
