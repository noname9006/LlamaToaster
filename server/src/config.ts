import type { Backend } from "../../shared/types.js";

export interface WorkerDef {
  name: string;
  // Any non-empty string -- not restricted to KNOWN_BACKENDS, see
  // shared/types.ts and server/src/github-releases.ts's generic asset
  // matching for why this stays open-ended.
  backend: Backend;
  llama_cpp_build: string;
  url: string;
}

function isWorkerDef(value: unknown): value is WorkerDef {
  if (!value || typeof value !== "object") return false;
  const w = value as Record<string, unknown>;
  return (
    typeof w.name === "string" &&
    w.name.length > 0 &&
    typeof w.url === "string" &&
    w.url.length > 0 &&
    typeof w.llama_cpp_build === "string" &&
    typeof w.backend === "string" &&
    w.backend.length > 0
  );
}

function loadWorkers(): WorkerDef[] {
  const raw = process.env.WORKERS;
  if (!raw) {
    return [
      {
        name: process.env.DEFAULT_WORKER_NAME ?? "default",
        backend: (process.env.DEFAULT_WORKER_BACKEND as WorkerDef["backend"]) ?? "vulkan",
        llama_cpp_build: process.env.DEFAULT_WORKER_BUILD ?? "b10068",
        url: process.env.DEFAULT_WORKER_URL ?? "http://100.0.0.0:8080",
      },
    ];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `WORKERS env var is not valid JSON: ${err instanceof Error ? err.message : String(err)}`
    );
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || !parsed.every(isWorkerDef)) {
    throw new Error(
      'WORKERS env var must be a non-empty JSON array of {name, backend, llama_cpp_build, url} objects'
    );
  }
  return parsed;
}

const workers: WorkerDef[] = loadWorkers();

export function getWorkerForRun(name: string): WorkerDef | undefined {
  return workers.find((w) => w.name === name);
}

export function getDefaultWorker(): WorkerDef | undefined {
  return workers[0];
}

export function listWorkers(): WorkerDef[] {
  return workers;
}
