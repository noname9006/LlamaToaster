import type { Backend } from "../../shared/types.js";

export interface WorkerDef {
  name: string;
  backend: Backend;
  llama_cpp_build: string;
  url: string;
}

// Each worker already knows its own name/backend/active build and reports
// them live via /health (see worker/src/index.ts) -- WORKERS only needs to
// say *where* a worker is (its Tailscale URL), never what it is. That's
// discovered here instead of being hand-declared and left to go stale.
const WORKER_DISCOVERY_TIMEOUT_MS = 5_000;

function loadWorkerUrls(): string[] {
  const raw = process.env.WORKERS;
  if (!raw) {
    const url = process.env.DEFAULT_WORKER_URL;
    return url ? [url] : [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(`WORKERS env var is not valid JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || !parsed.every((v) => typeof v === "string" && v.length > 0)) {
    throw new Error('WORKERS env var must be a non-empty JSON array of worker URLs, e.g. ["http://100.x.x.x:8080"]');
  }
  return parsed;
}

const workerUrls: string[] = loadWorkerUrls();

// Last successful discovery per URL. A worker that's offline the moment
// it's looked up falls back to this (or, if it's never once answered, a
// placeholder derived from its URL) so it still shows up -- as
// "Inaccessible", same as today -- instead of silently disappearing from
// the list.
const cache = new Map<string, WorkerDef>();

function placeholderName(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

async function discover(url: string): Promise<WorkerDef> {
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(WORKER_DISCOVERY_TIMEOUT_MS) });
    if (!res.ok) throw new Error(`worker responded ${res.status}`);
    const health = (await res.json()) as { worker?: unknown; backend?: unknown; active_build?: unknown };
    if (typeof health.worker === "string" && health.worker && typeof health.backend === "string") {
      const def: WorkerDef = {
        name: health.worker,
        backend: health.backend,
        llama_cpp_build: typeof health.active_build === "string" ? health.active_build : "none",
        url,
      };
      cache.set(url, def);
      return def;
    }
    throw new Error("worker's /health response is missing worker/backend");
  } catch {
    return cache.get(url) ?? { name: placeholderName(url), backend: "unknown", llama_cpp_build: "unknown", url };
  }
}

export async function listWorkers(): Promise<WorkerDef[]> {
  return Promise.all(workerUrls.map(discover));
}

export async function getWorkerForRun(name: string): Promise<WorkerDef | undefined> {
  return (await listWorkers()).find((w) => w.name === name);
}

export async function getDefaultWorker(): Promise<WorkerDef | undefined> {
  return (await listWorkers())[0];
}
