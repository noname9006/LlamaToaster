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
// How long a discovered (or placeholder) identity is trusted before the next
// listWorkers() call re-probes it. Without this, every single page load of
// the Workers/Models/Dashboard pages -- each of which calls listWorkers()
// one or more times -- re-fetches /health from every configured worker, so
// one slow/offline worker (up to WORKER_DISCOVERY_TIMEOUT_MS) stalls the
// whole page every time, not just once. A worker that's actually online
// still gets re-checked at most once per TTL, so its reported busy/build
// state can't go stale for more than this long.
const DISCOVERY_TTL_MS = 15_000;

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

// Last discovery per URL, successful or not, plus when it happened. A
// worker that's offline the moment it's looked up falls back to whatever
// was last cached (or, if it's never once answered, a placeholder derived
// from its URL) so it still shows up -- as "Inaccessible", same as today --
// instead of silently disappearing from the list. Re-probing an offline
// worker also only happens once per TTL, not on every request, so a
// persistently-down worker can't single-handedly keep every page load slow.
interface CacheEntry {
  def: WorkerDef;
  fetchedAt: number;
}
const cache = new Map<string, CacheEntry>();

function placeholderName(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

async function probe(url: string): Promise<WorkerDef> {
  const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(WORKER_DISCOVERY_TIMEOUT_MS) });
  if (!res.ok) throw new Error(`worker responded ${res.status}`);
  const health = (await res.json()) as { worker?: unknown; backend?: unknown; active_build?: unknown };
  if (typeof health.worker === "string" && health.worker && typeof health.backend === "string") {
    return {
      name: health.worker,
      backend: health.backend,
      llama_cpp_build: typeof health.active_build === "string" ? health.active_build : "none",
      url,
    };
  }
  throw new Error("worker's /health response is missing worker/backend");
}

async function discover(url: string): Promise<WorkerDef> {
  const cached = cache.get(url);
  if (cached && Date.now() - cached.fetchedAt < DISCOVERY_TTL_MS) {
    return cached.def;
  }
  try {
    const def = await probe(url);
    cache.set(url, { def, fetchedAt: Date.now() });
    return def;
  } catch {
    const def = cached?.def ?? { name: placeholderName(url), backend: "unknown", llama_cpp_build: "unknown", url };
    cache.set(url, { def, fetchedAt: Date.now() });
    return def;
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
