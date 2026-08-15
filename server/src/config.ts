import type { Backend } from "../../shared/types.js";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export interface WorkerDef {
  name: string;
  backend: Backend;
  llama_cpp_build: string;
  url: string;
  // True for the tailnet node this orchestrator process itself runs on
  // (typically the VPS, with its worker co-located and managed by the
  // deploy playbook/systemd) -- used to decide which cards get the
  // copy-paste bootstrap/restart commands, since those only make sense for
  // a worker you'd manually set up on a *different* box.
  is_self: boolean;
}

// Every machine on the Tailscale network is a worker candidate: rather than
// hand-maintaining a list of URLs (which goes stale the moment someone adds
// or reinstalls a box, and silently duplicates a worker if the same machine
// ever ends up reachable under two different tailnet identities), the
// orchestrator asks Tailscale itself who's on the network via `tailscale
// status --json`, then probes each peer's own /health on WORKER_PORT to see
// what (if anything) is running there. A peer that isn't running a worker
// yet still shows up -- as "can't be reached" with setup instructions,
// same as an offline worker -- rather than being silently omitted.
const WORKER_PORT = Number(process.env.WORKER_PORT) || 8080;
const WORKER_DISCOVERY_TIMEOUT_MS = 5_000;
// How long a discovered (or placeholder) worker identity is trusted before
// the next listWorkers() call re-probes it. Without this, every single page
// load of the Workers/Models/Dashboard pages -- each of which calls
// listWorkers() one or more times -- re-fetches /health from every tailnet
// peer, so one slow/offline peer (up to WORKER_DISCOVERY_TIMEOUT_MS) stalls
// the whole page every time, not just once.
const DISCOVERY_TTL_MS = 15_000;
const TAILSCALE_STATUS_TIMEOUT_MS = 5_000;
// Separate, shorter-lived cache for the tailnet member list itself -- a
// peer joining/leaving should show up reasonably promptly, but every page
// load still shouldn't shell out to `tailscale status` more than once per
// TTL.
const TAILSCALE_STATUS_TTL_MS = 15_000;

interface TailnetNode {
  id: string;
  ip: string;
  hostname: string;
  loginName: string;
  isSelf: boolean;
}

interface RawTailscalePeer {
  ID: string;
  HostName?: string;
  UserID?: number;
  TailscaleIPs?: string[];
}

interface TailscaleStatusJson {
  Self?: RawTailscalePeer;
  Peer?: Record<string, RawTailscalePeer>;
  User?: Record<string, { LoginName?: string }>;
}

function firstIPv4(ips: string[] | undefined): string | undefined {
  return ips?.find((ip) => /^\d+\.\d+\.\d+\.\d+$/.test(ip));
}

// Tailscale login names are full identifiers like "alice@github" or
// "alice@gmail.com" -- the part before "@" is what `tailscale status`'s own
// table output shows, so mirror that convention rather than the full string.
function shortLogin(loginName: string | undefined): string {
  if (!loginName) return "unknown";
  const at = loginName.indexOf("@");
  return at === -1 ? loginName : loginName.slice(0, at);
}

function parseTailscaleStatus(json: string): TailnetNode[] {
  const data = JSON.parse(json) as TailscaleStatusJson;
  const users = data.User ?? {};
  const raw: Array<{ peer: RawTailscalePeer; isSelf: boolean }> = [
    ...(data.Self ? [{ peer: data.Self, isSelf: true }] : []),
    ...Object.values(data.Peer ?? {}).map((peer) => ({ peer, isSelf: false })),
  ];
  const nodes: TailnetNode[] = [];
  for (const { peer, isSelf } of raw) {
    const ip = firstIPv4(peer.TailscaleIPs);
    if (!ip || !peer.HostName) continue;
    const login = peer.UserID !== undefined ? users[String(peer.UserID)]?.LoginName : undefined;
    nodes.push({ id: peer.ID, ip, hostname: peer.HostName.toLowerCase(), loginName: shortLogin(login), isSelf });
  }
  // Self first, then peers alphabetically by hostname -- gives a stable
  // "first worker" (the default trigger target, see getDefaultWorker) and a
  // stable Workers-page order, since Go's JSON map iteration order for
  // "Peer" isn't guaranteed from one `tailscale status` call to the next.
  nodes.sort((a, b) => (a.isSelf === b.isSelf ? a.hostname.localeCompare(b.hostname) : a.isSelf ? -1 : 1));
  return nodes;
}

let tailnetCache: { nodes: TailnetNode[]; fetchedAt: number } | null = null;
let warnedTailscaleUnavailable = false;

async function fetchTailnetNodes(): Promise<TailnetNode[]> {
  const { stdout } = await execFileAsync("tailscale", ["status", "--json"], {
    timeout: TAILSCALE_STATUS_TIMEOUT_MS,
  });
  return parseTailscaleStatus(stdout);
}

async function getTailnetNodes(): Promise<TailnetNode[]> {
  if (tailnetCache && Date.now() - tailnetCache.fetchedAt < TAILSCALE_STATUS_TTL_MS) {
    return tailnetCache.nodes;
  }
  try {
    const nodes = await fetchTailnetNodes();
    tailnetCache = { nodes, fetchedAt: Date.now() };
    warnedTailscaleUnavailable = false;
    return nodes;
  } catch (err) {
    if (!warnedTailscaleUnavailable) {
      warnedTailscaleUnavailable = true;
      const message = err instanceof Error ? err.message : String(err);
      console.error(
        `"tailscale status --json" failed (${message}) -- is Tailscale installed, running, and on PATH for this process? Workers list will be empty until it succeeds.`
      );
    }
    // Keep serving whatever we last had (even if stale) rather than having
    // every worker vanish from the UI over one transient failure.
    return tailnetCache?.nodes ?? [];
  }
}

interface HealthCacheEntry {
  backend: Backend;
  llama_cpp_build: string;
  fetchedAt: number;
}
const healthCache = new Map<string, HealthCacheEntry>();

async function probeHealth(url: string): Promise<{ backend: Backend; llama_cpp_build: string } | undefined> {
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(WORKER_DISCOVERY_TIMEOUT_MS) });
    if (!res.ok) return undefined;
    const health = (await res.json()) as { backend?: unknown; active_build?: unknown };
    if (typeof health.backend !== "string") return undefined;
    return {
      backend: health.backend,
      llama_cpp_build: typeof health.active_build === "string" ? health.active_build : "none",
    };
  } catch {
    return undefined;
  }
}

async function discover(node: TailnetNode): Promise<WorkerDef> {
  const url = `http://${node.ip}:${WORKER_PORT}`;
  const name = `${node.loginName}@${node.hostname}`;
  const cached = healthCache.get(node.id);
  if (cached && Date.now() - cached.fetchedAt < DISCOVERY_TTL_MS) {
    return { name, backend: cached.backend, llama_cpp_build: cached.llama_cpp_build, url, is_self: node.isSelf };
  }
  const health = await probeHealth(url);
  // On a failed probe, fall back to whatever this node last successfully
  // reported (so a momentary blip doesn't flash "unknown" over real data)
  // rather than a placeholder -- same as before, just keyed by tailnet node
  // identity instead of a hand-configured URL. A node that's never once
  // answered has no such fallback, so it's genuinely "unknown".
  const backend = health?.backend ?? cached?.backend ?? "unknown";
  const llama_cpp_build = health?.llama_cpp_build ?? cached?.llama_cpp_build ?? "unknown";
  healthCache.set(node.id, { backend, llama_cpp_build, fetchedAt: Date.now() });
  return { name, backend, llama_cpp_build, url, is_self: node.isSelf };
}

export async function listWorkers(): Promise<WorkerDef[]> {
  const nodes = await getTailnetNodes();
  return Promise.all(nodes.map(discover));
}

export async function getWorkerForRun(name: string): Promise<WorkerDef | undefined> {
  return (await listWorkers()).find((w) => w.name === name);
}

export async function getDefaultWorker(): Promise<WorkerDef | undefined> {
  return (await listWorkers())[0];
}
