import { primaryGpu, readHelper, type HelperReading } from "./helper";
// Service worker: everything that talks to the network. Content scripts on
// huggingface.co ask it for a repo's file list and for each file's header
// facts over a port, and get progress back on the same port. Facts are cached
// in chrome.storage.local (they are a few KB per file), so a page opened again
// costs no download.

import { HeaderError, readModelFacts, type ModelFacts } from "./hfHeader";
import { getSettings } from "./settings";

// The toolbar button opens the popup, or the side panel once the user has
// docked it there (the popup's "Side panel" button; "Use popup" undoes it).
async function applyDock() {
  try {
    const { sidePanel } = await getSettings();
    await chrome.action.setPopup({ popup: sidePanel ? "" : "popup.html" });
    await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: sidePanel });
  } catch (err) {
    console.warn("LlamaToaster: could not set the toolbar button", err);
  }
}
let lastDock: boolean | undefined;
chrome.storage.onChanged.addListener((changes) => {
  const next = (changes.settings?.newValue as { sidePanel?: boolean } | undefined)?.sidePanel;
  if (changes.settings && next !== lastDock) {
    lastDock = next;
    void applyDock();
  }
});
chrome.runtime.onStartup.addListener(() => void applyDock());
chrome.runtime.onInstalled.addListener(() => void applyDock());
void applyDock();

const FACTS_VERSION = 1;
const MAX_CACHED = 400;

interface CachedFacts {
  v: number;
  key: string;
  at: number;
  facts: ModelFacts;
}

const factsKey = (repo: string, file: string, sha: string | null) => `facts:${repo}/${file}@${sha ?? "main"}`;

async function cachedFacts(key: string): Promise<ModelFacts | null> {
  const got = await chrome.storage.local.get(key);
  const c = got[key] as CachedFacts | undefined;
  return c && c.v === FACTS_VERSION ? c.facts : null;
}

async function storeFacts(key: string, facts: ModelFacts): Promise<void> {
  await chrome.storage.local.set({ [key]: { v: FACTS_VERSION, key, at: Date.now(), facts } satisfies CachedFacts });
  const all = await chrome.storage.local.get(null);
  const keys = Object.keys(all).filter((k) => k.startsWith("facts:"));
  if (keys.length > MAX_CACHED) {
    const oldest = keys.sort((a, b) => (all[a] as CachedFacts).at - (all[b] as CachedFacts).at).slice(0, keys.length - MAX_CACHED);
    await chrome.storage.local.remove(oldest);
  }
}

export interface RepoFile {
  path: string;
  size: number | null;
}

async function listRepo(repo: string): Promise<{ sha: string | null; files: RepoFile[] }> {
  const res = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`, { credentials: "include" });
  if (!res.ok) throw new Error(`Hugging Face API: HTTP ${res.status}`);
  const body = (await res.json()) as { sha?: string; siblings?: { rfilename: string; size?: number }[] };
  return {
    sha: body.sha ?? null,
    files: (body.siblings ?? []).map((s) => ({ path: s.rfilename, size: s.size ?? null })),
  };
}

type Req =
  | { type: "list"; id: number; repo: string }
  | { type: "facts"; id: number; repo: string; file: string; sha: string | null; force?: boolean; cacheOnly?: boolean }
  | { type: "hasCached"; id: number; repo: string }
  | { type: "ram"; id: number };

// One helper call every few seconds at most: it spawns a process.
let helperCache: { at: number; reading: HelperReading | null } | null = null;
async function helperReading(): Promise<HelperReading | null> {
  if (helperCache && Date.now() - helperCache.at < 5000) return helperCache.reading;
  const r = await readHelper();
  helperCache = { at: Date.now(), reading: r.ok ? r.reading : null };
  return helperCache.reading;
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "lt") return;
  let alive = true;
  port.onDisconnect.addListener(() => {
    alive = false;
  });
  const send = (msg: unknown) => {
    if (alive) port.postMessage(msg);
  };

  port.onMessage.addListener(async (req: Req) => {
    try {
      if (req.type === "list") {
        send({ id: req.id, ok: true, result: await listRepo(req.repo) });
      } else if (req.type === "hasCached") {
        const all = await chrome.storage.local.get(null);
        send({ id: req.id, ok: true, result: Object.keys(all).some((k) => k.startsWith(`facts:${req.repo}/`)) });
      } else if (req.type === "ram") {
        const [m, h] = await Promise.all([chrome.system.memory.getInfo(), helperReading()]);
        send({ id: req.id, ok: true, result: { totalMib: m.capacity / 1048576, availableMib: m.availableCapacity / 1048576, helper: primaryGpu(h), llamaRamMib: h?.llamaRamMib ?? 0 } });
      } else if (req.type === "facts") {
        const key = factsKey(req.repo, req.file, req.sha);
        const hit = req.force ? null : await cachedFacts(key);
        if (hit || req.cacheOnly) {
          send({ id: req.id, ok: true, result: { facts: hit, cached: true } });
          return;
        }
        const facts = await readModelFacts(req.repo, req.file, {
          revision: req.sha ?? "main",
          onProgress: (p) => send({ id: req.id, progress: p }),
        });
        await storeFacts(key, facts);
        send({ id: req.id, ok: true, result: { facts, cached: false } });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      send({ id: req.id, ok: false, error: message, reason: err instanceof HeaderError ? err.reason : "http" });
    }
  });
});
