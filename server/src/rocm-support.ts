// Keeps the "which AMD GPUs does ROCm support" list current by re-reading AMD's
// own documentation once a day, instead of relying on a hand-copied snapshot
// that silently goes stale (shared/rocmSupport.ts is the matcher; its built-in
// list is only the fallback).
//
// Two pages, one per platform, each a plain server-rendered Sphinx table with
// a ✅/❌ cell per GPU:
//   Windows -- HIP SDK "System requirements" (Name | Architecture | LLVM target
//              | Runtime | HIP SDK | ...); we key off the Runtime column
//              (Runtime and HIP SDK agree on every row today).
//   Linux   -- ROCm "System requirements" (GPU | Architecture | LLVM target |
//              Support).
// Ryzen APU rows are skipped: AMD lists them by CPU name ("Ryzen AI Max+ 395")
// but a machine reports its iGPU ("Radeon 8060S"), so those rows could never
// match a detected GPU.
//
// A refresh only ever replaces the stored list with one that passes sanity
// checks (both pages parsed, enough rows, not drastically smaller than the
// previous list) -- a redesigned page must degrade to "keep yesterday's list",
// never to "every AMD GPU now falls back to Vulkan".

import { getDb } from "./db/migrate.js";
import { log } from "./log.js";
import {
  BUILTIN_ROCM_SUPPORT,
  isRocmSupportList,
  normalizeGpuName,
  type RocmPlatformSupport,
  type RocmSupportList,
  type RocmSupportSnapshot,
} from "../../shared/rocmSupport.js";

export const ROCM_WINDOWS_URL = "https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html";
export const ROCM_LINUX_URL = "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html";

const META_LIST_KEY = "rocm_support_list";
const META_FETCHED_AT_KEY = "rocm_support_fetched_at";
const META_ATTEMPT_AT_KEY = "rocm_support_attempt_at";

const REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000;
// After a failed attempt, wait this long before trying again (the timer below
// just checks "is a refresh due?" hourly, so this is also the check cadence).
const RETRY_BACKOFF_MS = 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 20_000;
const MAX_PAGE_CHARS = 5_000_000;
const STARTUP_DELAY_MS = 15_000;

// AMD publishes ~19 supported Windows and ~24 supported Linux GPUs today. A
// parse yielding far fewer means the page layout changed, not that support
// collapsed.
const MIN_SUPPORTED_PER_PLATFORM = 5;
// Refuse a list that lost more than this share of the previous one's supported
// markers in a single refresh.
const MAX_SHRINK_SHARE = 0.5;

// ── parsing ──────────────────────────────────────────────────────────────

function decodeEntities(s: string): string {
  return s
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(parseInt(d, 10)));
}

function cellText(html: string): string {
  return decodeEntities(html.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

// Every <table> as rows of cell texts. Regex-based on purpose: the pages are
// generated, well-formed Sphinx output and this avoids a new HTML-parser
// dependency; the sanity checks below catch a layout change.
export function extractTables(html: string): string[][][] {
  const tables: string[][][] = [];
  for (const t of html.matchAll(/<table[\s\S]*?<\/table>/g)) {
    const rows: string[][] = [];
    for (const r of t[0].matchAll(/<tr[\s\S]*?<\/tr>/g)) {
      const cells = [...r[0].matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/g)].map((c) => cellText(c[1]));
      if (cells.length > 0) rows.push(cells);
    }
    if (rows.length > 0) tables.push(rows);
  }
  return tables;
}

function addRow(out: { supported: Set<string>; unsupported: Set<string> }, name: string, verdict: string): void {
  if (/^\s*amd\s+ryzen/i.test(name)) return;
  const marker = normalizeGpuName(name);
  if (!marker) return;
  if (verdict.includes("✅")) out.supported.add(marker);
  else if (verdict.includes("❌")) out.unsupported.add(marker);
}

function finish(out: { supported: Set<string>; unsupported: Set<string> }): RocmPlatformSupport {
  return { supported: [...out.supported].sort(), unsupported: [...out.unsupported].sort() };
}

// Shared shape: find the tables whose header has a name column, an "LLVM
// target" column and a verdict column, and read name + verdict per row.
function parseGpuTables(html: string, nameHeader: string, verdictHeader: string): RocmPlatformSupport {
  const out = { supported: new Set<string>(), unsupported: new Set<string>() };
  for (const rows of extractTables(html)) {
    const header = rows[0];
    const nameIdx = header.indexOf(nameHeader);
    const verdictIdx = header.indexOf(verdictHeader);
    if (nameIdx < 0 || verdictIdx < 0 || !header.includes("LLVM target")) continue;
    for (const row of rows.slice(1)) {
      if (row.length <= Math.max(nameIdx, verdictIdx)) continue;
      addRow(out, row[nameIdx], row[verdictIdx]);
    }
  }
  return finish(out);
}

export function parseWindowsRocmPage(html: string): RocmPlatformSupport {
  return parseGpuTables(html, "Name", "Runtime");
}

export function parseLinuxRocmPage(html: string): RocmPlatformSupport {
  return parseGpuTables(html, "GPU", "Support");
}

// null = acceptable; otherwise why the parsed list must not be stored.
export function validateRocmList(next: RocmSupportList, previous: RocmSupportList | null): string | null {
  for (const platform of ["win32", "linux"] as const) {
    const n = next[platform].supported.length;
    if (n < MIN_SUPPORTED_PER_PLATFORM) {
      return `${platform}: only ${n} supported GPUs parsed (need >= ${MIN_SUPPORTED_PER_PLATFORM}) -- page layout probably changed`;
    }
    const prev = previous?.[platform].supported.length ?? 0;
    if (prev > 0 && n < prev * (1 - MAX_SHRINK_SHARE)) {
      return `${platform}: supported list shrank from ${prev} to ${n} -- refusing a drop of more than ${MAX_SHRINK_SHARE * 100}%`;
    }
  }
  return null;
}

// ── storage ──────────────────────────────────────────────────────────────

function getMeta(key: string): string | null {
  const row = getDb().prepare(`SELECT value FROM meta WHERE key = ?`).get(key) as { value: string } | undefined;
  return row?.value ?? null;
}

function setMeta(key: string, value: string): void {
  getDb()
    .prepare(`INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`)
    .run(key, value);
}

function readStoredList(): RocmSupportList | null {
  const raw = getMeta(META_LIST_KEY);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRocmSupportList(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function getRocmSupportSnapshot(): RocmSupportSnapshot {
  const stored = readStoredList();
  if (!stored) return { list: BUILTIN_ROCM_SUPPORT, source: "builtin", fetchedAt: null };
  return { list: stored, source: "live", fetchedAt: getMeta(META_FETCHED_AT_KEY) };
}

// ── refresh ──────────────────────────────────────────────────────────────

type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

async function fetchPage(url: string, fetchImpl: FetchLike): Promise<string> {
  const res = await fetchImpl(url, {
    headers: { "user-agent": "LlamaToaster (ROCm support list refresh)", accept: "text/html" },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  if (text.length > MAX_PAGE_CHARS) throw new Error(`page unexpectedly large (${text.length} chars)`);
  return text;
}

export type RocmRefreshOutcome = { ok: true; changed: boolean } | { ok: false; reason: string };

export async function refreshRocmSupport(fetchImpl: FetchLike = fetch): Promise<RocmRefreshOutcome> {
  setMeta(META_ATTEMPT_AT_KEY, String(Date.now()));
  let next: RocmSupportList;
  try {
    const [win, lin] = await Promise.all([fetchPage(ROCM_WINDOWS_URL, fetchImpl), fetchPage(ROCM_LINUX_URL, fetchImpl)]);
    next = { win32: parseWindowsRocmPage(win), linux: parseLinuxRocmPage(lin) };
  } catch (err) {
    return { ok: false, reason: `fetch failed: ${err instanceof Error ? err.message : String(err)}` };
  }

  const previous = readStoredList();
  const invalid = validateRocmList(next, previous);
  if (invalid) return { ok: false, reason: invalid };

  const serialized = JSON.stringify(next);
  const changed = serialized !== JSON.stringify(previous);
  setMeta(META_LIST_KEY, serialized);
  setMeta(META_FETCHED_AT_KEY, new Date().toISOString());
  return { ok: true, changed };
}

// Due when the last successful fetch is over a day old (or there never was
// one), but not more than once per RETRY_BACKOFF_MS so a persistently failing
// source isn't hammered. Timestamps live in the DB, so a restart neither skips
// a due refresh nor triggers a redundant one.
export function isRocmRefreshDue(now = Date.now()): boolean {
  const attempt = Number(getMeta(META_ATTEMPT_AT_KEY) ?? 0);
  if (attempt && now - attempt < RETRY_BACKOFF_MS) return false;
  const fetchedAt = Date.parse(getMeta(META_FETCHED_AT_KEY) ?? "");
  return Number.isNaN(fetchedAt) || now - fetchedAt >= REFRESH_INTERVAL_MS;
}

async function tick(): Promise<void> {
  if (!isRocmRefreshDue()) return;
  const outcome = await refreshRocmSupport();
  if (!outcome.ok) {
    log.warn(`[rocm-support] refresh failed, keeping previous list: ${outcome.reason}`);
    return;
  }
  const snap = getRocmSupportSnapshot();
  log.info(
    `[rocm-support] refreshed (${outcome.changed ? "changed" : "unchanged"}): ` +
      `${snap.list.win32.supported.length} Windows / ${snap.list.linux.supported.length} Linux supported GPUs`
  );
}

let timer: NodeJS.Timeout | null = null;
let startupTimer: NodeJS.Timeout | null = null;
let running = false;

function runTick(): void {
  if (running) return;
  running = true;
  tick()
    .catch((err) => log.warn(`[rocm-support] tick failed: ${err instanceof Error ? err.message : String(err)}`))
    .finally(() => {
      running = false;
    });
}

// ROCM_SUPPORT_REFRESH=false turns the background fetch off (air-gapped
// deployments); the built-in list, or the last stored one, keeps being served.
export function startRocmSupportService(): void {
  if (timer) return;
  if (process.env.ROCM_SUPPORT_REFRESH === "false") {
    log.info("[rocm-support] ROCM_SUPPORT_REFRESH=false -- not refreshing the ROCm support list");
    return;
  }
  // Hourly "is one due?" check; the first one runs shortly after boot so a
  // slow AMD docs site never delays startup.
  timer = setInterval(runTick, RETRY_BACKOFF_MS);
  timer.unref();
  startupTimer = setTimeout(runTick, STARTUP_DELAY_MS);
  startupTimer.unref();
}

export function stopRocmSupportService(): void {
  if (timer) clearInterval(timer);
  if (startupTimer) clearTimeout(startupTimer);
  timer = null;
  startupTimer = null;
}
