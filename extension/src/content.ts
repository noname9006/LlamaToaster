// Hugging Face model pages: every GGUF quant chip gets the web UI's own fit
// bread mark (shared/breadMarks.ts: a stack of toast, pale to golden, burnt for not enough memory) in place of
// Hugging Face's icons. The "Hardware compatibility" heading above the chips
// becomes a toaster: in manual mode clicking it starts the check, in auto mode
// the check starts by itself; its dial turns while headers are read. All
// headers are read first (background worker, 2 MiB range reads, cached), then
// every chip is marked at once. Settings live in the extension popup.

import toasterSvg from "virtual:toaster-svg";
import { deviceBudget, deviceInUseMib, evaluateFit, type FitAnswer } from "./fit";
import { detectGpu, memoryPicture, type MemoryPicture, type RamReading } from "./hardware";
import type { ModelFacts } from "./hfHeader";
import { CTX_SLIDER_MAX, ctxFromSlider, ctxLabel, getGpu, getSettings, setSettings, sliderFromCtx, ticksHtml, type GpuInfo, type Settings } from "./settings";
import { mapFiles } from "./fileMap";
import { gradeSvg } from "../../shared/breadMarks";

/** Hugging Face's dark theme puts a "dark" class on <html>. */
function pageIsDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

// ---------------------------------------------------------------- messaging

let port: chrome.runtime.Port | null = null;
let nextId = 1;
const waiting = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; progress?: (p: any) => void }>();

function getPort(): chrome.runtime.Port {
  if (port) return port;
  port = chrome.runtime.connect({ name: "lt" });
  port.onMessage.addListener((msg: { id: number; ok?: boolean; result?: unknown; error?: string; progress?: unknown }) => {
    const w = waiting.get(msg.id);
    if (!w) return;
    if (msg.progress) return w.progress?.(msg.progress);
    waiting.delete(msg.id);
    if (msg.ok) w.resolve(msg.result);
    else w.reject(new Error(msg.error ?? "failed"));
  });
  port.onDisconnect.addListener(() => {
    port = null;
    for (const w of waiting.values()) w.reject(new Error("The extension restarted; try again"));
    waiting.clear();
  });
  return port;
}

function ask<T>(msg: Record<string, unknown>, progress?: (p: any) => void): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    waiting.set(id, { resolve, reject, progress });
    getPort().postMessage({ ...msg, id });
  });
}

// ---------------------------------------------------------------- page model

interface Chip {
  label: string;
  button: HTMLElement;
  mark: HTMLSpanElement;
  file: string | null;
  state: "idle" | "pending" | "done" | "error";
  facts: ModelFacts | null;
  answer: FitAnswer | null;
  error: string | null;
  progress: string | null;
}

interface PageState {
  repo: string;
  sha: string | null;
  chips: Chip[];
  running: boolean;
  /** Marks are shown only once every chip has an answer. */
  ready: boolean;
  /** A check was started here (toaster clicked, or auto mode): chips that
   * appear later ("show more") are checked too. */
  armed: boolean;
  /** VRAM in use when the check started, for "keep free: in use now". */
  inUseAtStartMib: number | null;
  bar: ToasterBar | null;
}

interface ToasterBar {
  root: HTMLDivElement;
  button: HTMLButtonElement;
  text: HTMLSpanElement;
  /** Hugging Face's heading row we draw over. */
  head: HTMLElement | null;
  spinGen: number;
  /** Under the toaster: what the model is, and the context slider. */
  info: ModelInfo;
}

interface ModelInfo {
  root: HTMLDivElement;
  facts: HTMLDivElement;
  slider: HTMLInputElement;
  value: HTMLSpanElement;
  ticks: HTMLDivElement;
  /** The model's maximum context the slider is capped at (null until a header is read). */
  maxCtx: number | null;
}

let page: PageState | null = null;
let settings: Settings;
let gpu: GpuInfo | null = null;
let ram: RamReading | null = null;

function repoFromPath(): string | null {
  const m = /^\/([^/]+)\/([^/]+)\/?$/.exec(location.pathname);
  if (!m) return null;
  const reserved = new Set(["models", "datasets", "spaces", "docs", "blog", "settings", "organizations", "papers", "collections", "join", "login", "pricing", "enterprise", "learn", "tasks", "api"]);
  return reserved.has(m[1]) ? null : `${m[1]}/${m[2]}`;
}

function findChipButtons(): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>('div[class*="group/gguf"] button')].filter((b) => !b.closest("[inert]") && b.querySelector("span") && !isAuxiliaryChip(b));
}

/** Hugging Face tags chips of files that are not a model by themselves with a
 * badge span before the quant name (<span>MTP</span> BF16 855 MB): an MTP
 * draft head, not the BF16 model. They are left alone. */
function isAuxiliaryChip(btn: HTMLElement): boolean {
  return [...btn.querySelectorAll("span")].some((s) => /^(MTP|mmproj|imatrix)$/i.test((s.textContent ?? "").trim()));
}

function chipLabel(btn: HTMLElement): string {
  return (btn.querySelector("span:not(.lt-mark)")?.textContent ?? "").trim();
}

// ---------------------------------------------------------------- marks

function memNow(): MemoryPicture {
  return memoryPicture(gpu, ram, settings, ram?.helper ?? null);
}

// Hugging Face-style tooltip: a dark rounded bubble with an arrow, above the mark.
let tip: HTMLDivElement | null = null;
function showTip(m: HTMLElement) {
  if (!m.dataset.tip) return;
  tip ??= Object.assign(document.createElement("div"), { className: "lt-tip" });
  tip.textContent = m.dataset.tip;
  document.body.append(tip);
  const r = m.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  tip.style.left = `${Math.max(4, Math.min(window.innerWidth - t.width - 4, r.left + r.width / 2 - t.width / 2)) + window.scrollX}px`;
  tip.style.top = `${r.top - t.height - 8 + window.scrollY}px`;
  tip.style.setProperty("--arrow-x", `${r.left + r.width / 2 - (parseFloat(tip.style.left) - window.scrollX)}px`);
}
function hideTip() {
  tip?.remove();
}

function renderChip(c: Chip) {
  const m = c.mark;
  if (!page?.ready || c.state === "idle" || c.state === "pending") {
    m.remove();
    c.button.classList.remove("lt-marked");
    return;
  }
  c.button.classList.add("lt-marked");
  m.className = "lt-mark";
  if (c.state === "error") {
    m.textContent = "⚠️";
    m.dataset.tip = c.error ?? "Could not read this file";
  } else if (c.answer) {
    // Built from constants in shared/breadMarks.ts, never from page content.
    if (c.answer.grade != null) m.innerHTML = gradeSvg(c.answer.scale, c.answer.grade, { px: 14, dark: pageIsDark() });
    else m.textContent = c.answer.mark;
    m.dataset.tip = c.answer.headline;
  }
  m.onmouseenter = () => showTip(m);
  m.onmouseleave = hideTip;
  if (!m.isConnected) c.button.insertBefore(m, c.button.firstChild);
}

function reevaluate() {
  if (!page) return;
  const mem = memNow();
  const budget = deviceBudget(mem, settings, page.inUseAtStartMib);
  for (const c of page.chips) {
    if (c.facts) {
      c.answer = mem.vramUnknown ? null : evaluateFit(c.facts, mem, settings, budget);
      c.state = c.answer ? "done" : "error";
      c.error = c.answer ? null : "Enter your GPU's VRAM size in the LlamaToaster popup";
    }
    renderChip(c);
  }
  updateBar();
}

// ---------------------------------------------------------------- toaster

/** The GGUF box's heading row ("Hardware compatibility" + log in / your hardware). */
function hardwareHead(chipButton: HTMLElement): { box: HTMLElement; head: HTMLElement | null } | null {
  const grid = chipButton.closest('div[class*="group/gguf"]')?.parentElement;
  const box = grid?.parentElement;
  if (!grid || !box) return null;
  const head = [...box.children].find((el): el is HTMLElement => el !== grid && el instanceof HTMLElement && !el.contains(grid)) ?? null;
  return { box, head };
}

function installBar(p: PageState) {
  const first = p.chips[0]?.button;
  if (!first) return;
  const where = hardwareHead(first);
  if (!where) return;
  if (!p.bar) {
    // Hugging Face's own heading classes: same size, font and colour as "Hardware compatibility".
    const root = Object.assign(document.createElement("div"), { className: "lt-bar text-smd -ml-0.5 flex items-center gap-1" });
    const button = Object.assign(document.createElement("button"), { type: "button", className: "lt-toaster" });
    button.innerHTML = toasterSvg;
    button.setAttribute("aria-label", "LlamaToaster: check which quants fit this machine");
    const text = Object.assign(document.createElement("span"), { className: "lt-bar-text text-gray-500" });
    root.append(button, text); // toaster before (left of) the words
    // The whole line is the control: toaster and words both start the check.
    root.addEventListener("click", () => void calculate());
    p.bar = { root, button, text, head: null, spinGen: 0, info: buildInfo() };
  }
  const bar = p.bar;
  if (bar.head && bar.head !== where.head) bar.head.classList.remove("lt-hw-head");
  bar.head = where.head;
  if (where.head) {
    where.head.classList.add("lt-hw-head");
    if (bar.root.parentElement !== where.head) where.head.prepend(bar.root);
  } else if (bar.root.parentElement !== where.box) where.box.prepend(bar.root);
  // The model line and slider go right under the heading row.
  const anchor = where.head ?? bar.root;
  if (bar.info.root.previousElementSibling !== anchor) anchor.after(bar.info.root);
  updateBar();
}

function updateBar() {
  const p = page;
  const bar = p?.bar;
  if (!p || !bar) return;
  const total = p.chips.length;
  let text: string;
  let title: string;
  if (p.running) {
    const done = p.chips.filter((c) => c.facts || c.state === "error").length;
    text = done < total ? `Toasting… read ${done} of ${total} quants` : "Toasting…";
    title = "Reading each quant's header from Hugging Face (only the header, a few MB)";
  } else if (p.ready && p.armed) {
    const failed = p.chips.every((c) => c.state === "error");
    text = failed ? "Couldn't check these files. Click the toaster to try again." : "Checked for this machine. Click to check again.";
    title = "Hover a mark for the figures. Context and memory settings are under the LlamaToaster button in the toolbar.";
  } else {
    text = "Will it fit? Click the toaster to check.";
    title = "LlamaToaster reads each quant's header and marks what fits this machine";
  }
  bar.text.textContent = text;
  bar.root.title = title;
  bar.root.classList.toggle("lt-done", p.ready && p.armed && !p.running);
  renderInfo(p);
}

// ---------------------------------------------------------------- model line + context slider

function buildInfo(): ModelInfo {
  const root = Object.assign(document.createElement("div"), { className: "lt-info" });
  const facts = Object.assign(document.createElement("div"), { className: "lt-info-facts" });
  const row = Object.assign(document.createElement("div"), { className: "lt-info-ctx" });
  const label = Object.assign(document.createElement("span"), { className: "lt-info-k", textContent: "Context:" });
  const value = Object.assign(document.createElement("span"), { className: "lt-info-v" });
  row.append(label, value);
  const slider = Object.assign(document.createElement("input"), { type: "range", min: "0", step: "1", className: "lt-range" });
  slider.setAttribute("aria-label", "Context size for the fit check");
  const ticks = Object.assign(document.createElement("div"), { className: "lt-ticks" });
  root.append(facts, slider, ticks, row); // "Context  N tokens" reads under the scale
  const info: ModelInfo = { root, facts, slider, value, ticks, maxCtx: null };
  // Same setting as the popup's slider: moving either moves the other.
  slider.addEventListener("input", () => {
    const c = Math.min(ctxFromSlider(Number(slider.value)), info.maxCtx ?? Infinity);
    settings = { ...settings, ctx: c };
    paintSlider(info);
    queueCtxSave(c);
    // Every chip is re-simulated: once per frame is plenty while dragging.
    if (!reevalFrame) reevalFrame = requestAnimationFrame(() => ((reevalFrame = 0), reevaluate()));
  });
  slider.addEventListener("change", () => flushCtxSave());
  // Clicks on the slider must not reach Hugging Face's handlers.
  root.addEventListener("click", (e) => e.stopPropagation());
  return info;
}

let reevalFrame = 0;
let ctxSaveTimer: ReturnType<typeof setTimeout> | null = null;
let ctxPending: number | null = null;
function queueCtxSave(ctx: number) {
  ctxPending = ctx;
  if (ctxSaveTimer) clearTimeout(ctxSaveTimer);
  ctxSaveTimer = setTimeout(flushCtxSave, 150);
}
function flushCtxSave() {
  if (ctxSaveTimer) clearTimeout(ctxSaveTimer);
  ctxSaveTimer = null;
  if (ctxPending == null) return;
  const ctx = ctxPending;
  ctxPending = null;
  void setSettings({ ctx }).catch((err) => console.warn("LlamaToaster: could not save the context", err));
}

/** One quant's header describes the model: layers, experts and context are the same in every quant. */
function modelFacts(p: PageState): ModelFacts | null {
  return p.chips.find((c) => c.facts)?.facts ?? null;
}

function renderInfo(p: PageState) {
  const info = p.bar?.info;
  if (!info) return;
  info.root.classList.toggle("lt-dark", pageIsDark());
  const f = modelFacts(p);
  if (f) {
    // Headers cached by an older version have no expert count: the tensor list still tells MoE apart.
    const isMoe = !!f.expertCount || (f.breakdown.moe?.some((b) => b > 0) ?? false);
    const parts = [isMoe ? "MoE" : "Dense", `${f.nLayer} layers`];
    if (isMoe && f.expertCount) parts.push(`${f.expertCount} experts${f.expertUsedCount ? ` (${f.expertUsedCount} active)` : ""}`);
    if (f.trainedCtx) parts.push(`max context ${ctxLabel(f.trainedCtx)}`);
    info.facts.textContent = parts.join(" · ");
    info.facts.title = f.trainedCtx ? "Max context is the header's context_length: the longest context the model accepts, not necessarily what it was trained on." : "";
    info.facts.classList.remove("lt-empty");
  } else {
    // Keep the row (CSS min-height) so the block does not jump when the header arrives.
    info.facts.textContent = p.running ? "Reading the model…" : " ";
    info.facts.title = "";
    info.facts.classList.add("lt-empty");
  }
  const maxCtx = f?.trainedCtx ?? null;
  // Grey and locked until the model's maximum context is known; CSS fades it to orange.
  info.slider.disabled = maxCtx == null;
  if (maxCtx !== info.maxCtx || !info.ticks.hasChildNodes()) {
    info.maxCtx = maxCtx;
    const maxPos = maxCtx ? Math.max(1, sliderFromCtx(maxCtx)) : CTX_SLIDER_MAX;
    info.slider.max = String(maxPos);
    info.ticks.innerHTML = ticksHtml(maxPos); // built from constants only
  }
  paintSlider(info);
}

function paintSlider(info: ModelInfo) {
  const ctx = info.maxCtx ? Math.min(settings.ctx, info.maxCtx) : settings.ctx;
  if (document.activeElement !== info.slider) info.slider.value = String(sliderFromCtx(ctx));
  info.slider.style.setProperty("--frac", String(Number(info.slider.value) / Number(info.slider.max)));
  info.value.textContent = `${ctxLabel(ctx)} tokens${info.maxCtx && settings.ctx > info.maxCtx ? " (model maximum)" : ""}`;
}

/** Turn the dial while reading; when done it finishes its turn and stops where it started. */
function setSpinning(on: boolean) {
  const bar = page?.bar;
  if (!bar) return;
  const gen = ++bar.spinGen;
  const btn = bar.button;
  if (on) {
    btn.classList.add("lt-spin");
    btn.setAttribute("aria-busy", "true");
    return;
  }
  btn.removeAttribute("aria-busy");
  if (!btn.classList.contains("lt-spin")) return;
  const stop = () => {
    if (bar.spinGen === gen) btn.classList.remove("lt-spin");
  };
  btn.querySelector(".lt-knob")?.addEventListener("animationiteration", stop, { once: true });
  setTimeout(stop, 1600); // reduced motion or a hidden tab: no iteration event
}

// ---------------------------------------------------------------- calculation

async function refreshRam() {
  try {
    ram = await ask<RamReading>({ type: "ram" });
  } catch {
    // keep the last reading
  }
}

async function ensureGpu() {
  gpu = await getGpu();
  if (!gpu) {
    gpu = await detectGpu();
    await chrome.storage.local.set({ gpu });
  }
}

async function mapChips(p: PageState): Promise<boolean> {
  const listing = await ask<{ sha: string | null; files: { path: string }[] }>({ type: "list", repo: p.repo });
  p.sha = listing.sha;
  const map = mapFiles(p.chips.map((c) => c.label), listing.files.map((f) => f.path));
  for (const c of p.chips) c.file = map.get(c.label) ?? null;
  return true;
}

async function runChip(p: PageState, c: Chip, force: boolean, cacheOnly = false) {
  if (!c.file) {
    c.state = "error";
    c.error = "No matching .gguf file found in this repo";
    return;
  }
  try {
    const res = await ask<{ facts: ModelFacts | null }>({ type: "facts", repo: p.repo, file: c.file, sha: p.sha, force, cacheOnly }, (prog: { shard: number; shards: number; fetchedBytes: number }) => {
      c.progress = `${(prog.fetchedBytes / 1048576).toFixed(0)} MB${prog.shards > 1 ? ` · part ${prog.shard}/${prog.shards}` : ""}`;
    });
    if (!res.facts) return; // cache miss in cache-only mode
    c.facts = res.facts;
    c.error = null;
  } catch (err) {
    c.state = "error";
    c.error = err instanceof Error ? err.message : String(err);
  }
}

async function calculate() {
  const p = page;
  if (!p || p.running) return;
  p.armed = true;
  p.running = true;
  const queue = p.chips.filter((c) => !c.facts);
  // Checking again with every header known keeps the marks up meanwhile.
  if (queue.length) p.ready = false;
  for (const c of queue) c.state = "pending";
  setSpinning(true);
  updateBar();
  try {
    await Promise.all([ensureGpu(), refreshRam()]);
    p.inUseAtStartMib = deviceInUseMib(memNow());
    if (queue.length) await mapChips(p);
    // Two headers at a time: quick, and gentle on Hugging Face. A file's
    // header never changes for a revision, so it is read only once.
    const workers = Array.from({ length: 2 }, async () => {
      for (let c = queue.shift(); c; c = queue.shift()) {
        if (page !== p) return;
        await runChip(p, c, false);
        updateBar();
      }
    });
    await Promise.all(workers);
  } catch (err) {
    for (const c of p.chips) {
      if (c.facts) continue;
      c.state = "error";
      c.error = `Could not list this repo's files: ${err instanceof Error ? err.message : String(err)}`;
    }
  } finally {
    p.running = false;
    if (page === p) {
      setSpinning(false);
      p.ready = true;
      reevaluate();
      // Chips that appeared while we were reading ("show more", re-renders).
      if (p.chips.some((c) => c.state === "idle")) void calculate();
    }
  }
}

// ---------------------------------------------------------------- lifecycle

function teardown() {
  if (!page) return;
  hideTip();
  for (const c of page.chips) {
    c.mark.remove();
    c.button.classList.remove("lt-marked");
  }
  page.bar?.root.remove();
  page.bar?.info.root.remove();
  page.bar?.head?.classList.remove("lt-hw-head");
  page = null;
}

/** A check that was running when the page was frozen continues after restore. */
let resumeArmed = false;

// Back/forward can restore this page from the browser's back-forward cache with
// its script state exactly as it was frozen -- "Toasting… read 6 of 24" -- but
// the connection to the service worker is gone, so the requests in flight are
// never answered and the check would hang there forever. Start over instead:
// the headers already read are cached in the service worker, so only the
// missing ones are downloaded.
window.addEventListener("pageshow", (e) => {
  if (!e.persisted || !settings) return;
  try {
    port?.disconnect();
  } catch {
    // already closed
  }
  port = null;
  const stale = [...waiting.values()];
  waiting.clear();
  for (const w of stale) w.reject(new Error("The page was restored; checking again"));
  if (page) {
    resumeArmed = page.armed;
    teardown();
  }
  scan();
});

function scan() {
  const repo = repoFromPath();
  if (page && page.repo !== repo) teardown();
  if (!repo) return;
  const buttons = findChipButtons();
  if (buttons.length === 0) return;

  if (!page) {
    page = { repo, sha: null, chips: [], running: false, ready: false, armed: resumeArmed, inUseAtStartMib: null, bar: null };
    resumeArmed = false;
  }
  let added = false;
  // New chips (HF re-renders, "show more"): mark them too.
  for (const b of buttons) {
    if (page.chips.some((c) => c.button === b)) continue;
    const chip: Chip = { label: chipLabel(b), button: b, mark: document.createElement("span"), file: null, state: "idle", facts: null, answer: null, error: null, progress: null };
    page.chips.push(chip);
    added = true;
  }
  page.chips = page.chips.filter((c) => c.button.isConnected);
  for (const c of page.chips) if (c.state === "done" || c.state === "error") renderChip(c);
  // Hugging Face re-renders can drop the toaster: put it back.
  if (!page.bar?.root.isConnected || !page.bar.info.root.isConnected || (page.bar.head && !page.bar.head.classList.contains("lt-hw-head"))) installBar(page);
  if (added && !page.running && (page.armed || settings.mode === "auto")) void calculate();
  else if (added) updateBar();
}

async function init() {
  settings = await getSettings();
  gpu = await getGpu();
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.settings) settings = { ...settings, ...(changes.settings.newValue as Settings) };
    if (changes.gpu) gpu = changes.gpu.newValue as GpuInfo;
    if (changes.settings || changes.gpu) reevaluate();
    if (changes.settings && page) renderInfo(page);
    // Switched to auto with this page open: check it now.
    if (changes.settings && settings.mode === "auto" && page && !page.armed && !page.running) void calculate();
  });
  let scheduled = false;
  new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      scan();
    });
  }).observe(document.body, { childList: true, subtree: true });
  scan();
  // Available RAM moves; keep the marks honest while the tab is in view.
  setInterval(() => {
    if (page && document.visibilityState === "visible" && page.chips.some((c) => c.facts)) void refreshRam().then(reevaluate);
  }, 10_000);
}

void init();
