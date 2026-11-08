import { measureBandwidth, type MeasurePhase } from "./bandwidth";
import { deviceInUseMib } from "./fit";
import { primaryGpu, ramLabel, readHelper, type HelperApp, type HelperGpu, type HelperRam, type HelperResult } from "./helper";
import { detectGpu, memoryPicture, readCpu, readRam, type MemoryPicture } from "./hardware";
import { CTX_SLIDER_MAX, ctxFromSlider, ctxLabel, getBandwidth, getGpu, getSettings, setSettings, sliderFromCtx, ticksHtml, type BandwidthResult, type GpuInfo, type Settings } from "./settings";
import { gradeSvg } from "../../shared/breadMarks";
import { fitGradeLegendLabel, fitGrades } from "../../shared/fitGrade";
import { memoryScale } from "./fit";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const gb = (mib: number, digits = 1) => `${(mib / 1024).toFixed(digits)} GB`;
/** "8 GB", "7.5 GB": whole sizes without a trailing ".0". */
const size = (mib: number) => `${+(mib / 1024).toFixed(1)} GB`;
const WORKERS_URL = "https://llamatoaster.com/workers";

/** This page is the side panel (panel.html), not the toolbar popup. */
const inPanel = document.body.classList.contains("panel");

let settings: Settings;
let gpu: GpuInfo | null;
let helperRes: HelperResult | null = null;
let helper: HelperGpu | null = null;
/** RAM type and speed: read once from the helper and kept, it doesn't change. */
let ramInfo: HelperRam | null = null;
/** The user clicked "edit" next to the GPU size. */
let editingVram = false;
let windowId: number | undefined;

/** Apps shown on a bar: RAM from 1 GB, VRAM from 512 MB (smaller ones stay in the grey "in use"). */
const APP_MIN_MIB = { ram: 1024, vram: 512 } as const;
/** Barely-there tints so neighbouring apps tell apart; cycled in order of size. */
const APP_TINTS = 4;

/** Light grey = free (the bar itself), yellow from the left = LlamaToaster's
 * llama.cpp, dark grey at the right = everything else in use. The biggest
 * apps (from the worker's helper) are their own faintly tinted segments at
 * the right end, named on hover. */
function setBar(id: string, llama: number, other: number, total: number, apps: HelperApp[] = []) {
  const pct = (v: number) => (total > 0 ? Math.max(0, Math.min(100, (v / total) * 100)) : 0);
  const l = pct(llama);
  const o = Math.min(pct(other), 100 - l);
  const [li, oi] = $(id).children as unknown as HTMLElement[];
  li.style.width = `${l}%`;
  oi.style.width = `${o}%`;
  // Apps can't add up to more than "in use" (the readings are a moment apart).
  const sum = apps.reduce((a, x) => a + x.mib, 0);
  const scale = sum > other && sum > 0 ? other / sum : 1;
  oi.replaceChildren(
    ...apps.map((a, i) => {
      const seg = document.createElement("b");
      seg.className = `app t${i % APP_TINTS}`;
      seg.style.flexBasis = `${other > 0 ? ((a.mib * scale) / other) * 100 : 0}%`;
      seg.title = `${a.name} · ${gb(a.mib)}`;
      return seg;
    })
  );
  oi.title = `in use ${gb(other)}${apps.length ? ` · of which ${apps.map((a) => `${a.name} ${gb(a.mib)}`).join(", ")}` : ""}`;
  $(id).title = `${llama > 0 ? `LlamaToaster ${gb(llama)} · ` : ""}in use ${gb(other)} · free ${gb(Math.max(0, total - llama - other))}`;
}

/** The helper's app list for a bar: above the bar's threshold, LlamaToaster's own llama.cpp left out (it has the yellow segment). */
function barApps(list: HelperApp[] | null | undefined, min: number): HelperApp[] {
  return (list ?? []).filter((a) => a.mib >= min && !/^llama-/i.test(a.name));
}

/** The VRAM size shown next to the GPU name: measured, typed, or from the table of known cards. */
function vramSize(mem: MemoryPicture): number | null {
  if (mem.kind !== "vram" || mem.vramUnknown) return null;
  return mem.deviceTotalMib;
}

/** The bread marks this machine can show, best first (the popup is dark). */
function renderLegend(mem: MemoryPicture) {
  const scale = memoryScale(mem);
  // Built from constants in shared/breadMarks.ts and shared/fitGrade.ts.
  $("fitLegend").innerHTML = fitGrades(scale)
    .map((g) => `<span class="mark">${gradeSvg(scale, g, { px: 14, dark: true })}</span><span>${fitGradeLegendLabel(scale, g, "yours")}</span>`)
    .join("");
}

/** Fills a value cell: plain strings are dimmed hardware names, [text] is a bold memory amount. */
function parts(id: string, ...bits: (string | [string])[]) {
  $(id).replaceChildren(
    ...bits.map((b) => {
      const el = document.createElement(Array.isArray(b) ? "b" : "span");
      el.className = Array.isArray(b) ? "amt" : "hw";
      el.textContent = Array.isArray(b) ? b[0] : b;
      return el;
    }),
  );
}

async function renderMemory() {
  const ram = await readRam();
  const mem = memoryPicture(gpu, ram, settings, helper);
  const llamaRam = ram ? Math.min(helperRes?.ok ? helperRes.reading.llamaRamMib : 0, ram.totalMib - ram.availableMib) : 0;

  const type = ramLabel(ramInfo);
  if (ram) parts("ram", ...(type ? [`${type} · `] : []), [gb(ram.availableMib)], " free of ", [size(ram.totalMib)]);
  else parts("ram", "unknown");
  $("ram").title = ramInfo ? `${ramInfo.modules} module${ramInfo.modules === 1 ? "" : "s"}${ramInfo.ratedMts ? `, rated ${ramInfo.ratedMts} MT/s` : ""}` : "RAM type and speed come from a LlamaToaster worker on this machine";
  if (ram) setBar("ramBar", llamaRam, ram.totalMib - ram.availableMib - llamaRam, ram.totalMib, barApps(helperRes?.ok ? helperRes.reading.topRam : null, APP_MIN_MIB.ram));

  // GPU: name · VRAM size (or the memory kind when there is no own VRAM).
  const name = helper?.name ?? (gpu?.noGpu ? "none found" : gpu?.name ?? "unknown");
  const vram = vramSize(mem);
  if (mem.kind === "vram" && vram != null) parts("gpu", `${name} · ${size(vram)}`);
  else parts("gpu", mem.kind === "cpu" ? name : mem.kind === "unified" ? `${name} · shared RAM` : `${name} · VRAM size unknown`);
  $("editVram").style.display = mem.kind === "vram" ? "" : "none";

  // The size input shows only when it's needed: unknown card, a typed size, or "edit".
  const totalInput = $<HTMLInputElement>("vramTotal");
  $("vramTotalRow").style.display = mem.kind === "vram" && (mem.vramUnknown || editingVram || settings.vramOverrideMib != null) ? "" : "none";
  if (document.activeElement !== totalInput) totalInput.value = vram != null ? String(+(vram / 1024).toFixed(1)) : "";
  const src = $("vramSrc");
  src.textContent = settings.vramOverrideMib != null ? "entered · clear to reset" : mem.measured ? "measured" : gpu?.tableVramMib != null ? "known card" : "enter it";
  src.className = `pill${mem.vramUnknown ? " warn" : mem.measured && settings.vramOverrideMib == null ? " ok" : ""}`;

  renderLegend(mem);

  // "Auto" says what it found.
  const auto = memoryPicture(gpu, ram, { ...settings, cpuOnly: false, unifiedOverride: null }, helper).kind;
  $("kindAuto").textContent = `Auto · ${auto === "vram" ? "Dedicated VRAM" : auto === "unified" ? "Unified (shared RAM)" : "CPU only"}`;
  $<HTMLSelectElement>("kind").value = settings.cpuOnly ? "cpu" : settings.unifiedOverride == null ? "auto" : settings.unifiedOverride ? "unified" : "vram";

  // Without a worker the browser can't see what other apps hold: the user says.
  $("reserveRow").style.display = mem.kind === "vram" && !mem.measured ? "" : "none";
  const resInput = $<HTMLInputElement>("vramReserved");
  if (document.activeElement !== resInput) resInput.value = String(+(settings.vramReservedMib / 1024).toFixed(2));

  $("availLabel").textContent = mem.kind === "unified" ? "GPU can use" : mem.kind === "cpu" ? "Model memory" : "VRAM available";
  const unmeasured = mem.kind === "vram" && !mem.measured;
  if (mem.kind === "cpu") parts("vramAvail", "RAM only");
  else if (unmeasured) parts("vramAvail", "n/a free of ", [vram != null ? size(vram) : "?"]);
  else parts("vramAvail", [gb(mem.deviceFreeMib)], " free of ", [size(mem.deviceTotalMib)]);
  // VRAM: free vs in use. LlamaToaster's own llama.cpp counts as in use here.
  setBar("vramBar", 0, deviceInUseMib(mem), mem.deviceTotalMib, mem.measured ? barApps(helper?.apps, APP_MIN_MIB.vram) : []);
  $("vramBar").style.display = mem.kind === "cpu" || unmeasured ? "none" : "";
  $("legendLlama").style.display = llamaRam > 0 ? "" : "none";

  const note = $("memNote");
  note.className = "note";
  const workerLink = `<a href="${WORKERS_URL}" target="_blank" rel="noopener">llamatoaster.com/workers</a>`;
  if (mem.kind === "vram" && mem.vramUnknown) {
    note.className = "note warn";
    note.innerHTML = `This GPU is not in the built-in list: enter its VRAM size, or install a LlamaToaster worker on this machine to measure it (${workerLink}).`;
  } else if (mem.measured) {
    note.textContent = `Measured through the LlamaToaster worker (${helper?.vendor}): free = total − what every app on this GPU uses right now.`;
  } else if (mem.kind === "vram") {
    const err = helperRes && !helperRes.ok && !helperRes.missing ? `Worker helper error: ${escapeHtml(helperRes.error)}. ` : "";
    note.innerHTML = `${err}A browser can't read free VRAM, so the marks assume other apps use ${gb(settings.vramReservedMib)}. To measure it, install a LlamaToaster worker on this machine: ${workerLink}`;
  } else if (mem.kind === "unified") {
    note.textContent = /apple/i.test(`${gpu?.vendor} ${gpu?.name}`) ? "Apple silicon: the GPU can use about 3/4 of RAM." : "The GPU shares system RAM.";
  } else note.textContent = "";

  renderMargin(mem);
}

/** "Keep free on the GPU": the "In use now" choice shows the live figure it would capture. */
function renderMargin(mem: MemoryPicture) {
  const opt = $<HTMLOptionElement>("marginCurrent");
  if (mem.kind === "cpu") {
    opt.textContent = "In use now";
    opt.disabled = true;
  } else {
    opt.disabled = false;
    opt.textContent = `In use now (${mem.measured || mem.kind === "unified" ? "" : "est. "}${gb(deviceInUseMib(mem))})`;
  }
  // Shown on hover, not as a note under the row: a note appearing and vanishing made everything below jump.
  $("keepRow").title =
    settings.marginMode === "current"
      ? "Captured when a check starts on a page, so the marks don't move as other apps (or llama.cpp itself) take memory afterwards."
      : "";
}

function renderFit() {
  const lever = $("lever");
  lever.dataset.mode = settings.mode;
  $("leverSlot").setAttribute("aria-checked", String(settings.mode === "auto"));

  const slider = $<HTMLInputElement>("ctx");
  slider.max = String(CTX_SLIDER_MAX);
  if (document.activeElement !== slider) slider.value = String(sliderFromCtx(settings.ctx));
  setFill(slider);
  $("ctxVal").textContent = `${ctxLabel(settings.ctx)} tokens`;

  const sel = $<HTMLSelectElement>("margin");
  sel.value = settings.marginMode === "preset" ? String(settings.marginMib) : settings.marginMode;
  if (sel.selectedIndex < 0) sel.value = "1024";
  const custom = settings.marginMode === "custom";
  const input = $<HTMLInputElement>("customMargin");
  input.style.display = custom ? "" : "none";
  $("customUnit").style.display = custom ? "" : "none";
  sel.classList.toggle("chev-only", custom);
  if (document.activeElement !== input) input.value = String(+(settings.customMarginMib / 1024).toFixed(2));
}

/** How far along the slot the heating element glows (to the handle's centre). */
function setFill(slider: HTMLInputElement) {
  slider.style.setProperty("--frac", String(Number(slider.value) / Number(slider.max || CTX_SLIDER_MAX)));
}

/** Memory speed folds away; when folded its header shows the best figures. */
function setSpeedOpen(open: boolean) {
  $("speedHead").setAttribute("aria-expanded", String(open));
  $("speedBody").hidden = !open;
}

function speedSummary(b: BandwidthResult | null) {
  const sum = $("speedSum");
  if (!b) {
    sum.textContent = "not measured";
    return;
  }
  const ram = Math.max(...b.ram.points.map((p) => p.readGBs));
  sum.textContent = `RAM ${ram.toFixed(0)} GB/s${b.vram ? ` · VRAM ${b.vram.readGBs.toFixed(0)} GB/s` : ""}`;
}

function renderDock() {
  const b = $("dock");
  b.textContent = inPanel ? "Hide sidebar" : "Side panel";
  b.title = inPanel ? "Close the side panel; the toolbar button opens LlamaToaster as a popup again" : "Open in Chrome's side panel: it stays open next to the page instead of covering it";
}

function renderSpeed(b: BandwidthResult | null) {
  speedSummary(b);
  const out = $("speedOut");
  if (!b) {
    out.innerHTML = `<div class="note">How fast this machine moves bytes: the ceiling for token generation, which reads every active weight once per token.</div>`;
    return;
  }
  const pts = b.ram.points;
  const best = (k: "readGBs" | "writeGBs" | "copyGBs") => pts.reduce((bi, p, i) => (p[k] > pts[bi][k] ? i : bi), 0);
  const [br, bw, bc] = [best("readGBs"), best("writeGBs"), best("copyGBs")];
  const f = (n: number) => `${n.toFixed(1)} GB/s`;
  let html = `<table><thead><tr><th>RAM threads</th><th>Read</th><th>Write</th><th>Copy</th></tr></thead><tbody>`;
  pts.forEach((p, i) => {
    html += `<tr><td>${p.threads}</td><td class="${i === br ? "best" : ""}">${f(p.readGBs)}</td><td class="${i === bw ? "best" : ""}">${f(p.writeGBs)}</td><td class="${i === bc ? "best" : ""}">${f(p.copyGBs)}</td></tr>`;
  });
  html += `</tbody></table>`;
  if (b.vram) {
    html += `<table><thead><tr><th>VRAM</th><th>Read</th><th>Write</th><th>Copy</th></tr></thead><tbody><tr><td>GPU</td><td class="best">${f(b.vram.readGBs)}</td><td>${f(b.vram.writeGBs)}</td><td>${f(b.vram.copyGBs)}</td></tr></tbody></table>`;
  } else if (b.vramUnavailableReason) {
    html += `<div class="note">VRAM: ${escapeHtml(b.vramUnavailableReason)}</div>`;
  }
  html += `<div class="note">Measured ${new Date(b.measuredAt).toLocaleString()} in the browser: browser overhead puts these a little under native tools.</div>`;
  if (b.ram.readKernel !== "wasm-simd") {
    html += `<div class="note">RAM Read here was a JavaScript loop, which is limited by the loop rather than by memory and reads far too low. Run the measurement again with the current version for a real figure.</div>`;
  }
  out.innerHTML = html;
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

// Local state updates at once; storage gets the latest value, coalesced so a
// slider drag is a handful of writes rather than one per tick.
let pending: Partial<Settings> = {};
let flushTimer: ReturnType<typeof setTimeout> | null = null;
function flush() {
  flushTimer = null;
  const patch = pending;
  pending = {};
  return setSettings(patch).catch((err) => console.warn("LlamaToaster: could not save settings", err));
}
async function save(patch: Partial<Settings>, now = false) {
  settings = { ...settings, ...patch };
  pending = { ...pending, ...patch };
  if (flushTimer) clearTimeout(flushTimer);
  if (now) await flush();
  else flushTimer = setTimeout(flush, 120);
  renderFit();
  await renderMemory();
}

const gbInput = (el: HTMLInputElement) => {
  const v = parseFloat(el.value);
  return Number.isFinite(v) && v >= 0 ? Math.round(v * 1024) : null;
};

async function main() {
  renderDock();
  void chrome.windows?.getCurrent().then((w) => (windowId = w.id));
  settings = await getSettings();
  gpu = await getGpu();
  ramInfo = ((await chrome.storage.local.get("ramInfo")).ramInfo as HelperRam | undefined) ?? null;
  // Detect again each time the popup opens: cheap, and catches a GPU switch.
  const fresh = await detectGpu();
  gpu = fresh;
  await chrome.storage.local.set({ gpu: fresh });

  const cpu = await readCpu();
  if (cpu) parts("cpu", `${cpu.model} · ${cpu.threads} threads`);
  else parts("cpu", "unknown");

  $("ticks").innerHTML = ticksHtml();

  // Manual / Auto: a wall toggle switch. The switch flips it; each word picks its side.
  $("leverSlot").addEventListener("click", () => void save({ mode: settings.mode === "auto" ? "manual" : "auto" }, true));
  for (const el of document.querySelectorAll<HTMLButtonElement>(".toggle-label")) {
    el.addEventListener("click", () => {
      if (el.dataset.mode !== settings.mode) void save({ mode: el.dataset.mode as Settings["mode"] }, true);
    });
  }

  $<HTMLInputElement>("ctx").addEventListener("input", (e) => {
    const c = ctxFromSlider(Number((e.target as HTMLInputElement).value));
    setFill(e.target as HTMLInputElement);
    $("ctxVal").textContent = `${ctxLabel(c)} tokens`;
    void save({ ctx: c });
  });
  // Releasing the slider (or the popup closing) saves straight away.
  $<HTMLInputElement>("ctx").addEventListener("change", () => void save({}, true));
  window.addEventListener("pagehide", () => {
    if (flushTimer) void flush();
  });
  $<HTMLSelectElement>("margin").addEventListener("change", (e) => {
    const v = (e.target as HTMLSelectElement).value;
    if (v === "current" || v === "custom") void save({ marginMode: v }, true).then(() => v === "custom" && $<HTMLInputElement>("customMargin").focus());
    else void save({ marginMode: "preset", marginMib: Number(v) }, true);
  });
  $<HTMLInputElement>("customMargin").addEventListener("change", (e) => {
    const v = gbInput(e.target as HTMLInputElement);
    void save({ customMarginMib: v ?? settings.customMarginMib }, true);
  });
  $<HTMLSelectElement>("kind").addEventListener("change", (e) => {
    const v = (e.target as HTMLSelectElement).value;
    void save({ cpuOnly: v === "cpu", unifiedOverride: v === "auto" || v === "cpu" ? null : v === "unified" });
  });
  $("editVram").addEventListener("click", () => {
    editingVram = !editingVram;
    void renderMemory().then(() => editingVram && $<HTMLInputElement>("vramTotal").focus());
  });
  $<HTMLInputElement>("vramTotal").addEventListener("change", (e) => {
    const v = parseFloat((e.target as HTMLInputElement).value);
    void save({ vramOverrideMib: Number.isFinite(v) && v > 0 ? Math.round(v * 1024) : null });
  });
  $<HTMLInputElement>("vramReserved").addEventListener("change", (e) => {
    void save({ vramReservedMib: gbInput(e.target as HTMLInputElement) ?? 512 });
  });

  // Side panel: open it from the popup's click (it needs the user gesture,
  // so no await before it); the background makes the toolbar button follow.
  $("dock").addEventListener("click", () => {
    if (inPanel) {
      void save({ sidePanel: false }, true).then(() => window.close());
      return;
    }
    const opening = windowId != null ? chrome.sidePanel.open({ windowId }) : Promise.reject(new Error("no window"));
    void save({ sidePanel: true }, true);
    opening.then(() => window.close()).catch((err) => console.warn("LlamaToaster: could not open the side panel", err));
  });

  // Memory speed is folded by default, the marks legend open; both are remembered.
  const ui = ((await chrome.storage.local.get("ui")).ui ?? {}) as { speedOpen?: boolean; legendOpen?: boolean };
  const saveUi = (patch: typeof ui) => {
    Object.assign(ui, patch);
    void chrome.storage.local.set({ ui: { ...ui } });
  };
  setSpeedOpen(!!ui.speedOpen);
  $("speedHead").addEventListener("click", () => {
    const open = $("speedHead").getAttribute("aria-expanded") !== "true";
    setSpeedOpen(open);
    saveUi({ speedOpen: open });
  });
  const setLegendOpen = (open: boolean) => {
    $("legendHead").setAttribute("aria-expanded", String(open));
    $("fitLegend").hidden = !open;
  };
  setLegendOpen(ui.legendOpen !== false);
  $("legendHead").addEventListener("click", () => {
    const open = $("legendHead").getAttribute("aria-expanded") !== "true";
    setLegendOpen(open);
    saveUi({ legendOpen: open });
  });

  // The lever goes down and stays down while measuring, then pops back up.
  let measuring = false;
  $("measure").addEventListener("click", async () => {
    if (measuring) return;
    measuring = true;
    const btn = $<HTMLButtonElement>("measure");
    btn.classList.add("down");
    btn.setAttribute("aria-busy", "true");
    const label = $("measureLabel");
    label.textContent = "Measuring…";
    const status = $("speedStatus");
    status.textContent = "";
    $("speedSum").textContent = "measuring…";
    // The last figures stay in their table, faded, until the new ones replace them.
    $("speedOut").classList.add("measuring");
    // RAM and VRAM each count 0..100%. A counter eases between trial reports:
    // after each one it keeps creeping forward at the pace the last trial took,
    // never past the next report.
    const meter = () => ({ reported: 0, trialShare: 0, lastAt: performance.now(), pace: 0, shown: 0 });
    const meters: Record<MeasurePhase, ReturnType<typeof meter>> = { ram: meter(), vram: meter() };
    let phase: MeasurePhase = "ram";
    const pct = (m: ReturnType<typeof meter>, live: boolean) => {
      const creep = live && m.pace > 0 ? Math.min(0.95, (performance.now() - m.lastAt) / m.pace) * m.trialShare : 0;
      m.shown = Math.max(m.shown, Math.min(1, m.reported + creep));
      return `${Math.floor(m.shown * 100)}%`;
    };
    // A timer, not requestAnimationFrame: animation frames stop in a hidden view.
    const paint = () => {
      label.textContent = `RAM ${pct(meters.ram, phase === "ram")} · VRAM ${pct(meters.vram, phase === "vram")}`;
    };
    paint();
    const frame = setInterval(paint, 100);
    try {
      const result = await measureBandwidth((p, d, done, trial) => {
        const now = performance.now();
        if (p !== phase) {
          meters[phase].reported = 1; // the phase before is complete
          phase = p;
        }
        const m = meters[p];
        if (done > m.reported) {
          m.pace = (now - m.lastAt) / ((done - m.reported) / trial);
          m.reported = done;
        }
        m.lastAt = now;
        m.trialShare = trial;
        status.textContent = d;
      });
      clearInterval(frame);
      await chrome.storage.local.set({ bandwidth: result });
      renderSpeed(result);
      status.textContent = "";
      label.textContent = "Done. Push to measure again";
    } catch (err) {
      status.textContent = `failed: ${err instanceof Error ? err.message : String(err)}`;
      label.textContent = "Push the lever to try again";
      renderSpeed(await getBandwidth());
    } finally {
      clearInterval(frame);
      btn.classList.remove("down");
      $("speedOut").classList.remove("measuring");
      btn.removeAttribute("aria-busy");
      measuring = false;
    }
  });

  chrome.storage.onChanged.addListener((changes) => {
    // Another view (popup vs panel) or a page changed a setting.
    if (changes.settings?.newValue && !flushTimer) {
      settings = { ...settings, ...(changes.settings.newValue as Settings) };
      renderFit();
      void renderMemory();
    }
  });

  renderFit();
  await renderMemory();
  renderSpeed(await getBandwidth());
  setInterval(() => void renderMemory(), 2000);
  let askRam = true;
  const pollHelper = async () => {
    helperRes = await readHelper({ ram: askRam, apps: true });
    helper = helperRes.ok ? primaryGpu(helperRes.reading) : null;
    if (helperRes.ok) {
      // Asked once per open: it doesn't change, and on Linux it's a dmidecode run.
      askRam = false;
      if (helperRes.reading.ram) {
        ramInfo = helperRes.reading.ram;
        await chrome.storage.local.set({ ramInfo });
      }
    }
    await renderMemory();
  };
  void pollHelper();
  setInterval(() => void pollHelper(), 5000);
}

void main();
