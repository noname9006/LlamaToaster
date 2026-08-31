#!/usr/bin/env node
/**
 * LlamaToaster — pitch deck builder v2.
 * Renders docs/PITCH_DECKv2.md to a dark minimal 1280x720 PDF via headless Edge/Chrome.
 * Zero npm dependencies.
 *
 * Usage:
 *   node scripts/pitch-to-pdf-v2.mjs <in.md> <out.pdf>
 *   node scripts/pitch-to-pdf-v2.mjs --html-only <in.md> <out.html>
 *   node scripts/pitch-to-pdf-v2.mjs --shots <in.md> <outDir> [slide numbers...]
 *
 * ── Design system (2 fonts, minimal hues) ─────────────────────────
 * Fonts:  Bahnschrift   (display / big numbers, weights 300 + 600)
 *         Segoe UI      (body text)
 * Hues:   #0C0E14  ink-field  (bg; panels/borders are tints of it)
 *         #F4F6FB  ink-text   (muted = ink at reduced opacity)
 *         #7C5CFF  single accent (violet)  |  #4ADE80 reserved for
 *         "verified" semantics only (slides 6-7)
 * Rules:  <=5 blocks per slide, big numbers as hero element,
 *         no photos, no 3D, no decorative gradients, no tiny text.
 *
 * ── Per-slide blueprint ───────────────────────────────────────────
 * 1 Cover        Layout: centered column; waist-height dot grid + mini toaster glyph.
 *                Hero: "4 000 000 · 40 · 1" big numbers.
 *                Visual: sparse dot field + toaster line icon.
 *                   tag 10% · hero block 55% · brand 20% · footnote 10%
 * 2 Problem      Layout: header full width; 58% left (3 pains) / 42% right (illustration).
 *                Hero: violet "hot cell" inside a huge catalog grid.
 *                Visual: 13x9 cell grid SVG + "the right config" marker + 2 chips.
 *                   header 22% · pains 42% · grid+chips 36%
 * 3 Missing Half Layout: 50/50. Left text, right Quality chart vs Performance "0".
 *                Hero: giant "0" — answers for YOUR hardware.
 *                   header 16% · text 38% · panels 46%
 * 4 Why Now      Layout: 50/50. Left 3 "x" points, right 27B-hero + curve + chips.
 *                Hero: "27B =~ flagship quality" on an ordinary PC.
 *                   header 14% · left 43% · right 43%
 * 5 Fix          Layout: 44/56. Left text, right INTENT -> GRID+SWEEP -> 4 profile cards.
 *                Hero: 4 profile cards (Max Speed · Balanced · Max Context · Low Memory).
 *                   header 18% · text 36% · pipeline 46%
 * 6 Trust the Card Layout: 55/45. Left 4 checks, right mock card with rejected row.
 *                Hero: "1e6 tok/s -> REJECTED" row + rotated VERIFIED stamp.
 *                   header 16% · bullets 44% · mock card 40%
 * 7 Competitors  Layout: full-width 2x2 matrix (~70%) + bottom strip.
 *                Hero: 2x2 matrix with glowing "open spot".
 *                   header 18% · matrix 52% · strip 30%
 * 8 Ask          Layout: centered column. Giant "1" + URL pill + 3 direction cards.
 *                Hero: "1" command + llamatoaster.com/benchmark.
 *                   header 16% · hero 34% · directions 38%
 * ──────────────────────────────────────────────────────────────────
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";

const ACCENT = "#7C5CFF";
const OK = "#4ADE80";
const INK = "#F4F6FB";
const MUTED = "#9BA5BC";
const FAINT = "#6E7A99";
const BG = "#0C0E14";
const PANEL_A = "#171C2C";
const PANEL_B = "#12151F";
const BORDER = "#1E2436";

// deterministic pseudo-random (fractal), so renders are stable
const rand = (i) => {
  const s = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inline(s) {
  return esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<i class=\"t\">$1</i>");
}

function extract(raw) {
  const o = { kicker: "", headline: "", sub: "", bullets: [], paras: [], table: [] };
  for (const rawLine of raw) {
    const l = rawLine.trim();
    if (!l || l.startsWith("<!--")) continue;
    if (l.startsWith("### ")) { o.sub = l.slice(4).trim(); continue; }
    if (l.startsWith("## ")) { o.headline = l.slice(3).trim(); continue; }
    if (l.startsWith("# ")) { o.kicker = l.slice(2).trim(); continue; }
    if (l.startsWith("|") && l.endsWith("|")) {
      o.table.push(l.split("|").slice(1, -1).map((c) => c.trim()));
      continue;
    }
    if (l.startsWith("- ")) { o.bullets.push(l.slice(2).trim()); continue; }
    o.paras.push(l);
  }
  return o;
}

function parseSlides(md) {
  const slides = [];
  let cur = [];
  const flush = () => {
    if (cur.length) slides.push(cur);
    cur = [];
  };
  for (const line of md.split(/\r?\n/)) {
    if (line.trim() === "---") { flush(); continue; }
    cur.push(line);
  }
  flush();
  return slides.filter((s) => s.some((l) => l.trim() !== ""));
}

function dotGrid(w, h, step, r, color) {
  const dots = [];
  for (let x = step / 2; x < w; x += step) {
    for (let y = step / 2; y < h; y += step) {
      dots.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}"/>`);
    }
  }
  return `<svg class="bg-dots" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" fill="${color}">${dots.join("")}</svg>`;
}

function glowLayer(className) {
  return `<div class="bg-layer">${dotGrid(1280, 720, 44, 1.3, "rgba(244,246,251,0.045)")}<div class="bg-glow ${className}"></div></div>`;
}

function headerBlock(o) {
  return `<header class="shead">
  <div class="kicker">${inline(o.kicker)}</div>
  <h1 class="headline">${inline(o.headline)}</h1>
</header>`;
}

function footerBlock(index, total) {
  return `<footer class="pfoot">
  <div class="f-brand">LlamaToaster</div>
  <div class="pagenum">${String(index + 1).padStart(2, "0")} / ${String(total).padStart(2, "0")}</div>
</footer>`;
}

function chip(text, cls) {
  return `<span class="chip ${cls || ""}">${inline(text)}</span>`;
}

const CSS = `
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: ${BG}; }
body {
  font-family: 'Segoe UI', 'Segoe UI Variable Text', 'Helvetica Neue', Arial, sans-serif;
  color: ${INK}; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
.slide {
  position: relative; width: 1280px; height: 720px; overflow: hidden;
  padding: 62px 84px 92px;
  display: flex; flex-direction: column;
  page-break-after: always; break-after: page;
  background: ${BG};
}
.slide:last-child { page-break-after: auto; break-after: auto; }
body.preview .slide { display: none; }
body.preview .slide:target { display: flex; }
.disp { font-family: 'Bahnschrift', 'Segoe UI', sans-serif; }
b { color: ${INK}; font-weight: 700; }

/* background */
.bg-layer { position: absolute; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.bg-glow { position: absolute; width: 840px; height: 840px; border-radius: 50%;
  background: radial-gradient(circle, rgba(124,92,255,0.16), transparent 62%); filter: blur(46px); }
.bg-glow.g-left { left: -220px; top: 40px; }
.bg-glow.g-right { right: -260px; top: 10px; }
.bg-glow.g-tr { right: -180px; top: -220px; }
.bg-glow.g-br { right: -240px; bottom: -260px; }
.bg-glow.g-center { left: 50%; top: 26%; transform: translateX(-50%); }

/* header */
.shead { position: relative; z-index: 2; display: flex; flex-direction: column; gap: 14px; }
.kicker { display: flex; align-items: center; gap: 14px;
  font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-size: 14.5px; font-weight: 600;
  letter-spacing: 0.3em; text-transform: uppercase; color: ${ACCENT}; }
.kicker::before { content: ""; width: 30px; height: 2px; border-radius: 2px; background: ${ACCENT}; }
.headline { font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-weight: 600;
  font-size: 45px; line-height: 1.07; letter-spacing: -0.6px; max-width: 1160px; }
.headline em { font-style: normal; color: ${ACCENT}; }

/* body */
.sbody { position: relative; z-index: 2; flex: 1; display: flex; gap: 44px; margin-top: 26px; min-height: 0; }
.col { display: flex; flex-direction: column; justify-content: center; gap: 18px; min-width: 0; }

/* panels / chips / bullets */
.card { background: linear-gradient(180deg, ${PANEL_A} 0%, ${PANEL_B} 100%);
  border: 1px solid ${BORDER}; border-radius: 18px; }
.chip { display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
  padding: 7px 14px; border-radius: 999px; border: 1px solid #242C42; background: #13172A;
  font-size: 13px; font-weight: 600; color: #B9C2D8; }
.chip.vip { border-color: rgba(124,92,255,0.65); background: rgba(124,92,255,0.14); }
.chip.vip .v { color: ${ACCENT}; }
.chip b { color: ${INK}; }
.chip .v { color: ${ACCENT}; font-weight: 700; }
.chip .ok { color: ${OK}; font-weight: 700; }

.pains { display: flex; flex-direction: column; gap: 13px; }
.pain { display: flex; gap: 16px; align-items: flex-start; }
.pn { flex: none; width: 46px; height: 46px; border-radius: 13px; margin-top: 2px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 18px;
  color: ${BG}; background: ${ACCENT}; }
.pain-body b { display: block; font-family: 'Bahnschrift', sans-serif; font-size: 20.5px;
  font-weight: 600; line-height: 1.16; color: ${INK}; }
.pain-body span { display: block; margin-top: 5px; font-size: 14.5px; color: ${MUTED}; line-height: 1.36; }

ul.list { list-style: none; display: flex; flex-direction: column; gap: 11px; }
ul.list li { position: relative; padding-left: 26px; font-size: 16px; line-height: 1.42; color: #C4CCDE; }
ul.list li::before { content: ""; position: absolute; left: 2px; top: 9px; width: 9px; height: 9px;
  border-radius: 3px; background: ${ACCENT}; }
ul.list li b { color: ${INK}; }

/* ---- footer ---- */
.pfoot { position: absolute; left: 84px; right: 84px; bottom: 30px; z-index: 3;
  display: flex; justify-content: space-between; align-items: center;
  font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-size: 12px;
  letter-spacing: 0.14em; color: ${FAINT}; text-transform: uppercase; }
.pfoot .f-brand { font-weight: 700; letter-spacing: 0.22em; color: #39415A; }
.pfoot .pagenum b { color: ${ACCENT}; font-weight: 600; }

/* ==== 1. cover ==== */
.slide.cover { align-items: center; justify-content: center; text-align: center; padding-top: 54px; }
.cover .cv-inner { position: relative; z-index: 2; display: flex; flex-direction: column;
  align-items: center; gap: 26px; }
.cv-tag { display: inline-flex; align-items: center; gap: 12px; padding: 9px 22px;
  border-radius: 999px; border: 1px solid #2A2F4A; background: rgba(124,92,255,0.08);
  font-family: 'Bahnschrift', sans-serif; font-size: 14px; font-weight: 600;
  letter-spacing: 0.26em; text-transform: uppercase; color: #B9C2D8; }
.cv-tag b { color: ${ACCENT}; font-weight: 700; }
.cv-hero { display: flex; flex-direction: column; gap: 4px; }
.cv-row { display: flex; align-items: baseline; justify-content: center; }
.cv-num { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 108px;
  line-height: 1; color: ${ACCENT}; font-feature-settings: 'tnum'; letter-spacing: -2px; margin-right: 34px; }
.cv-num b { font-weight: 600; color: ${ACCENT}; }
.cv-lab { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 42px;
  color: ${INK}; letter-spacing: -0.5px; }
.cv-brand { margin-top: 6px; }
.cv-brand .t1 { font-family: 'Bahnschrift', sans-serif; font-weight: 700; font-size: 30px;
  letter-spacing: 0.3em; text-transform: uppercase; color: ${INK}; }
.cv-brand .t1 b { color: ${ACCENT}; }
.cv-brand .t2 { margin-top: 10px; font-size: 19px; color: ${MUTED}; }
.cv-foot { font-size: 14.5px; color: ${FAINT}; max-width: 760px; line-height: 1.5; }

/* ==== 2. problem ==== */
.sbody.problem { gap: 36px; margin-top: 34px; }
.problem .col-left { width: 50%; justify-content: flex-start; }
.problem .col-right { width: 44%; justify-content: center; align-items: flex-end; gap: 22px; }
.cat-wrap { position: relative; width: 100%; max-width: 440px; }
.cat-label { position: absolute; top: -26px; right: 0; z-index: 3; }
.problem .chips { display: flex; gap: 10px; flex-wrap: wrap; max-width: 450px; }

/* ==== 3. missing half ==== */
.sbody.missing { margin-top: 26px; }
.missing .col-left { width: 50%; }
.missing .col-right { width: 50%; gap: 20px; }
.barpanel { padding: 22px 26px 20px; }
.barpanel .hp { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; }
.barpanel .hp .hp-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 14.5px;
  letter-spacing: 0.12em; text-transform: uppercase; color: ${MUTED}; }
.barpanel .hp .hp-n { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 26px; color: ${INK}; }
.bars { display: flex; align-items: flex-end; gap: 14px; height: 78px; }
.bars .bar { flex: 1; border-radius: 5px 5px 2px 2px; background: currentColor; }
.zero-panel { padding: 24px 30px 26px; display: flex; align-items: center; gap: 30px; }
.zero-hero { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 190px;
  line-height: 0.85; color: ${ACCENT}; font-feature-settings: 'tnum'; }
.zero-side { display: flex; flex-direction: column; gap: 10px; }
.zero-side .zs-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px;
  letter-spacing: 0.06em; color: ${INK}; line-height: 1.25; }
.zero-metrics { display: flex; flex-wrap: wrap; gap: 8px; }

/* ==== 4. why now ==== */
.sbody.why { gap: 52px; margin-top: 28px; }
.why .col-left { width: 46%; justify-content: center; }
.why .col-right { width: 54%; justify-content: center; gap: 20px; }
.xrow { display: flex; gap: 14px; align-items: flex-start; }
.xmark { flex: none; width: 34px; height: 34px; margin-top: 2px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center; font-size: 16px;
  color: ${MUTED}; border: 1px solid ${BORDER}; background: ${PANEL_B}; }
.xbody { font-size: 16.5px; line-height: 1.4; color: #C4CCDE; }
.xbody b { color: ${INK}; }

/* ==== 5. fix ==== */
.sbody.fix { gap: 50px; margin-top: 30px; }
.fix .col-left { width: 42%; justify-content: center; gap: 22px; }
.fix .col-right { width: 58%; justify-content: center; gap: 26px; }
.pipe { display: flex; flex-direction: column; gap: 14px; }
.pipe-row { display: flex; align-items: center; gap: 18px; }
.pipe-box { flex: 1; padding: 18px 22px; text-align: center; }
.pipe-box .pt { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 21px;
  letter-spacing: 0.14em; color: ${INK}; }
.pipe-box .pv { margin-top: 7px; font-size: 13.5px; color: ${MUTED}; line-height: 1.4; }
.pipe-arrow { flex: none; color: ${ACCENT}; font-size: 26px; line-height: 1; }
.profiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.prof { padding: 16px 14px 14px; text-align: center; position: relative; overflow: hidden; }
.prof::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: ${ACCENT}; opacity: 0.85; }
.prof .pr-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 15.5px;
  letter-spacing: 0.02em; color: ${INK}; line-height: 1.15; }
.prof .pr-s { margin-top: 8px; font-size: 11.5px; color: ${FAINT}; line-height: 1.3; }
.fix .col-right .caps { display: flex; flex-wrap: wrap; gap: 9px; justify-content: center; }

/* ==== 6. trust ==== */
.sbody.trust { gap: 50px; margin-top: 26px; }
.trust .col-left { width: 54%; justify-content: center; }
.trust .col-right { width: 46%; justify-content: center; }
.checkrow { display: flex; gap: 14px; align-items: flex-start; }
.checkmark { flex: none; width: 26px; height: 26px; margin-top: 2px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700;
  color: #0C2A16; background: ${OK}; }
.checkbody { font-size: 15.5px; line-height: 1.42; color: #C4CCDE; }
.checkbody b { color: ${INK}; }
.mock { position: relative; width: 100%; max-width: 460px; padding: 20px 24px 16px; }
.mock .mh { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }
.mock .mh .mt { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px; color: ${INK}; }
.mock .mh .ms { font-size: 12.5px; color: ${FAINT}; letter-spacing: 0.06em; }
.mrow { display: flex; justify-content: space-between; align-items: center;
  padding: 11px 14px; border-radius: 10px; margin-bottom: 8px;
  background: ${PANEL_B}; border: 1px solid ${BORDER}; }
.mrow .ml { font-size: 14px; color: ${MUTED}; }
.mrow .mv { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 19px; color: ${INK}; }
.mrow.bad { border-color: rgba(124,92,255,0.45); }
.mrow.bad .ml { color: ${FAINT}; text-decoration: line-through; }
.tag-rej { font-family: 'Bahnschrift', sans-serif; font-size: 12px; font-weight: 600;
  letter-spacing: 0.14em; color: ${ACCENT}; border: 1px solid rgba(124,92,255,0.6);
  padding: 3px 10px; border-radius: 999px; display: inline-block; }
.mfoot { display: flex; align-items: center; justify-content: space-between; margin-top: 10px; gap: 12px; }
.stampwrap { display: flex; flex-direction: column; align-items: center; gap: 8px; flex: none; }
.stamp { width: 90px; height: 90px; border-radius: 50%; border: 3px solid ${OK}; opacity: 0.9;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1px;
  transform: rotate(8deg); }
.stamp .st-v { font-family: 'Bahnschrift', sans-serif; font-weight: 700; font-size: 13.5px;
  letter-spacing: 0.12em; color: ${OK}; }
.stamp .st-s { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 10.5px;
  letter-spacing: 0.16em; color: ${OK}; opacity: 0.85; }
.mhash { font-family: 'Bahnschrift', sans-serif; font-size: 11.5px; letter-spacing: 0.1em; color: ${FAINT}; }
.mhash b { color: ${OK}; }

/* ==== 7. competitors ==== */
.sbody.comp { display: block; margin-top: 18px; }
.matrixwrap { position: relative; z-index: 2; padding-left: 32px; }
.matrix { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
  width: 100%; height: 300px; border: 1px solid ${BORDER}; border-radius: 20px; overflow: hidden;
  background: ${PANEL_B}; }
.axis-y { position: absolute; left: 0; top: 0; width: 18px; height: 300px;
  display: flex; flex-direction: column; justify-content: space-between; align-items: center; }
.axis-y span { font-family: 'Bahnschrift', sans-serif; font-size: 11.5px; letter-spacing: 0.18em;
  text-transform: uppercase; color: ${FAINT}; white-space: nowrap;
  writing-mode: vertical-rl; transform: rotate(180deg); }
.quad { position: relative; padding: 18px 24px; display: flex; flex-direction: column;
  justify-content: flex-end; gap: 7px; }
.quad + .quad { border-left: 1px solid ${BORDER}; }
.quad.bot { border-top: 1px solid ${BORDER}; padding-top: 14px; padding-bottom: 14px; }
.quad .qx { font-family: 'Bahnschrift', sans-serif; font-size: 12px; letter-spacing: 0.16em;
  text-transform: uppercase; color: ${FAINT}; }
.quad .qb { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 20px; color: ${INK}; }
.quad .qs { font-size: 13px; color: ${MUTED}; line-height: 1.4; }
.quad.spot { background: radial-gradient(circle at 50% 42%, rgba(124,92,255,0.16), transparent 68%);
  justify-content: center; align-items: center; text-align: center; gap: 8px;
  border: 2px dashed rgba(124,92,255,0.55); border-radius: 18px; margin: 10px; }
.quad.spot .starn { width: 28px; height: 28px; }
.crow { display: flex; align-items: center; gap: 9px; font-size: 13.5px; color: ${MUTED}; }
.crow .v { color: ${ACCENT}; font-size: 9px; }
.axis-x { position: relative; display: flex; justify-content: space-between; margin-top: 8px; }
.axis-x span { font-family: 'Bahnschrift', sans-serif; font-size: 12px; letter-spacing: 0.18em;
  text-transform: uppercase; color: ${FAINT}; padding: 0 8px; background: ${BG}; }
.compstrip { position: relative; z-index: 2; display: flex; gap: 40px; margin-top: 14px; }
.stripcol { flex: 1; display: flex; flex-direction: column; gap: 7px; }
.stripcol .st-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 12px;
  letter-spacing: 0.18em; text-transform: uppercase; color: ${MUTED}; }
.irow { display: flex; gap: 9px; align-items: baseline; }
.irow .ik { flex: none; width: 20px; text-align: center; font-family: 'Bahnschrift', sans-serif;
  font-weight: 700; }
.irow.ok .ik { color: ${OK}; }
.irow.no .ik { color: ${ACCENT}; }
.irow span.t { font-size: 13.5px; color: #C4CCDE; }
.irow.no span.t b { color: ${INK}; }

/* ==== 8. ask ==== */
.slide.ask { align-items: center; }
.ask .awrap { position: relative; z-index: 2; display: flex; flex-direction: column;
  align-items: center; gap: 28px; margin-top: -4px; }
.ask .a-hero { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.a-numrow { display: flex; align-items: baseline; gap: 24px; }
.a-one { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 170px;
  line-height: 0.88; color: ${ACCENT}; font-feature-settings: 'tnum'; }
.a-lab { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 38px; color: ${INK}; }
.a-lab b { color: ${ACCENT}; }
.ask .a-url { display: inline-flex; align-items: center; gap: 12px; padding: 13px 30px;
  border-radius: 16px; border: 1px solid rgba(124,92,255,0.6); background: rgba(124,92,255,0.1);
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 29px; color: ${INK};
  letter-spacing: 0.01em; }
.ask .a-url b { color: ${ACCENT}; }
.adirs { display: flex; gap: 16px; width: 100%; max-width: 1080px; }
.adir { flex: 1; padding: 17px 20px; }
.adir .ad-n { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 15px;
  color: ${ACCENT}; letter-spacing: 0.06em; }
.adir .ad-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px;
  color: ${INK}; margin-top: 6px; line-height: 1.2; }
.adir .ad-s { font-size: 13px; color: ${MUTED}; margin-top: 6px; line-height: 1.4; }
`;
function coverSlide(o, index, total) {
  const tag = o.paras[0] || "Local LLMs · your hardware · your data";
  const hero = o.paras[1] || "4 000 000 files. 40 knobs. One right answer.";
  const brand = o.paras.find((p) => p.includes("LlamaToaster")) || "LlamaToaster — the appliance that finds it.";
  const foot = o.paras.find((p) => p.includes("llama.cpp expert")) || "";
  const rows = [];
  for (const part of hero.split(/\.\s+/).filter(Boolean)) {
    const m = part.match(/^([\d\s]+)/);
    let num, lab;
    if (m) {
      num = m[1].trim();
      lab = part.slice(m[0].length).trim();
    } else if (/^one\b/i.test(part)) {
      num = "1";
      lab = part.replace(/^one\s+/i, "").trim();
    } else {
      num = "";
      lab = part;
    }
    rows.push(
      `<div class="cv-row">
        <span class="cv-num">${num}</span>${lab ? `<span class="cv-lab">${lab.replace(/\b\w/g, (c) => c.toUpperCase())}</span>` : ""}
      </div>`
    );
  }
  return `<section class="slide cover" id="slide-${index + 1}">
${glowLayer("g-center")}
  <div class="cv-inner">
    <div class="cv-tag"><b>Local LLMs</b> · your hardware · your data</div>
    <div class="cv-hero">${rows.join("")}</div>
    <div class="cv-brand">
      <div class="t1"><b>LlamaToaster</b> — the appliance that finds it.</div>
      <div class="t2">4 000 000 files. 40 knobs. One right answer.</div>
    </div>
    <div class="cv-foot">${inline(foot)}</div>
    </div>
${footerBlock(index, total)}
</section>`;
}

function toasterIcon(size) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 120 120" fill="none">
  <rect x="20" y="36" width="80" height="54" rx="10" stroke="${ACCENT}" stroke-width="3"/>
  <rect x="12" y="28" width="96" height="8" rx="4" stroke="${ACCENT}" stroke-width="3"/>
  <path d="M30 50h9M30 62h14M30 74h7" stroke="${ACCENT}" stroke-width="3" stroke-linecap="round"/>
  <path d="M52 34c8-10 20-10 28 0" stroke="${ACCENT}" stroke-width="3" stroke-linecap="round" opacity="0.55"/>
</svg>`;
}

function makeCatalogGrid(w, h, step) {
  const items = [];
  const cols = Math.floor(w / step);
  const rows = Math.floor(h / step);
  for (let i = 0; i < cols * rows; i++) {
    const cx = (i % cols) * step + step / 2;
    const cy = Math.floor(i / cols) * step + step / 2;
    const r = rand(i);
    let fill = "rgba(148,163,190,0.10)";
    let extra = "";
    if (r > 0.94) { fill = "rgba(124,92,255,0.50)"; }
    if (r > 0.985) {
      fill = ACCENT;
      extra = `<rect x="${(cx - step / 2 + 3).toFixed(1)}" y="${(cy - step / 2 + 3).toFixed(1)}" width="${(step - 6).toFixed(1)}" height="${(step - 6).toFixed(1)}" rx="6" fill="${ACCENT}" opacity="0.35" filter="blur(4px)"/>`;
    }
    items.push(`${extra}<rect x="${(cx - step / 2 + 2).toFixed(1)}" y="${(cy - step / 2 + 2).toFixed(1)}" width="${(step - 4).toFixed(1)}" height="${(step - 4).toFixed(1)}" rx="6" fill="${fill}"/>`);
  }
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${items.join("")}</svg>`;
}

function problemSlide(o, index, total) {
  const pains = [
    { n: "01", t: "One bad knob — silent quality loss, or a hard crash", s: "Wrong KV cache → degraded tokens. Context too large → OOM." },
    { n: "02", t: "One good config on GPU X — broken on GPU Y", s: "Reddit recipes fail across architectures, drivers, OS versions." },
    { n: "03", t: "The stale oracle", s: "Ask ChatGPT “best local LLM 2026” → confident, outdated folklore." },
  ];
  const painHtml = pains
    .map((p) => `<div class="pain">
  <div class="pn">${p.n}</div>
  <div class="pain-body"><b>${p.t}</b><span>${p.s}</span></div>
</div>`)
    .join("");
  const grid = makeCatalogGrid(440, 250, 30);
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-left")}
${headerBlock(o)}
  <div class="sbody problem">
    <div class="col col-left">
      <div class="pains">${painHtml}</div>
    </div>
    <div class="col col-right">
      <div class="cat-wrap">
        ${grid}
        <span class="chip cat-label"><span class="v">✦</span> the right config</span>
      </div>
      <div class="chips">
        <span class="chip"><span class="v">~200 000</span> GGUF ready</span>
        <span class="chip"><span class="v">≈4 000 000</span> files</span>
        <span class="chip"><span class="v">10+</span> architectures</span>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function missingSlide(o, index, total) {
  const bars = [38, 52, 44, 70, 58, 82, 64].map((h, i) =>
    `<div class="bar" style="height:${h}%;color:${i === 5 ? ACCENT : "rgba(124,92,255,0.35)"}"></div>`
  ).join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-tr")}
${headerBlock(o)}
  <div class="sbody missing">
    <div class="col col-left">
      <p style="font-size:17px;line-height:1.5;color:#C4CCDE">${inline(o.bullets[0] || "")}</p>
      <ul class="list">
        <li>${inline("LMArena · Open LLM Leaderboard · MMLU · GPQA · HumanEval · **MT-Bench** — every release is scored nightly.")}</li>
        <li>${inline("But almost nothing answers the other half: **tok/s · latency · genuinely usable context — on YOUR hardware.**")}</li>
      </ul>
      <p style="font-size:15.5px;line-height:1.55;color:${FAINT}">Millions of files and mountains of quality scores — while performance is still folklore and trial-and-error.</p>
    </div>
    <div class="col col-right">
      <div class="card barpanel">
        <div class="hp">
          <div class="hp-t">Quality — scored nightly</div>
          <div class="hp-n">every release</div>
        </div>
        <div class="bars">${bars}</div>
      </div>
      <div class="card zero-panel">
        <div class="zero-hero">0</div>
        <div class="zero-side">
          <div class="zs-t">answers for<br>your hardware</div>
          <div class="zero-metrics">
            <span class="chip">tok/s</span>
            <span class="chip">latency</span>
            <span class="chip">usable context</span>
            <span class="chip">OOM-free</span>
          </div>
        </div>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}
function whySlide(o, index, total) {
  const x = [
    { t: "Blindly chasing the biggest model", s: "→ swap, crash, 0.3 tok/s" },
    { t: "Chasing hype", s: "→ Llama 4, when Qwen3.8 is better for YOUR task" },
    { t: "Following last month's advice", s: "→ already stale" },
  ];
  const curve = `<svg width="440" height="190" viewBox="0 0 440 190" fill="none">
  <line x1="40" y1="170" x2="430" y2="170" stroke="rgba(148,163,190,0.35)" stroke-width="2"/>
  <line x1="40" y1="170" x2="40" y2="8" stroke="rgba(148,163,190,0.35)" stroke-width="2"/>
  <path d="M40 168 C 90 150, 110 120, 140 104 S 220 74, 260 52 S 380 22, 422 18" stroke="${ACCENT}" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M40 168 C 90 150, 110 120, 140 104 S 220 74, 260 52 S 380 22, 422 18" stroke="${ACCENT}" stroke-width="10" fill="none" opacity="0.18" stroke-linecap="round"/>
  <line x1="260" y1="166" x2="260" y2="52" stroke="rgba(124,92,255,0.6)" stroke-width="2" stroke-dasharray="4 5"/>
  <circle cx="260" cy="52" r="7" fill="${ACCENT}"/>
  <circle cx="260" cy="52" r="13" stroke="${ACCENT}" stroke-width="2" opacity="0.4"/>
  <text x="272" y="42" fill="${INK}" font-family="Bahnschrift" font-weight="600" font-size="17">27B ≈ flagship</text>
  <text x="272" y="60" fill="${MUTED}" font-family="Bahnschrift" font-size="13" letter-spacing="1">on an ordinary PC</text>
  <text x="44" y="186" fill="${FAINT}" font-family="Bahnschrift" font-size="12" letter-spacing="2">MODEL SIZE →</text>
</svg>`;
  const xhtml = x
    .map(
      (p) => `<div class="xrow">
  <div class="xmark">✕</div>
  <div class="xbody"><b>${p.t}</b><br>${p.s}</div>
</div>`
    )
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-right")}
${headerBlock(o)}
  <div class="sbody why">
    <div class="col col-left">
      <div style="font-size:15.5px;color:${MUTED};line-height:1.45">Qwen3.8 27B ≈ previous-generation flagship quality — on an ordinary PC.</div>
      <div style="display:flex;flex-direction:column;gap:12px">${xhtml}</div>
      <div style="font-size:15px;color:${FAINT};line-height:1.45">Proprietary APIs: outages, refusals, price changes — nothing you control.<br>Your data stays on your machine.</div>
    </div>
    <div class="col col-right">
      ${curve}
      <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center">
        <span class="chip">Gemma 3</span>
        <span class="chip">Kimi K2</span>
        <span class="chip">GLM-4</span>
        <span class="chip"><span class="v">every month</span> a new “impossible” small model</span>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function fixSlide(o, index, total) {
  const profs = [
    { t: "Max Speed", s: "lowest latency\nfor your GPU" },
    { t: "Balanced", s: "speed × quality\nsweet spot" },
    { t: "Max Context", s: "longest verified\ncontext ceiling" },
    { t: "Low Memory", s: "runs on modest\nhardware" },
  ];
  const profHtml = profs
    .map(
      (p) => `<div class="card prof">
  <div class="pr-t">${p.t}</div>
  <div class="pr-s">${p.s.replace(/\n/g, "<br>")}</div>
</div>`
    )
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-left")}
${headerBlock(o)}
  <div class="sbody fix">
    <div class="col col-left">
      <p style="font-size:16px;line-height:1.5;color:#C4CCDE">${inline("One command connects any GPU or CPU box — **pull-only, zero open ports**, no firewall gymnastics.")}</p>
      <ul class="list">
        <li>${inline("You state intent: **goal × workload × target context**.")}</li>
        <li>${inline("The grid builds itself. **The sweep runs itself.**")}</li>
      </ul>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <span class="chip"><span class="v">5 min</span> set up tonight</span>
        <span class="chip">Intent in · <span class="v">Decision out</span></span>
      </div>
    </div>
    <div class="col col-right">
      <div class="pipe">
        <div class="pipe-row">
          <div class="card pipe-box"><div class="pt">INTENT</div><div class="pv">goal × workload × target context</div></div>
          <div class="pipe-arrow">→</div>
          <div class="card pipe-box"><div class="pt">GRID + SWEEP</div><div class="pv">the grid builds itself<br>the sweep runs itself</div></div>
        </div>
      </div>
      <div class="profiles">${profHtml}</div>
      <div class="caps">
        <span class="chip">verified context ceilings</span>
        <span class="chip">concurrency knees</span>
        <span class="chip">fair model-vs-model</span>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}
function trustSlide(o, index, total) {
  const checks = [
    { t: "Eligibility gates", s: "stability, suspect samples, stddev floors — timer-bug results like <b>1e6 tok/s</b> go to the trash, not the average." },
    { t: "Memory from real placement", s: "per-tensor GGUF placement catches “31/31 offloaded” while secretly running in system RAM." },
    { t: "Fair by construction", s: "a different machine, build, or GPU rejects the member, not the verdict." },
    { t: "Tamper-evident", s: "a hash per row, methodology versions never mixed." },
  ];
  const checkHtml = checks
    .map(
      (c) => `<div class="checkrow">
  <div class="checkmark">✓</div>
  <div class="checkbody"><b>${c.t}.</b> ${c.s}</div>
</div>`
    )
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-right")}
${headerBlock(o)}
  <div class="sbody trust">
    <div class="col col-left">
      <div style="display:flex;flex-direction:column;gap:13px">${checkHtml}</div>
    </div>
    <div class="col col-right">
      <div class="card mock">
        <div class="mh"><div class="mt">Profile · Balanced</div><div class="ms">RTX 4090 · ggml-v6</div></div>
        <div class="mrow"><span class="ml">tok/s</span><span class="mv">42.1</span></div>
        <div class="mrow"><span class="ml">context ceiling</span><span class="mv">32 768</span></div>
        <div class="mrow"><span class="ml">peak memory</span><span class="mv">11.2 GB</span></div>
        <div class="mrow bad"><span class="ml">burst “tok/s” <s>1e6</s></span><span class="tag-rej">REJECTED</span></div>
        <div class="mfoot">
          <div class="mhash">method v2 · row-hash <b>d4f9·3c81</b></div>
          <div class="stampwrap">
            <div class="stamp"><div class="st-v">✓VERIFIED</div><div class="st-s">method v2</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}
function compSlide(o, index, total) {
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-tr")}
${headerBlock(o)}
  <div class="sbody comp">
    <div class="matrixwrap">
      <div class="axis-y"><span>high optimization</span><span>low optimization</span></div>
      <div class="matrix">
        <div class="quad spot">
          <svg class="starn" viewBox="0 0 32 32" fill="${ACCENT}"><path d="M16 2l3.6 7.8 8.2 1-6 5.7 1.5 8.2L16 21l-7.3 3.7 1.5-8.2-6-5.7 8.2-1z"/></svg>
          <div class="qb" style="color:${ACCENT}">the open spot</div>
          <div class="qs">easy to use<br>+ high optimization</div>
        </div>
        <div class="quad">
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px">
            <span class="chip vip"><span class="v">●</span> LlamaToaster</span>
            <span class="qs" style="display:flex;align-items:center;gap:6px">← into the open spot</span>
          </div>
          <div class="qx">Hard to use</div>
        </div>
        <div class="quad bot">
          <div style="display:flex;flex-direction:column;gap:7px">
            <div class="crow"><span class="v">●</span> LM Studio</div>
            <div class="crow"><span class="v">●</span> Ollama</div>
            <div class="crow"><span class="v">●</span> Jan.ai</div>
          </div>
          <div class="qx">Easy to use</div>
        </div>
        <div class="quad bot">
          <div class="qs" style="opacity:0.5">unserved<br>— nobody optimizes this corner</div>
        </div>
      </div>
      <div class="axis-x"><span>easy to use</span><span>hard to use</span></div>
    </div>
    <div class="compstrip">
      <div class="stripcol">
        <div class="st-t">What they do</div>
        <div class="irow ok"><span class="ik">✓</span><span class="t">pick a model that “probably works”</span></div>
        <div class="irow ok"><span class="ik">✓</span><span class="t">abstract complexity · one-click install</span></div>
      </div>
      <div class="stripcol">
        <div class="st-t">What they don't do</div>
        <div class="irow no"><span class="ik">✗</span><span class="t"><b>find optimal config</b> for YOUR hardware</span></div>
        <div class="irow no"><span class="ik">✗</span><span class="t"><b>sweep</b> parameters · detect <b>silent quality degradation</b></span></div>
        <div class="irow no"><span class="ik">✗</span><span class="t">community grid · <b>tamper-evident</b> benchmark cards</span></div>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function askSlide(o, index, total) {
  const dirs = [
    { n: "01", t: "Run your rig tonight", s: "no port forwarding, no manual build — llama.cpp installs from the web UI" },
    { n: "02", t: "Join the community grid", s: "opt into the k-anonymized set → the whole catalog becomes searchable for everyone" },
    { n: "03", t: "Star it while it's early", s: "llamatoaster.com/benchmark — one command connects your machine" },
  ];
  const dirHtml = dirs
    .map(
      (d) => `<div class="card adir">
  <div class="ad-n">${d.n} / 03</div>
  <div class="ad-t">${d.t}</div>
  <div class="ad-s">${d.s}</div>
</div>`
    )
    .join("");
  return `<section class="slide ask" id="slide-${index + 1}">
${glowLayer("g-center")}
${headerBlock(o)}
  <div class="awrap">
    <div class="a-hero">
      <div class="a-numrow">
        <div class="a-one">1</div>
        <div class="a-lab">command<br><b>connects your machine</b></div>
      </div>
    </div>
    <div class="a-url">llamatoaster.com/<b>benchmark</b></div>
    <div class="adirs">${dirHtml}</div>
  </div>
${footerBlock(index, total)}
</section>`;
}
function slidePicker(o) {
  const h = (o.headline || "").toLowerCase();
  if (h.includes("problem")) return "problem";
  if (h.includes("missing")) return "missing";
  if (h.includes("why now")) return "why";
  if (h.includes("fix")) return "fix";
  if (h.includes("trust")) return "trust";
  if (h.includes("competit")) return "comp";
  if (h.includes("ask")) return "ask";
  return "cover";
}

function buildHtml(mdText, lang) {
  const slides = parseSlides(mdText);
  const body = slides
    .map((raw, i) => {
      const o = extract(raw);
      const kind = slidePicker(o);
      const fn = {
        cover: coverSlide,
        problem: problemSlide,
        missing: missingSlide,
        why: whySlide,
        fix: fixSlide,
        trust: trustSlide,
        comp: compSlide,
        ask: askSlide,
      }[kind];
      return fn(o, i, slides.length);
    })
    .join("\n");
  return `<!DOCTYPE html>
<html lang="${lang}">
<head>
<meta charset="utf-8">
<title>LlamaToaster — Pitch Deck v2</title>
<style>
${CSS}
</style>
</head>
<body>
${body}
<script>
(function () {
  if (location.hash && location.hash.indexOf("slide-") === 1) {
    document.body.classList.add("preview");
  }
})();
(function () {
  if (location.hash !== "#check") return;
  var issues = [];
  var slides = document.querySelectorAll(".slide");
  slides.forEach(function (s, i) {
    var sr = s.getBoundingClientRect();
    var bad = [];
    s.querySelectorAll("*").forEach(function (el) {
      var r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      if (r.left < sr.left - 1 || r.right > sr.right + 1 || r.top < sr.top - 1 || r.bottom > sr.bottom + 1) {
        var nm = el.tagName + "." + String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className);
        bad.push(nm + " [" + Math.round(r.left) + "," + Math.round(r.top) + " " + Math.round(r.width) + "x" + Math.round(r.height) + "]");
      }
      if (el.scrollWidth > el.clientWidth + 2 && /^(div|span|h1|p|li|ul)$/.test(el.tagName.toLowerCase())) {
        bad.push("OVERFLOW " + el.tagName + "." + el.className + " scroll=" + el.scrollWidth + " client=" + el.clientWidth);
      }
    });
    if (bad.length) issues.push("SLIDE " + (i + 1) + ": " + bad.slice(0, 12).join(" | "));
  });
  var pre = document.createElement("pre");
  pre.id = "check-out";
  pre.textContent = issues.length ? issues.join("\\n") : "ALL SLIDES OK";
  document.body.appendChild(pre);
})();
</script>
</body>
</html>
`;
}

const BROWSER_CANDIDATES = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
];

function findBrowser() {
  return BROWSER_CANDIDATES.find((p) => existsSync(p)) ?? null;
}
function runBrowser(extra, url) {
  const browser = findBrowser();
  if (!browser) {
    console.error("No Edge/Chrome found. Install Microsoft Edge or edit BROWSER_CANDIDATES.");
    process.exit(1);
  }
  const profile = join(tmpdir(), `lt-pitch-v2-${Date.now()}-${Math.round(Math.random() * 1e6)}`);
  const res = launch(browser, profile, extra, url);
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {}
  if (res.error) {
    console.error(`browser spawn failed (${res.error.code}): ${browser}`);
    process.exit(1);
  }
  if (res.status !== 0) {
    console.error((res.stderr || res.stdout || "browser failed").slice(0, 2000));
    process.exit(res.status ?? 1);
  }
}

function launch(browser, profile, extra, url) {
  const args = [
    "--headless",
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    ...extra,
    url,
    `--user-data-dir=${profile}`,
  ];
  return spawnSync(browser, args, { stdio: ["ignore", "pipe", "pipe"], encoding: "utf8", timeout: 120_000 });
}

function dumpDom(htmlPath, hash) {
  const browser = findBrowser();
  if (!browser) {
    console.error("No Edge/Chrome found.");
    process.exit(1);
  }
  const profile = join(tmpdir(), `lt-pitch-v2-${Date.now()}-${Math.round(Math.random() * 1e6)}`);
  const res = launch(browser, profile, ["--dump-dom"], `${pathToFileURL(htmlPath).href}${hash}`);
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {}
  if (res.error) {
    console.error(`browser spawn failed (${res.error.code}): ${browser}`);
    process.exit(1);
  }
  if (res.status !== 0) {
    console.error((res.stderr || res.stdout || "browser failed").slice(0, 2000));
    process.exit(res.status ?? 1);
  }
  const html = res.stdout || "";
  const m = html.match(/<pre id="check-out">([\s\S]*?)<\/pre>/);
  if (m) {
    console.log(m[1].replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&"));
  } else {
    console.log("(no check output in DOM; check script may not have run)");
  }
}

function printPdf(htmlPath, outPdf) {
  if (existsSync(outPdf)) {
    try {
      rmSync(outPdf, { force: true });
    } catch (e) {
      console.error(`WARN: cannot replace ${outPdf} — it may be open in a PDF viewer.`);
      console.error(String(e.message || e));
      process.exit(2);
    }
  }
  runBrowser(
    [
      "--no-pdf-header-footer",
      "--print-to-pdf-no-header",
      `--print-to-pdf=${resolve(outPdf)}`,
    ],
    pathToFileURL(htmlPath).href
  );
  if (!existsSync(outPdf)) {
    console.error("PDF was not produced.");
    process.exit(1);
  }
}

function screenshotSlide(htmlPath, outPng, n) {
  runBrowser(
    [
      "--window-size=1280,720",
      "--virtual-time-budget=1500",
      `--screenshot=${resolve(outPng)}`,
    ],
    `${pathToFileURL(htmlPath).href}#slide-${n}`
  );
  if (!existsSync(outPng)) {
    console.error(`Screenshot not produced: ${outPng}`);
    process.exit(1);
  }
}

function langFrom(file) {
  return /\.ru\.md$/i.test(file) ? "ru" : "en";
}

function writeTempHtml(mdText, inPath) {
  const html = buildHtml(mdText, langFrom(inPath));
  const tmpHtml = join(tmpdir(), `lt-pitch-v2-${Date.now()}.html`);
  writeFileSync(tmpHtml, html);
  return tmpHtml;
}

function main() {
  const args = process.argv.slice(2);

  if (args[0] === "--check" && args.length >= 2) {
    const inPath = resolve(args[1]);
    const mdText = readFileSync(inPath, "utf8");
    const html = buildHtml(mdText, langFrom(inPath));
    const tmpHtml = join(tmpdir(), `lt-pitch-check-${Date.now()}.html`);
    writeFileSync(tmpHtml, html);
    dumpDom(tmpHtml, "#check");
    try {
      rmSync(tmpHtml, { force: true });
    } catch {}
    return;
  }

  if (args[0] === "--html-only" && args.length >= 3) {
    const mdText = readFileSync(resolve(args[1]), "utf8");
    writeFileSync(resolve(args[2]), buildHtml(mdText, langFrom(args[1])));
    console.log(`HTML written: ${resolve(args[2])}`);
    return;
  }

  if (args[0] === "--shots" && args.length >= 3) {
    const inPath = resolve(args[1]);
    const mdText = readFileSync(inPath, "utf8");
    const tmpHtml = writeTempHtml(mdText, inPath);
    const outDir = resolve(args[2]);
    mkdirSync(outDir, { recursive: true });
    const total = parseSlides(mdText).length;
    const picks = args.slice(3).filter((a) => /^\d+$/.test(a)).map(Number);
    const shots = picks.length ? picks : Array.from({ length: total }, (_, i) => i + 1);
    for (const n of shots) {
      if (n < 1 || n > total) continue;
      const png = join(outDir, `slide-${String(n).padStart(2, "0")}.png`);
      screenshotSlide(tmpHtml, png, n);
      console.log(`shot: ${png}`);
    }
    try {
      rmSync(tmpHtml, { force: true });
    } catch {}
    console.log(`slides rendered: ${total}`);
    return;
  }

  if (args.length >= 2) {
    const inPath = resolve(args[0]);
    const outPdf = resolve(args[1]);
    const mdText = readFileSync(inPath, "utf8");
    const tmpHtml = writeTempHtml(mdText, inPath);
    console.log(`Rendering ${basename(inPath)} -> ${outPdf} (${parseSlides(mdText).length} slides)`);
    printPdf(tmpHtml, outPdf);
    try {
      rmSync(tmpHtml, { force: true });
    } catch {}
    console.log(`PDF written: ${outPdf}`);
    return;
  }

  console.log(`Usage:
  node scripts/pitch-to-pdf-v2.mjs <in.md> <out.pdf>
  node scripts/pitch-to-pdf-v2.mjs --html-only <in.md> <out.html>
  node scripts/pitch-to-pdf-v2.mjs --shots <in.md> <outDir> [slide numbers...]
  node scripts/pitch-to-pdf-v2.mjs --check <in.md>`);
}

main();