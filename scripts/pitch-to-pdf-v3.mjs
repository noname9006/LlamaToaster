#!/usr/bin/env node
/**
 * LlamaToaster — pitch deck builder v3.
 * Renders docs/PITCH_DECK_v3.md to a dark 1280x720 PDF via headless Edge/Chrome.
 * Zero npm dependencies.
 *
 * Usage:
 *   node scripts/pitch-to-pdf-v3.mjs <in.md> <out.pdf>
 *   node scripts/pitch-to-pdf-v3.mjs --html-only <in.md> <out.html>
 *   node scripts/pitch-to-pdf-v3.mjs --shots <in.md> <outDir> [slide numbers...]
 *
 * Same design system as v2 (Bahnschrift display + Segoe UI body, one violet
 * accent, green reserved for "verified" semantics) but a 10-slide blueprint:
 * v2's overloaded slide 2 is split into Catalog / Configuration, and two new
 * slides carry what the earlier decks never showed — what is already built,
 * and where the product sits against the tools people already use.
 *
 *  1 Cover        centered; hero "≈4 000 000 · 14 · 1"
 *  2 Catalog      50/50; big catalog numbers left, cell-grid + hot cell right
 *  3 Configuration 52/48; the 14 axes as a chip field + 3 numbered pains
 *  4 Missing Half 50/50; quality bars scored nightly vs a giant "0"
 *  5 Why Now      46/54; three ✗ traps left, the crossed line right
 *  6 Fix          42/58; INTENT -> 3-stage chain -> 4 profile cards
 *  7 Already Built full-width 3x3 capability grid, "runs today" stamp
 *  8 Trust        54/46; 4 method checks + a mock card with a rejected row
 *  9 Where It Sits full-width 5-row comparison table
 * 10 Ask          centered; "1" command, URL pills, audience strip
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
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

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inline(s) {
  return esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, '<i class="t">$1</i>');
}

// One slide's raw lines -> the fields every renderer reads. `##` is the
// kicker (the small violet eyebrow), `###` the headline; everything else is
// a paragraph, a bullet, or a table row, in source order.
function extract(raw) {
  const o = { kicker: "", headline: "", paras: [], bullets: [], table: [], sections: [] };
  let section = null;
  for (const rawLine of raw) {
    const l = rawLine.trim();
    if (!l || l.startsWith("<!--")) continue;
    if (l.startsWith("### ")) {
      const t = l.slice(4).trim();
      if (!o.headline) o.headline = t;
      else { section = { title: t, lines: [] }; o.sections.push(section); }
      continue;
    }
    if (l.startsWith("## ")) { o.kicker = l.slice(3).trim(); continue; }
    if (l.startsWith("|") && l.endsWith("|")) {
      o.table.push(l.split("|").slice(1, -1).map((c) => c.trim()));
      continue;
    }
    if (l.startsWith("- ")) { o.bullets.push(l.slice(2).trim()); continue; }
    if (section) section.lines.push(l);
    else o.paras.push(l);
  }
  return o;
}

function parseSlides(md) {
  const slides = [];
  let cur = [];
  const flush = () => { if (cur.length) slides.push(cur); cur = []; };
  for (const line of md.split(/\r?\n/)) {
    if (line.trim() === "---") { flush(); continue; }
    cur.push(line);
  }
  flush();
  return slides.filter((s) => s.some((l) => l.trim() !== ""));
}

function dotGrid(w, h, step, r, color) {
  const dots = [];
  for (let x = step / 2; x < w; x += step)
    for (let y = step / 2; y < h; y += step)
      dots.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}"/>`);
  return `<svg class="bg-dots" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" fill="${color}">${dots.join("")}</svg>`;
}

function glowLayer(cls) {
  return `<div class="bg-layer">${dotGrid(1280, 720, 44, 1.3, "rgba(244,246,251,0.045)")}<div class="bg-glow ${cls}"></div></div>`;
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
  padding: 58px 84px 88px;
  display: flex; flex-direction: column;
  page-break-after: always; break-after: page;
  background: ${BG};
}
.slide:last-child { page-break-after: auto; break-after: auto; }
body.preview .slide { display: none; }
body.preview .slide:target { display: flex; }
b { color: ${INK}; font-weight: 700; }
i.t { font-style: normal; font-family: Consolas, 'Cascadia Mono', monospace; color: ${ACCENT}; }

.bg-layer { position: absolute; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.bg-glow { position: absolute; width: 840px; height: 840px; border-radius: 50%;
  background: radial-gradient(circle, rgba(124,92,255,0.16), transparent 62%); filter: blur(46px); }
.bg-glow.g-left { left: -220px; top: 40px; }
.bg-glow.g-right { right: -260px; top: 10px; }
.bg-glow.g-tr { right: -180px; top: -220px; }
.bg-glow.g-br { right: -240px; bottom: -260px; }
.bg-glow.g-center { left: 50%; top: 24%; transform: translateX(-50%); }

.shead { position: relative; z-index: 2; display: flex; flex-direction: column; gap: 13px; }
.kicker { display: flex; align-items: center; gap: 14px;
  font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-size: 14.5px; font-weight: 600;
  letter-spacing: 0.3em; text-transform: uppercase; color: ${ACCENT}; }
.kicker::before { content: ""; width: 30px; height: 2px; border-radius: 2px; background: ${ACCENT}; }
.headline { font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-weight: 600;
  font-size: 43px; line-height: 1.07; letter-spacing: -0.6px; max-width: 1140px; }

.sbody { position: relative; z-index: 2; flex: 1; display: flex; gap: 44px; margin-top: 24px; min-height: 0; }
.col { display: flex; flex-direction: column; justify-content: center; gap: 18px; min-width: 0; }

.card { background: linear-gradient(180deg, ${PANEL_A} 0%, ${PANEL_B} 100%);
  border: 1px solid ${BORDER}; border-radius: 18px; }
.chip { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
  padding: 6px 13px; border-radius: 999px; border: 1px solid #242C42; background: #13172A;
  font-size: 12.5px; font-weight: 600; color: #B9C2D8; }
.chip.vip { border-color: rgba(124,92,255,0.65); background: rgba(124,92,255,0.14); color: ${INK}; }
.chip .v { color: ${ACCENT}; font-weight: 700; }
.chip .ok { color: ${OK}; font-weight: 700; }

.pains { display: flex; flex-direction: column; gap: 14px; }
.pain { display: flex; gap: 15px; align-items: flex-start; }
.pn { flex: none; width: 42px; height: 42px; border-radius: 12px; margin-top: 1px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px;
  color: ${BG}; background: ${ACCENT}; }
.pain-body b { display: block; font-family: 'Bahnschrift', sans-serif; font-size: 19px;
  font-weight: 600; line-height: 1.16; color: ${INK}; }
.pain-body span { display: block; margin-top: 5px; font-size: 14px; color: ${MUTED}; line-height: 1.36; }

ul.list { list-style: none; display: flex; flex-direction: column; gap: 11px; }
ul.list li { position: relative; padding-left: 26px; font-size: 15.5px; line-height: 1.42; color: #C4CCDE; }
ul.list li::before { content: ""; position: absolute; left: 2px; top: 9px; width: 9px; height: 9px;
  border-radius: 3px; background: ${ACCENT}; }

.pfoot { position: absolute; left: 84px; right: 84px; bottom: 28px; z-index: 3;
  display: flex; justify-content: space-between; align-items: center;
  font-family: 'Bahnschrift', 'Segoe UI', sans-serif; font-size: 12px;
  letter-spacing: 0.14em; color: ${FAINT}; text-transform: uppercase; }
.pfoot .f-brand { font-weight: 700; letter-spacing: 0.22em; color: #39415A; }

/* 1 cover */
.slide.cover { align-items: center; justify-content: center; text-align: center; }
.cv-inner { position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 24px; }
.cv-tag { display: inline-flex; align-items: center; gap: 12px; padding: 9px 22px;
  border-radius: 999px; border: 1px solid #2A2F4A; background: rgba(124,92,255,0.08);
  font-family: 'Bahnschrift', sans-serif; font-size: 13.5px; font-weight: 600;
  letter-spacing: 0.26em; text-transform: uppercase; color: #B9C2D8; }
.cv-hero { display: flex; align-items: flex-start; justify-content: center; gap: 30px; }
.cvh { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.cvh .n { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 88px; line-height: 1;
  color: ${ACCENT}; font-feature-settings: 'tnum'; letter-spacing: -2px; white-space: nowrap; }
.cvh .l { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 14px;
  letter-spacing: 0.2em; text-transform: uppercase; color: ${MUTED}; }
.cv-sep { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 52px; color: #2C3350;
  line-height: 1.5; }
.cv-brand .t1 { font-family: 'Bahnschrift', sans-serif; font-weight: 700; font-size: 30px;
  letter-spacing: 0.3em; text-transform: uppercase; color: ${INK}; }
.cv-brand .t1 b { color: ${ACCENT}; }
.cv-brand .t2 { margin-top: 10px; font-size: 19px; color: ${MUTED}; }
.cv-foot { font-size: 14.5px; color: ${FAINT}; max-width: 780px; line-height: 1.5; }

/* 2 catalog */
.sbody.catalog { gap: 40px; margin-top: 30px; }
.catalog .col-left { width: 52%; justify-content: center; gap: 16px; }
.catalog .col-right { width: 48%; justify-content: center; align-items: center; gap: 20px; }
.bignums { display: flex; flex-direction: column; gap: 13px; }
.bignum { display: flex; align-items: baseline; gap: 16px; }
.bignum .bn { flex: none; min-width: 205px; text-align: right; font-family: 'Bahnschrift', sans-serif;
  font-weight: 300; font-size: 42px; line-height: 1.05; color: ${ACCENT}; font-feature-settings: 'tnum'; }
.bignum .bl { font-size: 15.5px; line-height: 1.3; color: #C4CCDE; }
.bignum .bl b { color: ${INK}; }
.cat-wrap { position: relative; width: 100%; max-width: 430px; }
.cat-label { position: absolute; top: -28px; right: 0; z-index: 3; }
.catnote { font-size: 14px; line-height: 1.45; color: ${FAINT}; max-width: 430px; }
.catnote b { color: ${MUTED}; }

/* 3 configuration */
.sbody.config { gap: 40px; margin-top: 22px; }
.config .col-left { width: 47%; justify-content: center; gap: 20px; }
.config .col-right { width: 53%; justify-content: center; gap: 14px; }
.axisfield { display: flex; flex-wrap: wrap; gap: 8px; }
.axisfield .chip { font-size: 12px; padding: 5px 12px; }
.axeshead { display: flex; align-items: center; gap: 16px; }
.axeshead .ah-n { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 60px;
  line-height: 1; color: ${ACCENT}; }
.axeshead .ah-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 19px;
  color: ${INK}; line-height: 1.2; }
.axeshead .ah-t span { display: block; font-family: 'Segoe UI', sans-serif; font-weight: 400;
  font-size: 13.5px; color: ${MUTED}; margin-top: 5px; }
.kicknote { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 19px;
  line-height: 1.35; color: ${INK}; }
.kicknote em { font-style: normal; color: ${ACCENT}; }
.kicknote b { color: ${INK}; }

/* 4 missing half */
.sbody.missing { margin-top: 24px; }
.missing .col-left { width: 50%; }
.missing .col-right { width: 50%; gap: 18px; }
.barpanel { padding: 20px 26px 18px; }
.barpanel .hp { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }
.barpanel .hp-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 14px;
  letter-spacing: 0.12em; text-transform: uppercase; color: ${MUTED}; }
.barpanel .hp-n { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 24px; color: ${INK}; }
.bars { display: flex; align-items: flex-end; gap: 13px; height: 72px; }
.bars .bar { flex: 1; border-radius: 5px 5px 2px 2px; background: currentColor; }
.zero-panel { padding: 22px 30px 24px; display: flex; align-items: center; gap: 28px; }
.zero-hero { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 172px;
  line-height: 0.85; color: ${ACCENT}; font-feature-settings: 'tnum'; }
.zero-side { display: flex; flex-direction: column; gap: 10px; }
.zero-side .zs-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px;
  letter-spacing: 0.06em; color: ${INK}; line-height: 1.25; }
.zero-metrics { display: flex; flex-wrap: wrap; gap: 8px; }

/* 5 why now */
.sbody.why { gap: 48px; margin-top: 26px; }
.why .col-left { width: 47%; justify-content: center; }
.why .col-right { width: 53%; justify-content: center; gap: 18px; }
.xrow { display: flex; gap: 14px; align-items: flex-start; }
.xmark { flex: none; width: 32px; height: 32px; margin-top: 2px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center; font-size: 15px;
  color: ${MUTED}; border: 1px solid ${BORDER}; background: ${PANEL_B}; }
.xbody { font-size: 16px; line-height: 1.4; color: #C4CCDE; }
.xbody b { color: ${INK}; }
.linepanel { padding: 24px 28px; }
.linepanel .lp-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 14px;
  letter-spacing: 0.14em; text-transform: uppercase; color: ${MUTED}; margin-bottom: 14px; }
.linepanel .lp-h { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 26px;
  color: ${INK}; line-height: 1.22; }
.linepanel .lp-h b { color: ${ACCENT}; }
.linepanel .lp-s { margin-top: 12px; font-size: 14.5px; color: ${MUTED}; line-height: 1.45; }
.linepanel .lp-bridge { margin-top: 14px; padding-left: 14px; border-left: 2px solid ${ACCENT};
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px;
  color: ${INK}; line-height: 1.35; }
.whychips { display: flex; flex-wrap: wrap; gap: 9px; }

/* 6 fix */
.sbody.fix { gap: 48px; margin-top: 26px; }
.fix .col-left { width: 41%; justify-content: center; gap: 20px; }
.fix .col-right { width: 59%; justify-content: center; gap: 18px; }
.pipe { display: flex; align-items: center; gap: 14px; }
.pipe-box { flex: 1; padding: 16px 18px; text-align: center; }
.pipe-box .pt { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 18px;
  letter-spacing: 0.1em; color: ${INK}; }
.pipe-box .pv { margin-top: 6px; font-size: 12.5px; color: ${MUTED}; line-height: 1.35; }
.pipe-arrow { flex: none; color: ${ACCENT}; font-size: 24px; line-height: 1; }
.profiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.prof { padding: 15px 13px 13px; text-align: center; position: relative; overflow: hidden; }
.prof::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: ${ACCENT}; opacity: 0.85; }
.prof .pr-i { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 11px;
  letter-spacing: 0.18em; color: ${ACCENT}; line-height: 1; }
.profcap { font-size: 13.5px; line-height: 1.45; color: ${MUTED}; text-align: center; }
.prof .pr-t { margin-top: 7px; font-family: 'Bahnschrift', sans-serif; font-weight: 600;
  font-size: 15px; color: ${INK}; line-height: 1.15; }
.prof .pr-s { margin-top: 7px; font-size: 11.5px; color: ${FAINT}; line-height: 1.3; }
.cmdline { padding: 13px 18px; border-radius: 12px; background: ${PANEL_B};
  border: 1px solid ${BORDER}; font-family: Consolas, 'Cascadia Mono', monospace;
  font-size: 12.5px; color: #9FB0D8; white-space: nowrap; overflow: hidden; }
.cmdline b { color: ${OK}; font-weight: 600; }

/* 7 already built */
.sbody.built { display: flex; flex-direction: column; justify-content: center; gap: 22px; margin-top: 22px; }
.builtgrid { position: relative; z-index: 2; display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 14px; }
.bcell { padding: 19px 20px 18px; display: flex; gap: 13px; align-items: flex-start; }
.bcell .bi { flex: none; width: 30px; height: 30px; border-radius: 9px; display: flex;
  align-items: center; justify-content: center; font-size: 13px;
  background: rgba(74,222,128,0.12); color: ${OK}; border: 1px solid rgba(74,222,128,0.3); }
.bcell .bt { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 16px;
  color: ${INK}; line-height: 1.15; }
.bcell .bs { margin-top: 5px; font-size: 12.5px; color: ${MUTED}; line-height: 1.35; }
.builtfoot { position: relative; z-index: 2; display: flex; align-items: center;
  justify-content: space-between; gap: 20px; }
.builtfoot .bf-t { font-size: 14.5px; color: ${FAINT}; line-height: 1.45; }
.builtfoot .bf-t b { color: ${MUTED}; }
.runstamp { flex: none; display: inline-flex; align-items: center; gap: 10px; padding: 9px 20px;
  border-radius: 999px; border: 1px solid rgba(74,222,128,0.55); background: rgba(74,222,128,0.08);
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 13px;
  letter-spacing: 0.18em; text-transform: uppercase; color: ${OK}; }

/* 8 trust */
.sbody.trust { gap: 46px; margin-top: 24px; }
.trust .col-left { width: 55%; justify-content: center; gap: 15px; }
.trust .col-right { width: 45%; justify-content: center; }
.checkrow { display: flex; gap: 14px; align-items: flex-start; }
.checkmark { flex: none; width: 25px; height: 25px; margin-top: 2px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
  color: #0C2A16; background: ${OK}; }
.checkbody { font-size: 15px; line-height: 1.42; color: #C4CCDE; }
.checkbody b { color: ${INK}; }
.mock { position: relative; width: 100%; max-width: 450px; padding: 20px 24px 16px; }
.mock .mh { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 13px; }
.mock .mh .mt { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 17px; color: ${INK}; }
.mock .mh .ms { font-size: 12px; color: ${FAINT}; letter-spacing: 0.06em; }
.mrow { display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; border-radius: 10px; margin-bottom: 7px;
  background: ${PANEL_B}; border: 1px solid ${BORDER}; }
.mrow .ml { font-size: 13.5px; color: ${MUTED}; }
.mrow .mv { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 18px; color: ${INK}; }
.mrow.good { border-color: rgba(74,222,128,0.4); }
.mrow.good .mv { color: ${OK}; }
.mrow.bad { border-color: rgba(124,92,255,0.45); }
.mrow.bad .ml { color: ${FAINT}; text-decoration: line-through; }
.mrow.bad .mv { color: ${FAINT}; }
.tag-rej { font-family: 'Bahnschrift', sans-serif; font-size: 11px; font-weight: 600;
  letter-spacing: 0.14em; color: ${ACCENT}; border: 1px solid rgba(124,92,255,0.6);
  padding: 3px 10px; border-radius: 999px; display: inline-block; }
.mfoot { display: flex; align-items: center; justify-content: space-between; margin-top: 12px; gap: 12px; }
.mhash { font-family: Consolas, 'Cascadia Mono', monospace; font-size: 11px;
  letter-spacing: 0.04em; color: ${FAINT}; }
.mhash b { color: ${OK}; font-weight: 600; }
.stamp { flex: none; width: 78px; height: 78px; border-radius: 50%; border: 3px solid ${OK}; opacity: 0.9;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1px;
  transform: rotate(8deg); }
.stamp .st-v { font-family: 'Bahnschrift', sans-serif; font-weight: 700; font-size: 12px;
  letter-spacing: 0.12em; color: ${OK}; }
.stamp .st-s { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 9.5px;
  letter-spacing: 0.14em; color: ${OK}; opacity: 0.85; }

/* 9 where it sits */
.sbody.sits { display: flex; flex-direction: column; justify-content: center; margin-top: 20px; }
.ctable { position: relative; z-index: 2; width: 100%; border-collapse: separate;
  border-spacing: 0 8px; }
.ctable th { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 12px;
  letter-spacing: 0.16em; text-transform: uppercase; color: ${FAINT}; text-align: left;
  padding: 0 20px 6px; }
.ctable td { padding: 15px 20px; background: ${PANEL_B}; border-top: 1px solid ${BORDER};
  border-bottom: 1px solid ${BORDER}; font-size: 14.5px; color: #C4CCDE; vertical-align: middle; }
.ctable td:first-child { border-left: 1px solid ${BORDER}; border-radius: 12px 0 0 12px;
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 16px; color: ${INK}; width: 27%; }
.ctable td:last-child { border-right: 1px solid ${BORDER}; border-radius: 0 12px 12px 0; color: ${MUTED}; }
.ctable tr.us td { background: rgba(124,92,255,0.11); border-color: rgba(124,92,255,0.5); }
.ctable tr.us td:first-child { color: ${ACCENT}; }
.ctable tr.us td { color: ${INK}; }
.sitsfoot { position: relative; z-index: 2; margin-top: 16px;
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 21px; color: ${INK}; }
.sitsfoot b { color: ${ACCENT}; }

/* 10 ask */
/* The header stays left-aligned like every other slide; only the ask block
   below it centres, so slide 10 reads as the same deck rather than a poster. */
.awrap { position: relative; z-index: 2; flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 28px; margin-top: 10px; }
.a-numrow { display: flex; align-items: baseline; gap: 24px; }
.a-one { font-family: 'Bahnschrift', sans-serif; font-weight: 300; font-size: 158px;
  line-height: 0.88; color: ${ACCENT}; font-feature-settings: 'tnum'; }
.a-lab { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 36px; color: ${INK}; }
.a-lab span { display: block; font-family: 'Segoe UI', sans-serif; font-weight: 400;
  font-size: 15.5px; color: ${MUTED}; margin-top: 8px; }
.a-urls { display: flex; gap: 14px; }
.a-url { display: inline-flex; align-items: center; gap: 10px; padding: 12px 26px;
  border-radius: 14px; border: 1px solid rgba(124,92,255,0.6); background: rgba(124,92,255,0.1);
  font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 22px; color: ${INK}; }
.a-url .u { color: ${ACCENT}; }
.adirs { display: flex; gap: 16px; width: 100%; max-width: 1060px; }
.adir { flex: 1; padding: 16px 20px; }
.adir .ad-n { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 14px;
  color: ${ACCENT}; letter-spacing: 0.06em; }
.adir .ad-t { font-family: 'Bahnschrift', sans-serif; font-weight: 600; font-size: 16.5px;
  color: ${INK}; margin-top: 6px; line-height: 1.2; }
.adir .ad-s { font-size: 12.5px; color: ${MUTED}; margin-top: 6px; line-height: 1.4; }
`;

// A sparse cell grid standing in for the catalog, with one violet "hot" cell:
// the config that is right for this box, unlabeled among the rest.
function catalogGrid(w, h, step) {
  const cols = Math.floor(w / step);
  const rows = Math.floor(h / step);
  const hotC = Math.floor(cols * 0.62);
  const hotR = Math.floor(rows * 0.38);
  const cells = [];
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * step + 1;
      const y = r * step + 1;
      const s = step - 4;
      if (r === hotR && c === hotC) continue;
      const a = 0.05 + rnd() * 0.09;
      cells.push(`<rect x="${x}" y="${y}" width="${s}" height="${s}" rx="3" fill="rgba(244,246,251,${a.toFixed(3)})"/>`);
    }
  }
  const hx = hotC * step + 1;
  const hy = hotR * step + 1;
  const hs = step - 4;
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  ${cells.join("")}
  <rect x="${hx - 4}" y="${hy - 4}" width="${hs + 8}" height="${hs + 8}" rx="6" fill="none" stroke="${ACCENT}" stroke-width="2"/>
  <rect x="${hx}" y="${hy}" width="${hs}" height="${hs}" rx="3" fill="${ACCENT}"/>
</svg>`;
}

function coverSlide(o, index, total) {
  const tag = o.paras[0] || "";
  const brand = o.paras.find((p) => p.includes("LlamaToaster")) || "";
  const foot = o.paras[o.paras.length - 1] || "";
  // The hero numbers are the deck's thesis in three tokens: the size of the
  // haystack, the number of axes that actually get swept, the one answer out.
  const heroes = [
    { n: "≈4 000 000", l: "GGUF files" },
    { n: "14", l: "swept axes" },
    { n: "1", l: "answer" },
  ];
  const heroHtml = heroes
    .map((h) => `<div class="cvh"><div class="n">${h.n}</div><div class="l">${h.l}</div></div>`)
    .join(`<div class="cv-sep">·</div>`);
  const brandParts = brand.split("—");
  return `<section class="slide cover" id="slide-${index + 1}">
${glowLayer("g-center")}
  <div class="cv-inner">
    <div class="cv-tag">${inline(tag)}</div>
    <div class="cv-hero">${heroHtml}</div>
    <div class="cv-brand">
      <div class="t1">Llama<b>Toaster</b></div>
      <div class="t2">${inline((brandParts[1] || "the appliance that finds it.").trim())}</div>
    </div>
    <div class="cv-foot">${inline(foot)}</div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function catalogSlide(o, index, total) {
  // Numbers come from the markdown so the deck stays the single source of
  // truth for every claim someone might ask a source for.
  const nums = o.paras
    .filter((p) => /^[~≈\d]/.test(p))
    .slice(0, 3)
    .map((p) => {
      const m = p.match(/^([~≈]?\s?[\d\s]+)(.*)$/);
      return m ? { n: m[1].trim(), l: m[2].trim() } : { n: "", l: p };
    });
  const numHtml = nums
    .map((x) => `<div class="bignum"><div class="bn">${esc(x.n)}</div><div class="bl">${inline(x.l)}</div></div>`)
    .join("");
  const archLine = o.paras.find((p) => p.startsWith("10+")) || "";
  const tail = o.paras.filter((p) => !/^[~≈\d]/.test(p));
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-left")}
${headerBlock(o)}
  <div class="sbody catalog">
    <div class="col col-left">
      <div class="bignums">${numHtml}</div>
      <div class="chip vip"><span class="v">10+</span> ${inline(archLine.replace(/^10\+\s*architectures:\s*/, ""))}</div>
    </div>
    <div class="col col-right">
      <div class="cat-wrap">
        ${catalogGrid(430, 250, 30)}
        <span class="chip vip cat-label"><span class="v">✦</span> the right one for your box</span>
      </div>
      <div class="catnote">${tail.map((t) => inline(t)).join("<br>")}</div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function configSlide(o, index, total) {
  const AXES = [
    "context", "offload layers", "KV cache K ×9", "KV cache V ×9", "flash attention",
    "batch", "ubatch", "threads", "prefill depth", "concurrency",
    "MoE CPU offload", "speculative decoding", "draft offload", "llama.cpp build",
  ];
  const axisHtml = AXES.map((a, i) =>
    `<span class="chip${i === 2 || i === 10 ? " vip" : ""}">${esc(a)}</span>`
  ).join("");
  // "»" marks the section's closing punch -- without it a trailing line is
  // indistinguishable from more body copy for the pain above it.
  const pains = [];
  const closers = [];
  for (const s of o.sections) {
    for (const l of s.lines) {
      if (l.startsWith("»")) { closers.push(l.slice(1).trim()); continue; }
      const m = l.match(/^([❶❷❸])\s*(.*)$/);
      if (m) pains.push({ n: m[1], t: m[2], s: "" });
      else if (pains.length) pains[pains.length - 1].s += (pains[pains.length - 1].s ? " " : "") + l;
    }
  }
  const painHtml = pains
    .map((p, i) => `<div class="pain">
  <div class="pn">${String(i + 1).padStart(2, "0")}</div>
  <div class="pain-body"><b>${inline(p.t)}</b><span>${inline(p.s)}</span></div>
</div>`)
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-tr")}
${headerBlock(o)}
  <div class="sbody config">
    <div class="col col-left">
      <div class="axeshead">
        <div class="ah-n">14</div>
        <div class="ah-t">axes, cross-multiplied<span>on every run, on every machine, for every model</span></div>
      </div>
      <div class="axisfield">${axisHtml}</div>
      <div class="kicknote">${closers.map((c) => inline(c)).join("<br>")}</div>
    </div>
    <div class="col col-right">
      <div class="pains">${painHtml}</div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function missingSlide(o, index, total) {
  const bars = [38, 52, 44, 70, 58, 82, 64]
    .map((h, i) => `<div class="bar" style="height:${h}%;color:${i === 5 ? ACCENT : "rgba(124,92,255,0.35)"}"></div>`)
    .join("");
  const lines = o.paras;
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-tr")}
${headerBlock(o)}
  <div class="sbody missing">
    <div class="col col-left">
      <ul class="list">
        <li>${inline(lines[0] || "")}<br><span style="color:${FAINT}">${inline(lines[1] || "")}</span></li>
        <li>${inline(lines[2] || "")}<br><b>${inline(lines[3] || "")}</b></li>
      </ul>
      <p style="font-size:15px;line-height:1.5;color:${FAINT}">${inline(lines[4] || "")}<br>${inline(lines[5] || "")}</p>
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
            <span class="chip">TTFT</span>
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
  const traps = o.paras
    .filter((p) => p.startsWith("✗"))
    .map((p) => {
      const body = p.replace(/^✗\s*/, "");
      const [t, s] = body.split("→");
      return { t: (t || "").trim(), s: s ? `→ ${s.trim()}` : "" };
    });
  const trapHtml = traps
    .map((x) => `<div class="xrow"><div class="xmark">✗</div><div class="xbody"><b>${inline(x.t)}</b><br>${inline(x.s)}</div></div>`)
    .join("");
  const head = o.paras.filter((p) => !p.startsWith("»")).slice(0, 5);
  // "»" marks the bridge line into the product -- the one sentence on this
  // slide that is about LlamaToaster rather than about the model landscape.
  const bridge = o.paras.filter((p) => p.startsWith("»")).map((p) => p.slice(1).trim());
  const tail = o.paras.filter((p) => p.startsWith("The missing piece"));
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-br")}
${headerBlock(o)}
  <div class="sbody why">
    <div class="col col-left">
      <p style="font-size:15px;line-height:1.55;color:#C4CCDE;margin-bottom:6px">${inline(head[3] || "")}<br><span style="color:${FAINT}">${inline(head[4] || "")}</span></p>
      ${trapHtml}
    </div>
    <div class="col col-right">
      <div class="card linepanel">
        <div class="lp-t">The line models just crossed</div>
        <div class="lp-h">${inline(head[0] || "")}<br><b>${inline(head[1] || "")}</b></div>
        <div class="lp-s">${inline(head[2] || "")}</div>
        <div class="lp-bridge">${bridge.map((t) => inline(t)).join("<br>")}</div>
        <div class="lp-s">${tail.map((t) => inline(t)).join("<br>")}</div>
      </div>
      <div class="whychips">
        <span class="chip">no outages</span>
        <span class="chip">no refusals</span>
        <span class="chip">no price changes</span>
        <span class="chip vip"><span class="v">✦</span> nothing leaves your box</span>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function fixSlide(o, index, total) {
  const stages = [
    { t: "TUNING", v: "find the batch/ubatch that this box actually likes" },
    { t: "REFINE", v: "narrow to the neighbourhood worth measuring properly" },
    { t: "SWEEP", v: "the real grid, repeated, with stddev that means something" },
  ];
  const pipeHtml = stages
    .map((s) => `<div class="card pipe-box"><div class="pt">${s.t}</div><div class="pv">${esc(s.v)}</div></div>`)
    .join(`<div class="pipe-arrow">→</div>`);
  // Short metric labels rather than emoji: the pictographs that fit these four
  // cards are exactly the ones Bahnschrift/Segoe UI have no glyph for, and a
  // tofu box on a pitch slide is worse than no icon at all.
  const profs = [
    { i: "TOK/S", t: "Max Speed", s: "highest tok/s that clears the stability gate" },
    { i: "MIX", t: "Balanced", s: "speed against context, weighted by your workload" },
    { i: "CTX", t: "Max Context", s: "the largest ceiling a probe actually verified" },
    { i: "VRAM", t: "Low Memory", s: "leaves the most headroom on the card" },
  ];
  const profHtml = profs
    .map((p) => `<div class="card prof"><div class="pr-i">${p.i}</div><div class="pr-t">${esc(p.t)}</div><div class="pr-s">${esc(p.s)}</div></div>`)
    .join("");
  const left = o.paras;
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-left")}
${headerBlock(o)}
  <div class="sbody fix">
    <div class="col col-left">
      <p style="font-size:15.5px;line-height:1.5;color:#C4CCDE">${inline(left[0] || "")} ${inline(left[1] || "")}</p>
      <p style="font-size:15.5px;line-height:1.5;color:#C4CCDE"><b>${inline(left[2] || "")}</b><br>${inline(left[3] || "")} <span style="color:${ACCENT}">${inline(left[4] || "")}</span></p>
      <p style="font-family:'Bahnschrift',sans-serif;font-weight:600;font-size:25px;color:${INK};line-height:1.25">Intent in.<br><span style="color:${ACCENT}">Decision out.</span></p>
      <p style="font-size:14px;line-height:1.45;color:${FAINT}">${inline(left[8] || "")}</p>
    </div>
    <div class="col col-right">
      <div class="pipe">${pipeHtml}</div>
      <div class="profiles">${profHtml}</div>
      <div class="profcap">${inline(left[5] || "")} ${inline(left[6] || "")}</div>
      <div class="cmdline">llama-bench -m model.gguf -fa on -ctk q8_0 -ctv q8_0 <b>-ngl 31</b> -b 512 -ub 512 -d 4096</div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function builtSlide(o, index, total) {
  // Each cell is one shipped surface, taken verbatim from the markdown's
  // "⚙ Name — description" lines so the slide can never drift from the copy.
  const cells = o.paras
    .filter((p) => p.startsWith("⚙"))
    .map((p) => {
      const body = p.replace(/^⚙\s*/, "");
      const dash = body.indexOf("—");
      return dash === -1
        ? { t: body, s: "" }
        : { t: body.slice(0, dash).trim(), s: body.slice(dash + 1).trim() };
    });
  const cellHtml = cells
    .map((c) => `<div class="card bcell"><div class="bi">✓</div><div><div class="bt">${inline(c.t)}</div><div class="bs">${inline(c.s)}</div></div></div>`)
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-br")}
${headerBlock(o)}
  <div class="sbody built">
    <div class="builtgrid">${cellHtml}</div>
    <div class="builtfoot">
      <div class="bf-t">Nine surfaces, one queue, <b>one command to attach a machine</b>.<br>Every one of these is in the repository today — none of it is a promise.</div>
      <div class="runstamp">Runs today</div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function trustSlide(o, index, total) {
  // "✓" opens a check; every unprefixed line after it belongs to that check.
  // Grouping on sentence shape instead would silently drop or merge a check the
  // moment the copy is rewrapped.
  const checks = [];
  for (const p of o.paras) {
    if (p.startsWith("✓")) checks.push(p.slice(1).trim());
    else if (checks.length) checks[checks.length - 1] += " " + p;
  }
  const checkHtml = checks
    .map((c) => `<div class="checkrow"><div class="checkmark">✓</div><div class="checkbody">${inline(c)}</div></div>`)
    .join("");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-right")}
${headerBlock(o)}
  <div class="sbody trust">
    <div class="col col-left">${checkHtml}</div>
    <div class="col col-right">
      <div class="card mock">
        <div class="mh">
          <div class="mt">Max Speed</div>
          <div class="ms">method v3 · 5 repeats</div>
        </div>
        <div class="mrow good"><span class="ml">tg &nbsp;q8_0/q8_0 &nbsp;ngl 31</span><span class="mv">42.7 tok/s</span></div>
        <div class="mrow"><span class="ml">pp &nbsp;q8_0/q8_0 &nbsp;ngl 31</span><span class="mv">918 tok/s</span></div>
        <div class="mrow bad"><span class="ml">tg &nbsp;f16/f16 &nbsp;ngl 33</span><span class="mv">1.0e6 tok/s</span></div>
        <div class="mfoot">
          <div>
            <span class="tag-rej">rejected · suspect samples</span>
            <div class="mhash" style="margin-top:9px">config_hash <b>a91f2c7e…</b> · ctx 16 384 verified</div>
          </div>
          <div class="stamp"><div class="st-v">VERIFIED</div><div class="st-s">PROBED</div></div>
        </div>
      </div>
    </div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function sitsSlide(o, index, total) {
  const rows = o.table.filter((r) => !r.every((c) => /^-+:?$|^:?-+:?$/.test(c)));
  const head = rows[0] || ["Tool", "What it gives you", "What it never answers"];
  const body = rows.slice(1);
  const bodyHtml = body
    .map((r) => {
      const us = r[0].includes("**");
      const cells = r.map((c) => `<td>${inline(c.replace(/\*\*/g, ""))}</td>`).join("");
      return `<tr class="${us ? "us" : ""}">${cells}</tr>`;
    })
    .join("");
  const closer = o.paras.find((p) => p.startsWith("The gap")) || "";
  const [a, b] = closer.split(".");
  return `<section class="slide" id="slide-${index + 1}">
${glowLayer("g-tr")}
${headerBlock(o)}
  <div class="sbody sits">
    <table class="ctable">
      <thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${bodyHtml}</tbody>
    </table>
    <div class="sitsfoot">${esc((a || "").trim())}. <b>${esc((b || "").trim())}.</b></div>
  </div>
${footerBlock(index, total)}
</section>`;
}

function askSlide(o, index, total) {
  const dirs = [
    { n: "01", t: "Attach a machine", s: "One command. Pull-only — nothing to port-forward, no manual llama.cpp build." },
    { n: "02", t: "Share your numbers", s: "Opt in and your results join the k-anonymised community set — never under 5 contributors." },
    { n: "03", t: "Star it while it's early", s: "Solo GPU owners, mixed-hardware teams, and anyone publishing reproducible numbers." },
  ];
  const dirHtml = dirs
    .map((d) => `<div class="card adir"><div class="ad-n">${d.n}</div><div class="ad-t">${esc(d.t)}</div><div class="ad-s">${esc(d.s)}</div></div>`)
    .join("");
  return `<section class="slide ask" id="slide-${index + 1}">
${glowLayer("g-center")}
${headerBlock(o)}
  <div class="awrap">
    <div class="a-numrow">
      <div class="a-one">1</div>
      <div class="a-lab">command connects your rig<span>a spare GPU box, an old laptop, or the CPU on your VPS</span></div>
    </div>
    <div class="a-urls">
      <div class="a-url">github.com/<span class="u">noname9006/LlamaToaster</span></div>
      <div class="a-url"><span class="u">llamatoaster.com</span></div>
    </div>
    <div class="adirs">${dirHtml}</div>
  </div>
${footerBlock(index, total)}
</section>`;
}

const RENDERERS = [
  coverSlide, catalogSlide, configSlide, missingSlide, whySlide,
  fixSlide, builtSlide, trustSlide, sitsSlide, askSlide,
];

function buildHtml(mdText) {
  const slides = parseSlides(mdText);
  const body = slides
    .map((raw, i) => {
      const o = extract(raw);
      const fn = RENDERERS[i] || coverSlide;
      return fn(o, i, slides.length);
    })
    .join("\n");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>LlamaToaster — pitch deck</title>
<style>${CSS}</style>
</head><body>
${body}
<script>
(function () {
  // #slide-N renders that one slide alone at exactly 1280x720, which is what
  // the --shots path screenshots. Without the hash every slide stacks, which
  // is what the print-to-PDF path needs.
  if (location.hash && location.hash.indexOf("slide-") === 1) {
    document.body.classList.add("preview");
  }
})();
</script>
</body></html>`;
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

function runBrowser(extra, url) {
  const browser = findBrowser();
  if (!browser) {
    console.error("No Edge/Chrome found. Install Microsoft Edge or edit BROWSER_CANDIDATES.");
    process.exit(1);
  }
  const profile = join(tmpdir(), `lt-pitch-v3-${Date.now()}-${Math.round(Math.random() * 1e6)}`);
  const res = launch(browser, profile, extra, url);
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
  if (res.error) {
    console.error(`browser spawn failed (${res.error.code}): ${browser}`);
    process.exit(1);
  }
  if (res.status !== 0) {
    console.error((res.stderr || res.stdout || "browser failed").slice(0, 2000));
    process.exit(res.status ?? 1);
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
    ["--no-pdf-header-footer", "--print-to-pdf-no-header", `--print-to-pdf=${resolve(outPdf)}`],
    pathToFileURL(htmlPath).href
  );
  if (!existsSync(outPdf)) {
    console.error("PDF was not produced.");
    process.exit(1);
  }
}

function screenshotSlide(htmlPath, outPng, n) {
  runBrowser(
    ["--window-size=1280,720", "--virtual-time-budget=1500", `--screenshot=${resolve(outPng)}`],
    `${pathToFileURL(htmlPath).href}#slide-${n}`
  );
  if (!existsSync(outPng)) {
    console.error(`Screenshot not produced: ${outPng}`);
    process.exit(1);
  }
}

function writeTempHtml(mdText) {
  const tmpHtml = join(tmpdir(), `lt-pitch-v3-${Date.now()}.html`);
  writeFileSync(tmpHtml, buildHtml(mdText));
  return tmpHtml;
}

function main() {
  const args = process.argv.slice(2);

  if (args[0] === "--html-only" && args.length >= 3) {
    const mdText = readFileSync(resolve(args[1]), "utf8");
    writeFileSync(resolve(args[2]), buildHtml(mdText));
    console.log(`HTML written: ${resolve(args[2])}`);
    return;
  }

  if (args[0] === "--shots" && args.length >= 3) {
    const mdText = readFileSync(resolve(args[1]), "utf8");
    const tmpHtml = writeTempHtml(mdText);
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
    try { rmSync(tmpHtml, { force: true }); } catch {}
    console.log(`slides rendered: ${total}`);
    return;
  }

  if (args.length >= 2) {
    const inPath = resolve(args[0]);
    const outPdf = resolve(args[1]);
    const mdText = readFileSync(inPath, "utf8");
    const tmpHtml = writeTempHtml(mdText);
    console.log(`Rendering ${basename(inPath)} -> ${outPdf} (${parseSlides(mdText).length} slides)`);
    printPdf(tmpHtml, outPdf);
    try { rmSync(tmpHtml, { force: true }); } catch {}
    console.log(`PDF written: ${outPdf}`);
    return;
  }

  console.log(`Usage:
  node scripts/pitch-to-pdf-v3.mjs <in.md> <out.pdf>
  node scripts/pitch-to-pdf-v3.mjs --html-only <in.md> <out.html>
  node scripts/pitch-to-pdf-v3.mjs --shots <in.md> <outDir> [slide numbers...]`);
}

main();
