#!/usr/bin/env node
/**
 * LlamaToaster — pitch deck builder.
 * Renders a Markdown deck to a dark 16:9 HTML + PDF (via headless Edge/Chrome).
 * Zero npm dependencies.
 *
 * Usage:
 *   node scripts/pitch-to-pdf.mjs <in.md> <out.pdf>
 *   node scripts/pitch-to-pdf.mjs --html-only <in.md> <out.html>
 *   node scripts/pitch-to-pdf.mjs --shots <in.md> <outDir> [slide numbers...]
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";

const CSS = `
@page { size: 1280px 720px; margin: 0; }
* { margin: 0; padding: 0; box-sizing: border-box; }
html { background: #0b0e14; }
body { background: #0b0e14; font-family: 'Segoe UI', 'Segoe UI Variable Text', 'Helvetica Neue', Arial, sans-serif; color: #e8eef6; }
.slide {
  position: relative;
  width: 1280px; height: 720px;
  padding: 78px 100px 92px;
  overflow: hidden;
  display: flex; flex-direction: column; justify-content: center; gap: 20px;
  page-break-after: always; break-after: page;
  background:
    radial-gradient(1100px 480px at 84% -14%, rgba(240,165,74,0.17), transparent 62%),
    radial-gradient(760px 420px at -10% 110%, rgba(77,214,181,0.07), transparent 55%),
    linear-gradient(155deg, #141a29 0%, #0b0e16 72%, #0a0d12 100%);
}
.slide:last-child { page-break-after: auto; break-after: auto; }
.kicker {
  font-size: 21px; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase;
  color: #f0a54a;
}
.headline { font-size: 50px; font-weight: 700; line-height: 1.16; letter-spacing: -0.01em; color: #f4f7fb; max-width: 1080px; }
.headline strong { color: #ffc773; }
.lead {
  font-size: 28px; line-height: 1.34; color: #f3e7d3;
  border-left: 6px solid #f0a54a; padding: 4px 0 4px 26px; margin: 2px 0; max-width: 1060px;
}
.lead strong { color: #ffd9a0; }
ul.bullets { list-style: none; display: flex; flex-direction: column; gap: 12px; max-width: 1080px; }
ul.bullets li { position: relative; padding-left: 30px; font-size: 21px; line-height: 1.4; color: #d3dcea; }
ul.bullets li::before {
  content: ""; position: absolute; left: 2px; top: 12px; width: 10px; height: 10px;
  border-radius: 3px; background: linear-gradient(135deg, #ffc773, #e08a2e);
}
ul.bullets li strong { color: #ffffff; }
ul.bullets li em { color: #f2d8b8; }
code {
  font-family: Consolas, 'Cascadia Mono', Menlo, monospace; font-size: 0.88em;
  background: rgba(255,255,255,0.08); padding: 1px 8px; border-radius: 6px; color: #ffd9a0;
}
.stats { display: flex; gap: 30px; margin-top: 6px; }
.stat {
  flex: 1; text-align: center; padding: 34px 24px 30px;
  background: linear-gradient(150deg, #1b2330 0%, #121825 100%);
  border: 1px solid rgba(255,255,255,0.09); border-radius: 20px;
  box-shadow: 0 16px 44px rgba(0,0,0,0.38);
}
.stat-num { font-size: 56px; font-weight: 800; color: #ffc773; font-feature-settings: 'tnum'; letter-spacing: -0.01em; }
.stat-label { font-size: 19px; color: #93a1b5; margin-top: 10px; line-height: 1.3; }
.cover { text-align: center; align-items: center; }
.cover .kicker { letter-spacing: 0.3em; }
.cover .headline { font-size: 66px; line-height: 1.1; }
.cover .lead { border-left: none; padding: 0; font-size: 33px; color: #ffcf8f; }
.cover ul.bullets { align-items: center; }
footer {
  position: absolute; left: 100px; right: 100px; bottom: 32px; z-index: 2;
  display: flex; justify-content: space-between; align-items: center;
  font-size: 14px; letter-spacing: 0.05em; color: rgba(147,161,181,0.55);
}
footer .brand { font-weight: 700; color: rgba(240,165,74,0.5); letter-spacing: 0.08em; text-transform: uppercase; font-size: 13px; }
body.preview .slide { display: none; }
body.preview .slide:target { display: flex; }
`;

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(s) {
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
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

function renderStat(q) {
  const m = q.match(/\*\*([^*]+)\*\*/);
  const num = (m ? m[1] : q).trim();
  const label = m ? q.slice(m.index + m[0].length).trim() : "";
  return (
    `<div class="stat"><div class="stat-num">${inline(num)}</div>` +
    (label ? `<div class="stat-label">${inline(label)}</div>` : "") +
    `</div>`
  );
}

function renderSlide(raw, index, total) {
  const isStats = raw.some((l) => l.includes("layout: stats"));
  const kicker = raw.find((l) => l.startsWith("# "))?.slice(2).trim() ?? "";
  const headline = raw.find((l) => l.startsWith("## "))?.slice(3).trim() ?? "";
  const quotes = raw.filter((l) => l.startsWith("> ")).map((l) => l.slice(2).trim());
  const bullets = raw.filter((l) => l.startsWith("- ")).map((l) => l.slice(2).trim());

  const parts = [];
  if (isStats) {
    parts.push(`<div class="stats">${quotes.map(renderStat).join("")}</div>`);
    if (bullets.length) {
      parts.push(`<ul class="bullets">${bullets.map((b) => `<li>${inline(b)}</li>`).join("")}</ul>`);
    }
  } else {
    if (quotes.length) {
      parts.push(`<blockquote class="lead">${quotes.map(inline).join("<br>")}</blockquote>`);
    }
    if (bullets.length) {
      parts.push(`<ul class="bullets">${bullets.map((b) => `<li>${inline(b)}</li>`).join("")}</ul>`);
    }
  }

  const classes = ["slide", index === 0 ? "cover" : "", isStats ? "stats-layout" : ""].filter(Boolean).join(" ");
  return [
    `<section class="${classes}" id="slide-${index + 1}">`,
    `  <div class="kicker">${inline(kicker)}</div>`,
    `  <h1 class="headline">${inline(headline)}</h1>`,
    ...parts.map((p) => `  ${p}`),
    `  <footer>`,
    `    <div class="brand">LlamaToaster</div>`,
    `    <div class="pagenum">${index + 1} / ${total}</div>`,
    `  </footer>`,
    `</section>`,
  ].join("\n");
}

function buildHtml(mdText, lang) {
  const slides = parseSlides(mdText);
  const body = slides.map((s, i) => renderSlide(s, i, slides.length)).join("\n");
  return `<!DOCTYPE html>
<html lang="${lang}">
<head>
<meta charset="utf-8">
<title>LlamaToaster — Pitch Deck</title>
<style>${CSS}</style>
</head>
<body>
${body}
<script>if (location.hash) document.body.classList.add("preview");</script>
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
  const profile = join(tmpdir(), `lt-pitch-${Date.now()}-${Math.round(Math.random() * 1e6)}`);
  const args = [
    "--headless",
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--hide-scrollbars",
    ...extra,
    url,
    `--user-data-dir=${profile}`,
  ];
  const res = spawnSync(browser, args, { stdio: ["ignore", "pipe", "pipe"], encoding: "utf8", timeout: 120_000 });
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

function printPdf(htmlPath, outPdf) {
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

function main() {
  const args = process.argv.slice(2);

  if (args[0] === "--html-only" && args.length >= 3) {
    const mdText = readFileSync(resolve(args[1]), "utf8");
    const out = resolve(args[2]);
    writeFileSync(out, buildHtml(mdText, langFrom(args[1])));
    console.log(`HTML written: ${out}`);
    return;
  }

  if (args[0] === "--shots" && args.length >= 3) {
    const inPath = resolve(args[1]);
    const mdText = readFileSync(inPath, "utf8");
    const html = buildHtml(mdText, langFrom(inPath));
    const tmpHtml = join(tmpdir(), `lt-pitch-${Date.now()}.html`);
    writeFileSync(tmpHtml, html);
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
    const html = buildHtml(mdText, langFrom(inPath));
    const tmpHtml = join(tmpdir(), `lt-pitch-${Date.now()}.html`);
    writeFileSync(tmpHtml, html);
    console.log(`Rendering ${basename(inPath)} -> ${outPdf} (${parseSlides(mdText).length} slides)`);
    printPdf(tmpHtml, outPdf);
    try {
      rmSync(tmpHtml, { force: true });
    } catch {}
    console.log(`PDF written: ${outPdf}`);
    return;
  }

  console.log(`Usage:
  node scripts/pitch-to-pdf.mjs <in.md> <out.pdf>
  node scripts/pitch-to-pdf.mjs --html-only <in.md> <out.html>
  node scripts/pitch-to-pdf.mjs --shots <in.md> <outDir> [slide numbers...]`);
}

main();