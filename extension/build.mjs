// Builds the LlamaToaster Chrome extension into extension/dist (load that
// folder with chrome://extensions -> "Load unpacked").
//
//   node extension/build.mjs          one build
//   node extension/build.mjs --watch  rebuild on change
//
// The fit logic is the app's own: shared/gguf.ts, shared/fitSimulate.ts and
// shared/fitLevel.ts are bundled straight from the repo, so the extension and
// the web UI give the same answer for the same header and memory.

import * as esbuild from "esbuild";
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const out = join(here, "dist");
const watch = process.argv.includes("--watch");

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "icons"), { recursive: true });
mkdirSync(join(out, "fonts"), { recursive: true });

for (const f of ["manifest.json", "popup.html", "popup.css", "content.css"]) cpSync(join(here, f), join(out, f));
// The side panel is the popup page, marked so it fills the panel's width.
writeFileSync(join(out, "panel.html"), readFileSync(join(here, "popup.html"), "utf8").replace("<body>", '<body class="panel">'));
cpSync(join(root, "client", "public", "toaster_favicon.png"), join(out, "icons", "icon.png"));
cpSync(join(root, "client", "public", "logo.png"), join(out, "icons", "logo.png"));
cpSync(join(here, "icons", "llama.png"), join(out, "icons", "llama.png"));
cpSync(join(here, "icons", "wordmark.png"), join(out, "icons", "wordmark.png"));

const fontDir = join(root, "client", "node_modules", "@fontsource");
const fonts = {
  "barlow-400.woff2": "barlow/files/barlow-latin-400-normal.woff2",
  "barlow-500.woff2": "barlow/files/barlow-latin-500-normal.woff2",
  "barlow-600.woff2": "barlow/files/barlow-latin-600-normal.woff2",
  "barlow-condensed-600.woff2": "barlow-condensed/files/barlow-condensed-latin-600-normal.woff2",
  "plex-mono-400.woff2": "ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2",
};
for (const [name, src] of Object.entries(fonts)) cpSync(join(fontDir, src), join(out, "fonts", name));

// The toaster drawn on Hugging Face pages: the "Option 1 minimal" 32x32 icon
// (icons/toaster.svg embeds it as PNG, cut from the design sheet); its dial is a
// circle-clipped copy in the .lt-knob group, which turns while headers are read.
function toasterSvg() {
  const svg = readFileSync(join(here, "icons", "toaster.svg"), "utf8");
  if (!svg.includes('class="lt-knob"')) throw new Error("toaster.svg: the dial must be a <g class=\"lt-knob\">");
  return svg.replace(/\s*\n\s*/g, " ").trim();
}
const virtualPlugin = {
  name: "lt-virtual",
  setup(b) {
    b.onResolve({ filter: /^virtual:toaster-svg$/ }, (a) => ({ path: a.path, namespace: "lt-virtual" }));
    b.onLoad({ filter: /.*/, namespace: "lt-virtual" }, () => ({ contents: `export default ${JSON.stringify(toasterSvg())};`, loader: "js" }));
  },
};

const options = {
  entryPoints: {
    background: join(here, "src", "background.ts"),
    content: join(here, "src", "content.ts"),
    popup: join(here, "src", "popup.ts"),
    bwWorker: join(here, "src", "bwWorker.ts"),
  },
  outdir: out,
  bundle: true,
  format: "esm",
  target: "chrome120",
  sourcemap: "linked",
  logLevel: "info",
  plugins: [virtualPlugin],
};

// Content scripts are classic scripts: no top-level import/export allowed.
const contentOptions = { ...options, entryPoints: { content: options.entryPoints.content }, format: "iife" };
const moduleOptions = { ...options, entryPoints: { background: options.entryPoints.background, popup: options.entryPoints.popup, bwWorker: options.entryPoints.bwWorker } };

if (watch) {
  const a = await esbuild.context(contentOptions);
  const b = await esbuild.context(moduleOptions);
  await Promise.all([a.watch(), b.watch()]);
} else {
  await esbuild.build(contentOptions);
  await esbuild.build(moduleOptions);
}
