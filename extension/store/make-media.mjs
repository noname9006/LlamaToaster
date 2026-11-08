// Renders the Chrome Web Store images from store/media/source/*.
//   node store/make-media.mjs     (needs `npm run build` first for the fonts; uses a one-shot headless Chrome)
//
// The sources are real captures of the extension (a Hugging Face model page with the
// marks, the popup on a real machine); this script only lays them out with captions.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const media = join(here, "media");
const src = (f) => pathToFileURL(join(media, "source", f)).href;
const fonts = pathToFileURL(join(here, "..", "dist", "fonts")).href;
const icons = pathToFileURL(join(here, "..", "icons")).href;
const chrome = process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const tmp = join(here, "build", "media-tmp");
mkdirSync(tmp, { recursive: true });

const css = `
@font-face{font-family:Barlow;font-weight:400;src:url(${fonts}/barlow-400.woff2)}
@font-face{font-family:Barlow;font-weight:500;src:url(${fonts}/barlow-500.woff2)}
@font-face{font-family:Barlow;font-weight:600;src:url(${fonts}/barlow-600.woff2)}
@font-face{font-family:"Barlow Condensed";font-weight:600;src:url(${fonts}/barlow-condensed-600.woff2)}
*{box-sizing:border-box;margin:0}
html,body{width:100%;height:100%;background:#13100c;color:#f4efe6;font:500 22px/1.4 Barlow,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
.stage{position:relative;width:100%;height:100%;background:radial-gradient(1200px 600px at 85% -10%,#2a1d0c 0%,#13100c 60%)}
h1{font:600 52px/1.05 "Barlow Condensed",Barlow,sans-serif;text-transform:uppercase;letter-spacing:.02em}
h1 b{color:#f5a623;font-weight:600}
p.sub{color:#d2c8b9;font-size:24px;margin-top:12px}
.shot{position:absolute;border:1px solid #3d3329;border-radius:10px;box-shadow:0 18px 50px rgba(0,0,0,.55);overflow:hidden;background:#0b0907}
.shot img{display:block;width:100%}
ul{list-style:none;padding:0}
li{display:flex;gap:14px;margin:0 0 22px;color:#d2c8b9;font-size:26px;line-height:1.3}
li::before{content:"";flex:none;width:12px;height:12px;margin-top:11px;background:#f5a623;border-radius:2px}
li b{color:#f4efe6;font-weight:600}
.tag{display:inline-block;font:600 18px "Barlow Condensed";letter-spacing:.12em;text-transform:uppercase;color:#ab9f8d;margin-bottom:14px}
.brand{position:absolute;right:48px;bottom:30px;display:flex;align-items:center;gap:12px}
.brand img.l{width:40px;height:40px}.brand img.w{height:26px}
`;
const page = (body) => `<!doctype html><html><head><meta charset="utf-8"><style>${css}</style></head><body><div class="stage">${body}</div></body></html>`;
const brandL = `<div class="brand" style="left:60px;right:auto"><img class="l" src="${icons}/llama.png"><img class="w" src="${icons}/wordmark.png"></div>`;
const brand = brandL.replace(' style="left:60px;right:auto"', "");

const jobs = {
  "screenshot-1-hf-page": [1280, 800, page(`
    <div style="position:absolute;left:60px;top:38px;right:60px"><h1>Will this GGUF fit <b>your machine?</b></h1><p class="sub">Every quant on a Hugging Face model page, marked before you download it.</p></div>
    <div class="shot" style="left:90px;top:168px;width:1100px;height:606px"><img src="${src("hf-page.jpg")}" style="width:1100px"></div>`)],
  "screenshot-2-marks": [1280, 800, page(`
    <div style="position:absolute;left:60px;top:38px;right:60px"><h1>Every quant, <b>marked at a glance</b></h1></div>
    <div class="shot" style="left:60px;top:150px;width:700px"><img src="${src("hf-gguf-panel.png")}"></div>
    <div style="position:absolute;left:810px;top:176px;width:410px"><span class="tag">Read from the model header</span>
      <ul><li><span><b>Fits on the GPU</b> at your context, or only with experts in RAM</span></li>
          <li><span><b>Partial offload</b> when only some layers fit in VRAM</span></li>
          <li><span><b>Not enough memory</b> flagged before you spend the bandwidth</span></li>
          <li><span>Move the <b>context slider</b>: every chip updates at once</span></li></ul></div>${brand}`)],
  "screenshot-3-machine": [1280, 800, page(`
    <div style="position:absolute;left:60px;top:110px;width:640px"><span class="tag">This machine</span><h1>Knows your RAM, <b>VRAM</b> and memory type</h1>
      <p class="sub" style="margin-top:22px">Set how much to keep free, or use what every app holds right now. With the optional LlamaToaster worker, VRAM is measured, not guessed.</p></div>
    <div class="shot" style="left:760px;top:24px;width:401px;height:752px"><img src="${src("popup.png")}"></div>${brandL}`)],
  "screenshot-4-speed": [1280, 800, page(`
    <div style="position:absolute;left:60px;top:110px;width:600px"><span class="tag">Memory speed</span><h1>Push the lever, <b>measure your memory</b></h1>
      <p class="sub" style="margin-top:22px">RAM read, write and copy by thread count, and GPU memory bandwidth through WebGPU. It is what decides tokens per second.</p></div>
    <div class="shot" style="left:720px;top:110px;width:520px;height:571px"><img src="${src("popup-memory-speed.png")}" style="width:520px;margin-top:-800px"></div>${brandL}`)],
  "screenshot-5-privacy": [1280, 800, page(`
    <div style="position:absolute;left:90px;top:110px;right:90px"><span class="tag">Private by design</span><h1>Runs in your browser. <b>Nothing is sent to us.</b></h1>
      <ul style="margin-top:44px;width:900px">
        <li><span>Reads only the <b>GGUF header</b> with small range requests, about 6 MB per file, never the weights</span></li>
        <li><span>Hardware figures and settings <b>stay on your device</b></span></li>
        <li><span><b>No account, no analytics, no tracking</b></span></li>
        <li><span>Works only on <b>huggingface.co</b></span></li></ul></div>${brand}`)],
  "promo-small-440x280": [440, 280, page(`
    <div style="position:absolute;left:26px;top:30px;right:26px;display:flex;align-items:center;gap:16px"><img src="${icons}/llama.png" style="width:84px;height:84px"><img src="${icons}/wordmark.png" style="height:40px"></div>
    <div style="position:absolute;left:26px;right:26px;top:140px"><h1 style="font-size:36px">Will this GGUF <b>fit?</b></h1><p class="sub" style="font-size:17px;margin-top:8px">Fit marks on every Hugging Face quant.</p></div>`)],
  "promo-marquee-1400x560": [1400, 560, page(`
    <div style="position:absolute;left:70px;top:70px;width:640px"><div style="display:flex;align-items:center;gap:20px"><img src="${icons}/llama.png" style="width:110px;height:110px"><img src="${icons}/wordmark.png" style="height:52px"></div>
      <h1 style="font-size:64px;margin-top:40px">Will this GGUF <b>fit your machine?</b></h1><p class="sub">See which quants fit, right on Hugging Face.</p></div>
    <div class="shot" style="left:790px;top:60px;width:540px"><img src="${src("hf-gguf-panel.png")}" style="width:540px"></div>`)],
};

for (const [name, [w, h, html]] of Object.entries(jobs)) {
  const f = join(tmp, name + ".html");
  writeFileSync(f, html);
  const out = join(media, name + ".png");
  execFileSync(chrome, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", `--window-size=${w},${h}`, `--user-data-dir=${join(tmp, "profile")}`, `--screenshot=${out}`, "--virtual-time-budget=4000", pathToFileURL(f).href], { stdio: "ignore" });
  console.log("rendered", name, w + "x" + h);
}
rmSync(tmp, { recursive: true, force: true });
