# LlamaToaster for Chrome

A Chrome extension that answers "will this GGUF fit my machine?" on Hugging Face model pages. It uses the LlamaToaster web UI's own fit logic and marks.

- **Popup** (or Chrome's side panel: the **Side panel** button docks it there and the toolbar button then opens the panel; **Use popup** undoes it):
  - **Fit check** at the top: **Manual / Auto** lever, context slider, and how much VRAM to keep free: 0.5 / 1 / 1.5 GB, **In use now** (what every app holds on the GPU when a check starts, captured once so the marks don't move afterwards), or a custom amount. The KV cache is sized as f16.
  - **This machine**: CPU, RAM type and speed (e.g. `DDR4-3600`, from a worker) plus free/total, GPU name with its VRAM size, memory kind (Auto says what it detected), VRAM available (`n/a / 8 GB` without a worker, with a link to the worker install commands).
  - **Memory speed** (folded by default; the header shows the best figures): push the toaster lever to measure RAM read/write/copy by thread count and VRAM read/write/copy via WebGPU. The lever stays down while it runs and pops back up when done.
- **Hugging Face model pages**: the "Hardware compatibility" heading above the GGUF quants is replaced by "Will it fit? Click the toaster to check." in Hugging Face's own heading style, with the toaster icon (`icons/toaster.svg`, the "Option 1 minimal" 32 px design) to its right. In **Manual** mode nothing is read until you click it; in **Auto** mode every GGUF page is checked as it opens. The toaster's dial turns while the headers are read, then each quant chip gets the web UI's mark in place of Hugging Face's icon:
  - ✅✅ fits in VRAM, unified memory or RAM
  - ✅ fits with partial offload
  - ❗ not enough memory

  Hover a mark for the figures. Moving the popup's slider re-marks every chip at once: the slow part is reading the headers, and that happens only once per file. Click the toaster again to re-read free memory.

## How it sizes a model

1. It lists the repo's files and matches each chip to its `.gguf`. For a split model, that is shard 1 plus every other part.
2. It reads only the GGUF header (metadata and tensor list) with HTTP range requests, **2 MB per request**, and stops where the header ends. That is about 6 MB for a typical file and never touches the weights. Headers are cached, so a page you open again shows its marks straight away.
3. It runs the same code as the app: `shared/gguf.ts` parses the header, `shared/fitSimulate.ts` places the model the way `llama-fit-params` does (output layer and blocks onto the GPU, MoE experts moved to the CPU first), and `shared/fitLevel.ts` picks the mark.

## What a browser can't see

- **Free VRAM.** No browser API reports VRAM (unless a worker is installed, see below). The total comes from a table of known cards (you can override it in the popup). "Available" is that total minus what you say other apps use (0.5 GB by default).
- **Unified memory** is detected from the GPU name: Apple, Intel iGPU, Radeon integrated graphics. Apple silicon gets about 3/4 of RAM, like Metal's working-set budget. You can override the memory type in the popup.
- **Memory speed** reads a little below native tools: RAM Read is a WebAssembly SIMD kernel (a JavaScript sum loop is bound by the engine, not by memory), Write/Copy are the browser's native `fill`/`set`, VRAM is WebGPU kernels. The manifest allows `'wasm-unsafe-eval'` for that kernel; without it Read falls back to a JavaScript loop and the popup says so.

## Measured VRAM through the worker

Chrome reads VRAM internally but gives extensions no way to get it. A LlamaToaster worker on the same machine fills the gap: its installer and updater register `worker/chrome-host/` as a Chrome native-messaging host, and Chrome starts it on demand whenever the extension asks. The worker service doesn't have to be running, and the order doesn't matter (worker first or extension first). A worker installed before this existed picks it up on its next update.

| OS | Source | GPUs |
|---|---|---|
| Windows | DXGI (total per adapter) + `GPU Adapter Memory` counters (live usage, all processes) | Intel, AMD, NVIDIA, Qualcomm, any WDDM GPU |
| Linux | DRM sysfs (`mem_info_vram_*`, xe tile VRAM, i915 `lmem_*`), `nvidia-smi` | AMD, Intel Arc, NVIDIA |
| macOS | `sysctl hw.memsize` + `vm_stat` | Apple silicon (unified) |

The extension has a fixed ID (`odghfahbjacleniojhemnojbiabfebon`, from the `key` in `manifest.json`), and only that ID may call the host. The same host reports the installed RAM's type and speed (Windows: `Win32_PhysicalMemory`; macOS: `system_profiler`; Linux: `dmidecode`, which usually needs root, so Linux shows no type). With it, the popup marks VRAM as **measured**, and "free" is the total minus what every app on that GPU uses right now. Integrated GPUs (512 MB carve-out or less) are treated as unified memory. Without a worker, everything falls back to the estimate.

## Build and load

```bash
cd extension
npm install        # type definitions only
npm run build      # -> extension/dist
npm run typecheck
```

Then open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and pick `extension/dist`. To pick up a rebuild, click the reload icon on the extension's card.
