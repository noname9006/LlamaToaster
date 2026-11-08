// "Measure memory speed" in the browser -- the same read / write / copy
// table the Workers page shows (worker/src/memSpeed.ts), measured with what a
// browser has:
//   RAM   Web Workers sweeping 1, 2, 4 ... all threads over 256 MiB in total,
//         row GB/s = all bytes moved / the slowest thread's time. READ runs a
//         WebAssembly SIMD kernel (a JS sum loop is bound by the engine, not by
//         memory); WRITE/COPY are the native fill()/set() on plain buffers.
//   VRAM  WebGPU: a compute kernel that reads, one that writes, and a buffer
//         copy, over the largest storage buffer the adapter allows (<= 256 MiB).
// Browser numbers sit somewhat under native ones (JavaScript loops on the
// RAM side, API overhead on the GPU side); they are a floor, not the ceiling.

import type { BandwidthResult } from "./settings";
import { buildReadKernelWasm } from "../../shared/memReadKernel";

const RAM_MIB = 256;
const RAM_TRIALS = 6;
const RAM_REPS = 4;
type Op = "read" | "write" | "copy";

function threadCounts(logical: number): number[] {
  const set = new Set<number>([1]);
  if (logical >= 2) set.add(2);
  if (logical >= 4) set.add(4);
  if (logical >= 8) set.add(Math.floor(logical / 2));
  set.add(Math.max(1, logical));
  return [...set].sort((x, y) => x - y);
}

function call<T>(w: Worker, msg: unknown): Promise<T> {
  return new Promise((resolve, reject) => {
    w.onmessage = (e) => resolve(e.data as T);
    w.onerror = (e) => reject(new Error(e.message));
    w.postMessage(msg);
  });
}

/** VRAM timing trials per operation (see timeOp). */
const VRAM_TRIALS = 6;
/** Called once per finished trial, so progress can move in small steps. */
type Tick = () => void;

async function ramRow(threads: number, tick: Tick): Promise<{ threads: number; readGBs: number; writeGBs: number; copyGBs: number; readKernel: "wasm-simd" | "js" }> {
  let readKernel: "wasm-simd" | "js" = "js";
  const elems = Math.floor((RAM_MIB * 1048576) / 8 / threads);
  const workers = Array.from({ length: threads }, () => new Worker(chrome.runtime.getURL("bwWorker.js"), { type: "module" }));
  try {
    const kernel = buildReadKernelWasm();
    const ready = await Promise.all(workers.map((w) => call<{ readKernel: string }>(w, { cmd: "init", elems, kernel })));
    readKernel = ready.every((r) => r.readKernel === "wasm-simd") ? "wasm-simd" : "js";
    const best: Record<Op, number> = { read: 0, write: 0, copy: 0 };
    for (const op of ["read", "write", "copy"] as const) {
      for (let t = 0; t < RAM_TRIALS; t++) {
        const res = await Promise.all(workers.map((w) => call<{ bytes: number; elapsedMs: number }>(w, { cmd: "run", op, reps: RAM_REPS })));
        const bytes = res.reduce((s, r) => s + r.bytes, 0);
        const slowest = Math.max(...res.map((r) => r.elapsedMs));
        if (slowest > 0) best[op] = Math.max(best[op], bytes / (slowest / 1000) / 1e9);
        tick();
      }
    }
    return { threads, readGBs: best.read, writeGBs: best.write, copyGBs: best.copy, readKernel };
  } finally {
    workers.forEach((w) => w.terminate());
  }
}

const READ_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  let n = arrayLength(&src);
  var acc = vec4<f32>(0.0);
  for (var i = gid.x; i < n; i += stride) { acc += src[i]; }
  if (acc.x == 123456.789) { out[0] = acc; }
}`;

const WRITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  let n = arrayLength(&dst);
  let v = vec4<f32>(f32(gid.x));
  for (var i = gid.x; i < n; i += stride) { dst[i] = v; }
}`;

async function vramBandwidth(tick: Tick): Promise<{ vram: BandwidthResult["vram"]; reason?: string }> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { vram: null, reason: "This browser has no WebGPU." };
  const adapter = await gpu.requestAdapter(/Windows/.test(navigator.userAgent) ? undefined : { powerPreference: "high-performance" });
  if (!adapter) return { vram: null, reason: "WebGPU found no GPU adapter." };
  const size = Math.min(256 * 1048576, adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: size, maxBufferSize: size },
  });
  try {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const a = device.createBuffer({ size, usage });
    const b = device.createBuffer({ size, usage });
    const out = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const groups = Math.min(65535, Math.ceil(size / 16 / 256 / 16)); // ~16 vec4 per thread

    const readPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: READ_WGSL }), entryPoint: "main" } });
    const writePipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WRITE_WGSL }), entryPoint: "main" } });
    const readBind = device.createBindGroup({ layout: readPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: out } }] });
    const writeBind = device.createBindGroup({ layout: writePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }] });

    const encode = (op: Op, reps: number) => {
      const enc = device.createCommandEncoder();
      for (let r = 0; r < reps; r++) {
        if (op === "copy") enc.copyBufferToBuffer(r % 2 ? b : a, 0, r % 2 ? a : b, 0, size);
        else {
          const pass = enc.beginComputePass();
          pass.setPipeline(op === "read" ? readPipe : writePipe);
          pass.setBindGroup(0, op === "read" ? readBind : writeBind);
          pass.dispatchWorkgroups(groups);
          pass.end();
        }
      }
      return enc.finish();
    };
    const timeOp = async (op: Op): Promise<number> => {
      device.queue.submit([encode(op, 2)]);
      await device.queue.onSubmittedWorkDone(); // warm-up
      // Grow the batch until it runs long enough to time with a wall clock.
      let reps = 8;
      let best = 0;
      for (let trial = 0; trial < VRAM_TRIALS; trial++) {
        const t0 = performance.now();
        device.queue.submit([encode(op, reps)]);
        await device.queue.onSubmittedWorkDone();
        const ms = performance.now() - t0;
        const bytes = size * reps * (op === "copy" ? 2 : 1);
        best = Math.max(best, bytes / (ms / 1000) / 1e9);
        if (ms < 150 && reps < 512) reps *= 2;
        tick();
      }
      return best;
    };
    const readGBs = await timeOp("read");
    const writeGBs = await timeOp("write");
    const copyGBs = await timeOp("copy");
    const info = adapter.info;
    return {
      vram: {
        deviceName: [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ") || "GPU",
        bufferMib: size / 1048576,
        readGBs,
        writeGBs,
        copyGBs,
      },
    };
  } finally {
    device.destroy();
  }
}

export type MeasurePhase = "ram" | "vram";

/** phase: what is being measured now; detail: its current step; done: the
 * share of that phase's trials finished (0..1); trial: the share one trial is
 * (so a view can ease between reports). RAM and VRAM each count 0..100%. */
export async function measureBandwidth(
  onProgress: (phase: MeasurePhase, detail: string, done: number, trial: number) => void
): Promise<BandwidthResult> {
  const logical = navigator.hardwareConcurrency || 1;
  const points = [];
  const counts = threadCounts(logical);
  // RAM: 3 ops x RAM_TRIALS per thread count. VRAM: 3 ops x VRAM_TRIALS.
  const totals: Record<MeasurePhase, number> = { ram: counts.length * 3 * RAM_TRIALS, vram: 3 * VRAM_TRIALS };
  let phase: MeasurePhase = "ram";
  let finished = 0;
  let detail = "";
  const report = () => onProgress(phase, detail, Math.min(1, finished / totals[phase]), 1 / totals[phase]);
  const tick = () => {
    finished++;
    report();
  };
  for (const t of counts) {
    detail = `${t} thread${t === 1 ? "" : "s"}…`;
    report();
    points.push(await ramRow(t, tick));
  }
  phase = "vram";
  finished = 0;
  detail = "";
  report();
  let vram: BandwidthResult["vram"] = null;
  let reason: string | undefined;
  try {
    ({ vram, reason } = await vramBandwidth(tick));
  } catch (err) {
    reason = `VRAM measurement failed: ${err instanceof Error ? err.message : String(err)}`;
  }
  const readKernel = points.every((p) => p.readKernel === "wasm-simd") ? "wasm-simd" : "js";
  return {
    measuredAt: Date.now(),
    ram: { logicalCores: logical, readKernel, points: points.map(({ readKernel: _k, ...p }) => p) },
    vram,
    vramUnavailableReason: reason,
  };
}
