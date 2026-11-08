// One RAM-bandwidth measurement thread, spawned by memSpeed.ts's
// measureRamRow via worker_threads -- a plain, untranspiled .mjs (not .ts) so
// it loads under `node worker_threads` regardless of how the parent process
// itself was started (tsx vs a bundled build), see memSpeed.ts's own doc
// comment for why.
//
// Each thread allocates and owns its own slice (two arrays: the source and the
// copy destination) and times its own read/write/copy loop with
// performance.now() -- the orchestrator takes the SLOWEST thread's elapsed
// time as the row's wall-clock (the other threads finished no later), which is
// the standard way to report a parallel-threads bandwidth figure.
//
// The slice is deliberately NOT a SharedArrayBuffer: V8 runs fill()/set() on
// shared buffers through a relaxed-atomic element loop instead of memset/memcpy
// (measured on one thread: fill 15.5 vs 29.2 GB/s, copy 22.2 vs 36.9 GB/s on
// plain buffers), which made WRITE and COPY read about half of what the
// memory does. Threads never touch each other's data, so nothing needs to be
// shared -- and allocating the slice on the thread that uses it also keeps its
// pages local on NUMA machines.
//
// The slice lives in a WebAssembly.Memory (a plain, non-shared ArrayBuffer under
// the hood): that lets READ use the SIMD kernel from shared/memReadKernel.ts,
// which is bound by memory rather than by JavaScript loop overhead. If the
// runtime can't run it, READ falls back to a unrolled JS loop and the
// orchestrator is told, so it can warn that the read column is a lower bound.
import { parentPort, workerData } from "node:worker_threads";
import { performance } from "node:perf_hooks";

// 64-byte multiple so the SIMD kernel's stride (64 B) always fits exactly.
const elems = workerData.elems - (workerData.elems % 8);
const sliceBytes = elems * 8;
const pages = Math.ceil((sliceBytes * 2) / 65536);
const memory = new WebAssembly.Memory({ initial: pages });
const a = new Float64Array(memory.buffer, 0, elems);
const b = new Float64Array(memory.buffer, sliceBytes, elems);
const aBytes = new Uint8Array(memory.buffer, 0, sliceBytes);
a.fill(1); // touch every page up front so first-touch faults never land in a timed loop
b.fill(1);

let simdRead = null;
if (workerData.kernel) {
  try {
    const bytes = workerData.kernel;
    if (WebAssembly.validate(bytes)) {
      simdRead = new WebAssembly.Instance(new WebAssembly.Module(bytes), { e: { m: memory } }).exports.r;
    }
  } catch {
    simdRead = null;
  }
}

// Defeats dead-code elimination of the read loop without materially
// affecting timing (the branch is never taken).
let sink = 0;

function runRead(reps) {
  if (simdRead) {
    sink += simdRead(0, sliceBytes, reps) === 123456789.123456 ? 1 : 0;
    return sliceBytes * reps;
  }
  // Fallback: 4-way unrolled with independent accumulators.
  const x = a;
  let acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  const stop = elems - (elems % 4);
  for (let r = 0; r < reps; r++) {
    let i = 0;
    for (; i < stop; i += 4) {
      acc0 += x[i];
      acc1 += x[i + 1];
      acc2 += x[i + 2];
      acc3 += x[i + 3];
    }
    for (; i < elems; i++) acc0 += x[i];
  }
  sink += acc0 + acc1 + acc2 + acc3 === 123456789.123456 ? 1 : 0;
  return sliceBytes * reps;
}

function runWrite(reps) {
  // Byte-wise fill is a real memset (Float64Array.fill(v) is an element loop in
  // some engines: 16 vs 29 GB/s on one thread in Chromium).
  for (let r = 0; r < reps; r++) aBytes.fill(r + 1);
  return sliceBytes * reps;
}

function runCopy(reps) {
  for (let r = 0; r < reps; r++) b.set(a);
  return sliceBytes * 2 * reps; // bytes read + bytes written
}

function run(op, reps) {
  if (op === "read") return runRead(reps);
  if (op === "write") return runWrite(reps);
  return runCopy(reps);
}

parentPort.on("message", (msg) => {
  if (msg.cmd !== "run") return;
  // One untimed warm-up rep first -- lets the JIT settle on this thread
  // before the timed loop, so the FIRST timed sample isn't penalized by
  // interpreter-speed execution the way a cold single-shot call would be.
  run(msg.op, 1);
  const t0 = performance.now();
  const bytes = run(msg.op, msg.reps);
  const elapsedMs = performance.now() - t0;
  parentPort.postMessage({ done: true, bytes, elapsedMs, sink });
});

parentPort.postMessage({ ready: true, readKernel: simdRead ? "wasm-simd" : "js" });
