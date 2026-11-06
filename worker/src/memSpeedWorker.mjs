// One RAM-bandwidth measurement thread, spawned by memSpeed.ts via
// worker_threads -- a plain, untranspiled .mjs (not .ts) so it loads under
// `node worker_threads` regardless of how the parent process itself was
// started (tsx vs a bundled build), see memSpeed.ts's own doc comment.
//
// The thread is a small command server. The orchestrator hands it shared
// WebAssembly.Memory regions ("attach"), then sends "commit" (touch the pages
// of this thread's slice so first-touch page faults aren't measured) and
// "run" (execute one op over this thread's slice).
//
// "run" is what makes the numbers trustworthy: every active thread first
// meets at a barrier (Atomics on a shared control block), so they all start
// the timed work within microseconds of each other; each records its own
// start/end from process.hrtime (one monotonic clock for every thread --
// performance.now() is per-thread-origin); the orchestrator then reports
// total bytes / (latest end - earliest start). The earlier version let each
// thread time itself from whenever its message happened to arrive and divided
// by the slowest thread's own elapsed time, which under-counted overlap (so a
// run could read high) and was at the mercy of one late thread.
import { parentPort, workerData } from "node:worker_threads";
import { instantiateKernels, KERNEL_STEP_BYTES } from "./memSpeedKernels.mjs";

const ctl = new Int32Array(workerData.ctl); // [0] arrived, [1] go generation
const times = new BigInt64Array(workerData.times); // [2*slot] start, [2*slot+1] end (ns)

// Work is issued to the kernels in blocks so (a) the wasm engine's dynamic
// tier-up (Liftoff -> optimizing compiler) takes effect between calls -- there
// is no on-stack replacement for wasm, so one giant call would run entirely in
// the baseline tier -- and (b) a long pass doesn't sit in one uninterruptible
// call.
const BLOCK_BYTES = 4 * 1024 * 1024;

const regions = new Map(); // id -> { r, w, c }
let sink = 0n;

function attach(id, memory) {
  const k = instantiateKernels(memory);
  // Tiny warm-up so the kernels are already optimized before the first timed
  // window: a few hundred calls over 64 KiB at the start of the region, which
  // is the thread's own memory anyway (the first run's commit pass rewrites it).
  const span = 64 * 1024;
  for (let i = 0; i < 400; i++) {
    sink += k.r(0, span, 1);
    k.w(0, span, 1);
    k.c(0, span / 2, span / 2, 1);
  }
  regions.set(id, k);
}

function runBlocks(start, end, fn) {
  for (let b = start; b < end; b += BLOCK_BYTES) fn(b, Math.min(b + BLOCK_BYTES, end));
}

function execute(op, region, start, end, delta, passes) {
  const k = regions.get(region);
  if (!k) throw new Error(`unknown region ${region}`);
  if ((end - start) % KERNEL_STEP_BYTES !== 0 || end <= start) throw new Error("slice must be a positive multiple of 64 bytes");
  if (op === "read") {
    let acc = 0n;
    for (let p = 0; p < passes; p++) runBlocks(start, end, (b, e) => { acc += k.r(b, e, 1); });
    sink += acc;
  } else if (op === "write") {
    for (let p = 0; p < passes; p++) runBlocks(start, end, (b, e) => k.w(b, e, 1));
  } else if (op === "copy") {
    for (let p = 0; p < passes; p++) runBlocks(start, end, (b, e) => k.c(b, e, delta, 1));
  } else {
    throw new Error(`unknown op ${op}`);
  }
}

function barrier(gen, n) {
  const arrived = Atomics.add(ctl, 0, 1) + 1;
  if (arrived === n) {
    Atomics.store(ctl, 1, gen);
    Atomics.notify(ctl, 1);
    return;
  }
  // Spin briefly (the other threads are normally microseconds behind), then
  // park on the futex so a straggler isn't starved by spinners.
  for (let i = 0; i < 20000; i++) if (Atomics.load(ctl, 1) === gen) return;
  for (;;) {
    const cur = Atomics.load(ctl, 1);
    if (cur === gen) return;
    Atomics.wait(ctl, 1, cur, 100);
  }
}

function handle(msg) {
  switch (msg.cmd) {
    case "attach":
      attach(msg.region, msg.memory);
      return {};
    case "detach":
      regions.delete(msg.region);
      return {};
    case "commit": {
      // Touch (write) this thread's slice once: page faults and first-touch
      // placement are paid here, never inside a timed window.
      execute("write", msg.region, msg.start, msg.end, 0, 1);
      return {};
    }
    case "run": {
      barrier(msg.gen, msg.threads);
      const t0 = process.hrtime.bigint();
      execute(msg.op, msg.region, msg.start, msg.end, msg.delta, msg.passes);
      const t1 = process.hrtime.bigint();
      times[2 * msg.slot] = t0;
      times[2 * msg.slot + 1] = t1;
      return {};
    }
    default:
      throw new Error(`unknown command ${msg.cmd}`);
  }
}

parentPort.on("message", (msg) => {
  try {
    parentPort.postMessage({ reqId: msg.reqId, ...handle(msg) });
  } catch (err) {
    parentPort.postMessage({ reqId: msg.reqId, error: err instanceof Error ? err.message : String(err) });
  }
});

parentPort.postMessage({ ready: true });
