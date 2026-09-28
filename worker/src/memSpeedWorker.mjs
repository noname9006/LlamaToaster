// One RAM-bandwidth measurement thread, spawned by memSpeed.ts's
// runRamSweep via worker_threads -- a plain, untranspiled .mjs (not .ts) so
// it loads under `node worker_threads` regardless of how the parent process
// itself was started (tsx vs a bundled build), see memSpeed.ts's own doc
// comment for why.
//
// Each thread owns a disjoint element range [start, end) of two shared
// Float64Array views (bufA the source, bufB the copy destination) and times
// its own read/write/copy loop with performance.now() -- the orchestrator
// takes the SLOWEST thread's elapsed time as the row's wall-clock (the
// other threads finished no later), which is the standard way to report a
// parallel-threads bandwidth figure.
import { parentPort, workerData } from "node:worker_threads";
import { performance } from "node:perf_hooks";

const a = new Float64Array(workerData.bufA);
const b = new Float64Array(workerData.bufB);
const { start, end } = workerData;
const len = end - start;

// Defeats dead-code elimination of the read loop without materially
// affecting timing (the branch is never taken) -- same trick used in the
// C#/PowerShell prototype this replaces.
let sink = 0;

function runRead(reps) {
  // 4-way unrolled with independent accumulators -- lets the engine issue
  // overlapping loads instead of stalling on one dependency chain, so this
  // reflects the memory system's throughput rather than being an artifact
  // of a single-accumulator loop's own per-element overhead.
  let acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  const stop = end - ((end - start) % 4);
  for (let r = 0; r < reps; r++) {
    let i = start;
    for (; i < stop; i += 4) {
      acc0 += a[i];
      acc1 += a[i + 1];
      acc2 += a[i + 2];
      acc3 += a[i + 3];
    }
    for (; i < end; i++) acc0 += a[i];
  }
  const acc = acc0 + acc1 + acc2 + acc3;
  sink += acc === 123456789.123456 ? 1 : 0;
  return len * 8 * reps;
}

function runWrite(reps) {
  for (let r = 0; r < reps; r++) a.fill(r + 1, start, end);
  return len * 8 * reps;
}

function runCopy(reps) {
  const src = a.subarray(start, end);
  for (let r = 0; r < reps; r++) b.set(src, start);
  return len * 8 * 2 * reps; // bytes read + bytes written
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

parentPort.postMessage({ ready: true });
