// One RAM-bandwidth thread -- the browser port of worker/src/memSpeedWorker.mjs.
// Each thread owns its own slice of the test size (a browser extension page is
// not cross-origin isolated, so no SharedArrayBuffer -- which is also what we
// want: fill()/set() on a plain buffer are native memset/memcpy, on a shared
// one V8 uses a slow atomic loop); the total traffic is the same as one array
// split between threads.
//
// The slice sits in a WebAssembly.Memory so READ can use the SIMD kernel from
// shared/memReadKernel.ts. A JavaScript sum loop is bound by the engine, not by
// memory (5.3 GB/s on one thread in Chromium 152 against ~30 for the kernel),
// which made READ come out below WRITE and COPY. If WebAssembly is blocked
// (an extension CSP without 'wasm-unsafe-eval') or lacks SIMD, READ falls back
// to the JS loop and the page is told, so it can label the column a lower bound.

let a = new Float64Array(0);
let b = new Float64Array(0);
let aBytes = new Uint8Array(0);
let sliceBytes = 0;
let simdRead: ((start: number, len: number, reps: number) => number) | null = null;
let sink = 0;

function runRead(reps: number): number {
  if (simdRead) {
    sink += simdRead(0, sliceBytes, reps) === 123456789.123456 ? 1 : 0;
    return sliceBytes * reps;
  }
  const x = a;
  let acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  const end = x.length;
  const stop = end - (end % 4);
  for (let r = 0; r < reps; r++) {
    let i = 0;
    for (; i < stop; i += 4) {
      acc0 += x[i];
      acc1 += x[i + 1];
      acc2 += x[i + 2];
      acc3 += x[i + 3];
    }
    for (; i < end; i++) acc0 += x[i];
  }
  sink += acc0 + acc1 + acc2 + acc3 === 123456789.123456 ? 1 : 0;
  return sliceBytes * reps;
}

function runWrite(reps: number): number {
  // Byte-wise fill is a real memset; Float64Array.fill(v) is an element loop in
  // Chromium (16 vs 29 GB/s on one thread, same buffer).
  for (let r = 0; r < reps; r++) aBytes.fill(r + 1);
  return sliceBytes * reps;
}

function runCopy(reps: number): number {
  for (let r = 0; r < reps; r++) b.set(a);
  return sliceBytes * 2 * reps; // read + write
}

type Msg = { cmd: "init"; elems: number; kernel: Uint8Array<ArrayBuffer> | null } | { cmd: "run"; op: "read" | "write" | "copy"; reps: number };

self.onmessage = (e: MessageEvent<Msg>) => {
  const msg = e.data;
  if (msg.cmd === "init") {
    const elems = msg.elems - (msg.elems % 8); // 64-byte multiple: the kernel's stride
    sliceBytes = elems * 8;
    const memory = new WebAssembly.Memory({ initial: Math.ceil((sliceBytes * 2) / 65536) });
    a = new Float64Array(memory.buffer, 0, elems);
    b = new Float64Array(memory.buffer, sliceBytes, elems);
    aBytes = new Uint8Array(memory.buffer, 0, sliceBytes);
    a.fill(1); // touch every page now so first-touch faults never land in a timed loop
    b.fill(1);
    simdRead = null;
    if (msg.kernel) {
      try {
        if (WebAssembly.validate(msg.kernel)) {
          const inst = new WebAssembly.Instance(new WebAssembly.Module(msg.kernel), { e: { m: memory } });
          simdRead = inst.exports.r as unknown as typeof simdRead;
        }
      } catch {
        simdRead = null; // CSP or no SIMD: JS fallback
      }
    }
    (self as unknown as Worker).postMessage({ ready: true, readKernel: simdRead ? "wasm-simd" : "js" });
    return;
  }
  const run = msg.op === "read" ? runRead : msg.op === "write" ? runWrite : runCopy;
  run(1); // untimed warm-up so the JIT has settled
  const t0 = performance.now();
  const bytes = run(msg.reps);
  const elapsedMs = performance.now() - t0;
  (self as unknown as Worker).postMessage({ done: true, bytes, elapsedMs, sink });
};
