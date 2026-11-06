// WebAssembly SIMD kernels for the RAM-bandwidth measurement (see
// memSpeedWorker.mjs / memSpeed.ts). Plain, untranspiled JavaScript like the
// worker script, and for the same reason: it is loaded by a worker_threads
// Worker by path, so it cannot rely on a TypeScript loader.
//
// Why WASM at all. The first version of this test used JavaScript: a scalar
// loop for read and TypedArray.fill / TypedArray.set for write and copy.
//   * The JS read loop (one Float64 per iteration, bounds-checked) is limited
//     by the loop's own speed, not by memory: ~5 GB/s per thread on a Ryzen
//     5600X, "scaling" with thread count and peaking on the SMT siblings --
//     the signature of a compute-bound test.
//   * fill/set over a SharedArrayBuffer are NOT memset/memcpy in V8: shared
//     memory must be accessed with relaxed-atomic semantics, so V8 falls back
//     to a slower word-at-a-time path (measured 2-3x slower than the same call
//     on a plain ArrayBuffer). That made write and copy look low and flat.
// The kernels below do 64 bytes per iteration with 128-bit loads/stores, so a
// single core can pull as much from DRAM as the memory controller will give it
// and the thread sweep saturates where the hardware does.
//
// The module is assembled by hand (no toolchain, no network) from these three
// functions, WAT-style. All three loop `reps` times over [ptr, end) in
// 64-byte steps -- (end - ptr) must be a positive multiple of 64:
//   r(ptr, end, reps) -> i64        sums every i64 lane with four independent
//                                   accumulators (the checksum keeps the loads
//                                   alive); returns the sum
//   w(ptr, end, reps)               stores a fixed 128-bit pattern
//   c(ptr, end, delta, reps)        copies [ptr,end) to [ptr+delta, end+delta)
//                                   (loads first, then stores, 4x16 bytes)
// The memory is imported as shared (env.m), so every worker thread
// instantiates the same module over the same bytes.

const u = (n) => {
  // unsigned LEB128
  const out = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
};
const str = (s) => [...u(s.length), ...[...s].map((c) => c.charCodeAt(0))];
const section = (id, body) => [id, ...u(body.length), ...body];

// The largest memory a wasm32 module can address: 65536 pages of 64 KiB.
export const WASM_MAX_PAGES = 65536;
export const WASM_PAGE_BYTES = 65536;
export const KERNEL_STEP_BYTES = 64;

const V128 = 0x7b;
const I32 = 0x7f;
const I64 = 0x7e;

const LOCAL_GET = 0x20;
const LOCAL_SET = 0x21;
const LOCAL_TEE = 0x22;
const I32_CONST = 0x41;
const I32_ADD = 0x6a;
const I32_SUB = 0x6b;
const I32_EQZ = 0x45;
const I32_LT_U = 0x49;
const I64_ADD = 0x7c;
const BR = 0x0c;
const BR_IF = 0x0d;
const END = 0x0b;
const BLOCK = [0x02, 0x40];
const LOOP = [0x03, 0x40];
// 64 as signed LEB128 is 0xc0 0x00 -- a bare 0x40 would decode as -64.
const I32_CONST_64 = [I32_CONST, 0xc0, 0x00];
const V128_LOAD = (offset) => [0xfd, 0x00, 0x04, offset]; // align=16
const V128_STORE = (offset) => [0xfd, 0x0b, 0x04, offset];
const I64X2_ADD = [0xfd, 0xce, 0x01];
const I64X2_EXTRACT = (lane) => [0xfd, 0x1d, lane];
const V128_CONST_PATTERN = [
  0xfd, 0x0c,
  0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
  0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
];

// Outer loop over reps wrapped around a body that walks p from `ptrLocal` to
// `endLocal`. `inner` is spliced in at the top of the inner loop and must
// leave the stack empty; the p += 64 / bounds check is appended here.
function repsLoop({ repsLocal, ptrLocal, pLocal, endLocal, inner }) {
  return [
    ...BLOCK,
    ...LOOP,
    LOCAL_GET, repsLocal, I32_EQZ, BR_IF, 1, // reps == 0: leave
    LOCAL_GET, ptrLocal, LOCAL_SET, pLocal, // p = ptr
    ...LOOP,
    ...inner,
    LOCAL_GET, pLocal, ...I32_CONST_64, I32_ADD, LOCAL_TEE, pLocal, // p += 64
    LOCAL_GET, endLocal, I32_LT_U, BR_IF, 0, // while (p <u end)
    END,
    LOCAL_GET, repsLocal, I32_CONST, 1, I32_SUB, LOCAL_SET, repsLocal, // reps -= 1
    BR, 0,
    END,
    END,
  ];
}

function readBody() {
  // params: 0 ptr, 1 end, 2 reps; locals: 3 p (i32), 4..7 accumulators (v128)
  const acc = (local, offset) => [LOCAL_GET, local, LOCAL_GET, 3, ...V128_LOAD(offset), ...I64X2_ADD, LOCAL_SET, local];
  const code = [
    ...repsLoop({
      repsLocal: 2, ptrLocal: 0, pLocal: 3, endLocal: 1,
      inner: [...acc(4, 0), ...acc(5, 16), ...acc(6, 32), ...acc(7, 48)],
    }),
    // sum = a0 + a1 + a2 + a3 (kept in a0), then add its two lanes
    LOCAL_GET, 4, LOCAL_GET, 5, ...I64X2_ADD,
    LOCAL_GET, 6, ...I64X2_ADD,
    LOCAL_GET, 7, ...I64X2_ADD,
    LOCAL_SET, 4,
    LOCAL_GET, 4, ...I64X2_EXTRACT(0),
    LOCAL_GET, 4, ...I64X2_EXTRACT(1),
    I64_ADD,
    END,
  ];
  return [0x02, 0x01, I32, 0x04, V128, ...code];
}

function writeBody() {
  // params: 0 ptr, 1 end, 2 reps; locals: 3 p (i32), 4 pattern (v128)
  const st = (offset) => [LOCAL_GET, 3, LOCAL_GET, 4, ...V128_STORE(offset)];
  const code = [
    ...V128_CONST_PATTERN, LOCAL_SET, 4,
    ...repsLoop({
      repsLocal: 2, ptrLocal: 0, pLocal: 3, endLocal: 1,
      inner: [...st(0), ...st(16), ...st(32), ...st(48)],
    }),
    END,
  ];
  return [0x02, 0x01, I32, 0x01, V128, ...code];
}

function copyBody() {
  // params: 0 ptr, 1 end, 2 delta, 3 reps; locals: 4 p, 5 q (i32), 6..9 (v128)
  const ld = (local, offset) => [LOCAL_GET, 4, ...V128_LOAD(offset), LOCAL_SET, local];
  const st = (local, offset) => [LOCAL_GET, 5, LOCAL_GET, local, ...V128_STORE(offset)];
  const code = [
    ...repsLoop({
      repsLocal: 3, ptrLocal: 0, pLocal: 4, endLocal: 1,
      inner: [
        LOCAL_GET, 4, LOCAL_GET, 2, I32_ADD, LOCAL_SET, 5, // q = p + delta
        ...ld(6, 0), ...ld(7, 16), ...ld(8, 32), ...ld(9, 48),
        ...st(6, 0), ...st(7, 16), ...st(8, 32), ...st(9, 48),
      ],
    }),
    END,
  ];
  return [0x02, 0x02, I32, 0x04, V128, ...code];
}

export function buildKernelModuleBytes() {
  const bodies = [readBody(), writeBody(), copyBody()];
  const bytes = [
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, [
      0x03,
      0x60, 0x03, I32, I32, I32, 0x01, I64, // (i32 i32 i32) -> i64
      0x60, 0x03, I32, I32, I32, 0x00, // (i32 i32 i32) -> ()
      0x60, 0x04, I32, I32, I32, I32, 0x00, // (i32 i32 i32 i32) -> ()
    ]),
    // import env.m: shared memory, min 1 page, max WASM_MAX_PAGES
    ...section(2, [0x01, ...str("env"), ...str("m"), 0x02, 0x03, ...u(1), ...u(WASM_MAX_PAGES)]),
    ...section(3, [0x03, 0x00, 0x01, 0x02]), // three functions: types 0, 1, 2
    ...section(7, [0x03, ...str("r"), 0x00, 0x00, ...str("w"), 0x00, 0x01, ...str("c"), 0x00, 0x02]),
    ...section(10, [0x03, ...bodies.flatMap((b) => [...u(b.length), ...b])]),
  ];
  return new Uint8Array(bytes);
}

// A shared WebAssembly.Memory of at least `bytes` (rounded up to whole pages).
// Shared so every worker thread can be handed the same memory; fixed size
// (initial == maximum) so its buffer never detaches or moves.
export function allocSharedMemory(bytes) {
  const pages = Math.ceil(bytes / WASM_PAGE_BYTES);
  if (pages > WASM_MAX_PAGES) throw new RangeError(`a wasm memory holds at most ${WASM_MAX_PAGES * WASM_PAGE_BYTES} bytes`);
  return new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
}

let compiled = null;
// Compiles once per thread; returns { r, w, c } bound to `memory` (see the
// header comment for the signatures).
export function instantiateKernels(memory) {
  compiled ??= new WebAssembly.Module(buildKernelModuleBytes());
  return new WebAssembly.Instance(compiled, { env: { m: memory } }).exports;
}
