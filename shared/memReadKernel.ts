// A tiny hand-assembled WebAssembly SIMD kernel for the "Measure memory speed"
// READ column, shared by the worker (Node worker_threads) and the browser
// extension.
//
// Why it exists: a JavaScript loop that sums a Float64Array is bound by the
// engine's own per-element work (bounds checks, boxed-looking loads), not by
// memory. Measured on a Ryzen 5 5600X: the 4-way-unrolled JS loop gave 5.3 GB/s
// on one thread in Chromium 152 (a plain loop 11.7, Node 22 22.6) while the
// same box reads ~49 GB/s with several threads -- so the READ column came out
// below WRITE and COPY, which real memory never does. This kernel issues four
// independent 16-byte vector loads per iteration, so one thread can pull
// roughly what a core's memory pipeline allows and the thread sweep then shows
// where DRAM saturates.
//
// Contract of the exported function "r":
//   r(start, len, reps) -> f64
//   sums `len` bytes of the imported memory starting at byte offset `start`,
//   `reps` times (f64x2 lanes). `len` must be a multiple of 64 and `start`
//   16-aligned. The returned sum exists only to keep the loads observable.
// The module imports its memory as module "e", name "m".
//
// It is plain bytes (no wabt at build time) -- the unit test validates and
// runs it, and WebAssembly.validate() is the first thing consumers call so a
// runtime without SIMD, or an extension CSP without 'wasm-unsafe-eval', falls
// back to the JS loop instead of failing the whole measurement.

function u32leb(n: number): number[] {
  const out: number[] = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
}

function section(id: number, body: number[]): number[] {
  return [id, ...u32leb(body.length), ...body];
}

export function buildReadKernelWasm(): Uint8Array<ArrayBuffer> {
  // Locals: 0 start, 1 len, 2 reps (params); 3 p, 4 end (i32); 5..8 v128 accumulators.
  const F64X2_ADD = [0xfd, 0xf0, 0x01];
  const body: number[] = [];
  // locals: 2 x i32, 4 x v128
  body.push(0x02, 0x02, 0x7f, 0x04, 0x7b);
  // end = start + len
  body.push(0x20, 0x00, 0x20, 0x01, 0x6a, 0x21, 0x04);
  body.push(0x02, 0x40); // block $done
  body.push(0x03, 0x40); //   loop $rep
  body.push(0x20, 0x02, 0x45, 0x0d, 0x01); //     if reps == 0 break $done
  body.push(0x20, 0x00, 0x21, 0x03); //     p = start
  body.push(0x02, 0x40); //     block $inner_done
  body.push(0x03, 0x40); //       loop $inner
  body.push(0x20, 0x03, 0x20, 0x04, 0x4f, 0x0d, 0x01); //         if p >= end break $inner_done
  for (let k = 0; k < 4; k++) {
    // acc[k] = f64x2.add(acc[k], v128.load offset=16*k (align 16))
    body.push(0x20, 5 + k, 0x20, 0x03, 0xfd, 0x00, 0x04, 16 * k, ...F64X2_ADD, 0x21, 5 + k);
  }
  body.push(0x20, 0x03, 0x41, 0xc0, 0x00, 0x6a, 0x21, 0x03); //         p += 64
  body.push(0x0c, 0x00); //         continue $inner
  body.push(0x0b, 0x0b); //       end loop, end block
  body.push(0x20, 0x02, 0x41, 0x01, 0x6b, 0x21, 0x02); //     reps -= 1
  body.push(0x0c, 0x00); //     continue $rep
  body.push(0x0b, 0x0b); //   end loop, end block
  // (acc5 + acc6) + (acc7 + acc8), then lane0 + lane1
  body.push(0x20, 0x05, 0x20, 0x06, ...F64X2_ADD, 0x20, 0x07, 0x20, 0x08, ...F64X2_ADD, ...F64X2_ADD, 0x21, 0x05);
  body.push(0x20, 0x05, 0xfd, 0x21, 0x00, 0x20, 0x05, 0xfd, 0x21, 0x01, 0xa0);
  body.push(0x0b); // end function

  const bytes = [
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    // type 0: (i32 i32 i32) -> f64
    ...section(1, [0x01, 0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7c]),
    // import memory e.m, min 0 pages
    ...section(2, [0x01, 0x01, 0x65, 0x01, 0x6d, 0x02, 0x00, 0x00]),
    // one function of type 0
    ...section(3, [0x01, 0x00]),
    // export "r" = func 0
    ...section(7, [0x01, 0x01, 0x72, 0x00, 0x00]),
    // code
    ...section(10, [0x01, ...u32leb(body.length), ...body]),
  ];
  return new Uint8Array(bytes);
}
