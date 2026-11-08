import { describe, expect, it } from "vitest";
import { buildReadKernelWasm } from "./memReadKernel.js";

// The root tsconfig carries no WebAssembly typings; Node has the real thing.
const WebAssembly = (globalThis as unknown as { WebAssembly: any }).WebAssembly;

describe("buildReadKernelWasm", () => {
  const bytes = buildReadKernelWasm();

  it("is a valid module", () => {
    expect(WebAssembly.validate(bytes)).toBe(true);
  });

  function load(pages: number) {
    const memory = new WebAssembly.Memory({ initial: pages });
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { e: { m: memory } });
    return { f64: new Float64Array(memory.buffer), r: inst.exports.r as (s: number, l: number, n: number) => number };
  }

  it("sums len bytes, reps times", () => {
    const { f64, r } = load(1);
    f64.fill(1.5);
    expect(r(0, 8192, 3)).toBe(1024 * 1.5 * 3);
  });

  it("respects the start offset and length", () => {
    const { f64, r } = load(1);
    f64.fill(1);
    f64.fill(100, 64, 128); // bytes 512..1024 hold 100s
    expect(r(512, 512, 1)).toBe(64 * 100);
    expect(r(0, 512, 1)).toBe(64);
  });

  it("returns 0 for zero reps", () => {
    const { f64, r } = load(1);
    f64.fill(2);
    expect(r(0, 4096, 0)).toBe(0);
  });
});
