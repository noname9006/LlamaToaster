export const WASM_MAX_PAGES: number;
export const WASM_PAGE_BYTES: number;
export const KERNEL_STEP_BYTES: number;
export function buildKernelModuleBytes(): Uint8Array;
// A shared WebAssembly.Memory (the DOM lib that declares WebAssembly isn't in
// this project's tsconfig, so it's described structurally).
export interface SharedWasmMemory {
  readonly buffer: SharedArrayBuffer;
}
export function allocSharedMemory(bytes: number): SharedWasmMemory;
export interface MemSpeedKernels {
  r(ptr: number, end: number, reps: number): bigint;
  w(ptr: number, end: number, reps: number): void;
  c(ptr: number, end: number, delta: number, reps: number): void;
}
export function instantiateKernels(memory: SharedWasmMemory): MemSpeedKernels;
