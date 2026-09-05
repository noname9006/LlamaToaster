import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GpuMemoryReading } from "./vram.js";

// sampler.ts's own field comment on vram_peak_mib claims it's "per-process
// when the backend could read it that tick, whole-adapter otherwise" -- but
// every readGpuMemory backend implementation (vram.ts's readCudaGpuMemory /
// readGenericGpuMemory / readRocmGpuMemory) returns `used` (whole-adapter,
// GpuMemoryReading's own doc comment: "every process on the GPU combined...
// never process-isolated") and `process` (this benchmark's own pid) as two
// independently-populated fields -- nothing in sample() ever substitutes one
// for the other. This file confirms that directly against the real
// MemorySampler class, with readGpuMemory/systeminformation mocked so the
// two streams can be driven to different, known values.

const { mockMem, mockProcesses, mockReadGpuMemory, mockReadGpuSensors } = vi.hoisted(() => ({
  mockMem: vi.fn(),
  mockProcesses: vi.fn(),
  mockReadGpuMemory: vi.fn(),
  mockReadGpuSensors: vi.fn(),
}));

vi.mock("systeminformation", () => ({
  default: {
    mem: mockMem,
    processes: mockProcesses,
  },
}));

vi.mock("./vram.js", () => ({
  readGpuMemory: mockReadGpuMemory,
}));

vi.mock("./sensors.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./sensors.js")>();
  return { ...actual, readGpuSensors: mockReadGpuSensors };
});

const { MemorySampler } = await import("./sampler.js");

const TEST_PID = 4242;

function wholeAdapter(mib: number): GpuMemoryReading["used"] {
  return { mib, accuracy: "high", source: "driver_reported_memory" };
}

function process(mib: number): GpuMemoryReading["process"] {
  return { mib, accuracy: "exact", source: "process_gpu_usage" };
}

// Drives one MemorySampler.sample() tick directly (bypassing start()'s
// setInterval entirely -- private-in-TS-only fields are plain runtime
// properties, so this is the deterministic way to control which of the
// every-3rd-tick VRAM reads fires without fake timers racing the mocks).
async function tick(sampler: InstanceType<typeof MemorySampler>): Promise<void> {
  await (sampler as unknown as { sample(): Promise<void> }).sample();
}

describe("MemorySampler VRAM peak semantics", () => {
  beforeEach(() => {
    mockMem.mockReset().mockResolvedValue({ active: 1024 * 1024 * 1024 });
    mockProcesses.mockReset().mockResolvedValue({ list: [{ pid: TEST_PID, memRss: 2048 }] });
    mockReadGpuSensors.mockReset().mockResolvedValue({ clockMhz: null, tempC: null, source: null });
    mockReadGpuMemory.mockReset();
  });

  it("vram_peak_mib and vram_total_used_peak_mib both track the whole-adapter reading, never the process reading", async () => {
    mockReadGpuMemory.mockResolvedValueOnce({
      total: wholeAdapter(24000),
      used: wholeAdapter(8000), // everything on the card, not just this load
      process: process(3000), // llama.cpp's own share of that 8000
    } satisfies GpuMemoryReading);

    const sampler = new MemorySampler();
    (sampler as unknown as { pid: number; backend: string }).pid = TEST_PID;
    (sampler as unknown as { pid: number; backend: string }).backend = "cuda";
    await tick(sampler);

    const stats = sampler.stats;
    expect(stats.vram_peak_mib).toBe(8000);
    expect(stats.vram_total_used_peak_mib).toBe(8000);
    // Confirms the two are not just coincidentally equal here -- they are fed
    // literally the same input every tick.
    expect(stats.vram_peak_mib).toBe(stats.vram_total_used_peak_mib);
    // The process-scoped stream is genuinely different data, tracked
    // separately -- this is the number a llama.cpp-only "VRAM peak" column
    // would need to read instead.
    expect(stats.vram_process_peak_mib).toBe(3000);
    expect(stats.vram_process_peak_mib).not.toBe(stats.vram_peak_mib);
  });

  it("a tick where the backend can't yet attribute usage to the pid still grows the adapter-wide peak, but leaves the process peak stale", async () => {
    // Tick 1 (vramTickCount 0, due): both streams present.
    mockReadGpuMemory.mockResolvedValueOnce({
      total: wholeAdapter(24000),
      used: wholeAdapter(7000),
      process: process(2000),
    } satisfies GpuMemoryReading);
    // Ticks 2-3 (vramTickCount 1-2): not due for a VRAM sample at all
    // (VRAM_SAMPLE_EVERY_N_TICKS = 3) -- readGpuMemory must not even be
    // called, so no mock queued for these.
    // Tick 4 (vramTickCount 3, due again): other GPU load pushed the
    // whole-adapter figure up, but nvidia-smi's compute-apps polling hasn't
    // caught this process yet this tick -- `process` is absent, not zero.
    mockReadGpuMemory.mockResolvedValueOnce({
      total: wholeAdapter(24000),
      used: wholeAdapter(9000),
    } satisfies GpuMemoryReading);

    const sampler = new MemorySampler();
    (sampler as unknown as { pid: number; backend: string }).pid = TEST_PID;
    (sampler as unknown as { pid: number; backend: string }).backend = "cuda";
    await tick(sampler); // vramTickCount 0 -> 1, due
    await tick(sampler); // 1 -> 2, not due
    await tick(sampler); // 2 -> 3, not due
    await tick(sampler); // 3 -> 4, due

    expect(mockReadGpuMemory).toHaveBeenCalledTimes(2);

    const stats = sampler.stats;
    // Whole-adapter-derived streams follow the second, higher reading --
    // "VRAM peak" today would report a jump caused by something else on the
    // card, not by this load.
    expect(stats.vram_peak_mib).toBe(9000);
    expect(stats.vram_total_used_peak_mib).toBe(9000);
    // The process-scoped stream never saw that second tick's reading at all,
    // so it stays at the last tick that actually attributed usage to this
    // pid -- switching the UI column to it would sometimes under-report (or,
    // for a load whose whole run never gets picked up, read "--" entirely)
    // rather than over-report from background GPU activity.
    expect(stats.vram_process_peak_mib).toBe(2000);
  });
});
