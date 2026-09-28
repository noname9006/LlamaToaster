import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock systeminformation so detectHardware's vram_dynamic correction can be
// asserted deterministically without querying real GPUs. readNvidiaDriverInfo
// is only called for NVIDIA boxes -- mocked out so the nvidia-smi spawn path
// never runs in a test.
vi.mock("systeminformation", () => ({
  default: {
    cpu: vi.fn(),
    graphics: vi.fn(),
    mem: vi.fn(),
  },
}));

vi.mock("./vram.js", () => ({
  readNvidiaDriverInfo: vi.fn().mockResolvedValue(null),
}));

import siModule from "systeminformation";
import { detectHardware, describeOs, describeMemType, mergeListedDevices, type HardwareInfo } from "./hardware.js";

const si = vi.mocked(siModule);

// systeminformation's Windows vramDynamic derivation flags VideoMemoryType
// === '2', which is the enum value for *dedicated* VRAM (graphics.js:1227).
// These cases mirror what the library actually returns for the two adapter
// classes this logic must tell apart.
interface FakeController {
  vendor?: string;
  model?: string;
  vram?: number;
  vramDynamic?: boolean;
}

function setUp(gpuControllers: FakeController[]) {
  si.cpu.mockResolvedValue({
    manufacturer: "AMD",
    brand: "AMD Ryzen 5 5600X",
    cores: 6,
  } as never);
  si.graphics.mockResolvedValue({
    controllers: gpuControllers,
  } as never);
  si.mem.mockResolvedValue({ total: 16 * 1024 * 1024 * 1024 } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("detectHardware gpu vram_dynamic correction", () => {
  it("keeps a discrete GPU with dedicated VRAM (vram>0, VideoMemoryType=2) unlabeled as shared", async () => {
    // RX 6600 XT style: systeminformation reports real VRAM AND (Windows
    // false-positive) vramDynamic true.
    setUp([{ vendor: "Advanced Micro Devices, Inc.", model: "AMD Radeon RX 6600 XT", vram: 8192, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_mb).toBe(8192);
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });

  it("resets the false-positive flag even when vramDynamic is true but VRAM is real", async () => {
    // Convention: any non-zero dedicated vram_mb is ground truth that the
    // card is not shared/unified -- regardless of what the library says.
    setUp([{ vendor: "Intel Corporation", model: "Intel Arc A770", vram: 16384, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });

  it("preserves 'shared' for unified adapters that report no dedicated VRAM", async () => {
    // iGPU with no on-die pool (unified memory): vram null, vramDynamic true.
    setUp([{ vendor: "Intel Corporation", model: "Intel(R) UHD Graphics 620", vram: 0, vramDynamic: true }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_mb).toBeNull();
    expect(hw.gpu[0].vram_dynamic).toBe(true);
  });

  it("non-dynamic discrete GPU stays non-dynamic", async () => {
    setUp([{ vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram: 12288, vramDynamic: false }]);
    const hw = await detectHardware();
    expect(hw.gpu[0].vram_dynamic).toBe(false);
  });
});
describe("describeOs", () => {
  it("names Windows by edition with the NT build in parentheses", () => {
    expect(describeOs("win32", { distro: "Microsoft Windows 10 Pro N", release: "10.0.19045" })).toEqual({
      family: "Windows",
      name: "Windows 10 Pro N (10.0.19045)",
    });
  });
  it("names Linux by distro + version and macOS with its codename", () => {
    expect(describeOs("linux", { distro: "Ubuntu", release: "22.04" })).toEqual({ family: "Linux", name: "Ubuntu 22.04" });
    expect(describeOs("darwin", { distro: "macOS", release: "14.5", codename: "macOS Sonoma" }).name).toBe("macOS 14.5 (Sonoma)");
  });
  it("falls back to the family when nothing is reported", () => {
    expect(describeOs("linux", null)).toEqual({ family: "Linux", name: "Linux" });
  });
});

describe("describeMemType", () => {
  it("reports type and speed from the first module that names a DDR type", () => {
    expect(describeMemType([{ type: "" }, { type: "DDR4", clockSpeed: 3200 }])).toBe("DDR4-3200");
    expect(describeMemType([{ type: "ddr5" }])).toBe("DDR5");
  });
  it("is null when nothing usable is reported", () => {
    expect(describeMemType(null)).toBeNull();
    expect(describeMemType([{ type: "Unknown" }])).toBeNull();
  });
});

describe("detectHardware OS / memory extras", () => {
  it("survives systeminformation lacking osInfo/memLayout", async () => {
    setUp([{ vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram: 12288 }]);
    const hw = await detectHardware();
    expect(hw.os?.family).toBeTruthy();
    expect(hw.mem_type).toBeNull();
  });
});

describe("mergeListedDevices", () => {
  const base = (gpu: HardwareInfo["gpu"], extra: Partial<HardwareInfo> = {}): HardwareInfo => ({
    platform: "win32",
    arch: "x64",
    cpu: { manufacturer: "AMD", brand: "Ryzen", flags: [], cores: 12 },
    gpu,
    ...extra,
  });
  const dev = (name: string, description: string, totalMib: number, freeMib: number) => ({ name, description, totalMib, freeMib });

  it("attaches max usable and listed total to a matching GPU", () => {
    const hw = mergeListedDevices(
      base([{ vendor: "AMD", model: "AMD Radeon RX 6600 XT", vram_mb: 8192 }]),
      [dev("Vulkan0", "AMD Radeon RX 6600 XT", 8176, 7378)]
    );
    expect(hw.gpu[0]).toMatchObject({ vram_usable_mb: 7378, vram_listed_total_mb: 8176 });
  });

  it("maps every GPU on a multi-GPU box, identical cards to distinct devices", () => {
    const hw = mergeListedDevices(
      base([
        { vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram_mb: 12288 },
        { vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram_mb: 12288 },
        { vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3090", vram_mb: 24576 },
      ]),
      [
        dev("CUDA0", "NVIDIA GeForce RTX 3060", 12288, 11000),
        dev("CUDA1", "NVIDIA GeForce RTX 3060", 12288, 9000),
        dev("CUDA2", "NVIDIA GeForce RTX 3090", 24576, 23000),
      ]
    );
    expect(hw.gpu.map((g) => g.vram_usable_mb)).toEqual([11000, 9000, 23000]);
  });

  it("leaves iGPUs and unified memory untouched", () => {
    const igpu = mergeListedDevices(
      base([{ vendor: "Intel", model: "Intel(R) UHD Graphics 620", vram_mb: null, vram_dynamic: true }]),
      [dev("Vulkan0", "Intel(R) UHD Graphics 620", 8000, 6000)]
    );
    expect(igpu.gpu[0].vram_usable_mb).toBeUndefined();
    const apple = mergeListedDevices(
      base([{ vendor: "Apple", model: "Apple M2 Pro", vram_mb: null }], { unified_memory: true }),
      [dev("Metal", "Apple M2 Pro", 21845, 20000)]
    );
    expect(apple.gpu[0].vram_usable_mb).toBeUndefined();
  });

  it("does not mutate its input and skips unmatched GPUs", () => {
    const input = base([{ vendor: "NVIDIA", model: "NVIDIA GeForce RTX 3060", vram_mb: 12288 }, { vendor: "AMD", model: "Radeon RX 7900 XTX", vram_mb: 24576 }]);
    const out = mergeListedDevices(input, [dev("CUDA0", "NVIDIA GeForce RTX 3060", 12288, 11000)]);
    expect(input.gpu[0].vram_usable_mb).toBeUndefined();
    expect(out.gpu[0].vram_usable_mb).toBe(11000);
    expect(out.gpu[1].vram_usable_mb).toBeUndefined();
  });
});
