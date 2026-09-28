import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { HardwareInfo } from "../types";
import { HardwareSummary } from "./HardwareSummary";

function hw(patch: Partial<HardwareInfo>): HardwareInfo {
  return {
    platform: "win32",
    arch: "x64",
    cpu: { manufacturer: "AMD", brand: "AMD Ryzen 5 5600X", flags: [], cores: 12 },
    gpu: [],
    ...patch,
  };
}

const GIB = 1024 ** 3;

describe("HardwareSummary", () => {
  it("renders OS · CPU · RAM type · GPU (max usable / total)", () => {
    const { container } = render(
      <HardwareSummary
        hardware={hw({
          os: { family: "Windows", name: "Windows 10 Pro N (10.0.19045)" },
          mem_total_bytes: 32 * GIB,
          mem_type: "DDR4-3200",
          gpu: [{ vendor: "AMD", model: "AMD Radeon RX 6600 XT", vram_mb: 8192, vram_usable_mb: 7378, vram_listed_total_mb: 8176 }],
        })}
      />
    );
    expect(container.textContent).toBe(
      "Windows 10 Pro N (10.0.19045) · AMD Ryzen 5 5600X (12 threads) · RAM 32.0 GB DDR4-3200 · AMD Radeon RX 6600 XT (7.2 / 8.0 GB)"
    );
  });

  it("lists every GPU on a multi-GPU machine", () => {
    const { container } = render(
      <HardwareSummary
        hardware={hw({
          gpu: [
            { vendor: "NVIDIA", model: "RTX 3090", vram_mb: 24576, vram_usable_mb: 23000, vram_listed_total_mb: 24576 },
            { vendor: "NVIDIA", model: "RTX 3060", vram_mb: 12288, vram_usable_mb: 11000, vram_listed_total_mb: 12288 },
          ],
        })}
      />
    );
    expect(container.textContent).toContain("RTX 3090 (22.5 / 24.0 GB), RTX 3060 (10.7 / 12.0 GB)");
  });

  it("shows no memory for an iGPU and says 'unified memory' on Apple Silicon", () => {
    const igpu = render(
      <HardwareSummary
        hardware={hw({ mem_total_bytes: 16 * GIB, gpu: [{ vendor: "Intel", model: "Intel(R) UHD Graphics 620", vram_mb: 1024 }] })}
      />
    );
    expect(igpu.container.textContent).toContain("Intel(R) UHD Graphics 620");
    expect(igpu.container.textContent).not.toMatch(/GB\)/);
    igpu.unmount();
    const apple = render(
      <HardwareSummary
        hardware={hw({ platform: "darwin", unified_memory: true, mem_total_bytes: 32 * GIB, gpu: [{ vendor: "Apple", model: "Apple M2 Pro", vram_mb: null }] })}
      />
    );
    expect(apple.container.textContent).toContain("32.0 GB unified memory · Apple M2 Pro");
    expect(apple.container.textContent).not.toContain("RAM");
  });

  it("explains max usable vs total VRAM on hover", () => {
    render(
      <HardwareSummary hardware={hw({ gpu: [{ vendor: "NVIDIA", model: "RTX 3060", vram_mb: 12288, vram_usable_mb: 11000, vram_listed_total_mb: 12288 }] })} />
    );
    fireEvent.mouseEnter(screen.getByText("10.7 / 12.0 GB"));
    expect(screen.getByRole("tooltip").textContent).toMatch(/Max usable VRAM/);
  });
});
