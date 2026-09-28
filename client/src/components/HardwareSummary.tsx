import { Fragment, type ReactNode } from "react";
import type { HardwareInfo } from "../types";
import { formatBytes, formatGpuVram } from "../utils";
import { Tooltip } from "./Tooltip";

const VRAM_TIP_WITH_USABLE =
  "Max usable VRAM: the free video memory llama.cpp reports for this GPU right now (llama --list-devices) -- what a model can really use once the OS, display and other apps have taken their share. Total VRAM: the card's full memory.";
const VRAM_TIP_TOTAL_ONLY =
  "Total VRAM: the card's full memory. Max usable VRAM (what is free for a model) shows here once this machine has read it from an installed llama.cpp build.";

interface HardwareItem {
  label: string;
  value: ReactNode;
}

// One entry per piece of hardware: OS, CPU, RAM (or unified memory), and every
// GPU. A GPU that has no memory of its own (an iGPU, or Apple Silicon's
// unified memory) shows its name alone -- its "VRAM" is system RAM, already
// counted in the memory entry. The VRAM figure carries the hover explanation
// of max usable vs total.
function hardwareItems(hardware: HardwareInfo, platform?: string | null): HardwareItem[] {
  const items: HardwareItem[] = [];
  const os = hardware.os?.name || platform || hardware.platform;
  if (os) items.push({ label: "OS", value: os });

  const cpu = hardware.cpu.brand || hardware.cpu.manufacturer;
  if (cpu) items.push({ label: "CPU", value: `${cpu}${hardware.cpu.cores ? ` (${hardware.cpu.cores} threads)` : ""}` });

  if (hardware.mem_total_bytes) {
    const total = formatBytes(hardware.mem_total_bytes);
    items.push(
      hardware.unified_memory
        ? { label: "Memory", value: `${total} unified memory` }
        : { label: "RAM", value: `${total}${hardware.mem_type ? ` ${hardware.mem_type}` : ""}` }
    );
  }

  if (hardware.gpu.length === 0) {
    items.push({ label: "GPU", value: "none detected" });
  }
  hardware.gpu.forEach((g, i) => {
    const vram = formatGpuVram(g, hardware.unified_memory);
    items.push({
      label: hardware.gpu.length > 1 ? `GPU ${i + 1}` : "GPU",
      value: (
        <>
          {g.model || g.vendor || "unknown"}
          {vram && (
            <>
              {" ("}
              <Tooltip text={g.vram_usable_mb != null ? VRAM_TIP_WITH_USABLE : VRAM_TIP_TOTAL_ONLY}>{vram}</Tooltip>
              {")"}
            </>
          )}
        </>
      ),
    });
  });
  return items;
}

// The machine's hardware description, in one of two layouts:
//   "inline" (default): OS · CPU (N threads) · RAM 32.0 GB DDR4-3200 · GPU (max usable / total VRAM)
//   "lines": one labelled row per piece of hardware -- easier to scan, and the
//            natural fit when a machine has several GPUs.
export function HardwareSummary({
  hardware,
  platform,
  layout = "inline",
  className,
}: {
  hardware: HardwareInfo | null | undefined;
  /** Fallback OS label when the worker predates HardwareInfo.os. */
  platform?: string | null;
  layout?: "inline" | "lines";
  className?: string;
}) {
  if (!hardware) return <span className={className ?? "text-muted"}>hardware unknown</span>;
  const items = hardwareItems(hardware, platform);

  if (layout === "lines") {
    return (
      <dl className={className ?? "grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm"}>
        {items.map((item, i) => (
          <Fragment key={i}>
            <dt className="text-muted">{item.label}</dt>
            <dd className="text-fg">{item.value}</dd>
          </Fragment>
        ))}
      </dl>
    );
  }

  // Inline: the GPU entries stay together (", "-joined) as one segment.
  const gpuItems = items.filter((it) => it.label.startsWith("GPU"));
  const segments: ReactNode[] = [
    ...items.filter((it) => !it.label.startsWith("GPU")).map((it) => (it.label === "RAM" ? <>RAM {it.value}</> : it.value)),
    gpuItems.length > 0
      ? gpuItems.map((it, i) => (
          <Fragment key={i}>
            {i > 0 && ", "}
            {it.value}
          </Fragment>
        ))
      : null,
  ].filter((s) => s != null);

  return (
    <span className={className}>
      {segments.map((s, i) => (
        <Fragment key={i}>
          {i > 0 && " · "}
          {s}
        </Fragment>
      ))}
    </span>
  );
}
