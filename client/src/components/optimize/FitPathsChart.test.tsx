import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { FitLegend, FitPathsChart } from "./FitPathsChart";
import type { FitPoint } from "../../../../shared/fitParams.js";
import { buildFitSeries } from "../../../../shared/optimizeFlow.js";
import { detectMoe } from "../../../../shared/moePlacement.js";

function pt(over: Partial<FitPoint>): FitPoint {
  return {
    ctx: 4096, fa: "on", ctk: "f16", ctv: "f16", margin_mib: 1024, verdict: "partial", reason: null, ngl: 16, layers_gpu: 16,
    layers_total: 65, overflow_layers: 0, ot: null, dev_used_mib: 6000, dev_free_mib: 1100, host_used_mib: 14000, total_mib: 20000,
    kv_mib: 500, compute_mib: 500, need_all_gpu_mib: 20000, splits: 2, fa_disabled: false, n_expert: 0, n_expert_used: 0,
    inferred: false, raw_args: null, error: null, ...over,
  };
}

const points: FitPoint[] = [
  pt({ ctx: 4096, layers_gpu: 17, total_mib: 19500 }),
  pt({ ctx: 32768, layers_gpu: 16, total_mib: 21300 }),
  pt({ ctx: 262144, layers_gpu: 7, total_mib: 36000 }),
  pt({ fa: "off", ctx: 4096, layers_gpu: 18, total_mib: 19400 }),
  pt({ fa: "off", ctx: 32768, layers_gpu: 12, total_mib: 22600 }),
  pt({ fa: "off", ctx: 262144, layers_gpu: 10, verdict: "doesnt_fit", reason: "ram", total_mib: 60000 }),
];

describe("FitPathsChart", () => {
  it("draws one path per setup, marks the target and lets every point be picked", () => {
    const series = buildFitSeries(points, []);
    const onSelect = vi.fn();
    render(
      <FitPathsChart series={series} selection={{ seriesId: "f16-on", ctx: 32768 }} onSelect={onSelect} moe={null} machine="dgpu" targetCtx={32768} layersTotal={65} />
    );
    expect(screen.getByText("32k · target")).toBeTruthy();
    expect(screen.getByText(/LAYERS ON GPU \(OF 65\)/)).toBeTruthy();
    const target = screen.getByRole("button", { name: /f16 · FA off · 32k/ });
    fireEvent.click(target);
    expect(onSelect).toHaveBeenCalledWith({ seriesId: "f16-off", ctx: 32768 });
    // The doesn't-fit point is still clickable and says so.
    expect(screen.getByRole("button", { name: /f16 · FA off · 256k · doesn't fit/ })).toBeTruthy();
    // Keyboard works too.
    fireEvent.keyDown(screen.getByRole("button", { name: /f16 · FA on · 4k/ }), { key: "Enter" });
    expect(onSelect).toHaveBeenLastCalledWith({ seriesId: "f16-on", ctx: 4096 });
  });

  it("MoE: dense-phase points fall below a labelled break", () => {
    const moe = detectMoe({ dense: Array(40).fill(1), moe: Array(40).fill(100), embed: 0, output: 0, other: 0 });
    const series = buildFitSeries(
      [
        pt({ ctx: 32768, ngl: 41, layers_gpu: 41, layers_total: 41, overflow_layers: 33, total_mib: 19000 }),
        pt({ ctx: 262144, ngl: 5, layers_gpu: 5, layers_total: 41, total_mib: 25000 }),
      ],
      []
    );
    render(<FitPathsChart series={series} selection={null} onSelect={() => undefined} moe={moe} machine="dgpu" targetCtx={null} layersTotal={41} />);
    expect(screen.getByText(/dense layers only/)).toBeTruthy();
    expect(screen.getByText(/MOE LAYERS WITH EXPERTS ON GPU \(OF 40\)/)).toBeTruthy();
  });

  it("shows an empty state", () => {
    render(<FitPathsChart series={[]} selection={null} onSelect={() => undefined} moe={null} machine="dgpu" targetCtx={null} layersTotal={null} />);
    expect(screen.getByText("No answers to plot yet.")).toBeTruthy();
  });
});

describe("FitLegend", () => {
  it("marks the selected setup", () => {
    const series = buildFitSeries(points, []);
    const onPick = vi.fn();
    render(<FitLegend series={series} selectedId="f16-off" onPick={onPick} />);
    expect(screen.getByRole("button", { name: /f16 · FA off/ }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: /f16 · FA on/ }));
    expect(onPick).toHaveBeenCalledWith("f16-on");
  });
});
