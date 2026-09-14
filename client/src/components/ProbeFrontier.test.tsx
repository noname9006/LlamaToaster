// Real render against crafted rungs, with the fetch and the canvas mocked.
// What matters here is that the table says where each number CAME FROM: a
// stop nobody loaded must not read like a measurement, a stop the budget never
// reached must not read like a boundary, and a pass whose cache placement was
// never judged has to carry its warning.

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import type { ProbeAttemptDto } from "../types";

const attempts: ProbeAttemptDto[] = [];
vi.mock("../api/client", () => ({
  api: { getProbeAttempts: vi.fn(async () => ({ attempts })) },
}));
// chart.js draws to a canvas jsdom does not implement, and the curve's
// correctness lives in resolveFrontier (covered in shared/probeLadder.test.ts)
// rather than in the drawing.
vi.mock("./Chart", () => ({ Chart: () => <div data-testid="chart" /> }));

import { ProbeFrontier } from "./ProbeFrontier";

function row(overrides: Partial<ProbeAttemptDto>): ProbeAttemptDto {
  return {
    id: `a${attempts.length}`, run_id: "r1", worker_id: "w1", model_id: "m1", seq: attempts.length,
    candidate_ctx: 1024, ngl: 12, ok: 1, oom: 0, spill: 0,
    vram_needed_mib: 9000, vram_free_mib: 9500, vram_peak_mib: 8800, ram_needed_mib: 2000, ram_free_mib: 20000,
    ram_peak_mib: 1500, vram_process_peak_mib: 6200, ram_total_peak_mib: 12000, vram_shared_peak_mib: 300,
    vram_shared_total_peak_mib: null, vram_claimed_peak_mib: null,
    gen_tps: 22.5, pp_tps: 400, ttft_ms: 700, prefill_cliff: 0,
    host_backed_method: null, host_backed_slope: null, kv_host_backed_frac: null,
    error: null, created_at: 0, reused_from_run_id: null, vram_discrepancy: 0,
    gpu_layers_resident_est: null, gpu_layers_resident_exact: null,
    ...overrides,
  };
}

async function renderCurve(rows: ProbeAttemptDto[]) {
  attempts.splice(0, attempts.length, ...rows);
  render(<ProbeFrontier testId="r1" />);
  const table = await screen.findByRole("table");
  const bodyRows = within(table).getAllByRole("row").slice(1);
  return { table: within(table), bodyRows };
}

// A flat curve: 12 layers at the floor and at the ceiling, with every stop
// between them settled by monotonicity rather than by a load.
const FLAT = [
  row({ candidate_ctx: 1024, ngl: 12, ok: 1 }),
  row({ candidate_ctx: 1024, ngl: 13, ok: 0 }),
  row({ candidate_ctx: 8192, ngl: 12, ok: 1, kv_host_backed_frac: 0.1 }),
];

describe("ProbeFrontier", () => {
  it("shows one row per context stop, with the layer count that fits there", async () => {
    const { bodyRows } = await renderCurve(FLAT);
    // 1024, 2048, 4096, 8192
    expect(bodyRows).toHaveLength(4);
    for (const r of bodyRows) expect(within(r).getByText("12")).toBeInTheDocument();
  });

  it("says which stops were loaded and which were settled without a load", async () => {
    const { bodyRows } = await renderCurve(FLAT);
    expect(within(bodyRows[0]).getByText("measured")).toBeInTheDocument();
    expect(within(bodyRows[1]).getByText("implied")).toBeInTheDocument();
    expect(within(bodyRows[3]).getByText("measured")).toBeInTheDocument();
  });

  it("shows measured figures only for stops that were really loaded", async () => {
    const { bodyRows } = await renderCurve(FLAT);
    expect(within(bodyRows[0]).getByText("6,200 MiB")).toBeInTheDocument();
    expect(within(bodyRows[0]).getByText("22.5")).toBeInTheDocument();
    // An implied stop has no rung of its own, so it has no measurements.
    expect(within(bodyRows[1]).queryByText("6,200 MiB")).not.toBeInTheDocument();
  });

  it("captions the rate as an empty-cache figure, never the speed at that context", async () => {
    await renderCurve(FLAT);
    expect(screen.getByText(/gen tok\/s \(empty cache\)/)).toBeInTheDocument();
  });

  it("warns when a stop rests on a pass whose cache placement was never judged", async () => {
    const { bodyRows } = await renderCurve([
      row({ candidate_ctx: 1024, ngl: 12, ok: 1 }),
      row({ candidate_ctx: 1024, ngl: 13, ok: 0 }),
      // No kv_host_backed_frac: the check could not run on this worker.
      row({ candidate_ctx: 8192, ngl: 12, ok: 1, kv_host_backed_frac: null }),
    ]);
    expect(within(bodyRows[3]).getByTitle(/cache stayed in VRAM was never measured/)).toBeInTheDocument();
    expect(screen.getByText(/allocation ceilings, not speeds/)).toBeInTheDocument();
  });

  it("marks a stop the budget never reached as not measured, rather than guessing it", async () => {
    const { bodyRows } = await renderCurve([
      row({ candidate_ctx: 1024, ngl: 12, ok: 1 }),
      row({ candidate_ctx: 1024, ngl: 13, ok: 0 }),
      row({ candidate_ctx: 8192, ngl: 12, ok: 0, kv_host_backed_frac: 0.9 }),
    ]);
    const top = bodyRows[3];
    expect(within(top).getByText("not measured")).toBeInTheDocument();
    // No layer count either -- "not measured" must never be rendered as a 0 or
    // as the neighbour's value.
    expect(within(top).getAllByRole("cell")[1].textContent).toBe("—");
  });

  // A stop where something loaded but the budget ran out before the boundary
  // was found knows a FLOOR, not an answer. Printing it as a plain number
  // would read exactly like a measured boundary.
  it("shows a layer count the budget never pinned down as a floor, not a boundary", async () => {
    const { bodyRows } = await renderCurve([
      row({ candidate_ctx: 1024, ngl: 12, ok: 1 }),
      row({ candidate_ctx: 1024, ngl: 13, ok: 0 }),
      row({ candidate_ctx: 4096, ngl: 9, ok: 1, kv_host_backed_frac: 0.2 }),
      row({ candidate_ctx: 8192, ngl: 12, ok: 0, kv_host_backed_frac: 0.9 }),
    ]);
    const partial = bodyRows[2]; // 4096: 9 loaded, 12 unproven, boundary unknown
    expect(within(partial).getByText("not measured")).toBeInTheDocument();
    expect(within(partial).getByText(/≥ 9/)).toBeInTheDocument();
  });

  it("says outright when nothing fits at a context", async () => {
    const { bodyRows } = await renderCurve([
      row({ candidate_ctx: 1024, ngl: 0, ok: 0, oom: 1, gen_tps: null }),
    ]);
    expect(within(bodyRows[0]).getByText("none fit")).toBeInTheDocument();
  });
});
