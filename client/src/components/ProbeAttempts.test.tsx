// A rung the worker failed because its layers were measurably in system RAM
// has to say so: under the old label it read as a bare "failed", which is
// indistinguishable from a load that did not fit at all. Real render against
// crafted rows, with only the fetch mocked.

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ProbeAttemptDto } from "../types";

const attempts: ProbeAttemptDto[] = [];
vi.mock("../api/client", () => ({
  api: { getProbeAttempts: vi.fn(async () => ({ attempts })) },
}));

import { ProbeAttempts } from "./ProbeAttempts";

function row(overrides: Partial<ProbeAttemptDto>): ProbeAttemptDto {
  return {
    id: `a${attempts.length}`, run_id: "r1", worker_id: "w1", model_id: "m1", seq: attempts.length,
    candidate_ctx: 1024, ngl: 19, ok: 1, oom: 0, spill: 0,
    vram_needed_mib: 9000, vram_free_mib: 9500, vram_peak_mib: 8800, ram_needed_mib: 2000, ram_free_mib: 20000,
    ram_peak_mib: 1500, vram_process_peak_mib: null, ram_total_peak_mib: 12000, vram_shared_peak_mib: 600,
    gen_tps: 30, pp_tps: 400, ttft_ms: 700, prefill_cliff: 0,
    host_backed_method: null, host_backed_slope: null, kv_host_backed_frac: null,
    error: null, created_at: 0, reused_from_run_id: null, vram_discrepancy: 0,
    gpu_layers_resident_est: null, gpu_layers_resident_exact: null,
    ...overrides,
  };
}

describe("ProbeAttempts result badge", () => {
  it("labels a host-backed failure, and nothing else, as layers in system RAM", async () => {
    attempts.splice(
      0,
      attempts.length,
      // The regression itself: every layer claimed, most of them in shared memory.
      row({ ngl: 31, ok: 0, vram_shared_peak_mib: 7400, gen_tps: 4.1, vram_discrepancy: 1, host_backed_method: "ratio" }),
      row({ ngl: 26, ok: 0, oom: 1, gen_tps: null }),
      row({ ngl: 24, ok: 0, gen_tps: null, vram_discrepancy: 1, host_backed_method: "ratio" }),
      // Weaker evidence stays a pass with the warning badge.
      row({ ngl: 22, ok: 1, vram_discrepancy: 1, host_backed_method: "slope", host_backed_slope: 0.9 })
    );
    render(
      <MemoryRouter>
        <ProbeAttempts testId="r1" />
      </MemoryRouter>
    );

    // Scoped to the table: the legend under it names both badges too.
    const table = within(await screen.findByRole("table"));
    expect(table.getAllByText("layers in system RAM")).toHaveLength(1);
    expect(table.getByText("out of memory")).toBeTruthy();
    // No generation is not proof the layers moved, whatever the discrepancy says.
    expect(table.getByText("failed")).toBeTruthy();
    expect(table.getAllByText(/possible VRAM fallback/)).toHaveLength(1);
  });
});
