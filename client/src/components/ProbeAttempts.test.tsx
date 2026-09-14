// Real render against crafted rows, with only the fetch mocked. Covers what
// the result column says about a failed rung -- a rung the worker failed
// because its layers or its cache were measurably in system RAM has to say
// so, since a bare "failed" is indistinguishable from a load that did not fit
// at all -- and what the claimed-VRAM and in-system-RAM columns show.

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
    vram_shared_total_peak_mib: null, vram_claimed_peak_mib: null,
    gen_tps: 30, pp_tps: 400, ttft_ms: 700, prefill_cliff: 0,
    host_backed_method: null, host_backed_slope: null, kv_host_backed_frac: null,
    gpu_buffers_mib: null, gpu_in_system_ram_mib: null, gpu_spill_jitter_mib: null, host_backed_fail: null,
    error: null, created_at: 0, reused_from_run_id: null, vram_discrepancy: 0,
    gpu_layers_resident_est: null, gpu_layers_resident_exact: null,
    ...overrides,
  };
}

async function renderTable(rows: ProbeAttemptDto[]) {
  attempts.splice(0, attempts.length, ...rows);
  render(
    <MemoryRouter>
      <ProbeAttempts testId="r1" />
    </MemoryRouter>
  );
  // Scoped to the table: the legend under it names the badges too.
  const table = await screen.findByRole("table");
  // Header rows first, then one row per attempt.
  const bodyRows = within(table).getAllByRole("row").slice(2);
  return { table: within(table), bodyRows };
}

describe("ProbeAttempts result badge", () => {
  it("labels an older worker's host-backed failures by what spilled, and only those", async () => {
    const { table } = await renderTable([
      // Every layer claimed, most of them in shared memory.
      row({ ngl: 31, ok: 0, vram_shared_peak_mib: 7400, gen_tps: 4.1, vram_discrepancy: 1, host_backed_method: "ratio" }),
      // Probe 124c2ab1's 262144-token rung: the layers fit, the cache didn't.
      row({ candidate_ctx: 262144, ngl: 15, ok: 0, gen_tps: 9, host_backed_method: "slope", kv_host_backed_frac: 0.97 }),
      row({ ngl: 26, ok: 0, oom: 1, gen_tps: null }),
      row({ ngl: 24, ok: 0, gen_tps: null, vram_discrepancy: 1, host_backed_method: "ratio" }),
      // Weaker evidence stays a pass with the warning badge.
      row({ ngl: 22, ok: 1, vram_discrepancy: 1, host_backed_method: "slope", host_backed_slope: 0.9 }),
    ]);

    expect(table.getAllByText("layers in system RAM")).toHaveLength(1);
    expect(table.getAllByText("cache in system RAM")).toHaveLength(1);
    expect(table.getByText("out of memory")).toBeTruthy();
    // No generation is not proof the layers moved, whatever the discrepancy says.
    expect(table.getByText("failed")).toBeTruthy();
    expect(table.getAllByText(/possible VRAM fallback/)).toHaveLength(1);
  });

  it("names the spill the worker recorded", async () => {
    const { table } = await renderTable([
      row({ ngl: 8, ok: 0, gen_tps: 11.8, vram_discrepancy: 1, gpu_buffers_mib: 3634.7, gpu_in_system_ram_mib: 828.43, gpu_spill_jitter_mib: 12, host_backed_fail: "layers" }),
      row({ candidate_ctx: 262144, ngl: 4, ok: 0, gen_tps: 9.3, vram_discrepancy: 1, gpu_buffers_mib: 3319.56, gpu_in_system_ram_mib: 234.33, gpu_spill_jitter_mib: 0, host_backed_fail: "cache" }),
    ]);
    expect(table.getAllByText("layers in system RAM")).toHaveLength(1);
    expect(table.getAllByText("cache in system RAM")).toHaveLength(1);
    expect(screen.getByText(/of what llama.cpp put/)).toBeTruthy();
  });

  // ok is a SQLite 0/1: a bare `a.ok && ...` rendered a literal "0" beside the
  // badge on every failed row.
  it("renders nothing but the badge in a failed row's result cell", async () => {
    const { bodyRows } = await renderTable([
      row({ ngl: 31, ok: 0, gen_tps: 4.1, vram_discrepancy: 1, host_backed_method: "ratio" }),
    ]);
    expect(bodyRows[0].lastElementChild?.textContent).toBe("layers in system RAM");
  });
});

describe("ProbeAttempts in system RAM", () => {
  const spillText = (r: HTMLElement) => r.children[r.children.length - 2].textContent;

  it("shows the measured spill, none for a clean load, and a dash where nothing was measured", async () => {
    const { bodyRows } = await renderTable([
      row({ ngl: 8, ok: 0, gen_tps: 11.8, gpu_buffers_mib: 3634.7, gpu_in_system_ram_mib: 828.43, gpu_spill_jitter_mib: 12, host_backed_fail: "layers" }),
      row({ ngl: 4, gpu_buffers_mib: 2001.57, gpu_in_system_ram_mib: -15.76, gpu_spill_jitter_mib: 3 }),
      row({ ngl: 4, gpu_buffers_mib: null, gpu_in_system_ram_mib: null }),
    ]);
    expect(spillText(bodyRows[0])).toBe("828 MiB");
    expect(spillText(bodyRows[1])).toBe("none");
    expect(spillText(bodyRows[2])).toBe("—");
  });

  it("does not show a difference within the counter's own movement as a spill", async () => {
    const { bodyRows } = await renderTable([row({ gpu_buffers_mib: 2000, gpu_in_system_ram_mib: 20, gpu_spill_jitter_mib: 40 })]);
    expect(spillText(bodyRows[0])).toBe("none");
  });
});

describe("ProbeAttempts claimed VRAM", () => {
  it("shows the measured claimed figure, and marks the peak-sum fallback for older rows", async () => {
    const { bodyRows } = await renderTable([
      row({ vram_claimed_peak_mib: 11040, vram_process_peak_mib: 6321, vram_shared_peak_mib: 5000 }),
      // Recorded before claimed existed: the two separate peaks, added.
      row({ vram_claimed_peak_mib: null, vram_process_peak_mib: 6000, vram_shared_peak_mib: 2000 }),
      row({ vram_claimed_peak_mib: null, vram_process_peak_mib: null }),
    ]);
    const claimedText = (i: number) => bodyRows[i].children[5].textContent;
    expect(claimedText(0)).toBe("11,040 MiB");
    expect(claimedText(1)).toBe("~8,000 MiB");
    expect(claimedText(2)).toBe("—");
  });

  it("splits shared into total and llama", async () => {
    const { bodyRows } = await renderTable([row({ vram_shared_total_peak_mib: 7100, vram_shared_peak_mib: 6900 })]);
    // #, context, offload, resident, free, claimed, peak total, peak llama, shared total, shared llama
    expect(bodyRows[0].children[8].textContent).toBe("7,100 MiB");
    expect(bodyRows[0].children[9].textContent).toBe("6,900 MiB");
  });
});
