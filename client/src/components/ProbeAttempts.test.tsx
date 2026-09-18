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
    gpu_buffers_mib: null, gpu_in_system_ram_mib: null, gpu_spill_jitter_mib: null, host_backed_fail: null, list_devices_free_mib: null, claim_fits_free: null,
    load_kind: null, spill_ready_mib: null, spill_ready_jitter_mib: null, spill_work_mib: null, spill_work_jitter_mib: null,
    spill_method: null, spill_shared_growth_mib: null, spill_unlanded_growth_mib: null,
    ladder_ngl_max: null, ladder_max_ctx: null,
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

describe("ProbeAttempts speeds", () => {
  // The probe's request is too short to time anything at the row's context,
  // so no rate -- not even a stored one -- is shown.
  it("shows no generation, prompt or first-token figures", async () => {
    const { table } = await renderTable([row({ gen_tps: 30.4, pp_tps: 412.7, ttft_ms: 7654, prefill_cliff: 1 })]);
    expect(table.queryByText(/tok\/s/i)).toBeNull();
    expect(table.queryByText(/TTFT/)).toBeNull();
    expect(table.queryByText(/30\.4|412\.7|7\.65s/)).toBeNull();
  });
});

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
  // in system RAM, claim vs free, result
  const spillText = (r: HTMLElement) => r.children[r.children.length - 3].textContent;

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

describe("ProbeAttempts claim vs free", () => {
  const claimText = (r: HTMLElement) => r.children[r.children.length - 2].textContent;

  it("holds llama.cpp's claim against the --list-devices free reading", async () => {
    const { bodyRows } = await renderTable([
      row({ gpu_buffers_mib: 7274, list_devices_free_mib: 7378 }),
      row({ gpu_buffers_mib: 7673, list_devices_free_mib: 7378 }),
      // Died before reporting buffers: it did not fit.
      row({ ok: 0, oom: 1, gen_tps: null, gpu_buffers_mib: null, list_devices_free_mib: 7378 }),
      row({ gpu_buffers_mib: 7000, list_devices_free_mib: null }),
    ]);
    expect(claimText(bodyRows[0])).toBe("104 MiB under");
    expect(claimText(bodyRows[1])).toBe("295 MiB over");
    expect(claimText(bodyRows[2])).toBe("no");
    expect(claimText(bodyRows[3])).toBe("—");
  });

  it("follows the worker's per-device verdict, and reads a non-memory failure as no verdict", async () => {
    const { bodyRows, table } = await renderTable([
      // In total under free, but one GPU's own claim did not fit.
      row({ gpu_buffers_mib: 7000, list_devices_free_mib: 22378, claim_fits_free: 0 }),
      row({ ok: 0, gen_tps: null, gpu_buffers_mib: null, list_devices_free_mib: 7378, load_kind: "error" }),
    ]);
    expect(claimText(bodyRows[0])).toBe("over on one GPU");
    expect(claimText(bodyRows[1])).toBe("—");
    expect(table.getByText("failed — not a memory verdict")).toBeTruthy();
  });

  it("shows a claim-only load's claim, no spill verdict, and no pass or failure", async () => {
    const { bodyRows, table } = await renderTable([
      row({ candidate_ctx: 2048, ngl: 23, ok: 0, gen_tps: null, gpu_buffers_mib: 7172, list_devices_free_mib: 7378, claim_fits_free: 1, load_kind: "claim_only" }),
    ]);
    expect(claimText(bodyRows[0])).toBe("206 MiB under");
    expect(bodyRows[0].children[bodyRows[0].children.length - 3].textContent).toBe("—");
    expect(bodyRows[0].lastElementChild?.textContent).toBe("claim fits — not generated");
    expect(table.queryByText("failed")).toBeNull();
  });
});

describe("ProbeAttempts claimed VRAM", () => {
  it("shows llama.cpp's own GPU claim, never dedicated plus shared", async () => {
    const { bodyRows } = await renderTable([
      // The 1,024-token / 1-layer load: 995 model + 238 compute claimed, while
      // dedicated + shared read 1,005 + 1,272.
      row({ gpu_buffers_mib: 1232.6, vram_claimed_peak_mib: 2277, vram_process_peak_mib: 1005, vram_shared_peak_mib: 1272 }),
      row({ gpu_buffers_mib: null, vram_claimed_peak_mib: 8000, vram_process_peak_mib: 6000, vram_shared_peak_mib: 2000 }),
    ]);
    const claimedText = (i: number) => bodyRows[i].children[5].textContent;
    expect(claimedText(0)).toBe("1,233 MiB");
    expect(claimedText(1)).toBe("—");
  });

  it("splits shared into total and llama", async () => {
    const { bodyRows } = await renderTable([row({ vram_shared_total_peak_mib: 7100, vram_shared_peak_mib: 6900 })]);
    // #, context, offload, resident, free, claimed, peak total, peak llama, shared total, shared llama
    expect(bodyRows[0].children[8].textContent).toBe("7,100 MiB");
    expect(bodyRows[0].children[9].textContent).toBe("6,900 MiB");
  });
});

describe("ProbeAttempts growth over the anchor", () => {
  const spillText = (r: HTMLElement) => r.children[r.children.length - 3].textContent;

  it("labels the anchor loads, and judges the rest on min(s, d) against their tolerance", async () => {
    const { bodyRows, table } = await renderTable([
      row({ ngl: 1, gpu_buffers_mib: 1233, gpu_in_system_ram_mib: 0, gpu_spill_jitter_mib: 0, spill_method: "anchor", load_kind: "full" }),
      row({ ngl: 1, gpu_buffers_mib: 1233, gpu_in_system_ram_mib: 0, gpu_spill_jitter_mib: 0, spill_method: "anchor", load_kind: "full" }),
      row({ ngl: 4, gpu_buffers_mib: 1841, gpu_in_system_ram_mib: -3, gpu_spill_jitter_mib: 3, spill_method: "growth", spill_shared_growth_mib: 0, spill_unlanded_growth_mib: -3 }),
      row({ ngl: 9, ok: 0, gpu_buffers_mib: 3330, gpu_in_system_ram_mib: 130, gpu_spill_jitter_mib: 15, spill_method: "growth", spill_shared_growth_mib: 145, spill_unlanded_growth_mib: 130, host_backed_fail: "layers" }),
    ]);
    expect(spillText(bodyRows[0])).toBe("anchor");
    expect(bodyRows[0].lastElementChild?.textContent).toBe("anchor");
    expect(spillText(bodyRows[2])).toBe("none");
    expect(spillText(bodyRows[3])).toBe("130 MiB");
    expect(table.getByText("layers in system RAM")).toBeTruthy();
    expect(bodyRows[3].children[bodyRows[3].children.length - 3].getAttribute("title")).toMatch(/shared GPU memory \+145 MiB, claim not taken by VRAM \+130 MiB/);
  });

  it("still shows a failed anchor as failed", async () => {
    const { bodyRows } = await renderTable([row({ ngl: 1, ok: 0, gen_tps: null, gpu_buffers_mib: 1233, spill_method: "anchor", load_kind: "full" })]);
    expect(bodyRows[0].lastElementChild?.textContent).not.toBe("anchor");
  });
});
