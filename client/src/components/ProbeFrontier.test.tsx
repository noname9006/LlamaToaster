// Real render against rows the real search produced (shared/probeLadder.ts run
// against a simulated card), with the fetch and the canvas mocked. What matters
// here is that the table says where each number came from: an unconfirmed
// answer must say so, a context the budget never reached must not read like an
// answer, and the stop once nothing fits must be named.

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import type { ProbeAttemptDto } from "../types";
import { claimFitsFree, nextLadderRung, type LadderAttempt, type LadderRung } from "../../../shared/probeLadder";

const attempts: ProbeAttemptDto[] = [];
vi.mock("../api/client", () => ({
  api: { getProbeAttempts: vi.fn(async () => ({ attempts })) },
}));
// chart.js draws to a canvas jsdom does not implement.
vi.mock("./Chart", () => ({ Chart: () => <div data-testid="chart" /> }));

import { ProbeFrontier, outcomeFrom } from "./ProbeFrontier";

const FREE = 7378;
const claim = (r: LadderRung) => 280 + 405 * r.ngl + r.ctx * 0.004;
const clean = (r: LadderRung) => (r.ctx <= 32_768 ? [0, 1, 2, 3, 4, 6].includes(r.ngl) : r.ctx <= 131_072 ? r.ngl <= 2 : false);

function row(r: LadderRung, seq: number, overrides: Partial<ProbeAttemptDto> = {}): ProbeAttemptDto {
  const c = claim(r);
  const fits = c < FREE;
  const ok = fits && clean(r);
  return {
    id: `a${seq}`, run_id: "r1", worker_id: "w1", model_id: "m1", seq,
    candidate_ctx: r.ctx, ngl: r.ngl, ok: ok ? 1 : 0, oom: 0, spill: 0,
    vram_needed_mib: null, vram_free_mib: 7600, vram_peak_mib: 7000, ram_needed_mib: null, ram_free_mib: 20000,
    ram_peak_mib: 1500, vram_process_peak_mib: fits ? Math.round(c) : null, ram_total_peak_mib: 12000, vram_shared_peak_mib: 450,
    vram_shared_total_peak_mib: null, vram_claimed_peak_mib: null,
    gen_tps: fits ? 12 - r.ngl * 0.1 : null, pp_tps: fits ? 100 : null, ttft_ms: 700, prefill_cliff: 0,
    host_backed_method: null, host_backed_slope: null, kv_host_backed_frac: null,
    gpu_buffers_mib: c, gpu_in_system_ram_mib: fits ? (ok ? -10 : 150) : null, gpu_spill_jitter_mib: fits ? 2 : null,
    host_backed_fail: fits && !ok ? "cache" : null, list_devices_free_mib: FREE, claim_fits_free: null,
    load_kind: fits ? "full" : "claim_stop", spill_ready_mib: null, spill_ready_jitter_mib: null,
    spill_work_mib: null, spill_work_jitter_mib: null,
    spill_method: null, spill_shared_growth_mib: null, spill_unlanded_growth_mib: null, ladder_ngl_max: 41, ladder_max_ctx: 262_144,
    error: null, created_at: 0, reused_from_run_id: null, vram_discrepancy: 0,
    gpu_layers_resident_est: null, gpu_layers_resident_exact: null,
    ...overrides,
  };
}

/** Runs the real Wizard search against the simulated card for `loads` loads. */
function wizardRows(loads: number): ProbeAttemptDto[] {
  const history: LadderAttempt[] = [];
  const rows: ProbeAttemptDto[] = [];
  for (let i = 0; i < loads; i++) {
    const next = nextLadderRung({
      mode: "frontier", candidateCtx: 1024, candidateNgl: 10, nglMax: 41, maxCtx: 262_144, maxLoads: loads,
      history, freeVramMib: FREE, calculateNgl: () => 12,
    });
    if (!next) break;
    const r = row(next, rows.length);
    rows.push(r);
    history.push({
      ...next, ok: r.ok === 1, placementJudged: true, claimedMib: r.gpu_buffers_mib,
      fitsFree: claimFitsFree({ claimedMib: r.gpu_buffers_mib, freeMib: FREE, loaded: true }),
    });
  }
  return rows;
}

async function renderCurve(rows: ProbeAttemptDto[]) {
  attempts.splice(0, attempts.length, ...rows);
  render(<ProbeFrontier testId="r1" />);
  const table = await screen.findByRole("table");
  const bodyRows = within(table).getAllByRole("row").slice(1);
  return { table: within(table), bodyRows };
}

describe("ProbeFrontier", () => {
  it("shows both answers per context, re-resolved exactly as the search found them", async () => {
    const { bodyRows } = await renderCurve(wizardRows(200));
    expect(bodyRows).toHaveLength(9);
    const cells = (i: number) => within(bodyRows[i]).getAllByRole("cell").map((c) => c.textContent);
    expect(cells(0)[1]).toMatch(/^6/);
    expect(cells(0)[2]).toBe("17");
    expect(cells(6)[1]).toMatch(/^2/);
    expect(cells(6)[2]).toBe("16");
    expect(cells(8)[1]).toBe("none");
    expect(cells(8)[2]).toBe("14");
  });

  it("shows no generation rate", async () => {
    const { table } = await renderCurve(wizardRows(200));
    expect(table.queryByText(/tok\/s/i)).toBeNull();
  });

  it("marks a confirmed answer, and the answer the budget left unconfirmed", async () => {
    const full = await renderCurve(wizardRows(200));
    expect(within(full.bodyRows[0]).getByTitle(/second time, clean again/)).toBeInTheDocument();
  });

  it("says the probe stopped early and leaves unreached contexts not measured", async () => {
    const { bodyRows } = await renderCurve(wizardRows(20));
    expect(screen.getByText(/stopped before the search finished/)).toBeInTheDocument();
    const last = within(bodyRows[bodyRows.length - 1]);
    expect(last.getByText("not measured")).toBeInTheDocument();
    expect(last.getAllByRole("cell")[1].textContent).toBe("—");
  });

  it("names the context from which nothing fits", async () => {
    // Same search, on a card whose KV cache is so expensive nothing fits at 128k.
    const expensive = (r: LadderRung) => 280 + 405 * r.ngl + r.ctx * 0.1;
    const history: LadderAttempt[] = [];
    const rows: ProbeAttemptDto[] = [];
    for (let i = 0; i < 200; i++) {
      const next = nextLadderRung({
        mode: "frontier", candidateCtx: 1024, candidateNgl: 10, nglMax: 41, maxCtx: 262_144, maxLoads: 200,
        history, freeVramMib: FREE, calculateNgl: () => 12,
      });
      if (!next) break;
      const c = expensive(next);
      const fits = c < FREE;
      const r = row(next, rows.length, {
        gpu_buffers_mib: c, ok: fits && next.ngl <= 2 ? 1 : 0, load_kind: fits ? "full" : "claim_stop", gen_tps: fits ? 10 : null,
      });
      rows.push(r);
      history.push({ ...next, ok: r.ok === 1, placementJudged: true, claimedMib: c, fitsFree: fits });
    }
    await renderCurve(rows);
    expect(screen.getByText(/Nothing fits free VRAM from 131,072 tokens up/)).toBeInTheDocument();
  });

  it("says when a probe was recorded by an earlier version of the context test", async () => {
    const rows = wizardRows(200).map((r) => ({ ...r, ladder_ngl_max: null, ladder_max_ctx: null }));
    await renderCurve(rows);
    expect(screen.getByText(/recorded by an earlier version of the context test/)).toBeInTheDocument();
    expect(screen.queryByText(/stopped before the search finished/)).not.toBeInTheDocument();
  });

  it("re-resolves a Targets probe from its first load's pinned axis", () => {
    const rows = [row({ ctx: 4096, ngl: 6 }, 0)];
    expect(outcomeFrom(rows, "custom")?.clean).toMatchObject({ ctx: 4096, ngl: 6 });
  });

  it("re-resolves an anchored probe with its anchor first, reading the pinned axis past it", () => {
    const anchor = { spill_method: "anchor" as const };
    const anchorsOnly = [row({ ctx: 1024, ngl: 1 }, 0, anchor)];
    expect(outcomeFrom(anchorsOnly, "custom")).toMatchObject({ next: { ctx: 1024, ngl: 1 }, nextIsAnchor: true });
    const rows = [...anchorsOnly, row({ ctx: 1024, ngl: 1 }, 1, anchor), row({ ctx: 4096, ngl: 6 }, 2, { spill_method: "growth" })];
    expect(outcomeFrom(rows, "custom")?.clean).toMatchObject({ ctx: 4096, ngl: 6 });
  });
});
