import { describe, expect, it } from "vitest";
import { coreMask, defaultThreadPlan, fallbackTopology, pickThreadWinner, threadCandidates, type CpuCore, type CpuTopology } from "./threadPlan.js";

function cores(n: number, opts: { smt?: boolean; l3?: (i: number) => number; die?: (i: number) => number | null; eff?: (i: number) => number } = {}): CpuCore[] {
  return Array.from({ length: n }, (_, i) => ({
    id: i,
    logical: opts.smt === false ? [i] : [2 * i, 2 * i + 1],
    l3: opts.l3 ? opts.l3(i) : 0,
    die: opts.die ? opts.die(i) : 0,
    effClass: opts.eff ? opts.eff(i) : 0,
  }));
}
const topo = (c: CpuCore[], extra: Partial<CpuTopology> = {}): CpuTopology => ({
  source: "windows",
  cores: c,
  logicalCount: c.reduce((a, x) => a + x.logical.length, 0),
  masksSupported: true,
  ...extra,
});

describe("defaultThreadPlan", () => {
  it("single CCD (Ryzen 5 5600X): all physical cores, one logical CPU each", () => {
    const p = defaultThreadPlan(topo(cores(6)));
    expect(p.t).toMatchObject({ threads: 6, mask: "0x555" });
    expect(p.tb).toMatchObject({ threads: 6, mask: "0x555" });
    expect(p.strict).toBe(true);
    expect(p.summary).toBe("1 CCD · 6 cores / 12 threads");
  });
  it("2 CCDs: one CCD for both", () => {
    const p = defaultThreadPlan(topo(cores(16, { die: (i) => (i < 8 ? 0 : 1), l3: (i) => (i < 8 ? 0 : 1) })));
    expect(p.t.threads).toBe(8);
    expect(p.tb.mask).toBe(coreMask(cores(8)));
    expect(p.domains).toBe(2);
  });
  it("X3D: picks the CCD with the larger L3", () => {
    const t = topo(cores(16, { die: (i) => (i < 8 ? 0 : 1), l3: (i) => (i < 8 ? 0 : 1) }), { l3SizesKib: { "0": 32768, "1": 98304 } });
    const p = defaultThreadPlan(t);
    expect(p.t.mask).toBe(coreMask(t.cores.slice(8)));
    expect(p.reason).toMatch(/larger L3/);
  });
  it("Zen 2: one die, two L3s → labelled CCX", () => {
    const p = defaultThreadPlan(topo(cores(6, { l3: (i) => (i < 3 ? 0 : 1) })));
    expect(p.domainLabel).toBe("CCX");
    expect(p.t.threads).toBe(3);
  });
  it("Intel hybrid: -t P-cores, -tb P+E", () => {
    const c = [...cores(8, { eff: () => 1 }), ...cores(16, { smt: false, eff: () => 0 }).map((x, i) => ({ ...x, id: 8 + i, logical: [16 + i] }))];
    const p = defaultThreadPlan(topo(c));
    expect(p.hybrid).toBe(true);
    expect(p.t.threads).toBe(8);
    expect(p.tb.threads).toBe(24);
    const t = threadCandidates(topo(c), "t");
    expect(t[0]).toMatchObject({ kind: "p_plus_e", preselected: true, threads: 24 });
    const tb = threadCandidates(topo(c), "tb");
    expect(tb[0]).toMatchObject({ kind: "p_only", preselected: true, threads: 8 });
  });
  it("macOS: no masks", () => {
    const p = defaultThreadPlan(topo(cores(10, { smt: false }), { source: "macos", masksSupported: false, perfCores: 6, effCores: 4 }));
    expect(p.t).toMatchObject({ threads: 6, mask: null });
    expect(p.tb).toMatchObject({ threads: 10, mask: null });
    expect(p.strict).toBe(false);
  });
  it("fallback topology never passes masks", () => {
    expect(defaultThreadPlan(fallbackTopology(8)).t).toMatchObject({ threads: 8, mask: null });
  });
});

describe("threadCandidates", () => {
  it("multi-CCD: +1 CCD first and pre-ticked, then 1 core, half, default", () => {
    const c = threadCandidates(topo(cores(16, { die: (i) => (i < 8 ? 0 : 1), l3: (i) => (i < 8 ? 0 : 1) })), "t");
    expect(c.map((x) => [x.kind, x.threads])).toEqual([
      ["two_ccd", 16],
      ["one_core", 1],
      ["half", 4],
      ["default", 8],
    ]);
    expect(c[0].preselected).toBe(true);
  });
  it("candidates are prefixes of the default set's masks", () => {
    const c = threadCandidates(topo(cores(6)), "t");
    expect(c.map((x) => x.mask)).toEqual(["0x1", "0x15", "0x555"]);
  });
});

describe("pickThreadWinner", () => {
  it("needs a gap larger than both stddevs combined", () => {
    expect(pickThreadWinner([
      { kind: "default", mean: 40, stddev: 2, samples: 3 },
      { kind: "half", mean: 43, stddev: 2, samples: 3 },
    ])).toEqual({ kind: "default_kept", reason: "no_clear_winner" });
    expect(pickThreadWinner([
      { kind: "default", mean: 40, stddev: 1, samples: 3 },
      { kind: "two_ccd", mean: 45, stddev: 1, samples: 3 },
    ])).toEqual({ kind: "winner", winner: "two_ccd" });
  });
  it("one repeat never replaces the default", () => {
    expect(pickThreadWinner([
      { kind: "default", mean: 40, stddev: 0, samples: 1 },
      { kind: "half", mean: 90, stddev: 0, samples: 1 },
    ])).toEqual({ kind: "default_kept", reason: "no_spread" });
  });
});

describe("processor groups", () => {
  it("no masks once a core sits beyond the first 64 CPUs", () => {
    const c = Array.from({ length: 40 }, (_, i) => ({ id: i, logical: [2 * i, 2 * i + 1], l3: 0, die: 0, effClass: 0, group: i < 32 ? 0 : 1 }));
    const p = defaultThreadPlan({ source: "windows", cores: c, logicalCount: 80, masksSupported: true });
    expect(p.t.mask).toBeNull();
    expect(p.t.threads).toBe(40);
    expect(p.note).toMatch(/processor groups/);
  });
});
