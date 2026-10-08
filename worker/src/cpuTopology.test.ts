import { describe, expect, it } from "vitest";
import { buildLinuxTopology, parseCpuList, synthesizeWindowsTopology } from "./cpuTopology.js";
import { defaultThreadPlan, threadCandidates } from "../../shared/threadPlan.js";

const win = (over: Partial<Parameters<typeof synthesizeWindowsTopology>[0]>) =>
  synthesizeWindowsTopology({ brand: "", manufacturer: "", logical: 12, physical: 6, efficiencyCores: 0, l3Bytes: null, ...over })!;

describe("synthesizeWindowsTopology", () => {
  it("Ryzen 5 5600X (this box's real si.cpu() reading)", () => {
    const t = win({ brand: "Ryzen 5 5600X 6-Core Processor", manufacturer: "AMD", l3Bytes: 33554432 });
    expect(t.cores).toHaveLength(6);
    expect(t.cores[1].logical).toEqual([2, 3]);
    expect(t.estimated).toBeUndefined();
    expect(defaultThreadPlan(t).t).toMatchObject({ threads: 6, mask: "0x555" });
  });
  it("Ryzen 9 7950X: two CCDs estimated from 64 MiB L3", () => {
    const t = win({ brand: "Ryzen 9 7950X 16-Core Processor", manufacturer: "AMD", logical: 32, physical: 16, l3Bytes: 64 * 2 ** 20 });
    expect(t.estimated).toBe(true);
    const p = defaultThreadPlan(t);
    expect(p.domains).toBe(2);
    expect(p.t.threads).toBe(8);
    expect(threadCandidates(t, "t")[0]).toMatchObject({ kind: "two_ccd", threads: 16 });
  });
  it("Intel hybrid reported with E-cores", () => {
    const t = win({ brand: "Core i9-12900K", manufacturer: "Intel", logical: 24, physical: 16, efficiencyCores: 8 });
    expect(t.cores.filter((c) => c.effClass === 1)).toHaveLength(8);
    expect(t.cores[8].logical).toEqual([16]);
    const p = defaultThreadPlan(t);
    expect([p.t.threads, p.tb.threads]).toEqual([8, 16]);
  });
  it("Intel hybrid inferred when E-cores are not reported", () => {
    const t = win({ brand: "Core i7-13700K", manufacturer: "Intel", logical: 24, physical: 16, efficiencyCores: 0 });
    expect(defaultThreadPlan(t).hybrid).toBe(true);
    expect(t.cores.reduce((a, c) => a + c.logical.length, 0)).toBe(24);
  });
  it("no SMT", () => {
    expect(win({ logical: 8, physical: 8 }).cores[3].logical).toEqual([3]);
  });
  it("rejects nonsense counts", () => {
    expect(synthesizeWindowsTopology({ brand: "", manufacturer: "", logical: 4, physical: 0, efficiencyCores: 0, l3Bytes: null })).toBeNull();
  });
});

describe("linux", () => {
  it("parses cpu lists", () => {
    expect(parseCpuList("0-3,8,10-11")).toEqual([0, 1, 2, 3, 8, 10, 11]);
  });
  it("groups SMT siblings into cores and L3 lists into domains", () => {
    const t = buildLinuxTopology([
      { cpu: 0, coreKey: "0:0:0", die: null, l3List: "0-3", l3SizeKib: 16384, effClass: 0 },
      { cpu: 1, coreKey: "0:0:1", die: null, l3List: "0-3", l3SizeKib: 16384, effClass: 0 },
      { cpu: 2, coreKey: "0:0:0", die: null, l3List: "0-3", l3SizeKib: 16384, effClass: 0 },
      { cpu: 3, coreKey: "0:0:1", die: null, l3List: "0-3", l3SizeKib: 16384, effClass: 0 },
      { cpu: 4, coreKey: "0:0:4", die: null, l3List: "4-5", l3SizeKib: 16384, effClass: 0 },
      { cpu: 5, coreKey: "0:0:4", die: null, l3List: "4-5", l3SizeKib: 16384, effClass: 0 },
    ])!;
    expect(t.cores.map((c) => c.logical)).toEqual([[0, 2], [1, 3], [4, 5]]);
    expect(t.cores.map((c) => c.l3)).toEqual([0, 0, 1]);
  });
});
