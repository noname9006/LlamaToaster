import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  buildFitArgs,
  fitCtxStops,
  fitPointFromOutput,
  keepFitLogLine,
  kvSizeKey,
  KV_CACHE_TYPES,
  otForBench,
  parseFitArgs,
  parseFitOutput,
} from "./fitParams.js";

const dir = join(__dirname, "__fixtures__", "fit");
function fixture(name: string) {
  return parseFitOutput(readFileSync(join(dir, `${name}.stdout`), "utf8"), readFileSync(join(dir, `${name}.log`), "utf8"));
}
const ctx = (over: Partial<Parameters<typeof fitPointFromOutput>[1]> = {}) => ({
  ctx: 32768,
  fa: "on" as const,
  ctk: "f16",
  ctv: "f16",
  marginMib: 1024,
  ...over,
});

describe("fitCtxStops", () => {
  it("starts at 4096 and ends at the trained context", () => {
    expect(fitCtxStops(262144)).toEqual([4096, 8192, 16384, 32768, 65536, 131072, 262144]);
    expect(fitCtxStops(40960)).toEqual([4096, 8192, 16384, 32768, 40960]);
    expect(fitCtxStops(2048)).toEqual([4096]);
    expect(fitCtxStops(null)).toEqual([4096]);
  });
});

describe("buildFitArgs", () => {
  it("never passes -ngl for a fit-map call and always passes -c", () => {
    const a = buildFitArgs({ modelPath: "m.gguf", ctx: 32768, fa: "on", marginMib: 1536 });
    expect(a).not.toContain("-ngl");
    expect(a.slice(0, 4)).toEqual(["-m", "m.gguf", "-c", "32768"]);
    expect(a).toContain("-fitt");
    expect(a[a.indexOf("-fitt") + 1]).toBe("1536");
    expect(a).toContain("-v");
  });
  it("passes -ngl 99 on a KV probe so fit skips its search", () => {
    const a = buildFitArgs({ modelPath: "m", ctx: 4096, fa: "off", ctk: "q8_0", ctv: "f16", kvProbe: true, deviceArgs: ["-dev", "none"] });
    expect(a.join(" ")).toContain("-ngl 99");
    expect(a).not.toContain("-fitt");
    expect(a.slice(-2)).toEqual(["-dev", "none"]);
  });
});

describe("keepFitLogLine", () => {
  it("keeps decision lines and drops loader noise", () => {
    expect(keepFitLogLine("0.01 I common_params_fit_impl: id=0, target=6204 MiB")).toBe(true);
    expect(keepFitLogLine("0.01 I llama_model_loader: - tensor   12: blk.1.attn_q.weight")).toBe(false);
    expect(keepFitLogLine("0.01 I sched_reserve: graph: nodes = 1638, splits = 2, input objects = 4")).toBe(true);
  });
});

describe("parseFitOutput + fitPointFromOutput on real b11226 logs", () => {
  it("dense model that fits whole", () => {
    const p = fixture("dense-fits");
    expect(p.ngl).toBe(-1);
    expect(p.noChangesNeeded).toBe(true);
    expect(p.nLayer).toBe(48);
    const pt = fitPointFromOutput(p, ctx());
    expect(pt.verdict).toBe("full");
    expect(pt.layers_gpu).toBe(49);
    expect(pt.layers_total).toBe(49);
    expect(pt.dev_used_mib).toBe(1931);
    expect(pt.host_used_mib).toBe(173);
    expect(pt.total_mib).toBe(2104);
    expect(pt.kv_mib).toBe(768);
    expect(pt.splits).toBe(2);
  });

  it("hybrid 27B partial offload picks the chosen trial's table, not the last one", () => {
    const p = fixture("hybrid27b-partial");
    expect(p.ngl).toBe(16);
    expect(p.finalDevices).toEqual([
      { name: "Vulkan0 (AMD Radeon RX 6600 XT)", layers: 16, overflowing: 0, usedMib: 6156, freeMib: 1072 },
    ]);
    const pt = fitPointFromOutput(p, ctx());
    expect(pt.verdict).toBe("partial");
    expect(pt.layers_gpu).toBe(16);
    expect(pt.layers_total).toBe(65);
    expect(pt.host_used_mib).toBe(15184);
    expect(pt.total_mib).toBe(6156 + 15184);
    expect(pt.need_all_gpu_mib).toBe(20530);
  });

  it("FA off costs layers on the same model", () => {
    expect(fitPointFromOutput(fixture("hybrid27b-faoff"), ctx({ fa: "off" })).layers_gpu).toBe(12);
  });

  it("MoE expert phase: all layers, experts of the back layers on CPU, boundary fraction", () => {
    const p = fixture("moe35b");
    expect(p.ngl).toBe(41);
    expect(p.ot).toMatch(/^blk\\\.8\\\.ffn_down\.\*=CPU,blk\\\.9\\\.ffn_\(up\|down\|gate_up\|gate\)_\(ch\|\)exps=CPU/);
    expect(p.nExpert).toBe(256);
    expect(p.nExpertUsed).toBe(8);
    const pt = fitPointFromOutput(p, ctx());
    expect(pt.verdict).toBe("partial");
    expect(pt.layers_gpu).toBe(41);
    expect(pt.overflow_layers).toBe(33);
    expect(pt.dev_used_mib).toBe(6250);
    expect(otForBench(pt.ot!)).toContain(";blk\\.9\\.");
  });

  it("gemma-4 MoE (fused gate_up experts)", () => {
    const pt = fitPointFromOutput(fixture("moe-gemma26b"), ctx());
    expect(pt.verdict).toBe("partial");
    expect(pt.layers_gpu).toBe(31);
    expect(pt.ot).toContain("blk\\.30\\.");
  });

  it("MoE dense phase: a few layers, their experts still on CPU", () => {
    const p = fixture("moe35b-dense-phase");
    expect(p.ngl).toBe(5);
    const pt = fitPointFromOutput(p, ctx({ ctx: 262144, marginMib: 4500 }));
    expect(pt.verdict).toBe("partial");
    expect(pt.layers_gpu).toBe(5);
    expect(pt.ot).toContain("blk\\.37\\.");
  });

  it("cannot fit: fit still answers -ngl 0 with exit 0; free < margin is the verdict", () => {
    const p = fixture("cannot-fit");
    expect(p.ngl).toBe(0);
    const pt = fitPointFromOutput(p, ctx({ ctx: 131072, marginMib: 7300 }));
    expect(pt.verdict).toBe("doesnt_fit");
    expect(pt.reason).toBe("vram");
  });

  it("CPU-only fits and doesn't fit", () => {
    const ok = fitPointFromOutput(fixture("cpu-only"), ctx({ cpuOnly: true }));
    expect(ok.verdict).toBe("cpu_only");
    expect(ok.host_used_mib).toBe(2069);
    expect(ok.total_mib).toBe(2069);
    const no = fitPointFromOutput(fixture("cpu-doesnt-fit"), ctx({ cpuOnly: true, marginMib: 40000 }));
    expect(no.verdict).toBe("doesnt_fit");
    expect(no.reason).toBe("ram");
  });

  it("RAM budget turns a device fit into doesn't-fit", () => {
    const pt = fitPointFromOutput(fixture("hybrid27b-partial"), ctx({ ramBudgetMib: 8000 }));
    expect(pt.verdict).toBe("doesnt_fit");
    expect(pt.reason).toBe("ram");
  });

  it("KV probes: splits, FA fallback and the context-creation failure", () => {
    expect(fixture("kv-f16-f16-off").splits).toBe(2);
    expect(fixture("kv-q8_0-f16-off").splits).toBe(98);
    expect(fixture("kv-bf16-q8_0-on").splits).toBe(98);
    expect(fixture("kv-q8_0-q8_0-on").splits).toBe(2);
    expect(fixture("kv-bf16-f16-auto").faDisabled).toBe(true);
    const bad = fixture("kv-f16-q8_0-off");
    expect(bad.error).toMatch(/failed to create llama_context/);
    expect(fitPointFromOutput(bad, ctx({ ctx: 4096 })).verdict).toBe("error");
  });
});

describe("parseFitArgs", () => {
  it("reads -ngl all", () => {
    expect(parseFitArgs("-c 4096 -ngl all").ngl).toBe(-1);
  });
});

describe("kvSizeKey", () => {
  it("collapses the 64 non-f32 pairs into 18 sizes", () => {
    const keys = new Set<string>();
    for (const k of KV_CACHE_TYPES) for (const v of KV_CACHE_TYPES) keys.add(kvSizeKey(k, v));
    expect(keys.size).toBe(18);
    expect(kvSizeKey("q4_0", "q8_0")).toBe(kvSizeKey("q8_0", "q4_0"));
    expect(kvSizeKey("q4_0", "q8_0", false)).not.toBe(kvSizeKey("q8_0", "q4_0", false));
  });
});
