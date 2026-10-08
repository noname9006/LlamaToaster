import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { kvRowsComplete, runFitMap, type FitCallFn } from "./fitMap.js";
import type { FitPoint } from "../../shared/fitParams.js";
import type { KvSupportRow } from "../../shared/kvSupport.js";

const dir = join(__dirname, "..", "..", "shared", "__fixtures__", "fit");
const fx = (name: string) => ({
  stdout: readFileSync(join(dir, `${name}.stdout`), "utf8"),
  log: readFileSync(join(dir, `${name}.log`), "utf8"),
  code: 0,
  timedOut: false,
  ms: 1,
});
const arg = (args: string[], flag: string) => args[args.indexOf(flag) + 1];

// Fit calls answer with the 27B partial log; KV probes answer like the 1B
// model's real probes did (q8_0 K without FA falls back, quant V without FA
// fails, everything else matches the baseline).
const fakeCall: FitCallFn = async (args) => {
  if (args.includes("-ngl")) {
    const k = arg(args, "-ctk"), v = arg(args, "-ctv"), fa = arg(args, "-fa");
    if (fa === "off" && v !== "f16" && v !== "bf16") return { ...fx("kv-f16-q8_0-off"), code: 1 };
    if (fa === "off" && k !== "f16" && k !== "bf16") return fx("kv-q8_0-f16-off");
    return fx(fa === "off" ? "kv-f16-f16-off" : "kv-f16-f16-on");
  }
  return arg(args, "-fa") === "off" ? fx("hybrid27b-faoff") : fx("hybrid27b-partial");
};

function run(over: Partial<Parameters<typeof runFitMap>[0]> = {}) {
  const points: FitPoint[] = [];
  const kv: KvSupportRow[][] = [];
  const progress: string[] = [];
  return runFitMap({
    spec: { margin_mib: 1024, ctx_stops: [4096, 8192, 32768], machine: "dgpu", kv_detect: true, kv_fit: true },
    modelPath: "m.gguf",
    backend: "vulkan",
    deviceArgs: [],
    ramBudgetMib: null,
    call: fakeCall,
    report: { point: (p) => void points.push(p), kvRows: (r) => void kv.push(r), progress: (_d, _t, s) => void progress.push(s) },
    shouldStop: () => false,
    ...over,
  }).then((out) => ({ out, points, kv, progress }));
}

describe("runFitMap", () => {
  it("walks f16 FA on/off largest context first, detects KV, then one path per KV size", async () => {
    const { out, points, kv } = await run();
    expect(points.slice(0, 3).map((p) => [p.fa, p.ctx])).toEqual([["on", 32768], ["on", 8192], ["on", 4096]]);
    expect(points.slice(3, 6).every((p) => p.fa === "off")).toBe(true);
    expect(kv).toHaveLength(1);
    expect(kv[0]).toHaveLength(128);
    expect(kv[0].find((r) => r.ctk === "q8_0" && r.ctv === "f16" && r.fa === "off")?.status).toBe("cpu_fallback");
    expect(kv[0].find((r) => r.ctk === "f16" && r.ctv === "q8_0" && r.fa === "off")?.status).toBe("invalid");
    // 64 pairs at FA on are ok -> 17 non-f16 sizes x 3 stops.
    const kvPoints = points.slice(6);
    expect(kvPoints).toHaveLength(17 * 3);
    expect(kvPoints.every((p) => p.fa === "on")).toBe(true);
    expect(out.calls).toBe(6 + 2 * 64 + 17 * 3);
    expect(points[0].verdict).toBe("partial");
  });

  it("reuses complete known KV rows instead of probing again", async () => {
    const first = await run({ spec: { margin_mib: 1024, ctx_stops: [4096], machine: "dgpu", kv_detect: true, kv_fit: false } });
    expect(kvRowsComplete(first.kv[0])).toBe(true);
    const second = await run({
      spec: { margin_mib: 1024, ctx_stops: [4096], machine: "dgpu", kv_detect: true, kv_fit: false },
      kvKnown: first.kv[0],
    });
    expect(second.out.calls).toBe(2);
    expect(second.kv[0]).toHaveLength(128);
  });

  it("stops between calls when asked", async () => {
    let n = 0;
    const { out, points } = await run({ shouldStop: () => ++n > 2 });
    expect(out.stopped).toBe(true);
    expect(points).toHaveLength(2);
  });

  it("a timed-out call becomes an error point and the map continues", async () => {
    const { points } = await run({
      spec: { margin_mib: 1024, ctx_stops: [4096], machine: "dgpu", kv_detect: false, kv_fit: false },
      call: async () => ({ stdout: "", log: "", code: null, timedOut: true, ms: 60000 }),
    });
    expect(points.map((p) => p.verdict)).toEqual(["error", "error"]);
    expect(points[0].error).toMatch(/no answer within 60 s/);
  });
});
