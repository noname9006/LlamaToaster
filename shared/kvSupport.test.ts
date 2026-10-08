import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseFitOutput } from "./fitParams.js";
import { classifyKvProbe, kvCandidatePairs, kvSizeGroups, type KvSupportRow } from "./kvSupport.js";

const fx = (n: string) =>
  parseFitOutput(readFileSync(join(__dirname, "__fixtures__", "fit", `${n}.stdout`), "utf8"), readFileSync(join(__dirname, "__fixtures__", "fit", `${n}.log`), "utf8"));

describe("classifyKvProbe on real Vulkan b11226 probes", () => {
  const baseOff = fx("kv-f16-f16-off");
  const baseOn = fx("kv-f16-f16-on");
  it("quant K without FA falls back to the CPU", () => {
    expect(classifyKvProbe({ ctk: "q8_0", ctv: "f16", fa: "off", probe: fx("kv-q8_0-f16-off"), baseline: baseOff, backend: "vulkan" }).status).toBe("cpu_fallback");
  });
  it("quant V without FA cannot create a context", () => {
    const r = classifyKvProbe({ ctk: "f16", ctv: "q8_0", fa: "off", probe: fx("kv-f16-q8_0-off"), baseline: baseOff, backend: "vulkan" });
    expect(r.status).toBe("invalid");
    expect(r.detail).toMatch(/llama_context/);
  });
  it("bf16 mixed with a quant falls back, q8_0/q8_0 with FA is fine", () => {
    expect(classifyKvProbe({ ctk: "bf16", ctv: "q8_0", fa: "on", probe: fx("kv-bf16-q8_0-on"), baseline: baseOn, backend: "vulkan" }).status).toBe("cpu_fallback");
    expect(classifyKvProbe({ ctk: "q8_0", ctv: "q8_0", fa: "on", probe: fx("kv-q8_0-q8_0-on"), baseline: baseOn, backend: "vulkan" }).status).toBe("ok");
  });
  it("FA turned off by the backend", () => {
    expect(classifyKvProbe({ ctk: "bf16", ctv: "f16", fa: "on", probe: fx("kv-bf16-f16-auto"), baseline: baseOn, backend: "vulkan" }).status).toBe("fa_disabled");
  });
  it("CUDA marks pairs outside the default vec kernels", () => {
    expect(classifyKvProbe({ ctk: "q8_0", ctv: "q4_0", fa: "on", probe: baseOn, baseline: baseOn, backend: "cuda" }).status).toBe("cuda_slow");
    expect(classifyKvProbe({ ctk: "q8_0", ctv: "q8_0", fa: "on", probe: baseOn, baseline: baseOn, backend: "cuda" }).status).toBe("ok");
  });
});

describe("kvSizeGroups", () => {
  it("one group per size, FA on preferred, symmetric pair as representative", () => {
    const rows: KvSupportRow[] = kvCandidatePairs().flatMap(([ctk, ctv]) =>
      (["on", "off"] as const).map((fa) => ({ ctk, ctv, fa, status: fa === "on" ? "ok" : "invalid", splits: 2, baseline_splits: 2, detail: null }) as KvSupportRow)
    );
    const groups = kvSizeGroups(rows);
    expect(groups).toHaveLength(18);
    expect(groups.every((g) => g.fa === "on")).toBe(true);
    expect(groups[0].rep).toEqual(["f16", "f16"]);
    const q4 = groups.find((g) => g.pairs.some(([k, v]) => k === "q4_0" && v === "q4_0"))!;
    expect(q4.rep).toEqual(["q4_0", "q4_0"]);
    expect(q4.pairs).toHaveLength(4);
  });
  it("skips unusable pairs entirely", () => {
    expect(kvSizeGroups([{ ctk: "q8_0", ctv: "f16", fa: "off", status: "cpu_fallback", splits: 98, baseline_splits: 2, detail: null }])).toEqual([]);
  });
});
