import { describe, expect, it } from "vitest";
import {
  baselineCandidate,
  batchCandidates,
  candidateExtraServerArgs,
  hasBatchArgs,
  candidateKey,
  compareToBaseline,
  dedupeCandidates,
  serverCommandLine,
  shellQuote,
  speedWorkload,
  threadSweepBenchArgs,
  validateSpeedRunSpec,
  validateThreadSweepSpec,
  type SpeedCandidate,
} from "./speedRun.js";

const OT = "blk\\.8\\.ffn_down.*=CPU,blk\\.9\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU";
const fitCand: SpeedCandidate = {
  label: "Most layers",
  source: "fit",
  ngl: 41,
  ot: OT,
  ctk: "q8_0",
  ctv: "q8_0",
  fa: "on",
  threads: { t: 6, tb: 6, mask: "0x555", maskBatch: "0x555", strict: true },
};

describe("speedWorkload", () => {
  it("4096 + 512 with 256 tokens of headroom", () => {
    expect(speedWorkload(32768)).toEqual({ promptTokens: 4096, nGen: 512, shrunk: false });
    expect(speedWorkload(4096)).toEqual({ promptTokens: 3328, nGen: 512, shrunk: true });
    expect(speedWorkload(4864)).toEqual({ promptTokens: 4096, nGen: 512, shrunk: false });
  });
});

describe("args", () => {
  it("adds only what the base server args lack", () => {
    expect(candidateExtraServerArgs(fitCand)).toEqual(["-ot", OT, "-tb", "6", "-C", "0x555", "-Cb", "0x555", "--cpu-strict", "1"]);
    expect(candidateExtraServerArgs(baselineCandidate())).toEqual([]);
  });
  it("llama-bench wants ';' between -ot entries", () => {
    const a = threadSweepBenchArgs(OT, { role: "t", kind: "default", label: "x", threads: 6, mask: "0x555", strict: true });
    expect(a[1]).toBe(OT.replace(",", ";"));
    expect(a.slice(2)).toEqual(["-C", "0x555", "--cpu-strict", "1"]);
  });
  it("the copyable command keeps fit's regex intact", () => {
    const cmd = serverCommandLine("model.gguf", 32768, fitCand);
    expect(cmd).toContain(`-ot '${OT}'`);
    expect(cmd.startsWith("llama-server -m model.gguf -c 32768 -ngl 41 -fa on -ctk q8_0 -ctv q8_0")).toBe(true);
    expect(cmd.endsWith("--fit off")).toBe(true);
    expect(shellQuote("it's")).toBe(`'it'"'"'s'`);
  });
});

describe("batch sizes", () => {
  it("the batch step measures each -ub with -b at least 2048, default included", () => {
    const list = batchCandidates(fitCand);
    expect(list.map((c) => [c.b, c.ub])).toEqual([[2048, 256], [2048, 512], [2048, 1024], [2048, 2048]]);
    expect(list.every((c) => c.ot === fitCand.ot && c.ctk === "q8_0")).toBe(true);
    expect(new Set(list.map(candidateKey)).size).toBe(4);
  });
  it("-b/-ub appear in the command only when set, and old keys are unchanged", () => {
    expect(serverCommandLine("m.gguf", 32768, fitCand)).not.toContain("-ub");
    expect(serverCommandLine("m.gguf", 32768, { ...fitCand, b: 4096, ub: 1024 })).toContain("-b 4096 -ub 1024");
    expect(candidateKey(fitCand)).toBe(JSON.stringify([41, OT, "q8_0", "q8_0", "on", fitCand.threads]));
    expect(hasBatchArgs(fitCand)).toBe(false);
    expect(hasBatchArgs({ ub: 256 })).toBe(true);
  });
  it("validation wants powers of two and ub <= b", () => {
    const spec = { target_ctx: 32768, prompt_tokens: 4096, n_gen: 512, repeats: 3 };
    expect(validateSpeedRunSpec({ ...spec, candidates: batchCandidates(fitCand) }, null)).toBeNull();
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, ub: 1000 }] }, null)).toMatch(/power of two/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, b: 512, ub: 1024 }] }, null)).toMatch(/ub must not exceed b/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, ub: 4096 }] }, null)).toMatch(/ub must not exceed b/);
  });
});

describe("dedupe", () => {
  it("merges identical configs regardless of label", () => {
    const list = dedupeCandidates([fitCand, { ...fitCand, label: "again" }, { ...fitCand, ngl: 40 }]);
    expect(list).toHaveLength(2);
    expect(candidateKey(fitCand)).not.toBe(candidateKey({ ...fitCand, threads: null }));
  });
});

describe("compareToBaseline", () => {
  it("calls a gap clear only beyond both spreads, never without spread", () => {
    expect(compareToBaseline({ mean: 110, stddev: 2, samples: 3 }, { mean: 100, stddev: 2, samples: 3 })).toMatchObject({ clear: true });
    expect(compareToBaseline({ mean: 103, stddev: 2, samples: 3 }, { mean: 100, stddev: 2, samples: 3 })).toMatchObject({ clear: false });
    expect(compareToBaseline({ mean: 150, stddev: 0, samples: 1 }, { mean: 100, stddev: 0, samples: 1 })).toMatchObject({ clear: false, noSpread: true });
  });
});

describe("validateSpeedRunSpec", () => {
  const spec = { target_ctx: 32768, prompt_tokens: 4096, n_gen: 512, repeats: 3, candidates: [baselineCandidate(), fitCand] };
  it("accepts a real spec", () => {
    expect(validateSpeedRunSpec(spec, 262144)).toBeNull();
  });
  it("rejects bad workloads, repeats, overrides and masks", () => {
    expect(validateSpeedRunSpec({ ...spec, prompt_tokens: 8000 }, 262144)).toMatch(/4096-token prompt/);
    expect(validateSpeedRunSpec({ ...spec, target_ctx: 300000 }, 262144)).toMatch(/target_ctx/);
    expect(validateSpeedRunSpec({ ...spec, repeats: 0 }, null)).toMatch(/repeats/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, ot: "x;rm -rf /=CPU" }] }, null)).toMatch(/ot/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, ctk: "f32" }] }, null)).toMatch(/cache/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [{ ...fitCand, threads: { ...fitCand.threads!, mask: "555" } }] }, null)).toMatch(/mask/);
    expect(validateSpeedRunSpec({ ...spec, candidates: [] }, null)).toMatch(/candidates/);
  });
});

describe("validateThreadSweepSpec", () => {
  const spec = {
    placement: { ngl: 16, ot: null, ctk: "f16", ctv: "f16", fa: "on" },
    repeats: 3,
    items: [{ role: "t", kind: "default", label: "all", threads: 6, mask: "0x555", strict: true }],
  };
  it("accepts and rejects", () => {
    expect(validateThreadSweepSpec(spec)).toBeNull();
    expect(validateThreadSweepSpec({ ...spec, placement: { ...spec.placement, fa: "auto" } })).toMatch(/fa/);
    expect(validateThreadSweepSpec({ ...spec, items: [{ ...spec.items[0], role: "x" }] })).toMatch(/role/);
    expect(validateThreadSweepSpec({ ...spec, items: [{ ...spec.items[0], threads: 0 }] })).toMatch(/threads/);
  });
});
