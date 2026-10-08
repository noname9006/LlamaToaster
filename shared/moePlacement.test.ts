import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describePlacement, detectMoe, fromFit, ladderIndex, placementLadder, toArgs } from "./moePlacement.js";
import { parseFitArgs } from "./fitParams.js";

const fit = (name: string) => parseFitArgs(readFileSync(join(__dirname, "__fixtures__", "fit", `${name}.stdout`), "utf8"));
const tlb = (nBlocks: number, moe: (i: number) => number) => ({
  dense: Array.from({ length: nBlocks }, () => 30 * 2 ** 20),
  moe: Array.from({ length: nBlocks }, (_, i) => moe(i)),
  embed: 0,
  output: 0,
  other: 0,
});

const qwen35b = detectMoe(tlb(40, () => 364 * 2 ** 20));
const gemma26b = detectMoe(tlb(30, () => 408 * 2 ** 20));

describe("detectMoe", () => {
  it("reads MoE blocks from expert bytes, skipping leading dense blocks", () => {
    const m = detectMoe(tlb(6, (i) => (i < 2 ? 0 : 100)));
    expect(m.isMoe).toBe(true);
    expect(m.moeLayers).toEqual([2, 3, 4, 5]);
    expect(detectMoe(tlb(4, () => 0)).isMoe).toBe(false);
    expect(detectMoe(null).isMoe).toBe(false);
  });
});

describe("placementLadder", () => {
  it("dense models are plain -ngl 0..all", () => {
    const l = placementLadder(detectMoe(tlb(4, () => 0)));
    expect(l.map((s) => s.ngl)).toEqual([0, 1, 2, 3, 4, 5]);
  });
  it("MoE: dense phase, then expert phase front-first with UP/GATE between whole steps", () => {
    const l = placementLadder(qwen35b);
    const firstExpert = l.findIndex((s) => s.phase === "expert");
    expect(l[firstExpert]).toEqual({ phase: "expert", ngl: 41, expertsGpu: 0, boundary: "none" });
    expect(l.slice(firstExpert, firstExpert + 4).map((s) => s.boundary)).toEqual(["none", "UP", "GATE", "none"]);
    expect(l[l.length - 1]).toEqual({ phase: "expert", ngl: 41, expertsGpu: 40, boundary: "none" });
  });
  it("drops boundary states fit-params didn't size strictly larger", () => {
    const l = placementLadder(gemma26b, { dropFractions: new Set(["UP"]) });
    expect(l.some((s) => s.boundary === "UP")).toBe(false);
    expect(l.some((s) => s.boundary === "GATE")).toBe(true);
  });
});

describe("toArgs / fromFit", () => {
  it("round-trips every state", () => {
    for (const moe of [qwen35b, gemma26b, detectMoe(tlb(6, (i) => (i < 2 ? 0 : 100)))]) {
      const l = placementLadder(moe);
      l.forEach((s, i) => {
        const back = fromFit(toArgs(s, moe), moe);
        expect(back && ladderIndex(l, back), JSON.stringify(s)).toBe(i);
      });
    }
  });
  it("maps fit's own expert-phase answer exactly (boundary block 8 = GATE)", () => {
    const s = fromFit(fit("moe35b"), qwen35b);
    expect(s).toEqual({ phase: "expert", ngl: 41, expertsGpu: 8, boundary: "GATE" });
    expect(describePlacement(s!, qwen35b)).toBe("all 41 layers on GPU · experts of layers 8–39 on CPU (layer 8: up+gate on GPU)");
    expect(fromFit(fit("moe-gemma26b"), gemma26b)).toEqual({ phase: "expert", ngl: 31, expertsGpu: 8, boundary: "GATE" });
  });
  it("maps fit's dense-phase answer", () => {
    const s = fromFit(fit("moe35b-dense-phase"), qwen35b);
    expect(s).toEqual({ phase: "dense", ngl: 5, expertsGpu: 0, boundary: "none" });
    expect(describePlacement(s!, qwen35b)).toBe("5 layers on GPU, all experts on CPU");
  });
  it("whole-model answers", () => {
    expect(fromFit({ ngl: -1, ot: null }, qwen35b)).toEqual({ phase: "expert", ngl: 41, expertsGpu: 40, boundary: "none" });
    expect(fromFit({ ngl: 16, ot: null }, detectMoe(tlb(64, () => 0)))).toEqual({ phase: "dense", ngl: 16, expertsGpu: 0, boundary: "none" });
  });
});
