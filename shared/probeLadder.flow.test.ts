// The optimization flow's step 2 on top of the frontier search: explicit
// context stops (>= 4096), fit-seeded openers, per-stop caps, and the MoE
// placement ladder as the search axis (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md).
import { describe, expect, it } from "vitest";
import { claimFitsFree, nextLadderRung, probeOutcome, type LadderAttempt, type LadderInput, type LadderRung } from "./probeLadder.js";
import { detectMoe, placementLadder, fromFit, ladderIndex } from "./moePlacement.js";

const FREE = 7315;

function simulate(input: Omit<LadderInput, "history">, claim: (r: LadderRung) => number, cleanBelowMib: number) {
  const history: LadderAttempt[] = [];
  const full: LadderInput = { ...input, history };
  for (let guard = 0; guard < 400; guard++) {
    const next = nextLadderRung(full);
    if (!next) break;
    const c = claim(next);
    history.push({
      ...next,
      ok: c < cleanBelowMib,
      placementJudged: true,
      claimedMib: c,
      fitsFree: claimFitsFree({ claimedMib: c, freeMib: FREE, loaded: true }),
    });
  }
  return { history, outcome: probeOutcome(full) };
}

describe("frontier with explicit stops, seeds and caps", () => {
  const dense = (r: LadderRung) => 280 + 405 * r.ngl + r.ctx * 0.01;
  it("settles only the chosen stops (>= 4096) besides the anchor", () => {
    const { history, outcome } = simulate(
      { mode: "frontier", anchored: true, candidateCtx: 4096, candidateNgl: 0, nglMax: 65, maxCtx: 262144, freeVramMib: FREE, stops: [32768, 131072], calculateNgl: () => 14 },
      dense,
      6500
    );
    const contexts = new Set(history.map((h) => h.ctx));
    expect([...contexts].sort((a, b) => a - b)).toEqual([1024, 32768, 131072]);
    expect(outcome.curve!.map((c) => c.ctx)).toEqual([32768, 131072]);
    expect(outcome.curve!.every((c) => c.clean.resolved)).toBe(true);
  });

  it("opens each stop at fit's seed and never loads above the cap", () => {
    const seed: Record<number, number> = { 32768: 14, 131072: 12 };
    const cap: Record<number, number> = { 32768: 15, 131072: 13 };
    const { history } = simulate(
      {
        mode: "frontier", anchored: true, candidateCtx: 4096, candidateNgl: 0, nglMax: 65, maxCtx: 262144, freeVramMib: FREE,
        stops: [32768, 131072], calculateNgl: (ctx) => seed[ctx], capNgl: (ctx) => cap[ctx],
      },
      dense,
      99999
    );
    const at32 = history.filter((h) => h.ctx === 32768);
    expect(at32[0].ngl).toBe(14);
    expect(history.filter((h) => h.ctx === 32768).every((h) => h.ngl <= 15)).toBe(true);
    expect(history.filter((h) => h.ctx === 131072).every((h) => h.ngl <= 13)).toBe(true);
  });
});

describe("MoE placement ladder as the search axis", () => {
  // Qwen3.6-35B-A3B at 32k FA on, measured by fit-params (plan §1.3): ~3123 MiB
  // with every expert on CPU, ~370 MiB per expert layer brought back, the
  // boundary fractions in between; dense phase ~45 MiB per dense-only layer.
  const moe = detectMoe({ dense: Array(40).fill(30 * 2 ** 20), moe: Array(40).fill(364 * 2 ** 20), embed: 0, output: 0, other: 0 });
  const ladder = placementLadder(moe);
  const mib = (index: number, ctx: number) => {
    const st = ladder[index];
    const kv = ctx / 32768 * 702;
    if (st.phase === "dense") return 900 + st.ngl * 45 + (st.boundary === "ATTN" ? 20 : 0) + kv * (st.ngl / 41);
    const frac = st.boundary === "UP" ? 0.3 : st.boundary === "GATE" ? 0.6 : 0;
    return 2400 + kv + 370 * (st.expertsGpu + frac);
  };
  it("finds the expert-phase boundary within the load budget, seeded by fit", () => {
    // fit's own answer at 32k: experts of blocks 0..7 on GPU, block 8 GATE.
    const fitState = fromFit({ ngl: 41, ot: "blk\\.8\\.ffn_down.*=CPU,blk\\.9\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU" }, moe)!;
    const seedIndex = ladderIndex(ladder, fitState);
    expect(seedIndex).toBeGreaterThan(0);
    const { history, outcome } = simulate(
      {
        mode: "frontier", anchored: true, candidateCtx: 4096, candidateNgl: 0, nglMax: ladder.length - 1, maxCtx: 262144,
        freeVramMib: FREE, stops: [32768], calculateNgl: () => seedIndex,
      },
      (r) => mib(r.ngl, r.ctx),
      FREE - 400
    );
    expect(history.length).toBeLessThanOrEqual(40);
    const stop = outcome.curve![0];
    expect(stop.clean.resolved).toBe(true);
    const clean = ladder[stop.clean.value!];
    expect(clean.phase).toBe("expert");
    // The answer really is the last state below the clean line, and the next one isn't.
    expect(mib(stop.clean.value!, 32768)).toBeLessThan(FREE - 400);
    expect(mib(stop.clean.value! + 1, 32768)).toBeGreaterThanOrEqual(FREE - 400);
    // The anchor is -ngl 1 with every expert on CPU.
    expect(ladder[history[0].ngl]).toEqual({ phase: "dense", ngl: 1, expertsGpu: 0, boundary: "none" });
  });
});
