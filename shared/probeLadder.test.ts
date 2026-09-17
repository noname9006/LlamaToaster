import { describe, expect, it } from "vitest";
import {
  claimFitsFree,
  claimFitsFreeByDevice,
  computeCtxStops,
  ctxLadderStops,
  isProbeMode,
  nextLadderRung,
  predictFitNgl,
  probeOutcome,
  PROBE_ANCHOR_NGL,
  PROBE_LADDER_MIN_CTX,
  PROBE_MAX_LOADS,
  snapToSafeCtx,
  type LadderAttempt,
  type LadderInput,
  type LadderRung,
  type ProbeMode,
} from "./probeLadder.js";

const TRAINED = 262_144;
const LAYERS = 41;
const FREE = 7378;

// A machine shaped on the RX 6600 XT sweep (docs/research/spill-tax-dataset.json):
// ~405MiB claimed per layer, 7,378MiB free per --list-devices, and spill that
// grows steadily with neither layers (5 spills, 6 is clean) nor context (8
// spills at 1k-8k and not at 16k).
interface Machine {
  claim: (r: LadderRung) => number;
  clean: (r: LadderRung, loadIndex: number) => boolean;
}

const cleanLayers = (ctx: number): Set<number> =>
  ctx <= 8192
    ? new Set([0, 1, 2, 3, 4, 6])
    : ctx <= 32_768
      ? new Set([0, 1, 2, 3, 4, 6, 8])
      : ctx <= 131_072
        ? new Set([0, 1, 2])
        : new Set<number>();

const CARD: Machine = {
  claim: (r) => 280 + 405 * r.ngl + r.ctx * 0.004,
  clean: (r) => cleanLayers(r.ctx).has(r.ngl),
};

function measure(machine: Machine, r: LadderRung, history: LadderAttempt[], free: number | null): LadderAttempt {
  const claim = machine.claim(r);
  const index = history.filter((h) => h.ctx === r.ctx && h.ngl === r.ngl).length;
  return {
    ...r,
    ok: machine.clean(r, index),
    placementJudged: true,
    claimedMib: claim,
    fitsFree: claimFitsFree({ claimedMib: claim, freeMib: free, loaded: true }),
  };
}

function run(
  mode: ProbeMode,
  opts: { ctx?: number; ngl?: number; machine?: Machine; free?: number | null; maxLoads?: number; maxCtx?: number; anchored?: boolean } = {}
): { history: LadderAttempt[]; input: LadderInput } {
  const machine = opts.machine ?? CARD;
  const free = opts.free === undefined ? FREE : opts.free;
  const history: LadderAttempt[] = [];
  const input: LadderInput = {
    mode,
    candidateCtx: opts.ctx ?? 1024,
    candidateNgl: opts.ngl ?? 10,
    nglMax: LAYERS,
    maxCtx: opts.maxCtx ?? TRAINED,
    maxLoads: opts.maxLoads ?? 200,
    history,
    freeVramMib: free,
    calculateNgl: () => 12,
    anchored: opts.anchored,
  };
  for (let guard = 0; guard < 300; guard++) {
    const next = nextLadderRung(input);
    if (next == null) return { history, input };
    history.push(measure(machine, next, history, free));
  }
  throw new Error("ladder did not terminate");
}

const key = (r: LadderRung) => `${r.ctx}/${r.ngl}`;

describe("vocabulary", () => {
  it("defaults to 40 loads", () => {
    expect(PROBE_MAX_LOADS).toBe(40);
  });

  it("no longer accepts the removed modes", () => {
    for (const gone of ["max_gpu", "max_context", "balanced"]) expect(isProbeMode(gone)).toBe(false);
    for (const kept of ["frontier", "keep_context", "fixed_offload", "custom"]) expect(isProbeMode(kept)).toBe(true);
  });
});

describe("claimFitsFree", () => {
  it("strictly below free fits, a load that never allocated does not, missing readings are unknown", () => {
    expect(claimFitsFree({ claimedMib: 7274, freeMib: 7378, loaded: true })).toBe(true);
    expect(claimFitsFree({ claimedMib: 7378, freeMib: 7378, loaded: true })).toBe(false);
    expect(claimFitsFree({ claimedMib: null, freeMib: 7378, loaded: false })).toBe(false);
    expect(claimFitsFree({ claimedMib: null, freeMib: 7378, loaded: true })).toBeNull();
    expect(claimFitsFree({ claimedMib: 7000, freeMib: null, loaded: true })).toBeNull();
  });
});

describe("claimFitsFreeByDevice", () => {
  it("needs every used device to fit its own free memory -- an integrated GPU's room cannot hide a full card", () => {
    const free = { Vulkan0: 7378, Vulkan1: 15000 };
    expect(claimFitsFreeByDevice({ Vulkan0: 7000, Vulkan1: 200 }, free)).toBe(true);
    // 7,500 + 200 is far below the 22,378 total, but Vulkan0 alone is over.
    expect(claimFitsFreeByDevice({ Vulkan0: 7500, Vulkan1: 200 }, free)).toBe(false);
  });

  it("is unknown when a used device was not listed, or nothing was claimed", () => {
    expect(claimFitsFreeByDevice({ CUDA0: 100 }, { Vulkan0: 7378 })).toBeNull();
    expect(claimFitsFreeByDevice({ Vulkan0: 0 }, { Vulkan0: 7378 })).toBeNull();
    expect(claimFitsFreeByDevice(null, { Vulkan0: 7378 })).toBeNull();
  });
});

describe("predictFitNgl", () => {
  const at = (ctx: number, ngl: number): LadderAttempt => ({ ctx, ngl, ok: false, claimedMib: CARD.claim({ ctx, ngl }) });

  it("errs low from one load, and reads the per-layer claim from two", () => {
    expect(predictFitNgl([at(1024, 12)], 1024, FREE)).toBe(17);
    expect(predictFitNgl([at(1024, 12)], 1024, FREE)!).toBeLessThanOrEqual(17);
    expect(predictFitNgl([at(1024, 12), at(1024, 17)], 1024, FREE)).toBe(17);
  });

  it("assumes the fixed part does not grow until a second context is measured, then interpolates", () => {
    const floor = [at(1024, 12), at(1024, 17)];
    expect(predictFitNgl(floor, 262_144, FREE)).toBe(17);
    expect(predictFitNgl([...floor, at(262_144, 17)], 262_144, FREE)).toBe(14);
    expect(predictFitNgl([...floor, at(262_144, 17)], 65_536, FREE)).toBe(16);
  });
});

describe("the Wizard (frontier)", () => {
  it("opens at the estimate, never at every layer, and settles target 2 at the smallest context in three loads", () => {
    const { history } = run("frontier");
    expect(history[0]).toEqual(expect.objectContaining({ ctx: 1024, ngl: 12 }));
    expect(history.some((h) => h.ngl === LAYERS)).toBe(false);
    expect(history.slice(0, 3).map(key)).toEqual(["1024/12", "1024/17", "1024/18"]);
  });

  it("finds where spill starts, then the two layers above it -- 6 past a spilling 5", () => {
    const { history, input } = run("frontier");
    const stop = probeOutcome(input).curve![0];
    expect(stop.fit).toMatchObject({ value: 17, resolved: true });
    expect(stop.clean).toMatchObject({ value: 6, resolved: true, control: "confirmed" });
    // 7 is where spill starts above 6; 8 and 9 are the two layers above it.
    for (const n of [7, 8, 9]) expect(history.some((h) => h.ctx === 1024 && h.ngl === n)).toBe(true);
  });

  it("finishes each context before the next, with the control after the next context's target 2", () => {
    const { history } = run("frontier");
    const order = history.map(key);
    const fit2k = order.indexOf("2048/17");
    const control1k = order.lastIndexOf("1024/6");
    const clean2k = order.indexOf("2048/6");
    expect(order.indexOf("1024/6")).toBeLessThan(fit2k);
    expect(fit2k).toBeLessThan(control1k);
    expect(control1k).toBeLessThan(clean2k);
    expect(order.findIndex((k) => k.startsWith("2048/"))).toBeGreaterThan(order.indexOf("1024/18"));
  });

  it("never tries more layers for no spill than the smaller context held clean -- 8 at 16k stays untried", () => {
    const { history, input } = run("frontier");
    expect(history.some((h) => h.ctx === 16_384 && h.ngl === 8)).toBe(false);
    const curve = probeOutcome(input).curve!;
    expect(curve.find((s) => s.ctx === 16_384)!.clean.value).toBe(6);
    expect(curve.find((s) => s.ctx === 65_536)!.clean.value).toBe(2);
    expect(curve.find((s) => s.ctx === TRAINED)!.clean).toMatchObject({ value: null, resolved: true });
  });

  it("resolves the whole curve for this card in 43 loads, every answer controlled", () => {
    const { history, input } = run("frontier");
    expect(history).toHaveLength(43);
    const outcome = probeOutcome(input);
    for (const stop of outcome.curve!) {
      expect(stop.fit?.resolved).toBe(true);
      expect(stop.clean.resolved).toBe(true);
      if (stop.clean.value != null) {
        expect(stop.clean.control).toBe("confirmed");
        expect(stop.clean.value).toBeLessThanOrEqual(stop.fit!.value!);
      }
    }
    expect(outcome.next).toBeNull();
  });

  it("lowers an answer whose control spills, and gives the next context the lowered ceiling", () => {
    // The second load of 6 layers at 1k spills: that reading is not reproducible.
    const flaky: Machine = {
      claim: CARD.claim,
      clean: (r, index) => (r.ctx === 1024 && r.ngl === 6 && index >= 1 ? false : CARD.clean(r, index)),
    };
    const { history, input } = run("frontier", { machine: flaky });
    const curve = probeOutcome(input).curve!;
    expect(curve[0].clean).toMatchObject({ value: 4, control: "confirmed" });
    expect(history.some((h) => h.ctx === 2048 && h.ngl > 4 && h.ok)).toBe(false);
    expect(curve[1].clean.value).toBe(4);
  });

  it("stops once nothing fits: no load at any larger context", () => {
    const expensiveKv: Machine = { claim: (r) => 280 + 405 * r.ngl + r.ctx * 0.1, clean: (r) => r.ngl <= 2 };
    const { history, input } = run("frontier", { machine: expensiveKv });
    const outcome = probeOutcome(input);
    const firstNone = outcome.nothingFitsFrom!;
    expect(firstNone).toBe(131_072);
    expect(history.some((h) => h.ctx > firstNone)).toBe(false);
    expect(outcome.curve!.filter((s) => s.ctx > firstNone).every((s) => s.fit?.resolved && s.fit.value == null)).toBe(true);
  });

  it("at the default 40 loads, leaves only the largest context's no-spill answer unmeasured on this card", () => {
    const { history, input } = run("frontier", { maxLoads: PROBE_MAX_LOADS });
    expect(history).toHaveLength(PROBE_MAX_LOADS);
    const curve = probeOutcome(input).curve!;
    expect(curve.slice(0, -1).every((s) => s.fit?.resolved && s.clean.resolved)).toBe(true);
    expect(curve[curve.length - 1].fit).toMatchObject({ value: 14, resolved: true });
    expect(curve[curve.length - 1].clean.resolved).toBe(false);
  });

  it("leaves the stops it never reached unmeasured when the budget runs out", () => {
    const { history, input } = run("frontier", { maxLoads: 16 });
    expect(history).toHaveLength(16);
    const curve = probeOutcome(input).curve!;
    expect(curve[0].clean.resolved).toBe(true);
    expect(curve[curve.length - 1].clean.source).toBe("unmeasured");
    expect(curve[curve.length - 1].fit?.source).toBe("unmeasured");
  });

  describe("claim-only loads", () => {
    // Runs the ladder the way the worker does: a load flagged nextClaimOnly stops
    // at ready, so it carries its claim verdict but no clean one.
    function runStopping(machine: Machine = CARD) {
      const history: LadderAttempt[] = [];
      const flagged: LadderRung[] = [];
      const input: LadderInput = { ...run("frontier", { machine, maxLoads: 0 }).input, history, maxLoads: 200 };
      for (let guard = 0; guard < 300; guard++) {
        const outcome = probeOutcome(input);
        if (outcome.next == null) return { history, flagged, input };
        const measured = measure(machine, outcome.next, history, FREE);
        if (outcome.nextClaimOnly) {
          flagged.push(outcome.next);
          if (measured.fitsFree) {
            history.push({ ...measured, ok: false, placementJudged: false });
            continue;
          }
        }
        history.push(measured);
      }
      throw new Error("ladder did not terminate");
    }

    it("flags only loads above the stop below's final no-spill answer, never at the smallest context", () => {
      const { flagged, input } = runStopping();
      expect(flagged.length).toBeGreaterThan(0);
      const curve = probeOutcome(input).curve!;
      for (const r of flagged) {
        const i = curve.findIndex((s) => s.ctx === r.ctx);
        expect(i).toBeGreaterThan(0);
        const below = curve[i - 1].clean.value;
        if (below != null) expect(r.ngl).toBeGreaterThan(below);
      }
    });

    it("reaches the same curve as generating on every load", () => {
      const full = probeOutcome(run("frontier").input).curve;
      expect(probeOutcome(runStopping().input).curve).toEqual(full);
    });

    it("flags every load past a smallest context with no clean answer", () => {
      const nothingClean: Machine = { claim: CARD.claim, clean: () => false };
      const { history, flagged, input } = runStopping(nothingClean);
      expect(probeOutcome(input).curve![0].clean).toMatchObject({ value: null, resolved: true });
      expect(flagged.map(key)).toEqual(history.filter((h) => h.ctx > 1024).map(key));
    });
  });

  it("without a --list-devices reading searches no spill alone, opening at the estimate", () => {
    const { history, input } = run("frontier", { free: null });
    expect(history[0]).toEqual(expect.objectContaining({ ctx: 1024, ngl: 12 }));
    const outcome = probeOutcome(input);
    expect(outcome.curve![0].fit).toBeNull();
    expect(outcome.curve![0].clean.value).toBe(6);
    expect(outcome.fit).toBeNull();
  });
});

describe("the anchor", () => {
  const anchorKey = `${PROBE_LADDER_MIN_CTX}/${PROBE_ANCHOR_NGL}`;

  it("loads the anchor twice before any mode searches, flagged as anchor loads", () => {
    for (const [mode, opts] of [
      ["frontier", {}],
      ["keep_context", { ctx: 8192 }],
      ["fixed_offload", { ngl: 6 }],
      ["custom", { ctx: 4096, ngl: 6 }],
    ] as const) {
      const { history, input } = run(mode, { ...opts, anchored: true });
      expect(history.slice(0, 2).map(key)).toEqual([anchorKey, anchorKey]);
      expect(probeOutcome({ ...input, history: [] })).toMatchObject({ next: { ctx: 1024, ngl: 1 }, nextIsAnchor: true });
      expect(probeOutcome({ ...input, history: history.slice(0, 2) }).nextIsAnchor).toBeFalsy();
    }
  });

  it("reaches the same Wizard curve, opening target 2 from the anchor claim", () => {
    const plain = probeOutcome(run("frontier").input).curve!;
    const { history, input } = run("frontier", { anchored: true });
    // One claim is known (the anchor), so the opener is its prediction, which errs low.
    expect(history[2]).toEqual(expect.objectContaining({ ctx: 1024, ngl: 10 }));
    expect(probeOutcome(input).curve!.map((s) => [s.ctx, s.fit?.value, s.clean.value, s.clean.control])).toEqual(
      plain.map((s) => [s.ctx, s.fit?.value, s.clean.value, s.clean.control])
    );
  });

  it("ends the probe on a failed anchor, and reports nothing fits when its claim did not fit", () => {
    const failing: Machine = { claim: CARD.claim, clean: (r) => !(r.ctx === 1024 && r.ngl === 1) };
    const failed = run("frontier", { machine: failing, anchored: true });
    expect(failed.history).toHaveLength(1);
    expect(probeOutcome(failed.input)).toMatchObject({ next: null, clean: null });

    // A claim over free stops at ready and is never clean.
    const huge: Machine = { claim: () => FREE + 1, clean: () => false };
    const none = run("frontier", { machine: huge, anchored: true });
    expect(none.history).toHaveLength(1);
    const outcome = probeOutcome(none.input);
    expect(outcome.nothingFitsFrom).toBe(1024);
    expect(outcome.curve!.every((s) => s.fit?.resolved && s.fit.value == null && s.clean.value == null)).toBe(true);
  });
});

describe("the stored ceiling", () => {
  it("prefers the largest context whose answer a control confirmed over a larger unconfirmed one", () => {
    // Cut the Wizard off at every budget; wherever it stops right after an answer
    // but before that answer's control, the stored ceiling must skip it.
    let checked = 0;
    for (let budget = 10; budget <= 43; budget++) {
      const outcome = probeOutcome(run("frontier", { maxLoads: budget }).input);
      const answered = outcome.curve!.filter((s) => s.clean.resolved && s.clean.value != null);
      const last = answered.at(-1);
      if (!last || last.clean.control !== "pending" || answered.length < 2) continue;
      expect(outcome.clean?.control).toBe("confirmed");
      expect(outcome.clean!.ctx).toBeLessThan(last.ctx);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });
});

describe("target 1's window of two layers above where spill starts", () => {
  function atOneContext(cleanSet: number[]) {
    const machine: Machine = { claim: CARD.claim, clean: (r) => cleanSet.includes(r.ngl) };
    return run("keep_context", { machine, ctx: 1024, ngl: 10 });
  }

  it("repeats from each clean layer the window finds", () => {
    const { input } = atOneContext([0, 1, 2, 3, 5, 7]);
    expect(probeOutcome(input).clean?.ngl).toBe(7);
  });

  it("misses a clean island three layers above where spill starts, by design", () => {
    const { input } = atOneContext([0, 1, 2, 3, 4, 8]);
    expect(probeOutcome(input).clean?.ngl).toBe(4);
  });

  it("answers none when even 0 layers spill", () => {
    const { input } = atOneContext([]);
    expect(probeOutcome(input).clean).toBeNull();
    expect(probeOutcome(input).fit?.ngl).toBe(17);
  });
});

describe("Targets", () => {
  it("keep_context: target 2 from the user's layers, then target 1, then the control last", () => {
    const { history, input } = run("keep_context", { ctx: 1024, ngl: 10 });
    const outcome = probeOutcome(input);
    expect(history[0]).toEqual(expect.objectContaining({ ctx: 1024, ngl: 10 }));
    expect(outcome.fit).toEqual({ ctx: 1024, ngl: 17 });
    expect(outcome.clean).toMatchObject({ ctx: 1024, ngl: 6, control: "confirmed" });
    expect(key(history[history.length - 1])).toBe("1024/6");
  });

  it("fixed_offload: both targets over the context stops, two stops above where spill starts", () => {
    // 6 layers: clean up to 32k, spilling above it.
    const { history, input } = run("fixed_offload", { ctx: 8192, ngl: 6 });
    const outcome = probeOutcome(input);
    expect(outcome.fit).toEqual({ ctx: TRAINED, ngl: 6 });
    expect(outcome.clean).toMatchObject({ ctx: 32_768, ngl: 6, control: "confirmed" });
    expect(history.every((h) => ctxLadderStops(TRAINED).includes(h.ctx))).toBe(true);
  });

  it("fixed_offload keeps a real miss above a fit when a flaky miss sits below it", () => {
    const at = (ctx: number, fits: boolean): LadderAttempt => ({ ctx, ngl: 6, ok: false, fitsFree: fits, claimedMib: 1 });
    const outcome = probeOutcome({
      mode: "fixed_offload", candidateCtx: 8192, candidateNgl: 6, nglMax: LAYERS, maxCtx: TRAINED, freeVramMib: FREE,
      // 2k missed (flaky), 8k fits, 16k misses: target 2 is 8k, not "every stop fits".
      history: [at(2048, false), at(8192, true), at(16_384, false)],
    });
    expect(outcome.fit).toEqual({ ctx: 8192, ngl: 6 });
  });

  it("keep_context does not read a load that failed for a non-memory reason as a miss", () => {
    // 18 timed out before ready: it must not become target 2's boundary.
    const history: LadderAttempt[] = [
      { ctx: 1024, ngl: 16, ok: false, fitsFree: true, claimedMib: 6800 },
      { ctx: 1024, ngl: 17, ok: false, fitsFree: true, claimedMib: 7205 },
      { ctx: 1024, ngl: 18, ok: false, fitsFree: false, inconclusive: true },
    ];
    const outcome = probeOutcome({
      mode: "keep_context", candidateCtx: 1024, candidateNgl: 16, nglMax: LAYERS, maxCtx: TRAINED, freeVramMib: FREE, history,
    });
    expect(outcome.fit).toBeNull();
    expect(outcome.next).not.toEqual({ ctx: 1024, ngl: 18 });
    expect(outcome.next?.ngl).toBeGreaterThan(18);
  });

  it("custom: exactly one load, no control", () => {
    const { history, input } = run("custom", { ctx: 4096, ngl: 6 });
    expect(history.map(key)).toEqual(["4096/6"]);
    expect(probeOutcome(input).clean).toMatchObject({ ctx: 4096, ngl: 6, control: "none" });
  });
});

describe("ctxLadderStops", () => {
  it("is power-of-two doublings from the floor to the model's own ceiling, included exactly", () => {
    expect(ctxLadderStops(262_144)).toEqual([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]);
    expect(ctxLadderStops(50_000)).toEqual([1024, 2048, 4096, 8192, 16384, 32768, 50_000]);
    expect(ctxLadderStops(500)).toEqual([PROBE_LADDER_MIN_CTX]);
  });
});

describe("snapToSafeCtx", () => {
  it("snaps a stored off-stop value DOWN to a stop, never up", () => {
    expect(snapToSafeCtx(43_581, TRAINED)).toBe(32_768);
    expect(snapToSafeCtx(32_768, TRAINED)).toBe(32_768);
    for (const v of [1024, 5000, 43_581, 200_000, TRAINED]) expect(snapToSafeCtx(v, TRAINED)).toBeLessThanOrEqual(v);
  });

  it("uses the probe's grid, not the client slider's fraction grid", () => {
    expect(computeCtxStops(50_000)).not.toEqual(ctxLadderStops(50_000));
    expect(snapToSafeCtx(40_000, 50_000)).toBe(32_768);
  });
});
