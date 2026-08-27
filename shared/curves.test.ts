import { describe, expect, it } from "vitest";
import { buildCurve, buildLadder, deriveKnee, CURVE_LADDER, type CurveSourceRow } from "./curves.js";

let seq = 0;

function row(over: Partial<CurveSourceRow> & { test_type: "pp" | "tg"; n_prompt: number; avg_tps: number }): CurveSourceRow {
  return {
    id: `r${seq++}`,
    run_id: over.run_id ?? "run-1",
    idx: over.idx ?? 0,
    n_gen: 128,
    n_depth: 0,
    stddev_tps: 0,
    vram_peak_mib: 4000,
    ram_peak_mib: 2000,
    method_version: 2,
    caveat_flags: [],
    created_at: 1_000,
    ...over,
  };
}

let nextItemIdx = 0;

function point(
  ctx: number,
  pp: number,
  tg: number,
  over: Omit<Partial<CurveSourceRow>, "test_type"> = {}
): CurveSourceRow[] {
  // One point = one run item, so each call gets its own idx unless the caller
  // pins one.
  const idx = over.idx ?? nextItemIdx++;
  over = { ...over, idx };
  return [
    row({ test_type: "pp", n_prompt: ctx, avg_tps: pp, ttft_ms_p50: (ctx / pp) * 1000, ttft_n: 1, ...over }),
    row({ test_type: "tg", n_prompt: ctx, avg_tps: tg, e2e_ms_mean: 39_500, ...over }),
  ];
}

describe("N1 context curves", () => {
  it("keys points on effective context (n_depth + n_prompt), so both engines land on one x-axis", () => {
    // Different runs, as they would be in practice: one llama-bench depth
    // sweep, one llama-server curve point.
    const benchRow = row({ test_type: "tg", n_prompt: 512, n_depth: 16_384, avg_tps: 38, created_at: 1, run_id: "bench-run" });
    const serverRow = row({ test_type: "tg", n_prompt: 16_896, n_depth: 0, avg_tps: 38.4, created_at: 2, run_id: "server-run" });
    const curve = buildCurve([benchRow, serverRow]);
    expect(curve.map((p) => p.effectiveCtx)).toEqual([16_896, 16_896]);
  });

  it("labels a cold point as single-shot rather than pretending it is a p50", () => {
    const curve = buildCurve(point(8_192, 310, 39.1));
    expect(curve[0].ttftN).toBe(1);
    expect(curve[0].ttftMs).toBeCloseTo((8192 / 310) * 1000, 3);
  });

  it("greys the superseded point on a re-measure instead of averaging the two", () => {
    const curve = buildCurve([
      ...point(16_384, 305, 38.2, { created_at: 100 }),
      ...point(16_384, 310, 38.8, { created_at: 200 }),
    ]);
    expect(curve).toHaveLength(2);
    expect(curve[0].superseded).toBe(true);
    expect(curve[0].tg).toBe(38.2);
    expect(curve[1].superseded).toBe(false);
    expect(curve[1].tg).toBe(38.8);
  });

  it("breaks a created_at collision by id, so the ordering is a total order", () => {
    const a = point(8_192, 300, 38, { created_at: 500 });
    const b = point(8_192, 320, 39, { created_at: 500 });
    const curve = buildCurve([...a, ...b]);
    // Same timestamp: both survive as distinct cells, and the sort is stable
    // rather than nondeterministic.
    expect(curve).toHaveLength(2);
    expect(curve.every((p) => p.effectiveCtx === 8_192)).toBe(true);
  });

  it("drops a cache-evicted row from the curve with an explanatory reason", () => {
    const curve = buildCurve(point(32_768, 300, 38, { caveat_flags: ["cache_evicted"] }));
    expect(curve[0].excluded).toBe(true);
    expect(curve[0].excludedReason).toContain("prefix cache did not hold");
  });

  it("flags a context-shifted row too -- shifted contexts corrupt TTFT comparability", () => {
    const curve = buildCurve(point(32_768, 300, 38, { caveat_flags: ["context_shift"] }));
    expect(curve[0].excluded).toBe(true);
    expect(curve[0].excludedReason).toContain("--no-context-shift");
  });
});

describe("N1 sizing ladder", () => {
  it("renders points beyond trained context as unavailable-with-reason, never as silent gaps", () => {
    const ladder = buildLadder([8_192, 16_384, 32_768], { trainedCtx: 32_768 });
    expect(ladder.map((c) => c.effectiveCtx)).toEqual([...CURVE_LADDER]);
    const beyond = ladder.filter((c) => c.effectiveCtx > 32_768);
    expect(beyond.length).toBeGreaterThan(0);
    expect(beyond.every((c) => !c.available && c.unavailableReason!.includes("trained ctx"))).toBe(true);
    expect(ladder.find((c) => c.effectiveCtx === 8_192)!.measured).toBe(true);
    expect(ladder.find((c) => c.effectiveCtx === 4_096)!.measured).toBe(false);
  });

  it("clamps to M1 affordability too, naming that as the reason", () => {
    const ladder = buildLadder([], { trainedCtx: 131_072, affordableTokens: 44_000 });
    expect(ladder.find((c) => c.effectiveCtx === 32_768)!.available).toBe(true);
    expect(ladder.find((c) => c.effectiveCtx === 65_536)!.unavailableReason).toContain("affords about 44000");
  });
});

describe("N5 concurrency knee", () => {
  const ladder: CurveSourceRow[] = [
    row({ test_type: "tg", n_prompt: 32_768, avg_tps: 39, concurrency: 1, ttft_ms_p95: 107_000 }),
    row({ test_type: "tg", n_prompt: 32_768, avg_tps: 68, concurrency: 2, ttft_ms_p95: 118_000 }),
    row({ test_type: "tg", n_prompt: 32_768, avg_tps: 101, concurrency: 4, ttft_ms_p95: 226_000 }),
    row({ test_type: "tg", n_prompt: 32_768, avg_tps: 108, concurrency: 8, ttft_ms_p95: 415_000 }),
  ];

  it("finds the smallest slot count whose TTFT p95 exceeds 2x its slots=1 value", () => {
    const knee = deriveKnee(ladder);
    expect(knee.thresholdMs).toBe(214_000);
    expect(knee.knee).toBe(4);
    expect(knee.summary).toBe("knee at slots = 4 — TTFT p95 doubles past this point");
  });

  it("shows aggregate throughput still rising past the knee -- the trade is visible, not asserted", () => {
    const knee = deriveKnee(ladder);
    expect(knee.samples.map((s) => s.aggregateTps)).toEqual([39, 68, 101, 108]);
  });

  it("reports no knee rather than inventing one when the ladder never doubles", () => {
    const flat = deriveKnee([
      row({ test_type: "tg", n_prompt: 4096, avg_tps: 39, concurrency: 1, ttft_ms_p95: 100 }),
      row({ test_type: "tg", n_prompt: 4096, avg_tps: 70, concurrency: 2, ttft_ms_p95: 120 }),
    ]);
    expect(flat.knee).toBeNull();
    expect(flat.summary).toContain("no knee within the measured ladder");
  });

  it("refuses to derive a knee with no slots = 1 baseline", () => {
    const noBaseline = deriveKnee([
      row({ test_type: "tg", n_prompt: 4096, avg_tps: 70, concurrency: 2, ttft_ms_p95: 120 }),
    ]);
    expect(noBaseline.knee).toBeNull();
    expect(noBaseline.summary).toContain("no slots = 1 baseline");
  });
});
