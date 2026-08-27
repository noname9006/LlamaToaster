import { describe, expect, it } from "vitest";
import {
  ETA_UNAVAILABLE,
  formatDuration,
  priceCell,
  priceMatrix,
  priceRate,
  RATE_SOURCE_LABEL,
  type RateCandidate,
} from "./pricing.js";

const query = { model_id: "m1", worker_id: "w1", llama_cpp_build: "b10516" };

function candidate(over: Partial<RateCandidate> & { tps: number; test_type: "pp" | "tg" }): RateCandidate {
  return {
    engine: "bench",
    spec: null,
    model_id: "m1",
    worker_id: "w1",
    llama_cpp_build: "b10516",
    created_at: 1_000,
    ...over,
  };
}

describe("§0.6 rate selection order", () => {
  it("prefers the most recent server/spec-off rate", () => {
    const rate = priceRate(
      [
        candidate({ tps: 1500, test_type: "pp", engine: "bench", created_at: 3_000 }),
        candidate({ tps: 280, test_type: "pp", engine: "server", spec: "off", created_at: 1_000 }),
        candidate({ tps: 310, test_type: "pp", engine: "server", spec: "off", created_at: 2_000 }),
      ],
      query,
      "pp"
    );
    expect(rate).toMatchObject({ tps: 310, source: "server_measured" });
  });

  it("never prices from a SPECULATIVE server row", () => {
    const rate = priceRate(
      [candidate({ tps: 900, test_type: "tg", engine: "server", spec: "mtp", created_at: 9_000 })],
      query,
      "tg"
    );
    // The mtp row is not a {server, off} baseline, so it falls through to the
    // bench tier -- and that row is a server row, so nothing prices it.
    expect(rate.source).toBe("unavailable");
  });

  it("falls back to llama-bench with the MANDATORY label", () => {
    const rate = priceRate([candidate({ tps: 1500, test_type: "pp", engine: "bench" })], query, "pp");
    expect(rate).toMatchObject({ tps: 1500, source: "bench_derived" });
    expect(rate.label).toBe("derived from llama-bench");
  });

  it("returns unavailable rather than a number when nothing matches", () => {
    expect(priceRate([], query, "pp")).toMatchObject({ tps: null, source: "unavailable" });
    // Another machine's rate is not this machine's rate.
    expect(
      priceRate([candidate({ tps: 1500, test_type: "pp", worker_id: "w2" })], query, "pp").source
    ).toBe("unavailable");
  });

  it("prices generation symmetrically with prompt processing", () => {
    const candidates = [
      candidate({ tps: 39, test_type: "tg", engine: "server", spec: "off", created_at: 5_000 }),
      candidate({ tps: 45, test_type: "tg", engine: "bench", created_at: 9_000 }),
    ];
    expect(priceRate(candidates, query, "tg")).toMatchObject({ tps: 39, source: "server_measured" });
  });
});

describe("§0.6 cell pricing", () => {
  const pp = { tps: 310, source: "server_measured" as const, label: RATE_SOURCE_LABEL.server_measured };
  const tg = { tps: 39, source: "server_measured" as const, label: RATE_SOURCE_LABEL.server_measured };

  it("prices prefill and generation from their own rates and names both sources", () => {
    const cell = priceCell({ nPrompt: 8192, nGen: 512, repeats: 1, loadSeconds: 50, ppRate: pp, tgRate: tg });
    expect(cell.prefillSeconds).toBeCloseTo(8192 / 310, 6);
    expect(cell.generationSeconds).toBeCloseTo(512 / 39, 6);
    expect(cell.seconds).toBeCloseTo(50 + 8192 / 310 + 512 / 39, 6);
    expect(cell.display).toContain("measured on this machine");
  });

  it("multiplies by the repeat count", () => {
    const one = priceCell({ nPrompt: 8192, nGen: 512, repeats: 1, ppRate: pp, tgRate: tg }).seconds!;
    const five = priceCell({ nPrompt: 8192, nGen: 512, repeats: 5, ppRate: pp, tgRate: tg }).seconds!;
    expect(five).toBeCloseTo(one * 5, 6);
  });

  it("renders ETA unavailable rather than half a number when one side has no rate", () => {
    const cell = priceCell({
      nPrompt: 8192,
      nGen: 512,
      repeats: 1,
      ppRate: pp,
      tgRate: { tps: null, source: "unavailable", label: RATE_SOURCE_LABEL.unavailable },
    });
    expect(cell.seconds).toBeNull();
    expect(cell.display).toContain(ETA_UNAVAILABLE);
    expect(cell.display).toContain("generation");
  });

  it("labels a bench-derived cell as such, in the total too", () => {
    const benchPp = { tps: 1500, source: "bench_derived" as const, label: RATE_SOURCE_LABEL.bench_derived };
    const cell = priceCell({ nPrompt: 8192, nGen: 512, repeats: 1, ppRate: benchPp, tgRate: tg });
    expect(cell.display).toContain("derived from llama-bench");
    const matrix = priceMatrix([{ nPrompt: 8192, nGen: 512, repeats: 1, ppRate: benchPp, tgRate: tg }]);
    expect(matrix.display).toContain("derived from llama-bench");
  });

  it("makes the whole matrix unavailable when any single cell cannot be priced", () => {
    const matrix = priceMatrix([
      { nPrompt: 8192, nGen: 512, repeats: 1, ppRate: pp, tgRate: tg },
      {
        nPrompt: 65_536,
        nGen: 512,
        repeats: 1,
        ppRate: { tps: null, source: "unavailable", label: RATE_SOURCE_LABEL.unavailable },
        tgRate: tg,
      },
    ]);
    expect(matrix.seconds).toBeNull();
    expect(matrix.display).toContain(ETA_UNAVAILABLE);
  });
});

describe("duration formatting", () => {
  it("renders hours, minutes and seconds", () => {
    expect(formatDuration(7_080)).toBe("1 h 58 m");
    expect(formatDuration(125)).toBe("2 m 05 s");
    expect(formatDuration(9)).toBe("9 s");
  });
});
