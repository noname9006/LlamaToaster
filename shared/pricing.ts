// BENCHMARKING_PLAN_V8.md §0.6 -- the ETA pricing rule. Pre-flight and live
// ETAs price from, in order:
//
//   1. the most recent {engine:"server", spec:"off"} pp rate for
//      (model_id, worker_id[, llama_cpp_build]);
//   2. the parent sweep's llama-bench pp for the same model, rendered with
//      the MANDATORY label "derived from llama-bench";
//   3. neither -> `ETA unavailable`.
//
// Generation prices symmetrically: server-measured tg first, then labeled
// bench-derived tg, then that component renders unavailable.
//
// The invariant this file exists to enforce: NO CELL RENDERS A NUMBER WITHOUT
// NAMING ITS SOURCE. Every priced value is returned together with the
// provenance label that must be shown beside it.

export type RateSource = "server_measured" | "bench_derived" | "unavailable";

export const RATE_SOURCE_LABEL: Record<RateSource, string> = {
  server_measured: "measured on this machine",
  bench_derived: "derived from llama-bench",
  unavailable: "unavailable",
};

export const ETA_UNAVAILABLE = "ETA unavailable";

export interface RateCandidate {
  /** tokens per second */
  tps: number;
  engine: string | null;
  /** "off" for a non-speculative row; anything else disqualifies it as a baseline. */
  spec: string | null;
  test_type: "pp" | "tg" | "pg";
  model_id: string;
  worker_id: string | null;
  llama_cpp_build: string | null;
  created_at: number;
}

export interface RateQuery {
  model_id: string;
  worker_id?: string | null;
  llama_cpp_build?: string | null;
}

export interface PricedRate {
  tps: number | null;
  source: RateSource;
  /** Rendered verbatim beside the number. Never omitted. */
  label: string;
}

const UNAVAILABLE_RATE: PricedRate = {
  tps: null,
  source: "unavailable",
  label: RATE_SOURCE_LABEL.unavailable,
};

function matches(candidate: RateCandidate, query: RateQuery): boolean {
  if (candidate.model_id !== query.model_id) return false;
  if (query.worker_id != null && candidate.worker_id !== query.worker_id) return false;
  // The build is an OPTIONAL narrowing (§0.6's "[, llama_cpp_build]"): a
  // same-machine rate from another build still prices better than nothing.
  if (query.llama_cpp_build != null && candidate.llama_cpp_build != null) {
    if (candidate.llama_cpp_build !== query.llama_cpp_build) return false;
  }
  return true;
}

function newest(candidates: RateCandidate[]): RateCandidate | undefined {
  return candidates.sort((a, b) => b.created_at - a.created_at)[0];
}

export function priceRate(
  candidates: RateCandidate[],
  query: RateQuery,
  testType: "pp" | "tg"
): PricedRate {
  const usable = candidates.filter((c) => c.tps > 0 && c.test_type === testType && matches(c, query));
  const server = newest(usable.filter((c) => c.engine === "server" && (c.spec ?? "off") === "off"));
  if (server) {
    return { tps: server.tps, source: "server_measured", label: RATE_SOURCE_LABEL.server_measured };
  }
  const bench = newest(usable.filter((c) => c.engine !== "server"));
  if (bench) {
    return { tps: bench.tps, source: "bench_derived", label: RATE_SOURCE_LABEL.bench_derived };
  }
  return UNAVAILABLE_RATE;
}

export interface CellPricingInput {
  nPrompt: number;
  nGen: number;
  repeats: number;
  /** Model load time, seconds -- one per item, measured or assumed. */
  loadSeconds?: number;
  ppRate: PricedRate;
  tgRate: PricedRate;
}

export interface PricedCell {
  /** Null whenever ANY priced component is unavailable -- never a partial number presented as a total. */
  seconds: number | null;
  prefillSeconds: number | null;
  generationSeconds: number | null;
  loadSeconds: number;
  /** What each half was priced from; rendered beside the number. */
  sources: { prefill: RateSource; generation: RateSource };
  /** The single line the UI shows -- either a duration with its sources, or ETA unavailable with the reason. */
  display: string;
}

export function priceCell(input: CellPricingInput): PricedCell {
  const loadSeconds = input.loadSeconds ?? 0;
  const prefillSeconds =
    input.ppRate.tps != null && input.ppRate.tps > 0 ? (input.nPrompt / input.ppRate.tps) * input.repeats : null;
  const generationSeconds =
    input.tgRate.tps != null && input.tgRate.tps > 0 ? (input.nGen / input.tgRate.tps) * input.repeats : null;

  // A component priced from nothing makes the total unavailable: half a
  // number presented as a whole one is exactly what §0.6 forbids.
  const unavailable: string[] = [];
  if (prefillSeconds == null && input.nPrompt > 0) unavailable.push("prompt processing");
  if (generationSeconds == null && input.nGen > 0) unavailable.push("generation");

  const seconds =
    unavailable.length > 0 ? null : loadSeconds + (prefillSeconds ?? 0) + (generationSeconds ?? 0);

  return {
    seconds,
    prefillSeconds,
    generationSeconds,
    loadSeconds,
    sources: { prefill: input.ppRate.source, generation: input.tgRate.source },
    display:
      seconds == null
        ? `${ETA_UNAVAILABLE} — no rate for ${unavailable.join(" or ")} on this machine yet`
        : `${formatDuration(seconds)} · prompt ${input.ppRate.label}, generation ${input.tgRate.label}`,
  };
}

export function priceMatrix(cells: CellPricingInput[]): { seconds: number | null; display: string } {
  const priced = cells.map(priceCell);
  if (priced.some((c) => c.seconds == null)) {
    return { seconds: null, display: `${ETA_UNAVAILABLE} — at least one cell has no rate to price from` };
  }
  const total = priced.reduce((acc, c) => acc + (c.seconds ?? 0), 0);
  const sources = new Set(priced.flatMap((c) => [c.sources.prefill, c.sources.generation]));
  const label = sources.has("bench_derived")
    ? `${formatDuration(total)} · includes cells derived from llama-bench`
    : `${formatDuration(total)} · measured on this machine`;
  return { seconds: total, display: label };
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return ETA_UNAVAILABLE;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours} h ${String(minutes).padStart(2, "0")} m`;
  if (minutes > 0) return `${minutes} m ${String(secs).padStart(2, "0")} s`;
  return `${secs} s`;
}
