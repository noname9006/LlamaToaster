// BENCHMARKING_PLAN_V8.md N3 -- model-vs-model comparison across weight
// quants. Zero new measurement: every row is an ordinary `results` row on its
// own model_id, grouped by runs.comparison_id.
//
// The fairness rules are BLOCKING, at trigger and re-checked per member.
// Comparisons are the one place a silent confound poisons exactly the
// conclusion being sold, so drift fails the affected member loudly instead of
// quietly shifting the answer.

export const MAX_COMPARISON_MEMBERS = 5;
export const MIN_COMPARISON_MEMBERS = 2;

export interface ComparisonFairnessFacts {
  worker_id: string | null;
  llama_cpp_build: string | null;
  llama_cpp_backend: string | null;
  backend_device_name: string | null;
  repeats: number | null;
  method_version: number | null;
  /** Every non-model axis of the frozen grid, canonicalized. */
  grid_signature: string | null;
}

export interface FairnessViolation {
  field: keyof ComparisonFairnessFacts;
  expected: string | number | null;
  found: string | number | null;
  message: string;
}

const FIELD_COPY: Record<keyof ComparisonFairnessFacts, string> = {
  worker_id: "machine",
  llama_cpp_build: "llama.cpp build",
  llama_cpp_backend: "backend",
  backend_device_name: "GPU",
  repeats: "repeats",
  method_version: "methodology version",
  grid_signature: "grid",
};

// Compares a prospective (or completed) member against the group's first
// member. A null on EITHER side of a field is "not known yet" and is not
// drift -- e.g. method_version is only known once rows land, and the
// per-member re-check is what catches it then.
export function checkComparisonFairness(
  reference: ComparisonFairnessFacts,
  candidate: ComparisonFairnessFacts
): FairnessViolation[] {
  const violations: FairnessViolation[] = [];
  for (const field of Object.keys(FIELD_COPY) as (keyof ComparisonFairnessFacts)[]) {
    const expected = reference[field];
    const found = candidate[field];
    if (expected == null || found == null) continue;
    if (expected !== found) {
      violations.push({
        field,
        expected,
        found,
        message: `${FIELD_COPY[field]} differs from the rest of this comparison (expected ${String(
          expected
        )}, found ${String(found)}) — the interesting variable is the model file, so everything else must be held fixed`,
      });
    }
  }
  return violations;
}

// The frozen grid, minus the model: what every member must share. Sorted and
// joined so two identical grids expressed in different key order still match.
export function gridSignature(sweep: Record<string, unknown> | null | undefined): string | null {
  if (!sweep || typeof sweep !== "object") return null;
  const parts: string[] = [];
  for (const key of Object.keys(sweep).sort()) {
    if (key === "model_id" || key === "repeats") continue;
    const value = (sweep as Record<string, unknown>)[key];
    parts.push(`${key}=${Array.isArray(value) ? [...value].map(String).sort().join("|") : String(value)}`);
  }
  return parts.join(";");
}

// --- The comparison view ----------------------------------------------------

export interface ComparisonMemberRow {
  run_id: string;
  model_id: string;
  model_filename: string;
  /** Parsed from GGUF metadata (general.file_type), not from the filename. */
  quant_label: string | null;
  /** Identical labels with different hashes display separately: labels lie, hashes don't. */
  file_sha256: string | null;
  status: string;
  /** Null until the member produced a scoreable row. */
  pp: number | null;
  tg: number | null;
  vram_peak_mib: number | null;
  ram_peak_mib: number | null;
  /** N4, when a quality row exists for this model. Never a score, never ranked on. */
  ppl: number | null;
  kld_vs_baseline: number | null;
  dataset_hash: string | null;
  /** N2, when a probe verified a ceiling for this quant on this machine. */
  verified_ctx_tokens: number | null;
  /** Non-empty when this member drifted -- a FATAL, marked in the view. */
  violations: FairnessViolation[];
}

export interface ParetoPoint {
  model_id: string;
  label: string;
  /** x axis: measured speed. */
  tg: number;
  /** y axis: measured memory. */
  peakMib: number;
  dominated: boolean;
}

// Speed x memory across models. A point is dominated when another point is at
// least as fast AND uses no more memory -- and strictly better on one of them.
export function paretoFrontier(rows: ComparisonMemberRow[]): ParetoPoint[] {
  const points: ParetoPoint[] = rows
    .filter((r) => r.tg != null)
    .map((r) => ({
      model_id: r.model_id,
      label: r.quant_label ?? r.model_filename,
      tg: r.tg!,
      peakMib: (r.vram_peak_mib ?? 0) + (r.ram_peak_mib ?? 0),
      dominated: false,
    }));
  for (const point of points) {
    point.dominated = points.some(
      (other) =>
        other !== point &&
        other.tg >= point.tg &&
        other.peakMib <= point.peakMib &&
        (other.tg > point.tg || other.peakMib < point.peakMib)
    );
  }
  return points;
}
