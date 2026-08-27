// BENCHMARKING_PLAN_V8.md N7 -- reproducible exchange. Results travel without
// losing their meaning: every shared number carries its own methods section,
// every row carries a hash recomputed at export time from its canonical form
// (§0.4), and an import validates per row so tampering one field breaks
// exactly that row.

import { configHash, type ConfigHashInput } from "./configHash.js";
import type { CaveatFlag, TestType } from "./types.js";
import type { GoalsConfig } from "./goals.js";

export const BUNDLE_FORMAT = "llamatoaster.bundle";
export const BUNDLE_FORMAT_VERSION = 1;

// Anonymized hardware class: GPU name + a VRAM bucket. No serials, no
// hostnames, no machine ids.
export interface HardwareClass {
  gpu_name: string | null;
  backend: string | null;
  vram_class_mib: number | null;
  cpu_isa: string | null;
}

export const VRAM_CLASS_BUCKETS_MIB = [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304] as const;

// Bucketed, not exact: a precise VRAM total is closer to a fingerprint than a
// hardware class.
export function vramClass(totalMib: number | null | undefined): number | null {
  if (totalMib == null || !Number.isFinite(totalMib) || totalMib <= 0) return null;
  for (const bucket of VRAM_CLASS_BUCKETS_MIB) {
    if (totalMib <= bucket) return bucket;
  }
  return VRAM_CLASS_BUCKETS_MIB[VRAM_CLASS_BUCKETS_MIB.length - 1];
}

export interface BundleRow {
  /** Recomputed at export time from this row's canonical form -- never read back from results.config_hash. */
  config_hash: string;
  test_type: TestType;
  n_prompt: number;
  n_gen: number;
  n_depth: number;
  n_threads: number;
  n_gpu_layers: number;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  n_gpu_layers_draft: number;
  n_cpu_moe: number;
  engine: string | null;
  concurrency: number | null;
  method_version: number | null;
  avg_tps: number;
  stddev_tps: number;
  sample_count: number | null;
  suspect_count: number | null;
  vram_peak_mib: number | null;
  ram_peak_mib: number;
  ttft_ms_p50: number | null;
  ttft_ms_p95: number | null;
  ttft_n: number | null;
  e2e_ms_mean: number | null;
  gpu_temp_c_max: number | null;
  gpu_clock_mhz_min: number | null;
  caveat_flags: CaveatFlag[];
  created_at: number;
}

export interface BundleQualityRow {
  model_id: string;
  ctx_tokens: number;
  cache_type_k: string;
  cache_type_v: string;
  ppl: number | null;
  kld_vs_baseline: number | null;
  /** A perplexity number without its corpus hash is meaningless. */
  dataset_hash: string;
  method_version: number | null;
}

export interface Bundle {
  format: typeof BUNDLE_FORMAT;
  format_version: number;
  exported_at: number;
  run: {
    id: string;
    root_run_id: string | null;
    kind: string | null;
    model_filename: string;
    model_quant: string | null;
    /**
     * The model file's own sha256 -- content-addressed, not a machine
     * fingerprint, and the only way an importer can tell two files apart when
     * their quant LABELS are identical. Labels lie, hashes don't.
     */
    model_sha256?: string | null;
    llama_cpp_build: string | null;
    /** The full stored configuration, including M2's goals block. */
    config: { goals?: GoalsConfig } & Record<string, unknown>;
  };
  hardware_class: HardwareClass;
  rows: BundleRow[];
  quality_rows: BundleQualityRow[];
  /** Every method_version present, so §0.1's display rules can be honored on the far side. */
  method_versions: (number | null)[];
  /** A rendered summary of the pipeline, keyed to method_version. */
  methods: MethodsSection[];
}

export interface MethodsSection {
  method_version: number;
  summary: string;
  pipeline: string[];
}

// Every shared number carries its own methods section. Keyed to
// method_version so a mixed-vintage bundle explains both vintages.
export function methodsFor(methodVersion: number): MethodsSection {
  if (methodVersion >= 2) {
    return {
      method_version: 2,
      summary:
        "Context-curve choreography: a discarded warm-up on a nonce prompt, one cold timed prefill whose first streamed chunk is the TTFT reading, then warm repeats against the prefix cache for generation statistics.",
      pipeline: [
        "warm-up: a short nonce prompt, n_predict 8, excluded from statistics by construction",
        "cold timed prefill: full prompt, streamed, n_predict 1, ignore_eos -- first-chunk arrival is TTFT (single sample, labeled as such)",
        "pp for the point comes from that same response's timings.prompt_ms / prompt_n",
        "warm repeats: identical prompt with cache_prompt -- any repeat reporting prompt_n > 0 re-prefilled and flags cache_evicted",
        "implausibility filter rejects physically impossible rates; wall-clock fallback readings are marked suspect",
        "stability gate: stddev <= max(10 % of mean, 0.5 tok/s), n >= 3",
      ],
    };
  }
  return {
    method_version: 1,
    summary:
      "Core methodology: warmed benchmark repeats, streamed clock where the engine has a request lifecycle, an implausibility filter on both sides of any pair, and a stddev floor in the stability gate.",
    pipeline: [
      "warm-up before the timed repeats",
      "streamed clock on the llama-server path; llama-bench does its own internal repeat averaging",
      "implausibility filter rejects physically impossible rates (the ~1e6 tok/s timer-bug class)",
      "suspect readings are kept and flagged, never silently erased",
      "sample (n-1) standard deviation, matching llama-bench's own formula",
      "stability gate: stddev <= max(10 % of mean, 0.5 tok/s), n >= 3",
    ],
  };
}

// The canonical form a bundle row's hash is computed from -- §0.4's axes, and
// nothing else. Recomputed at EXPORT time rather than read back from
// results.config_hash, which exists only where the twin join wrote it.
export function canonicalFormOf(row: Omit<BundleRow, "config_hash">): ConfigHashInput {
  return {
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth,
    threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    engine: (row.engine as "bench" | "server" | null) ?? undefined,
    concurrency: row.concurrency ?? undefined,
  };
}

export function stampConfigHash(row: Omit<BundleRow, "config_hash">): BundleRow {
  return { ...row, config_hash: configHash(canonicalFormOf(row)) };
}

export interface ImportRowVerdict {
  index: number;
  ok: boolean;
  reason: string | null;
}

export interface ImportValidation {
  ok: boolean;
  /** Populated only when the bundle envelope itself is unusable. */
  fatal: string | null;
  rows: ImportRowVerdict[];
  acceptedRows: BundleRow[];
  methodVersions: (number | null)[];
  /** True when the bundle mixes vintages -- §0.1 says surface them, never average them. */
  mixedVintages: boolean;
}

// Per-row validation: tampering ONE field breaks exactly that row, and the
// rest of the bundle still imports.
export function validateBundle(raw: unknown): ImportValidation {
  const empty: ImportValidation = {
    ok: false,
    fatal: null,
    rows: [],
    acceptedRows: [],
    methodVersions: [],
    mixedVintages: false,
  };
  if (!raw || typeof raw !== "object") return { ...empty, fatal: "bundle must be an object" };
  const bundle = raw as Partial<Bundle>;
  if (bundle.format !== BUNDLE_FORMAT) {
    return { ...empty, fatal: `not a ${BUNDLE_FORMAT} bundle` };
  }
  if (typeof bundle.format_version !== "number" || bundle.format_version > BUNDLE_FORMAT_VERSION) {
    return {
      ...empty,
      fatal: `bundle format version ${String(bundle.format_version)} is newer than this install understands (${BUNDLE_FORMAT_VERSION})`,
    };
  }
  if (!Array.isArray(bundle.rows)) return { ...empty, fatal: "bundle has no rows array" };

  const verdicts: ImportRowVerdict[] = [];
  const accepted: BundleRow[] = [];
  bundle.rows.forEach((row, index) => {
    const problem = rowProblem(row);
    if (problem) {
      verdicts.push({ index, ok: false, reason: problem });
      return;
    }
    const recomputed = configHash(canonicalFormOf(row));
    if (recomputed !== row.config_hash) {
      verdicts.push({
        index,
        ok: false,
        reason: `config_hash does not match this row's canonical form (expected ${recomputed}) — the row was altered after export`,
      });
      return;
    }
    verdicts.push({ index, ok: true, reason: null });
    accepted.push(row);
  });

  const versions = [...new Set(accepted.map((r) => r.method_version ?? null))];
  return {
    ok: accepted.length > 0,
    fatal: null,
    rows: verdicts,
    acceptedRows: accepted,
    methodVersions: versions,
    mixedVintages: versions.length > 1,
  };
}

function rowProblem(row: unknown): string | null {
  if (!row || typeof row !== "object") return "row must be an object";
  const r = row as Partial<BundleRow>;
  if (typeof r.config_hash !== "string" || !/^[0-9a-f]{64}$/.test(r.config_hash)) {
    return "config_hash must be 64 lowercase hex characters";
  }
  if (r.test_type !== "pp" && r.test_type !== "tg" && r.test_type !== "pg") {
    return "test_type must be pp/tg/pg";
  }
  for (const field of ["avg_tps", "stddev_tps", "ram_peak_mib"] as const) {
    const value = r[field];
    if (typeof value !== "number" || !Number.isFinite(value)) return `${field} must be a finite number`;
  }
  for (const field of [
    "n_prompt",
    "n_gen",
    "n_depth",
    "n_threads",
    "n_gpu_layers",
    "batch_size",
    "ubatch_size",
    "n_gpu_layers_draft",
    "n_cpu_moe",
  ] as const) {
    const value = r[field];
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
      return `${field} must be a non-negative integer`;
    }
  }
  for (const field of ["cache_type_k", "cache_type_v", "flash_attn", "mtp"] as const) {
    if (typeof r[field] !== "string") return `${field} must be a string`;
  }
  return null;
}
