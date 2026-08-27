// BENCHMARKING_PLAN_V8.md N7 -- export a run (or a whole chain root) as a
// self-describing bundle, and import one back with per-row validation.

import type { FastifyInstance } from "fastify";
import { v4 as uuid } from "uuid";
import { repo } from "../db/repo.js";
import { resolveAuthUser } from "../auth-middleware.js";
import type { ResultRow, Run, RunConfig } from "../../../shared/types.js";
import {
  BUNDLE_FORMAT,
  BUNDLE_FORMAT_VERSION,
  methodsFor,
  stampConfigHash,
  validateBundle,
  vramClass,
  type Bundle,
  type BundleRow,
} from "../../../shared/exchange.js";

const MAX_IMPORT_BYTES = 8 * 1024 * 1024;

function toBundleRow(row: ResultRow): BundleRow {
  return stampConfigHash({
    test_type: row.test_type,
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_depth: row.n_depth ?? 0,
    n_threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    mtp: row.mtp,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    n_cpu_moe: row.n_cpu_moe,
    engine: row.engine ?? null,
    concurrency: row.concurrency ?? null,
    method_version: row.method_version ?? null,
    avg_tps: row.avg_tps,
    stddev_tps: row.stddev_tps,
    sample_count: row.sample_count ?? null,
    suspect_count: row.suspect_count ?? null,
    vram_peak_mib: row.vram_peak_mib,
    ram_peak_mib: row.ram_peak_mib,
    ttft_ms_p50: row.ttft_ms_p50 ?? null,
    ttft_ms_p95: row.ttft_ms_p95 ?? null,
    ttft_n: row.ttft_n ?? null,
    e2e_ms_mean: row.e2e_ms_mean ?? null,
    gpu_temp_c_max: row.gpu_temp_c_max ?? null,
    gpu_clock_mhz_min: row.gpu_clock_mhz_min ?? null,
    caveat_flags: row.caveat_flags ?? [],
    created_at: row.created_at,
  });
}

function buildBundle(run: Run, rows: ResultRow[]): Bundle {
  const model = repo.getModel(run.model_id);
  const bundleRows = rows.map(toBundleRow);
  const versions = [...new Set(bundleRows.map((r) => r.method_version ?? null))];
  const sample = rows.find((r) => r.gpu_memory_total_mb != null);
  const quality = repo.qualityRepo.listForRoot(run.root_run_id ?? run.id);
  return {
    format: BUNDLE_FORMAT,
    format_version: BUNDLE_FORMAT_VERSION,
    exported_at: Date.now(),
    run: {
      id: run.id,
      root_run_id: run.root_run_id ?? null,
      kind: run.kind ?? null,
      model_filename: model?.filename ?? run.model_id,
      model_quant: model?.metadata.quant ?? null,
      // models.id IS the file's sha256 (see schema.sql) -- content-addressed,
      // so it identifies the FILE without identifying the machine.
      model_sha256: run.model_id,
      llama_cpp_build: run.llama_cpp_build ?? null,
      // The full stored configuration, M2's goals block included, so results
      // stay reproducible without reference to whatever the defaults were.
      config: (run.config ?? {}) as RunConfig & Record<string, unknown>,
    },
    // Anonymized hardware class -- GPU name and a VRAM bucket, no serials and
    // no machine identity.
    hardware_class: {
      gpu_name: run.backend_device_name ?? null,
      backend: run.llama_cpp_backend ?? null,
      vram_class_mib: vramClass(sample?.gpu_memory_total_mb ?? null),
      cpu_isa: rows.find((r) => r.cpu_isa)?.cpu_isa ?? null,
    },
    rows: bundleRows,
    quality_rows: quality.map((q) => ({
      model_id: q.model_id,
      ctx_tokens: q.ctx_tokens,
      cache_type_k: q.cache_type_k,
      cache_type_v: q.cache_type_v,
      ppl: q.ppl,
      kld_vs_baseline: q.kld_vs_baseline,
      dataset_hash: q.dataset_hash,
      method_version: q.method_version,
    })),
    method_versions: versions,
    // Every shared number carries its own methods section.
    methods: [...new Set(versions.filter((v): v is number => v != null))].map(methodsFor),
  };
}

export async function exchangeRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Params: { id: string }; Querystring: { scope?: string } }>(
    "/api/runs/:id/export",
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const userId = authed?.user.id;
      const run = repo.getRun(userId, request.params.id);
      if (!run) return reply.code(404).send({ error: "run not found" });

      const rows: ResultRow[] = [];
      if (request.query.scope === "root") {
        for (const member of repo.listRunsUnderRoot(userId, run.root_run_id ?? run.id)) {
          rows.push(...repo.getResultsForRun(member.id));
        }
      } else {
        rows.push(...repo.getResultsForRun(run.id));
      }

      const bundle = buildBundle(run, rows);
      reply.header("content-type", "application/json; charset=utf-8");
      reply.header("content-disposition", `attachment; filename="llamatoaster-${run.id}.json"`);
      return reply.send(JSON.stringify(bundle, null, 2));
    }
  );

  app.post<{ Body: { bundle?: unknown; opt_in_scoring?: boolean } }>(
    "/api/import",
    { bodyLimit: MAX_IMPORT_BYTES },
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const userId = authed?.user.id;
      const validation = validateBundle(request.body?.bundle);
      if (validation.fatal) {
        request.log.warn({ import_rejected: true, reason: validation.fatal, row_id: null }, "import_rejected");
        return reply.code(400).send({ error: validation.fatal, rows: [] });
      }
      for (const verdict of validation.rows) {
        if (!verdict.ok) {
          request.log.warn(
            { import_rejected: true, reason: verdict.reason, row_id: verdict.index },
            "import_rejected"
          );
        }
      }
      if (validation.acceptedRows.length === 0) {
        return reply.code(400).send({ error: "every row in that bundle failed validation", rows: validation.rows });
      }

      const bundle = request.body!.bundle as Bundle;
      // Imported rows are badged and never merge into local profile scoring
      // unless opted in for THIS import.
      const optIn = request.body?.opt_in_scoring === true;
      const bundleId = uuid();
      const imported = repo.importBundleRows(userId, {
        bundleId,
        optIn,
        run: bundle.run,
        hardwareClass: bundle.hardware_class,
        rows: validation.acceptedRows,
      });

      return reply.code(201).send({
        bundle_id: bundleId,
        run_id: imported.runId,
        imported_rows: validation.acceptedRows.length,
        rejected_rows: validation.rows.filter((r) => !r.ok),
        // §0.1's display rule: surfaced together, never averaged together.
        method_versions: validation.methodVersions,
        mixed_vintages: validation.mixedVintages,
        opted_into_scoring: optIn,
        notice: optIn
          ? "These rows were opted into local scoring for this import only; they stay badged as imported wherever they appear."
          : "Imported rows are badged and excluded from local profile scoring. Re-import with opt_in_scoring to include them.",
      });
    }
  );
}
