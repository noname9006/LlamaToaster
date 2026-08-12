import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { getDb } from "../db/migrate.js";

interface ResultExportRow {
  run_id: string;
  worker_name: string;
  // Denormalized from the parent run via loadExportRows' own JOIN, same as
  // worker_name above -- see shared/types.ts's ResultRow doc comment for why
  // these aren't physical columns on `results` itself.
  backend_type: string;
  backend_device_name: string | null;
  model_id: string;
  model_filename: string;
  test_type: string;
  n_prompt: number;
  n_gen: number;
  n_threads: number;
  n_gpu_layers: number;
  // Read from llama.cpp's own runtime output, never inferred -- see
  // worker/src/index.ts's parseOffloadLayers.
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  batch_size: number;
  ubatch_size: number;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  mtp: string;
  avg_tps: number;
  stddev_tps: number;
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  // Raw DB columns (SELECT r.* below), not the API's _mb-suffixed
  // ResultRow shape -- this route has never transformed column names, only
  // passed them through, so these keep the same _mib convention every
  // sibling field in this interface already uses. accuracy/source are
  // string | null (not the narrower GpuMemoryAccuracyLevel/
  // GpuMemoryMeasurementSource types) since a row from before this
  // migration genuinely has SQL NULL here, unlike the API's mapResult,
  // which coalesces that to "unavailable".
  system_memory_total_mib: number | null;
  gpu_memory_total_mib: number | null;
  gpu_memory_total_accuracy: string | null;
  gpu_memory_total_source: string | null;
  gpu_memory_free_start_accuracy: string | null;
  gpu_memory_free_start_source: string | null;
  gpu_memory_model_avg_accuracy: string | null;
  gpu_memory_model_avg_source: string | null;
  gpu_memory_model_peak_accuracy: string | null;
  gpu_memory_model_peak_source: string | null;
  // Only ever populated by the llama-server/MTP path -- see shared/types.ts's
  // ResultRow doc comment. suspect_samples comes straight out of the DB as
  // JSON text (SQLite has no array column type), NULL when there were none.
  sample_count: number | null;
  suspect_count: number | null;
  suspect_samples: string | null;
  // Every individual repeat's raw reading for this test_type, in repeat
  // order -- see shared/types.ts's ResultRow doc comment for how this
  // differs from suspect_samples (a flagged-only subset). Same JSON-text
  // storage as suspect_samples (SQLite has no array column type).
  repeat_samples: string | null;
  // llama-server /metrics speculative-decoding counters (tg row of an MTP
  // item only) -- see shared/types.ts's ResultRow doc comment. Both null
  // means /metrics was unreachable or had no counters at all, distinct from
  // spec_drafted: 0 (draft head loaded but drafted nothing).
  spec_drafted: number | null;
  spec_accepted: number | null;
}

// Neutralize CSV/formula injection: a field starting with =, +, -, @, tab, or
// CR gets interpreted as a formula by Excel/Sheets when the export is opened
// there. model_filename/worker_name/model_id are free text a client can set
// via the API, so they need this before being written out.
function neutralizeFormula(value: string): string {
  return /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
}

function fmtNullable(value: number | null): string {
  return value === null ? "n/a" : String(value);
}

function fmtNullableStr(value: string | null): string {
  return value === null ? "n/a" : value;
}

// spec_drafted/spec_accepted are raw counters (see ResultExportRow) -- this
// derives the accept rate a reader actually wants, e.g. "210/340 (61.8%)",
// rather than making every consumer of the export recompute it. "n/a" means
// /metrics never confirmed anything (unreachable or no counters at all);
// spec_drafted: 0 renders as "0/0" -- draft head loaded but drafted nothing,
// a distinct, worse condition than a missing reading.
function fmtSpecDecode(drafted: number | null, accepted: number | null): string {
  if (drafted === null || accepted === null) return "n/a";
  const rate = drafted > 0 ? ` (${((accepted / drafted) * 100).toFixed(1)}%)` : "";
  return `${accepted}/${drafted}${rate}`;
}

// suspect_samples/repeat_samples are stored as JSON text (see
// ResultExportRow) -- reformatted here as a semicolon-separated list so it
// reads as plain data in a CSV/MD cell rather than a nested JSON blob inside
// a quoted field. Shared by both columns since the storage/formatting is
// identical, only which array each holds differs.
function fmtSampleList(json: string | null): string {
  if (!json) return "";
  try {
    const values = JSON.parse(json);
    return Array.isArray(values) ? values.map((v) => Number(v).toFixed(1)).join("; ") : "";
  } catch {
    return "";
  }
}

function loadExportRows(runIds?: string[]): ResultExportRow[] {
  const db = getDb();
  let rows: any[];
  if (runIds && runIds.length) {
    const placeholders = runIds.map(() => "?").join(",");
    rows = db
      .prepare(
        `SELECT r.*, runs.worker_name AS worker_name, m.filename AS model_filename,
                runs.llama_cpp_backend AS backend_type, runs.backend_device_name AS backend_device_name
         FROM results r
         JOIN runs ON runs.id = r.run_id
         JOIN models m ON m.id = r.model_id
         WHERE r.run_id IN (${placeholders})
         ORDER BY r.run_id, r.created_at`
      )
      .all(...runIds);
  } else {
    rows = db
      .prepare(
        `SELECT r.*, runs.worker_name AS worker_name, m.filename AS model_filename,
                runs.llama_cpp_backend AS backend_type, runs.backend_device_name AS backend_device_name
         FROM results r
         JOIN runs ON runs.id = r.run_id
         JOIN models m ON m.id = r.model_id
         ORDER BY r.run_id, r.created_at`
      )
      .all();
  }
  return rows as ResultExportRow[];
}

export async function resultsRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Querystring: { format?: string; runs?: string } }>(
    "/api/results/export",
    async (request, reply) => {
      const format = (request.query.format ?? "json").toLowerCase();
      const runIds = request.query.runs
        ? request.query.runs.split(",").map((s) => s.trim()).filter(Boolean)
        : undefined;
      const rows = loadExportRows(runIds);

      if (format === "csv") {
        const header =
          "run_id,worker_name,backend_type,backend_device_name,model_id,model_filename,test_type,n_prompt,n_gen,n_threads,n_gpu_layers,gpu_layers_loaded,total_model_layers,batch_size,ubatch_size,cache_type_k,cache_type_v,flash_attn,mtp,avg_tps,stddev_tps,ram_peak_mib,vram_peak_mib,system_memory_total_mib,gpu_memory_total_mib,gpu_memory_total_accuracy,gpu_memory_total_source,gpu_memory_free_start_accuracy,gpu_memory_free_start_source,gpu_memory_model_avg_accuracy,gpu_memory_model_avg_source,gpu_memory_model_peak_accuracy,gpu_memory_model_peak_source,sample_count,suspect_count,suspect_samples,repeat_samples,spec_accepted_drafted";
        const lines = rows.map((r) =>
          [
            neutralizeFormula(r.run_id),
            neutralizeFormula(r.worker_name),
            r.backend_type,
            neutralizeFormula(fmtNullableStr(r.backend_device_name)),
            neutralizeFormula(r.model_id),
            neutralizeFormula(r.model_filename),
            r.test_type,
            r.n_prompt,
            r.n_gen,
            r.n_threads,
            r.n_gpu_layers,
            fmtNullable(r.gpu_layers_loaded),
            fmtNullable(r.total_model_layers),
            r.batch_size,
            r.ubatch_size,
            r.cache_type_k,
            r.cache_type_v,
            r.flash_attn,
            r.mtp,
            r.avg_tps,
            r.stddev_tps,
            r.ram_peak_mib,
            fmtNullable(r.vram_peak_mib),
            fmtNullable(r.system_memory_total_mib),
            fmtNullable(r.gpu_memory_total_mib),
            fmtNullableStr(r.gpu_memory_total_accuracy),
            fmtNullableStr(r.gpu_memory_total_source),
            fmtNullableStr(r.gpu_memory_free_start_accuracy),
            fmtNullableStr(r.gpu_memory_free_start_source),
            fmtNullableStr(r.gpu_memory_model_avg_accuracy),
            fmtNullableStr(r.gpu_memory_model_avg_source),
            fmtNullableStr(r.gpu_memory_model_peak_accuracy),
            fmtNullableStr(r.gpu_memory_model_peak_source),
            fmtNullable(r.sample_count),
            fmtNullable(r.suspect_count),
            fmtSampleList(r.suspect_samples),
            fmtSampleList(r.repeat_samples),
            fmtSpecDecode(r.spec_drafted, r.spec_accepted),
          ]
            .map((v) => `"${String(v).replace(/"/g, '""')}"`)
            .join(",")
        );
        const csv = [header, ...lines].join("\n");
        reply.header("content-type", "text/csv");
        reply.header(
          "content-disposition",
          'attachment; filename="results.csv"'
        );
        return csv;
      }

      if (format === "md") {
        // backend/layers/vram total are condensed cells here (same idea as
        // the existing "suspect"/"mtp accepted/drafted" columns, which
        // already combine two raw fields into one) rather than the full
        // per-field accuracy/source breakdown CSV/JSON export -- MD has
        // always been a lighter, more human-skimmable summary than CSV
        // (already fewer/combined columns even before this change), not a
        // complete dump; reach for CSV or JSON for full provenance detail.
        const header =
          "| run | worker | backend | model | test | n_prompt | n_gen | threads | ngls | layers | batch | ubatch | ctk | ctv | fa | mtp | avg_tps | stddev | ram_mib | vram_mib | vram total | suspect | suspect raw | per-run samples | mtp accepted/drafted |";
        const sep =
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |";
        const lines = rows.map(
          (r) =>
            `| ${neutralizeFormula(r.run_id.slice(0, 8))} | ${neutralizeFormula(r.worker_name)} | ${neutralizeFormula(`${r.backend_type}${r.backend_device_name ? ` (${r.backend_device_name})` : ""}`)} | ${neutralizeFormula(r.model_filename)} | ${r.test_type} | ${r.n_prompt} | ${r.n_gen} | ${r.n_threads} | ${r.n_gpu_layers} | ${fmtNullable(r.gpu_layers_loaded)}/${fmtNullable(r.total_model_layers)} | ${r.batch_size} | ${r.ubatch_size} | ${r.cache_type_k} | ${r.cache_type_v} | ${r.flash_attn} | ${r.mtp} | ${r.avg_tps.toFixed(2)} | ${r.stddev_tps.toFixed(2)} | ${r.ram_peak_mib} | ${fmtNullable(r.vram_peak_mib)} | ${fmtNullable(r.gpu_memory_total_mib)} | ${fmtNullable(r.suspect_count)}/${fmtNullable(r.sample_count)} | ${fmtSampleList(r.suspect_samples)} | ${fmtSampleList(r.repeat_samples)} | ${fmtSpecDecode(r.spec_drafted, r.spec_accepted)} |`
        );
        const md = ["# Benchmark results", "", header, sep, ...lines].join("\n");
        reply.header("content-type", "text/markdown");
        reply.header(
          "content-disposition",
          'attachment; filename="results.md"'
        );
        return md;
      }

      reply.header("content-type", "application/json");
      reply.header("content-disposition", 'attachment; filename="results.json"');
      return JSON.stringify(rows, null, 2);
    }
  );
}
