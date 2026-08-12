import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { getDb } from "../db/migrate.js";

interface ResultExportRow {
  run_id: string;
  worker_name: string;
  model_id: string;
  model_filename: string;
  test_type: string;
  n_prompt: number;
  n_gen: number;
  n_threads: number;
  n_gpu_layers: number;
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
        `SELECT r.*, runs.worker_name AS worker_name, m.filename AS model_filename
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
        `SELECT r.*, runs.worker_name AS worker_name, m.filename AS model_filename
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
          "run_id,worker_name,model_id,model_filename,test_type,n_prompt,n_gen,n_threads,n_gpu_layers,batch_size,ubatch_size,cache_type_k,cache_type_v,flash_attn,mtp,avg_tps,stddev_tps,ram_peak_mib,vram_peak_mib,sample_count,suspect_count,suspect_samples,repeat_samples,spec_accepted_drafted";
        const lines = rows.map((r) =>
          [
            neutralizeFormula(r.run_id),
            neutralizeFormula(r.worker_name),
            neutralizeFormula(r.model_id),
            neutralizeFormula(r.model_filename),
            r.test_type,
            r.n_prompt,
            r.n_gen,
            r.n_threads,
            r.n_gpu_layers,
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
        const header =
          "| run | worker | model | test | n_prompt | n_gen | threads | ngls | batch | ubatch | ctk | ctv | fa | mtp | avg_tps | stddev | ram_mib | vram_mib | suspect | suspect raw | per-run samples | mtp accepted/drafted |";
        const sep =
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |";
        const lines = rows.map(
          (r) =>
            `| ${neutralizeFormula(r.run_id.slice(0, 8))} | ${neutralizeFormula(r.worker_name)} | ${neutralizeFormula(r.model_filename)} | ${r.test_type} | ${r.n_prompt} | ${r.n_gen} | ${r.n_threads} | ${r.n_gpu_layers} | ${r.batch_size} | ${r.ubatch_size} | ${r.cache_type_k} | ${r.cache_type_v} | ${r.flash_attn} | ${r.mtp} | ${r.avg_tps.toFixed(2)} | ${r.stddev_tps.toFixed(2)} | ${r.ram_peak_mib} | ${fmtNullable(r.vram_peak_mib)} | ${fmtNullable(r.suspect_count)}/${fmtNullable(r.sample_count)} | ${fmtSampleList(r.suspect_samples)} | ${fmtSampleList(r.repeat_samples)} | ${fmtSpecDecode(r.spec_drafted, r.spec_accepted)} |`
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
