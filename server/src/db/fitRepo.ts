// Storage for the optimization flow's fit map (fit_points) and KV-cache
// support detection (kv_support) -- see schema.sql and
// docs/plans/OPTIMIZATION_FLOW_REDESIGN.md steps 1, 4 and 5.

import { getDb } from "./migrate.js";
import type { FitPoint } from "../../../shared/fitParams.js";
import type { KvSupportRow, KvSupportStatus } from "../../../shared/kvSupport.js";

export class FitPointLimitError extends Error {
  constructor(readonly maxRows: number) {
    super(`a fit run holds at most ${maxRows} points`);
  }
}

interface FitPointRow {
  ctx: number;
  fa: string;
  ctk: string;
  ctv: string;
  margin_mib: number;
  verdict: string;
  reason: string | null;
  ngl: number | null;
  layers_gpu: number | null;
  layers_total: number | null;
  overflow_layers: number | null;
  ot: string | null;
  dev_used_mib: number | null;
  dev_free_mib: number | null;
  host_used_mib: number | null;
  total_mib: number | null;
  kv_mib: number | null;
  compute_mib: number | null;
  need_all_gpu_mib: number | null;
  splits: number | null;
  fa_disabled: number | null;
  n_expert: number | null;
  n_expert_used: number | null;
  inferred: number | null;
  raw_args: string | null;
  error: string | null;
}

function mapPoint(r: FitPointRow): FitPoint {
  return {
    ctx: r.ctx,
    fa: r.fa === "off" ? "off" : "on",
    ctk: r.ctk,
    ctv: r.ctv,
    margin_mib: r.margin_mib,
    verdict: r.verdict as FitPoint["verdict"],
    reason: (r.reason as FitPoint["reason"]) ?? null,
    ngl: r.ngl,
    layers_gpu: r.layers_gpu,
    layers_total: r.layers_total,
    overflow_layers: r.overflow_layers ?? 0,
    ot: r.ot,
    dev_used_mib: r.dev_used_mib,
    dev_free_mib: r.dev_free_mib,
    host_used_mib: r.host_used_mib,
    total_mib: r.total_mib,
    kv_mib: r.kv_mib,
    compute_mib: r.compute_mib,
    need_all_gpu_mib: r.need_all_gpu_mib,
    splits: r.splits,
    fa_disabled: r.fa_disabled === 1,
    n_expert: r.n_expert,
    n_expert_used: r.n_expert_used,
    inferred: r.inferred === 1,
    raw_args: r.raw_args,
    error: r.error,
  };
}

export const fitRepo = {
  /** Throws FitPointLimitError (rolling the whole post back) when the run
   * would end up holding more than `max_rows` points. Re-posting a key that
   * is already stored never counts against it. */
  upsertPoints(input: {
    run_id: string;
    worker_id: string | null;
    model_id: string;
    build: string | null;
    points: FitPoint[];
    max_rows?: number;
  }): number {
    const db = getDb();
    const stmt = db.prepare(
      `INSERT INTO fit_points
         (run_id, worker_id, model_id, ctx, fa, ctk, ctv, margin_mib, verdict, reason, ngl, layers_gpu, layers_total,
          overflow_layers, ot, dev_used_mib, dev_free_mib, host_used_mib, total_mib, kv_mib, compute_mib, need_all_gpu_mib,
          splits, fa_disabled, n_expert, n_expert_used, inferred, raw_args, error, llama_cpp_build, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(run_id, ctx, fa, ctk, ctv) DO UPDATE SET
         margin_mib = excluded.margin_mib, verdict = excluded.verdict, reason = excluded.reason, ngl = excluded.ngl,
         layers_gpu = excluded.layers_gpu, layers_total = excluded.layers_total, overflow_layers = excluded.overflow_layers,
         ot = excluded.ot, dev_used_mib = excluded.dev_used_mib, dev_free_mib = excluded.dev_free_mib,
         host_used_mib = excluded.host_used_mib, total_mib = excluded.total_mib, kv_mib = excluded.kv_mib,
         compute_mib = excluded.compute_mib, need_all_gpu_mib = excluded.need_all_gpu_mib, splits = excluded.splits,
         fa_disabled = excluded.fa_disabled, n_expert = excluded.n_expert, n_expert_used = excluded.n_expert_used,
         inferred = excluded.inferred, raw_args = excluded.raw_args, error = excluded.error, created_at = excluded.created_at`
    );
    const now = Date.now();
    const tx = db.transaction(() => {
      for (const p of input.points) {
        stmt.run(
          input.run_id, input.worker_id, input.model_id, p.ctx, p.fa, p.ctk, p.ctv, p.margin_mib, p.verdict, p.reason,
          p.ngl, p.layers_gpu, p.layers_total, p.overflow_layers, p.ot, p.dev_used_mib, p.dev_free_mib, p.host_used_mib,
          p.total_mib, p.kv_mib, p.compute_mib, p.need_all_gpu_mib, p.splits, p.fa_disabled ? 1 : 0, p.n_expert,
          p.n_expert_used, p.inferred ? 1 : 0, p.raw_args, p.error, input.build, now
        );
      }
      if (input.max_rows !== undefined) {
        const { n } = db.prepare(`SELECT COUNT(*) AS n FROM fit_points WHERE run_id = ?`).get(input.run_id) as { n: number };
        if (n > input.max_rows) throw new FitPointLimitError(input.max_rows);
      }
    });
    tx();
    return input.points.length;
  },

  listPoints(runId: string): FitPoint[] {
    const rows = getDb()
      .prepare(`SELECT * FROM fit_points WHERE run_id = ? ORDER BY ctk, ctv, fa, ctx`)
      .all(runId) as FitPointRow[];
    return rows.map(mapPoint);
  },

  upsertKv(input: { worker_id: string; build: string; backend: string; model_id: string; run_id: string; rows: KvSupportRow[] }): number {
    const db = getDb();
    const stmt = db.prepare(
      `INSERT INTO kv_support (worker_id, llama_cpp_build, backend, model_id, ctk, ctv, fa, status, splits, baseline_splits, detail, run_id, checked_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(worker_id, llama_cpp_build, backend, model_id, ctk, ctv, fa) DO UPDATE SET
         status = excluded.status, splits = excluded.splits, baseline_splits = excluded.baseline_splits,
         detail = excluded.detail, run_id = excluded.run_id, checked_at = excluded.checked_at`
    );
    const now = Date.now();
    db.transaction(() => {
      for (const r of input.rows) {
        stmt.run(input.worker_id, input.build, input.backend, input.model_id, r.ctk, r.ctv, r.fa, r.status, r.splits, r.baseline_splits, r.detail, input.run_id, now);
      }
    })();
    return input.rows.length;
  },

  listKv(filter: { worker_id: string; build: string; backend: string; model_id: string }): KvSupportRow[] {
    const rows = getDb()
      .prepare(
        `SELECT ctk, ctv, fa, status, splits, baseline_splits, detail FROM kv_support
         WHERE worker_id = ? AND llama_cpp_build = ? AND backend = ? AND model_id = ?
         ORDER BY fa, ctk, ctv`
      )
      .all(filter.worker_id, filter.build, filter.backend, filter.model_id) as {
      ctk: string;
      ctv: string;
      fa: string;
      status: string;
      splits: number | null;
      baseline_splits: number | null;
      detail: string | null;
    }[];
    return rows.map((r) => ({
      ctk: r.ctk,
      ctv: r.ctv,
      fa: r.fa === "off" ? "off" : "on",
      status: r.status as KvSupportStatus,
      splits: r.splits,
      baseline_splits: r.baseline_splits,
      detail: r.detail,
    }));
  },
};
