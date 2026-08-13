import { v4 as uuid } from "uuid";
import { createHash } from "node:crypto";
import { getDb } from "./migrate.js";
import type {
  Model,
  ModelMetadata,
  RegisterModelInput,
  Run,
  RunConfig,
  RunStatus,
  ResultRow,
  IngestResultInput,
  RunItem,
  RunItemTickInput,
  RunItemTerminalInput,
  GpuMemoryAccuracyLevel,
} from "../../../shared/types.js";
import type { SweepItem } from "../../../shared/sweep.js";

interface ModelRow {
  id: string;
  filename: string;
  size_bytes: number;
  source: string;
  hf_repo: string | null;
  hf_file: string | null;
  metadata: string;
  created_at: number;
}

interface RunRow {
  id: string;
  worker_name: string;
  llama_cpp_build: string;
  llama_cpp_backend: string;
  backend_device_name: string | null;
  model_id: string;
  config: string;
  status: string;
  error: string | null;
  started_at: number;
  completed_at: number | null;
  model_filename?: string | null;
  items_total?: number;
  items_done?: number;
  items_failed?: number;
  items_cancelled?: number;
}

interface RunItemRow {
  id: string;
  run_id: string;
  idx: number;
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
  n_gpu_layers_draft: number;
  status: string;
  detail: string | null;
  ram_mib: number | null;
  vram_mib: number | null;
  ram_peak_mib: number | null;
  vram_peak_mib: number | null;
  ram_avg_mib: number | null;
  vram_avg_mib: number | null;
  ram_free_before_mib: number | null;
  vram_free_before_mib: number | null;
  live_tps: number | null;
  error: string | null;
  started_at: number | null;
  completed_at: number | null;
}

interface ResultRowRaw {
  id: string;
  run_id: string;
  idx: number | null;
  model_id: string;
  // Denormalized from the parent run via getResultsForRun's JOIN, not a
  // physical column on `results` itself -- see ResultRow's own doc comment.
  backend_type: string;
  backend_device_name: string | null;
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
  n_gpu_layers_draft: number;
  avg_tps: number;
  stddev_tps: number;
  ram_peak_mib: number;
  vram_peak_mib: number | null;
  ram_avg_mib: number;
  vram_avg_mib: number | null;
  ram_free_before_mib: number | null;
  vram_free_before_mib: number | null;
  // Raw storage is nullable regardless of shared/types.ts's ResultRow
  // typing accuracy fields as non-nullable strings: NULL here means this row
  // predates the migration that added these columns, not "unavailable" as a
  // deliberate measurement -- mapResult below coalesces that distinction
  // away for anything reading through the public type.
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
  gpu_layers_loaded: number | null;
  total_model_layers: number | null;
  gpu_layers_loaded_draft: number | null;
  total_model_layers_draft: number | null;
  sample_count: number | null;
  suspect_count: number | null;
  suspect_samples: string | null;
  repeat_samples: string | null;
  spec_drafted: number | null;
  spec_accepted: number | null;
  raw_json_path: string | null;
  created_at: number;
}

function mapModel(row: ModelRow): Model {
  return {
    id: row.id,
    filename: row.filename,
    size_bytes: row.size_bytes,
    source: row.source as Model["source"],
    hf_repo: row.hf_repo ?? undefined,
    hf_file: row.hf_file ?? undefined,
    metadata: JSON.parse(row.metadata || "{}"),
    created_at: row.created_at,
  };
}

function mapRun(row: RunRow): Run {
  return {
    id: row.id,
    worker_name: row.worker_name,
    llama_cpp_build: row.llama_cpp_build,
    llama_cpp_backend: row.llama_cpp_backend as Run["llama_cpp_backend"],
    backend_device_name: row.backend_device_name ?? undefined,
    model_id: row.model_id,
    model_filename: row.model_filename ?? undefined,
    config: JSON.parse(row.config),
    status: row.status as RunStatus,
    error: row.error ?? undefined,
    started_at: row.started_at,
    completed_at: row.completed_at ?? undefined,
    items_total: row.items_total ?? 0,
    items_done: row.items_done ?? 0,
    items_failed: row.items_failed ?? 0,
    items_cancelled: row.items_cancelled ?? 0,
  };
}

function mapRunItem(row: RunItemRow): RunItem {
  return {
    id: row.id,
    run_id: row.run_id,
    idx: row.idx,
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    mtp: row.mtp,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    status: row.status as RunItem["status"],
    detail: row.detail ?? undefined,
    ram_mib: row.ram_mib,
    vram_mib: row.vram_mib,
    ram_peak_mib: row.ram_peak_mib,
    vram_peak_mib: row.vram_peak_mib,
    ram_avg_mib: row.ram_avg_mib,
    vram_avg_mib: row.vram_avg_mib,
    ram_free_before_mib: row.ram_free_before_mib,
    vram_free_before_mib: row.vram_free_before_mib,
    live_tps: row.live_tps,
    error: row.error ?? undefined,
    started_at: row.started_at ?? undefined,
    completed_at: row.completed_at ?? undefined,
  };
}

function safeParseNumberArray(json: string): number[] | undefined {
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) && parsed.every((v) => typeof v === "number") ? parsed : undefined;
  } catch {
    return undefined;
  }
}

// A raw DB NULL here means this row predates the migration that added these
// columns -- reads the same as a deliberate "unavailable" measurement to
// anything consuming the public ResultRow type, per shared/types.ts's
// GpuMemoryAccuracyLevel doc comment (accuracy is always a concrete level,
// never absent).
function coalesceAccuracy(raw: string | null): GpuMemoryAccuracyLevel {
  return (raw as GpuMemoryAccuracyLevel | null) ?? "unavailable";
}

function mapResult(row: ResultRowRaw): ResultRow {
  return {
    id: row.id,
    run_id: row.run_id,
    // -1 only for pre-run_items-era historical rows that backfillResultIdx
    // (migrate.ts) couldn't recover -- harmless, it just never matches a
    // real item.idx in the client's join.
    idx: row.idx ?? -1,
    model_id: row.model_id,
    test_type: row.test_type as ResultRow["test_type"],
    n_prompt: row.n_prompt,
    n_gen: row.n_gen,
    n_threads: row.n_threads,
    n_gpu_layers: row.n_gpu_layers,
    batch_size: row.batch_size,
    ubatch_size: row.ubatch_size,
    cache_type_k: row.cache_type_k,
    cache_type_v: row.cache_type_v,
    flash_attn: row.flash_attn,
    mtp: row.mtp,
    n_gpu_layers_draft: row.n_gpu_layers_draft,
    avg_tps: row.avg_tps,
    stddev_tps: row.stddev_tps,
    ram_peak_mib: row.ram_peak_mib,
    vram_peak_mib: row.vram_peak_mib,
    ram_avg_mib: row.ram_avg_mib,
    vram_avg_mib: row.vram_avg_mib,
    ram_free_before_mib: row.ram_free_before_mib,
    vram_free_before_mib: row.vram_free_before_mib,
    backend_type: row.backend_type,
    backend_device_name: row.backend_device_name,
    system_memory_total_mb: row.system_memory_total_mib,
    gpu_memory_total_mb: row.gpu_memory_total_mib,
    gpu_memory_total_accuracy: coalesceAccuracy(row.gpu_memory_total_accuracy),
    gpu_memory_total_source: row.gpu_memory_total_source as ResultRow["gpu_memory_total_source"],
    gpu_memory_free_start_accuracy: coalesceAccuracy(row.gpu_memory_free_start_accuracy),
    gpu_memory_free_start_source: row.gpu_memory_free_start_source as ResultRow["gpu_memory_free_start_source"],
    gpu_memory_model_avg_accuracy: coalesceAccuracy(row.gpu_memory_model_avg_accuracy),
    gpu_memory_model_avg_source: row.gpu_memory_model_avg_source as ResultRow["gpu_memory_model_avg_source"],
    gpu_memory_model_peak_accuracy: coalesceAccuracy(row.gpu_memory_model_peak_accuracy),
    gpu_memory_model_peak_source: row.gpu_memory_model_peak_source as ResultRow["gpu_memory_model_peak_source"],
    gpu_layers_loaded: row.gpu_layers_loaded,
    total_model_layers: row.total_model_layers,
    gpu_layers_loaded_draft: row.gpu_layers_loaded_draft ?? undefined,
    total_model_layers_draft: row.total_model_layers_draft ?? undefined,
    sample_count: row.sample_count ?? undefined,
    suspect_count: row.suspect_count ?? undefined,
    // Stored as JSON text (SQLite has no array column type) -- tolerate a
    // corrupt/unexpected value by dropping it rather than throwing, since
    // this is diagnostic-only data, not load-bearing for the run itself.
    suspect_samples: row.suspect_samples ? safeParseNumberArray(row.suspect_samples) : undefined,
    repeat_samples: row.repeat_samples ? safeParseNumberArray(row.repeat_samples) : undefined,
    spec_drafted: row.spec_drafted ?? undefined,
    spec_accepted: row.spec_accepted ?? undefined,
    raw_json_path: row.raw_json_path ?? undefined,
    created_at: row.created_at,
  };
}

// The VPS has no filesystem access to the actual model file (it lives on the
// worker's machine), so it can't compute a real sha256 of the bytes. When the
// caller omits an id, derive a stable one from the identifying fields it did
// supply, so re-registering the same model doesn't create a duplicate row
// with a fresh random id each time.
function deriveModelId(input: RegisterModelInput): string {
  const key =
    input.source === "huggingface"
      ? `hf:${input.hf_repo ?? ""}/${input.hf_file ?? ""}`
      : `local:${input.filename ?? ""}:${input.size_bytes ?? 0}`;
  return createHash("sha256").update(key).digest("hex");
}

export const repo = {
  listModels(): Model[] {
    const rows = getDb()
      .prepare("SELECT * FROM models ORDER BY created_at DESC")
      .all() as ModelRow[];
    return rows.map(mapModel);
  },

  getModel(id: string): Model | undefined {
    const row = getDb().prepare("SELECT * FROM models WHERE id = ?").get(id) as
      | ModelRow
      | undefined;
    return row ? mapModel(row) : undefined;
  },

  registerModel(input: RegisterModelInput): Model {
    const id = input.id ?? deriveModelId(input);
    const now = Date.now();
    const metadata = JSON.stringify(input.metadata ?? {});
    getDb()
      .prepare(
        `INSERT INTO models (id, filename, size_bytes, source, hf_repo, hf_file, metadata, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           filename = excluded.filename,
           size_bytes = excluded.size_bytes,
           source = excluded.source,
           hf_repo = excluded.hf_repo,
           hf_file = excluded.hf_file,
           metadata = excluded.metadata`
      )
      .run(
        id,
        input.filename ?? "",
        input.size_bytes ?? 0,
        input.source,
        input.hf_repo ?? null,
        input.hf_file ?? null,
        metadata,
        now
      );
    return this.getModel(id)!;
  },

  // Merges into a model's existing metadata rather than replacing it
  // wholesale like registerModel's upsert does -- used to backfill a single
  // derived field (e.g. n_layer) onto a model that predates it without
  // disturbing anything else already recorded (arch, quant, ...).
  updateModelMetadata(id: string, patch: Partial<ModelMetadata>): Model | undefined {
    const existing = this.getModel(id);
    if (!existing) return undefined;
    const metadata = JSON.stringify({ ...existing.metadata, ...patch });
    getDb().prepare("UPDATE models SET metadata = ? WHERE id = ?").run(metadata, id);
    return this.getModel(id);
  },

  // results.model_id is a foreign key (foreign_keys=ON, see migrate.ts) with
  // no ON DELETE clause, so SQLite rejects deleting a model any existing
  // result row still references -- check first so the route can return a
  // clean 409 instead of a raw SQLite constraint error.
  countResultsForModel(modelId: string): number {
    const row = getDb().prepare("SELECT COUNT(*) as n FROM results WHERE model_id = ?").get(modelId) as {
      n: number;
    };
    return row.n;
  },

  // A model deleted while a run against it is still in flight would let the
  // worker's eventual per-item terminal report (POST /api/runs/:id/items/:idx,
  // see routes/runs.ts) hit the same FK constraint above once it tries to
  // insert a `results` row for a model that's already gone -- that insert
  // would fail, the worker's safeItemTerminal would retry and give up, and
  // that item (and potentially the whole run) would be stuck "running"
  // forever with no result and no clear error. Block deletion up front
  // instead.
  countRunningRunsForModel(modelId: string): number {
    const row = getDb()
      .prepare("SELECT COUNT(*) as n FROM runs WHERE model_id = ? AND status = 'running'")
      .get(modelId) as { n: number };
    return row.n;
  },

  deleteModel(id: string): Model | undefined {
    const model = this.getModel(id);
    if (!model) return undefined;
    getDb().prepare("DELETE FROM models WHERE id = ?").run(id);
    return model;
  },

  listRuns(): Run[] {
    const rows = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'cancelled') AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         ORDER BY runs.started_at DESC`
      )
      .all() as RunRow[];
    return rows.map(mapRun);
  },

  getRun(id: string): Run | undefined {
    const row = getDb()
      .prepare(
        `SELECT runs.*, m.filename AS model_filename,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id) AS items_total,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'done') AS items_done,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status IN ('failed','failed_oom')) AS items_failed,
                (SELECT COUNT(*) FROM run_items WHERE run_items.run_id = runs.id AND status = 'cancelled') AS items_cancelled
         FROM runs
         LEFT JOIN models m ON m.id = runs.model_id
         WHERE runs.id = ?`
      )
      .get(id) as RunRow | undefined;
    return row ? mapRun(row) : undefined;
  },

  getRunWithResults(id: string): { run: Run; results: ResultRow[]; items: RunItem[] } | undefined {
    const run = this.getRun(id);
    if (!run) return undefined;
    const results = this.getResultsForRun(id);
    const items = this.getRunItems(id);
    return { run, results, items };
  },

  // backend_type/backend_device_name are denormalized from the parent run
  // here, not physical columns on `results` itself -- see ResultRow's own
  // doc comment for why (backend can't vary between one run's own items,
  // so repeating an identical string on every row would be pure noise).
  // Mirrors worker_name's own existing JOIN in the CSV/MD export
  // (server/src/routes/results.ts).
  getResultsForRun(runId: string): ResultRow[] {
    const rows = getDb()
      .prepare(
        `SELECT r.*, runs.llama_cpp_backend AS backend_type, runs.backend_device_name AS backend_device_name
         FROM results r
         JOIN runs ON runs.id = r.run_id
         WHERE r.run_id = ?
         ORDER BY r.created_at ASC`
      )
      .all(runId) as ResultRowRaw[];
    return rows.map(mapResult);
  },

  createRun(run: Run): void {
    getDb()
      .prepare(
        `INSERT INTO runs (id, worker_name, llama_cpp_build, llama_cpp_backend, backend_device_name, model_id, config, status, error, started_at, completed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        run.id,
        run.worker_name,
        run.llama_cpp_build,
        run.llama_cpp_backend,
        run.backend_device_name ?? null,
        run.model_id,
        JSON.stringify(run.config),
        run.status,
        run.error ?? null,
        run.started_at,
        run.completed_at ?? null
      );
  },

  // Pre-creates the full planned item list at trigger time (before the
  // worker is even contacted) so the UI can show "N total tests" and a
  // queued list immediately. idx must match what the worker will report
  // against -- both sides compute it by calling shared/sweep.ts's
  // expandSweep on the exact same sweep JSON, so there's no separate
  // registration round trip.
  createRunItems(runId: string, items: SweepItem[]): void {
    const database = getDb();
    const insert = database.prepare(
      `INSERT INTO run_items
         (id, run_id, idx, n_prompt, n_gen, n_threads, n_gpu_layers,
          batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft, status)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')`
    );
    const tx = database.transaction((rows: SweepItem[]) => {
      for (const item of rows) {
        insert.run(
          uuid(),
          runId,
          item.idx,
          item.n_prompt,
          item.n_gen,
          item.threads,
          item.n_gpu_layers,
          item.batch_size,
          item.ubatch_size,
          item.cache_type_k,
          item.cache_type_v,
          item.flash_attn,
          item.mtp,
          item.n_gpu_layers_draft
        );
      }
    });
    tx(items);
  },

  getRunItems(runId: string): RunItem[] {
    const rows = getDb()
      .prepare("SELECT * FROM run_items WHERE run_id = ? ORDER BY idx ASC")
      .all(runId) as RunItemRow[];
    return rows.map(mapRunItem);
  },

  // Best-effort progress tick (loading/processing/generating/benchmarking) --
  // called far more often than the terminal update below, so this is a
  // plain UPDATE with no completion bookkeeping. ram_free_before_mib/
  // vram_free_before_mib are set once via COALESCE-on-existing-value (same
  // pattern as started_at) since they're only ever sent on an item's very
  // first tick.
  updateRunItemTick(runId: string, idx: number, patch: RunItemTickInput): void {
    getDb()
      .prepare(
        `UPDATE run_items SET
           status = ?,
           detail = COALESCE(?, detail),
           ram_mib = ?,
           vram_mib = ?,
           live_tps = COALESCE(?, live_tps),
           ram_free_before_mib = COALESCE(ram_free_before_mib, ?),
           vram_free_before_mib = COALESCE(vram_free_before_mib, ?),
           started_at = COALESCE(started_at, ?)
         WHERE run_id = ? AND idx = ?`
      )
      .run(
        patch.status,
        patch.detail ?? null,
        patch.ram_mib ?? null,
        patch.vram_mib ?? null,
        patch.live_tps ?? null,
        patch.ram_free_before_mib ?? null,
        patch.vram_free_before_mib ?? null,
        Date.now(),
        runId,
        idx
      );
  },

  // Records one item's final outcome, writes a `results` row for a
  // successful item, and -- once every item for the run is terminal --
  // finalizes the run itself (done/partial/failed). Returns undefined if the
  // run doesn't exist so the route can 404 instead of silently no-op'ing.
  recordRunItemTerminal(runId: string, idx: number, input: RunItemTerminalInput): Run | undefined {
    const database = getDb();
    const run = this.getRun(runId);
    if (!run) return undefined;

    const tx = database.transaction(() => {
      const now = Date.now();
      database
        .prepare(
          `UPDATE run_items SET
             status = ?, error = ?, completed_at = ?,
             ram_peak_mib = ?, vram_peak_mib = ?,
             ram_avg_mib = ?, vram_avg_mib = ?
           WHERE run_id = ? AND idx = ?`
        )
        .run(
          input.status,
          input.error ?? null,
          now,
          input.ram_peak_mib ?? null,
          input.vram_peak_mib ?? null,
          input.ram_avg_mib ?? null,
          input.vram_avg_mib ?? null,
          runId,
          idx
        );

      if (input.status === "done" && input.results) {
        const insertResult = database.prepare(
          `INSERT INTO results
             (id, run_id, idx, model_id, test_type, n_prompt, n_gen, n_threads, n_gpu_layers,
              batch_size, ubatch_size, cache_type_k, cache_type_v, flash_attn, mtp, n_gpu_layers_draft,
              avg_tps, stddev_tps, ram_peak_mib, vram_peak_mib,
              ram_avg_mib, vram_avg_mib, ram_free_before_mib, vram_free_before_mib,
              system_memory_total_mib, gpu_memory_total_mib,
              gpu_memory_total_accuracy, gpu_memory_total_source,
              gpu_memory_free_start_accuracy, gpu_memory_free_start_source,
              gpu_memory_model_avg_accuracy, gpu_memory_model_avg_source,
              gpu_memory_model_peak_accuracy, gpu_memory_model_peak_source,
              gpu_layers_loaded, total_model_layers, gpu_layers_loaded_draft, total_model_layers_draft,
              sample_count, suspect_count, suspect_samples, repeat_samples, spec_drafted, spec_accepted,
              raw_json_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        );
        // Up to two rows for one idx (a pp row and a tg row from the same
        // benchmark process) -- distinguished by test_type, see
        // worker/src/index.ts's runSweepItem/runSweepItemViaServer.
        for (const resultInput of input.results) {
          const row = buildResultRow(runId, run.model_id, idx, resultInput);
          insertResult.run(
            row.id,
            row.run_id,
            row.idx,
            row.model_id,
            row.test_type,
            row.n_prompt,
            row.n_gen,
            row.n_threads,
            row.n_gpu_layers,
            row.batch_size,
            row.ubatch_size,
            row.cache_type_k,
            row.cache_type_v,
            row.flash_attn,
            row.mtp,
            row.n_gpu_layers_draft,
            row.avg_tps,
            row.stddev_tps,
            row.ram_peak_mib,
            row.vram_peak_mib,
            row.ram_avg_mib,
            row.vram_avg_mib,
            row.ram_free_before_mib,
            row.vram_free_before_mib,
            row.system_memory_total_mb,
            row.gpu_memory_total_mb,
            row.gpu_memory_total_accuracy,
            row.gpu_memory_total_source,
            row.gpu_memory_free_start_accuracy,
            row.gpu_memory_free_start_source,
            row.gpu_memory_model_avg_accuracy,
            row.gpu_memory_model_avg_source,
            row.gpu_memory_model_peak_accuracy,
            row.gpu_memory_model_peak_source,
            row.gpu_layers_loaded,
            row.total_model_layers,
            row.gpu_layers_loaded_draft ?? null,
            row.total_model_layers_draft ?? null,
            row.sample_count ?? null,
            row.suspect_count ?? null,
            row.suspect_samples ? JSON.stringify(row.suspect_samples) : null,
            row.repeat_samples ? JSON.stringify(row.repeat_samples) : null,
            row.spec_drafted ?? null,
            row.spec_accepted ?? null,
            null,
            row.created_at
          );
        }
      }

      // Tier-2 backend_device_name upgrade (see shared/types.ts's
      // Run.backend_device_name) -- a plain UPDATE, not COALESCE: the Tier-1
      // fallback is always written first at dispatch time (markRunRunning/
      // createRun), strictly before any item can go terminal, so this write
      // is always the intended upgrade, never a downgrade.
      if (input.backend_device_name) {
        database.prepare(`UPDATE runs SET backend_device_name = ? WHERE id = ?`).run(input.backend_device_name, runId);
      }

      if (this.countUnfinishedItems(runId) === 0) {
        this.finalizeRun(runId, now);
      }
    });
    tx();

    return this.getRun(runId);
  },

  countUnfinishedItems(runId: string): number {
    const row = getDb()
      .prepare(
        `SELECT COUNT(*) as n FROM run_items
         WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','cancelled')`
      )
      .get(runId) as { n: number };
    return row.n;
  },

  // Derives the run's overall status from its items' final statuses. Only
  // meaningful once countUnfinishedItems(runId) === 0 -- callers are
  // responsible for that check (recordRunItemTerminal does it after every
  // terminal write; failAllRunItems calls it directly since it just made
  // that true itself).
  //
  // cancelReason distinguishes *why* items ended up 'cancelled': omitted
  // (the default) means a genuine per-item user stop (worker/src/index.ts's
  // stopRequested loop, reported one item at a time through the normal
  // recordRunItemTerminal path) -- "stopped by user" is accurate there.
  // reconcileStaleRun passes an explicit reason instead, since its
  // 'cancelled' items were never actually confirmed stopped by anyone --
  // the DB just lost track of a run the worker itself had already moved on
  // from (its own terminal report may have been rejected or never arrived).
  // Labelling that "stopped by user" too would be a lie the user has no way
  // to tell apart from a real stop.
  finalizeRun(runId: string, completedAt: number = Date.now(), cancelReason?: string): void {
    const counts = getDb()
      .prepare(
        `SELECT
           SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done,
           SUM(CASE WHEN status = 'failed_oom' THEN 1 ELSE 0 END) as oom,
           SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
           COUNT(*) as total
         FROM run_items WHERE run_id = ?`
      )
      .get(runId) as { done: number; oom: number; cancelled: number; total: number };

    let status: RunStatus;
    let error: string | null = null;
    if (counts.cancelled > 0) {
      // A genuine user stop always reads "cancelled" regardless of how much
      // finished first -- that's the outcome the user asked for. An
      // unconfirmed/reconciled stop (cancelReason set) is different: if some
      // combinations *did* complete and are saved, "partial" reflects that
      // there's real, usable data here rather than implying the whole run
      // is worthless.
      status = cancelReason && counts.done > 0 ? "partial" : "cancelled";
      const reason = cancelReason ?? "stopped by user";
      error = `${reason} after ${counts.done} of ${counts.total} test${counts.total === 1 ? "" : "s"} completed`;
    } else if (counts.total === 0 || counts.done === 0) {
      status = "failed";
    } else if (counts.done === counts.total) {
      status = "done";
    } else {
      status = "partial";
    }

    // Guards on counts.cancelled (not status) -- a reconciled run with some
    // completions now reads status "partial" too (see above), but its error
    // was already set from cancelReason and must not be overwritten with a
    // "tests failed" message; nothing here was a bench failure.
    if (counts.cancelled === 0) {
      const failedCount = counts.total - counts.done;
      if (failedCount > 0) {
        error = `${failedCount} of ${counts.total} test${counts.total === 1 ? "" : "s"} failed`;
        if (counts.oom > 0) error += ` (${counts.oom} OOM)`;
      }
    }

    getDb()
      .prepare(`UPDATE runs SET status = ?, error = ?, completed_at = ? WHERE id = ?`)
      .run(status, error, completedAt, runId);
  },

  // Used when the worker rejects a trigger or is unreachable -- the run's
  // items already exist (created up front, see routes/runs.ts) but nothing
  // ever ran, so mark all of them failed rather than leaving a permanently
  // "queued" list for a run that never started.
  failAllRunItems(runId: string, error: string): void {
    const database = getDb();
    const now = Date.now();
    const tx = database.transaction(() => {
      database
        .prepare(
          `UPDATE run_items SET status = 'failed', error = ?, completed_at = ?
           WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','cancelled')`
        )
        .run(error, now, runId);
      this.finalizeRun(runId, now);
    });
    tx();
  },

  // A worker only ever tracks one run at a time and has no persistent state
  // of its own (see worker/src/index.ts) -- if it restarts mid-run, or a
  // run's very last item terminal update exhausts safeItemTerminal's retries
  // without reaching us, the run's row here is left "running" forever with
  // no unfinished item ever going to arrive. Used by routes/runs.ts (GET
  // /api/runs/:id, self-healing against /health) and routes/workers.ts
  // (the /stop proxy, when the worker 409s "not running a benchmark") to
  // find the row that's stuck.
  getRunningRunForWorker(workerName: string): Run | undefined {
    const row = getDb()
      .prepare(`SELECT * FROM runs WHERE worker_name = ? AND status = 'running' ORDER BY started_at DESC LIMIT 1`)
      .get(workerName) as RunRow | undefined;
    return row ? mapRun(row) : undefined;
  },

  // True once a worker already has a run in flight or waiting its turn --
  // routes/runs.ts's trigger handler uses this to decide whether a new run
  // can dispatch immediately or has to join that worker's FIFO queue
  // instead (a worker only ever runs one benchmark at a time).
  hasActiveRunForWorker(workerName: string): boolean {
    const row = getDb()
      .prepare(`SELECT 1 FROM runs WHERE worker_name = ? AND status IN ('running','scheduled') LIMIT 1`)
      .get(workerName);
    return row !== undefined;
  },

  // Earliest-queued 'scheduled' run for a worker, if any -- pulled by
  // routes/runs.ts's dispatchScheduledRun once the run ahead of it on that
  // worker finalizes (or is reconciled away as stale).
  getNextScheduledRunForWorker(workerName: string): Run | undefined {
    const row = getDb()
      .prepare(`SELECT * FROM runs WHERE worker_name = ? AND status = 'scheduled' ORDER BY started_at ASC LIMIT 1`)
      .get(workerName) as RunRow | undefined;
    return row ? mapRun(row) : undefined;
  },

  // Flips a queued run to 'running' right before it's actually dispatched to
  // the worker. llama_cpp_build/backend and started_at are only meaningful
  // once dispatch actually begins -- the worker's active build can change
  // while a run sits queued, and "started" should mean "began executing",
  // not "was queued" (see Run.started_at's use elsewhere as "when
  // triggered", which for a queued run this now supersedes).
  markRunRunning(
    runId: string,
    patch: { llama_cpp_build: string; llama_cpp_backend: string; backend_device_name?: string; started_at: number }
  ): void {
    getDb()
      .prepare(
        `UPDATE runs SET status = 'running', llama_cpp_build = ?, llama_cpp_backend = ?, backend_device_name = ?, started_at = ? WHERE id = ?`
      )
      .run(patch.llama_cpp_build, patch.llama_cpp_backend, patch.backend_device_name ?? null, patch.started_at, runId);
  },

  // Marks a run's still-unfinished items 'cancelled' (not 'failed' --
  // nothing here indicates an actual bench failure, just that the worker
  // isn't tracking this run anymore) and finalizes it. `note` is both the
  // per-item error text and finalizeRun's cancelReason, so the run-level
  // message stays honest about this being an unconfirmed/lost run, not a
  // genuine user stop (see finalizeRun's own doc comment).
  reconcileStaleRun(runId: string, note: string): Run | undefined {
    const database = getDb();
    const now = Date.now();
    const tx = database.transaction(() => {
      database
        .prepare(
          `UPDATE run_items SET status = 'cancelled', error = ?, completed_at = ?
           WHERE run_id = ? AND status NOT IN ('done','failed','failed_oom','cancelled')`
        )
        .run(note, now, runId);
      this.finalizeRun(runId, now, note);
    });
    tx();
    return this.getRun(runId);
  },
};

// Builds the *physical-column* subset of ResultRow -- deliberately excludes
// backend_type/backend_device_name, which aren't physical columns on
// `results` at all (see ResultRow's own doc comment: they're a JOIN done in
// getResultsForRun, only known once a row is read back with its parent run,
// never at insert time here).
function buildResultRow(
  runId: string,
  modelId: string,
  idx: number,
  r: IngestResultInput
): Omit<ResultRow, "backend_type" | "backend_device_name"> {
  return {
    id: uuid(),
    run_id: runId,
    idx,
    model_id: modelId,
    test_type: r.test_type,
    n_prompt: r.n_prompt,
    n_gen: r.n_gen,
    n_threads: r.n_threads,
    n_gpu_layers: r.n_gpu_layers,
    batch_size: r.batch_size,
    ubatch_size: r.ubatch_size,
    cache_type_k: r.cache_type_k,
    cache_type_v: r.cache_type_v,
    flash_attn: r.flash_attn,
    mtp: r.mtp,
    // Defaults to 0 for a version-skewed worker that predates this field --
    // see shared/types.ts's IngestResultInput.n_gpu_layers_draft and
    // schema.sql's matching column default.
    n_gpu_layers_draft: r.n_gpu_layers_draft ?? 0,
    avg_tps: r.avg_tps,
    stddev_tps: r.stddev_tps,
    ram_peak_mib: r.ram_peak_mib,
    vram_peak_mib: r.vram_peak_mib,
    ram_avg_mib: r.ram_avg_mib,
    vram_avg_mib: r.vram_avg_mib,
    ram_free_before_mib: r.ram_free_before_mib,
    vram_free_before_mib: r.vram_free_before_mib,
    system_memory_total_mb: r.system_memory_total_mb,
    gpu_memory_total_mb: r.gpu_memory_total_mb,
    gpu_memory_total_accuracy: r.gpu_memory_total_accuracy,
    gpu_memory_total_source: r.gpu_memory_total_source,
    gpu_memory_free_start_accuracy: r.gpu_memory_free_start_accuracy,
    gpu_memory_free_start_source: r.gpu_memory_free_start_source,
    gpu_memory_model_avg_accuracy: r.gpu_memory_model_avg_accuracy,
    gpu_memory_model_avg_source: r.gpu_memory_model_avg_source,
    gpu_memory_model_peak_accuracy: r.gpu_memory_model_peak_accuracy,
    gpu_memory_model_peak_source: r.gpu_memory_model_peak_source,
    gpu_layers_loaded: r.gpu_layers_loaded,
    total_model_layers: r.total_model_layers,
    gpu_layers_loaded_draft: r.gpu_layers_loaded_draft ?? null,
    total_model_layers_draft: r.total_model_layers_draft ?? null,
    sample_count: r.sample_count,
    suspect_count: r.suspect_count,
    suspect_samples: r.suspect_samples,
    repeat_samples: r.repeat_samples,
    spec_drafted: r.spec_drafted,
    spec_accepted: r.spec_accepted,
    raw_json_path: undefined,
    created_at: Date.now(),
  };
}
