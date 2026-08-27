-- SQLite schema for the LLM Benchmark Web App (Lean MVP)

CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,          -- sha256 of the gguf file
  filename TEXT,
  size_bytes INTEGER,
  source TEXT,                  -- 'local' | 'huggingface'
  hf_repo TEXT,
  hf_file TEXT,
  metadata TEXT,                 -- JSON: arch, quant, trained ctx, etc
  created_at INTEGER
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  -- §0.5 -- denormalized at creation, immutable; points at the run itself
  -- for standalone runs. Chain-scoped reads are indexed equality predicates.
  root_run_id TEXT REFERENCES runs(id),
  -- §0.5 -- 'tuning'|'refine'|'sweep'|'runtime'|'probe'|'quality'; NULL = standalone
  kind TEXT,
  -- N3 -- groups model-vs-model comparison members; NULL otherwise
  comparison_id TEXT,
  worker_name TEXT,              -- just a label, e.g. 'Local' — no registration needed
  llama_cpp_build TEXT,          -- pinned build tag, e.g. 'b10068'
  llama_cpp_backend TEXT,        -- 'vulkan' | 'cpu' | 'cuda'
  backend_device_name TEXT,      -- e.g. 'AMD Radeon RX 6600 XT' -- see shared/types.ts's Run.backend_device_name
  model_id TEXT,
  config TEXT,                   -- JSON: the exact sweep passed to llama-bench
  status TEXT,                   -- 'running' | 'done' | 'failed'
  error TEXT,
  started_at INTEGER,
  completed_at INTEGER
);

CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES runs(id),
  idx INTEGER,                    -- matches the run_items row this came from, for a client-side join
  model_id TEXT REFERENCES models(id),
  test_type TEXT,                -- 'pp' | 'tg' | 'pg'
  n_prompt INTEGER,
  n_gen INTEGER,
  -- §0.2 -- KV-prefill depth; always 0 on server-engine rows (depth rule)
  n_depth INTEGER DEFAULT 0,
  -- §0.1 -- worker-stamped methodology version; NULL = pre-v8-core
  method_version INTEGER,
  -- §0.8 twin-join columns (config_hash excludes spec fields, which is what
  -- makes the join work); speedup_status ∈ ok|unverified|unavailable
  config_hash TEXT,
  prompt_offset INTEGER,
  spec_type TEXT,
  spec_n_max INTEGER,
  spec_n_min INTEGER,
  speedup REAL,
  speedup_status TEXT,
  -- §0.10 -- JSON string[] of caveat flags from the closed registry
  caveat_flags TEXT,
  -- N5 -- concurrent streaming slots this row measured (1 = solo)
  concurrency INTEGER,
  -- M6 -- adapter peak °C, lowest sampled sclk, and the sample series
  -- (JSON number[], ~6s cadence; prunable 30 days after ingest)
  gpu_temp_c_max INTEGER,
  gpu_clock_mhz_min INTEGER,
  gpu_clock_samples TEXT,
  -- N6 -- llama.cpp startup-banner ISA provenance (CPU-bound rows)
  cpu_isa TEXT,
  -- N1/Test B -- MEASURED time-to-first-token from the streamed response
  -- (llama-server path only; the llama-bench path has no request lifecycle
  -- and leaves these NULL, which the UI renders as a derived estimate
  -- instead). ttft_n is the sample count behind p50/p95: a choreographed
  -- curve point is single-shot by construction (ttft_n = 1) and is labeled
  -- as such -- no p50/p95 pretense on cold columns.
  -- N7 -- imported rows are BADGED and never merge into local profile
  -- scoring unless opted in per import. NULL bundle id = a local measurement.
  imported_bundle_id TEXT,
  import_opt_in INTEGER DEFAULT 0,
  ttft_ms_p50 REAL,
  ttft_ms_p95 REAL,
  ttft_n INTEGER,
  e2e_ms_mean REAL,
  -- Denormalized from the parent run so idx_results_curve can cover the
  -- curve grouping keys exactly (N1's read path stays an index range scan)
  worker_id TEXT REFERENCES workers(id) ON DELETE SET NULL,
  llama_cpp_build TEXT,
  engine TEXT,                   -- 'bench' | 'server' (§0.2)
  n_threads INTEGER,
  n_gpu_layers INTEGER,
  batch_size INTEGER,
  ubatch_size INTEGER,
  cache_type_k TEXT,
  cache_type_v TEXT,
  flash_attn TEXT,                -- store as string ('on'/'off'/'auto')
  mtp TEXT DEFAULT 'off',         -- 'on'/'off' -- see worker/src/serverBench.ts
  n_gpu_layers_draft INTEGER DEFAULT 0,  -- -ngld for the MTP/draft companion model, 0 where not applicable
  n_cpu_moe INTEGER DEFAULT 0,    -- --n-cpu-moe, 0 where not applicable -- see shared/sweep.ts's SweepItem.n_cpu_moe
  avg_tps REAL,                   -- from llama-bench's own averaged output
  stddev_tps REAL,                -- from llama-bench's own stddev output
  ram_peak_mib INTEGER,           -- from external sampling — llama-bench doesn't report memory
  vram_peak_mib INTEGER,          -- best-effort
  ram_avg_mib INTEGER,            -- average of the same sampling, not just the peak
  vram_avg_mib INTEGER,           -- best-effort
  ram_free_before_mib INTEGER,    -- free host RAM sampled right before the process spawned
  vram_free_before_mib INTEGER,   -- best-effort
  system_memory_total_mib INTEGER,   -- total system RAM (Apple Silicon: total unified memory)
  gpu_memory_total_mib INTEGER,      -- total VRAM capacity (Metal: total unified memory) -- see worker/src/vram.ts
  -- Accuracy/provenance metadata for the four GPU-memory figures above/below
  -- (gpu_memory_total_mib and the existing vram_free_before_mib/vram_avg_mib/
  -- vram_peak_mib) -- see shared/types.ts's GpuMemoryAccuracyLevel/
  -- GpuMemoryMeasurementSource for the enum values. Null iff the paired
  -- value is null.
  gpu_memory_total_accuracy TEXT,
  gpu_memory_total_source TEXT,
  gpu_memory_free_start_accuracy TEXT,   -- pairs with vram_free_before_mib above
  gpu_memory_free_start_source TEXT,
  gpu_memory_model_avg_accuracy TEXT,    -- pairs with vram_avg_mib above
  gpu_memory_model_avg_source TEXT,
  gpu_memory_model_peak_accuracy TEXT,   -- pairs with vram_peak_mib above
  gpu_memory_model_peak_source TEXT,
  -- Whole-adapter and per-process VRAM usage avg/peak, plus whole-system RAM
  -- usage avg/peak -- see shared/types.ts's ResultRow doc comments and
  -- worker/src/vram.ts's per-backend readers for what populates these.
  -- vram_avg_mib/vram_peak_mib above remain the best-available hybrid
  -- (process when readable, else whole adapter). NULL on every row inserted
  -- before this migration, which correctly reads as "unavailable".
  gpu_memory_used_avg_mib INTEGER,
  gpu_memory_used_peak_mib INTEGER,
  gpu_memory_used_avg_accuracy TEXT,
  gpu_memory_used_avg_source TEXT,
  gpu_memory_used_peak_accuracy TEXT,
  gpu_memory_used_peak_source TEXT,
  gpu_memory_process_avg_mib INTEGER,
  gpu_memory_process_peak_mib INTEGER,
  gpu_memory_process_avg_accuracy TEXT,
  gpu_memory_process_avg_source TEXT,
  gpu_memory_process_peak_accuracy TEXT,
  gpu_memory_process_peak_source TEXT,
  ram_total_used_avg_mib INTEGER,
  ram_total_used_peak_mib INTEGER,
  -- Read from llama.cpp's own runtime output, never inferred -- see
  -- worker/src/index.ts's parseOffloadLayers. n_gpu_layers above is already
  -- the *requested* value; these are what actually happened. Always the
  -- *base* model's figures, even on an MTP row -- see gpu_layers_loaded_draft/
  -- total_model_layers_draft below for the draft companion's own.
  gpu_layers_loaded INTEGER,
  total_model_layers INTEGER,
  -- The MTP/draft companion model's own actual offload -- see
  -- shared/types.ts's ResultRow doc comment. NULL on every non-MTP row and
  -- on an MTP row where the draft model's own runtime line wasn't captured.
  gpu_layers_loaded_draft INTEGER,
  total_model_layers_draft INTEGER,
  raw_json_path TEXT,             -- full llama-bench JSON output for this run
  created_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model_id);

-- One row per sweep combination (one llama-bench process each -- see
-- shared/sweep.ts's expandSweep), pre-created as 'queued' when a run is
-- triggered so the UI can show the full planned list and "N total tests"
-- immediately, then updated live as the worker works through them. A
-- 'failed'/'failed_oom' item does not stop the rest of the run.
CREATE TABLE IF NOT EXISTS run_items (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES runs(id),
  idx INTEGER,
  n_prompt INTEGER,
  n_gen INTEGER,
  n_depth INTEGER DEFAULT 0, -- KV-prefill depth, llama-bench only (§0.2)
  concurrency INTEGER DEFAULT 1, -- N5 concurrent streaming slots
  n_threads INTEGER,
  n_gpu_layers INTEGER,
  batch_size INTEGER,
  ubatch_size INTEGER,
  cache_type_k TEXT,
  cache_type_v TEXT,
  flash_attn TEXT,
  mtp TEXT DEFAULT 'off',    -- 'on'/'off' -- see worker/src/serverBench.ts
  n_gpu_layers_draft INTEGER DEFAULT 0,  -- -ngld for the MTP/draft companion model, 0 where not applicable
  n_cpu_moe INTEGER DEFAULT 0,    -- --n-cpu-moe, 0 where not applicable -- see shared/sweep.ts's SweepItem.n_cpu_moe
  status TEXT,              -- queued|loading|processing|generating|benchmarking|done|failed|failed_oom
  detail TEXT,               -- best-effort human text, e.g. a stderr progress line
  ram_mib INTEGER,           -- live/current, updated by best-effort ticks
  vram_mib INTEGER,
  ram_peak_mib INTEGER,      -- final, set once terminal
  vram_peak_mib INTEGER,
  ram_avg_mib INTEGER,       -- final, set once terminal
  vram_avg_mib INTEGER,
  ram_free_before_mib INTEGER,   -- baseline, set once from the item's first tick
  vram_free_before_mib INTEGER,
  live_tps REAL,             -- measured throughput of the most recently completed repeat
  error TEXT,
  started_at INTEGER,
  completed_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_run_items_run ON run_items(run_id);

-- Multi-user Stage 1: pull queue -- see MULTIUSER_PLAN.md §1.2.

-- One row per machine. In Stage 1 rows are created by the worker's first
-- heartbeat (self-announce, keyed on the machine_id it persists locally).
-- Stage 3 adds enrolment columns; Stage 4 adds user_id.
CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,                       -- UUID, server-assigned
  machine_id TEXT UNIQUE NOT NULL,           -- stable id the worker persists in its config
  display_name TEXT NOT NULL,                -- user-editable; defaults to the reported hostname
  backend TEXT,                              -- from state push
  platform TEXT,                             -- from state push
  arch TEXT,                                 -- from state push
  hostname TEXT,                             -- from state push, the default display_name
  hardware_json TEXT,
  installed_builds_json TEXT,
  model_files_json TEXT,
  last_heartbeat_at INTEGER,                 -- liveness source of truth; status is DERIVED (see server/src/liveness.ts)
  active_job_id TEXT,                        -- what it is executing right now, or NULL
  -- §0.7/N2/N4/N1 capability advertisement (probe-v1, quality-v1, curve-v1...)
  capabilities_json TEXT,
  -- M6 sensor availability (clock/temp per backend) declared up front
  sensors_json TEXT,
  -- N6 llama.cpp startup-banner ISA provenance
  cpu_isa TEXT,
  pause_requested INTEGER NOT NULL DEFAULT 0, -- see MULTIUSER_PLAN.md §1.14 -- POST /api/runs/:id/pause|resume
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

-- Persistent queue. Leases and attempts are what make a crashed worker
-- recoverable.
CREATE TABLE IF NOT EXISTS worker_jobs (
  id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL,                    -- 'benchmark' | 'install_build' | 'download_model' | 'delete_model_file'
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',    -- pending | claimed | completed | failed | cancelled
  attempts INTEGER NOT NULL DEFAULT 0,
  lease_expires_at INTEGER,                  -- set on claim, extended by every heartbeat
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  progress_json TEXT,                        -- live phase/progress, pushed by heartbeat
  run_id TEXT,                               -- set for benchmark jobs, for reconciliation
  created_at INTEGER NOT NULL,
  claimed_at INTEGER,
  completed_at INTEGER,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_worker_jobs_worker  ON worker_jobs(worker_id, status);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_lease   ON worker_jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_workers_machine     ON workers(machine_id);

-- Multi-user Stage 2: auth -- see MULTIUSER_PLAN.md §2.1/§2.2.

-- THE account. Provider-independent -- every other table (sessions, workers
-- once Stage 4 adds user_id, ...) references users.id, never a provider's own
-- id directly. No is_superadmin column here -- see §5.1, evaluated per
-- request from env + linked identities, never a stored/cacheable flag.
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  display_name TEXT,
  avatar_url TEXT,
  share_benchmarks INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_login_at INTEGER
);

-- One row per (provider, that provider's account) a user has linked. GitHub
-- today; a second provider is a new row shape, not a new column, and not a
-- migration of anything that already exists.
CREATE TABLE IF NOT EXISTS identities (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  provider_user_id TEXT NOT NULL,
  provider_login TEXT,
  created_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_identities_provider ON identities(provider, provider_user_id);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);

-- The token itself is never stored, only its hash -- see server/src/session.ts's
-- hashToken and repo.sessionRepo.create.
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL,
  refresh_hash TEXT,
  prev_refresh_hash TEXT,
  expires_at INTEGER NOT NULL,
  is_worker INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT,
  label TEXT,
  last_seen_at INTEGER,
  created_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_token   ON sessions(token_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_refresh ON sessions(refresh_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_worker  ON sessions(worker_id);

-- Per-user/per-day/per-hour AI usage counters -- see MULTIUSER_PLAN.md §2.6.
-- One row per (user_id, day, hour); incremented on every /api/ai/chat call
-- that actually reaches the provider, checked before it.
CREATE TABLE IF NOT EXISTS ai_usage (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  day TEXT NOT NULL,     -- 'YYYY-MM-DD', UTC
  hour INTEGER NOT NULL, -- 0-23, UTC
  count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, day, hour)
);
CREATE INDEX IF NOT EXISTS idx_ai_usage_day ON ai_usage(day);

-- Multi-user Stage 4 (MULTIUSER_PLAN.md §4.2): tiny key-value store for
-- one-shot migration flags -- currently just "has the pre-auth legacy
-- history been claimed by the first superadmin yet" (see migrate.ts's
-- claimLegacyHistory, run from routes/auth.ts on that user's login, not at
-- boot -- the user row has to exist first). Not meant for anything else.
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

-- Hugging Face GGUF index -- see server/src/hf-index.ts for the full
-- indexing flow. One row per (sha256, repo_id, filename, revision)
-- combination. Composite PRIMARY KEY (sha256, repo_id, filename) so the same
-- content hash mirrored across repos (or renamed files) keeps distinct rows
-- rather than collapsing via sha256-only conflict. last_seen tracks when this
-- row was last confirmed present on HF (for incremental refresh -- repos
-- whose last_seen is older than the refresh interval get re-checked).
CREATE TABLE IF NOT EXISTS hf_gguf_index (
  sha256 TEXT NOT NULL,
  repo_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  revision TEXT,
  file_size INTEGER,
  last_seen INTEGER NOT NULL,
  deleted_at INTEGER,
  PRIMARY KEY (sha256, repo_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_hf_gguf_index_sha256 ON hf_gguf_index(sha256);
CREATE INDEX IF NOT EXISTS idx_hf_gguf_index_repo ON hf_gguf_index(repo_id);
CREATE INDEX IF NOT EXISTS idx_hf_gguf_index_last_seen ON hf_gguf_index(last_seen);
-- idx_hf_gguf_index_deleted_at is created in migrate.ts's
-- createHfGgufIndexDeletedAtIndex, not here -- deleted_at is added via
-- COLUMN_MIGRATIONS' ALTER TABLE on an already-existing DB (this
-- CREATE TABLE IF NOT EXISTS is a no-op there), so an index on it in this
-- file would run before the column exists and fail on any pre-existing
-- database.

-- idx_results_item (UNIQUE on results(run_id, idx, test_type), guarding
-- against a retried terminal item report inserting a second results row) is
-- deliberately NOT created here. results.idx is only added as a column by
-- migrate.ts's COLUMN_MIGRATIONS + backfillResultIdx, which run AFTER this
-- file is exec'd -- creating the index here would fail with "no such column"
-- on any DB whose results table predates that column. See migrate.ts's
-- createResultsItemUniqueIndex, called once idx is guaranteed to exist and
-- be backfilled.

-- Worker-local persistent cache of model file hashes and their HF
-- mappings. See worker/src/local-cache.ts for the full scan/validate/
-- hash/resolve flow. path is the PRIMARY KEY because it's worker-local:
-- each worker maintains its own cache for its own model_dir. sha256 and
-- hf_model_id are optional: a file that hasn't been hashed yet has neither,
-- and a hashed file with no HF match has sha256 but no hf_model_id.
-- last_verified tracks when this row was last confirmed valid (for
-- incremental re-validation -- files whose mtime changed or whose
-- last_verified is older than the recheck interval get re-hashed).
-- state is the ModelState enum (detected/hashing/verified/etc) - see
-- shared/types.ts. Present in worker DB, but also defined here for
-- tooling/inspection of a worker DB copied to server side.
CREATE TABLE IF NOT EXISTS local_model_cache (
  path TEXT PRIMARY KEY,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  sha256 TEXT,
  hf_model_id TEXT,
  last_verified INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'detected'
);

CREATE INDEX IF NOT EXISTS idx_local_model_cache_sha256 ON local_model_cache(sha256);
CREATE INDEX IF NOT EXISTS idx_local_model_cache_hf ON local_model_cache(hf_model_id);
CREATE INDEX IF NOT EXISTS idx_local_model_cache_state ON local_model_cache(state);

-- N2 -- largest-usable-config verification (estimate → verify). A re-probe
-- replaces the stale row through the same UNIQUE key. worker_id is derived
-- from the run named in the ingestion route's URL, never from the payload.
CREATE TABLE IF NOT EXISTS model_machine_limits (
  id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  model_id  TEXT NOT NULL REFERENCES models(id)  ON DELETE CASCADE,
  llama_cpp_build TEXT NOT NULL,
  kv_type   TEXT NOT NULL,            -- the K/V pair probed
  placement_hash TEXT NOT NULL,       -- hash of the placement the estimate assumed (ngl / n-cpu-moe / slots)
  verified_ctx_tokens INTEGER NOT NULL,
  margin_observed_frac REAL,          -- headroom left at the verified point
  method_version INTEGER,
  created_at INTEGER NOT NULL,
  UNIQUE(worker_id, model_id, llama_cpp_build, kv_type, placement_hash)
);
CREATE INDEX IF NOT EXISTS idx_model_machine_limits_model ON model_machine_limits(model_id);

-- N4 -- perplexity/KLD quality measurements (labeled synthetic proxy; never
-- feeds ranking until the calibration gate passes). root_run_id is NOT NULL:
-- quality work is run-scoped like everything else. Worker retries replace
-- through the six-column UNIQUE key instead of accumulating duplicates.
CREATE TABLE IF NOT EXISTS quality_results (
  id TEXT PRIMARY KEY,
  root_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  llama_cpp_build TEXT NOT NULL,
  ctx_tokens INTEGER NOT NULL,
  cache_type_k TEXT NOT NULL, cache_type_v TEXT NOT NULL,
  ppl REAL, kld_vs_baseline REAL,
  dataset_hash TEXT NOT NULL,
  method_version INTEGER,
  created_at INTEGER NOT NULL,
  UNIQUE(root_run_id, model_id, ctx_tokens, cache_type_k, cache_type_v, dataset_hash)
);
CREATE INDEX IF NOT EXISTS idx_quality_results_model ON quality_results(model_id);
