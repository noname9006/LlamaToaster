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
  worker_name TEXT,              -- just a label, e.g. 'Local' — no registration needed
  llama_cpp_build TEXT,          -- pinned build tag, e.g. 'b10068'
  llama_cpp_backend TEXT,        -- 'vulkan' | 'cpu' | 'cuda'
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
  n_threads INTEGER,
  n_gpu_layers INTEGER,
  batch_size INTEGER,
  ubatch_size INTEGER,
  cache_type_k TEXT,
  cache_type_v TEXT,
  flash_attn TEXT,                -- store as string ('on'/'off'/'auto')
  mtp TEXT DEFAULT 'off',         -- 'on'/'off' -- see worker/src/serverBench.ts
  avg_tps REAL,                   -- from llama-bench's own averaged output
  stddev_tps REAL,                -- from llama-bench's own stddev output
  ram_peak_mib INTEGER,           -- from external sampling — llama-bench doesn't report memory
  vram_peak_mib INTEGER,          -- best-effort
  ram_avg_mib INTEGER,            -- average of the same sampling, not just the peak
  vram_avg_mib INTEGER,           -- best-effort
  ram_free_before_mib INTEGER,    -- free host RAM sampled right before the process spawned
  vram_free_before_mib INTEGER,   -- best-effort
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
  n_threads INTEGER,
  n_gpu_layers INTEGER,
  batch_size INTEGER,
  ubatch_size INTEGER,
  cache_type_k TEXT,
  cache_type_v TEXT,
  flash_attn TEXT,
  mtp TEXT DEFAULT 'off',    -- 'on'/'off' -- see worker/src/serverBench.ts
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
