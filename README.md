# LlamaToaster — LLM Benchmark Web App (Lean MVP, v2)

Benchmark orchestrator: a **VPS** (Fastify API + a React SPA + SQLite) triggers
runs on one or more workers (typically named **Remote** for a CPU worker on the
orchestrator box itself, and **Local** for your GPU box) which run
`llama-bench` and post the averaged results back. Tailscale between them.

Architecture, schema, security model, and phasing follow
[`llm-benchmark-plan-v2.md`](./llm-benchmark-plan-v2.md).

## Layout

```
server/   VPS: Fastify API + SQLite (better-sqlite3), serves client/dist
client/   React + Vite + Tailwind SPA (the dashboard UI)
worker/   local box: HTTP /run listener, spawns llama-bench, samples RAM/VRAM
shared/   TypeScript types shared by server, worker, and client
```

## Prerequisites

- Node.js 22+ (required by `better-sqlite3` v13, and by Vite 8 for the client build)
- A built `llama-bench` (and `llama.cpp`) on the worker box
- Both machines on the same Tailscale network

Install once (root deps, then the client's own deps), then build the SPA —
wherever the **server** runs needs this; a worker-only box doesn't:

```bash
npm install
npm run install:client
npm run build
```

Re-run `npm run build` after pulling changes that touch `client/` — it isn't
run automatically on `npm run server` startup.

## Running the VPS (orchestrator)

```bash
PORT=4010 BIND_HOST=<vps-tailscale-ip> \
DEFAULT_WORKER_URL=http://<worker-tailscale-ip>:8080 \
npm run server
```

For frontend development with hot reload instead of rebuilding on every
change, run the Vite dev server alongside `npm run server` in a second
terminal — it proxies `/api` to whatever `PORT` the server above is using
(default `3000`; override with `VITE_API_PROXY_TARGET` if you set a
different `PORT`):

```bash
npm run dev:client
```

- `BIND_HOST` — **bind to the VPS's own Tailscale IP, not `0.0.0.0`** (see Security).
- `DB_PATH` — optional SQLite file location (default `./llamatoaster.db`).
- `DEFAULT_WORKER_URL` / `DEFAULT_WORKER_NAME` / `DEFAULT_WORKER_BACKEND` / `DEFAULT_WORKER_BUILD` — fallback worker.
- `WORKERS` — optional JSON array of worker defs `[{name, backend, llama_cpp_build, url}]` for multi-worker labels.
- `LOG_LEVEL` — optional, `debug` | `info` (default) | `warn` | `error`. `debug` adds full sweep/trigger detail to the timestamped request log.
- `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` — optional, enable the dashboard's AI Assistant panel (parameter/model suggestions, informed by your hardware and benchmark results). Unlike the vars above, these are read from a `.env` file at the repo root (copy [`.env.example`](.env.example)) rather than the shell/systemd — it's gitignored and loaded automatically on startup. Without it the panel just reports "not configured."

## Running the worker (GPU box)

**Brand-new machine, nothing downloaded yet?** One command fetches the repo,
installs dependencies, and starts the worker (needs PowerShell/bash +
Node.js 22+; uses `git` if present, otherwise falls back to a zip/tarball
download). It asks which drive/volume to use — showing free space, since
models are often tens of GB each — and a folder name, then creates it:

```powershell
# Windows
iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) }"
```

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
```

Pass a folder explicitly to skip the prompts (e.g. for unattended/scripted
use): `-Dir F:\LlamaToaster` / `-- --dir ~/LlamaToaster`.

It's safe to re-run: if the folder already has a checkout, the download step
is skipped, and [`worker/setup-worker.ps1`](worker/setup-worker.ps1) /
[`.sh`](worker/setup-worker.sh) underneath only write `config.json` once —
after that, the exact same command just (re)starts the worker (no prompt,
since there's nothing left to configure). Pass `-Force` / `--force` to redo
config generation (e.g. to pick a different drive or bind IP), and see
either script's header comment for the full set of overrides
(`-WorkerName`, `-Backend`, `-VpsUrl`, `-BindHost`, `-Port`).

Already have the repo checked out? Skip straight to
`.\worker\setup-worker.ps1` (or the `.sh` equivalent) — same idempotent
behavior and folder prompt, no download step.

To configure by hand instead, edit `worker/config.json`:

```json
{
  "worker_name": "Local",
  "backend": "vulkan",
  "llama_cpp_build": "b10068",
  "llama_bench_path": "C:\\llama\\llama-bench.exe",
  "model_dir": "C:\\llm",
  "vps_url": "http://<vps-tailscale-ip>:4010",
  "bind_host": "<worker-tailscale-ip>",
  "port": 8080,
  "raw_json_dir": "C:\\llm\\raw",
  "llama_cpp_builds_dir": "C:\\llm\\llama-builds"
}
```

`llama_cpp_builds_dir` is optional (defaults to a `llama-builds/` folder next
to `worker/`) — it's where builds installed from the Workers page land, one
subfolder per tag. `llama_bench_path`/`llama_cpp_build` are the *initial*
active build; switching versions from the Workers page rewrites both fields
in this file, so a restart doesn't revert the choice. If `llama_bench_path`
doesn't exist at startup the worker still starts (logs a warning) so a brand
new worker can have its first build installed entirely from the web UI.

Every completed test also dumps its raw llama-bench/llama-server
stdout/stderr under `raw_json_dir` (`<ok|failed>/<run_id>-<idx>.json`) —
purely a manual-debugging aid, nothing in the app reads these back. A daily
janitor prunes them automatically: successful tests' dumps after
`raw_json_retention_days` (optional, default `14`); failed tests' dumps are
kept forever unless you set `raw_json_retention_days_failed` too, since a
failure's raw output is the main reason these files are worth keeping at all.

`log_dir` is optional (defaults to a `logs/` folder next to `worker/`) — every
run/download/build-management event is timestamped and written both to the
console and to a daily file there (`worker-YYYY-MM-DD.log`), so a worker
started hidden (`start-hidden.vbs`) or as a background service still leaves a
trail to inspect after the fact. Set `LOG_LEVEL=debug` (env var) for full
sweep/command-line detail on top of the default lifecycle logging.

Then:

```bash
npm run worker
```

`bind_host` is this machine's Tailscale IP (`tailscale ip -4`) — the listener
binds to it directly, never `0.0.0.0`.

## Deploying a CPU worker on the VPS itself

Same worker process as above, just declared `"backend": "cpu"` and running
on the same box as the orchestrator instead of a separate GPU machine.
Thanks to the Workers page's version manager, **you don't need to manually
build or download llama-bench first** — install it from the browser after
the worker is up.

1. On the VPS: clone this repo to e.g. `/home/ubuntu/LlamaToaster`, run
   `npm install`, `npm run install:client`, and `npm run build` (builds the
   SPA into `client/dist`, which the server serves directly — the worker
   process itself needs none of this), and `mkdir -p
   /home/ubuntu/LlamaToaster/data /home/ubuntu/LlamaToaster/models`. The
   provided systemd units run the
   services as your regular login user (`User=ubuntu`/`Group=ubuntu`) rather
   than a dedicated system account — simpler for a single-user VPS, at the
   cost of the app running with the same privileges as your login user
   instead of an isolated one. Adjust `User=`/`Group=`/paths in both
   `deploy/*.service` files if your username or clone path differs.
2. Copy [`worker/config.vps.json.example`](worker/config.vps.json.example)
   to `worker/config.vps.json` and copy
   [`deploy/orchestrator.env.example`](deploy/orchestrator.env.example) to
   `deploy/orchestrator.env`. Fill in the VPS's real Tailscale IP in both —
   **`orchestrator.env`'s `BIND_HOST` and `worker/config.vps.json`'s
   `vps_url` must be the same address**: the worker posts results back to
   whatever the orchestrator actually listens on, and if the orchestrator
   binds only to the Tailscale IP (needed so you can reach the dashboard
   from your own machine), `vps_url: http://127.0.0.1:...` will not reach
   it even though they're on the same box — loopback traffic doesn't land
   on a socket that's bound to a different specific IP. The worker's own
   `bind_host` can stay `127.0.0.1`, though: nothing but this same-host
   orchestrator needs to reach it directly.
3. Install the two systemd units:
   ```bash
   sudo cp deploy/llamatoaster-worker.service deploy/llamatoaster-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now llamatoaster-server llamatoaster-worker
   ```
4. Firewall, same rule as the GPU worker (see "Security / Networking" below)
   — scope port 4010 (and 8080 if you chose the Tailscale-IP option for the
   worker's own `bind_host`) to the tailscale interface, never a bare `allow`.
5. Open the dashboard from your own machine (`http://<vps-tailscale-ip>:4010`),
   go to **Workers**, confirm the `Remote` card loads, and install a build
   from the dropdown — this is the step that replaces manually building
   llama.cpp on the VPS.
6. Register a small model (search-and-download panel on **Models**, or place
   a `.gguf` in `/home/ubuntu/LlamaToaster/models` by hand) and trigger a small run
   against `Remote` from **New Run** before doing a full sweep — the VPS
   likely has less RAM than your GPU box, and the dashboard may lag while a
   benchmark is pegging every CPU core on the same machine that's serving it.

## Using it

1. **Models** (`/models`) — register a model (local path or Hugging Face repo/file),
   or use the "Search & download (Hugging Face)" panel: search by model name, expand a
   repo to see its quant files, pick a worker, download straight to its `model_dir`.
2. **New Run** (`/new-run`) — pick a model + worker, set the sweep with graphical
   controls (chip inputs for numeric fields, toggles for cache types/flash attention,
   a repeats stepper), or open "Advanced: raw JSON" to edit/paste the equivalent JSON
   directly, then trigger.
3. **Runs** (`/runs`) — list; polls every 5s while anything is `running`.
4. **Run Detail** (`/runs/:id`) — results table + Chart.js bar chart + raw JSON download.
5. **Compare** (`/compare`) — pick 2+ runs for side-by-side bars.
6. **Workers** (`/workers`) — one card per worker: online/**Inaccessible** status,
   platform/backend, detected CPU/GPU (best-effort, informational), and two separate
   lists — **Downloaded** (installed builds, active one marked, with
   Activate/Delete) and **Available to install** (from GitHub, checked every few
   minutes). Every action is a manual click — nothing downloads, switches, or
   deletes on its own.
7. Export: `GET /api/results/export?format=json|csv|md[&runs=id1,id2]`.
8. **AI Assistant** — collapsible panel on the right of every page (collapsed by
   default). Chats with whatever OpenAI-compatible provider you configure via
   `AI_API_KEY`/`AI_BASE_URL`/`AI_MODEL` (see above), with your hardware, registered
   models, and recent benchmark results automatically included as context.

### Sweep config

A sweep (comma-separated values per field) expands to the full combination
matrix (`shared/sweep.ts`'s `expandSweep`), the same way a single CSV-args
`llama-bench` call would:

```
n_prompt=512,2048  n_gen=128  threads=4,6  n_gpu_layers=0,999
batch_size=512  ubatch_size=256,512  cache_type_k/v=f16,q8_0  flash_attn=on,off
```

Unlike passing all of that to `llama-bench` as one CSV-args call, each
combination now runs as its **own** `llama-bench` process (`-r 5` repetitions
still average within that one process, same as before) -- one crashing
combination doesn't cost the rest of the sweep, and the worker reports each
combination's live status individually as it runs (see the API table below).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs/trigger` | VPS calls the worker directly (no polling) |
| POST | `/api/runs/:id/items/:idx` | Worker posts live progress/results per sweep test as it runs |
| GET | `/api/runs` | list runs |
| GET | `/api/runs/:id` | run detail + results |
| GET | `/api/models` | list models |
| POST | `/api/models` | register a model |
| GET | `/api/workers` | list configured workers |
| GET | `/api/workers/:name/llama-cpp` | installed + installable builds for a worker |
| POST | `/api/workers/:name/llama-cpp/install` | download+extract a build (manual trigger) |
| POST | `/api/workers/:name/llama-cpp/activate` | switch the worker's active build |
| DELETE | `/api/workers/:name/llama-cpp/:tag` | remove an installed build (not the active one) |
| GET | `/api/hf/search?q=` | search Hugging Face repos tagged gguf |
| GET | `/api/hf/repo/:repo` | list `.gguf` files + sizes in a repo |
| POST | `/api/workers/:name/models/download` | start a download on a worker (returns as soon as it's accepted, not once it finishes) |
| GET | `/api/workers/:name/models/download/progress` | poll an in-flight download's byte progress |
| POST | `/api/models/download-callback` | worker posts a download's outcome once it finishes; registers the model |
| GET | `/api/results/export` | json \| csv \| md |

## Security / Networking

No app-level secret. Protection is at the network layer:

1. **Bind each listener to its own Tailscale IP**, not `0.0.0.0`.
2. **Firewall scoped to the tailscale interface** (backstop):
   - VPS: `sudo ufw allow in on tailscale0 to any port 4010 proto tcp` (never a bare `allow 4010`).
   - Worker (Windows Firewall): scope the inbound rule's remote IP to the VPS Tailscale IP or `100.64.0.0/10`.
3. Check for a provider-level network firewall that can override `ufw`.
4. No nginx/TLS — Tailscale is already WireGuard-encrypted.

Verify from outside the tailnet (e.g. phone on cellular):

```bash
curl http://<vps-public-ip>:4010/api/runs   # should hang or refuse
```

## Known limitations

- VRAM peak on **Windows** workers is read from the OS's own "GPU Process
  Memory" performance counter (per-process, vendor-neutral — works the same
  for AMD/Intel/NVIDIA under any backend). On **Linux/macOS** it still falls
  back to `systeminformation.graphics()`, which only gets real numbers from
  `nvidia-smi` — non-NVIDIA GPUs there report VRAM as `n/a`. RAM peak is
  always the trustworthy number regardless of platform.
