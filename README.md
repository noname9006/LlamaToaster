<p align="center">
  <img src="assets/logo.png" alt="LlamaToaster" width="320">
</p>

<p align="center">
  <a href="https://llamatoaster.com"><img src="https://img.shields.io/badge/try%20it-llamatoaster.com-f5a623" alt="Try it at llamatoaster.com"></a>
  <a href="https://llamatoaster.com"><img src="https://img.shields.io/website?url=https%3A%2F%2Fllamatoaster.com%2Fhealth&up_message=online&down_message=offline&label=status" alt="Hosted instance status"></a>
  <img src="https://img.shields.io/github/last-commit/noname9006/LlamaToaster" alt="Last commit">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/node-%3E%3D22-brightgreen" alt="Node >= 22">
</p>

**Find the llama.cpp settings your own hardware actually runs best, then
compare them against everyone else's.** Point LlamaToaster at a model and a
machine, and it sweeps the parameter space with `llama-bench` /
`llama-server`, records every measurement, and hands you the configuration to
put in your real inference command line.

👉 **[llamatoaster.com](https://llamatoaster.com)** — the hosted instance.
Sign in with GitHub, connect a machine, start testing.

## What you get

- **Optimal configs for your box, not someone's blog post.** Sweep `-ngl`,
  context, batch sizes, `--n-cpu-moe`, speculative decoding and more, on
  *your* GPU, with *your* build of llama.cpp. What comes out is the setting
  you paste into your own `llama-server`/`llama-cli` run.
- **An AI assistant that reads the numbers so you don't have to.** Instead of
  staring at a results table, ask what's actually limiting you, which
  parameter moved the needle, and what to try next. It reads your own tests
  directly.
- **A GGUF index keyed by content hash.** Around 4 million GGUF files from
  Hugging Face, indexed by sha256 — so a model file is identified by what it
  *is*, not by whatever it was renamed to on disk, and the same file mirrored
  across repos doesn't turn into three different entries.
- **A shared results base that gets better as it grows.** As more machines
  report in, the assistant can answer from the accumulated data rather than
  from your tests alone — eventually predicting good settings for hardware
  it has seen before, without you re-running the whole sweep yourself.

That last point is the reason to use the hosted instance rather than a
private copy: the value compounds across contributors. One person's sweep on
an RTX 4090 is a data point; a few thousand of them is a map of local LLM
inference.

## Two ways to run it

**Use the hosted instance** — nothing to deploy. Sign in at
[llamatoaster.com](https://llamatoaster.com), then run one command on the
machine you want to benchmark (Windows PowerShell shown; macOS/Linux and the
full flow are under
[Running a worker](#running-a-worker-gpu-or-cpu-box)):

```powershell
iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) } -Url https://llamatoaster.com"
```

The worker prints a short code, you approve it in the browser, and it starts
taking jobs. It only ever makes outbound HTTPS calls — nothing to
port-forward, nothing listening on your network.

**Self-host the whole thing** — server, workers and all, on your own
infrastructure. Everything below covers this. It works completely
standalone (including with no login at all), but a private instance is an
island: it never sees, and never contributes to, the shared results base.

## What is collected, and what isn't

If you use llamatoaster.com, this is what leaves your machine:

- **Hardware and OS**: CPU model/flags/core count, GPU vendor/model/VRAM,
  total RAM, OS platform and architecture, NVIDIA driver and CUDA version,
  and the machine's hostname (it becomes the default display name — rename it
  in the UI any time).
- **Benchmark results**: the configurations tested and what they measured.
- **Your GitHub account**, because that's how sign-in works: the numeric
  account id, login and avatar. The OAuth scope requested is `read:user`
  only — **not** `user:email`, and nothing that grants access to your
  repositories.

What is *not* collected: your email address, your prompts or any content you
generate (benchmarks run on synthetic tokens — LlamaToaster never sees what
you actually use a model for), the model files themselves, or anything about
your machine beyond the hardware inventory above.

**Sharing with other users is opt-in and identity-free.** Settings →
"Share my benchmarks with the community" controls whether your results feed
the shared base at all. When they do, other people reach them only as
aggregates through the assistant — averages grouped by model, backend,
platform and GPU model, each carrying how many contributors it came from.
No username, no account id, no hostname, and never your own rows back to you.

## Layout

```
server/   Fastify API + SQLite (better-sqlite3), serves client/dist and admin/dist
client/   React + Vite + Tailwind SPA (the main dashboard every user sees)
admin/    React + Vite + Tailwind SPA (superadmin-only cross-tenant view, served from a separate origin)
worker/   runs on each benchmark box: long-polls the server, spawns llama-bench/llama-server
shared/   TypeScript types shared by server, worker, and client
```

Workers are **pull-only**: they long-poll the server for work and heartbeat
their own status, and the server never opens a connection to a worker. A
worker can therefore live anywhere with outbound HTTPS — a home GPU rig, a
cloud box, the server's own machine — with nothing to port-forward or
firewall-open on its side.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers the rest: the
pull-queue protocol and its lease/reaper machinery, the schema, auth and
multi-tenancy, device-flow machine enrolment, and the admin surface — plus
the reasoning behind most of what's described in this document.

## Prerequisites

- Node.js 22+ (required by `better-sqlite3` v13, and by Vite 8 for the client/admin builds)
- A built `llama-bench` (and `llama.cpp`) on each worker box — or install one from the Workers page after the worker is up, no manual build needed
- A GitHub OAuth App if you're running with accounts enabled (see below) — not needed for a single-user/no-login deployment

Install once (root deps, then the client's and admin's own deps — `postinstall`
does both automatically), then build both SPAs — wherever the **server**
runs needs this; a worker-only box doesn't:

```bash
npm install
npm run build
npm run build:admin
```

Re-run these after pulling changes that touch `client/`/`admin/` — neither is
built automatically on `npm run server` startup.

## Running the server

```bash
PORT=3000 npm run server
```

For frontend development with hot reload instead of rebuilding on every
change, run the Vite dev server(s) alongside `npm run server` in a second
terminal — each proxies its own API calls to whatever `PORT` the server
above is using (default `3000`; override with `VITE_API_PROXY_TARGET` if you
set a different `PORT`):

```bash
npm run dev:client   # client/, port 5173
npm run dev:admin    # admin/, port 5174 -- only useful with ADMIN_PUBLIC_URL/AUTH_ENABLED set
```

Core env vars (all optional unless noted — a bare `npm run server` with
nothing set gives you a single-user, no-login deployment identical to how
this app originally worked):

- `PORT` — default `3000`.
- `BIND_HOST` — which interface to listen on. Default `127.0.0.1`. Set to
  your VPS's Tailscale IP for a tailnet-only deployment with no reverse
  proxy in front, or leave at `127.0.0.1` once nginx/Caddy is fronting the
  app (see "Deploying publicly" below) — never `0.0.0.0` directly.
- `DB_PATH` — SQLite file location (default `./llamatoaster.db`).
- `LOG_LEVEL` — `debug` | `info` (default) | `warn` | `error`.
- `AUTH_ENABLED` — unset/false: no login, single-tenant, every machine and
  run visible to whoever's using the dashboard (the original behavior).
  `true`: GitHub sign-in required, every user sees only their own data. See
  "Accounts & multi-user" below before turning this on.
- `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` — optional, enable the dashboard's
  AI Assistant panel. Read from a `.env` file at the repo root (copy
  [`.env.example`](.env.example)) or the shell/systemd env — both work
  identically. Without these the panel just reports "not configured."

See [`deploy/orchestrator.env.example`](deploy/orchestrator.env.example) for
every env var (including the multi-user/OAuth/admin ones below), with
comments on what each does.

## Accounts & multi-user (`AUTH_ENABLED=true`)

Sign-in is GitHub OAuth only today — no separate username/password, no
email/verification step. To enable it:

1. Register a GitHub OAuth App (GitHub → Settings → Developer settings →
   OAuth Apps). Callback URL must be exactly `<PUBLIC_URL>/auth/github/callback`.
2. Set `AUTH_ENABLED=true`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, and
   `PUBLIC_URL` (your server's real origin, e.g. `https://llamatoaster.com`).
3. Restart the server. Visiting it now redirects an unauthenticated browser
   to `/login`.

Each account sees only its own machines, runs, and models it registered —
enforced in SQL, not just hidden in the UI. The one exception is the AI
assistant, which can also show **anonymised, aggregated** benchmark numbers
contributed by other users who opted in (Settings → "Share my benchmarks
with the community") — never a username, account id or hostname, never the
caller's own rows, and always with the number of contributors the aggregate
came from. Set `COMMUNITY_MIN_CONTRIBUTORS=<n>` to additionally suppress any
aggregate built from fewer than *n* distinct contributors (default `1`, i.e.
a single opted-in machine may be summarised — labelled as such).

**Superadmin** (optional, on top of `AUTH_ENABLED`): set
`SUPERADMIN_IDENTITIES=github:<your-numeric-github-id>` (comma-separated for
more than one) and `ADMIN_PUBLIC_URL` (e.g. `https://supervise.llamatoaster.com`
— must resolve to the same server, see "Deploying publicly" below). A
superadmin-listed account gets a read-only, cross-tenant view (every user's
machines/runs, an export, basic stats) at that separate origin — nothing
about their experience on the *main* site changes; there's no admin link or
special mode there. Revoking access is just removing the entry and
restarting the process, no DB write needed.

## Running a worker (GPU or CPU box)

**Brand-new machine, nothing downloaded yet?** One command fetches the
repo, installs dependencies, and starts the worker (needs
PowerShell/bash + Node.js 22+; uses `git` if present, otherwise falls back
to a zip/tarball download). `-Url`/`--url` (your server's real
origin) is required. It asks which drive/volume to use — showing free
space, since models are often tens of GB each — and a folder name, then
creates it:

```powershell
# Windows
iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) } -Url https://llamatoaster.com"
```

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --url https://llamatoaster.com
```

Pass `-Dir`/`--dir` to skip the prompt (e.g. for unattended/scripted use):
`-Dir F:\LlamaToaster` / `-- --dir ~/LlamaToaster`.

**First run only:** the worker has no config yet, so it generates a stable
machine id, asks the server for a short code, and prints something like:

```
To connect this machine, visit https://llamatoaster.com/device
Code: ABCD-EFGH   (expires in 15 minutes)
```

Open that link (already logged in, or log in first), confirm the
hostname/platform/GPU shown matches the machine you just started, and
approve — one click. The worker picks this up within a few seconds and is
connected from then on; the saved session survives ordinary restarts, so it
never asks again on its own.
**Only approve a code you generated yourself, on a machine you trust** — a
leaked code can be used to attach someone else's machine to your account
within its 15-minute window.

If a machine's session is revoked (Settings → Active sessions → revoke) or
its refresh token expires, it needs to reconnect:
`./worker/setup-worker.sh --reconnect` (or `-Reconnect` on the `.ps1`, or
`--reconnect`/`-Reconnect` on `bootstrap.sh`/`.ps1` to also pull the latest
code first) clears just the saved session, keeping `machine_id` and every
other setting -- the server recognizes it as the *same* machine and keeps
its history/display name. `--force`/`-Force` instead regenerates the whole
file from scratch, including a fresh `machine_id` -- only use that if you
actually want this to enrol as a brand-new machine.

It's safe to re-run the same command any time to pull the latest code: an
existing install is updated in place (git fetch + hard reset where it is a
git checkout, a fresh-tarball sync over same-named files otherwise), leaving
every user-local file alone -- config.json, models, .db files, logs,
llama.cpp builds -- and
[`worker/setup-worker.ps1`](worker/setup-worker.ps1) /
[`.sh`](worker/setup-worker.sh) underneath only write `config.json` once —
after that, the exact same command just (re)starts the worker (no prompt,
no re-enrolment, since it already has a saved session). Pass `-Force` /
`--force` to redo config generation (e.g. to pick a different drive), and
see either script's header comment for the full set of overrides
(`-WorkerName`, `-Backend`).

Already have the repo checked out? Skip straight to
`.\worker\setup-worker.ps1 -Url https://llamatoaster.com` (or the `.sh`
equivalent) — same idempotent behavior and folder prompt, no download step.

To configure by hand instead, edit `worker/config.json`:

```json
{
  "worker_name": "Local",
  "backend": "vulkan",
  "llama_cpp_build": "b10068",
  "llama_bench_path": "C:\\llama\\llama-bench.exe",
  "model_dir": "C:\\llm",
  "url": "https://llamatoaster.com",
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

Every run also gets its own structured log file under `logs/runs/<run_id>.log`
— a header (llama.cpp build/backend, OS, CPU, RAM, GPU, model, total
tests/runs), one line per test and per repeat as it happens, and an
end-of-run summary (tests/runs completed vs. failed vs. cancelled, and
whether the run finished on its own or was stopped early). Downloadable from
the Run page next to the CSV export whenever a run has at least one failed
test (`GET /api/runs/:id/log`, proxied from this worker's own local log file).

Then, once `config.json` exists:

```bash
npm run worker
```

## Deploying publicly / as a service

The app defaults to a single-user, tailnet-or-loopback-only deployment —
nothing extra is required for that. Putting it behind nginx/Caddy + TLS for
real accounts, or running the server + a CPU worker as systemd/PM2 services
on the VPS itself, is covered in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
(including the security model behind multi-tenancy and the admin surface).

## Using it

1. **Dashboard** (`/`) — your own machines (status, backend, live job
   progress) and recent runs. Nothing here is ever cross-tenant, even for a
   superadmin — the cross-tenant view lives entirely on the separate admin
   origin, if configured.
2. **Models** (`/models`) — register a model (local path or Hugging Face repo/file),
   or use the "Search & download (Hugging Face)" panel: search by model name, expand a
   repo to see its quant files, pick a machine, download straight to its `model_dir`.
3. **New Run** (`/new-run`) — pick a model + machine, set the sweep with graphical
   controls (chip inputs for numeric fields, toggles for cache types/flash attention,
   a repeats stepper), or open "Advanced: raw JSON" to edit/paste the equivalent JSON
   directly, then trigger.
4. **Runs** (`/runs`) — list; polls every 5s while anything is `running`.
5. **Run Detail** (`/runs/:id`) — results table + Chart.js bar chart + raw JSON download.
6. **Compare** (`/compare`) — pick 2+ runs for side-by-side bars.
7. **Workers** (`/workers`) — one card per machine: online/offline/**Inaccessible**
   status, platform/backend, detected CPU/GPU/RAM (best-effort, informational), and
   two separate lists — **Downloaded** (installed builds, active one marked, with
   Activate/Delete) and **Available to install** (from GitHub, checked every few
   minutes). Every action is a manual click — nothing downloads, switches, or
   deletes on its own.
8. **Device** (`/device`) — approve a machine enrolment code (see "Running a worker"
   above); bookmarkable, for enrolling a headless box you set up over SSH.
9. **Settings** (`/settings`, only with `AUTH_ENABLED`) — connected sign-in providers,
   active sessions (revoke individually or "sign out everywhere else"), and the
   community-benchmarks sharing toggle.
10. Export: `GET /api/results/export?format=json|csv|md[&runs=id1,id2]`.
11. **AI Assistant** — collapsible panel on the right of every page (collapsed by
    default). Chats with whatever OpenAI-compatible provider you configure via
    `AI_API_KEY`/`AI_BASE_URL`/`AI_MODEL`, with your hardware, registered
    models, and recent benchmark results automatically included as context, plus
    live Hugging Face lookups and (when signed in) anonymised community
    benchmark aggregates. Rate-limited per signed-in user (30/hour, 150/day)
    and by a global daily cap across everyone, to protect the configured
    provider's budget.

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
combination's live status individually as it runs.

`expandSweep` also silently drops any `flash_attn:"off"` combination paired
with a quantized `cache_type_k`/`cache_type_v` -- llama.cpp refuses to create
a context for that combination outright ("quantized V cache requires
flash_attn to be enabled"), so it can never produce a result. A sweep made
up entirely of such combinations is rejected at trigger time instead of
creating a run with nothing to do.

## API

A representative slice — [`docs/API.md`](docs/API.md) has the full table.
See `server/src/routes/*.ts` for everything else (auth, device-flow
enrolment, sessions, and admin routes all exist too, but are meant to be
driven by the SPA, not called directly).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs/trigger` | Queue a sweep against one of your own machines |
| GET | `/api/runs` | list your own runs |
| GET | `/api/models` | list models |
| GET | `/api/workers` | list your own machines |
| GET | `/api/results/export` | json \| csv \| md |
| POST | `/api/ai/chat` | AI assistant (SSE streamed) |

## Security / Networking

Every write is scoped to the caller's own account in SQL, machine enrolment
is worker-initiated (device flow, nothing sensitive crosses browser→
terminal), community benchmark data is consent-gated and exposed only as
identity-free aggregates, and the superadmin surface lives on a separate
hostname that 404s for anyone else.
Full detail, plus how to put the app behind TLS for a public deployment, is
in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Known limitations

- VRAM peak is best-effort via `systeminformation.graphics()` (no clean AMD/Vulkan
  equivalent of `nvidia-smi`). RAM peak is the trustworthy number.
- Only GitHub is supported as a sign-in provider today; the auth layer is
  written to make adding a second provider (e.g. Google) a config-map change,
  not a rewrite, but none is wired up yet.
