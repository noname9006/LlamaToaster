# LlamaToaster — LLM Benchmark Web App

Benchmark orchestrator: a **server** (Fastify API + a React SPA + SQLite)
dispatches sweeps to one or more **workers** (your own GPU/CPU boxes), which
run `llama-bench` (or `llama-server`, for MTP/speculative-decoding
benchmarks) and post results back as they complete. Multi-user: sign in with
GitHub, each account sees only its own machines and runs; a superadmin gets
a separate, read-only cross-tenant view on its own origin.

Workers are **pull-only** — they long-poll the server for work and heartbeat
their own status; the server never opens a connection to a worker. That
means a worker can live anywhere with outbound HTTPS (your home GPU rig, a
cloud box, the server's own machine) with nothing to port-forward or
firewall-open on the worker side.

Architecture, schema, and security model follow
[`MULTIUSER_PLAN.md`](./MULTIUSER_PLAN.md) — auth, the pull-queue worker
protocol, device-flow machine enrolment, the admin surface, and the
reasoning behind most of what's described in this document.

## Layout

```
server/   Fastify API + SQLite (better-sqlite3), serves client/dist and admin/dist
client/   React + Vite + Tailwind SPA (the main dashboard every user sees)
admin/    React + Vite + Tailwind SPA (superadmin-only cross-tenant view, served from a separate origin)
worker/   runs on each benchmark box: long-polls the server, spawns llama-bench/llama-server
shared/   TypeScript types shared by server, worker, and client
```

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
with the community") — never a username, never a group of fewer than 5
distinct contributors.

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
to a zip/tarball download). `-VpsUrl`/`--vps-url` (your server's real
origin) is required. It asks which drive/volume to use — showing free
space, since models are often tens of GB each — and a folder name, then
creates it:

```powershell
# Windows
iex "& { $(irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1) } -VpsUrl https://llamatoaster.com"
```

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash -s -- --vps-url https://llamatoaster.com
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

It's safe to re-run the same command any time: if the folder already has a
checkout, the download step is skipped, and
[`worker/setup-worker.ps1`](worker/setup-worker.ps1) /
[`.sh`](worker/setup-worker.sh) underneath only write `config.json` once —
after that, the exact same command just (re)starts the worker (no prompt,
no re-enrolment, since it already has a saved session). Pass `-Force` /
`--force` to redo config generation (e.g. to pick a different drive), and
see either script's header comment for the full set of overrides
(`-WorkerName`, `-Backend`).

Already have the repo checked out? Skip straight to
`.\worker\setup-worker.ps1 -VpsUrl https://llamatoaster.com` (or the `.sh`
equivalent) — same idempotent behavior and folder prompt, no download step.

To configure by hand instead, edit `worker/config.json`:

```json
{
  "worker_name": "Local",
  "backend": "vulkan",
  "llama_cpp_build": "b10068",
  "llama_bench_path": "C:\\llama\\llama-bench.exe",
  "model_dir": "C:\\llm",
  "vps_url": "https://llamatoaster.com",
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

## Deploying publicly

The app defaults to a single-user, tailnet-or-loopback-only deployment —
nothing here is required for that. To open it up to real accounts on the
public Internet:

1. Point a domain (and, for the optional admin surface, a subdomain — e.g.
   `supervise.llamatoaster.com`) at the server's public IP.
2. Put nginx or Caddy in front of it and get real TLS certs — see
   [`deploy/nginx.conf`](deploy/nginx.conf) for a ready-to-adapt config
   (security headers, CSP, both hostnames) and its own header comment for
   the Caddy-equivalent shape. `certbot --nginx -d yourdomain.com -d
   supervise.yourdomain.com` handles the cert.
3. Set `BIND_HOST=127.0.0.1` (the app should not be reachable on any public
   interface directly — confirm with the VPS firewall too, not just this
   value) and `PORT` to whatever the proxy config forwards to.
4. Set `AUTH_ENABLED=true` and the OAuth/`PUBLIC_URL`/`ADMIN_PUBLIC_URL` vars
   from "Accounts & multi-user" above.
5. Back up the DB on a schedule — [`scripts/backup-db.sh`](scripts/backup-db.sh)
   (WAL-safe `sqlite3 .backup`, integrity-checked, retention-pruned) is meant
   to run daily from cron.

## Deploying a CPU worker on the server's own box

Same worker process as any other, just declared `"backend": "cpu"`. As of
the pull-queue architecture, this machine is **not** special-cased at all —
it enrolls exactly the way any user's GPU box does (see "Running a worker"
above): run `worker/bootstrap.sh` on the VPS itself, approve the resulting
device-flow code from your own account, done. `worker/config.vps.json.example`
exists as a reference for what the resulting config roughly looks like, but
there's no need to hand-author it — bootstrap.sh does it, including
generating and persisting the session credentials.

1. On the server: clone this repo to e.g. `/home/ubuntu/LlamaToaster`, run
   `npm install`, `npm run build`, `npm run build:admin`, and
   `mkdir -p /home/ubuntu/LlamaToaster/data`. The provided systemd units run
   the services as your regular login user (`User=ubuntu`/`Group=ubuntu`)
   rather than a dedicated system account — simpler for a single-operator
   VPS, at the cost of the app running with the same privileges as your
   login user instead of an isolated one. Adjust `User=`/`Group=`/paths in
   both `deploy/*.service` files if your username or clone path differs.
2. Copy [`deploy/orchestrator.env.example`](deploy/orchestrator.env.example)
   to `deploy/orchestrator.env` and fill it in (see "Running the server" and
   "Accounts & multi-user" above for what each var does).
3. Install the two systemd units:
   ```bash
   sudo cp deploy/llamatoaster-worker.service deploy/llamatoaster-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now llamatoaster-server
   ```
   Don't enable `llamatoaster-worker` yet — it needs a real, device-flow-
   enrolled `config.json` first (see step 5), or it'll crash-loop looking
   for one.

   Prefer PM2? [`deploy/ecosystem.server.config.cjs`](deploy/ecosystem.server.config.cjs)
   and [`deploy/ecosystem.worker.config.cjs`](deploy/ecosystem.worker.config.cjs)
   are the equivalent, kept as two separate files deliberately — server and
   worker are independent processes that should be deployable, and able to
   fail, independently. `pm2 start deploy/ecosystem.server.config.cjs` for
   one, `pm2 start deploy/ecosystem.worker.config.cjs` for the other (after
   step 5).
4. Once `llamatoaster-server` is up, open the dashboard from your own
   machine and confirm it loads.
5. Enrol the server's own worker exactly like any other machine:
   `./worker/bootstrap.sh --vps-url https://<your-domain> --dir ~/llamatoaster-worker`,
   then approve the printed code from your logged-in session. Note where it
   wrote `config.json`, then point `llamatoaster-worker.service`'s
   `WORKER_CONFIG=` (or `ecosystem.worker.config.cjs`'s
   `LLAMATOASTER_VPS_WORKER_CONFIG`) at that exact path and start the
   worker service/process.
6. Go to **Workers**, confirm the new card loads, and install a build from
   the dropdown — this replaces manually building llama.cpp on the server.
7. Register a small model (search-and-download panel on **Models**, or place
   a `.gguf` in the worker's `model_dir` by hand) and trigger a small run
   against that worker from **New Run** before doing a full sweep.

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

A representative slice — see `server/src/routes/*.ts` for the full set
(auth, device-flow enrolment, sessions, and admin routes all exist too, but
are meant to be driven by the SPA, not called directly).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs/trigger` | Queue a sweep against one of your own machines |
| GET | `/api/runs` | list your own runs |
| GET | `/api/runs/:id` | run detail + results |
| POST | `/api/runs/:id/pause` \| `/resume` \| `/stop` | control a running sweep |
| GET | `/api/models` | list models |
| POST | `/api/models` | register a model |
| GET | `/api/workers` | list your own machines |
| GET | `/api/workers/:id/available-builds` | installed + installable builds for a machine |
| POST | `/api/workers/:id/llama-cpp/install` | download+extract a build (manual trigger) |
| POST | `/api/workers/:id/llama-cpp/activate` | switch a machine's active build |
| DELETE | `/api/workers/:id/llama-cpp/:tag` | remove an installed build (not the active one) |
| GET | `/api/hf/search?q=` | search Hugging Face repos tagged gguf |
| GET | `/api/hf/repo/*` | list `.gguf` files + sizes in a repo (repo id as the wildcard tail) |
| POST | `/api/workers/:id/models/download` | start a download on a machine (returns as soon as it's accepted, not once it finishes) |
| GET | `/api/results/export` | json \| csv \| md |
| POST | `/api/ai/chat` | AI assistant (SSE streamed) |
| GET | `/api/auth/status` | who's logged in, and whether login is required at all |

## Security / Networking

- **Every write is scoped to the caller's own account** in SQL, not filtered
  in the UI — one user's machines/runs/models are invisible to another's API
  calls too, not just hidden from their screen. Enforced whether or not
  `AUTH_ENABLED` is on; with it off, everything belongs to nobody in
  particular (single-tenant, matching the original design).
- **Machine enrolment is worker-initiated** (device flow, RFC 8628-style):
  nothing sensitive ever crosses browser→terminal. The one accepted risk is
  device-code phishing — if a code meant for someone else leaks (shoulder-
  surfing, a shared screen), it can be approved into a different account
  within its 15-minute window. Mitigated, not eliminated, by the confirm
  card showing hostname/GPU/platform, a warning line, and per-user rate
  limiting on the approve endpoint — the blast radius is mis-attributed
  compute, not exposed secrets or data.
- **The AI assistant's system prompt is server-owned** — a client-supplied
  `system` message is discarded outright, so `curl`ing the chat endpoint
  directly can't override its instructions or its per-user data scoping.
- **Community benchmark aggregates are k-anonymised in SQL**, not just
  stripped of a username column: any group describing fewer than 5 distinct
  contributors is never returned, opt-in (Settings), and never includes the
  querying user's own data.
- **The superadmin surface lives on a separate hostname** (`ADMIN_PUBLIC_URL`)
  and is unreachable from the main site even with a valid superadmin
  session — a request with the wrong `Host` header 404s before any auth
  check runs, so it doesn't even reveal that the surface exists.
- **Put the app behind nginx/Caddy + TLS for any public deployment** — see
  "Deploying publicly" above. A tailnet-only deployment (bind to your VPS's
  Tailscale IP, no reverse proxy, `AUTH_ENABLED` left off) remains a valid,
  simpler alternative if you don't need multiple real-world accounts —
  Tailscale's own WireGuard encryption covers that topology. The worker
  installer refuses a plain `http://` `--vps-url` other than
  localhost/127.0.0.1 by default (a real bearer credential shouldn't go out
  in the clear) — pass `--allow-insecure-url`/`-AllowInsecureUrl` for this
  mode specifically, since Tailscale's own tunnel is the encryption there,
  not TLS.
- No app-level secret is stored anywhere reversible: sessions are looked up
  by `sha256(token)`, GitHub OAuth issues no long-lived shared secret this
  app has to protect beyond the client secret itself.

## Known limitations

- VRAM peak is best-effort via `systeminformation.graphics()` (no clean AMD/Vulkan
  equivalent of `nvidia-smi`). RAM peak is the trustworthy number.
- Only GitHub is supported as a sign-in provider today; the auth layer is
  written to make adding a second provider (e.g. Google) a config-map change,
  not a rewrite, but none is wired up yet.
