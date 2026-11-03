<p align="center">
  <img src="assets/logo.png" alt="LlamaToaster" width="320">
</p>

<p align="center">
  <a href="https://llamatoaster.com"><img src="https://img.shields.io/badge/try%20it-llamatoaster.com-f5a623" alt="Try it at llamatoaster.com"></a>
  <a href="https://llamatoaster.com"><img src="https://img.shields.io/website?url=https%3A%2F%2Fllamatoaster.com%2Fhealth&up_message=online&down_message=offline&label=status" alt="Hosted instance status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/node-%3E%3D22.22.2-brightgreen" alt="Node >= 22.22.2">
</p>

**LlamaToaster answers four questions: no, no, yes, no.**

- **No!** You don't need your own data center to run your own AI.
- **No!** You're not running your AI as well as your machine allows.
- **Yes!** Your machine can run local inference without a top GPU, or with no
  graphics card at all.
- **No!** You don't need to be an engineer to do this. It's simpler than you think.

It tells you whether a GGUF model will run on your machine before you download
it, finds the best [llama.cpp](https://github.com/ggml-org/llama.cpp) settings for
your hardware, and measures the speed you will really get.

👉 **[llamatoaster.com](https://llamatoaster.com)** — the hosted instance.
Sign in with GitHub, connect a machine, start testing.

## The questions it settles

| You're asking | Where you get the answer |
|---|---|
| Will this model run on my machine? Is there enough memory? | **Will it fit?** on the Models page, without downloading the model |
| How many tokens per second will I really get, in prefill and in generation? | **New test**: a real `llama-server` run at the context you need |
| Which launch parameters should I use? | **New test**: a map of context against layers, flash attention on and off, every K/V cache type that works, CPU threads |
| How do I make it faster? | **New test**: your chosen configs measured next to the defaults. **Compare** and the AI assistant show what moved the number |
| How do I tune it for the way I use it? | **Serve** already profiles your real traffic. Tuning from that profile is [in development](#in-development) |

## Know before you download

Existing predictors don't work. They have no idea whether you can really run a
model until you've downloaded it, because they estimate from a formula and a file
size. They don't know your llama.cpp build, your backend, or what else is holding
your memory right now.

LlamaToaster asks llama.cpp itself. It fetches only the model's GGUF header from
Hugging Face, pads it to full size as a sparse file, and runs llama.cpp's own
`llama-fit-params` on your machine against what is free right now.

**Download 10–15 MB instead of 20–30 GB, and get the answer.** On 12 models at
4 context sizes, the header gave the same answer as the real file at 45 of 45 points.

## Speed on your hardware, not someone else's

Quality benchmarks are everywhere. Performance benchmarks either aren't universal,
or don't show performance on your hardware, or both.

LlamaToaster runs on your machine, with your build, on the model you picked. Prefill
and generation are measured separately, because they behave differently and you feel
them differently. What comes out is the setting you paste into your own
`llama-server` or `llama-cli` run.

## What it does for you

- **Any platform, any backend.** Windows, Linux and macOS, on everything
  [llama.cpp](https://github.com/ggml-org/llama.cpp) ships a release for: CPU,
  Vulkan, CUDA, ROCm, SYCL, OpenVINO, Metal. The newest builds are listed within
  minutes of release and install with one click.
- **Any GGUF model on [Hugging Face](https://huggingface.co/models?library=gguf).**
  Search, pick a quantization, download to the machine. Gated repos need `HF_TOKEN`
  set on that machine. Around 4 million GGUF files are indexed by sha256, so a file is
  identified by what it *is*, not by what it was renamed to, and a mirrored copy
  doesn't turn into a second entry.
- **One command to install and run.** [See below](#two-ways-to-run-it). Node.js and
  the rest are handled for you.
- **K/V cache quantization, found automatically.** Don't guess which cache types will
  work and which will end in a silent fallback to the CPU. LlamaToaster tries all
  64 type pairs with flash attention on and off, and reads the answer from llama.cpp's
  own scheduler. No weights are loaded, so each check takes about half a second.
- **Two ways to find the offload when VRAM runs short:**
  - **llama-fit**, built into llama.cpp, quickly and quite accurately places both
    dense and MoE models: layers and experts.
  - **No-spill**, LlamaToaster's own method, takes longer and needs the model on disk,
    but squeezes out the last few percent. It puts as much as possible in VRAM, right
    up to the point where data starts silently spilling out of VRAM into system
    memory. Spill degrades performance with no crash and no error, so you'd otherwise
    never know.
- **No surprise out-of-memory.** llama-fit's answers keep a margin of free VRAM
  (1 GiB by default), and real loads with real requests can confirm them at your
  context, so you don't lose a session to a sudden OOM and have to restart. No-spill
  trades that margin for speed by design, and memory that other apps take later is
  outside LlamaToaster's control.
- **An AI assistant that reads the numbers for you.** Ask what's limiting you, which
  parameter moved the needle and what to try next. It reads your own tests directly.

## In development

**Our own daemon for analyzing how you use your model and optimizing for it.** The
Serve page already loads a model on your machine behind an OpenAI-compatible URL for
your agents. It shows VRAM and RAM split into this app, other apps and free, speed
against context length from real traffic, and your usage profile: tokens in, cached
and out. Still to come is tuning from that profile, finding the right balance between
generation and prefill for your workload.

## Why every test counts

The main value is the network effect. Every test you run is uploaded to a shared
database, so the next person can get an answer faster, without repeating the work.
Today, other people reach that data as anonymous aggregates through the AI assistant.
Over time it should answer directly for hardware it has already seen.

A million tests across different hardware, software and models is a million more
people who can get into local AI with less friction. That's the reason to use the
hosted instance rather than a private copy: one sweep on an RTX 4090 is a data point,
and thousands of them are a map of local inference.
[Here's what is collected.](#what-is-collected-and-what-isnt)

## Two ways to run it

**Use the hosted instance** — nothing to deploy. Sign in at
[llamatoaster.com](https://llamatoaster.com), then run one command on the
machine you want to benchmark (Windows PowerShell shown; macOS/Linux and the
full flow are under
[Running a worker](#running-a-worker-gpu-or-cpu-box)):

```powershell
irm https://llamatoaster.com/install.ps1 | iex
```

That URL is a 302 redirect to
[`worker/bootstrap.ps1`](worker/bootstrap.ps1) in this repo — the domain
supplies the short name, the bytes that run stay readable and diffable on
GitHub. Setup registers a `toaster` command, so every later start, restart
or update is one word from any folder.

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
- **If you use the Serve page**: for each request a machine serves, its token
  counts, timings, status and endpoint, never its text. Prompts and answers stay
  on that machine. Also the settings you save there, including a system prompt.
- **Your GitHub account**, because that's how sign-in works: the numeric
  account id, login and avatar. The OAuth scope requested is `read:user`
  only — **not** `user:email`, and nothing that grants access to your
  repositories.

What is *not* collected: your email address, your prompts or any content you
generate (benchmarks run on synthetic tokens, and the Serve page's traffic stays on
your machine), the model files themselves, or anything about
your machine beyond the hardware inventory above.

## Layout

```
server/   Fastify API + SQLite (better-sqlite3), serves client/dist and admin/dist
client/   React + Vite + Tailwind SPA (the main dashboard every user sees)
admin/    React + Vite + Tailwind SPA (superadmin-only cross-tenant view, served from a separate origin;
          bundles client/'s own test-detail page, so a test looks the same to an operator as to its owner)
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

- Node.js 22.22.2+ (22+ is required by `better-sqlite3` v13 and by Vite 8 for the client/admin builds; jsdom 30, used by the client tests, needs 22.22.2)
- A built `llama-bench` (and `llama.cpp`) on each worker box — or install one from the Machines page after the worker is up, no manual build needed
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
contributed by other users (once the operator enables community sharing in the
admin console) — never a username, account id or hostname, never
the caller's own rows, and always with the number of contributors the aggregate
came from. Set `COMMUNITY_MIN_CONTRIBUTORS=<n>` to additionally suppress any
aggregate built from fewer than *n* distinct contributors (default `1`, i.e.
a single contributing machine may be summarised — labelled as such).

**Superadmin** (optional, on top of `AUTH_ENABLED`): set
`SUPERADMIN_IDENTITIES=github:<your-numeric-github-id>` (comma-separated for
more than one) and `ADMIN_PUBLIC_URL` (e.g. `https://supervise.llamatoaster.com`
— must resolve to the same server, see "Deploying publicly" below). A
superadmin-listed account gets a read-only, cross-tenant view (every user's
tests, an export, basic stats) at that separate origin. Any test — including
one still running, which updates live — opens in the same test page its owner
sees (results, charts, scored profiles, probe ladders), just without the
controls that would change it (pause/stop/trigger/delete). Nothing about their
experience on the *main* site changes; there's no admin link or special mode
there. Revoking access is just removing the entry and
restarting the process, no DB write needed.

## Running a worker (GPU or CPU box)

**Brand-new machine, nothing downloaded yet?** One command fetches the
repo, installs dependencies, registers the `toaster` command, and starts the
worker (needs only PowerShell or bash + curl; Node.js 22+ is installed for you
if missing — via winget on Windows, or a checksum-verified download from
nodejs.org into the install folder's `.node/` on macOS/Linux, no sudo; uses
`git` if present, otherwise falls back to a zip/tarball download). It asks which drive/volume to use —
showing free space, since models are often tens of GB each — and a folder
name, then creates it:

```powershell
# Windows
irm https://llamatoaster.com/install.ps1 | iex
```

```bash
# macOS / Linux
curl -fsSL https://llamatoaster.com/install.sh | bash
```

Both short URLs are 302 redirects to the bootstrap script in this repo
([`.ps1`](worker/bootstrap.ps1) / [`.sh`](worker/bootstrap.sh)), served by
[`server/src/routes/install.ts`](server/src/routes/install.ts). The
redirect is deliberate: piping a URL into a shell is a trust decision, and
the bytes that actually execute stay in the public repo where they can be
read, blamed and diffed, rather than being generated by a server nobody can
audit. The raw GitHub URLs keep working directly if you prefer them, or the
origin is unreachable:

```powershell
irm https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.ps1 | iex
```

```bash
curl -fsSL https://raw.githubusercontent.com/noname9006/LlamaToaster/main/worker/bootstrap.sh | bash
```

`-Url`/`--url` defaults to `https://llamatoaster.com`. **Self-hosting?** Pass
your own origin — which a plain pipe can't carry, so use the argument-passing
form:

```powershell
iex "& { $(irm https://your-server.example.com/install.ps1) } -Url https://your-server.example.com"
```

```bash
curl -fsSL https://your-server.example.com/install.sh | bash -s -- --url https://your-server.example.com
```

The same forms take `-Dir`/`--dir` to skip the folder prompt (e.g. for
unattended/scripted use): `-Dir F:\LlamaToaster` / `--dir ~/LlamaToaster`.

**After the first install, use `toaster`** — from any folder, on any of the
three platforms:

| Command | What it does |
| --- | --- |
| `toaster` | Start the worker (`toaster restart` is the same thing) |
| `toaster update` | Pull the latest code + dependencies, then start — stops with the real error (and doesn't start old code) if any step fails |
| `toaster reconnect` | Re-approve this machine after its session was revoked |
| `toaster logs` | Open (Windows) or list (macOS/Linux) the worker log folder |
| `toaster where` | Print the install folder |
| `toaster uninstall` | Remove the command, leaving the install untouched |

Stop a running worker with Ctrl+C in its window. The command is per-user and
minimal by design: on Windows it's one `toaster.cmd` under
`%LOCALAPPDATA%\LlamaToaster\bin` plus that folder appended to your **user**
PATH; on macOS/Linux it's one executable at `~/.local/bin/toaster` and no
shell-rc edits at all (if that folder isn't on your PATH, setup prints the
line to add). No admin/sudo, nothing system-wide, no service. `toaster
uninstall` reverses exactly those changes.

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
its refresh token expires, it needs to reconnect: `toaster reconnect` (or,
without the shim, `./worker/setup-worker.sh --reconnect` / `-Reconnect` on
the `.ps1`, or the same flag on `bootstrap.sh`/`.ps1` to also pull the latest
code first) clears just the saved session, keeping `machine_id` and every
other setting -- the server recognizes it as the *same* machine and keeps
its history/display name. `--force`/`-Force` instead regenerates the whole
file from scratch, including a fresh `machine_id` -- only use that if you
actually want this to enrol as a brand-new machine.

`toaster update` — or re-running the install command, which does the same
thing — pulls the latest code any time: an existing install is updated in
place (git fetch + hard reset where it is a
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
That also installs the `toaster` command, so it's the one-time step an
install predating the shim needs to get it.

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
to `worker/`) — it's where builds installed from the Machines page land, one
subfolder per tag. `llama_bench_path`/`llama_cpp_build` are the *initial*
active build; switching versions from the Machines page rewrites both fields
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
the Test Detail page next to the CSV export whenever a run has at least one
failed test (`GET /api/tests/:id/log`, proxied from this worker's own local
log file).

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
   progress) and recent tests. Nothing here is ever cross-tenant, even for a
   superadmin — the cross-tenant view lives entirely on the separate admin
   origin, if configured.
2. **Models** (`/models`) — register a model (local path or Hugging Face repo/file),
   or use the "Search & download (Hugging Face)" panel: search by model name, expand a
   repo to see its quant files, pick a machine, download straight to its `model_dir`.
   **Will it fit?** sizes any of those files on a chosen machine, downloaded or not.
3. **New test** (`/benchmark`) — the guided path: pick a machine, a model and the
   context you need, then work through four steps, each its own test: **Baseline**
   (the model at that context, every layer on the GPU), **Find the best fit**
   (llama-fit's map of context against layers, K/V cache detection, an optional
   real-load no-spill check), **CPU threads**, and **Confirm** (your chosen configs
   measured next to the baseline).
4. **Custom Test** (`/custom-test`) — the hand-built path: set the sweep yourself with
   graphical controls (chip inputs for numeric fields, toggles for cache types/flash
   attention, a repeats stepper), or open "Advanced: raw JSON" to edit/paste the
   equivalent JSON directly, then trigger a single standalone test.
5. **Tests** (`/tests`) — list; polls every 5s while anything is `running`.
6. **Test Detail** (`/tests/:id`) — results table + Chart.js bar chart + raw JSON download.
7. **Compare** (`/compare`) — pick 2+ tests for side-by-side bars.
8. **Serve** (`/serve`) — load a model on a machine for your agents; see
   [Serving a model to agents](#serving-a-model-to-agents).
9. **Machines** (`/workers`) — one card per machine: online/offline/**Inaccessible**
   status, platform/backend, detected CPU/GPU/RAM (best-effort, informational), and
   two separate lists — **Downloaded** (installed builds, active one marked, with
   Activate/Delete) and **Available to install** (from GitHub, checked every few
   minutes). Every action is a manual click — nothing downloads, switches, or
   deletes on its own.
10. **Device** (`/device`) — approve a machine enrolment code (see "Running a worker"
    above); bookmarkable, for enrolling a headless box you set up over SSH.
11. **Settings** (`/settings`, only with `AUTH_ENABLED`) — connected sign-in providers,
    active sessions (revoke individually or "sign out everywhere else"), and the
    community-benchmarks sharing toggle.
12. Export: `GET /api/results/export?format=json|csv|md[&tests=id1,id2]`.
13. **AI Assistant** — collapsible panel on the right of every page (collapsed by
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

## Serving a model to agents

Open **Serve**, pick a machine and a model, and load it. Agents then connect to the
machine, not to the LlamaToaster server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8642/v1", api_key="your key, or anything without one")
client.chat.completions.create(model="the API identifier on the page", messages=[{"role": "user", "content": "Hello"}])
```

The endpoint also speaks `/v1/completions`, `/v1/responses`, `/v1/embeddings`,
`/v1/rerank` and `/v1/messages`. Images and other media must be sent inline as
`data:` URLs, because llama-server would otherwise fetch a remote URL itself from your
machine. It listens on that machine only until you choose
otherwise in Endpoint settings, and refuses a network address without an API key. It
is plain HTTP, so use a network you trust or put TLS in front of it.

Load settings need a reload. Sampling settings apply on the next request, and fill
in only what a request leaves unset unless you tell them to override it. Tests take
priority: a test unloads the model, and agents get a 503 with `Retry-After` until it
comes back when the machine's queue is clear.

## API

A representative slice — [`docs/API.md`](docs/API.md) has the full table.
See `server/src/routes/*.ts` for everything else (auth, device-flow
enrolment, sessions, and admin routes all exist too, but are meant to be
driven by the SPA, not called directly).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/tests/trigger` | Queue a sweep against one of your own machines |
| GET | `/api/tests` | list your own tests |
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

- VRAM readings depend on the platform: per-process where the OS or driver offers it
  (Windows GPU counters, `nvidia-smi`), whole-adapter otherwise. Each reading is
  labelled with its source and accuracy.
- A no-spill answer can differ between sittings, because the driver splits memory
  differently from one session to the next. Re-run it if the machine has changed
  state. llama-fit's answer depends on free memory at that moment too.
- Sizing a model without downloading it needs a filesystem with sparse files (NTFS,
  ext4, XFS, APFS, not FAT or exFAT).
- Only GitHub is supported as a sign-in provider today; the auth layer is
  written to make adding a second provider (e.g. Google) a config-map change,
  not a rewrite, but none is wired up yet.
