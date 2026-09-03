# Architecture

How LlamaToaster is put together: the processes, how they talk, what's in the
database, and the rules that keep one account's data out of another's.

For running it, see the [README](../README.md); for the HTTP surface, see
[API.md](API.md); for putting it on a server, [DEPLOYMENT.md](DEPLOYMENT.md).

## Processes

There are exactly two kinds of process, plus two static SPAs served by the
first one.

| | what it is | where it runs |
| --- | --- | --- |
| `server/` | Fastify API + SQLite (better-sqlite3). Also serves `client/dist` and `admin/dist` as static files. | one box, publicly reachable (or tailnet-reachable) |
| `worker/` | long-polls the server for jobs, spawns `llama-bench` / `llama-server`, reports results | every benchmark box: your GPU rig, a cloud instance, the server's own machine |
| `client/` | React + Vite + Tailwind SPA — the dashboard every signed-in user sees | built to static files, served by `server/` |
| `admin/` | React + Vite + Tailwind SPA — superadmin-only, read-only, cross-tenant | built to static files, served by `server/` on a **separate hostname** |
| `shared/` | TypeScript types and pure logic (sweep expansion, config hashing, scoring, VRAM estimation) imported by all of the above | — |

There is no Docker, no message broker, and no second Node process for the
admin surface: one Fastify instance tells the two hostnames apart by
`req.hostname` and gates the admin routes on that.

## Workers are pull-only

The server never opens a connection to a worker. Workers do all the calling:

```
worker                                          server
  |  POST /api/worker/queue   (long poll)          |
  |----------------------------------------------->|
  |             <- a job, or nothing after timeout  |
  |                                                 |
  |  POST /api/worker/heartbeat   (every few s)     |
  |----------------------------------------------->|  extends the job lease,
  |             <- cancel_requested, control        |  carries live progress
  |                                                 |
  |  POST results / job completion                  |
  |----------------------------------------------->|
```

This is the most load-bearing decision in the design. Because the worker only
ever makes **outbound** HTTPS calls, it can sit behind any NAT or firewall
with nothing to port-forward, and the server needs no inbound route into your
home network.

Consequences the code has to handle, and does:

- **Leases, not assignments.** `worker_jobs.lease_expires_at` is set on claim
  and extended by every heartbeat. A worker that dies mid-job stops
  heartbeating, its lease expires, and [`server/src/reaper.ts`](../server/src/reaper.ts)
  requeues the job (up to `MAX_ATTEMPTS`) or fails it. Without the reaper,
  cancelled and lease-expired jobs would stay permanently unclaimable.
- **Liveness is derived, never stored.** A worker is "online" if its last
  heartbeat is recent enough — there is no `is_online` column to go stale.
- **The queue handler is event-driven** ([`queue-events.ts`](../server/src/queue-events.ts)),
  not a polling loop: enqueueing a job wakes the long-poll already waiting.
- **Worker-reported state is validated** before it is trusted
  ([`validate-worker-state.ts`](../server/src/validate-worker-state.ts)).

Job types are `benchmark`, `install_build`, `download_model` and
`delete_model_file` — installing a llama.cpp build or downloading a GGUF from
the Workers page goes through the same queue as a benchmark.

## Data model

SQLite; schema in [`server/src/db/schema.sql`](../server/src/db/schema.sql),
migrations in [`migrate.ts`](../server/src/db/migrate.ts). All access goes
through [`repo.ts`](../server/src/db/repo.ts) — no route builds SQL itself.

**Benchmarking**

- `runs` — one test. (The API and UI call this entity a *test*; the table kept
  its original name.) Carries the pinned llama.cpp build, backend, device
  name, and the exact sweep config as JSON.
- `run_items` — the individual configurations a sweep expands into.
- `results` — one row per measured point: `pp` / `tg` / `pg`, `n_prompt`,
  `n_gen`, `n_depth`, throughput, plus `config_hash` (hashed over everything
  that is *not* a speculative-decoding spec field, which is what makes
  baseline-vs-speculative twin joins possible), caveat flags, and thermal
  telemetry.
- `probe_attempts`, `model_machine_limits` — the "largest config this model
  actually runs at on this machine" probe: every attempt, and the verified
  result.
- `quality_results` — perplexity / quality measurements, kept apart from
  throughput.

**Models**

- `models` — the catalog.
- `hf_gguf_index` — a Hugging Face GGUF index keyed by
  `(sha256, repo_id, filename)`. The sha256 is the point: the same file
  mirrored across repos or renamed keeps distinct rows, so a local GGUF is
  identified by content hash rather than by whatever it was named on disk.
  `last_seen` drives incremental refresh; `deleted_at` tombstones files that
  disappear from HF.
- `local_model_cache` — each worker's own scan of its `model_dir`
  (path → size, mtime, sha256, resolved HF id, state), so re-hashing happens
  only when a file actually changes.

**Accounts and machines**

- `users` is provider-agnostic; `identities` maps
  (provider, provider_user_id) → user. GitHub is the first provider, not a
  baked-in assumption.
- `sessions` never stores the session token — only its SHA-256
  ([`session.ts`](../server/src/session.ts)). A leaked database read yields no
  usable session cookie.
- `workers` — an enrolled machine, with a `machine_id` and a hardware
  fingerprint used for duplicate detection at enrolment time.
- `ai_usage` — per-day AI budget accounting.

## Auth, tenancy, and the admin surface

`AUTH_ENABLED` is the switch. Unset, the app is exactly what it originally
was: single-user, no login, every machine and test visible to whoever opens
the dashboard. Set, everything below applies.

**Sign-in** is GitHub OAuth. **Machine enrolment is worker-first**: the worker
starts, calls `/api/device/start`, and shows a user code you approve from the
browser while signed in. The worker never handles a password, and the
approval binds the machine to the account that clicked it. If the hardware
fingerprint matches a machine you already have, approving *merges* into that
row instead of creating a duplicate.

**Tenancy** is enforced in SQL, not in the UI: every read is scoped by
`user_id` inside `repo.ts`. Route handlers cannot forget to scope, because
they never write the query.

**The superadmin surface** lives on its own hostname (`ADMIN_PUBLIC_URL`) and
is gated in Fastify's `onRequest` phase — before the auth `preHandler` runs —
so an admin-route request on the wrong hostname is rejected before anything
else inspects it. It is read-only, and it shows machines, not users.

**Community aggregates** — the cross-user comparison data the AI assistant can
cite — are the one place where data crosses tenant lines. The constraints are
enforced in the SQL itself, in `repo.ts`'s `communityRepo`: the operator must
enable the feature, each contributing user must consent
(`users.share_benchmarks`), the calling user's own rows are always excluded,
and the projection carries no username, account id, hostname or build tag —
only an average per (model, backend, test type, platform, GPU model) group,
with the number of contributors it came from.

`COMMUNITY_MIN_CONTRIBUTORS` sets a k-anonymity floor on top of that:
groups combining fewer than *n* distinct contributors are dropped in the
`HAVING` clause, so they never reach the caller. It defaults to `1` — a
single opted-in machine may be summarised, carrying `contributorCount: 1` so
the assistant reports it as one machine rather than a consensus.
`BENCHMARKING_PLAN_V8.md` §0.9 originally mandated 5; that floor suppressed
essentially every group until the database is very large, which defeated the
purpose of having a shared base at all. Set the variable to `5` to restore it.

## Retention

The reaper also runs a maintenance sweep: terminal jobs are dropped after
`TERMINAL_JOB_RETENTION_DAYS`, GPU clock sample series after
`GPU_CLOCK_SAMPLE_RETENTION_DAYS`, and expired sessions and stale enrolments
are collected rather than accumulating forever.

## Design documents

The source cites two specs by section number (`§2.3`, `§0.9`, …). They are the
historical design documents this implementation was built against, kept in the
repo so those citations resolve:

- [`plans/MULTIUSER_PLAN.md`](plans/MULTIUSER_PLAN.md) — auth, sessions, the
  pull-queue protocol, device enrolment, multi-tenancy, the admin surface,
  deployment, and the security checklist.
- [`plans/BENCHMARKING_PLAN_V8.md`](plans/BENCHMARKING_PLAN_V8.md) — the
  measurement layer: normative definitions (§0), goal-parameterized scoring,
  thermal telemetry, context curves, the verify-by-probe advisor,
  model-vs-model comparison, quality measurement, and reproducible
  export/import.

They are **plans, not current documentation**. Where they disagree with the
code, the code wins — most visibly, the `Run` entity was later renamed to
`Test` across the API and UI while the `runs` table kept its name.
