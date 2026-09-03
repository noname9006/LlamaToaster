# LlamaToaster Multi-User Transformation Plan (v4.0)

> **v4.0 supersedes v3.3** (kept at `MULTIUSER_PLAN_v3.3_superseded.md`). It is a
> restructure, not a patch — see `MULTIUSER_PLAN_REVIEW.md` for the findings that drove it.
>
> **What changed:**
> 1. **Re-sequenced.** The pull-queue inversion is now **Stage 1**, built single-user on the
>    tailnet, before any auth exists. All nine v3.3 blockers lived there; they are fixed and
>    provable against a real GPU worker before multi-tenancy is layered on.
> 2. **Security fixes applied throughout** — session tokens are no longer stored in plaintext,
>    token generation is specified, the AI system prompt moves server-side, community aggregates
>    get k-anonymity, and `/api/ai/chat` gets a budget.
> 3. **Machine enrolment is worker-first**, RFC 8628 shape: the machine prints a short code, the
>    user types it into the browser and approves. One copy-paste (a constant install command, no
>    per-user templating), one click, no secret ever crosses browser→terminal.
> 4. **The username generator is deleted.** Identity is a `users` row keyed by internal UUID, with
>    N linked `identities` (provider + provider's own id) — GitHub today, any future OAuth provider
>    (Google, etc.) is a new adapter, not a schema change. Machines get user-chosen names,
>    defaulting to their own hostname.
> 5. **Dashboard shows the user's machines**, not a user count.
> 6. **Superadmin lives on its own origin** (`supervise.llamatoaster.com`) — its own login, its own
>    small SPA, its own session cookie — not a hidden route inside the app every tenant uses.

---

> **On `file:line` references.** All 26 in this document were verified against the working tree on
> 2026-08-17 and each resolves to the symbol it names. Treat them as *hints*, not contracts —
> several drifted by 3–21 lines within a single afternoon's editing, and v3.3 shipped with numbers
> that were already stale. **Grep for the symbol name, not the line.** Where a reference is
> load-bearing (a function to delete, a type to change) the symbol name is given alongside it, and
> that is the part that matters.

## Table of contents

- [Execution order](#execution-order)
- [Stage 0 — Prerequisites](#stage-0--prerequisites)
- [Stage 1 — Pull queue, single-user, on the tailnet](#stage-1--pull-queue-single-user-on-the-tailnet)
- [Stage 2 — Auth & sessions](#stage-2--auth--sessions)
- [Stage 3 — Machine enrolment (one-click approve)](#stage-3--machine-enrolment-one-click-approve)
- [Stage 4 — Multi-tenancy](#stage-4--multi-tenancy)
- [Stage 5 — Admin & community](#stage-5--admin--community)
- [Stage 6 — Deployment](#stage-6--deployment)
- [Security checklist](#security-checklist)
- [Appendix A: key types](#appendix-a-key-types)
- [Appendix B: review findings → where fixed](#appendix-b-review-findings--where-fixed)

---

## Execution order

v3.3 built auth first and left the queue inversion for the middle. That is backwards: the
inversion is the part that can fail, and it can be built and proven **without** auth, on the
tailnet you already trust, against the GPU box you already have.

| Stage | What | Ships behind | Exit criteria |
|---|---|---|---|
| **0** | Prerequisites: deps, `errors.ts`, `session.ts`, test harness, DB backup, origin decision | — | `npm test` runs; a backup+restore of the live DB is proven |
| **1** | **Pull queue + control channel + leases + liveness**, single-user, tailnet, shared-secret worker auth | nothing (replaces push dispatch outright) | A full sweep runs end-to-end pull-only; Stop works; killing the worker mid-run self-heals; download progress visible |
| **2** | Auth & sessions (OAuth — GitHub first, provider-agnostic identity model — opaque sessions, rate limits, AI budget) | `AUTH_ENABLED` env | Login works; every `/api/*` route 401s without a session; AI spend is capped |
| **3** | Machine enrolment (worker-first device code, one-click approve) — replaces Stage 1's shared secret | — | A machine enrols from a fresh box in under 60s with one copy-paste and one click |
| **4** | Multi-tenancy (`user_id` threading, isolation, model ownership, scoped AI context) | — | User A provably cannot read/write any of user B's rows |
| **5** | Admin & community (superadmin on its own origin, k-anonymised aggregates, dashboard) | — | Aggregate queries suppress cells below k=5; admin flag revocable without a DB write; `/api/admin/*` 404s off the main hostname |
| **6** | Deployment (existing PM2/systemd + nginx or Caddy in front, CSP, backups — no Docker, see §6.0) | — | Public URL live, HTTPS-only worker link, daily backup verified |

**Stage 1 is the critical path.** It is the only stage that changes how a benchmark actually
executes. Budget it generously; everything after it is comparatively mechanical.

Stages 1 and 2 are independently shippable to the current single-user deployment. Do not start
Stage 4 until Stage 1's exit criteria have held for at least a week of real use.

---

## Stage 0 — Prerequisites

### 0.1 Dependencies

```bash
npm i @fastify/cookie@^11 @fastify/rate-limit@^10
npm i -D vitest
```

- `@fastify/cookie` — Fastify v5 does **not** parse cookies on its own; `req.cookies` and the
  OAuth `state` check depend on it. Register with `app.register(fastifyCookie)`.
- `@fastify/rate-limit` — needed from Stage 2.
- **No `@fastify/cors`.** See §0.5 — the app stays same-origin.

### 0.2 `server/src/errors.ts` (new)

```ts
export class HttpError extends Error {
  constructor(message: string, readonly statusCode: number) { super(message); }
}
export class BadRequestError  extends HttpError { constructor(m = "bad request")  { super(m, 400); } }
export class UnauthorizedError extends HttpError { constructor(m = "unauthorized") { super(m, 401); } }
export class ForbiddenError   extends HttpError { constructor(m = "forbidden")    { super(m, 403); } }
export class NotFoundError    extends HttpError { constructor(m = "not found")    { super(m, 404); } }
export class ConflictError    extends HttpError { constructor(m = "conflict")     { super(m, 409); } }
```

Wire into the existing `app.setErrorHandler` (`server/src/index.ts:134`), which already maps
`error.statusCode`. Add: never leak `error.message` for a 500 — log it, return a generic string.

### 0.3 `server/src/session.ts` (new) — **specify the crypto, do not leave it to the implementer**

v3.3 named these five functions and never implemented them, while the only ID generator it
actually showed used `Math.random()`. They are the security foundation of the whole design.

```ts
import { randomBytes, createHash, timingSafeEqual } from "node:crypto";

// 32 bytes = 256 bits. base64url so the value is safe in a cookie, an
// Authorization header, a shell argument, and a URL without escaping.
// NEVER Math.random() -- it is seeded, predictable, and not a CSPRNG.
export function generateToken(): string {
  return randomBytes(32).toString("base64url"); // 43 chars
}

export const generateSessionId    = generateToken;   // browser + worker access token
export const generateRefreshToken = generateToken;   // worker refresh token
export const generateEnrolmentCode = generateToken;  // machine enrolment secret
export const generateState        = generateToken;   // OAuth CSRF state

// Plain SHA-256 is correct here and bcrypt/argon2 would be wrong: these are
// 256-bit uniformly-random values, not user-chosen passwords. There is no
// dictionary to slow down, and a per-request KDF would just add latency to
// every authenticated call.
export function hashToken(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

// Constant-time compare for anything an attacker can submit repeatedly
// (the OAuth state cookie, the enrolment user_code).
export function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (ab.length !== bb.length) return false;      // length leak is unavoidable and harmless here
  return timingSafeEqual(ab, bb);
}

// Human-typeable short code, ONLY ever a display hint scoped to one user's
// pending enrolments (see Stage 3.3) -- never a standalone credential.
// Alphabet excludes 0/O/1/I/L to survive being read off a screen.
const CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // 31 chars
export function generateUserCode(): string {
  // Rejection sampling from randomBytes -- a plain `% 31` would bias the
  // first 8 letters. Still crypto, still not Math.random().
  const out: string[] = [];
  while (out.length < 8) {
    for (const b of randomBytes(16)) {
      if (b >= 248) continue;                     // 248 = 31 * 8, the largest multiple of 31 < 256
      out.push(CODE_ALPHABET[b % 31]);
      if (out.length === 8) break;
    }
  }
  return `${out.slice(0, 4).join("")}-${out.slice(4).join("")}`; // "ABCD-EFGH"
}
```

### 0.4 Test harness

There is **no test runner in the repo today** — no `test` script, no vitest/jest. Stage 1's
atomic claim, lease expiry, and cancel semantics are unverifiable without one, so this is a
Stage 0 item, not a Stage 6 one.

```jsonc
// package.json
"scripts": { "test": "vitest run", "test:watch": "vitest" }
```

Minimum coverage to write **in Stage 0**, against an in-memory/temp-file SQLite DB:
- `claimNextJob` under two concurrent callers returns the job to exactly one.
- A job whose lease expires returns to `pending` with `attempts` incremented.
- `expandSweep` round-trips (already relied on by both sides; currently untested).

### 0.5 Decide: same-origin (decided)

v3.3 was internally inconsistent — `VITE_API_URL` and `credentials: 'include'` (split origin),
`SameSite=Lax` cookies (which are **not** sent on cross-site XHR), no CORS package, and a
Caddyfile defining only the API host. Three of those four cannot all be true.

**Decision: same-origin.** The server already serves `client/dist` through `@fastify/static`
(`server/src/index.ts:44`). Keep it that way.

- One hostname, e.g. `llamatoaster.com`, serving both the SPA and `/api/*`.
- No CORS, no preflights, `SameSite=Lax` works, CSRF is covered by Lax on state-changing methods.
- **Delete `VITE_API_URL`** from the client; all calls stay relative (`/api/...`), exactly as
  `client/src/api/client.ts` already does.
- Cookies: `secure` is derived, never hardcoded — `new URL(PUBLIC_URL).protocol === "https:"`.
  v3.3's hardcoded `secure: true` makes login impossible on `http://localhost` in dev.

**One deliberate, single exception, decided in §5.1:** the superadmin surface lives on its own
origin (`supervise.llamatoaster.com`), not `llamatoaster.com`. That doesn't reopen the
CORS/cross-origin mess this section just closed — it's one fixed, operator-only second hostname
with its own independent login, not a general multi-origin architecture serving arbitrary clients.
No CORS is needed there either: the admin SPA talks to `/api/admin/*` on the *same* origin it's
served from, exactly as the main app does on its own.

### 0.6 Pre-migration safety

Before **any** schema change touches the live VPS DB:

```bash
sqlite3 /data/llamatoaster.db ".backup '/backups/pre-multiuser-$(date +%F).db'"
sqlite3 /data/llamatoaster.db "PRAGMA foreign_key_check;"   # must return nothing
sqlite3 /backups/pre-multiuser-*.db "PRAGMA integrity_check;" # must return "ok"
```

Then **inspect the real data** before writing any backfill:

```sql
SELECT worker_name, COUNT(*) FROM runs GROUP BY worker_name;
```

Do not assume `'Local'`/`'Remote'`. This repo's worker names have been through two renames —
`vps-cpu`/`local-6600xt` → `Local`/`Remote` (see `scripts/backfill-worker-names.ts`) → and now
Tailscale-derived `login@hostname` (`server/src/config.ts:158`). The dev DB is empty and proves
nothing.

> **Schema rule, non-negotiable:** `server/src/db/migrate.ts:28-29` runs
> `database.exec(readFileSync("schema.sql"))` on **every process start**. Every `CREATE TABLE`
> and `CREATE INDEX` added to `schema.sql` **must** carry `IF NOT EXISTS`, or the server dies on
> the second boot. v3.3's §0.1 DDL omitted it on all eight new indexes.

---

## Stage 1 — Pull queue, single-user, on the tailnet

**Goal:** the worker pulls all work over one outbound HTTPS connection and pushes all state back.
No server→worker HTTP remains. Still one user, still on the tailnet, authenticated by a shared
secret that Stage 3 replaces with per-worker sessions. **The wire protocol does not change when
auth arrives** — only how the bearer token is obtained.

### 1.1 Job granularity: one job per **run** (decided)

v3.3 enqueued one `benchmark` job with `item_idx: 0` and had the worker execute `items[item_idx]`.
Nothing enqueued items 1..N, so a 24-combination sweep would run one combination and never
finalize. The two ways out:

| | one job per **item** | one job per **run** ✅ |
|---|---|---|
| Sweep completion | needs N enqueues + ordering | worker loops, as it does today |
| Run mutual exclusion | must be added — two users' runs would interleave on one box and collide on the llama-server port | free: one job = one run, worker executes one at a time |
| Diff from current worker | rewrite the sweep loop | keep `worker/src/index.ts`'s existing loop verbatim |
| Restart granularity | per item | per run |

**Decided: one job per run.** The payload carries the whole sweep, exactly as the current `/run`
body does. Per-item restart is not worth re-architecting the worker for; a failed item already
fails independently without stopping the run.

### 1.2 Schema

Two new tables. Note the table is called `workers` from the start (not `user_workers`) — Stage 4
adds `user_id` to it rather than renaming.

```sql
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
  last_heartbeat_at INTEGER,                 -- liveness source of truth; status is DERIVED (§1.6)
  active_job_id TEXT,                        -- what it is executing right now, or NULL
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

-- Persistent queue. Leases and attempts are what make a crashed worker
-- recoverable -- v3.3 had neither, so a lost job stuck 'claimed' forever.
CREATE TABLE IF NOT EXISTS worker_jobs (
  id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL,                    -- 'benchmark' | 'install_build' | 'download_model'
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',    -- pending | claimed | completed | failed | cancelled
  attempts INTEGER NOT NULL DEFAULT 0,
  lease_expires_at INTEGER,                  -- set on claim, extended by every heartbeat
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  progress_json TEXT,                        -- live phase/progress, pushed by heartbeat (§1.5)
  run_id TEXT,                               -- set for benchmark jobs, for reconciliation
  created_at INTEGER NOT NULL,
  claimed_at INTEGER,
  completed_at INTEGER,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_worker_jobs_worker  ON worker_jobs(worker_id, status);
CREATE INDEX IF NOT EXISTS idx_worker_jobs_lease   ON worker_jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_workers_machine     ON workers(machine_id);

-- Terminal item reports are retried by the worker (safeItemTerminal, worker/
-- src/index.ts:1081). Today a retried report inserts a SECOND set of results
-- rows. Over the public internet that stops being theoretical.
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_item ON results(run_id, idx, test_type);
```

`runs` needs one new column. It does **not** exist today — verified: `worker_id` appears nowhere
in `schema.sql` or `shared/types.ts`. Add it here, in Stage 1, because §1.12's trigger writes it:

```ts
// migrate.ts COLUMN_MIGRATIONS -- same idempotent PRAGMA-checked pattern as
// every existing entry (migrate.ts:47).
{ table: "runs", column: "worker_id", ddlType: "TEXT REFERENCES workers(id) ON DELETE SET NULL" },
```

and `Run.worker_id?: string` in `shared/types.ts`. `runs.worker_name` stays exactly as it is — a
plain snapshot the export reads directly (`server/src/routes/results.ts:140,152`), so legacy rows
with `worker_id = NULL` are harmless.

> **Check the live DB before creating `idx_results_item`** — it will fail if historical duplicates
> exist: `SELECT run_id, idx, test_type, COUNT(*) c FROM results GROUP BY 1,2,3 HAVING c > 1;`
> Dedupe (keep the earliest `created_at`) first. The dev DB has 0 results and proves nothing.

> The unique index above may fail on the live DB if historical duplicates already exist. Check
> first: `SELECT run_id, idx, test_type, COUNT(*) c FROM results GROUP BY 1,2,3 HAVING c > 1;`
> and dedupe (keep the earliest `created_at`) before creating it.

### 1.3 Two endpoints, two jobs

v3.3 had a single long-poll that the worker could only call when idle — so `status: 'busy'` was
unreachable, the machine looked offline for the whole run, and there was no channel for cancel or
progress. Split it:

```
POST /api/worker/heartbeat        every 10s, ALWAYS (idle and busy), returns fast
  body: { machine_id, state: WorkerStatePush, active_job: ActiveJobReport | null }
  200  { worker_id, control: { cancel_job_ids: string[], pause: boolean }, lease_until }

POST /api/worker/queue            only when IDLE, hangs up to 25s
  body: { machine_id, state: WorkerStatePush }
  200  QueueJob | 204 No Content | 401
```

`ActiveJobReport`:

```ts
interface ActiveJobReport {
  job_id: string;
  phase: "downloading" | "extracting" | "loading" | "benchmarking" | "finalizing";
  // Whatever is meaningful for the phase. Replaces the deleted
  // /models/download/progress polling route -- see §1.5.
  bytes?: number; total_bytes?: number;
  item_idx?: number; items_total?: number;
  detail?: string;
}
```

Why 10s heartbeat / 25s long poll: 25s keeps the hanging request comfortably under Caddy's and
any intermediary's 30s idle defaults. 10s gives cancel a worst-case 10s latency, which is fine
for a Stop button on a multi-hour job.

### 1.4 Server: queue handler (event-driven, not a 500 ms poll loop)

v3.3 ran a synchronous `better-sqlite3` transaction every 500 ms **per connected worker**, on the
event loop — 100 workers meant 200 blocking transactions/second to discover there was nothing to
do, which defeats the point of a long poll.

```ts
// server/src/queue-events.ts
import { EventEmitter } from "node:events";
export const queueEvents = new EventEmitter();     // emit(workerId) on enqueue/cancel
queueEvents.setMaxListeners(0);
```

```ts
// server/src/routes/queue.ts
const LONG_POLL_MS = 25_000;
const BACKSTOP_MS  = 5_000;   // safety net only; the event is the primary wake-up

app.post("/api/worker/queue", { config: { bodyLimit: 1_000_000 } }, async (req, reply) => {
  const worker = await authenticateWorker(req);            // Stage 1: shared secret; Stage 3: session
  const state = parseWorkerState(req.body);                // §1.8 -- VALIDATED, never a bare cast
  repo.workerRepo.recordHeartbeat(worker.id, state, null);

  // A worker that says it is busy must never be handed a second job.
  if (state.status === "busy") return reply.code(204).send();

  const claimed = repo.queueRepo.claimNextJob(worker.id);
  if (claimed) return reply.send(claimed);

  const job = await waitForJob(worker.id, req.raw);
  return job ? reply.send(job) : reply.code(204).send();
});

function waitForJob(workerId: string, raw: IncomingMessage): Promise<QueueJob | null> {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v: QueueJob | null) => {
      if (done) return;
      done = true;
      queueEvents.off(workerId, onEvent);
      raw.off("close", onClose);
      clearInterval(backstop); clearTimeout(deadline);
      resolve(v);
    };
    const tryClaim = () => { const j = repo.queueRepo.claimNextJob(workerId); if (j) finish(j); };
    const onEvent  = () => tryClaim();
    const onClose  = () => finish(null);          // client hung up -- do NOT claim a job for nobody
    const backstop = setInterval(tryClaim, BACKSTOP_MS);
    const deadline = setTimeout(() => finish(null), LONG_POLL_MS);
    queueEvents.on(workerId, onEvent);
    raw.on("close", onClose);
    tryClaim();                                    // race: a job enqueued between the sync claim and here
  });
}
```

Note `onClose` resolves `null` **without** claiming — v3.3's version could claim a job onto a
socket that had already gone away.

### 1.5 Server: heartbeat handler (state, progress, control, lease)

```ts
app.post("/api/worker/heartbeat", { logLevel: "silent", config: { bodyLimit: 1_000_000 } },
  async (req, reply) => {
    const worker = await authenticateWorker(req);
    const state  = parseWorkerState(req.body);
    const active = parseActiveJobReport(req.body);

    repo.workerRepo.recordHeartbeat(worker.id, state, active?.job_id ?? null);

    let control = { cancel_job_ids: [] as string[], pause: false };
    if (active) {
      const job = repo.queueRepo.extendLeaseAndGetFlags(active.job_id, worker.id, active);
      if (!job) {
        // The server no longer thinks this job is live (lease expired and it
        // was reassigned, or the run was deleted). Tell the worker to stop
        // rather than let two executions of the same run both report results.
        control.cancel_job_ids.push(active.job_id);
      } else if (job.cancel_requested) {
        control.cancel_job_ids.push(active.job_id);
      }
    }
    return { worker_id: worker.id, control, lease_until: Date.now() + LEASE_MS };
  });
```

`extendLeaseAndGetFlags` writes `progress_json` from the `ActiveJobReport` in the same statement.
That is what powers the UI's download/install/bench progress — **the replacement for the deleted
`/models/download/progress` route**, and it works while the worker is busy, which the v3.3 design
could not.

### 1.6 Liveness: derive, never store

v3.3 stored `workers.status` and only ever wrote it on a poll, so a powered-off machine stayed
`idle` forever. Derive it instead, so it cannot go stale:

```ts
export const HEARTBEAT_INTERVAL_MS = 10_000;
export const OFFLINE_AFTER_MS      = 35_000;   // ~3 missed heartbeats
export const LEASE_MS              = 60_000;   // ~6 missed heartbeats

export function deriveWorkerStatus(w: WorkerRow): "offline" | "busy" | "idle" {
  if (!w.last_heartbeat_at || Date.now() - w.last_heartbeat_at > OFFLINE_AFTER_MS) return "offline";
  return w.active_job_id ? "busy" : "idle";
}
```

### 1.7 The reaper — replaces the deleted self-healing path

v3.3 deleted `reconcileStaleRun`'s only two triggers (the `/health` probe in `GET /api/runs/:id`,
`server/src/routes/runs.ts, inside GET /api/runs/:id`, and the `/stop` proxy, `routes/workers.ts:258-265`) and put
nothing in their place, turning "stuck run" from self-healing into permanent.

One in-process `setInterval` in `index.ts`, same pattern as the existing polling-summary hook:

```ts
const REAP_INTERVAL_MS = 15_000;
const MAX_ATTEMPTS = 2;

function reapExpiredLeases(app: FastifyInstance): void {
  const expired = repo.queueRepo.listExpiredLeases(Date.now());   // status='claimed' AND lease_expires_at < now
  for (const job of expired) {
    if (job.attempts >= MAX_ATTEMPTS) {
      repo.queueRepo.markJobFailed(job.id, "worker stopped reporting while running this job");
      if (job.run_id) repo.reconcileStaleRun(job.run_id, "worker stopped reporting while running this run");
      app.log.warn({ job_id: job.id, run_id: job.run_id }, "job abandoned after max attempts");
    } else {
      repo.queueRepo.returnToPending(job.id);                     // attempts++, lease cleared
      queueEvents.emit(job.worker_id);
      app.log.warn({ job_id: job.id, attempts: job.attempts + 1 }, "job lease expired, requeued");
    }
  }
}
```

`reconcileStaleRun` already exists and does the right thing (`server/src/db/repo.ts:824`) — marks
unfinished items `cancelled` with an honest reason and finalizes the run as `partial`/`cancelled`
rather than lying about a user stop. It just needs a caller again.

The same scheduler also runs (from Stage 3/4 onward): expired-enrolment pruning, terminal-job
pruning (`pruneCompletedOlderThan`, 7 days), and **expired-session pruning** — which v3.3 never
had, so `sessions` would grow forever despite having an `expires_at` index.

### 1.8 Validate the worker state push

v3.3 did `const state = req.body as WorkerStatePush` — a bare cast — then `JSON.stringify`'d it
into three columns. Stage 5 then feeds `hardware_json.gpu[0].model` to **other users' LLM
context**, which turns an unvalidated worker string into a cross-tenant prompt-injection
primitive. And with Fastify's default 1 MiB `bodyLimit`, a machine with a large `model_dir` would
start getting 413s on every poll and wedge with no diagnostic.

```ts
const LIMITS = {
  maxModelFiles: 2_000,
  maxBuilds: 100,
  maxGpus: 16,
  maxString: 256,        // every free-text field: hostname, gpu model, build tag, file path
};

function parseWorkerState(body: unknown): WorkerStatePush {
  // Reject, don't coerce. Cap array lengths, truncate strings to maxString,
  // strip control characters and anything that reads as an instruction to an
  // LLM downstream. A worker is a semi-trusted client, not part of the server.
  ...
}
```

Set an explicit `bodyLimit` on both worker routes (1 MB) and make the worker cap `model_files` on
its side too, so a big model directory degrades to a truncated list rather than a wedged machine.

### 1.9 Worker: concurrent heartbeat + execution

The single change that unblocks cancel, progress, and liveness:

```ts
// worker/src/index.ts
async function workerMain(config: WorkerConfig): Promise<void> {
  startHeartbeatLoop(config);          // <-- runs FOREVER, independent of job execution
  while (true) {
    try {
      const job = await longPoll(config);       // only called while idle
      if (!job) continue;
      setActiveJob(job);                        // heartbeat now reports it every 10s
      try   { await executeJob(job, config); await reportJobDone(config, job.id); }
      catch (err) { await reportJobFailed(config, job.id, String(err)); }
      finally { clearActiveJob(); }
    } catch (err) {
      log.error("worker loop error", err);
      await sleep(10_000);
    }
  }
}

function startHeartbeatLoop(config: WorkerConfig): void {
  setInterval(async () => {
    try {
      const res = await postHeartbeat(config, collectState(), currentActiveJobReport());
      for (const jobId of res.control.cancel_job_ids) requestStop(jobId);   // sets stopRequested
      pauseRequested = res.control.pause;
    } catch { /* transient -- the reaper's lease is 6 heartbeats wide */ }
  }, HEARTBEAT_INTERVAL_MS).unref();
}
```

`requestStop`/`pauseRequested` map onto the flags the worker **already** declares
(`worker/src/index.ts:487-488`) and already checks between items (`:699`, `:705`, `:708`) — the
existing pause/stop machinery is reused verbatim, including its `safeItemTerminal(…, {status:
"cancelled"})` reporting (`:701`, `:710`). Only the trigger changes, from the inbound
`/run/stop` route (`:794`) to a heartbeat directive.

### 1.10 Run logs

`GET /api/runs/:id/log` currently proxies to the worker's `/logs`
(`server/src/routes/runs.ts:679-697`) — v3.3's route inventory omitted `/logs` entirely (it listed
14 routes; the worker has 16 — it also omitted `DELETE /llama-cpp/:tag` and `/logs`), so
the feature died silently.

**Push instead of pull:** on job completion the worker POSTs the run's log file to
`POST /api/runs/:id/log` (worker-authenticated, `Content-Type: text/plain`, capped at 5 MB,
gzip-encoded). The server stores it under `LOG_DIR/run-<id>.log.gz` and `GET /api/runs/:id/log`
serves it from disk. Same UI, no outbound HTTP, and the log survives the worker being wiped.

### 1.11 What gets deleted

**Server (`server/src/routes/runs.ts`):**
- `dispatchScheduledRun` — the function (`:555`) and **both** call sites (`:659` and `:940` —
  v3.3 said `:894`, which is stale).
- `sendRunToWorker` (`:502`), `ensureActiveBuild` (`:287`), `ensureBuildForBackend` (`:386`),
  `activateOnWorker` (`:258`), `resolveRunBuild` (`:453`), `resolveWorkerName` (`:951`).
- The import `{ getWorkerForRun, getDefaultWorker, type WorkerDef }` (`:22`). Note there is **no**
  `resolveWorker` in `config.ts` to delete — v3.3 named a function that does not exist.
- The `/health` probe + reconciliation block inside `GET /api/runs/:id` — replaced by
  §1.6/§1.7.

**Server (`server/src/config.ts`):** the whole Tailscale discovery module — `listWorkers`,
`getWorkerForRun`, `getDefaultWorker`, `parseTailscaleStatus`, `probeHealth`, the caches.
`WORKER_PORT` and `worker-reachability.ts` go with it.

**Server (`server/src/routes/workers.ts`):** every proxy route. Replaced by DB reads + queue
enqueues. `/api/hf/search` and `/api/hf/repo/*` stay (they talk to Hugging Face, not workers).

**Server (`server/src/routes/models.ts`):** `deleteFileFromAllWorkers`, `findGgufInfoFromWorkers`,
`listWorkerFiles`. Backfill reads `workers.model_files_json`; if a value is absent, leave it null
— **the server has no GGUF reader** (verified: `server/src` has only `hf.ts`; the scan lives in
`worker/src/gguf.ts`).

**Worker (`worker/src/index.ts`):** the entire inbound `createServer` (`:532`). With logs pushed,
progress pushed, and control pulled, **the worker needs no inbound HTTP at all** — a strict
improvement for a machine now sitting behind NAT on a home connection. Keep one optional
`127.0.0.1`-bound `/health` behind `WORKER_DEBUG_PORT` for local troubleshooting, off by default.

**Result:** `runs.ts` keeps read endpoints, `POST /api/runs/trigger` (creates run + enqueues), the
worker-reporting `POST /api/runs/:id/items/:idx`, and `POST /api/runs/:id/log`. Zero outbound HTTP
to workers anywhere in the server.

### 1.12 Trigger, rewritten

Fixes v3.3's mass assignment, its never-set `status`/`llama_cpp_build`, its self-contradictory
`InstalledBuild.backend` filter, its triple `getReleases()` call, and its orphaning 202.

```ts
app.post<{ Body: TriggerPayload }>("/api/runs/trigger", async (req, reply) => {
  const body = req.body;
  // ...all existing validation preserved verbatim (validateSweep, expandSweep
  // emptiness, main_gpu, isMtpDraftModel, the MTP companion checks) --
  // runs.ts, the trigger handler. None of it changes.

  const worker = repo.workerRepo.getWorker(body.worker_id);
  if (!worker) throw new BadRequestError("unknown machine");
  if (deriveWorkerStatus(worker) === "offline") {
    throw new ConflictError("that machine is offline -- start the worker and try again");
  }

  const targetBackend: Backend = body.main_gpu_backend ?? worker.backend;

  // Resolve the build ONCE. v3.3 called getReleases() three times and used a
  // different resolution for the install job than for the benchmark job, so a
  // release published between the two calls produced install=b10070,
  // benchmark=b10071 -> "build not installed".
  const installed: InstalledBuild[] = JSON.parse(worker.installed_builds_json ?? "[]");
  const resolved = await resolveBuildForRun(worker, targetBackend, installed);
  if ("error" in resolved) throw new BadRequestError(resolved.error);

  // EXPLICIT field list. Never spread the request body into an insert -- a
  // client-chosen runs.id is an invitation in a multi-tenant app.
  const run: Run = {
    id: uuid(),
    worker_id: worker.id,
    worker_name: worker.display_name,        // point-in-time snapshot; the export reads this
    llama_cpp_build: resolved.tag,           // set NOW, not "filled on dispatch" by nobody
    llama_cpp_backend: targetBackend,
    backend_device_name: resolved.deviceName,
    model_id: body.model_id,
    config: { model_id: body.model_id, mtp_model_id: mtpModel?.id,
              main_gpu: body.main_gpu, main_gpu_backend: body.main_gpu_backend,
              sweep: body.sweep } as RunConfig,
    // REUSE the existing status. `RunStatus` (shared/types.ts:290) is already
    // "running" | "scheduled" | "done" | "partial" | "failed" | "cancelled",
    // and "queued" is a *RunItemStatus*, not a run one -- inventing it here
    // would collide conceptually and force changes in StatusPill.tsx's
    // RUN_STATUS_TONE map (:25) and every status filter.
    status: "scheduled",                     // flipped to 'running' on claim (§1.13)
    started_at: Date.now(),
  };
  repo.createRun(run);
  repo.createRunItems(run.id, expandSweep(body.sweep));

  if (!resolved.alreadyInstalled) {
    repo.queueRepo.enqueueJob(worker.id, { type: "install_build", payload: resolved.installPayload });
  }
  repo.queueRepo.enqueueJob(worker.id, {
    type: "benchmark", run_id: run.id,
    payload: { run_id: run.id, model, mtp_model: mtpModel, sweep: body.sweep,
               main_gpu: body.main_gpu,
               llama_cpp_build: resolved.tag, llama_cpp_backend: targetBackend },
  });
  queueEvents.emit(worker.id);
  return reply.code(201).send({ run });
});
```

`resolveBuildForRun` matches installed builds **by tag and asset name**, never by
`InstalledBuild.backend` — that field does not exist (`shared/types.ts:669`). Reuse the existing
`assetMatchesWorker(b.asset_name, platform, arch, backend)` helper, which is exactly what
`ensureBuildForBackend` already does today (`routes/runs.ts:403`).

The "worker has never checked in" case is gone: §1.12 rejects an offline machine up front with a
clear message, instead of v3.3's 202 that created a run with no job and no way to retry.

### 1.13 Run status lifecycle

| Event | `runs.status` |
|---|---|
| trigger | `scheduled` (existing value — see §1.12) |
| worker claims the benchmark job | `running` + `markRunRunning(started_at)` |
| all items terminal | `done` / `partial` / `failed` (existing `finalizeRun`, unchanged) |
| user stops | items `cancelled` → `cancelled` |
| lease reaped past max attempts | `reconcileStaleRun` → `partial`/`cancelled` |

`claimNextJob` sets `runs.status='running'` in the **same transaction** as the claim, so the two
can never disagree.

### 1.14 Stop / pause / resume

```
POST /api/runs/:id/stop     → set cancel_requested on the run's claimed job; emit
POST /api/runs/:id/pause    → set workers.pause_requested
POST /api/runs/:id/resume   → clear it
```

Delivered on the next heartbeat (≤10s). The UI shows "stopping…" until the worker's terminal
reports land — an honest state rather than a button that looks like it did nothing. If the
machine is offline, the endpoint reconciles the run immediately instead of waiting for the lease.

These target the **run**, not the machine — a change from today's worker-scoped
`/api/workers/:name/stop`, and the right shape once one account owns several machines.

### 1.15 Stage 1 auth (temporary)

`WORKER_SHARED_TOKEN` env, checked as `Bearer` on both worker routes with `safeEqual`. The worker
reads it from its `config.json`. Stage 3 replaces this with per-worker sessions — the protocol,
the payloads, and every handler above stay identical; only `authenticateWorker()` changes.

### 1.16 Shared-type and client changes Stage 1 forces

Machines stop being identified by name and start being identified by id. That ripples further
than it looks — everything below is name-keyed today:

| Where | Today | Stage 1 |
|---|---|---|
| `shared/types.ts:643` | `TriggerPayload.worker_name: string` | `worker_id: string` |
| `client/src/pages/NewRun.tsx:700` | sends `worker_name: workerName` | sends `worker_id` |
| `shared/types.ts:292` `Run` | `worker_name` only | add `worker_id?: string` (snapshot name stays) |
| `client/src/api/useWorkerStatus.ts` | **entirely** keyed on `worker.name` (`:24,27,28,32,41,43`) | re-key on `worker.id`; drop the per-worker `getWorkerLlamaCpp` fetch — state now comes from one `/api/workers` read of cached heartbeat data |
| `client/src/api/client.ts` | 9 methods take `workerName` and build `/api/workers/:name/...` | take `workerId`; most disappear entirely with the proxy routes (§1.11) |
| `worker/src/index.ts:308` | `ModelDirFile` is worker-local | promote to `shared/types.ts` so the server can type the state push |

`useWorkerStatus` is the one to look at first: it currently fans out a live
`getWorkerLlamaCpp(name)` per worker on every mount. Under the pull model there is nothing live to
fetch — a single `GET /api/workers` returns every machine with its cached hardware, builds, model
files, derived status, and current job. That is a simplification, not just a port.

### 1.17 Stage 1 exit criteria

- [ ] A 20+ combination sweep completes end-to-end, pull-only, with no server→worker HTTP.
- [ ] `netstat` on the worker shows **no** listening port (except the opt-in debug one).
- [ ] Stop mid-run: run finalizes `cancelled` within 15s, machine returns to `idle`.
- [ ] `kill -9` the worker mid-run: within ~90s the run reconciles to `partial`, the machine reads
      `offline`, and the job is requeued once then failed.
- [ ] Machine reads `busy` and shows live progress for the whole of a long run.
- [ ] Multi-GB model download shows a live byte counter.
- [ ] Two concurrent `/api/worker/queue` calls never receive the same job (test, §0.4).
- [ ] Run log downloads for a failed run.

---

## Stage 2 — Auth & sessions

### 2.1 Identity: `users` is provider-agnostic from day one

**The username generator is deleted** (unchanged conclusion from the earlier draft — see the
reasoning box below) — but the identity table underneath it changes shape, because of a real
requirement: **GitHub today, Google (or others) later, on the same account.**

If `users` hardcodes `github_id` as its identity column — which the previous draft of this plan
did — adding a second provider means one of: a nullable `google_id` column bolted on later (and a
third provider makes it worse), an `OR`-chained lookup in every auth query, and an unanswered
question about what happens when someone who already has a GitHub-based account tries to sign in
with Google. None of that is hard, but all of it is a live-data migration on a running multi-tenant
app, which is a materially worse time to do it than now, before any user exists.

**Fix: split "account" from "how you proved who you are."** Standard federated-identity shape —
the same one Auth0/Clerk/NextAuth use for exactly this reason:

```sql
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,                     -- UUID, internal. THE account. Provider-independent.
  display_name TEXT,                       -- from whichever provider first supplied one; user-editable
  avatar_url TEXT,                         -- from whichever provider first supplied one
  share_benchmarks INTEGER NOT NULL DEFAULT 1,  -- §5.4; makes the "opt-in" claim true
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_login_at INTEGER
);

-- One row per (provider, that provider's account) a user has linked. GitHub
-- today; adding Google later is a new row shape, not a new column, and not a
-- migration of anything that already exists.
CREATE TABLE IF NOT EXISTS identities (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,                  -- 'github' | 'google' | ...
  provider_user_id TEXT NOT NULL,          -- THE identity within that provider. Immutable there.
  provider_login TEXT,                     -- display login/handle; refreshed on every login
  created_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_identities_provider ON identities(provider, provider_user_id);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);
```

`users.id` is what every other table (`sessions`, `workers`, `runs`, …) references — none of them
ever need to know which provider a user signed in with. That boundary is what makes Stage 3's
worker enrolment (§3) automatically work for Google the day it ships, with zero changes there —
see the closing note below.

**`github_email` is not stored, and `user:email` is dropped from every provider's OAuth scope.**
Nothing in the app sends mail. Removing it cuts consent friction on the authorize screen *and*
removes PII you would otherwise be liable for. Scope is `read:user` (GitHub) / `openid profile`
(Google) — profile only, no email, on every provider. (If run-completion email is built later, ask
for it then, with a reason the user can see.)

There is no `is_superadmin` column — see §5.1, which also moves off a bare numeric GitHub ID for
the same reason this table did.

> **Why the username generator is still deleted, restated briefly:** v3.3's `adjective_noun`
> scheme cost a wordlist, a UNIQUE-retry loop, a collision test, and a name snapshotted permanently
> into `runs.worker_name` — and the result was visible **nowhere**, since §8.0 makes the UI
> own-data-only and the AI is forbidden from surfacing it. `identities.provider_login` (whichever
> provider a user actually signed in with) is what the UI displays; it is **never** a join key —
> only `users.id` is. Machines get a user-chosen `display_name` defaulting to their own hostname
> (§3.4), never derived from any provider's login.

### 2.2 Sessions: the token is never stored

v3.3's schema made `sessions.id` **be** the session token, stored in plaintext, right beside its
own SHA-256. Any DB read — a backup, a stray `.db-wal` (there are three such files in this repo
root right now), a `sqlite3 .backup` artefact from the deploy section — was total session takeover
for every user and every 90-day worker session.

```sql
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,                     -- SURROGATE UUID. Never sent to any client.
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL,                -- sha256(access token). The token itself is never stored.
  refresh_hash TEXT,                       -- sha256(current refresh token), workers only
  prev_refresh_hash TEXT,                  -- sha256(previous), for replay detection (§2.5)
  expires_at INTEGER NOT NULL,
  is_worker INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT,                          -- set when is_worker
  label TEXT,                              -- "Chrome on Windows" / machine display name, for Settings
  last_seen_at INTEGER,
  created_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_token   ON sessions(token_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_refresh ON sessions(refresh_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_worker  ON sessions(worker_id);
```

Three fixes in that DDL beyond the surrogate id:

- **`idx_sessions_token` exists.** v3.3 indexed `user_id`, `expires_at` and `worker_id` but not
  `token_hash` — the one column read on *every single request*. Every authenticated call was a
  full table scan.
- **It is UNIQUE**, so a duplicate-insert bug can never produce two live sessions for one token.
- **`ip_hash` / `user_agent` are gone.** v3.3 declared them and never populated them. Don't collect
  what you don't use. `label` (derived server-side from the UA at creation, not stored raw) is what
  the Settings page actually needs.

`sessionRepo.create` returns the token to the caller and stores only its hash:

```ts
create(userId: string, opts: { isWorker?: boolean; workerId?: string; label?: string }) {
  const token = generateSessionId();
  const refresh = opts.isWorker ? generateRefreshToken() : null;
  const now = Date.now();
  const expiresAt = now + (opts.isWorker ? 90 : 30) * 24 * 3600 * 1000;
  getDb().prepare(`INSERT INTO sessions
      (id, user_id, token_hash, refresh_hash, expires_at, is_worker, worker_id, label, last_seen_at, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .run(uuid(), userId, hashToken(token), refresh && hashToken(refresh),
         expiresAt, opts.isWorker ? 1 : 0, opts.workerId ?? null, opts.label ?? null, now, now);
  return { token, refresh, expiresAt };     // the ONLY time these values exist outside the client
}
```

### 2.3 Middleware

```ts
const PUBLIC_PATHS = new Set([
  "/health", "/api/auth/status",
  "/api/device/start", "/api/device/token",             // worker enrolment -- see §3
  "/auth/:provider", "/auth/:provider/callback", "/auth/logout",
]);

export async function authMiddleware(req: FastifyRequest, reply: FastifyReply): Promise<void> {
  const routeUrl = req.routeOptions?.url;
  // FAIL CLOSED. v3.3 fell back to req.url, which carries the query string, so
  // "/api/auth/status?x=1" missed the public set -- and worse, a fallback that
  // can be influenced by the client is the wrong default for an allowlist.
  if (routeUrl === undefined) return;                       // unmatched route -> the 404 handler owns it
  if (PUBLIC_PATHS.has(routeUrl)) return;
  if (req.method === "GET" && !routeUrl.startsWith("/api/")) return;   // SPA static assets

  const token = extractToken(req);
  if (!token) throw new UnauthorizedError("no session");

  const session = repo.sessionRepo.getByTokenHash(hashToken(token));
  if (!session || session.expiresAt < Date.now()) throw new UnauthorizedError("session expired");

  const user = repo.userRepo.getUser(session.userId);
  if (!user) throw new UnauthorizedError("user not found");

  // Superadmin is evaluated PER REQUEST from env, never read from a DB column
  // (§5.1) -- and now from the user's LINKED IDENTITIES, not a single provider
  // id, since a superadmin's GitHub and Google accounts (once linked, §2.4)
  // must both grant the same access. One extra indexed lookup per request on
  // a table that will hold a handful of rows per user; not worth caching.
  const identities = repo.userRepo.getIdentities(user.id);
  const isSuperadmin = identities.some((i) => isSuperadminIdentity(i.provider, i.providerUserId));
  (req as AuthenticatedRequest).user = { ...user, isSuperadmin };
  (req as AuthenticatedRequest).session = session;

  touchSession(session);   // sliding expiry, throttled to one write per hour (§2.5)
}
```

Registered as `app.addHook("preHandler", authMiddleware)` in `index.ts` — v3.3 described the
middleware but never wired it.

`extractToken`: `Authorization: Bearer` first (workers, API), then the `lt_session` cookie
(browser). Behind `AUTH_ENABLED=false` the middleware is not registered at all, so Stage 1 and 2
can be deployed independently.

`/auth/:provider` and `/auth/:provider/callback` are Fastify **route patterns**, matched against
an explicit provider allowlist inside the handler (§2.4) — never an open passthrough that resolves
`:provider` from user input into a URL.

### 2.4 OAuth: one adapter per provider, GitHub is the first

Every provider-specific fact — authorize URL, token exchange, profile endpoint, how to read the
provider's own stable id out of the profile response — lives in one small object. The route
handler underneath never changes when a provider is added:

```ts
interface OAuthProvider {
  id: "github" | "google";                    // extend the union when adding one
  authorizeUrl: string;
  scope: string;                               // profile only, never email (§2.1)
  tokenUrl: string;
  profileUrl: string;
  clientId: string; clientSecret: string;       // read from env: `${PROVIDER}_CLIENT_ID` etc.
  // Maps that provider's own profile shape onto the one shape the rest of the
  // app deals with -- THIS is the only part that's provider-specific past this point.
  mapProfile(json: unknown): { providerUserId: string; login: string; avatarUrl: string | null } | null;
}

const PROVIDERS: Record<string, OAuthProvider> = {
  github: {
    id: "github",
    authorizeUrl: "https://github.com/login/oauth/authorize",
    scope: "read:user",                         // NOT user:email -- see §2.1
    tokenUrl: "https://github.com/login/oauth/access_token",
    profileUrl: "https://api.github.com/user",
    clientId: process.env.GITHUB_CLIENT_ID!, clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    mapProfile: (j) => {
      const p = j as { id?: number; login?: string; avatar_url?: string };
      return typeof p.id === "number" && p.login
        ? { providerUserId: String(p.id), login: p.login, avatarUrl: p.avatar_url ?? null }
        : null;
    },
  },
  // google: { ... } -- added later. Same shape, different URLs/scope
  // ("openid profile"), different mapProfile (Google's stable id is the
  // `sub` claim, not `id`). Nothing else in this file changes.
};

app.get<{ Params: { provider: string } }>("/auth/:provider", async (req, reply) => {
  const provider = PROVIDERS[req.params.provider];
  if (!provider) return reply.code(404).send({ error: "unknown provider" });
  const state = generateState();
  reply.setCookie("oauth_state", state, { httpOnly: true, secure: COOKIES_SECURE, sameSite: "lax", maxAge: 600 });
  // linking=1 set only by the authed Settings flow below -- carries the
  // "attach to my existing account" intent through the redirect round trip.
  const linking = req.query.link === "1" && (req as AuthenticatedRequest).user;
  if (linking) reply.setCookie("oauth_link_user", (req as AuthenticatedRequest).user.id, { httpOnly: true, secure: COOKIES_SECURE, sameSite: "lax", maxAge: 600 });
  const params = new URLSearchParams({ client_id: provider.clientId, scope: provider.scope, state,
    redirect_uri: `${PUBLIC_URL}/auth/${provider.id}/callback` });
  return reply.redirect(`${provider.authorizeUrl}?${params}`);
});

app.get<{ Params: { provider: string } }>("/auth/:provider/callback", async (req, reply) => {
  const provider = PROVIDERS[req.params.provider];
  if (!provider) return reply.code(404).send({ error: "unknown provider" });
  const { code, state } = req.query as { code?: string; state?: string };
  const cookieState = req.cookies?.oauth_state;
  const linkUserId = req.cookies?.oauth_link_user ?? null;
  reply.clearCookie("oauth_state", { path: "/" }); reply.clearCookie("oauth_link_user", { path: "/" });
  if (!code || !state || !cookieState || !safeEqual(state, cookieState)) {
    return reply.redirect("/login?error=oauth_state");      // constant-time compare
  }

  const tokenRes = await fetch(provider.tokenUrl, { ... });
  const tokenJson = await tokenRes.json() as { access_token?: string; error?: string };
  // GitHub returns HTTP 200 with {error: "bad_verification_code"} -- naively
  // destructuring access_token here turns a replayed code into a 500 on a
  // NOT NULL violation three calls later, instead of a clean redirect.
  if (!tokenJson.access_token) return reply.redirect("/login?error=oauth_exchange");

  const profileRes = await fetch(provider.profileUrl, { headers: { authorization: `Bearer ${tokenJson.access_token}` } });
  if (!profileRes.ok) return reply.redirect("/login?error=oauth_profile");
  const mapped = provider.mapProfile(await profileRes.json());
  if (!mapped) return reply.redirect("/login?error=oauth_profile");

  const user = linkUserId
    ? repo.userRepo.linkIdentity(linkUserId, provider.id, mapped)   // Settings "connect another account" (below)
    : repo.userRepo.upsertByIdentity(provider.id, mapped);           // normal login

  const { token } = repo.sessionRepo.create(user.id, { label: describeUserAgent(req) });
  reply.setCookie("lt_session", token, {
    httpOnly: true, secure: COOKIES_SECURE, sameSite: "lax", path: "/", maxAge: 30 * 24 * 3600,
  });
  return reply.redirect(linkUserId ? "/settings?linked=1" : "/");
});
```

`upsertByIdentity(provider, {providerUserId, login, avatarUrl})` looks up
`identities WHERE provider = ? AND provider_user_id = ?`; on a hit it updates `provider_login`
(mutable, refreshed every login) and returns the linked `users` row; on a miss it creates a fresh
`users` row **and** its first `identities` row in one transaction. **No username generation, no
retry loop, no collision handling** — the whole §5.4-of-v3.3 collision-safety machinery doesn't
exist here because nothing is generated.

`COOKIES_SECURE = new URL(PUBLIC_URL).protocol === "https:"` (§0.5).

**Account linking — the part that's easy to get wrong.** The naive move is auto-linking by email:
sign in with Google, look up a `users` row that has a GitHub identity with a matching email,
merge automatically. **Don't.** OAuth providers don't verify email ownership consistently, and
email reuse/typosquatting turns "matching email" into an account-takeover primitive — someone
signs up with a lookalike or reused address on a different provider and inherits your account.

The safe version requires proving ownership of **both** sides, in the same session:

- Logging in with a provider that has never been seen before **always creates a new, separate
  account.** No silent merging, ever.
- **Settings → Connected accounts → "+ Connect Google"** re-runs the exact same
  `/auth/google?link=1` flow, but *while already authenticated*. The `oauth_link_user` cookie
  carries that intent through the redirect; the callback calls `linkIdentity` instead of
  `upsertByIdentity`, attaching the new provider to the **current session's** account rather than
  creating or finding one. This is standard: the user proves the existing account first (their
  live session), then proves the new identity second (that provider's own OAuth flow) — ownership
  of both is established before anything is merged.
- A user who signs up via Google, then later clicks "Sign in with GitHub" **without** having
  linked it in Settings, gets a second, separate account — same behaviour every major SSO consumer
  has. A one-line hint on the login page ("already have an account? link providers in Settings
  after signing in") is a nice-to-have, not a requirement.

**Why this doesn't touch worker enrolment at all.** Stage 3's device flow authenticates purely
against `sessions.user_id → users.id` — it has never once read `identities`, `github_id`, or
anything provider-specific. That was already true in the browser-first draft (the approve handler
checked `worker.user_id === req.user.id`, both UUIDs) and stays true here. **Adding Google support
later is a new `PROVIDERS` entry and a login button — zero changes to Stage 3.** A worker should
never attempt an OAuth login itself: it is a headless process with no browser to complete an
interactive consent screen, and it doesn't need one — it only ever needs to be vouched for by an
already-authenticated human, which is exactly what the device code in §3 does.

### 2.5 Session lifetime, refresh, and revocation

**Browser sessions do not use refresh tokens.** v3.3 set an `lt_refresh` cookie that nothing ever
read, and `/api/auth/refresh` called `extractSessionId()` — which returns the *access* cookie for a
browser, so `getByRefreshHash(hash(access_token))` never matched and browser refresh was
permanently broken. Delete the cookie and the code path.

Instead: **sliding expiry.** `touchSession` extends `expires_at` to `now + 30d`, throttled to at
most one write per hour per session. An active user is never logged out; an idle one expires in 30
days.

**Worker sessions do rotate**, because they are long-lived bearer tokens on someone's home machine:

```ts
app.post("/api/auth/refresh", async (req, reply) => {
  const presented = bearerToken(req);                      // dedicated read, NOT extractToken
  if (!presented) throw new UnauthorizedError("no refresh token");
  const hash = hashToken(presented);

  // Replay detection: presenting the PREVIOUS refresh token means the token was
  // captured and reused. Kill the whole session rather than issue new tokens.
  const replayed = repo.sessionRepo.getByPrevRefreshHash(hash);
  if (replayed) {
    repo.sessionRepo.revokeById(replayed.id);
    app.log.warn({ session: replayed.id, worker: replayed.workerId }, "refresh token replay -- session revoked");
    throw new UnauthorizedError("refresh token reuse detected");
  }

  const session = repo.sessionRepo.getByRefreshHash(hash);
  if (!session || session.expiresAt < Date.now()) throw new UnauthorizedError("refresh token expired");

  // sessions.id is STABLE across rotation. v3.3 rotated the primary key, which
  // orphaned workers.session_id (written at approval) after the first refresh.
  const rotated = repo.sessionRepo.rotate(session.id);     // new token + refresh, prev_refresh_hash = old, expires_at += 90d
  return { session_token: rotated.token, refresh_token: rotated.refresh, expires_at: rotated.expiresAt };
});
```

**Session management** (the backend v3.3's "Settings page (revoke sessions)" never had):

```
GET    /api/sessions          → [{ id, label, isWorker, createdAt, lastSeenAt, current }]
DELETE /api/sessions/:id      → revoke one (own sessions only)
POST   /api/sessions/revoke-all → revoke every session except the current one
```

Revoking a worker's session marks the machine as needing re-enrolment (§3.5) rather than deleting
its history.

### 2.6 Rate limits and the AI budget

v3.3 rate-limited `/auth/*` and `/device/token` and left the expensive endpoint wide open.

| Route | Limit | Keyed on |
|---|---|---|
| `/auth/:provider*` | 10/min | IP |
| `/api/device/start` | 30/min | IP — see §3.4 for why this needs its own limit now |
| `/api/device/token` | 60/min | **IP** — and separately 12/min per device code |
| `/api/device/approve` | 20/min | authenticated user |
| `/api/runs/trigger` | 30/hour | user |
| `/api/models` (POST), `/api/models/*/backfill-*` | 60/hour | user |
| `/api/worker/heartbeat`, `/api/worker/queue` | 20/min | worker session |
| **`/api/ai/chat`** | **see below** | user |

> **Why per-IP on `/api/device/token`:** v3.3's checklist said "10/min **per device_code**". An
> attacker brute-forcing codes sends a *different* code every request, so a per-code limit is
> arithmetically a no-op. The per-code limit is still worth having — it stops a broken worker from
> hammering — but it is not the brute-force control.

**`/api/ai/chat` is the highest-probability launch-day incident** and v3.3 never mentioned it: an
unmetered proxy to a paid LLM, with a client-controlled message array, up to 4 tool rounds per
request, and no cap on message count or size (`server/src/routes/ai.ts:333`). One signed-up user
can drain the operator's budget in an afternoon.

```ts
const AI_LIMITS = {
  perUserPerHour: 30,
  perUserPerDay: 150,
  globalPerDay: Number(process.env.AI_GLOBAL_DAILY_CAP ?? 2000),  // circuit breaker
  maxMessages: 40,
  maxTotalChars: 60_000,          // whole conversation, enforced server-side
  maxToolRounds: 4,               // already enforced
};
```

Track usage in a small `ai_usage(user_id, day, hour, count)` table. On the global cap, return a
clear 503 ("the assistant is over its daily budget, try tomorrow") rather than failing opaquely.
Surface the user's remaining quota in the chat panel so the limit is visible before it bites.

### 2.7 The AI system prompt moves server-side

Today the system prompt is built **in the browser** — `client/src/components/ChatPanel.tsx:265`:

```tsx
{ role: "system", content: `${SYSTEM_PROMPT}${memoryBlock}\n\n${ctx}` }
```

and `/api/ai/chat` forwards whatever `messages` it is handed. Any user can `curl` the endpoint
with their own system message, or none. That makes Stage 5's entire anonymisation instruction
(§5.4) unenforceable, and it also means v3.3's §4.3 was editing a server-side
`buildContextSnapshot` that **does not exist** — context building lives in
`client/src/api/aiContext.ts:106`.

Changes to `server/src/routes/ai.ts`:

```ts
// 1. STRIP every client-supplied system message. The server owns the prompt.
const userMessages = messages.filter((m) => m.role !== "system");
if (userMessages.length !== messages.length) {
  req.log.warn({ user: req.user.id }, "ai chat: client-supplied system message discarded");
}
// 2. Server builds the context from the caller's OWN data (Stage 4 §4.6).
const conversation = [
  { role: "system", content: SERVER_SYSTEM_PROMPT + "\n\n" + await buildContextSnapshot(req.user.id) },
  ...userMessages,
];
// 3. Enforce AI_LIMITS.maxMessages / maxTotalChars before calling the provider.
```

The client keeps `SYSTEM_PROMPT` only as the source that gets **moved** to the server; the
per-user AI "memory" block (`client/src/api/aiStorage.ts`) stays client-side but is passed as a
`user` message, not a `system` one, so it can never override server instructions.

### 2.8 User journeys, end to end

Stages 2 and 3 are specified protocol-first, by necessity — but nobody experiences a protocol.
This walks the same machinery as three screen-by-screen stories, and calls out the couple of
places writing them out surfaced a gap.

**Sign-up and log-in are the same flow — there is no separate registration.** That's the point of
letting the identity provider *be* the account system (§2.1): nothing distinguishes "creating an
account" from "logging in" until the callback handler decides which one just happened, and the
user never sees that decision.

1. An unauthenticated visitor hits any route → the SPA's boot check (`GET /api/auth/status`)
   comes back `{ user: null }` → redirected to `/login`.
2. `/login` shows one button per configured provider — today, "Continue with GitHub"; the day
   `PROVIDERS.google` (§2.4) exists, a second button appears with no other change to this page.
   Nothing to type: no username, no password, no email.
3. Click → full-page navigation to `/auth/github` → server sets the `oauth_state` cookie →
   redirect to GitHub's own consent screen, which asks for profile access only (no email scope,
   §2.1) — a returning user who already authorized the app doesn't see this screen again at all,
   GitHub just bounces them straight back.
4. `/auth/github/callback` exchanges the code, fetches the profile, and calls `upsertByIdentity`
   (§2.4): a `provider_user_id` GitHub has never sent before creates a `users` row and its first
   `identities` row in one transaction — that *is* sign-up, with no separate step. A recognized one
   just updates `provider_login` and returns the existing account.
5. Session cookie set, redirect to `/`. First-ever login lands on the empty-state Dashboard
   (§5.2): "No machines yet — connect your first machine, about a minute." Every later login lands
   on the normal four-tile Dashboard.
6. Sessions slide (§2.5) — an active user is never logged out. Thirty days idle, and step 1 repeats
   with nothing lost: the account and every machine on it are exactly as they were.

**Worker registration — walking §3.1's protocol as a story instead of a sequence diagram:**

1. From the Dashboard's empty state, or "+ Add machine" once there's at least one, the user lands
   on one screen showing the OS-switched install command (§3.1) and, just below it, an empty code
   field.
2. They run the command on the GPU box. It prints an 8-character code and a URL. Nothing about
   *this* step required them to be logged in yet, or even to have an account — see the note below.
3. Back in the browser — same screen, no navigation — they type the 8 characters into the field
   that was already waiting there. It resolves to a confirm card: hostname, platform, GPU if
   already detected, and the phishing-mitigation line from §3.2. One click, "Approve."
4. Within one heartbeat interval (≤10s) the card flips to "connected," and the machine now appears
   in the strip on the Dashboard with its default `display_name` (the hostname) editable inline.

**Two properties worth stating explicitly, because they're easy to get backwards:**

- **Sign-up and machine enrolment have no required order.** Because `/api/device/start` is public
  and unauthenticated (§2.3, §3.4), someone can run the install command on a fresh GPU box *before*
  ever creating a llamatoaster account — the row just sits pending. Signing up via GitHub for the
  first time and landing straight on an approval card for the machine they already started is a
  legitimate, and arguably the most natural, path through this — not a workaround.
- **Re-enrolment is the identical story, not a different one.** A "Reconnect" affordance on an
  offline or session-expired machine's card does nothing but re-show the same OS-switched command
  from step 1 of the second flow — §3.4's `getByMachineId` reuse means the *server* already knows
  this is a reconnection and treats it as one (no re-approval required, history preserved); the
  *user* just sees "run this again."

---

## Stage 3 — Machine enrolment (one-click approve)

### 3.1 Worker-first, not browser-first — and why that's the actual fix

v3.3's flow: create worker in the UI → copy a 64-char `device_code` → paste into a `curl | bash` →
navigate to `/device` → **paste the same 64-char code again** → approve. Its stated rationale was
that the device code is "the only binding that proves the machine belongs to the account that
created it" — but the server handed **both** codes to the creator's browser at creation time, so
the browser echoing one back proved nothing the session didn't already prove. That flaw survived
into this plan's first draft too: `POST /api/workers` created the row **already owned**
(`user_id = req.user.id`) before the machine had done anything, which made the later "Approve"
click ceremonial — a gate on a decision that was already made.

**The actual fix is inverting who moves first, not just trimming the copy-paste.** RFC 8628 has
the *device* — not the browser — start the flow precisely because that is the only way the
approval click ends up meaning something: the row has no owner until a human, looking at what
they're approving, decides it does.

```
1. User runs ONE constant command on the GPU box — no flags, no per-user
   templating, nothing copied from the browser. The "Add machine" screen
   shows BOTH forms side by side (an OS switcher, not a guess) -- the repo
   already ships and maintains both installers (worker/bootstrap.sh,
   worker/bootstrap.ps1 -- the latter touched as recently as this plan, see
   "Self-heal blocked PowerShell script execution in worker bootstrap"), so
   this is presenting what already exists, not building a second installer:

     macOS/Linux:  curl -fsSL https://llamatoaster.com/install.sh | sh
     Windows:      irm https://llamatoaster.com/install.ps1 | iex

2. The worker generates (or reuses, see §3.4) a machine_id, then calls the
   PUBLIC, unauthenticated:

     POST /api/device/start  { machine_id, hostname, platform, arch }
     -> { device_code, user_code: "ABCD-EFGH", verification_uri: "/device",
          interval: 5, expires_in: 900 }

   and prints:

     To connect this machine, visit https://llamatoaster.com/device
     Code: ABCD-EFGH   (expires in 15 minutes)

3. Worker polls POST /api/device/token with device_code every 5s -> 400
   { error: "authorization_pending" } until approved.

4. The user types ABCD-EFGH into a code field. In the common case this is
   the SAME "Add machine" screen from step 1 -- no navigation, just a field
   under the install command that appears the moment they start typing.
   /device is the same component at a bookmarkable URL, for the case where
   enrolment starts somewhere the browser wasn't already open (a headless
   box set up by SSH, a code relayed from someone else). Either way it calls
   GET /api/device/status?user_code=ABCD-EFGH and shows:

     ┌──────────────────────────────────────────────┐
     │  🖥️  gpu-tower                                │
     │      Linux · x64 · 1× NVIDIA RTX 4090        │
     │                                              │
     │      Only approve a code YOU generated,      │
     │      on a machine you trust.                 │
     │                                              │
     │              [ Approve ]  [ Deny ]           │
     └──────────────────────────────────────────────┘

5. ONE CLICK -> POST /api/device/approve { user_code }  (from the approver's
   session). Sets workers.user_id = req.user.id -- the FIRST time anything
   sets it.
6. Worker's next poll returns its session + refresh token. Page shows
   "connected".
```

**One copy-paste — into the terminal, of a command that never changes — and one click.** No
per-user templating, no second secret, and the thing that crosses from the machine's screen to the
browser is eight low-value characters, not a 64-char bearer credential.

### 3.2 The honest cost, and why it's acceptable here

Moving first-mover from browser to worker gains something and gives something up; both need
stating, not just the first half.

**Gains:** nothing sensitive ever crosses browser→terminal (v3.3's actual weak point — a stolen
64-char code was a live credential). The install command becomes a genuine constant, safe to put
in a README, with no modal needed to generate it. And the approval click now *is* the identity
binding, not a rubber stamp on one made earlier.

**Gives up:** because the row is unowned at creation, `/api/device/approve` can only look up
`user_code` **globally** among pending enrolments — there is no `user_id` yet to scope the lookup
to. That opens a narrow, known class of risk: **device-code phishing.** If someone shoulder-surfs,
screen-shares, or otherwise sees a `user_code` meant for someone else, they can type it into their
*own* llamatoaster session within the 15-minute window and approve it — silently attaching a
stranger's physical machine to their own account. This is a documented risk class for OAuth device
flow generally (it's why Microsoft and Okta have both written about it for high-value SSO
deployments), and it is not something short TTL or rate limiting eliminates — both only shrink the
window, they don't close it.

**Why it's the right trade for this product specifically, not universally:** the blast radius here
is low. A hijacked machine lets someone else's benchmark runs execute on it and shows up as a
"machine" on the wrong account — mis-attributed compute, not exposed secrets, financial access, or
personal data. That is a materially different risk profile than the OAuth-device-flow-for-SSO case
those write-ups are about. The mitigations below are the accepted, standard set for this risk
class, sized to the actual stakes:

- The confirm card shows hostname/GPU/platform (§3.1 step 4) so an *accidental* mismatch is caught
  by the approver noticing "that's not my machine" — this doesn't stop a deliberate hijack (an
  attacker wants the machine, they won't self-report), but it does stop the far more common
  accident of two people enrolling machines around the same time and mixing up which code is
  which.
- The confirm card also carries an explicit warning line ("only approve a code you generated
  yourself, on a machine you trust") — cheap, and it's the standard mitigating copy for this
  attack class.
- 15-minute TTL, and `/api/device/approve` is rate-limited per authenticated user (§2.6) —
  narrows the window and the guess budget; neither is the security boundary, both are depth.
- `user_code` has a **unique partial index among pending rows** (§3.4) — no two concurrently
  pending enrolments can collide on the same code, so there is never ambiguity about which machine
  a correct guess resolves to.
- A "Reconnect" flow (§3.5) reuses the same machine's existing history, so a user who suspects
  their code leaked can simply re-run the install command and get a fresh code — the abandoned one
  expires in the remaining minutes and is never usable again.

If this product's stakes ever change — say, workers start handling something more sensitive than
benchmark execution — revisit toward Option A (a pre-authenticated Settings-issued token, no
approval step at all, closing this vector entirely at the cost of losing the "look before you
approve" moment) or a browser-first flow with the ceremony v3.1 avoided. Neither is needed today.

### 3.3 Schema

```sql
ALTER TABLE workers ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE;   -- NULL until approved
ALTER TABLE workers ADD COLUMN enrolment_code_hash TEXT;    -- sha256(device_code) -- worker-held, never shown to a human, NOT plaintext
ALTER TABLE workers ADD COLUMN user_code TEXT;              -- human-facing, the PRIMARY approval path now (not a fallback)
ALTER TABLE workers ADD COLUMN enrolment_expires_at INTEGER;
ALTER TABLE workers ADD COLUMN approved_at INTEGER;
CREATE INDEX IF NOT EXISTS idx_workers_enrolment ON workers(enrolment_code_hash);
CREATE INDEX IF NOT EXISTS idx_workers_user ON workers(user_id);
-- No two PENDING enrolments can share a user_code -- a correct guess is
-- never ambiguous about which machine it resolves to (§3.2).
CREATE UNIQUE INDEX IF NOT EXISTS idx_workers_user_code_pending
  ON workers(user_code) WHERE approved_at IS NULL;
```

The enrolment code is stored **hashed**, same rule as session tokens (§2.2) — lookup by hash, the
plaintext exists only in the `POST /api/device/start` response and the worker's own config.

**Naming.** No `username_rigNNN`, and no naming derived from any linked identity's login (§2.1).
`display_name` defaults to the machine's own reported hostname (`gpu-tower`), is user-editable
inline on the worker card, and only needs to be unique per user — enforced with a friendly error,
not a 500. This also avoids v3.3's `existing.length + 1` counter bug, where pruning an unapproved
`rig002` made the next create collide with `rig003` — there is no counter here at all.

### 3.4 Server: `/api/device/start` doubles as re-enrolment

```ts
app.post("/api/device/start", { config: { bodyLimit: 4_096 } }, async (req, reply) => {
  const { machine_id, hostname, platform, arch } = parseDeviceStart(req.body);   // validated, capped (§1.8's rules)
  const deviceCode = generateEnrolmentCode();
  const userCode = generateUserCode();
  const expiresAt = Date.now() + 15 * 60 * 1000;

  const existing = repo.workerRepo.getByMachineId(machine_id);
  if (existing) {
    // Same machine as before -- whether it was owned (re-enrolment: expired
    // session, revoked from Settings, rebuilt disk) or still-pending from an
    // abandoned attempt, reuse the ROW so history/display_name/past runs
    // survive. Revoking prior sessions here closes the exact hole v3.3 had
    // (review finding D6): re-registering never leaves an old session live
    // alongside a new one.
    repo.sessionRepo.revokeWorkerSessions(existing.id);
    repo.workerRepo.reissueEnrolment(existing.id, { hostname, platform, arch, deviceCode, userCode, expiresAt });
    // user_id is left EXACTLY as it was -- untouched for an owned machine
    // being reconnected, still NULL for a still-pending one. Re-enrolling an
    // owned machine does NOT require re-approval; only a first enrolment does.
  } else {
    repo.workerRepo.createPending({ machineId: machine_id, hostname, platform, arch, deviceCode, userCode, expiresAt });
  }
  return { device_code: deviceCode, user_code: userCode, verification_uri: "/device", interval: 5, expires_in: 900 };
});

app.get<{ Querystring: { user_code?: string } }>("/api/device/status", async (req, reply) => {
  // Authed (§2.3 -- not in PUBLIC_PATHS): only a logged-in browser can even
  // check what a code resolves to, which is itself part of the rate-limit
  // story alongside §2.6's per-user cap on /api/device/approve.
  const worker = repo.workerRepo.getByUserCode(req.query.user_code ?? "");
  if (!worker || worker.enrolmentExpiresAt < Date.now()) return { state: "not_found" };
  if (worker.approvedAt) return { state: "approved" };
  return { state: "pending", machine: { hostname: worker.hostname, platform: worker.platform, arch: worker.arch, gpu: worker.hardware?.gpu?.[0]?.model } };
});

app.post<{ Body: { user_code?: string } }>("/api/device/approve", async (req, reply) => {
  const worker = repo.workerRepo.getByUserCode(req.body.user_code ?? "");
  if (!worker || worker.enrolmentExpiresAt < Date.now()) throw new NotFoundError("invalid or expired code");
  if (worker.approvedAt) throw new ConflictError("already approved");
  repo.workerRepo.approve(worker.id, (req as AuthenticatedRequest).user.id);   // sets user_id + approved_at,
  return { ok: true, machine: { hostname: worker.hostname } };                 // nulls user_code + enrolment_code_hash
});
```

`getByMachineId` is what makes this unify **first enrolment of a brand-new machine**,
**re-enrolment of an already-owned one**, and — usefully — **the transition of an existing Stage 1
machine** (created via the shared secret, `user_id` still NULL) into the multi-tenant model: it
just needs to call `/api/device/start` once and get approved like any other machine, no special
migration path required.

### 3.5 Worker config

```jsonc
// ~/.llamatoaster/config.json  — chmod 0600, created with mode 0o600
{
  "server_url": "https://llamatoaster.com",   // https:// enforced; the installer refuses http://
  "machine_id": "…",                          // stable, generated once on first boot, reused across re-enrolment
  "session_token": "…",
  "refresh_token": "…",
  "model_dir": "…", "llama_cpp_builds_dir": "…", "raw_json_dir": "…", "log_dir": "…",
  "bench_timeout_ms": 0
}
```

Gone: `worker_name`, `bind_host`, `port` (no inbound server — §1.11), any enrolment code (single-use,
discarded once a session exists). Note there is no `--code` flag at all now — the worker generates
its own codes by calling `/api/device/start` itself, which is what makes the install command a
true constant (§3.1). The installer must `chmod 0600` and warn if the file is group/world-readable
— it now holds a 90-day bearer token on someone's home machine, which v3.3 never addressed.

Publish `install.sh`'s SHA-256 next to the command wherever it's shown. `curl | sh` is the
pragmatic choice, but the user should be able to verify what they are about to run.

"Reconnect" on a worker card (Settings or the machine strip, §5.2) is just a UI affordance that
tells the user to re-run the same install command on that box — §3.4 already handles the resulting
`/api/device/start` call as a re-enrolment, no separate endpoint needed.

---

## Stage 4 — Multi-tenancy

### 4.1 Schema

```sql
ALTER TABLE runs      ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE results   ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE run_items ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE models    ADD COLUMN created_by TEXT REFERENCES users(id);   -- §4.4

CREATE INDEX IF NOT EXISTS idx_runs_user      ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_worker    ON runs(worker_id);
CREATE INDEX IF NOT EXISTS idx_results_user   ON results(user_id);
CREATE INDEX IF NOT EXISTS idx_run_items_user ON run_items(user_id);
```

(`runs.worker_id` was added in Stage 1 alongside the `workers` table.)

### 4.2 Legacy migration — claim the operator's history

v3.3 assigned all existing rows to a synthetic `legacy-user`. When the operator then logs in via
GitHub they become a *new* user, so **their entire benchmark history becomes invisible in the
normal UI** and reachable only through the admin view. Fix it:

```sql
-- 1. Everything pre-auth belongs to the operator.
UPDATE runs      SET user_id = :first_superadmin_user_id WHERE user_id IS NULL;
UPDATE results   SET user_id = :first_superadmin_user_id WHERE user_id IS NULL;
UPDATE run_items SET user_id = :first_superadmin_user_id WHERE user_id IS NULL;
UPDATE workers   SET user_id = :first_superadmin_user_id WHERE user_id IS NULL;
```

Run this as a **one-shot claim on the first superadmin login**, not at boot — the user row has to
exist first. Guard it with a `meta` flag so it runs exactly once.

`runs.worker_name` needs **no migration at all**. The export reads it directly
(`server/src/routes/results.ts:140,152`) — it is a plain snapshot column, not a join — so legacy
runs with any historical name (`vps-cpu`, `Local`, `alice@host`) survive untouched. Linking legacy
runs to a `workers` row via `worker_id` is optional UI polish; `NULL` is harmless.

### 4.3 Repo threading

Every user-scoped read/write takes `userId` as its **first** parameter:

```ts
listRuns(userId): Run[]
getRun(userId, id): Run | undefined
getRunWithResults(userId, id): {...} | undefined
createRun(userId, run): void
createRunItems(userId, runId, items): void
markRunRunning(userId, runId, patch): void
failAllRunItems(userId, runId, error): void
reconcileStaleRun(userId, runId, reason): Run | undefined
loadExportRows(userId, runIds?): ResultExportRow[]
```

`listModels` / `getModel` / `registerModel` keep their current signatures — the catalog is global.
v3.3 marked this as "finding #19, fixed" and then still wrote `repo.registerModel(req.user.id,
req.body)` in §4.2; don't repeat that.

Worker-reported writes (`POST /api/runs/:id/items/:idx`) resolve `userId` from the **worker
session**, never from the request body, and verify `run.worker_id === session.workerId` — stricter
than v3.3's `run.user_id === session.userId`, which would have let one of a user's machines report
results for a run assigned to a different machine of theirs.

**Isolation test (Stage 4 exit criterion):** enumerate every registered route; for each, assert
that user B receives 403/404 for a resource owned by user A. Do it by enumeration, not by
hand-picking routes — the point is to catch the one that was forgotten.

### 4.4 The models catalog

`repo.registerModel` is an **upsert on id** (`server/src/db/repo.ts:315`) that overwrites
`filename`, `hf_repo`, `hf_file` and `metadata`. Left as-is, any authenticated user can POST an
existing model id and rewrite another user's model row — silently corrupting `n_layer` /
`mtp_layers` for a benchmark already in flight.

- Global **read** (dedup by sha256 is genuinely valuable — keep it).
- `created_by` is set on first insert.
- On conflict: **only the creator may update** identity fields. Anyone may contribute *missing*
  metadata (fill a null `n_layer`), nobody may overwrite a non-null value they don't own.
- `DELETE /api/models/:id` requires `created_by = req.user.id` **and** the existing
  `countResultsForModel` / `countRunningRunsForModel` guards — extended to count **all** users'
  results, not just the caller's. It removes the catalog row only; files live on workers.
- Deleting a model file now targets **one of the caller's own machines** (a `delete_model_file`
  queue job), never a fan-out to every worker as `deleteFileFromAllWorkers` does today.

> **Privacy note to make explicit in the UI:** the catalog is global, so **model filenames are
> visible to every user** (`acme-internal-finetune-v3.gguf`). That conflicts with §5.3's
> "UI = own data only" rule unless it is stated. Say so at download time. If that is unacceptable,
> the alternative is a per-user catalog with sha256 dedup behind the scenes — more code, and worth
> it only if private model names are a real concern.

### 4.5 Models routes without workers

- `POST /api/models/:id/backfill-layer-count` reads `n_layer`/`mtp_layers` from the caller's
  `workers.model_files_json` cache. Missing → leave null. **No server-side GGUF parsing** — the
  server has no reader (`worker/src/gguf.ts` is worker-only).
- `GET /api/models/locations` reads `model_files_json` for the caller's machines only.
- `POST /api/models/:id/backfill-param-count` is unchanged (it only talks to Hugging Face).
- `GET /api/models/hf-updates` fans out to HF per distinct repo. With a global catalog this scales
  with total users — add a 10-minute server-side cache keyed by repo.

**Net: zero `fetch(worker.url/...)` calls remain in `models.ts`, and it stops importing
`config.ts`.**

### 4.6 AI context, server-side and user-scoped

`buildContextSnapshot(userId)` moves from `client/src/api/aiContext.ts` into
`server/src/routes/ai.ts`:

```ts
async function buildContextSnapshot(userId: string): Promise<string> {
  return [
    "## Your machines",            hardwareSummary(userId),     // workers.hardware_json, caller's rows
    "## Registered models",        modelsSummary(),             // global catalog
    "## Your recent benchmarks",   resultsSummary(userId, 150), // loadExportRows(userId), capped
  ].join("\n");
}
```

Each source is user-scoped at the repo layer, so the context cannot leak across tenants even if
the prompt is ignored. Combined with §2.7 (strip client system messages), the boundary is enforced
by SQL, not by instruction.

---

## Stage 5 — Admin & community

Two independent design questions live in this stage — *who* can see cross-tenant data (§5.1) and
*where* they see it (also §5.1, the origin split below) — plus the separate question of what a
*non-admin* is allowed to learn about other tenants at all (§5.3–5.4, unchanged by anything here).

### 5.1 Superadmin, revocable for real

v3.3 wrote `is_superadmin` to the `users` row at login and claimed "de-admining is just removing
the ID from env". It isn't: the flag persists until the next login, and the demoted admin's
existing 30-day session keeps full cross-tenant read access the whole time.

**No DB column.** Evaluate per request — it is a `Set.has()`:

```ts
// One entry per (provider, provider's own id) -- "github:123456", "google:104852...".
// Keyed on the SAME pair identities.provider/provider_user_id uses (§2.1), so a
// superadmin who later links a second provider (§2.4) grants that provider's
// login the same access without a separate config entry, once BOTH are in
// the env list; listing every provider identity that should count is the
// deliberate, explicit alternative to inferring it from `users.id`.
const SUPERADMIN_IDENTITIES = new Set(
  (process.env.SUPERADMIN_IDENTITIES ?? "").split(",").map((s) => s.trim()).filter(Boolean)
  // e.g. SUPERADMIN_IDENTITIES=github:123456,google:104852001937446291234
);
export function isSuperadminIdentity(provider: string, providerUserId: string): boolean {
  return SUPERADMIN_IDENTITIES.has(`${provider}:${providerUserId}`);
}
```

Set in §2.3's middleware on every request, against **every** identity linked to the caller's
account (a superadmin's account, however many providers it has linked, is superadmin on all of
them or none — there's no partial state). Removing an entry from env + restart = instant, complete
revocation, including live sessions. Provider-qualified stable ids, never logins (§2.1).

Enforcement is a **single hook on the admin plugin**, not a per-handler `if` — v3.3 defined a
`requireSuperadmin` helper (with a typo, `FasthenticatedRequest`), never wired it, and then
open-coded the check in all three handlers, which is one forgotten line away from a cross-tenant
leak:

```ts
export async function adminRoutes(app: FastifyInstance) {
  app.addHook("preHandler", async (req) => {
    if (!(req as AuthenticatedRequest).user?.isSuperadmin) throw new ForbiddenError("superadmin required");
  });
  // ...every route in this plugin is now gated by construction
}
```

Read-only in v1: `/api/admin/stats`, `/api/admin/runs`, `/api/admin/results/export`,
`/api/admin/users`. No promote/demote, no ban, no editing others' runs.

**Reachable only from its own origin — `supervise.llamatoaster.com`, not a route in the main SPA.**
An earlier draft of this stage put the admin surface at `/admin` inside the same app every tenant
uses. On reflection that's the wrong shape for what this actually is: an *operator* tool, not a
*product feature* — nothing here is meant for a normal user to ever brush up against, not even as
a hidden route or an unreachable JS bundle chunk their browser downloads anyway. A dedicated
origin gets several real properties a same-app route can't, cheaply, because Stage 6 already put
an nginx (or Caddy) reverse proxy and a single Fastify process in front of everything:

- **Narrower blast radius.** A same-app admin route shares its session cookie's reach with every
  other page in the app — an XSS or CSRF bug anywhere in the (larger, more exposed-to-user-content)
  main app can pivot straight into cross-tenant reads. A separate origin means a separate cookie
  jar by construction; compromising the main site's session grants nothing on `supervise.`.
- **The admin UI's code never reaches a normal user's browser at all** — not gated-and-hidden,
  genuinely absent — since it's a different build, served only off the admin hostname.
- **A real path to tightening later without touching the main app**: firewall
  `supervise.llamatoaster.com` to the operator's own Tailscale IP, add step-up auth, shorten its
  session lifetime — all independent of anything the public app does.
- Matches the "no self-service" posture already decided above: this reads as what it is, an
  operator's own tool, not a feature toggle inside the product.

**The cost, stated plainly, so it's a real trade and not a free lunch:** a second DNS record, a
second nginx `server{}` block (no second Node process needed — same Fastify instance, see below),
and the superadmin logs in **separately** on `supervise.` — a second, independent OAuth round trip,
not a shared session. For a read-mostly tool an operator checks occasionally rather than constantly
(§5.1's own framing), that's a small, deliberate cost for a materially better isolation property,
consistent with this plan's posture everywhere else it had a similar choice (§8, §2.6).

**One Fastify process, two hostnames — nginx routes both to the same `127.0.0.1:3000`** (§6.1). The
split happens inside the app, on `req.hostname`:

```ts
const ADMIN_HOSTNAME = new URL(process.env.ADMIN_PUBLIC_URL ?? "http://localhost").hostname;

export async function adminRoutes(app: FastifyInstance): Promise<void> {
  app.addHook("preHandler", async (req) => {
    // Hostname check FIRST, and a 404 -- not 403 -- on a mismatch. Hitting
    // /api/admin/stats via the MAIN hostname should look exactly like a route
    // that doesn't exist, never confirm an admin surface exists to probe further.
    if (req.hostname !== ADMIN_HOSTNAME) throw new NotFoundError();
    if (!(req as AuthenticatedRequest).user?.isSuperadmin) throw new ForbiddenError("superadmin required");
  });
  // ...every §5.1 route above, unchanged, now gated by BOTH hostname and identity
}
```

The OAuth callback (§2.4) branches the same way, at the tail end, after the provider profile is
already resolved — everything upstream of this (state check, token exchange, `mapProfile`) is
identical on both origins, only what happens with the result differs:

```ts
if (req.hostname === ADMIN_HOSTNAME) {
  // LOOKUP, never upsert -- signing in on this origin must never silently
  // mint a normal-user account for someone who isn't already allowlisted.
  // No account linking here either (§2.4's linking flow is a main-site concept).
  const identity = repo.userRepo.findIdentity(provider.id, mapped.providerUserId);
  if (!identity || !isSuperadminIdentity(provider.id, mapped.providerUserId)) {
    return reply.redirect(`${PUBLIC_URL}/login?error=not_authorized`);   // bounces to the MAIN site
  }
  const { token } = repo.sessionRepo.create(identity.userId, { label: describeUserAgent(req) });
  // HOST-ONLY cookie -- no Domain= attribute. That absence is the entire
  // isolation mechanism: setting Domain=llamatoaster.com here would make this
  // cookie valid on the main site too and silently undo the whole point.
  reply.setCookie("lt_session", token, { httpOnly: true, secure: COOKIES_SECURE, sameSite: "lax", path: "/", maxAge: 30 * 24 * 3600 });
  return reply.redirect("/");
}
// ... existing upsertByIdentity / linking path (§2.4), unchanged, main hostname only
```

`findIdentity` (lookup-only, the admin path's counterpart to §2.4's `upsertByIdentity`) is a small
addition to `userRepo` — no schema change; `sessions` needs no new column, since which origin
issued a session is never queried after the fact, only enforced live by the two independent checks
above on every admin request.

The admin SPA itself is a **new, deliberately small** top-level package (`admin/`, its own Vite
build, its own `admin/dist`) — a stats page, the filterable global runs table, an export button.
Nothing else: no chat panel, no run-triggering UI, nothing a normal user's app needs. Fastify picks
`admin/dist` or `client/dist` as the static root per-request based on `req.hostname`, ahead of
`@fastify/static`'s own handling — mechanical wiring, not security-critical, unlike the two gates
above.

### 5.2 Dashboard: machines, not users

The dashboard shows **the user's own machines** — a "Users" tile that always reads `1` for a
normal account is noise.

**Normal user — four counters plus a machine strip.** `machinesEverConnected` counts machines
that have completed enrolment and heartbeated at least once —
`SELECT COUNT(*) FROM workers WHERE user_id = ? AND last_heartbeat_at IS NOT NULL`. Abandoned
enrolments never counted; a machine that is currently powered off still does.

```tsx
<section className="grid grid-cols-2 gap-4 md:grid-cols-4">
  {/* Total machines that have EVER connected -- a lifetime count, not a
      live one. An online/total ratio makes the tile flicker as boxes sleep
      and turns a headline number into a status widget; per-machine liveness
      belongs in the strip below, where it is actionable. */}
  <StatCard label="Machines"      value={machinesEverConnected} />
  <StatCard label="Runs"          value={runs.length} />
  <StatCard label="Tests"         value={sumItemsTotal} />
  <StatCard label="Models tested" value={distinctModelIds} />
</section>

{/* The machines they actually use -- status, what it is, what it's doing now. */}
<section className="mt-6 grid gap-3 md:grid-cols-2">
  {machines.map(m => (
    <MachineRow key={m.id}
      name={m.display_name}                    // "gpu-tower"
      status={deriveWorkerStatus(m)}           // offline | idle | busy
      gpu={m.hardware?.gpu?.[0]?.model}
      backend={m.backend}
      activity={m.active_job_id ? currentRunSummary(m) : `last run ${timeAgo(m.lastRunAt)}`}
    />
  ))}
</section>
```

**Empty state** (a brand-new account currently lands on four zeros with no call to action):

> **No machines yet.** LlamaToaster runs benchmarks on your own hardware.
> **[ Connect your first machine ]** — takes about a minute.

Keep "Recent runs" and `TokSpeedDemo` below. Drop the Tailscale-era live worker-status chips
(`Dashboard.tsx:42-60`, driven by `useWorkerStatuses`) — machine state now lives in the strip
above, from cached heartbeat data.

Two notes against the current file:

- **`StatCard` takes `{ label, value }` only** (`client/src/components/StatCard.tsx:1`) — no
  `hint` prop. Add one, or fold the qualifier into the label.
- **"Models tested" is not today's "Models" tile.** The current tile counts the whole catalog
  (`models.length`, `Dashboard.tsx:38`); the new one counts distinct `model_id` across the user's
  own runs. Different number, and a new user will see 0 next to a non-empty catalog — which is
  correct, and another reason the empty state matters.

**Superadmin** gets exactly this same personal dashboard on the main site — nothing about it is
different for them, no admin link, no branch, no special-cased tile. The cross-tenant counters
(Users, Machines, Runs, Tests, Models tested) and the global runs table with User / Machine /
Backend / Status filters live entirely on `supervise.llamatoaster.com` (§5.1). Global numbers don't
belong on the personal dashboard, and now there's no code path here that could leak one.

### 5.3 Two-tier visibility (unchanged rule)

- **UI:** a normal user sees only their own data. No community page, no public API.
- **AI assistant:** the one cross-user window, and it returns **anonymised aggregates only**.
- **Superadmin:** cross-tenant read, env-gated, not self-service, and reachable only from
  `supervise.llamatoaster.com` (§5.1) — never from the app every tenant uses.

### 5.4 Community aggregates need k-anonymity, not just a column allowlist

v3.3's `listAggregates` grouped by `(model_id, backend, build, platform, gpu_model)` and returned
`run_count`, `avg/min/max_tps`, RAM and VRAM. With a small user base most groups are **one
machine** — filter by `gpuModel: 'RTX 5090'` and `platform: 'darwin'` and you have singled out one
person's rig and can read their exact sweep parameters and throughput. Omitting the `username`
column does not make that anonymous; this is textbook singling-out.

The column allowlist stays (it is the right mechanism) and gets four additions:

```sql
SELECT
  r.model_id, m.filename AS model_filename,
  r.llama_cpp_backend AS backend,
  json_extract(w.hardware_json, '$.platform')     AS platform,
  json_extract(w.hardware_json, '$.gpu[0].model') AS gpu_model,
  COUNT(DISTINCT r.user_id)     AS contributor_count,
  COUNT(DISTINCT r.id)          AS run_count,
  ROUND(AVG(ri.avg_tps), 1)     AS avg_tps,          -- rounded
  ROUND(AVG(ri.ram_peak_mib))   AS avg_ram_peak_mib,
  ROUND(AVG(ri.vram_peak_mib))  AS avg_vram_peak_mib
FROM runs r
JOIN results ri ON ri.run_id = r.id
LEFT JOIN models m  ON m.id = r.model_id
LEFT JOIN workers w ON w.id = r.worker_id
JOIN users u ON u.id = r.user_id AND u.share_benchmarks = 1     -- (2) consent
WHERE r.user_id <> :caller_id                                   -- (3) "others' runs"
GROUP BY r.model_id, r.llama_cpp_backend, platform, gpu_model   -- (4) build dropped from the key
HAVING COUNT(DISTINCT r.user_id) >= 5                           -- (1) k-anonymity
ORDER BY avg_tps DESC
```

1. **`HAVING COUNT(DISTINCT user_id) >= 5`** — suppress cells that describe fewer than five
   people. This is the change that makes "anonymised" true.
2. **`share_benchmarks`** — §4.3's tool description promised "opt-in only" while §8 had no opt-in
   anywhere. Now it is a real column (`users.share_benchmarks`, default `1`), disclosed at signup
   and toggleable in Settings. Default-on is defensible for a comparative-benchmarking product
   *if* it is stated plainly; the point is that the claim is now backed by a column.
3. **Exclude the caller's own runs** — the tool is for others' data; the caller's own is already in
   their context (§4.6).
4. **`build` dropped from the group key** — high-cardinality and near-unique per user at any given
   moment, so it re-identifies through the k-threshold.
5. **`MIN`/`MAX` dropped** — extremes are the most identifying values in an aggregate.

`listFacets` gets the same k-threshold, or it leaks the existence of a single rare GPU.

**Unit tests:** no result row contains a UUID-shaped or username-shaped value; every returned row
has `contributor_count >= 5`; a synthetic single-contributor group is absent from the output.

The server-owned system prompt (§2.7) still instructs the model not to attribute or count users —
but it is now defence in depth on top of a query that cannot return the data, rather than the only
control.

---

## Stage 6 — Deployment

### 6.0 No Docker — this app already has a working non-Docker deployment

v3.3 specified Docker Compose here without checking it against what this repo actually runs.
It doesn't: `deploy/ecosystem.config.cjs` (PM2) and `deploy/llamatoaster-server.service` /
`llamatoaster-worker.service` (systemd) are both already written, both already used —
`deploy/push-client.ps1` deploys to the live VPS today by building the client, `scp`-ing
`client/dist`, and `pm2 restart llamatoaster-server` over SSH. Introducing Docker here would mean
**discarding a working pipeline to rebuild an equivalent one**, not adding something missing.

Concretely, for a single VPS running one app for one operator (soon many tenants, but still one
deploy target — not a fleet):

- **Docker's actual value proposition doesn't apply.** Compose earns its keep coordinating many
  services across many environments that need to behave identically. This is one Fastify process
  on one box. There is nothing to orchestrate and nowhere else this needs to run identically.
- **It would regress the fast path.** Today, a client-only change is `npm run build` + `scp` +
  `pm2 restart` — seconds, no image rebuild. Under Docker, the same change means rebuilding and
  redeploying an image. That's a real cost for no corresponding benefit here.
- **The systemd unit already has real sandboxing**, written and currently running:
  `NoNewPrivileges=true`, `ProtectSystem=strict`, `PrivateTmp=true`, and `ReadWritePaths` scoped to
  exactly `data/`. That's equivalent-or-better isolation to a naive `docker run`, at zero
  additional cost since it already exists.
- **It adds a privileged daemon as attack surface at exactly the wrong moment.** This whole plan is
  about hardening the app for a public launch (Stages 2–5). Adding Docker Engine as a new
  always-on root-capable process on the same box works against that, for a workload that doesn't
  need it.
- **The plan already rejected containers for the *other* half of this app.** Stage 3's worker
  installer is a plain native `curl | sh` / `irm | iex` script specifically because workers run on
  users' own GPU boxes, where GPU driver passthrough (CUDA/ROCm) into a container is real,
  ongoing friction — not a solved problem. Containerizing the *server* half while deliberately
  keeping the worker half native would be inconsistent without a reason that applies here.

better-sqlite3 (the one native module in this stack) isn't a counter-argument for Docker either:
that would matter if this needed to run identically across many *different* host environments —
it doesn't. The VPS already produces a working build today; there's nothing to make reproducible
across machines that don't exist.

**Keep the process managers already in `deploy/`** — PM2 or systemd, whichever is the daily driver
(both stay as valid alternatives, per `ecosystem.config.cjs`'s own comment; one manifest-level fix
to the PM2 path is below) — and add a reverse proxy in front, which is genuinely new: the app
today binds to a **Tailscale-only**
address (`deploy/orchestrator.env.example`'s `BIND_HOST=<vps-tailscale-ip>`) with nothing public in
front of it at all, because it never needed to be Internet-facing before. Going public is the one
real new deployment requirement this plan introduces, independent of Docker.

**Correction to an earlier draft of this section:** it previously said the VPS's own worker "stops
being part of the production topology" once Stage 1 ships. That's wrong — it keeps running in
production. What actually changes is what it *is*: no longer this repo's special-cased
local/remote worker pair under the old tailnet model, just **one more enrolled machine under the
operator's own account**, indistinguishable from any user's GPU box (Stage 3). That reframing is
what makes the next point matter.

**Server and worker are two independent processes and must be deployed, and fail, independently.**
They already run as separate systemd units (`llamatoaster-server.service`,
`llamatoaster-worker.service` — each start/stop/restarts on its own, always did). The PM2 path
doesn't currently match that: `deploy/ecosystem.config.cjs` declares **both** apps in one file's
`apps: []` array. PM2 itself can still target one by name (`pm2 restart llamatoaster-server`
doesn't touch the worker, and `push-client.ps1` already only ever restarts the server) — but one
shared manifest still invites exactly the mistake independent processes shouldn't allow: a
`pm2 start ecosystem.config.cjs` / `pm2 reload ecosystem.config.cjs` / `pm2 delete
ecosystem.config.cjs` run without `--only` touches both, coupling a server deploy to the worker's
lifecycle for no reason. Split it, mirroring the systemd units 1:1:

```js
// deploy/ecosystem.server.config.cjs
module.exports = { apps: [ /* the existing llamatoaster-server entry, unchanged */ ] };

// deploy/ecosystem.worker.config.cjs
module.exports = { apps: [ /* the existing llamatoaster-worker entry -- config source below */ ] };
```

`pm2 start deploy/ecosystem.server.config.cjs` and `pm2 start deploy/ecosystem.worker.config.cjs`
become two genuinely separate deploy actions — a server deploy (frequent: every code push) can
never touch a worker mid-benchmark, and a worker restart (rare, and never automated by
`push-client.ps1`) can never bounce the server's live sessions. This isn't just tidiness: it's the
same property the pull-queue architecture already assumes. Stage 1's lease/reaper machinery
(§1.6–1.7) exists specifically because the worker and server are meant to tolerate losing contact
with each other — a shared PM2 manifest that restarts them together undermines the one thing that
design already bought.

**The worker's config source needs to change too, for the same reason.** Today
`ecosystem.config.cjs` sets `WORKER_CONFIG: "worker/config.vps.json"` — a static, repo-relative
file, hand-provisioned outside the app's own auth model. Once the VPS's worker is "just another
enrolled machine" (above), it should read the **same** `~/.llamatoaster/config.json` every other
worker gets from running the install script and completing device-flow approval (§3.5) — not a
bespoke deploy-time file. Concretely: run `install.sh` on the VPS itself, approve it from the
operator's own logged-in session like any other machine, and point
`deploy/ecosystem.worker.config.cjs` at that config path instead of `WORKER_CONFIG`. This also
means the VPS worker survives a credential rotation (§2.5) the same way every other worker does —
"Reconnect" (§3.4), not a manual file edit.

### 6.1 nginx — or Caddy, your call, both work

The plan's earlier draft specified Caddy for its zero-config automatic HTTPS. If you already run
nginx for anything else on this VPS, use that instead — one front door for every site on the box
beats running two reverse proxies side by side, and the header/CSP config translates directly:

```nginx
server {
    listen 443 ssl http2;
    server_name llamatoaster.com;

    ssl_certificate     /etc/letsencrypt/live/llamatoaster.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llamatoaster.com/privkey.pem;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    # v3.3's checklist claimed CSP and never shipped it in the (Caddy) config -- ship it here.
    add_header Content-Security-Policy "default-src 'self'; img-src 'self' https://avatars.githubusercontent.com https://lh3.googleusercontent.com data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'" always;

    location / {
        proxy_pass http://127.0.0.1:3000;   # BIND_HOST changes to 127.0.0.1 -- see below
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 35s;   # >= the 25s long-poll in §1.4 -- an nginx default (60s) is fine
                                  # too, but state it explicitly rather than relying on the default
    }
}
server {
    listen 80;
    server_name llamatoaster.com;
    return 301 https://$host$request_uri;
}

# supervise.llamatoaster.com (§5.1) -- SAME backend, 127.0.0.1:3000. No second
# Node process, no second PM2/systemd unit: Fastify tells the two hostnames
# apart itself (req.hostname), so this block only needs to exist and forward.
server {
    listen 443 ssl http2;
    server_name supervise.llamatoaster.com;

    ssl_certificate     /etc/letsencrypt/live/llamatoaster.com/fullchain.pem;   # -d for both hostnames, one cert
    ssl_certificate_key /etc/letsencrypt/live/llamatoaster.com/privkey.pem;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    # A tighter CSP than the main site's is fine here (no avatar images, no AI
    # chat, no third-party embeds at all in the deliberately minimal admin SPA).
    add_header Content-Security-Policy "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'" always;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
server {
    listen 80;
    server_name supervise.llamatoaster.com;
    return 301 https://$host$request_uri;
}
```

- **The one thing nginx doesn't give you for free that Caddy does: certificate renewal.** Run
  `certbot --nginx -d llamatoaster.com -d supervise.llamatoaster.com` once (both hostnames, one
  cert), verify its systemd timer (`certbot.timer`, installed automatically on most distros)
  actually renews before relying on it — the standard, well-trodden path, just not zero-config the
  way Caddy is.
- **`BIND_HOST` changes from the Tailscale IP to `127.0.0.1`.** The app should not be reachable on
  any public interface directly — only nginx, on 80/443, is internet-facing; confirm with the
  VPS's firewall (`ufw`/`iptables`), not just by trusting the bind address.
- **`app.register(fastify, { trustProxy: true })` (or the VPS's own loopback address, more
  precisely) is required once nginx is in front of anything.** Without it, `req.ip` inside Fastify
  is nginx's own address for every request — meaning §2.6's per-IP rate limits (`/auth/:provider*`,
  `/api/device/start`, `/api/device/token`) silently collapse into one shared global limit instead
  of one per real client, since every request looks like it came from the same place. The
  `X-Forwarded-For` header the config above already sets is exactly what `trustProxy` needs to
  read; without both halves, one is dead weight.
- `img-src` allowlists each linked provider's avatar CDN individually (GitHub's shown; Google's
  `lh3.googleusercontent.com` added here in advance since §2.1 anticipates it) — extend this list
  when a new `PROVIDERS` entry (§2.4) is added, or proxy avatars server-side and drop the
  allowlist entirely. Doesn't apply to the admin origin's own, stricter CSP above.
- `style-src 'unsafe-inline'` is required by the current Tailwind build — remove it if the build
  is switched to extracted CSS.
- If Caddy is preferred instead (still entirely valid — no other change in this plan depends on
  which one), both blocks above become `reverse_proxy 127.0.0.1:3000` with their respective
  headers; automatic HTTPS (for both hostnames, same Caddyfile) is the trade for owning less of the
  TLS lifecycle yourself.

### 6.2 Everything else is unchanged by the Docker-vs-bare-metal question

- **Env delivery** stays exactly the pattern already in `deploy/`: `EnvironmentFile=` for
  `llamatoaster-server.service`, or `ecosystem.server.config.cjs`'s own `loadEnvFile` helper for
  PM2 (unchanged by the split above — it moves with the server's app entry) — both already parse
  `deploy/orchestrator.env`. New vars (`GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, and once added
  `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` — see §2.4's `PROVIDERS` map — `SUPERADMIN_IDENTITIES`,
  `PUBLIC_URL`, `ADMIN_PUBLIC_URL` (`https://supervise.llamatoaster.com` — §5.1), `AI_*`,
  `AI_GLOBAL_DAILY_CAP`) just get added to that same file. **No `SESSION_SECRET`** — the store is
  `sha256(token)`; do not ship a dead env var.
- `/health` returns `{ok, db}` only. It is public, and today it echoes `err.message`
  (`server/src/index.ts:101`) — log the detail, don't serve it.
- **Backups**: the WAL-aware `sqlite3 .backup` daily cron from v3.3 stands unchanged — it never
  depended on Docker — **plus** the pre-migration backup in §0.6. Test a restore before go-live.
- `push-client.ps1` needs one addition once the app is public: today it deploys over the Tailscale
  address; add (or parameterize) a path that deploys over the public hostname too, since the VPS
  will still be reachable over the tailnet for operator access but the app itself now serves
  `PUBLIC_URL`.

---

## Security checklist

**Tokens & sessions**
- [ ] Session tokens are **never stored** — `sessions.id` is a surrogate UUID; only `token_hash` /
      `refresh_hash` exist (§2.2)
- [ ] `UNIQUE INDEX` on `sessions(token_hash)` and `sessions(refresh_hash)` (§2.2)
- [ ] All tokens from `crypto.randomBytes(32)`; `Math.random()` appears nowhere near a credential (§0.3)
- [ ] `safeEqual` (timing-safe) for OAuth `state` and `user_code` (§0.3)
- [ ] Worker refresh rotates both tokens, keeps `sessions.id` stable, extends expiry, and
      **detects replay** via `prev_refresh_hash` (§2.5)
- [ ] Expired sessions pruned on the maintenance timer (§1.7)
- [ ] `GET/DELETE /api/sessions` + revoke-all shipped (§2.5)

**Cookies & origin**
- [ ] Same-origin; no CORS; `VITE_API_URL` deleted (§0.5)
- [ ] `lt_session`: HttpOnly, `SameSite=Lax`, `secure` **derived from PUBLIC_URL** (§0.5)
- [ ] `lt_refresh` cookie deleted — browser sessions use sliding expiry (§2.5)

**Enrolment**
- [ ] `enrolment_code_hash` stores a hash, never the plaintext `device_code` (§3.3)
- [ ] `user_code` has a unique index scoped to pending rows — no two live enrolments can collide
      on the same code (§3.3)
- [ ] Codes are **single-use**: `POST /api/device/approve` nulls `user_code` +
      `enrolment_code_hash` in the same write that sets `user_id`/`approved_at` (§3.4)
- [ ] `POST /api/device/start` and `POST /api/device/approve` are rate-limited independently
      (§2.6) — `/device/status` and `/device/approve` require an authenticated session, they are
      **not** public (§3.4)
- [ ] The `/device` approval card shows the machine (hostname/GPU/platform) **and** an explicit
      "only approve a code you generated yourself" warning — device-code phishing is an accepted,
      documented residual risk for this flow, mitigated but not eliminated (§3.2)
- [ ] Re-enrolling an already-owned machine revokes its prior sessions before issuing new
      enrolment codes on the same row (§3.4)
- [ ] Worker `config.json` written 0600; installer refuses a non-HTTPS server URL (§3.5)
- [ ] `install.sh` SHA-256 published next to the install command (§3.5)

**Authorization**
- [ ] Every user-scoped repo function takes `userId` first (§4.3)
- [ ] Route-enumerating isolation test: user B gets 403/404 on every one of user A's resources (§4.3)
- [ ] Worker writes verify `run.worker_id === session.workerId` (§4.3)
- [ ] `models` upsert cannot overwrite another user's row (§4.4)
- [ ] Admin gated by a **plugin-level hook**, not per-handler `if`s (§5.1)
- [ ] `is_superadmin` evaluated per request from env — revocable without a DB write (§5.1)
- [ ] `/api/admin/*` checks hostname **before** identity, and 404s (not 403s) on a hostname
      mismatch — the main site never confirms an admin surface exists (§5.1)
- [ ] The admin origin's login cookie is **host-only** (no `Domain=` attribute) — verified it is
      never sent to the main hostname (§5.1)
- [ ] Admin-origin OAuth callback only **looks up** an identity, never creates one — no account can
      be minted by signing in on `supervise.` (§5.1)
- [ ] No mass assignment: `createRun` builds its object field by field (§1.12)

**Input**
- [ ] Worker state push validated and capped (arrays, string lengths, control chars); explicit
      `bodyLimit` on worker routes (§1.8)
- [ ] Worker-supplied strings sanitised before they can reach an LLM context (§1.8, §5.4)

**Abuse & cost**
- [ ] `/api/device/token` rate-limited **per IP** (not per device_code) (§2.6)
- [ ] `/api/ai/chat`: per-user hourly/daily quota, global daily circuit breaker, message
      count/size caps (§2.6)
- [ ] `/api/runs/trigger` and model registration rate-limited per user (§2.6)

**Privacy**
- [ ] Server owns the AI system prompt; client `system` messages discarded (§2.7)
- [ ] AI context built server-side from user-scoped queries (§4.6)
- [ ] Community aggregates enforce `HAVING COUNT(DISTINCT user_id) >= 5` (§5.4)
- [ ] `users.share_benchmarks` exists, is disclosed at signup, and is toggleable (§5.4)
- [ ] No email collected from **any** linked provider; every `PROVIDERS` entry requests
      profile-only scope (`read:user` on GitHub, `openid profile` on Google) (§2.1, §2.4)
- [ ] `ip_hash` / `user_agent` not collected (§2.2)
- [ ] Account linking requires an authenticated session on the existing account **and** a fresh
      OAuth round trip on the new provider — never auto-linked by matching email (§2.4)
- [ ] Global model catalog's cross-user filename visibility disclosed in the UI (§4.4)
- [ ] **Account deletion** endpoint ships in v1: cascades `sessions`/`workers`/`runs`/`results`/
      `run_items`/`identities`, and either anonymises or removes the user's contribution to
      aggregates. Plus a data export (the CSV export already exists — point at it). A public SaaS
      storing OAuth-derived identity needs both.

**Transport & headers**
- [ ] HTTPS enforced end to end, including worker → server (§3.5)
- [ ] CSP shipped alongside HSTS / nosniff / frame-options / referrer-policy, from the reverse
      proxy in front of the existing PM2/systemd deploy — no Docker (§6.0, §6.1)
- [ ] `BIND_HOST` is `127.0.0.1`; only the reverse proxy is reachable on a public interface,
      confirmed at the firewall (§6.1)
- [ ] `trustProxy` configured in Fastify, matching nginx's `X-Forwarded-For` — otherwise every
      §2.6 per-IP rate limit silently collapses into one shared global limit (§6.1)
- [ ] `/health` does not leak error detail (§6.2)

---

## Appendix A: key types

```ts
// shared/types.ts additions. Field names are camelCase on the API and
// snake_case in SQLite; repo.ts owns the mappers. Every field listed here
// must exist in a mapper -- v3.3's mapSession omitted activeJobId while
// handlers read req.session.activeJobId.

export interface User {
  id: string;                       // THE identity. Provider-independent (§2.1).
  displayName: string | null;
  avatarUrl: string | null;
  shareBenchmarks: boolean;         // ← users.share_benchmarks
  createdAt: number; updatedAt: number; lastLoginAt: number | null;
}

// One per linked OAuth provider (§2.1/§2.4). A user has >=1; >1 once they've
// used Settings -> Connected accounts to link a second provider.
export interface Identity {
  id: string;
  userId: string;
  provider: "github" | "google";    // extend the union alongside PROVIDERS (§2.4)
  providerUserId: string;           // ← identities.provider_user_id -- stable within that provider
  providerLogin: string | null;     // display login/handle; refreshed every login
  createdAt: number;
}

// isSuperadmin is NOT on User -- it is computed per request from env, over
// ALL of the caller's linked identities (§5.1), and lives on
// AuthenticatedRequest.user only.
export interface AuthedUser extends User { isSuperadmin: boolean; }

export interface Session {
  id: string;                       // surrogate; never sent to a client
  userId: string;
  expiresAt: number;
  isWorker: boolean;
  workerId: string | null;
  label: string | null;
  lastSeenAt: number | null;
  createdAt: number;
}                                   // token_hash / refresh_hash never leave the repo layer

export interface Worker {
  id: string;
  userId: string | null;            // null only in Stage 1 / pre-claim legacy rows
  machineId: string;
  displayName: string;              // user-chosen; defaults to hostname
  hostname: string | null;
  backend: Backend | null;
  platform: string | null; arch: string | null;
  hardware: HardwareInfo | null;
  installedBuilds: InstalledBuild[];
  modelFiles: ModelDirFile[];
  status: "offline" | "idle" | "busy";   // DERIVED, never stored (§1.6)
  lastHeartbeatAt: number | null;
  activeJobId: string | null;
  approvedAt: number | null;
  createdAt: number; updatedAt: number;
}

// PREREQUISITE: `ModelDirFile` is currently a worker-LOCAL interface
// (worker/src/index.ts:308) and is not exported from shared/types.ts at all,
// so the server cannot reference it. Promote it to shared/types.ts first --
// the worker keeps using it unchanged, the server gains the type it needs.
export interface WorkerStatePush {
  machine_id: string;
  capabilities: string[];
  hostname: string;
  hardware: HardwareInfo;                  // shared/types.ts:676, gpu[] is {vendor, model, vram_mb?, vram_dynamic?}
  installed_builds: InstalledBuild[];      // extended with bench_path/server_path, below
  model_files: (ModelDirFile & { n_layer?: number | null; mtp_layers?: number | null })[];
  status: "idle" | "busy";
}

export interface ActiveJobReport {
  job_id: string;
  phase: "downloading" | "extracting" | "loading" | "benchmarking" | "finalizing";
  bytes?: number; total_bytes?: number;
  item_idx?: number; items_total?: number;
  detail?: string;
}

export interface HeartbeatResponse {
  worker_id: string;
  control: { cancel_job_ids: string[]; pause: boolean };
  lease_until: number;
}

export type QueueJob =
  | { job_id: string; type: "benchmark";      payload: BenchmarkJob }
  | { job_id: string; type: "install_build";  payload: InstallBuildJob }
  | { job_id: string; type: "download_model"; payload: DownloadModelJob }
  | { job_id: string; type: "delete_model_file"; payload: { filename: string } };

export interface BenchmarkJob {
  run_id: string;
  model: Model;
  mtp_model?: Model;
  // Omit<..., "model_id"> matches BOTH TriggerPayload.sweep (shared/types.ts:652)
  // and expandSweep's parameter (shared/sweep.ts:75). A bare SweepConfig here
  // would not type-check against either.
  sweep: Omit<SweepConfig, "model_id">;   // the WHOLE sweep -- the worker loops it (§1.1)
  main_gpu?: number;
  llama_cpp_build: string;          // tag; resolved to a path from installed_builds (§1.12)
  llama_cpp_backend: Backend;
}

// EXTEND the existing InstalledBuild (shared/types.ts:669) so the worker can
// resolve tag -> binary path. toInstalledBuildList (worker/src/index.ts:523)
// currently serialises only {tag, asset_name, installed_at, active}; the paths
// already exist on InstalledBuildInfo (worker/src/llama-builds.ts:71) as
// `bench_path` / `server_path` -- snake_case, matching this file's convention
// and set at llama-builds.ts:166 and :244-245 -- and are simply never pushed.
// There is NO `backend` field on InstalledBuild: match on tag/asset_name.
export interface InstalledBuild {
  tag: string; asset_name: string; installed_at: number; active: boolean;
  bench_path?: string; server_path?: string;    // ← add
}

// Response to the WORKER's own POST /api/device/start (§3.4) -- not shown to
// a human directly; the worker prints verification_uri + user_code itself.
export interface DeviceStartResponse {
  device_code: string;               // high-entropy, worker-held, POSTed to /api/device/token
  user_code: string;                 // "ABCD-EFGH" -- what the worker prints for a human to type
  verification_uri: string;          // "/device"
  expires_in: number;                // 900
  interval: number;                  // 5
}

// Polled by /device (authed browser) after the user types a user_code, to
// drive the one-click approve card (§3.4).
export interface DeviceStatusResponse {
  state: "pending" | "approved" | "not_found";
  machine?: { hostname: string; platform: string; arch: string; gpu?: string };
}
```

---

## Appendix B: review findings → where fixed

| Review § | Finding | Fixed in |
|---|---|---|
| A1 | Only sweep item 0 ever runs | §1.1 — one job per run, worker loops |
| A2 | Worker can't poll while executing | §1.3, §1.9 — heartbeat loop independent of execution |
| A3 | Stop/pause/resume deleted with no replacement | §1.14 — control directives on the heartbeat |
| A4 | No dead-worker detection, no job redelivery | §1.6 derived status, §1.7 lease reaper |
| A5 | Download/install progress disappears | §1.5 — `ActiveJobReport` → `progress_json` |
| A6 | Run-log download breaks | §1.10 — worker pushes the log on completion |
| A7 | §0.1 DDL crashes the server on boot #2 | §0.6 — `IF NOT EXISTS` is a stated rule |
| A8 | 202 path creates an orphan run | §1.12 — reject offline machines up front |
| A9 | `runs.status` / `llama_cpp_build` never set | §1.12 explicit fields, §1.13 lifecycle |
| B1 | Session token stored in plaintext | §2.2 — surrogate id, hash only |
| B2 | Token generation never specified | §0.3 — full implementation |
| B3 | No index on `token_hash` | §2.2 — `UNIQUE INDEX` |
| B4 | Privacy enforced by a client-side system prompt | §2.7 — server owns it, strips client system messages |
| B5 | Aggregates re-identify individuals | §5.4 — k=5, rounding, no min/max, no build |
| B6 | "opt-in only" claimed, never built | §5.4 — `users.share_benchmarks` |
| B7 | Superadmin not actually revocable | §5.1 — per-request env evaluation, no column |
| B8 | Rate limits keyed wrong; AI endpoint unmetered | §2.6 — per-IP device limit, AI quotas + circuit breaker |
| B9 | Device codes not single-use | §3.4 — nulled in the approval transaction |
| B10 | Mass assignment in the run trigger | §1.12 — explicit field list |
| B11 | Worker state stored unvalidated | §1.8 — validated, capped, sanitised |
| C1 | Two copies of a 64-char secret to enrol | §3.1 — worker-first device flow, one copy (a constant command), one click; §3.2 states the honest cost this trades for |
| C2 | Username generator buys nothing | §2.1 — deleted; federated `identities` table, provider-agnostic `users` |
| C3 | "Users: 1" tile; no empty state | §5.2 — machines strip + first-run CTA |
| C4 | No session management / re-enrolment / notifications | §2.5, §3.4/§3.5 (notifications still open) |
| D1 | `InstalledBuild.backend` doesn't exist | §1.12 — match on tag/asset_name |
| D2 | `registerModel(userId, …)` won't compile | §4.3 — signature unchanged |
| D3 | Browser refresh path broken | §2.5 — sliding expiry, no browser refresh token |
| D4 | Rotating `sessions.id` orphans `workers.session_id` | §2.5 — id stable, replay detection added |
| D5 | `existing.length + 1` name counter collides | §3.3 — no generated names |
| D6 | `createWorker` inserts a duplicate `(user_id, name)` | §3.4 — re-enrol reuses the row by `machine_id` |
| D7 | Three `getReleases()` calls, mismatched tags | §1.12 — resolve once |
| D8 | `req.session` field naming inconsistent | Appendix A — one shape, all mapped |
| E1 | LLM as the cross-tenant security boundary | §5.4 — SQL-enforced; prompt is defence in depth |
| E2 | Global catalog is an unaccounted read/write channel | §4.4 — ownership + disclosure |
| E3 | 500 ms poll loop defeats the long poll | §1.4 — event-driven |
| E4 | No test infrastructure | §0.4 — vitest is a Stage 0 item |
| E5 | Legacy migration orphans the operator's history | §4.2 — one-shot claim on first superadmin login |
| E6 | Cross-origin story unresolved | §0.5 — same-origin, decided |
| E7 | Sessions never garbage-collected | §1.7 — maintenance timer |

**Still open, deliberately:** run-completion notifications (C4) — worth doing, but not a blocker.
Per-item job restart granularity (§1.1) — rejected for now, revisit if long sweeps prove flaky.

---

*Document version: 4.0 — restructured around a staged execution order; security findings from
`MULTIUSER_PLAN_REVIEW.md` applied; one-click enrolment; username generator removed;
dashboard reoriented to machines.*
