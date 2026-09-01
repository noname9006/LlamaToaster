import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type {
  RunItemUpdateInput,
  ModelDownloadCallbackInput,
  WorkerStatePush,
  ActiveJobReport,
  HeartbeatResponse,
  QueueJob,
  DeviceStartResponse,
  DeviceTokenSuccess,
  DeviceTokenError,
  RefreshResponse,
  HardwareInfo,
  ProbeResultInput,
  ProbeAttemptReport,
  QualityResultInput,
} from "../../shared/types.js";
import type { ProbeDedupPoint } from "../../shared/api-v8.js";

export function writeRawJson(runDir: string, runId: string, data: unknown): string {
  if (!existsSync(runDir)) mkdirSync(runDir, { recursive: true });
  const path = join(runDir, `${runId}.json`);
  writeFileSync(path, JSON.stringify(data, null, 2), "utf8");
  return path;
}

// Single endpoint for both tiers of per-item reporting (see shared/types.ts's
// RunItemTickInput/RunItemTerminalInput) -- the caller decides how to treat
// failures: worker/src/index.ts awaits-and-swallows for ticks, and wraps
// this in its own retry-with-backoff for terminal outcomes.
//
// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3): now Bearer-authenticated like
// every other pull-queue call below -- this route used to accept an
// unauthenticated POST at all (a real gap the server-side fix closes; see
// server/src/routes/runs.ts's own comment on it), so worker/src/index.ts's
// callers now route through withAuth the same way postHeartbeat/pollQueue
// already do.
export async function postRunItemUpdate(
  url: string,
  token: string,
  runId: string,
  idx: number,
  payload: RunItemUpdateInput,
  timeoutMs = 5000
): Promise<void> {
  const res = await fetch(`${url}/api/runs/${runId}/items/${idx}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `item update failed (${res.status}): ${text}`);
  }
}

// BENCHMARKING_PLAN_V8.md N2/N4 result ingestion. Both routes require an
// ENROLLED WORKER SESSION server-side and refuse the shared deployment
// secret, so the token passed here must be the per-worker credential -- the
// same one every other authenticated call already uses.
export async function postProbeResult(
  url: string,
  token: string,
  runId: string,
  payload: ProbeResultInput,
  timeoutMs = 15_000
): Promise<void> {
  const res = await fetch(`${url}/api/runs/${runId}/probe-result`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `probe result failed (${res.status}): ${text}`);
  }
}

// N2 batch dedup -- every already-measured (ctx, ngl) point from an earlier
// sibling run under the same batch root (see server/src/routes/
// measurements.ts's own comment on the route). Best-effort, short timeout:
// on failure the caller just proceeds without dedup, exactly as if this
// probe weren't part of a batch at all.
export async function getProbeDedup(url: string, token: string, runId: string, timeoutMs = 5000): Promise<ProbeDedupPoint[]> {
  const res = await fetch(`${url}/api/runs/${runId}/probe-dedup`, {
    headers: { ...authHeader(token) },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `probe dedup fetch failed (${res.status}): ${text}`);
  }
  const body = (await res.json()) as { points: ProbeDedupPoint[] };
  return body.points;
}

// N2 live progress -- one rung, posted as it happens (see
// worker/src/index.ts's executeRunProbeJob), so the RunDetail page can show
// rows appearing during the ladder instead of all at once when
// postProbeResult finally lands at the end. Best-effort by design (short
// timeout, caller swallows the error) -- losing one tick just means the
// panel lags until the next one, same tolerance as sendTick/postRunItemUpdate.
export async function postProbeAttempt(
  url: string,
  token: string,
  runId: string,
  seq: number,
  attempt: ProbeAttemptReport,
  timeoutMs = 5000
): Promise<void> {
  const res = await fetch(`${url}/api/runs/${runId}/probe-attempt`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify({ seq, ...attempt }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `probe attempt tick failed (${res.status}): ${text}`);
  }
}

export async function postQualityResult(
  url: string,
  token: string,
  runId: string,
  payload: QualityResultInput,
  timeoutMs = 15_000
): Promise<void> {
  const res = await fetch(`${url}/api/runs/${runId}/quality-result`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `quality result failed (${res.status}): ${text}`);
  }
}

// Reports a model download's terminal outcome (success or failure) once the
// background transfer in worker/src/index.ts's POST /models/download
// settles. Same retry-with-backoff posture as postRunItemUpdate's terminal
// use above -- this is what actually gets a completed download registered
// as a Model server-side, so losing it silently would leave a fully
// downloaded file on disk that the app never learns about.
export async function postModelDownloadResult(
  url: string,
  token: string,
  payload: ModelDownloadCallbackInput,
  timeoutMs = 10_000
): Promise<void> {
  const res = await fetch(`${url}/api/models/download-callback`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new HttpError(res.status, `model download callback failed (${res.status}): ${text}`);
  }
}

// --- Pull queue (MULTIUSER_PLAN.md Stage 1) -- all Bearer-authenticated,
// either with a per-worker session token (Stage 3, the default for a newly
// enrolled worker) or the legacy shared secret (config.worker_shared_token,
// still supported for an already-deployed Stage 1 worker -- see
// server/src/worker-auth.ts's dual-mode authenticateWorker). The wire shape
// is identical either way; only which token worker/src/index.ts passes in
// changes. ---

function authHeader(token: string): Record<string, string> {
  return { authorization: `Bearer ${token}` };
}

// Carries the HTTP status alongside the message so a caller can distinguish
// "credential rejected, worth trying to refresh" (401) from any other
// failure (network error, 5xx, malformed request) without parsing the
// message string -- see worker/src/index.ts's withAuth, which retries once
// after a refresh on exactly this error/status combination.
export class HttpError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "HttpError";
  }
}

export async function postHeartbeat(
  url: string,
  token: string,
  state: WorkerStatePush,
  activeJob: ActiveJobReport | null,
  // Concurrently-running download_model jobs (index.ts's downloadJobPool) --
  // reported alongside activeJob, not exclusive with it. See
  // server/src/validate-worker-state.ts's parseActiveDownloads.
  activeDownloads: ActiveJobReport[],
  timeoutMs = 10_000
): Promise<HeartbeatResponse> {
  const res = await fetch(`${url}/api/worker/heartbeat`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify({ ...state, active_job: activeJob ?? undefined, active_downloads: activeDownloads }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new HttpError(res.status, `heartbeat failed (${res.status}): ${await res.text()}`);
  return (await res.json()) as HeartbeatResponse;
}

// Only ever called while idle. Hangs up to ~25s server-side (its own
// long-poll window) -- timeoutMs here must stay comfortably above that, or
// this call would abort spuriously on every poll that didn't get a job
// immediately.
export async function pollQueue(
  url: string,
  token: string,
  state: WorkerStatePush,
  timeoutMs: number
): Promise<QueueJob | null> {
  const res = await fetch(`${url}/api/worker/queue`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify(state),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (res.status === 204) return null;
  if (!res.ok) throw new HttpError(res.status, `queue poll failed (${res.status}): ${await res.text()}`);
  return (await res.json()) as QueueJob;
}

// The ONLY thing that ever moves a claimed job out of 'claimed' on success --
// see server/src/routes/queue.ts's doc comment on this endpoint for why it's
// required (without it, the reaper eventually treats a finished job as if
// the worker had crashed and re-hands it out).
export async function reportJobResult(
  url: string,
  token: string,
  machineId: string,
  jobId: string,
  outcome: { ok: true } | { ok: false; error: string },
  timeoutMs = 10_000
): Promise<void> {
  // Same body-based identity convention authenticateWorker expects on every
  // JSON-bodied worker route (queue/heartbeat) -- machine_id must travel in
  // the body itself, not just the Bearer token (which only proves "some
  // worker," not "which one").
  const res = await fetch(`${url}/api/worker/jobs/${jobId}/complete`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader(token) },
    body: JSON.stringify({ machine_id: machineId, ...outcome }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new HttpError(res.status, `job completion report failed (${res.status}): ${await res.text()}`);
}

// Pushes a completed run's gzipped log file -- see server/src/routes/runs.ts's
// POST /api/runs/:id/log. Identified by X-Machine-Id (not machine_id in a
// JSON body, since the body here is raw gzip bytes) so the server can verify
// this machine actually executed that run before accepting it.
export async function pushRunLog(
  url: string,
  token: string,
  machineId: string,
  runId: string,
  gzippedBytes: Buffer,
  timeoutMs = 30_000
): Promise<void> {
  const res = await fetch(`${url}/api/runs/${runId}/log`, {
    method: "POST",
    headers: {
      "content-type": "application/octet-stream",
      "x-machine-id": machineId,
      ...authHeader(token),
    },
    body: gzippedBytes,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new HttpError(res.status, `run log push failed (${res.status}): ${await res.text()}`);
}

// --- Device enrolment (MULTIUSER_PLAN.md §3.1/§3.5) -- unauthenticated,
// called before this worker holds any credential at all. ---

export async function startDeviceEnrolment(
  url: string,
  info: { machine_id: string; hostname: string; platform: string; arch: string; hardware: HardwareInfo },
  timeoutMs = 10_000
): Promise<DeviceStartResponse> {
  const res = await fetch(`${url}/api/device/start`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(info),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`device enrolment start failed (${res.status}): ${await res.text()}`);
  return (await res.json()) as DeviceStartResponse;
}

export type DeviceTokenPoll =
  | { state: "approved"; session_token: string; refresh_token: string }
  | { state: "pending" }
  | { state: "expired" };

// One poll of the device flow (§3.1 step 3) -- the caller loops this on its
// own `interval` cadence (from DeviceStartResponse) until "approved" or
// "expired". A 400 with a recognized `error` body is a normal, expected
// outcome here (not thrown as an error) since "still waiting on a human" is
// what most polls return; only a genuinely unreachable server or unexpected
// response shape throws.
export async function pollDeviceToken(url: string, deviceCode: string, timeoutMs = 10_000): Promise<DeviceTokenPoll> {
  const res = await fetch(`${url}/api/device/token`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ device_code: deviceCode }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (res.ok) {
    const body = (await res.json()) as DeviceTokenSuccess;
    return { state: "approved", session_token: body.session_token, refresh_token: body.refresh_token };
  }
  if (res.status === 400) {
    const body = (await res.json()) as DeviceTokenError;
    return body.error === "authorization_pending" ? { state: "pending" } : { state: "expired" };
  }
  throw new Error(`device token poll failed (${res.status}): ${await res.text()}`);
}

// Rotates a worker session (MULTIUSER_PLAN.md §2.5/§3.5) -- called reactively
// by worker/src/index.ts when a session-authenticated call comes back 401,
// not on a timer: the session's own sliding expiry (server/src/db/repo.ts's
// sessionRepo.touch, refreshed on every authenticated call up to once/hour)
// means an actively-heartbeating worker's session effectively never expires
// on its own, so there is no fixed schedule to refresh against in the first
// place. Authenticated with the REFRESH token, a different token space than
// the session token every other vps-client.ts function sends.
export async function refreshWorkerSession(
  url: string,
  refreshToken: string,
  timeoutMs = 10_000
): Promise<RefreshResponse> {
  const res = await fetch(`${url}/api/auth/refresh`, {
    method: "POST",
    headers: authHeader(refreshToken),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new HttpError(res.status, `session refresh failed (${res.status}): ${await res.text()}`);
  return (await res.json()) as RefreshResponse;
}
