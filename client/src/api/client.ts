import {
  WORKER_INACCESSIBLE_MESSAGE,
  type Model,
  type RegisterModelInput,
  type Run,
  type ResultRow,
  type RunItem,
  type TriggerPayload,
  type Worker,
  type WorkerVramInfo,
  type HfRepoSearchResult,
  type HfFileEntry,
  type LlamaCppRelease,
  type AuthStatus,
  type SessionInfo,
  type IdentityInfo,
  type DeviceStatusResponse,
  type DeviceApproveResponse,
} from "../types";

export class ApiError extends Error {
  status: number;
  inaccessible: boolean;
  constructor(message: string, status: number, inaccessible = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.inaccessible = inaccessible;
  }
}

function extractErrorMessage(data: unknown, status: number): string {
  if (data && typeof data === "object" && "error" in data) {
    const err = (data as { error?: unknown }).error;
    if (typeof err === "string") return err;
  }
  return `request failed: ${status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    // Same-origin call to our own server failing at the network level (dev
    // proxy not up, server down) -- distinct from the server reporting a
    // *worker* as inaccessible below, but the caller-facing shape is the same.
    throw new ApiError(err instanceof Error ? err.message : String(err), 0);
  }
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    const message = extractErrorMessage(data, res.status);
    throw new ApiError(message, res.status, message === WORKER_INACCESSIBLE_MESSAGE);
  }
  return data as T;
}

function postJson(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const api = {
  listModels: (): Promise<Model[]> => request<{ models: Model[] }>("/api/models").then((d) => d.models),

  registerModel: (body: RegisterModelInput): Promise<Model> =>
    request<{ model: Model }>("/api/models", postJson(body)).then((d) => d.model),

  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): no more file_deletion_queued_on
  // -- deleting the catalog row no longer fans out to every worker's
  // model_dir (most workers aren't even the caller's to touch, once
  // ownership exists). Freeing a file on a specific machine is
  // deleteModelFileFromWorker below, a separate, deliberate action.
  deleteModel: (id: string): Promise<{ ok: true }> =>
    request(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" }),

  // Reads n_layer/mtp_layers from whichever worker's cached model_files_json
  // already has this file (see server/src/db/repo.ts's findModelFileMeta) --
  // no live worker round trip anymore, just a DB read of the last heartbeat.
  backfillModelLayerCount: (id: string): Promise<{ ok: true; n_layer: number | null }> =>
    request(`/api/models/${encodeURIComponent(id)}/backfill-layer-count`, { method: "POST" }),

  // Looks up a model's real total parameter count from Hugging Face's own
  // GGUF-derived metadata -- backfills models registered before this
  // existed. Never touches a worker (unlike layer count), just the model's
  // hf_repo.
  backfillModelParamCount: (id: string): Promise<{ ok: true; param_count: number | null }> =>
    request(`/api/models/${encodeURIComponent(id)}/backfill-param-count`, { method: "POST" }),

  // Which machine(s) currently have each model's file, per their last
  // heartbeat -- powers the Models page's Local/Remote split.
  getModelLocations: (): Promise<{ locations: Record<string, string[]>; unreachable: string[] }> =>
    request("/api/models/locations"),

  // Queues deletion of a model's file on one machine's model_dir -- never
  // touches the model registry row. Fire-and-forget under the pull model:
  // resolves once the job is queued, not once the worker actually deletes
  // it (see server/src/routes/workers.ts).
  deleteModelFileFromWorker: (workerId: string, filename: string): Promise<{ ok: true; queued: true }> =>
    request(`/api/workers/${encodeURIComponent(workerId)}/models?file=${encodeURIComponent(filename)}`, {
      method: "DELETE",
    }),

  // One read of every machine's last-reported state -- hardware, installed
  // builds, model files, derived status, and (while a job is active) its
  // run id / live progress are all already inline on each Worker. No more
  // separate per-worker fetch (MULTIUSER_PLAN.md §1.16).
  listWorkers: (): Promise<Worker[]> => request<{ workers: Worker[] }>("/api/workers").then((d) => d.workers),

  // Permanently removes a machine from the Workers page -- distinct from
  // Settings' session revoke, which only kicks it off and leaves the row
  // (and card) in place. Run history is untouched (see server/src/db/
  // repo.ts's deleteWorker doc comment); reconnecting afterward re-enrols as
  // a brand-new machine.
  deleteWorker: (workerId: string): Promise<{ ok: true }> =>
    request(`/api/workers/${encodeURIComponent(workerId)}`, { method: "DELETE" }),

  // Which llama.cpp releases a specific machine could install, and whether
  // a newer one than what it has exists -- the one piece of per-worker
  // "live" info that's a GitHub lookup, not something the worker itself
  // reports (see server/src/routes/workers.ts).
  getAvailableBuilds: (workerId: string): Promise<{ available: LlamaCppRelease[]; update_available: boolean }> =>
    request(`/api/workers/${encodeURIComponent(workerId)}/available-builds`),

  // Fire-and-forget: queues the install/activate/delete on the machine,
  // resolves once queued, not once the machine actually does it (no more
  // outbound HTTP to workers -- MULTIUSER_PLAN.md §1.11).
  installBuild: (workerId: string, tag: string, assetName: string): Promise<{ ok: true; queued: true }> =>
    request(
      `/api/workers/${encodeURIComponent(workerId)}/llama-cpp/install`,
      postJson({ tag, asset_name: assetName })
    ),

  // Cached free-VRAM reading (last heartbeat, not a live proxy call -- see
  // shared/types.ts's WorkerVramInfo doc comment), for NewRun.tsx's
  // pre-flight VRAM-fit banner (see shared/vramEstimate.ts).
  getWorkerVram: (workerId: string): Promise<WorkerVramInfo> =>
    request<WorkerVramInfo>(`/api/workers/${encodeURIComponent(workerId)}/vram`),

  activateBuild: (workerId: string, tag: string): Promise<{ ok: true; queued: true }> =>
    request(`/api/workers/${encodeURIComponent(workerId)}/llama-cpp/activate`, postJson({ tag })),

  deleteBuild: (workerId: string, tag: string): Promise<{ ok: true; queued: true }> =>
    request(`/api/workers/${encodeURIComponent(workerId)}/llama-cpp/${encodeURIComponent(tag)}`, {
      method: "DELETE",
    }),

  searchHf: (q: string): Promise<HfRepoSearchResult[]> =>
    request<{ results: HfRepoSearchResult[] }>(`/api/hf/search?q=${encodeURIComponent(q)}`).then((d) => d.results),

  listHfFiles: (repo: string): Promise<HfFileEntry[]> =>
    request<{ files: HfFileEntry[] }>(
      `/api/hf/repo/${repo.split("/").map(encodeURIComponent).join("/")}`
    ).then((d) => d.files),

  // Resolves once the download is queued, not once it finishes -- progress
  // is now read off the matching Worker's activeJobProgress (see
  // listWorkers above), polled the same way the rest of that machine's live
  // state already is, instead of a separate per-file progress endpoint.
  downloadHfFile: (workerId: string, hfRepo: string, hfFile: string): Promise<{ ok: true; queued: true }> =>
    request<{ ok: true; queued: true }>(
      `/api/workers/${encodeURIComponent(workerId)}/models/download`,
      postJson({ hf_repo: hfRepo, hf_file: hfFile })
    ),

  listRuns: (): Promise<Run[]> => request<{ runs: Run[] }>("/api/runs").then((d) => d.runs),

  getRun: (id: string): Promise<{ run: Run; results: ResultRow[]; items: RunItem[]; paused?: boolean }> =>
    request(`/api/runs/${encodeURIComponent(id)}`),

  triggerRun: (payload: TriggerPayload): Promise<Run> =>
    request<{ run: Run }>("/api/runs/trigger", postJson(payload)).then((d) => d.run),

  // Pause/resume/stop target the RUN now, not a worker-scoped endpoint --
  // the right shape once one account can own several machines
  // (MULTIUSER_PLAN.md §1.14). Delivered to the worker on its next
  // heartbeat (≤10s), not synchronously.
  pauseRun: (runId: string): Promise<{ ok: true }> => request(`/api/runs/${encodeURIComponent(runId)}/pause`, postJson({})),

  resumeRun: (runId: string): Promise<{ ok: true }> =>
    request(`/api/runs/${encodeURIComponent(runId)}/resume`, postJson({})),

  stopRun: (runId: string): Promise<{ run: Run }> => request(`/api/runs/${encodeURIComponent(runId)}/stop`, postJson({})),

  exportUrl: (format: "json" | "csv" | "md", runIds?: string[]): string => {
    const params = new URLSearchParams({ format });
    if (runIds?.length) params.set("runs", runIds.join(","));
    return `/api/results/export?${params.toString()}`;
  },

  // The worker pushes its log on job completion; this just reads it back
  // off disk (see server/src/routes/runs.ts's GET /api/runs/:id/log).
  runLogUrl: (id: string): string => `/api/runs/${encodeURIComponent(id)}/log`,

  getAiStatus: (): Promise<{ configured: boolean; model?: string; quota?: { remainingHour: number; remainingDay: number } }> =>
    request("/api/ai/status"),

  // Multi-user Stage 2 (MULTIUSER_PLAN.md §2) -- always reachable regardless
  // of AUTH_ENABLED (see routes/auth.ts's own doc comment); `authEnabled`
  // lets the SPA decide whether to gate on `user` at all.
  getAuthStatus: (): Promise<AuthStatus> => request("/api/auth/status"),

  listSessions: (): Promise<SessionInfo[]> => request<{ sessions: SessionInfo[] }>("/api/sessions").then((d) => d.sessions),

  revokeSession: (id: string): Promise<void> =>
    request(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),

  revokeAllOtherSessions: (): Promise<void> => request("/api/sessions/revoke-all", postJson({})),

  listIdentities: (): Promise<IdentityInfo[]> =>
    request<{ identities: IdentityInfo[] }>("/api/auth/identities").then((d) => d.identities),

  // Multi-user Stage 5 (MULTIUSER_PLAN.md §5.4) -- Settings' own toggle for
  // the community-aggregates consent flag.
  setShareBenchmarks: (enabled: boolean): Promise<{ shareBenchmarks: boolean }> =>
    request("/api/auth/share-benchmarks", postJson({ enabled })),

  // Security checklist's own "Account deletion ships in v1" item --
  // irreversible, so the server itself also requires this exact
  // {confirm: true} body, not just a bare DELETE. Spread order matters:
  // method must come AFTER postJson's own (which is "POST") to actually win.
  deleteAccount: (): Promise<{ ok: true }> => request("/api/auth/account", { ...postJson({ confirm: true }), method: "DELETE" }),

  // Multi-user Stage 3 (MULTIUSER_PLAN.md §3.1) -- the "Add machine"/"/device"
  // screen's own poll and approve action. Never 404s/errors for an
  // unknown/expired code (see server/src/routes/device.ts) -- it resolves to
  // {state: "not_found"}, so this always succeeds; only approveDevice can
  // reject (unknown code, already approved).
  getDeviceStatus: (userCode: string): Promise<DeviceStatusResponse> =>
    request(`/api/device/status?user_code=${encodeURIComponent(userCode)}`),

  approveDevice: (userCode: string, confirmDuplicate = false): Promise<DeviceApproveResponse> =>
    request("/api/device/approve", postJson({ user_code: userCode, confirm_duplicate: confirmDuplicate })),

  // Live (not persisted) Hugging Face check, keyed by model id -- powers the
  // New Run model picker's "Updated X ago" label and "possibly newer on HF"
  // hint. See server/src/routes/models.ts's /api/models/hf-updates.
  getModelHfUpdates: (): Promise<Record<string, string | null>> =>
    request<{ updates: Record<string, string | null> }>("/api/models/hf-updates").then((d) => d.updates),
};
