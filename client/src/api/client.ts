import {
  WORKER_INACCESSIBLE_MESSAGE,
  type Model,
  type RegisterModelInput,
  type Run,
  type ResultRow,
  type RunItem,
  type TriggerPayload,
  type WorkerLlamaCppInfo,
  type HfRepoSearchResult,
  type HfFileEntry,
  type Backend,
  type InstalledBuild,
  type DownloadProgress,
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

export interface WorkerListEntry {
  name: string;
  backend: Backend;
  llama_cpp_build: string;
}

export const api = {
  listModels: (): Promise<Model[]> => request<{ models: Model[] }>("/api/models").then((d) => d.models),

  registerModel: (body: RegisterModelInput): Promise<Model> =>
    request<{ model: Model }>("/api/models", postJson(body)).then((d) => d.model),

  deleteModel: (
    id: string
  ): Promise<{ ok: true; file_deletions: { worker: string; ok: boolean; error?: string }[] }> =>
    request(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" }),

  // Asks every configured worker to read the model's GGUF header from
  // whichever of them already has the file on disk -- backfills n_layer for
  // models registered before that existed, without re-downloading anything.
  backfillModelLayerCount: (id: string): Promise<{ ok: true; n_layer: number | null }> =>
    request(`/api/models/${encodeURIComponent(id)}/backfill-layer-count`, { method: "POST" }),

  // Looks up a model's real total parameter count from Hugging Face's own
  // GGUF-derived metadata -- backfills models registered before this
  // existed. Never touches a worker (unlike layer count), just the model's
  // hf_repo.
  backfillModelParamCount: (id: string): Promise<{ ok: true; param_count: number | null }> =>
    request(`/api/models/${encodeURIComponent(id)}/backfill-param-count`, { method: "POST" }),

  // Which configured worker(s) currently have each model's file on disk --
  // powers the Models page's Local/Remote split. Live, not cached server-side.
  getModelLocations: (): Promise<{ locations: Record<string, string[]>; unreachable: string[] }> =>
    request("/api/models/locations"),

  // Deletes a model's file from exactly one worker's model_dir -- never
  // touches the model registry row (see server/src/routes/workers.ts's
  // DELETE /api/workers/:name/models, unchanged, just not previously called
  // from the client). Distinct from the old whole-registry deleteModel above.
  deleteModelFileFromWorker: (workerName: string, filename: string): Promise<{ ok: true }> =>
    request(`/api/workers/${encodeURIComponent(workerName)}/models?file=${encodeURIComponent(filename)}`, {
      method: "DELETE",
    }),

  listWorkers: (): Promise<WorkerListEntry[]> =>
    request<{ workers: WorkerListEntry[] }>("/api/workers").then((d) => d.workers),

  getWorkerLlamaCpp: (name: string): Promise<WorkerLlamaCppInfo> =>
    request<WorkerLlamaCppInfo>(`/api/workers/${encodeURIComponent(name)}/llama-cpp`),

  installBuild: (name: string, tag: string, assetName: string): Promise<{ ok: true; build: InstalledBuild }> =>
    request(`/api/workers/${encodeURIComponent(name)}/llama-cpp/install`, postJson({ tag, asset_name: assetName })),

  activateBuild: (name: string, tag: string): Promise<{ ok: true; active_tag: string }> =>
    request(`/api/workers/${encodeURIComponent(name)}/llama-cpp/activate`, postJson({ tag })),

  deleteBuild: (name: string, tag: string): Promise<{ ok: true }> =>
    request(`/api/workers/${encodeURIComponent(name)}/llama-cpp/${encodeURIComponent(tag)}`, { method: "DELETE" }),

  searchHf: (q: string): Promise<HfRepoSearchResult[]> =>
    request<{ results: HfRepoSearchResult[] }>(`/api/hf/search?q=${encodeURIComponent(q)}`).then((d) => d.results),

  listHfFiles: (repo: string): Promise<HfFileEntry[]> =>
    request<{ files: HfFileEntry[] }>(
      `/api/hf/repo/${repo.split("/").map(encodeURIComponent).join("/")}`
    ).then((d) => d.files),

  // Resolves once the worker acks that it has started the download, not
  // once the download finishes -- that now happens in the background and is
  // only observable by polling getDownloadProgress until it stops reporting
  // (see Models.tsx's poll effect).
  downloadHfFile: (workerName: string, hfRepo: string, hfFile: string): Promise<void> =>
    request<void>(
      `/api/workers/${encodeURIComponent(workerName)}/models/download`,
      postJson({ hf_repo: hfRepo, hf_file: hfFile })
    ),

  getDownloadProgress: (workerName: string, hfRepo: string, hfFile: string): Promise<DownloadProgress> =>
    request(
      `/api/workers/${encodeURIComponent(workerName)}/models/download/progress` +
        `?hf_repo=${encodeURIComponent(hfRepo)}&hf_file=${encodeURIComponent(hfFile)}`
    ),

  listRuns: (): Promise<Run[]> => request<{ runs: Run[] }>("/api/runs").then((d) => d.runs),

  getRun: (id: string): Promise<{ run: Run; results: ResultRow[]; items: RunItem[]; paused?: boolean }> =>
    request(`/api/runs/${encodeURIComponent(id)}`),

  triggerRun: (payload: TriggerPayload): Promise<Run> =>
    request<{ run: Run }>("/api/runs/trigger", postJson(payload)).then((d) => d.run),

  // Pause/resume/stop target the worker (it only ever runs one benchmark at
  // a time), not a specific run id -- see server/src/routes/workers.ts.
  pauseWorker: (workerName: string): Promise<{ ok: true; paused: true }> =>
    request(`/api/workers/${encodeURIComponent(workerName)}/pause`, postJson({})),

  resumeWorker: (workerName: string): Promise<{ ok: true; paused: false }> =>
    request(`/api/workers/${encodeURIComponent(workerName)}/resume`, postJson({})),

  stopWorker: (workerName: string): Promise<{ ok: true; stopping: true }> =>
    request(`/api/workers/${encodeURIComponent(workerName)}/stop`, postJson({})),

  exportUrl: (format: "json" | "csv" | "md", runIds?: string[]): string => {
    const params = new URLSearchParams({ format });
    if (runIds?.length) params.set("runs", runIds.join(","));
    return `/api/results/export?${params.toString()}`;
  },

  getAiStatus: (): Promise<{ configured: boolean; model?: string }> => request("/api/ai/status"),

  // Live (not persisted) Hugging Face check, keyed by model id -- powers the
  // New Run model picker's "Updated X ago" label and "possibly newer on HF"
  // hint. See server/src/routes/models.ts's /api/models/hf-updates.
  getModelHfUpdates: (): Promise<Record<string, string | null>> =>
    request<{ updates: Record<string, string | null> }>("/api/models/hf-updates").then((d) => d.updates),
};
