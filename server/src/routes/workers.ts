import type { FastifyInstance } from "fastify";
import { listWorkers } from "../config.js";
import { repo } from "../db/repo.js";
import { getReleases, filterReleasesForWorker } from "../github-releases.js";
import { describeWorkerError } from "../worker-errors.js";
import { isMtpDraftModel } from "../../../shared/types.js";
import type {
  WorkerLlamaCppInfo,
  InstalledBuild,
  HardwareInfo,
  HfRepoSearchResult,
  HfFileEntry,
  DownloadProgress,
  WorkerCurrentRun,
  ModelMetadata,
  ModelDownloadCallbackInput,
} from "../../../shared/types.js";
import { HF_REPO_PATTERN, searchHfGgufModels, listHfGgufFiles, getHfGgufMeta } from "../hf.js";

const WORKER_READ_TIMEOUT_MS = 15_000;
const WORKER_INSTALL_TIMEOUT_MS = 10 * 60 * 1000;

function validateModelDownloadCallback(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return "payload must be an object";
  const p = payload as Record<string, unknown>;
  if (typeof p.worker !== "string" || !p.worker) return "worker must be a non-empty string";
  if (typeof p.hf_repo !== "string" || !p.hf_repo) return "hf_repo must be a non-empty string";
  if (typeof p.hf_file !== "string" || !p.hf_file) return "hf_file must be a non-empty string";
  if (typeof p.ok !== "boolean") return "ok must be a boolean";
  if (p.ok) {
    if (typeof p.sha256 !== "string" || !p.sha256) return "sha256 is required when ok is true";
    if (p.size_bytes !== undefined && typeof p.size_bytes !== "number") return "size_bytes must be a number";
  } else if (p.error !== undefined && typeof p.error !== "string") {
    return "error must be a string";
  }
  return null;
}

async function findWorker(name: string) {
  return (await listWorkers()).find((w) => w.name === name);
}

async function fetchWorker(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  return fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
}

export async function workersRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/workers", async () => {
    const list = await listWorkers();
    return {
      workers: list.map((w) => ({
        name: w.name,
        backend: w.backend,
        llama_cpp_build: w.llama_cpp_build,
      })),
    };
  });

  app.get<{ Params: { name: string } }>(
    "/api/workers/:name/llama-cpp",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });

      let workerInfo: {
        platform: string;
        arch: string;
        backend: string;
        installed: InstalledBuild[];
        active_tag: string | null;
      };
      try {
        const res = await fetchWorker(`${worker.url}/llama-cpp`, {}, WORKER_READ_TIMEOUT_MS);
        if (!res.ok) throw new Error(`worker responded ${res.status}`);
        workerInfo = (await res.json()) as typeof workerInfo;
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }

      // Best-effort and fetched in parallel with the releases lookup below --
      // this is diagnostic detail (what CPU/GPU the worker actually has), not
      // something the "available" list's filtering depends on, so a worker
      // that's slow or doesn't have this endpoint yet shouldn't block the
      // rest of the response.
      let hardware: HardwareInfo | undefined;
      const hardwarePromise = fetchWorker(`${worker.url}/hardware`, {}, WORKER_READ_TIMEOUT_MS)
        .then((res) => (res.ok ? (res.json() as Promise<HardwareInfo>) : undefined))
        .catch(() => undefined);

      // Same best-effort story: lets the Workers page/dashboard show what a
      // worker is currently busy with without a separate round trip, but a
      // worker that's slow or predates /health's current_run field shouldn't
      // block the rest of this response either.
      let health: { busy: boolean; current_run: WorkerCurrentRun | null } | undefined;
      const healthPromise = fetchWorker(`${worker.url}/health`, {}, WORKER_READ_TIMEOUT_MS)
        .then((res) => (res.ok ? (res.json() as Promise<typeof health>) : undefined))
        .catch(() => undefined);

      let available: WorkerLlamaCppInfo["available"] = [];
      try {
        const releases = await getReleases();
        available = filterReleasesForWorker(releases, workerInfo.platform, workerInfo.arch, workerInfo.backend);
      } catch (err) {
        // GitHub being unreachable shouldn't block viewing installed builds --
        // just report an empty "installable" list.
        app.log.warn(`github releases fetch failed: ${err instanceof Error ? err.message : err}`);
      }

      const latestTag = available[0]?.tag;
      const updateAvailable = Boolean(
        latestTag && latestTag !== workerInfo.active_tag && !workerInfo.installed.some((b) => b.tag === latestTag)
      );
      hardware = await hardwarePromise;
      health = await healthPromise;

      const info: WorkerLlamaCppInfo = {
        worker_name: worker.name,
        platform: workerInfo.platform,
        arch: workerInfo.arch,
        backend: workerInfo.backend,
        installed: workerInfo.installed,
        active_tag: workerInfo.active_tag,
        available,
        update_available: updateAvailable,
        hardware,
        busy: health?.busy,
        current_run: health?.current_run,
      };
      return info;
    }
  );

  app.post<{ Params: { name: string }; Body: { tag?: string; asset_name?: string } }>(
    "/api/workers/:name/llama-cpp/install",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      const { tag, asset_name } = request.body ?? {};
      if (!tag || !asset_name) {
        return reply.code(400).send({ error: "tag and asset_name are required" });
      }

      // Resolve the download URL server-side from the cached GitHub data --
      // never take a URL straight from the request body. The worker will
      // download, extract, and eventually exec whatever's at that URL, so
      // the orchestrator only ever hands it a URL it fetched from GitHub
      // itself for this exact tag/asset pair.
      let downloadUrl: string | undefined;
      try {
        const releases = await getReleases();
        const release = releases.find((r) => r.tag === tag);
        downloadUrl = release?.assets.find((a) => a.name === asset_name)?.download_url;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return reply.code(502).send({ error: `could not fetch release info: ${message}` });
      }
      if (!downloadUrl) {
        return reply.code(400).send({ error: `no matching asset ${asset_name} for tag ${tag}` });
      }

      request.log.info({ worker: worker.name, tag, asset_name }, "llama-cpp install requested");
      try {
        const res = await fetchWorker(
          `${worker.url}/llama-cpp/install`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ tag, asset_name, download_url: downloadUrl }),
          },
          WORKER_INSTALL_TIMEOUT_MS
        );
        const data = await res.json();
        return reply.code(res.status).send(data);
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  app.post<{ Params: { name: string }; Body: { tag?: string } }>(
    "/api/workers/:name/llama-cpp/activate",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      if (!request.body?.tag) return reply.code(400).send({ error: "tag is required" });
      try {
        const res = await fetchWorker(
          `${worker.url}/llama-cpp/activate`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ tag: request.body.tag }),
          },
          WORKER_READ_TIMEOUT_MS
        );
        const data = await res.json();
        return reply.code(res.status).send(data);
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  // Pause/resume/stop a worker's in-flight run -- thin proxies to the
  // worker's own /run/pause, /run/resume, /run/stop (worker/src/index.ts).
  // A worker only ever runs one benchmark at a time, so these target the
  // worker itself rather than a specific run id; the worker 409s if it
  // isn't actually busy.
  app.post<{ Params: { name: string } }>("/api/workers/:name/pause", async (request, reply) => {
    const worker = await findWorker(request.params.name);
    if (!worker) return reply.code(404).send({ error: "unknown worker" });
    try {
      const res = await fetchWorker(`${worker.url}/run/pause`, { method: "POST" }, WORKER_READ_TIMEOUT_MS);
      const data = await res.json();
      return reply.code(res.status).send(data);
    } catch (err) {
      return reply.code(502).send({ error: describeWorkerError(err) });
    }
  });

  app.post<{ Params: { name: string } }>("/api/workers/:name/resume", async (request, reply) => {
    const worker = await findWorker(request.params.name);
    if (!worker) return reply.code(404).send({ error: "unknown worker" });
    try {
      const res = await fetchWorker(`${worker.url}/run/resume`, { method: "POST" }, WORKER_READ_TIMEOUT_MS);
      const data = await res.json();
      return reply.code(res.status).send(data);
    } catch (err) {
      return reply.code(502).send({ error: describeWorkerError(err) });
    }
  });

  app.post<{ Params: { name: string } }>("/api/workers/:name/stop", async (request, reply) => {
    const worker = await findWorker(request.params.name);
    if (!worker) return reply.code(404).send({ error: "unknown worker" });
    try {
      const res = await fetchWorker(`${worker.url}/run/stop`, { method: "POST" }, WORKER_READ_TIMEOUT_MS);
      const data = await res.json();
      if (res.status === 409) {
        // The worker disagrees it's running anything -- most likely it
        // already finished or restarted and our DB row never got the memo
        // (see repo.reconcileStaleRun). Reconcile instead of bubbling the
        // 409 back to a Stop button the user would otherwise keep clicking
        // forever with no way to clear the "running" state.
        const stale = repo.getRunningRunForWorker(worker.name);
        if (stale) {
          request.log.warn(
            { worker: worker.name, run_id: stale.id },
            "stop hit a worker that wasn't busy -- reconciling stale run"
          );
          repo.reconcileStaleRun(stale.id, "worker lost track of run");
          return reply.code(200).send({ ok: true, stopping: true, reconciled: true });
        }
      }
      return reply.code(res.status).send(data);
    } catch (err) {
      return reply.code(502).send({ error: describeWorkerError(err) });
    }
  });

  app.delete<{ Params: { name: string; tag: string } }>(
    "/api/workers/:name/llama-cpp/:tag",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      try {
        const res = await fetchWorker(
          `${worker.url}/llama-cpp/${encodeURIComponent(request.params.tag)}`,
          { method: "DELETE" },
          WORKER_READ_TIMEOUT_MS
        );
        const data = await res.json();
        return reply.code(res.status).send(data);
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  app.get<{ Params: { name: string }; Querystring: { hf_repo?: string; hf_file?: string } }>(
    "/api/workers/:name/models/download/progress",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      const { hf_repo, hf_file } = request.query;
      if (!hf_repo || !hf_file) {
        return reply.code(400).send({ error: "hf_repo and hf_file are required" });
      }
      try {
        const res = await fetchWorker(
          `${worker.url}/models/download/progress?hf_repo=${encodeURIComponent(hf_repo)}&hf_file=${encodeURIComponent(hf_file)}`,
          {},
          WORKER_READ_TIMEOUT_MS
        );
        const data = (await res.json()) as DownloadProgress | { error: string };
        return reply.code(res.status).send(data);
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  app.delete<{ Params: { name: string }; Querystring: { file?: string } }>(
    "/api/workers/:name/models",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      const { file } = request.query;
      if (!file) return reply.code(400).send({ error: "file is required" });
      try {
        const res = await fetchWorker(
          `${worker.url}/models?file=${encodeURIComponent(file)}`,
          { method: "DELETE" },
          WORKER_READ_TIMEOUT_MS
        );
        const data = await res.json();
        return reply.code(res.status).send(data);
      } catch (err) {
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  app.get<{ Querystring: { q?: string } }>("/api/hf/search", async (request, reply) => {
    const q = (request.query.q ?? "").trim();
    if (!q) return { results: [] as HfRepoSearchResult[] };
    try {
      const results: HfRepoSearchResult[] = await searchHfGgufModels(q, WORKER_READ_TIMEOUT_MS);
      return { results };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return reply.code(502).send({ error: message });
    }
  });

  app.get<{ Params: { "*": string } }>("/api/hf/repo/*", async (request, reply) => {
    const repoId = request.params["*"];
    if (!repoId || !HF_REPO_PATTERN.test(repoId)) {
      return reply.code(400).send({ error: "invalid repo id" });
    }
    try {
      const files: HfFileEntry[] = await listHfGgufFiles(repoId, WORKER_READ_TIMEOUT_MS);
      return { files };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return reply.code(502).send({ error: message });
    }
  });

  // Just triggers the download and returns as soon as the worker acks --
  // see worker/src/index.ts's POST /models/download for why this no longer
  // waits out the (potentially tens-of-minutes) file transfer itself: doing
  // so here too would still be bounded by Node's undici default per-request
  // timeout (5 minutes) regardless of any timeout this route configured,
  // silently reporting a healthy worker as "inaccessible" on any download
  // slower than that. The actual completion (success or failure) arrives
  // later via the worker's own callback to POST /api/models/download-callback
  // below, mirroring how /run's per-item results are reported rather than
  // returned on the trigger response.
  app.post<{ Params: { name: string }; Body: { hf_repo?: string; hf_file?: string } }>(
    "/api/workers/:name/models/download",
    async (request, reply) => {
      const worker = await findWorker(request.params.name);
      if (!worker) return reply.code(404).send({ error: "unknown worker" });
      const { hf_repo, hf_file } = request.body ?? {};
      if (!hf_repo || !hf_file) {
        return reply.code(400).send({ error: "hf_repo and hf_file are required" });
      }
      request.log.info({ worker: worker.name, hf_repo, hf_file }, "model download requested");
      try {
        const res = await fetchWorker(
          `${worker.url}/models/download`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ hf_repo, hf_file }),
          },
          WORKER_READ_TIMEOUT_MS
        );
        const data = await res.json();
        if (!res.ok) {
          request.log.warn(
            { worker: worker.name, hf_repo, hf_file, error: (data as { error?: string }).error },
            "model download rejected"
          );
        }
        return reply.code(res.status).send(data);
      } catch (err) {
        request.log.error({ worker: worker.name, hf_repo, hf_file, err: describeWorkerError(err) }, "model download error");
        return reply.code(502).send({ error: describeWorkerError(err) });
      }
    }
  );

  // Worker -> server callback reporting a download's terminal outcome (see
  // the trigger route above). Retried by the worker with backoff
  // (safeReportDownloadResult), so this must stay safe to receive more than
  // once for the same download -- repo.registerModel's upsert-on-id
  // semantics already guarantee that for the success path.
  app.post<{ Body: ModelDownloadCallbackInput }>(
    "/api/models/download-callback",
    // Same rationale as /api/runs/:id/items/:idx's logLevel below -- this
    // isn't a user-facing request, no need for Fastify's automatic pair.
    { logLevel: "silent" },
    async (request, reply) => {
      const validationError = validateModelDownloadCallback(request.body);
      if (validationError) {
        app.log.warn({ error: validationError }, "model download callback rejected: invalid payload");
        return reply.code(400).send({ error: validationError });
      }
      const { worker, hf_repo, hf_file, ok, error, sha256, size_bytes, n_layer, mtp_layers } = request.body;
      if (!ok) {
        app.log.error({ worker, hf_repo, hf_file, error }, "model download failed");
        return reply.code(200).send({ ok: true });
      }
      try {
        // See shared/types.ts's isMtpDraftModel for the full detection story
        // (content-based: has an MTP head but no real transformer stack of
        // its own; filename-based: the real-world "MTP/"-folder or
        // "mtp-"-prefixed convention) -- computed here from this download's
        // own metadata so it can be persisted, but every *read* site
        // recomputes it live from stored metadata too rather than trusting
        // this stored flag alone.
        const { param_count } = await getHfGgufMeta(hf_repo, WORKER_READ_TIMEOUT_MS);
        const metadata: ModelMetadata = {
          ...(typeof n_layer === "number" ? { n_layer } : {}),
          ...(typeof param_count === "number" ? { param_count } : {}),
          ...(typeof mtp_layers === "number" && mtp_layers > 0 ? { mtp_layers } : {}),
        };
        if (isMtpDraftModel({ metadata, hf_file, filename: hf_file })) {
          metadata.mtp_role = "draft";
        }
        app.log.info(
          { worker, hf_repo, hf_file, size_bytes, n_layer, param_count, mtp_layers, mtp_role: metadata.mtp_role },
          "model download complete"
        );
        repo.registerModel({
          id: sha256,
          filename: hf_file,
          size_bytes: size_bytes ?? 0,
          source: "huggingface",
          hf_repo,
          hf_file,
          metadata,
        });
        return reply.code(200).send({ ok: true });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        app.log.error({ worker, hf_repo, hf_file, err: message }, "model registration failed after download");
        return reply.code(500).send({ error: message });
      }
    }
  );
}
