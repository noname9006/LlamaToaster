import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { queueEvents } from "../queue-events.js";
import { getReleases, filterReleasesForWorker } from "../github-releases.js";
import { BadRequestError, ConflictError, NotFoundError } from "../errors.js";
import { resolveAuthUser, assertOwnsWorker } from "../auth-middleware.js";
import { authenticateWorker } from "../worker-auth.js";
import { isMtpDraftModel } from "../../../shared/types.js";
import type {
  HfRepoSearchResult,
  HfFileEntry,
  ModelMetadata,
  ModelDownloadCallbackInput,
  LlamaCppRelease,
  WorkerVramInfo,
} from "../../../shared/types.js";
import { HF_REPO_PATTERN, searchHfGgufModels, listHfGgufFiles, getHfGgufMeta, type HfSortField } from "../hf.js";

const WORKER_READ_TIMEOUT_MS = 5_000;

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

// Every route below that used to proxy an HTTP call to a worker now just
// enqueues a job and returns 202 -- there is no more outbound HTTP to
// workers anywhere in the server (MULTIUSER_PLAN.md §1.11). Rejecting an
// offline machine up front (same policy as /api/runs/trigger) gives the user
// immediate, clear feedback instead of a job that silently sits in the
// queue until the machine happens to reconnect.
//
// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): also the ownership gate for
// every action route below -- existence (404) is always checked before
// ownership (403), and ownership before the offline check (409), so an
// unauthorized caller never learns whether someone else's machine happens to
// be online.
function requireOnlineWorker(userId: string | undefined, id: string) {
  const worker = repo.workerRepo.getWorker(id);
  if (!worker) throw new NotFoundError("unknown machine");
  assertOwnsWorker(userId, worker.id);
  if (worker.status === "offline") {
    throw new ConflictError("that machine is offline -- start the worker and try again");
  }
  return worker;
}

export async function workersRoutes(app: FastifyInstance): Promise<void> {
  // One DB read of every machine's cached last-reported state -- no live
  // per-worker fan-out. Replaces the old Tailscale-discovery listing AND the
  // per-worker GET /api/workers/:name/llama-cpp fetch it used to require
  // (MULTIUSER_PLAN.md §1.16): hardware/installed builds/model files/status
  // are all already here.
  //
  // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.3/§4.4): scoped to the caller's
  // own machines once authenticated -- every machine in single-tenant mode
  // (AUTH_ENABLED off), unchanged from before this scoping existed.
  app.get("/api/workers", async (request) => {
    const authed = resolveAuthUser(request);
    return { workers: authed ? repo.workerRepo.listWorkersForUser(authed.user.id) : repo.workerRepo.listWorkers() };
  });

  // Which llama.cpp releases this specific machine could install (filtered
  // by its own reported platform/arch/backend) and whether a newer one than
  // whatever it has installed exists -- the one piece of the old per-worker
  // GET /api/workers/:name/llama-cpp response that couldn't just move onto
  // the Worker object itself, since it depends on a live GitHub lookup
  // (cached 5 minutes, see github-releases.ts) rather than anything the
  // worker reports about itself.
  app.get<{ Params: { id: string } }>("/api/workers/:id/available-builds", async (request, reply) => {
    const worker = repo.workerRepo.getWorker(request.params.id);
    if (!worker) return reply.code(404).send({ error: "unknown machine" });
    assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
    if (!worker.platform || !worker.arch || !worker.backend) {
      return { available: [] as LlamaCppRelease[], update_available: false };
    }
    let available: LlamaCppRelease[] = [];
    try {
      const releases = await getReleases();
      available = filterReleasesForWorker(releases, worker.platform, worker.arch, worker.backend);
    } catch (err) {
      // GitHub being unreachable shouldn't block viewing installed builds --
      // just report an empty "installable" list.
      app.log.warn(`github releases fetch failed: ${err instanceof Error ? err.message : err}`);
    }
    const latestTag = available.find((r) => r.assets.length > 0)?.tag;
    const activeTag = worker.installedBuilds.find((b) => b.active)?.tag;
    const updateAvailable = Boolean(
      latestTag && latestTag !== activeTag && !worker.installedBuilds.some((b) => b.tag === latestTag)
    );
    return { available, update_available: updateAvailable };
  });

  app.post<{ Params: { id: string }; Body: { tag?: string; asset_name?: string; download_url?: string } }>(
    "/api/workers/:id/llama-cpp/install",
    async (request, reply) => {
      const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
      const { tag, asset_name } = request.body ?? {};
      if (!tag || !asset_name) throw new BadRequestError("tag and asset_name are required");

      // Resolve the download URL server-side from the cached GitHub data --
      // never take a URL straight from the request body. The worker will
      // download, extract, and eventually exec whatever's at that URL, so
      // it only ever gets a URL this server fetched from GitHub itself for
      // this exact tag/asset pair.
      const releases = await getReleases();
      const release = releases.find((r) => r.tag === tag);
      const downloadUrl = release?.assets.find((a) => a.name === asset_name)?.download_url;
      if (!downloadUrl) throw new BadRequestError(`no matching asset ${asset_name} for tag ${tag}`);

      request.log.info({ worker: worker.id, tag, asset_name }, "llama-cpp install queued");
      repo.queueRepo.enqueueJob(worker.id, {
        type: "install_build",
        payload: { tag, asset_name, download_url: downloadUrl },
      });
      queueEvents.emit(worker.id);
      return reply.code(202).send({ ok: true, queued: true });
    }
  );

  app.post<{ Params: { id: string }; Body: { tag?: string } }>(
    "/api/workers/:id/llama-cpp/activate",
    async (request, reply) => {
      const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
      if (!request.body?.tag) throw new BadRequestError("tag is required");
      repo.queueRepo.enqueueJob(worker.id, { type: "activate_build", payload: { tag: request.body.tag } });
      queueEvents.emit(worker.id);
      return reply.code(202).send({ ok: true, queued: true });
    }
  );

  app.delete<{ Params: { id: string; tag: string } }>("/api/workers/:id/llama-cpp/:tag", async (request, reply) => {
    const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
    repo.queueRepo.enqueueJob(worker.id, { type: "delete_build", payload: { tag: request.params.tag } });
    queueEvents.emit(worker.id);
    return reply.code(202).send({ ok: true, queued: true });
  });

  // Queued like any other command (not pushed via HeartbeatResponse.control)
  // so it takes its place behind whatever's already running rather than
  // killing an in-progress benchmark -- see shared/types.ts's shutdown_worker.
  app.post<{ Params: { id: string } }>("/api/workers/:id/shutdown", async (request, reply) => {
    const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
    request.log.info({ worker: worker.id }, "shutdown queued");
    repo.queueRepo.enqueueJob(worker.id, { type: "shutdown_worker", payload: {} });
    queueEvents.emit(worker.id);
    return reply.code(202).send({ ok: true, queued: true });
  });

  // Permanently forgets this machine -- distinct from a session revoke
  // (Settings' "Active sessions"), which only kicks the machine off and
  // leaves the workers row (and its Workers-page card) in place forever
  // (schema.sql: rows are otherwise never deleted). Refuses a busy machine
  // (worker_jobs is ON DELETE CASCADE -- deleting mid-job would silently
  // orphan whatever's running) but allows idle/offline, unlike
  // requireOnlineWorker's routes above which need the opposite.
  app.delete<{ Params: { id: string } }>("/api/workers/:id", async (request, reply) => {
    const worker = repo.workerRepo.getWorker(request.params.id);
    if (!worker) throw new NotFoundError("unknown machine");
    assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
    if (worker.status === "busy") {
      throw new ConflictError("that machine is busy -- wait for it to finish, or cancel the run first");
    }
    repo.workerRepo.deleteWorker(worker.id);
    return reply.code(200).send({ ok: true });
  });

  // Cosmetic rename -- doesn't require the machine to be online (unlike the
  // requireOnlineWorker-gated routes above), since it's a server-side-only
  // label with no worker-side effect at all.
  app.patch<{ Params: { id: string }; Body: { display_name?: string } }>(
    "/api/workers/:id",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
      const raw = request.body?.display_name;
      if (typeof raw !== "string") throw new BadRequestError("display_name is required");
      const displayName = raw.trim();
      if (!displayName) throw new BadRequestError("display_name can't be empty");
      if (displayName.length > 100) throw new BadRequestError("display_name is too long (max 100 characters)");
      const updated = repo.workerRepo.renameWorker(worker.id, displayName);
      return reply.code(200).send({ worker: updated });
    }
  );

  app.delete<{ Params: { id: string }; Querystring: { file?: string } }>(
    "/api/workers/:id/models",
    async (request, reply) => {
      const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
      const { file } = request.query;
      if (!file) throw new BadRequestError("file is required");
      repo.queueRepo.enqueueJob(worker.id, { type: "delete_model_file", payload: { filename: file } });
      queueEvents.emit(worker.id);
      return reply.code(202).send({ ok: true, queued: true });
    }
  );

  // Cached free-VRAM reading -- worker/src/index.ts's collectState() reports
  // this on every idle heartbeat (see shared/types.ts's WorkerVramInfo doc
  // comment); there's no server->worker proxy to re-probe on demand anymore
  // (MULTIUSER_PLAN.md §1.11), so this just serves the last-known value.
  // Consumed by NewRun.tsx's pre-flight VRAM-fit banner (shared/vramEstimate.ts).
  app.get<{ Params: { id: string } }>("/api/workers/:id/vram", async (request, reply) => {
    const worker = repo.workerRepo.getWorker(request.params.id);
    if (!worker) return reply.code(404).send({ error: "unknown machine" });
    assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
    if (!worker.vram) {
      return reply.code(409).send({ error: "no VRAM reading from this machine yet" });
    }
    return worker.vram as WorkerVramInfo;
  });

  const HF_SORT_FIELDS = new Set<HfSortField>(["downloads", "likes", "createdAt", "lastModified"]);

  // q may be empty -- that's the Models page's default "top 15" listing
  // (search box empty, sort=downloads) shown before the user types anything.
  app.get<{ Querystring: { q?: string; sort?: string; direction?: string; cursor?: string } }>(
    "/api/hf/search",
    async (request, reply) => {
      const q = (request.query.q ?? "").trim();
      const sort = HF_SORT_FIELDS.has(request.query.sort as HfSortField) ? (request.query.sort as HfSortField) : undefined;
      const direction = request.query.direction === "1" ? 1 : request.query.direction === "-1" ? -1 : undefined;
      const cursor = request.query.cursor || undefined;
      try {
        const page = await searchHfGgufModels(q, WORKER_READ_TIMEOUT_MS, { sort, direction, cursor });
        return { results: page.items as HfRepoSearchResult[], next_cursor: page.nextCursor };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return reply.code(502).send({ error: message });
      }
    }
  );

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

  // Queues the download; the worker reports live byte progress via its
  // heartbeat's ActiveJobReport (phase "downloading") instead of the old
  // separate GET /models/download/progress polling route, and the terminal
  // outcome via POST /api/models/download-callback below, same as before.
  app.post<{ Params: { id: string }; Body: { hf_repo?: string; hf_file?: string } }>(
    "/api/workers/:id/models/download",
    async (request, reply) => {
      const worker = requireOnlineWorker(resolveAuthUser(request)?.user.id, request.params.id);
      const { hf_repo, hf_file } = request.body ?? {};
      if (!hf_repo || !hf_file) throw new BadRequestError("hf_repo and hf_file are required");
      request.log.info({ worker: worker.id, hf_repo, hf_file }, "model download queued");
      const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "download_model", payload: { hf_repo, hf_file } });
      queueEvents.emit(worker.id);
      // job_id lets the client target this specific download with the pause
      // route below -- multiple downloads for the same worker can be queued
      // and running concurrently now (worker/src/index.ts's downloadJobPool),
      // so hf_repo/hf_file alone no longer uniquely identifies "the" download.
      return reply.code(202).send({ ok: true, queued: true, job_id: jobId });
    }
  );

  // Pauses an in-flight (or still-queued) download -- flags the same
  // cancel_requested column a benchmark job's Stop button already uses
  // (requestCancel), delivered to the worker on its next heartbeat
  // (routes/queue.ts). The worker leaves the partial file's .part on disk
  // rather than deleting it (worker/src/index.ts's executeDownloadModelJob),
  // so "Resume" is just re-queuing the same download -- it picks the .part
  // file back up via an HTTP Range request automatically.
  app.post<{ Params: { id: string; jobId: string } }>(
    "/api/workers/:id/downloads/:jobId/pause",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
      const job = repo.queueRepo.getJob(request.params.jobId);
      if (!job || job.workerId !== worker.id || job.jobType !== "download_model") {
        throw new NotFoundError("unknown download");
      }
      repo.queueRepo.requestCancel(job.id);
      return reply.code(200).send({ ok: true });
    }
  );

  // Cancel Download -- unlike pause above, tells the worker to delete the
  // .part file too (worker/src/index.ts's discard handling), so there's
  // nothing left to resume; the client removes the row outright rather than
  // offering Resume. A still-'pending' download (never claimed, so no
  // partial file exists at all) is cancelled directly here instead of
  // round-tripping through a heartbeat that will never come for a job no
  // worker has picked up yet.
  app.post<{ Params: { id: string; jobId: string } }>(
    "/api/workers/:id/downloads/:jobId/cancel",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
      const job = repo.queueRepo.getJob(request.params.jobId);
      if (!job || job.workerId !== worker.id || job.jobType !== "download_model") {
        throw new NotFoundError("unknown download");
      }
      if (job.status === "pending") {
        repo.queueRepo.cancelPendingJob(job.id);
      } else {
        repo.queueRepo.requestDiscard(job.id);
      }
      return reply.code(200).send({ ok: true });
    }
  );

  // Generic job status (used for refresh_models polling); kept alongside the
  // legacy download-specific one for backwards compat. The refresh_models job
  // may be pending/claimed for a while (hashing large files), so the Models
  // page polls this instead of a fixed 2s timeout.
  app.get<{ Params: { id: string; jobId: string } }>(
    "/api/workers/:id/jobs/:jobId",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
      const job = repo.queueRepo.getJob(request.params.jobId);
      if (!job || job.workerId !== worker.id) {
        throw new NotFoundError("unknown job");
      }
      return reply.code(200).send({ status: job.status, job_type: job.jobType });
    }
  );

  // Lets the client tell a download that's genuinely gone (completed, failed,
  // cancelled, or never existed) apart from one that's still queued behind
  // the worker's max_concurrent_downloads cap -- such a job never appears in
  // the worker's heartbeat-reported activeDownloads (that only lists jobs
  // actually claimed/running), so without this the Models page poll loop had
  // no way to distinguish "queued, waiting for a slot" from "dropped" and
  // wrongly finalized it after a few missed polls.
  app.get<{ Params: { id: string; jobId: string } }>(
    "/api/workers/:id/downloads/:jobId",
    async (request, reply) => {
      const worker = repo.workerRepo.getWorker(request.params.id);
      if (!worker) throw new NotFoundError("unknown machine");
      assertOwnsWorker(resolveAuthUser(request)?.user.id, worker.id);
      const job = repo.queueRepo.getJob(request.params.jobId);
      if (!job || job.workerId !== worker.id || job.jobType !== "download_model") {
        throw new NotFoundError("unknown download");
      }
      return reply.code(200).send({ status: job.status });
    }
  );

  // Worker -> server callback reporting a download's terminal outcome.
  // Retried by the worker with backoff (safeReportDownloadResult), so this
  // must stay safe to receive more than once for the same download --
  // repo.registerModel's upsert-on-id semantics already guarantee that for
  // the success path. Separate from the worker's own generic per-job
  // completion report (POST /api/worker/jobs/:jobId/complete, routes/queue.ts)
  // -- that one just marks the worker_jobs bookkeeping row done; this one
  // does the actual model-catalog registration, unchanged from before.
  app.post<{ Body: ModelDownloadCallbackInput }>(
    "/api/models/download-callback",
    // Same rationale as /api/runs/:id/items/:idx's logLevel below -- this
    // isn't a user-facing request, no need for Fastify's automatic pair.
    { logLevel: "silent" },
    async (request, reply) => {
      // Worker credential, not a browser session -- see auth-middleware.ts's
      // WORKER_AUTHENTICATED_ROUTES entry for this route, same pattern as
      // every /api/worker/* route in routes/queue.ts. The body has no
      // machine_id, so this only resolves via a real Stage 3 worker session
      // (not the Stage 1 WORKER_SHARED_TOKEN+machine_id fallback) -- fine,
      // since every worker that can reach this point already authenticated
      // the same way for its heartbeat/queue-poll calls.
      await authenticateWorker(request);
      const validationError = validateModelDownloadCallback(request.body);
      if (validationError) {
        app.log.warn({ error: validationError }, "model download callback rejected: invalid payload");
        return reply.code(400).send({ error: validationError });
      }
      const { worker, hf_repo, hf_file, ok, error, sha256, size_bytes, n_layer, mtp_layers, expert_count } =
        request.body;
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
          ...(typeof expert_count === "number" && expert_count > 0 ? { expert_count } : {}),
        };
        if (isMtpDraftModel({ metadata, hf_file, filename: hf_file })) {
          metadata.mtp_role = "draft";
        }
        app.log.info(
          {
            worker,
            hf_repo,
            hf_file,
            size_bytes,
            n_layer,
            param_count,
            mtp_layers,
            expert_count,
            mtp_role: metadata.mtp_role,
          },
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
