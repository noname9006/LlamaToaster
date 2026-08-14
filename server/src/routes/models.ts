import type { FastifyInstance } from "fastify";
import { repo } from "../db/repo.js";
import { listWorkers } from "../config.js";
import { describeWorkerError } from "../worker-errors.js";
import { isRecentlyUnreachable, markReachable, markUnreachable } from "../worker-reachability.js";
import { getHfGgufMeta } from "../hf.js";
import { isMtpDraftModel } from "../../../shared/types.js";
import type { RegisterModelInput } from "../../../shared/types.js";

const WORKER_DELETE_TIMEOUT_MS = 15_000;
const WORKER_GGUF_INFO_TIMEOUT_MS = 15_000;
const WORKER_LIST_TIMEOUT_MS = 5_000;
const HF_META_TIMEOUT_MS = 15_000;

export interface FileDeletionResult {
  worker: string;
  ok: boolean;
  error?: string;
}

// A registered model isn't tied to a specific worker (any worker's model_dir
// might have the file), so freeing disk space means asking every configured
// worker to delete it and letting the ones that don't have it 404
// harmlessly -- best-effort, never blocks the registry deletion above.
async function deleteFileFromAllWorkers(filename: string): Promise<FileDeletionResult[]> {
  const workers = await listWorkers();
  return Promise.all(
    workers.map(async (worker): Promise<FileDeletionResult> => {
      try {
        const res = await fetch(`${worker.url}/models?file=${encodeURIComponent(filename)}`, {
          method: "DELETE",
          signal: AbortSignal.timeout(WORKER_DELETE_TIMEOUT_MS),
        });
        if (res.ok || res.status === 404) {
          return { worker: worker.name, ok: res.ok };
        }
        const text = await res.text();
        return { worker: worker.name, ok: false, error: text };
      } catch (err) {
        return { worker: worker.name, ok: false, error: describeWorkerError(err) };
      }
    })
  );
}

interface WorkerGgufInfo {
  n_layer: number | null;
  mtp_layers: number | null;
}

// Same "any worker's model_dir might have it" reasoning as
// deleteFileFromAllWorkers above, but stops at the first worker that
// actually has the file and can read metadata out of it -- one answer is
// enough, unlike delete where every worker's copy matters. Reads n_layer and
// mtp_layers together (same underlying GGUF header scan on the worker side,
// see worker/src/gguf.ts) so backfilling one opportunistically backfills the
// other too, rather than needing two separate worker round trips.
async function findGgufInfoFromWorkers(filename: string): Promise<WorkerGgufInfo> {
  for (const worker of await listWorkers()) {
    try {
      const res = await fetch(`${worker.url}/models/gguf-info?file=${encodeURIComponent(filename)}`, {
        signal: AbortSignal.timeout(WORKER_GGUF_INFO_TIMEOUT_MS),
      });
      if (!res.ok) continue; // not on this worker, or it couldn't read it -- try the next one
      const data = (await res.json()) as { n_layer?: number | null; mtp_layers?: number | null };
      if (typeof data.n_layer === "number" || typeof data.mtp_layers === "number") {
        return {
          n_layer: typeof data.n_layer === "number" ? data.n_layer : null,
          mtp_layers: typeof data.mtp_layers === "number" ? data.mtp_layers : null,
        };
      }
    } catch {
      // worker unreachable -- try the next one
    }
  }
  return { n_layer: null, mtp_layers: null };
}

// One call per worker (its own recursive model_dir listing, see worker/src/
// index.ts's listModelDirFiles) rather than one call per model per worker --
// cheap enough to run on every Models page load. A worker that errors/times
// out is reported back as unreachable rather than silently treated as "has
// nothing", so the client never mistakes "couldn't check" for "not there".
async function listWorkerFiles(worker: { name: string; url: string }): Promise<{ path: string }[] | null> {
  if (isRecentlyUnreachable(worker.url)) return null;
  try {
    const res = await fetch(`${worker.url}/models`, { signal: AbortSignal.timeout(WORKER_LIST_TIMEOUT_MS) });
    if (!res.ok) {
      markUnreachable(worker.url);
      return null;
    }
    const data = (await res.json()) as { files: { path: string; size_bytes: number }[] };
    markReachable(worker.url);
    return data.files;
  } catch {
    markUnreachable(worker.url);
    return null;
  }
}

export async function modelsRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/models", async () => {
    return { models: repo.listModels() };
  });

  app.post<{ Body: RegisterModelInput }>(
    "/api/models",
    async (request, reply) => {
      const body = request.body;
      if (!body || !body.source) {
        return reply.code(400).send({ error: "source is required" });
      }
      if (
        body.source === "huggingface" &&
        (!body.hf_repo || !body.hf_file)
      ) {
        return reply
          .code(400)
          .send({ error: "hf_repo and hf_file are required for huggingface source" });
      }
      const model = repo.registerModel(body);
      return reply.code(201).send({ model });
    }
  );

  app.delete<{ Params: { id: string } }>("/api/models/:id", async (request, reply) => {
    const existing = repo.getModel(request.params.id);
    if (!existing) return reply.code(404).send({ error: "model not found" });

    const resultCount = repo.countResultsForModel(existing.id);
    if (resultCount > 0) {
      return reply.code(409).send({
        error: `cannot delete model: ${resultCount} run result(s) reference it`,
      });
    }
    const runningCount = repo.countRunningRunsForModel(existing.id);
    if (runningCount > 0) {
      return reply.code(409).send({
        error: `cannot delete model: ${runningCount} run(s) still in progress against it`,
      });
    }

    const model = repo.deleteModel(existing.id)!;
    const filename = model.source === "local" ? model.filename : model.hf_file;
    const fileDeletions = filename ? await deleteFileFromAllWorkers(filename) : [];
    request.log.info(
      { model_id: model.id, filename, fileDeletions },
      "model deregistered"
    );
    return reply.code(200).send({ ok: true, file_deletions: fileDeletions });
  });

  app.post<{ Params: { id: string } }>("/api/models/:id/backfill-layer-count", async (request, reply) => {
    const model = repo.getModel(request.params.id);
    if (!model) return reply.code(404).send({ error: "model not found" });

    // Both must already be known to skip the worker round trip -- an
    // earlier version of this check only looked at n_layer, which meant a
    // model that already had n_layer (true for every model registered
    // before mtp_layers detection existed) could never get mtp_layers
    // backfilled at all, since the request never got this far.
    const hasLayerCount = typeof model.metadata.n_layer === "number";
    const hasMtpLayers = typeof model.metadata.mtp_layers === "number";
    if (hasLayerCount && hasMtpLayers) {
      return reply.code(200).send({ ok: true, n_layer: model.metadata.n_layer });
    }

    const filename = model.source === "local" ? model.filename : model.hf_file;
    if (!filename) {
      return reply.code(400).send({ error: "model has no filename to look up" });
    }

    const { n_layer, mtp_layers } = await findGgufInfoFromWorkers(filename);
    // Fall back to the already-known value rather than letting a worker
    // that's temporarily unreachable (or no longer has the file) regress an
    // n_layer this model already had, just because this call's real purpose
    // this time was picking up mtp_layers.
    const resolvedNLayer = n_layer ?? (hasLayerCount ? (model.metadata.n_layer as number) : null);
    if (resolvedNLayer == null) {
      return reply.code(200).send({ ok: true, n_layer: null });
    }
    // mtp_layers comes along for free from the same worker round trip --
    // only worth persisting when actually present (>0), same "don't clobber
    // with a meaningless 0" posture the download route already uses. Also
    // recompute mtp_role from the now-complete metadata (see shared/types.ts's
    // isMtpDraftModel) -- every read site recomputes this live regardless, but
    // keeping the persisted flag correct too avoids a misleading raw DB row.
    const patch = {
      n_layer: resolvedNLayer,
      ...(typeof mtp_layers === "number" && mtp_layers > 0 ? { mtp_layers } : {}),
    };
    const mergedMetadata = { ...model.metadata, ...patch };
    repo.updateModelMetadata(model.id, {
      ...patch,
      ...(isMtpDraftModel({ metadata: mergedMetadata, hf_file: model.hf_file, filename: model.filename })
        ? { mtp_role: "draft" as const }
        : {}),
    });
    request.log.info(
      { model_id: model.id, filename, n_layer: resolvedNLayer, mtp_layers },
      "backfilled model layer count"
    );
    return reply.code(200).send({ ok: true, n_layer: resolvedNLayer });
  });

  // Same "backfill" story as layer count above, but for models registered
  // before real parameter-count lookup existed (or before this model's own
  // download picked it up). Unlike layer count, this never touches a
  // worker -- HF's API returns it straight from the repo id, no file bytes
  // needed anywhere.
  app.post<{ Params: { id: string } }>("/api/models/:id/backfill-param-count", async (request, reply) => {
    const model = repo.getModel(request.params.id);
    if (!model) return reply.code(404).send({ error: "model not found" });

    if (typeof model.metadata.param_count === "number") {
      return reply.code(200).send({ ok: true, param_count: model.metadata.param_count });
    }
    if (!model.hf_repo) {
      return reply.code(400).send({ error: "model has no hf_repo to look up" });
    }

    const { param_count } = await getHfGgufMeta(model.hf_repo, HF_META_TIMEOUT_MS);
    if (param_count == null) {
      return reply.code(200).send({ ok: true, param_count: null });
    }
    repo.updateModelMetadata(model.id, { param_count });
    request.log.info({ model_id: model.id, hf_repo: model.hf_repo, param_count }, "backfilled model parameter count");
    return reply.code(200).send({ ok: true, param_count });
  });

  // Which configured worker(s) currently have each model's file on disk --
  // powers the Models page's "Local"/"Remote" split. Deliberately not
  // persisted anywhere: a model's file could be added/removed on a worker
  // outside this app (manual copy, disk cleanup), so this always reflects a
  // live check rather than a point-in-time snapshot that could go stale.
  app.get("/api/models/locations", async () => {
    const workers = await listWorkers();
    const filesByWorker = await Promise.all(workers.map((w) => listWorkerFiles(w)));

    const unreachable: string[] = [];
    const pathSetByWorker = new Map<string, Set<string>>();
    workers.forEach((w, i) => {
      const files = filesByWorker[i];
      if (files === null) {
        unreachable.push(w.name);
        return;
      }
      pathSetByWorker.set(w.name, new Set(files.map((f) => f.path)));
    });

    const locations: Record<string, string[]> = {};
    for (const model of repo.listModels()) {
      const filename = model.source === "local" ? model.filename : model.hf_file;
      if (!filename) continue;
      const workerNames = workers
        .filter((w) => pathSetByWorker.get(w.name)?.has(filename))
        .map((w) => w.name);
      if (workerNames.length > 0) locations[model.id] = workerNames;
    }
    return { locations, unreachable };
  });

  // Live per-model check against Hugging Face for the New Run model picker's
  // "Updated X ago" / "possibly newer on HF" hints -- deliberately not
  // persisted anywhere (unlike param_count/n_layer's backfill routes above):
  // this should reflect HF's *current* state on every picker load, not a
  // point-in-time snapshot that could go stale. Only models with an hf_repo
  // are checked; local models have nothing to compare against. Requests are
  // deduped by repo since several registered models (different quants) often
  // share one -- no reason to ask HF the same question twice.
  app.get("/api/models/hf-updates", async () => {
    const withRepo = repo.listModels().filter((m): m is typeof m & { hf_repo: string } => Boolean(m.hf_repo));
    const byRepo = new Map<string, Promise<string | null>>();
    for (const m of withRepo) {
      if (!byRepo.has(m.hf_repo)) {
        byRepo.set(
          m.hf_repo,
          getHfGgufMeta(m.hf_repo, HF_META_TIMEOUT_MS).then((meta) => meta.last_modified)
        );
      }
    }
    const updates: Record<string, string | null> = {};
    await Promise.all(
      withRepo.map(async (m) => {
        updates[m.id] = await byRepo.get(m.hf_repo)!;
      })
    );
    return { updates };
  });
}
