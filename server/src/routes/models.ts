import type { FastifyInstance } from "fastify";
import { deriveModelId, repo } from "../db/repo.js";
import { getHfGgufMeta } from "../hf.js";
import {
  lookupHfGgufHashes,
  verifyRepoInBackground,
  triggerDuplicateFactResolutionForCollisions,
  HF_INDEX_REFRESH_INTERVAL_MS,
} from "../hf-index.js";
import { isMtpDraftModel } from "../../../shared/types.js";
import type { Model, RegisterModelInput } from "../../../shared/types.js";
import { isPlausibleLayerCount } from "../../../shared/gguf.js";
import { userOrIpKeyGenerator, resolveAuthUser, assertOwnsWorker } from "../auth-middleware.js";
import { ForbiddenError, UnauthorizedError } from "../errors.js";
import { resolveWorkerSession } from "../worker-auth.js";

const HF_META_TIMEOUT_MS = 15_000;

// Multi-user Stage 4 (MULTIUSER_PLAN.md §4.5): GET /api/models/hf-updates
// fans out to HF per distinct repo, which with a global catalog scales with
// total users, not just this operator's own model list -- a 10-minute
// server-side cache keyed by repo keeps repeat Custom Test page loads (by
// anyone, for any model sharing that repo) from re-asking HF the same
// question. Module-level (not per-request), same pattern as ai.ts's
// modelCooldownUntil -- reset on process restart, which is fine for a
// last-modified check that's only ever a "possibly newer" hint anyway.
const HF_UPDATES_CACHE_TTL_MS = 10 * 60 * 1000;
const hfUpdatesCache = new Map<string, { value: string | null; expiresAt: number }>();

// §2.6: 60/hour, keyed on user (IP fallback, same reasoning as
// runs.ts's trigger limit) -- shared across POST /api/models and both
// backfill routes below, all comparably cheap-but-not-free registry writes.
const REGISTRY_WRITE_RATE_LIMIT = { max: 60, timeWindow: "1 hour", keyGenerator: userOrIpKeyGenerator } as const;
const HASH_LOOKUP_RATE_LIMIT = { max: 120, timeWindow: "1 hour", keyGenerator: userOrIpKeyGenerator } as const;

// A model the caller may act on: with accounts, only one in their own list
// (repo.listModelsVisibleTo), answered 404 otherwise -- same as a missing id,
// so the route doesn't confirm another tenant's catalog row exists.
function findVisibleModel(id: string, userId: string | undefined): Model | undefined {
  if (userId === undefined) return repo.getModel(id);
  return repo.listModelsVisibleTo(userId).find((m) => m.id === id);
}

export async function modelsRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/models", async (request) => {
    return { models: repo.listModelsVisibleTo(resolveAuthUser(request)?.user.id) };
  });

  // Trigger a full model refresh on a specific worker
  app.post<{ Body: { worker_id?: string } }>(
    "/api/models/refresh",
    async (request, reply) => {
      const { worker_id } = request.body ?? {};
      const authed = resolveAuthUser(request);
      
      // If worker_id specified, verify ownership and queue refresh job
      if (worker_id) {
        const worker = repo.workerRepo.getWorker(worker_id);
        if (!worker) {
          return reply.code(404).send({ error: "unknown machine" });
        }
        assertOwnsWorker(authed?.user.id, worker.id);
        
        const jobId = repo.queueRepo.enqueueJob(worker.id, { type: "refresh_models", payload: {} });
        return reply.code(202).send({ ok: true, queued: true, job_id: jobId, message: "Model refresh job queued" });
      }
      
      // If no worker_id, return guidance
      return reply.code(400).send({ error: "worker_id is required to trigger a model refresh" });
    }
  );

  // The Models page's "Add a model manually" form. Create-only: an existing
  // row -- shared catalog or someone else's -- is never changed from here
  // (audit 2026-10-11 N3). Only the fields below are read from the body, so a
  // client can't assert the download lock or an owner.
  app.post<{ Body: RegisterModelInput }>(
    "/api/models",
    { config: { rateLimit: REGISTRY_WRITE_RATE_LIMIT } },
    async (request, reply) => {
      const body = (request.body ?? {}) as Partial<Record<keyof RegisterModelInput, unknown>>;
      const { source, filename, size_bytes, hf_repo, hf_file, metadata } = body;
      if (source !== "local" && source !== "huggingface") {
        return reply.code(400).send({ error: "source must be \"local\" or \"huggingface\"" });
      }
      for (const [field, v] of [["id", body.id], ["filename", filename], ["hf_repo", hf_repo], ["hf_file", hf_file]] as const) {
        if (v !== undefined && (typeof v !== "string" || v.length > 512)) {
          return reply.code(400).send({ error: `${field} must be a string of at most 512 characters` });
        }
      }
      if (size_bytes !== undefined && (typeof size_bytes !== "number" || !Number.isSafeInteger(size_bytes) || size_bytes < 0)) {
        return reply.code(400).send({ error: "size_bytes must be a non-negative integer" });
      }
      if (metadata !== undefined && (metadata === null || typeof metadata !== "object" || Array.isArray(metadata))) {
        return reply.code(400).send({ error: "metadata must be an object" });
      }
      if (source === "huggingface" && (!hf_repo || !hf_file)) {
        return reply
          .code(400)
          .send({ error: "hf_repo and hf_file are required for huggingface source" });
      }
      let id = (body.id as string | undefined)?.trim() || undefined;
      // A 64-hex id is a file's own sha256 -- the key a worker's heartbeat and
      // the download callback write that file's shared row under. A client
      // may reserve one only as a bare local file: its HF identity and header
      // metadata come from those verified paths, which take the row over.
      if (id && /^[0-9a-f]{64}$/i.test(id)) {
        id = id.toLowerCase();
        if (source !== "local" || hf_repo || hf_file || (metadata && Object.keys(metadata).length > 0)) {
          return reply.code(400).send({
            error:
              "a sha256 id registers a local file only -- leave hf_repo, hf_file and metadata empty; they are filled in once a machine with the file reports it",
          });
        }
      }
      const authed = resolveAuthUser(request);
      const input: RegisterModelInput = {
        id,
        filename: filename as string | undefined,
        size_bytes: size_bytes as number | undefined,
        source,
        hf_repo: hf_repo as string | undefined,
        hf_file: hf_file as string | undefined,
        metadata: metadata as RegisterModelInput["metadata"],
        // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4) -- undefined in
        // single-tenant mode, matching every other userId in this app.
        created_by: authed?.user.id,
      };
      const model = repo.insertModelIfAbsent(input);
      if (model) {
        // Only a derived-id row can be shared by several adders; an explicit
        // id is the creator's alone (created_by covers it).
        if (authed && !id) repo.addModelRegistration(authed.user.id, model.id);
        return reply.code(201).send({ model });
      }
      // No id sent: the id is derived from the repo/file (or name/size), the
      // same for everyone, so an existing row is this very model. Adding it
      // again puts it on the caller's list without changing the row.
      if (!id) {
        const existing = repo.getModel(deriveModelId(input))!;
        if (authed) repo.addModelRegistration(authed.user.id, existing.id);
        return reply.code(200).send({ model: existing });
      }
      // An explicit id that's taken: never touched from here.
      return reply.code(409).send({ error: "a model with this id is already registered" });
    }
  );

  app.delete<{ Params: { id: string } }>("/api/models/:id", async (request, reply) => {
    const existing = repo.getModel(request.params.id);
    if (!existing) return reply.code(404).send({ error: "model not found" });

    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): only the creator may
    // delete. An unclaimed model (created_by NULL: worker-driven
    // registrations, imports, legacy rows) is a shared, content-addressed
    // catalog entry other tenants may depend on, so with auth on only a
    // superadmin may delete it. Single-tenant mode (no session) is unchanged.
    const authed = resolveAuthUser(request);
    const createdBy = repo.getModelCreatedBy(existing.id);
    // Someone else also added this model (POST /api/models joins a shared
    // derived row), so the row isn't the caller's alone: take it off their
    // list instead of deleting it for everyone.
    // (A creator from before model_registrations existed has no row there,
    // hence the created_by check.)
    const isCreator = !!authed && createdBy === authed.user.id;
    if (
      authed &&
      (isCreator || repo.hasModelRegistration(authed.user.id, existing.id)) &&
      (!isCreator || repo.hasOtherModelRegistrations(existing.id, authed.user.id))
    ) {
      repo.removeModelRegistration(authed.user.id, existing.id);
      request.log.info({ model_id: existing.id }, "model removed from caller's list");
      return reply.code(200).send({ ok: true });
    }
    if (authed && createdBy && createdBy !== authed.user.id) {
      throw new ForbiddenError("you don't own this model");
    }
    if (authed && !createdBy && !authed.user.isSuperadmin) {
      throw new ForbiddenError("shared catalog models can only be deleted by an admin");
    }

    const resultCount = repo.countResultsForModel(existing.id);
    if (resultCount > 0) {
      return reply.code(409).send({
        error: `cannot delete model: ${resultCount} run result(s) reference it`,
      });
    }
    const runningCount = repo.countRunningTestsForModel(existing.id);
    if (runningCount > 0) {
      return reply.code(409).send({
        error: `cannot delete model: ${runningCount} run(s) still running or scheduled against it`,
      });
    }

    const model = repo.deleteModel(existing.id)!;
    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.4): no more fan-out to every
    // worker's model_dir -- with a global catalog, most workers aren't even
    // the caller's to touch. Removing the catalog ROW is a metadata-only
    // action now; freeing the actual file on a specific machine is the
    // caller's own separate, deliberate action via DELETE
    // /api/workers/:id/models?file=... (routes/workers.ts).
    request.log.info({ model_id: model.id }, "model deregistered");
    return reply.code(200).send({ ok: true });
  });

  app.post<{ Params: { id: string } }>(
    "/api/models/:id/backfill-layer-count",
    { config: { rateLimit: REGISTRY_WRITE_RATE_LIMIT } },
    async (request, reply) => {
      const authed = resolveAuthUser(request);
      const model = findVisibleModel(request.params.id, authed?.user.id);
      if (!model) return reply.code(404).send({ error: "model not found" });

      // All three must already be known to skip the lookup -- an earlier
      // version of this check only looked at n_layer, which meant a model
      // that already had n_layer (true for every model registered before
      // mtp_layers detection existed) could never get mtp_layers backfilled
      // at all, since the request never got this far. tensor_layer_bytes
      // joins the same gate for the same reason (see ModelMetadata's doc
      // comment): a model registered before it existed should still pick it
      // up on the next manual backfill click.
      const hasLayerCount = typeof model.metadata.n_layer === "number";
      const hasMtpLayers = typeof model.metadata.mtp_layers === "number";
      const hasTensorLayerBytes = model.metadata.tensor_layer_bytes != null;
      if (hasLayerCount && hasMtpLayers && hasTensorLayerBytes) {
        return reply.code(200).send({ ok: true, n_layer: model.metadata.n_layer });
      }

      const filename = model.source === "local" ? model.filename : model.hf_file;
      if (!filename) {
        return reply.code(400).send({ error: "model has no filename to look up" });
      }

      // Reads from the caller's own machines' cached model_files_json (last
      // heartbeat) -- no live round trip, see repo.workerRepo.findModelFileMeta.
      // A stored reading from before the heartbeat bounds counts as no reading.
      const found = repo.workerRepo.findModelFileMeta(authed?.user.id, filename);
      const meta =
        found && (found.n_layer == null || isPlausibleLayerCount(found.n_layer))
          ? found
          : { n_layer: null, mtp_layers: null, tensor_layer_bytes: null };
      // Fall back to the already-known value rather than letting a worker
      // that hasn't reported this file (temporarily offline, or no longer has
      // it) regress an n_layer this model already had, just because this
      // call's real purpose this time was picking up mtp_layers.
      const resolvedNLayer = meta.n_layer ?? (hasLayerCount ? (model.metadata.n_layer as number) : null);
      if (resolvedNLayer == null) {
        return reply.code(200).send({ ok: true, n_layer: null });
      }
      // With accounts, the catalog row is shared: one user's machine reading
      // a file that merely has the same NAME must not rewrite what everyone
      // else sees (audit 2026-10-18 A3). The caller gets the value; their own
      // machine's reading of the file itself already reaches their pages and
      // jobs through the heartbeat (listModelsVisibleTo / getModelForWorker).
      if (authed) {
        return reply.code(200).send({ ok: true, n_layer: resolvedNLayer });
      }
      // mtp_layers comes along for free from the same lookup -- only worth
      // persisting when actually present (>0), same "don't clobber with a
      // meaningless 0" posture the download route already uses. Also recompute
      // mtp_role from the now-complete metadata (see shared/types.ts's
      // isMtpDraftModel) -- every read site recomputes this live regardless,
      // but keeping the persisted flag correct too avoids a misleading raw DB
      // row.
      const patch = {
        n_layer: resolvedNLayer,
        ...(typeof meta.mtp_layers === "number" && meta.mtp_layers > 0 ? { mtp_layers: meta.mtp_layers } : {}),
        ...(meta.tensor_layer_bytes ? { tensor_layer_bytes: meta.tensor_layer_bytes } : {}),
      };
      const mergedMetadata = { ...model.metadata, ...patch };
      repo.updateModelMetadata(model.id, {
        ...patch,
        ...(isMtpDraftModel({ metadata: mergedMetadata, hf_file: model.hf_file, filename: model.filename })
          ? { mtp_role: "draft" as const }
          : {}),
      });
      request.log.info(
        { model_id: model.id, filename, n_layer: resolvedNLayer, mtp_layers: meta.mtp_layers },
        "backfilled model layer count"
      );
      return reply.code(200).send({ ok: true, n_layer: resolvedNLayer });
    }
  );

  // Same "backfill" story as layer count above, but for models registered
  // before real parameter-count lookup existed (or before this model's own
  // download picked it up). Unlike layer count, this never touches a
  // worker -- HF's API returns it straight from the repo id, no file bytes
  // needed anywhere.
  app.post<{ Params: { id: string } }>(
    "/api/models/:id/backfill-param-count",
    { config: { rateLimit: REGISTRY_WRITE_RATE_LIMIT } },
    async (request, reply) => {
      const model = findVisibleModel(request.params.id, resolveAuthUser(request)?.user.id);
      if (!model) return reply.code(404).send({ error: "model not found" });

      if (typeof model.metadata.param_count === "number") {
        return reply.code(200).send({ ok: true, param_count: model.metadata.param_count });
      }
      if (!model.hf_repo) {
        return reply.code(400).send({ error: "model has no hf_repo to look up" });
      }

      const { param_count } = await getHfGgufMeta(model.hf_repo, HF_META_TIMEOUT_MS, "metadata-backfill");
      if (param_count == null) {
        return reply.code(200).send({ ok: true, param_count: null });
      }
      repo.updateModelMetadata(model.id, { param_count });
      request.log.info({ model_id: model.id, hf_repo: model.hf_repo, param_count }, "backfilled model parameter count");
      return reply.code(200).send({ ok: true, param_count });
    }
  );

  // Which machine(s) currently have each model's file, per their last
  // heartbeat -- powers the Models page's "Local"/"Remote" split. Reads the
  // cached model_files_json rather than a live per-worker fetch (no
  // outbound HTTP to workers anywhere, MULTIUSER_PLAN.md §1.11); a currently
  // offline machine's last-known file list is still shown (it just might be
  // stale), and is separately called out via `unreachable`.
  app.get("/api/models/locations", async (request) => {
    // Multi-user Stage 4 (MULTIUSER_PLAN.md §4.5): the caller's own machines
    // once authenticated; every machine in single-tenant mode.
    const authed = resolveAuthUser(request);
    const workers = authed ? repo.workerRepo.listWorkersForUser(authed.user.id) : repo.workerRepo.listWorkers();
    const locations: Record<string, string[]> = {};
    for (const model of repo.listModels()) {
      const filename = model.source === "local" ? model.filename : model.hf_file;
      // Hash-keyed catalog rows (id == the file's SHA-256 -- every download and
      // hash-identified scan registers this way) can be located even when the
      // local copy's path differs from the canonical HF filename (a manually
      // renamed/dropped file), since workers report each file's digest.
      const shaId = /^[0-9a-f]{64}$/.test(model.id) ? model.id : null;
      if (!filename && !shaId) continue;
      // Keyed by id, not display_name -- display names are only guaranteed
      // unique per user (MULTIUSER_PLAN.md §3.3), and the client matches
      // this against the worker it has selected by id, not by name.
      // state "missing" means the worker's own reconciliation confirmed the
      // file is gone from disk (its cache keeps the row for re-download
      // bookkeeping) -- that must not count as "this machine has it", or a
      // deleted model keeps showing under its former machine forever.
      const owners = workers
        .filter((w) =>
          w.modelFiles.some(
            (f) =>
              f.state !== "missing" &&
              ((filename != null && filename !== "" && f.path === filename) ||
                (shaId != null && f.sha256 != null && f.sha256.toLowerCase() === shaId))
          )
        )
        .map((w) => w.id);
      if (owners.length > 0) locations[model.id] = owners;
    }
    const unreachable = workers.filter((w) => w.status === "offline").map((w) => w.id);
    return { locations, unreachable };
  });

  // Live per-model check against Hugging Face for the Custom Test model picker's
  // "Updated X ago" / "possibly newer on HF" hints -- deliberately not
  // persisted anywhere (unlike param_count/n_layer's backfill routes above):
  // this should reflect HF's *current* state on every picker load, not a
  // point-in-time snapshot that could go stale. Only models with an hf_repo
  // are checked; local models have nothing to compare against. Requests are
  // deduped by repo since several registered models (different quants) often
  // share one -- no reason to ask HF the same question twice.
  app.get("/api/models/hf-updates", async (request) => {
    const withRepo = repo.listModelsVisibleTo(resolveAuthUser(request)?.user.id).filter((m): m is typeof m & { hf_repo: string } => Boolean(m.hf_repo));
    const byRepo = new Map<string, Promise<string | null>>();
    for (const m of withRepo) {
      if (byRepo.has(m.hf_repo)) continue;
      const cached = hfUpdatesCache.get(m.hf_repo);
      if (cached && cached.expiresAt > Date.now()) {
        byRepo.set(m.hf_repo, Promise.resolve(cached.value));
        continue;
      }
      byRepo.set(
        m.hf_repo,
        getHfGgufMeta(m.hf_repo, HF_META_TIMEOUT_MS, "hf-updates").then((meta) => {
          hfUpdatesCache.set(m.hf_repo, { value: meta.last_modified, expiresAt: Date.now() + HF_UPDATES_CACHE_TTL_MS });
          return meta.last_modified;
        })
      );
    }
    const updates: Record<string, string | null> = {};
    await Promise.all(
      withRepo.map(async (m) => {
        updates[m.id] = await byRepo.get(m.hf_repo)!;
      })
    );
    return { updates };
  });

  // Hash lookup: workers send SHA-256 hashes of local files, and the server
  // returns the matching HF metadata (repo_id, filename, revision) so the
  // client can display "this is bartowski/gemma-3-27b-it-GGUF/…". This is
  // the bridge between the worker's local file scan and the server's
  // hf_gguf_index table (see server/src/hf-index.ts for how that index is
  // built). Hashes not in the index are simply absent from the result --
  // the caller falls back to "unknown" state for those.
  // Rate-limited per IP/user (same posture as registry writes) to prevent abuse.
  app.post<{ Body: { hashes?: string[] } }>(
    "/api/models/hash-lookup",
    { config: { rateLimit: HASH_LOOKUP_RATE_LIMIT } },
    async (request, reply) => {
      // Listed in WORKER_AUTHENTICATED_ROUTES so the middleware lets a worker
      // session through; that makes this handler the gate for both callers.
      if (process.env.AUTH_ENABLED === "true" && !resolveAuthUser(request) && !resolveWorkerSession(request)) {
        throw new UnauthorizedError("no session");
      }
      const hashes = request.body?.hashes;
      if (!Array.isArray(hashes) || hashes.length === 0) {
        return reply.code(400).send({ error: "hashes array is required" });
      }
      // Cap at a reasonable batch size to bound the query and prevent
      // abuse -- a worker heartbeat carries at most its model_dir's files.
      if (hashes.length > 500) {
        return reply.code(400).send({ error: "too many hashes (max 500)" });
      }
      const results = lookupHfGgufHashes(hashes);
      // Lazy, on-demand verification instead of a background reconciliation
      // crawl: a matched row that's gone stale gets re-checked against HF
      // only because a real worker just depended on it, not on a schedule.
      // Fire-and-forget -- never delays this response; a soft-delete (or
      // updated files) lands in the DB in time for the *next* lookup.
      // Already-deleted rows are skipped -- no value in repeatedly re-asking
      // HF about a repo already confirmed gone.
      for (const r of results) {
        if (r.deleted_at == null && Date.now() - r.last_seen > HF_INDEX_REFRESH_INTERVAL_MS) {
          verifyRepoInBackground(r.repo_id);
        }
      }
      // Same fire-and-forget posture as verifyRepoInBackground above, for a
      // different gap: a hash genuinely live under more than one repo needs
      // to know which one is the real source vs. a fork/mirror (see
      // hf-index.ts's isBetterMatch) -- resolved here, once, the first time a
      // real worker's hash actually collides, rather than eagerly for every
      // scanned repo.
      triggerDuplicateFactResolutionForCollisions(hashes);
      return { results };
    }
  );
}
