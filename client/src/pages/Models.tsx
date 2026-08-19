import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { api } from "../api/client";
import { StatusPill } from "../components/StatusPill";
import { IconChevronDown, IconDownload, IconRefreshCw, IconTrash } from "../components/icons";
import { ParamsRangeSlider, PARAMS_STOPS, paramsInRange } from "../components/ParamsRangeSlider";
import { buildModelGroups, type ModelGroup } from "../modelGrouping";
import {
  formatBytes,
  formatParamsB,
  formatRelativeTime,
  hfFileUrl,
  hfRepoUrl,
  modelAuthor,
  modelFamily,
  modelParamsB,
  paramsBFromText,
  shortId,
} from "../utils";
import { isMtpDraftModel } from "../types";
import type {
  Model,
  ModelMetadata,
  ModelSource,
  HfRepoSearchResult,
  HfFileEntry,
  Worker,
} from "../types";

interface DownloadState {
  repoId: string;
  workerId: string;
  startedAt: number;
  file: HfFileEntry;
}

// Keyed by file path, mirrors `downloading` below -- persisted so an
// in-progress download survives a page refresh instead of vanishing from the
// UI while it keeps running server/worker-side regardless. See the poll
// effect further down for how a persisted-but-orphaned entry (no live
// handleDownload call left to notice its completion) gets reconciled.
const DOWNLOADS_STORAGE_KEY = "llamatoaster:active-downloads";

function loadPersistedDownloads(): Record<string, DownloadState> {
  try {
    const raw = localStorage.getItem(DOWNLOADS_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, DownloadState>) : {};
  } catch {
    return {};
  }
}

function persistDownloads(downloads: Record<string, DownloadState>): void {
  try {
    localStorage.setItem(DOWNLOADS_STORAGE_KEY, JSON.stringify(downloads));
  } catch {
    /* localStorage unavailable -- downloads just won't survive a refresh */
  }
}

type SortField = "created" | "author" | "family" | "params" | "size";
type SortDir = "asc" | "desc";

const SORT_OPTIONS: { value: SortField; label: string }[] = [
  { value: "created", label: "Date added" },
  { value: "author", label: "Author" },
  { value: "family", label: "Family" },
  { value: "params", label: "Parameters" },
  { value: "size", label: "Size" },
];

type HfSortField = "relevance" | "name" | "likes" | "downloads" | "params" | "newest";

const HF_SORT_OPTIONS: { value: HfSortField; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "newest", label: "Newest" },
  { value: "name", label: "Name (A-Z)" },
  { value: "likes", label: "Likes" },
  { value: "downloads", label: "Downloads" },
  { value: "params", label: "Parameters" },
];

// Session-scoped (not localStorage) -- a search is "current" for as long as
// the tab lives, but stale HF result counts/likes shouldn't linger forever
// across separate visits the way a remembered sweep should.
const HF_SEARCH_STORAGE_KEY = "llamatoaster:hf-search";

interface PersistedHfSearch {
  query: string;
  results: HfRepoSearchResult[];
  expandedRepo: string | null;
  filesByRepo: Record<string, HfFileEntry[]>;
  paramsLoIndex: number;
  paramsHiIndex: number;
  sortField: HfSortField;
  sortDir: SortDir;
}

function loadPersistedHfSearch(): PersistedHfSearch | null {
  try {
    const raw = sessionStorage.getItem(HF_SEARCH_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PersistedHfSearch) : null;
  } catch {
    return null;
  }
}

function persistHfSearch(state: PersistedHfSearch): void {
  try {
    sessionStorage.setItem(HF_SEARCH_STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* sessionStorage unavailable -- search just won't survive a refresh */
  }
}

const emptyForm = {
  source: "local" as ModelSource,
  id: "",
  filename: "",
  size_bytes: "",
  hf_repo: "",
  hf_file: "",
  metadata: "",
};

// Orders groups by a representative quant (the smallest, i.e. quants[0]
// after buildModelGroups' own size-ascending sort) using the same fields the
// old flat table's "Sort by" control offered -- applied to which group comes
// first, not to the quants/drafts within a group (those stay smallest-first).
function sortModelGroups(groups: ModelGroup[], field: SortField, dir: SortDir): ModelGroup[] {
  const d = dir === "asc" ? 1 : -1;
  return [...groups].sort((a, b) => {
    const ra = a.quants[0].base;
    const rb = b.quants[0].base;
    switch (field) {
      case "author":
        return d * a.author.localeCompare(b.author);
      case "family":
        return d * a.family.localeCompare(b.family);
      case "size":
        return d * (ra.size_bytes - rb.size_bytes);
      case "params":
        return d * ((modelParamsB(ra) ?? -1) - (modelParamsB(rb) ?? -1));
      default:
        return d * (ra.created_at - rb.created_at);
    }
  });
}

// buildModelGroups only nests a draft under a base that's present in the
// same input list -- within one worker's subset that's almost always true,
// but a draft can in principle exist on a worker its paired base doesn't
// (rare, but the file registry doesn't enforce they travel together). Those
// would otherwise vanish from that worker's section entirely; surface them
// as their own standalone rows instead of silently dropping a registered model.
function orphanDrafts(subset: Model[], groups: ModelGroup[]): Model[] {
  const paired = new Set<string>();
  for (const g of groups) for (const q of g.quants) for (const d of q.drafts) paired.add(d.id);
  return subset.filter((m) => isMtpDraftModel(m) && !paired.has(m.id));
}

export function Models() {
  const [models, setModels] = useState<Model[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [downloadWorker, setDownloadWorker] = useState("");
  const [pickWorkerWarning, setPickWorkerWarning] = useState(false);
  const workerPickerRef = useRef<HTMLSelectElement>(null);

  const [form, setForm] = useState(emptyForm);
  const [addMsg, setAddMsg] = useState("");

  const [hfQuery, setHfQuery] = useState(() => loadPersistedHfSearch()?.query ?? "");
  const [hfResults, setHfResults] = useState<HfRepoSearchResult[]>(() => loadPersistedHfSearch()?.results ?? []);
  const [expandedRepo, setExpandedRepo] = useState<string | null>(() => loadPersistedHfSearch()?.expandedRepo ?? null);
  const [filesByRepo, setFilesByRepo] = useState<Record<string, HfFileEntry[]>>(
    () => loadPersistedHfSearch()?.filesByRepo ?? {}
  );
  const [hfMsg, setHfMsg] = useState("");
  const [searching, setSearching] = useState(false);
  const [downloading, setDownloading] = useState<Record<string, DownloadState>>(loadPersistedDownloads);
  // { bytes, total } -- read off the owning worker's activeJobProgress
  // (MULTIUSER_PLAN.md §1.5), not a dedicated per-file progress endpoint
  // (deleted, see the poll effect below). total is null until the worker
  // reports Content-Length.
  const [progress, setProgress] = useState<Record<string, { bytes: number; total: number | null }>>({});
  const [speeds, setSpeeds] = useState<Record<string, number>>({});
  const prevSampleRef = useRef<Record<string, { bytes: number; time: number }>>({});
  // Exponential moving average of each download's speed, keyed by path --
  // the raw poll-to-poll delta (see the effect below) is noisy on its own:
  // a 750ms sampling window is short enough that normal jitter in when the
  // worker's event loop gets to flush a chunk (worse with multiple downloads
  // competing for the same worker's bandwidth/CPU) shows up as visible
  // spikes. Smoothing the *displayed* rate instead of widening the sampling
  // window keeps the number responsive while damping single-sample noise.
  const smoothedSpeedRef = useRef<Record<string, number>>({});
  const seenRef = useRef<Record<string, boolean>>({});
  const missesRef = useRef<Record<string, number>>({});

  const [hfParamsLoIndex, setHfParamsLoIndex] = useState(() => loadPersistedHfSearch()?.paramsLoIndex ?? 0);
  const [hfParamsHiIndex, setHfParamsHiIndex] = useState(
    () => loadPersistedHfSearch()?.paramsHiIndex ?? PARAMS_STOPS.length - 1
  );
  const [hfSortField, setHfSortField] = useState<HfSortField>(() => loadPersistedHfSearch()?.sortField ?? "relevance");
  const [hfSortDir, setHfSortDir] = useState<SortDir>(() => loadPersistedHfSearch()?.sortDir ?? "desc");

  // Keyed by "<model_id>:<worker_name>" since a delete now only ever targets
  // one worker's copy, and the same model can appear (with independent
  // in-flight state) under more than one worker's section at once.
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [modelsMsg, setModelsMsg] = useState("");
  // Per-row state for the "look up real parameter count" backfill action --
  // keyed by model id since, unlike delete, several rows could plausibly be
  // looked up around the same time.
  const [paramLookupBusyId, setParamLookupBusyId] = useState<string | null>(null);
  const [paramLookupErr, setParamLookupErr] = useState<Record<string, string>>({});

  // Live "which worker(s) have this model's file" map -- see
  // server/src/routes/models.ts's /api/models/locations. Drives the
  // Local/Remote (one collapsible section per configured worker) split
  // below. unreachableLocationWorkers lists workers that couldn't be
  // checked, so their models aren't wrongly assumed absent everywhere.
  const [locations, setLocations] = useState<Record<string, string[]>>({});
  const [unreachableLocationWorkers, setUnreachableLocationWorkers] = useState<string[]>([]);

  const [authorFilter, setAuthorFilter] = useState("");
  const [familyFilter, setFamilyFilter] = useState("");
  const [paramsLoIndex, setParamsLoIndex] = useState(0);
  const [paramsHiIndex, setParamsHiIndex] = useState(PARAMS_STOPS.length - 1);
  const [sortField, setSortField] = useState<SortField>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const enrichedModels = useMemo(
    () =>
      models.map((m) => ({
        model: m,
        author: modelAuthor(m),
        family: modelFamily(m),
        paramsB: modelParamsB(m),
      })),
    [models]
  );
  // Only offer authors/families that actually have a file on some configured
  // worker right now -- a registered-but-unlocated model (e.g. its file was
  // deleted from every worker) shouldn't populate these filters with a value
  // that would then show zero results everywhere.
  const presentModels = useMemo(
    () => enrichedModels.filter((e) => (locations[e.model.id]?.length ?? 0) > 0),
    [enrichedModels, locations]
  );
  const authorOptions = useMemo(
    () => Array.from(new Set(presentModels.map((e) => e.author))).sort(),
    [presentModels]
  );
  const familyOptions = useMemo(
    () => Array.from(new Set(presentModels.map((e) => e.family))).sort(),
    [presentModels]
  );
  const filteredModels = useMemo(() => {
    let list = enrichedModels;
    if (authorFilter) list = list.filter((e) => e.author === authorFilter);
    if (familyFilter) list = list.filter((e) => e.family === familyFilter);
    list = list.filter((e) => paramsInRange(e.paramsB, paramsLoIndex, paramsHiIndex));
    return list.map((e) => e.model);
  }, [enrichedModels, authorFilter, familyFilter, paramsLoIndex, paramsHiIndex]);

  // Buckets the filtered models by where their file actually lives -- one
  // entry per configured worker plus a trailing "not found anywhere" bucket,
  // each independently grouped via buildModelGroups (quant siblings +
  // nested MTP drafts, shared with NewRun's ModelPicker).
  const modelsByWorker = useMemo(
    () =>
      workers.map((w) => {
        const subset = filteredModels.filter((m) => locations[m.id]?.includes(w.id));
        const groups = sortModelGroups(buildModelGroups(subset), sortField, sortDir);
        return { worker: w, subset, groups, orphans: orphanDrafts(subset, groups) };
      }),
    [workers, filteredModels, locations, sortField, sortDir]
  );

  const visibleHfResults = useMemo(() => {
    const list = hfResults.filter((r) => paramsInRange(paramsBFromText(r.id), hfParamsLoIndex, hfParamsHiIndex));
    if (hfSortField === "relevance") return list;
    const dir = hfSortDir === "asc" ? 1 : -1;
    return [...list].sort((a, b) => {
      switch (hfSortField) {
        case "name":
          return dir * a.id.localeCompare(b.id);
        case "likes":
          return dir * (a.likes - b.likes);
        case "downloads":
          return dir * (a.downloads - b.downloads);
        case "params":
          return dir * ((paramsBFromText(a.id) ?? -1) - (paramsBFromText(b.id) ?? -1));
        case "newest": {
          const ta = a.created_at ? new Date(a.created_at).getTime() : -1;
          const tb = b.created_at ? new Date(b.created_at).getTime() : -1;
          return dir * (ta - tb);
        }
        default:
          return 0;
      }
    });
  }, [hfResults, hfParamsLoIndex, hfParamsHiIndex, hfSortField, hfSortDir]);

  async function loadModels() {
    setModels(await api.listModels());
  }

  async function loadLocations() {
    try {
      const res = await api.getModelLocations();
      setLocations(res.locations);
      setUnreachableLocationWorkers(res.unreachable);
    } catch {
      // Best-effort, same posture as NewRun's hfUpdates fetch -- if this
      // route is unreachable every worker section just shows no models
      // rather than breaking the page.
    }
  }

  // Deletes the file from exactly one worker's model_dir -- the registry row
  // (and every other worker's copy) is left untouched, so this never hits
  // the "N run result(s) reference it" conflict the old whole-registry
  // delete could -- there's no DB data being removed here at all.
  async function handleDeleteModelFile(m: Model, worker: Worker) {
    const filename = m.source === "local" ? m.filename : m.hf_file;
    if (!filename) return;
    if (
      !window.confirm(
        `Delete ${m.filename} from ${worker.displayName}? This only removes the file on that machine -- the model stays registered.`
      )
    ) {
      return;
    }
    const key = `${m.id}:${worker.id}`;
    setDeletingKey(key);
    setModelsMsg("");
    try {
      await api.deleteModelFileFromWorker(worker.id, filename);
      setModelsMsg(`Queued: delete ${m.filename} from ${worker.displayName}`);
      await Promise.all([loadModels(), loadLocations()]);
    } catch (err) {
      setModelsMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDeletingKey(null);
    }
  }

  // Backfills param_count for a model registered before real HF-derived
  // lookup existed (or downloaded before this app fetched it) -- see
  // server/src/routes/models.ts's backfill-param-count route. Never touches
  // a worker, just the model's hf_repo, so it's fast.
  async function lookupParamCount(m: Model) {
    setParamLookupBusyId(m.id);
    setParamLookupErr((s) => ({ ...s, [m.id]: "" }));
    try {
      const res = await api.backfillModelParamCount(m.id);
      if (res.param_count == null) {
        setParamLookupErr((s) => ({ ...s, [m.id]: "Not on Hugging Face." }));
      } else {
        await loadModels();
      }
    } catch (err) {
      setParamLookupErr((s) => ({ ...s, [m.id]: err instanceof Error ? err.message : String(err) }));
    } finally {
      setParamLookupBusyId(null);
    }
  }

  useEffect(() => {
    void loadModels();
    void loadLocations();
    (async () => {
      const list = await api.listWorkers();
      setWorkers(list);
      if (list.length === 1) setDownloadWorker(list[0].id);
    })();
  }, []);

  useEffect(() => {
    persistDownloads(downloading);
  }, [downloading]);

  useEffect(() => {
    persistHfSearch({
      query: hfQuery,
      results: hfResults,
      expandedRepo,
      filesByRepo,
      paramsLoIndex: hfParamsLoIndex,
      paramsHiIndex: hfParamsHiIndex,
      sortField: hfSortField,
      sortDir: hfSortDir,
    });
  }, [hfQuery, hfResults, expandedRepo, filesByRepo, hfParamsLoIndex, hfParamsHiIndex, hfSortField, hfSortDir]);

  // Cleans up a download once the worker stops reporting progress for it --
  // the only signal available that it's done, since the worker's download
  // now runs fully in the background (see worker/src/index.ts's POST
  // /models/download) and doesn't distinguish "finished" from "failed" via
  // this progress endpoint, only "no longer in flight". Refreshing the model
  // list either way is harmless: it either surfaces the newly-registered
  // model or changes nothing. This is the *only* completion path now -- the
  // POST that starts a download resolves as soon as the worker acks it, long
  // before the transfer itself finishes, so there's no blocking call left to
  // hang cleanup off of.
  function finalizeDownload(path: string) {
    setDownloading((d) => {
      const next = { ...d };
      delete next[path];
      return next;
    });
    setProgress((p) => {
      const next = { ...p };
      delete next[path];
      return next;
    });
    setSpeeds((s) => {
      const next = { ...s };
      delete next[path];
      return next;
    });
    delete prevSampleRef.current[path];
    delete smoothedSpeedRef.current[path];
    delete seenRef.current[path];
    delete missesRef.current[path];
    setHfMsg(`${path} is no longer downloading -- refreshed the model list.`);
    void loadModels();
    // Safety net: the worker's progress entry disappearing only means the
    // byte transfer is done, not that the server has finished registering
    // the model (that still needs the worker's callback to reach
    // POST /api/models/download-callback and hit repo.registerModel -- see
    // worker/src/index.ts's runModelDownload). The loadModels() above can
    // race ahead of that and miss the new row; this second pass catches it.
    setTimeout(() => void loadModels(), 2000);
  }

  // Reads progress off the owning worker's activeJobProgress (one
  // GET /api/workers read per tick, not one call per file) instead of a
  // dedicated per-file progress endpoint -- that endpoint doesn't exist
  // anymore (MULTIUSER_PLAN.md §1.5/§1.11: workers push progress via their
  // heartbeat, roughly every 10s, not continuously). Polling faster than
  // that heartbeat cadence wouldn't see any new data, so this ticks every 2s
  // (matching RunDetail's own live-poll interval) rather than the old 750ms.
  //
  // Known limitation: since a worker now runs exactly one job at a time, if
  // two downloads are queued on the SAME worker back to back, only the
  // first one's progress is visible (the worker hasn't started the second
  // at all yet) -- its entry may look stalled until the first finishes and
  // it actually becomes the active job. Downloads on different workers are
  // unaffected; each worker's own activeJobProgress is independent.
  useEffect(() => {
    const entries = Object.entries(downloading);
    if (entries.length === 0) return;

    async function poll() {
      let workerList: Worker[];
      try {
        workerList = await api.listWorkers();
      } catch {
        return; // transient -- try again next tick
      }
      const byId = new Map(workerList.map((w) => [w.id, w]));

      for (const [path, state] of entries) {
        const worker = byId.get(state.workerId);
        const active = worker?.activeJobProgress;
        const stillDownloadingThis = worker?.status === "busy" && active?.phase === "downloading";

        if (stillDownloadingThis && active) {
          seenRef.current[path] = true;
          missesRef.current[path] = 0;
          const bytes = active.bytes ?? 0;
          const total = active.total_bytes ?? null;
          setProgress((prev) => ({ ...prev, [path]: { bytes, total } }));
          const prevSample = prevSampleRef.current[path];
          const now = Date.now();
          if (prevSample && now > prevSample.time) {
            const instantBytesPerSec = Math.max(0, ((bytes - prevSample.bytes) / (now - prevSample.time)) * 1000);
            // EMA with alpha ~0.25 -- smooths poll-to-poll jitter without
            // lagging too far behind a real step change.
            const SPEED_SMOOTHING_ALPHA = 0.25;
            const prevSmoothed = smoothedSpeedRef.current[path];
            const smoothed =
              prevSmoothed == null
                ? instantBytesPerSec
                : prevSmoothed + SPEED_SMOOTHING_ALPHA * (instantBytesPerSec - prevSmoothed);
            smoothedSpeedRef.current[path] = smoothed;
            setSpeeds((s) => ({ ...s, [path]: smoothed }));
          }
          prevSampleRef.current[path] = { bytes, time: now };
        } else {
          // No longer this worker's active job -- either it just finished
          // (success or failure), or it hasn't started yet (still queued
          // behind another job). A grace period before finalizing either
          // way, since the worker's heartbeat cadence means "just changed"
          // and "genuinely done" look identical for a few ticks.
          const misses = (missesRef.current[path] ?? 0) + 1;
          missesRef.current[path] = misses;
          const SEEN_GRACE_MISSES = 3;
          const NEVER_SEEN_TIMEOUT_MISSES = 6;
          if ((seenRef.current[path] && misses >= SEEN_GRACE_MISSES) || misses >= NEVER_SEEN_TIMEOUT_MISSES) {
            finalizeDownload(path);
          }
        }
      }
    }
    void poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [downloading]);

  async function handleAdd(e: FormEvent) {
    e.preventDefault();
    setAddMsg("");
    let metadata: ModelMetadata = {};
    if (form.metadata.trim()) {
      try {
        metadata = JSON.parse(form.metadata);
      } catch {
        setAddMsg("Invalid metadata JSON");
        return;
      }
    }
    try {
      await api.registerModel({
        source: form.source,
        id: form.id || undefined,
        filename: form.filename || undefined,
        size_bytes: Number(form.size_bytes) || 0,
        hf_repo: form.hf_repo || undefined,
        hf_file: form.hf_file || undefined,
        metadata,
      });
      setAddMsg("Registered ✓");
      setForm(emptyForm);
      await loadModels();
    } catch (err) {
      setAddMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleSearch(e: FormEvent) {
    e.preventDefault();
    setHfMsg("");
    setHfResults([]);
    setExpandedRepo(null);
    if (!hfQuery.trim()) return;
    setSearching(true);
    try {
      const results = await api.searchHf(hfQuery);
      setHfResults(results);
      if (!results.length) setHfMsg("No repos found.");
    } catch (err) {
      setHfMsg(`Search error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSearching(false);
    }
  }

  async function toggleRepo(repoId: string) {
    if (expandedRepo === repoId) {
      setExpandedRepo(null);
      return;
    }
    setExpandedRepo(repoId);
    if (!filesByRepo[repoId]) {
      try {
        const files = await api.listHfFiles(repoId);
        setFilesByRepo((f) => ({ ...f, [repoId]: files }));
      } catch (err) {
        setHfMsg(`Files error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  async function handleDownload(repoId: string, file: HfFileEntry) {
    if (!downloadWorker) {
      setPickWorkerWarning(true);
      workerPickerRef.current?.focus();
      workerPickerRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const workerId = downloadWorker;
    setPickWorkerWarning(false);
    setHfMsg("");
    try {
      // Resolves once the download is queued, not once it's done -- the
      // actual transfer happens once the worker claims the job (may take up
      // to one queue-poll cycle even for an idle machine), so completion is
      // only observable via the progress-poll effect below, same as an
      // entry restored from localStorage after a refresh. finalizeDownload
      // is the one place that ever clears `downloading` now.
      await api.downloadHfFile(workerId, repoId, file.path);
      setDownloading((d) => ({ ...d, [file.path]: { repoId, workerId, startedAt: Date.now(), file } }));
    } catch (err) {
      setHfMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function renderModelRow(m: Model, worker: Worker, indent: boolean) {
    const paramsB = modelParamsB(m);
    const key = `${m.id}:${worker.id}`;
    return (
      <div
        key={key}
        className={`flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 ${indent ? "ml-6 border-l-2 border-border pl-3" : ""}`}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            {indent && !/^mtp\//i.test(m.filename) && <span className="text-xs text-muted">MTP/</span>}
            <span className="truncate text-fg">{m.filename}</span>
            {isMtpDraftModel(m) && (
              <span title="Standalone MTP/draft companion file -- not benchmarkable on its own, only usable as an MTP model paired with a base model on New Run">
                <StatusPill label="MTP draft" tone="accent" />
              </span>
            )}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
            <code>{shortId(m.id)}</code>
            <span>{modelAuthor(m)}</span>
            <span>{modelFamily(m)}</span>
            <span className="flex items-center gap-1">
              <span title={typeof m.metadata.param_count !== "number" ? "Estimated from the filename -- may be inaccurate" : undefined}>
                {formatParamsB(paramsB)}
              </span>
              {typeof m.metadata.param_count !== "number" && m.hf_repo && (
                <button
                  type="button"
                  disabled={paramLookupBusyId === m.id}
                  onClick={() => lookupParamCount(m)}
                  className="text-muted hover:text-accent disabled:opacity-50"
                  title="Look up the real parameter count from Hugging Face"
                  aria-label={`Look up real parameter count for ${m.filename}`}
                >
                  <IconRefreshCw width={12} height={12} className={paramLookupBusyId === m.id ? "animate-spin" : undefined} />
                </button>
              )}
            </span>
            <span>{formatBytes(m.size_bytes)}</span>
            <StatusPill label={m.source} tone="muted" />
            {m.hf_repo && m.hf_file && (
              <a href={hfFileUrl(m.hf_repo, m.hf_file)} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                {m.hf_repo}/{m.hf_file}
              </a>
            )}
          </div>
          {paramLookupErr[m.id] && <div className="text-xs text-danger">{paramLookupErr[m.id]}</div>}
        </div>
        <button
          type="button"
          disabled={deletingKey === key}
          onClick={() => handleDeleteModelFile(m, worker)}
          className="rounded-md border border-border p-1.5 text-muted hover:border-danger/40 hover:text-danger disabled:opacity-50"
          aria-label={`Delete ${m.filename} from ${worker.displayName}`}
          title={`Delete this file from ${worker.displayName} only -- the model stays registered`}
        >
          <IconTrash width={14} height={14} />
        </button>
      </div>
    );
  }

  function renderModelGroup(group: ModelGroup, worker: Worker) {
    return (
      <div key={`${group.key}:${worker.id}`} className="py-1">
        <div className="px-4 pt-2 pb-1 text-xs font-semibold uppercase tracking-wide text-muted">
          {group.author} · {group.family} · {group.label}
        </div>
        {group.quants.map(({ base }) => (
          <div key={base.id}>{renderModelRow(base, worker, false)}</div>
        ))}
        {group.quants.map(({ drafts }) =>
          drafts.map((draft) => <div key={draft.id}>{renderModelRow(draft, worker, true)}</div>)
        )}
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-semibold text-fg">Models</h1>
      <p className="mt-1 text-sm text-muted">
        The model catalog (filenames and metadata) is shared across every user, so downloads can be
        deduplicated — a filename you register or download may be visible to others. Benchmark
        results themselves are never shared unless you opt in (Settings).
      </p>

      <section className="mt-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">My models</h2>
        {models.length === 0 ? (
          <p className="mt-2 text-sm text-muted">No models registered yet.</p>
        ) : (
          <>
            <div className="mt-3 flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Author</span>
                <select
                  value={authorFilter}
                  onChange={(e) => setAuthorFilter(e.target.value)}
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                >
                  <option value="">All</option>
                  {authorOptions.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Family</span>
                <select
                  value={familyFilter}
                  onChange={(e) => setFamilyFilter(e.target.value)}
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                >
                  <option value="">All</option>
                  {familyOptions.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Parameters</span>
                <ParamsRangeSlider
                  loIndex={paramsLoIndex}
                  hiIndex={paramsHiIndex}
                  onChange={(lo, hi) => {
                    setParamsLoIndex(lo);
                    setParamsHiIndex(hi);
                  }}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Sort by</span>
                <div className="flex items-center gap-1.5">
                  <select
                    value={sortField}
                    onChange={(e) => setSortField(e.target.value as SortField)}
                    className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                  >
                    {SORT_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
                    className="rounded-lg border border-border px-2.5 py-1.5 text-fg hover:border-accent/40"
                    aria-label={sortDir === "asc" ? "Ascending" : "Descending"}
                    title={sortDir === "asc" ? "Ascending" : "Descending"}
                  >
                    {sortDir === "asc" ? "↑" : "↓"}
                  </button>
                </div>
              </label>
              {(authorFilter || familyFilter || paramsLoIndex !== 0 || paramsHiIndex !== PARAMS_STOPS.length - 1) && (
                <button
                  type="button"
                  onClick={() => {
                    setAuthorFilter("");
                    setFamilyFilter("");
                    setParamsLoIndex(0);
                    setParamsHiIndex(PARAMS_STOPS.length - 1);
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-accent"
                >
                  Clear filters
                </button>
              )}
            </div>

            {unreachableLocationWorkers.length > 0 && (
              <p className="mt-3 text-xs text-warning">
                Couldn't check what's on:{" "}
                {unreachableLocationWorkers
                  .map((id) => workers.find((w) => w.id === id)?.displayName ?? id)
                  .join(", ")}{" "}
                -- their models may be missing below even though they actually have them.
              </p>
            )}

            <div className="mt-3 flex flex-col gap-3">
              {modelsByWorker.map(({ worker, subset, groups, orphans }) => (
                <details key={worker.id} open className="group rounded-xl border border-border bg-surface">
                  <summary className="flex cursor-pointer items-center justify-between px-4 py-2.5 text-sm font-semibold text-fg">
                    <span className="flex items-center gap-2">
                      {worker.displayName}
                      <span className="text-xs font-normal text-muted">
                        ({subset.length} file{subset.length === 1 ? "" : "s"})
                      </span>
                    </span>
                    <IconChevronDown width={16} height={16} className="transition-transform group-open:rotate-180" />
                  </summary>
                  <div className="divide-y divide-border border-t border-border">
                    {subset.length === 0 ? (
                      <p className="px-4 py-3 text-sm text-muted">No models on this worker.</p>
                    ) : (
                      <>
                        {groups.map((group) => renderModelGroup(group, worker))}
                        {orphans.map((d) => renderModelRow(d, worker, false))}
                      </>
                    )}
                  </div>
                </details>
              ))}
            </div>
            {modelsMsg && <p className="mt-3 text-sm text-muted">{modelsMsg}</p>}
          </>
        )}
      </section>

      {Object.keys(downloading).length > 0 && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Active downloads</h2>
          <div className="mt-3 flex flex-col gap-2">
            {Object.entries(downloading).map(([path, state]) => {
              const prog = progress[path];
              const speed = speeds[path];
              const pct = prog?.total != null ? Math.min(100, Math.round((prog.bytes / prog.total) * 100)) : null;
              const totalBytes = prog?.total ?? state.file.size_bytes ?? null;
              return (
                <div
                  key={path}
                  className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <IconDownload width={14} height={14} className="text-accent" />
                      <span className="text-fg">{path}</span>
                      <span className="text-xs text-muted">
                        {state.repoId} · {workers.find((w) => w.id === state.workerId)?.displayName ?? state.workerId}
                      </span>
                    </div>
                    <span className="text-xs text-muted">{pct !== null ? `${pct}%` : "Starting…"}</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
                    {pct !== null ? (
                      <div
                        className="h-full rounded-full bg-accent transition-[width]"
                        style={{ width: `${pct}%` }}
                      />
                    ) : (
                      <div className="progress-indeterminate h-full w-full rounded-full" />
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <span>
                      {formatBytes(prog?.bytes ?? 0)}
                      {totalBytes != null ? ` / ${formatBytes(totalBytes)}` : ""}
                    </span>
                    {speed !== undefined && <span>· {formatBytes(speed)}/s</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          Search &amp; download (Hugging Face)
        </h2>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Search by model name, pick a repo to see its quant files, then download one straight to
          a worker's model directory. Large files can take a while -- progress is tracked under
          "Active downloads" above and survives a page refresh.
        </p>

        <form onSubmit={handleSearch} className="mt-3 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">Model name</span>
            <input
              value={hfQuery}
              onChange={(e) => setHfQuery(e.target.value)}
              placeholder="e.g. llama-3-8b"
              className="w-64 rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">Download to worker</span>
            <select
              ref={workerPickerRef}
              value={downloadWorker}
              onChange={(e) => {
                setDownloadWorker(e.target.value);
                if (e.target.value) setPickWorkerWarning(false);
              }}
              className={`w-48 rounded-lg border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent ${
                pickWorkerWarning ? "border-danger" : "border-border"
              }`}
            >
              <option value="">select…</option>
              {workers.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.displayName}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            disabled={searching}
            className="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg disabled:opacity-50"
          >
            {searching ? "Searching…" : "Search"}
          </button>
          {pickWorkerWarning && (
            <span className="text-sm text-danger">Choose a worker above before downloading.</span>
          )}
        </form>

        {hfResults.length > 0 && (
          <>
            <div className="mt-4 flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Parameters</span>
                <ParamsRangeSlider
                  loIndex={hfParamsLoIndex}
                  hiIndex={hfParamsHiIndex}
                  onChange={(lo, hi) => {
                    setHfParamsLoIndex(lo);
                    setHfParamsHiIndex(hi);
                  }}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">Sort by</span>
                <div className="flex items-center gap-1.5">
                  <select
                    value={hfSortField}
                    onChange={(e) => setHfSortField(e.target.value as HfSortField)}
                    className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                  >
                    {HF_SORT_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                  {hfSortField !== "relevance" && (
                    <button
                      type="button"
                      onClick={() => setHfSortDir((d) => (d === "asc" ? "desc" : "asc"))}
                      className="rounded-lg border border-border px-2.5 py-1.5 text-fg hover:border-accent/40"
                      aria-label={hfSortDir === "asc" ? "Ascending" : "Descending"}
                      title={hfSortDir === "asc" ? "Ascending" : "Descending"}
                    >
                      {hfSortDir === "asc" ? "↑" : "↓"}
                    </button>
                  )}
                </div>
              </label>
              {(hfParamsLoIndex !== 0 || hfParamsHiIndex !== PARAMS_STOPS.length - 1) && (
                <button
                  type="button"
                  onClick={() => {
                    setHfParamsLoIndex(0);
                    setHfParamsHiIndex(PARAMS_STOPS.length - 1);
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-accent"
                >
                  Clear filters
                </button>
              )}
            </div>
          <div className="mt-4 flex flex-col gap-2">
            {visibleHfResults.map((r) => (
              <div key={r.id} className="overflow-hidden rounded-lg border border-border">
                <button
                  type="button"
                  onClick={() => toggleRepo(r.id)}
                  className="flex w-full items-center justify-between gap-3 bg-surface px-4 py-2.5 text-left text-sm hover:bg-white/5"
                >
                  <span className="text-fg">
                    {r.id}{" "}
                    <a
                      href={hfRepoUrl(r.id)}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-xs text-accent hover:underline"
                    >
                      view on HF ↗
                    </a>
                  </span>
                  <span className="flex items-center gap-3 text-xs text-muted">
                    {r.downloads.toLocaleString()} downloads · {r.likes.toLocaleString()} likes
                    {formatRelativeTime(r.created_at) && <span>· added {formatRelativeTime(r.created_at)}</span>}
                    <IconChevronDown
                      width={16}
                      height={16}
                      className={`transition-transform ${expandedRepo === r.id ? "rotate-180" : ""}`}
                    />
                  </span>
                </button>
                {expandedRepo === r.id && (
                  <div className="border-t border-border bg-surface-raised">
                    {!filesByRepo[r.id] ? (
                      <p className="px-4 py-3 text-sm text-muted">Loading files…</p>
                    ) : filesByRepo[r.id].length === 0 ? (
                      <p className="px-4 py-3 text-sm text-muted">No .gguf files found in this repo.</p>
                    ) : (
                      <ul className="divide-y divide-border">
                        {filesByRepo[r.id].map((f) => {
                          const dl = downloading[f.path];
                          const prog = progress[f.path];
                          const speed = speeds[f.path];
                          const pct =
                            prog?.total != null ? Math.min(100, Math.round((prog.bytes / prog.total) * 100)) : null;
                          return (
                            <li key={f.path} className="flex flex-col gap-2 px-4 py-2.5 text-sm">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-2">
                                  <a
                                    href={hfFileUrl(r.id, f.path)}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-fg hover:text-accent hover:underline"
                                  >
                                    {f.path}
                                  </a>
                                  {f.quant && <StatusPill label={f.quant} tone="muted" />}
                                  <span className="text-xs text-muted">{formatBytes(f.size_bytes)}</span>
                                </div>
                                <button
                                  type="button"
                                  disabled={dl !== undefined}
                                  onClick={() => handleDownload(r.id, f)}
                                  className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs font-semibold text-fg hover:border-accent/40 hover:text-accent disabled:opacity-60"
                                >
                                  <IconDownload width={14} height={14} />
                                  {dl ? (pct !== null ? `${pct}%` : "Starting…") : "Download"}
                                </button>
                              </div>
                              {dl && (
                                <div className="flex flex-col gap-1">
                                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
                                    {pct !== null ? (
                                      <div
                                        className="h-full rounded-full bg-accent transition-[width]"
                                        style={{ width: `${pct}%` }}
                                      />
                                    ) : (
                                      <div className="progress-indeterminate h-full w-full rounded-full" />
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2 text-xs text-muted">
                                    <span>
                                      {formatBytes(prog?.bytes ?? 0)}
                                      {prog?.total != null ? ` / ${formatBytes(prog.total)}` : ""}
                                    </span>
                                    {speed !== undefined && <span>· {formatBytes(speed)}/s</span>}
                                  </div>
                                </div>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
          </>
        )}
        {hfMsg && <p className="mt-3 text-sm text-muted">{hfMsg}</p>}
      </section>

      <details className="group mt-8 rounded-xl border border-border bg-surface">
        <summary className="flex cursor-pointer items-center justify-between px-5 py-3 text-sm font-semibold uppercase tracking-wide text-muted">
          Add a model manually
          <IconChevronDown width={16} height={16} className="transition-transform group-open:rotate-180" />
        </summary>
        <form onSubmit={handleAdd} className="flex flex-col gap-3 border-t border-border px-5 py-4">
          <div className="flex flex-wrap gap-3">
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Source</span>
              <select
                value={form.source}
                onChange={(e) => setForm({ ...form, source: e.target.value as ModelSource })}
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              >
                <option value="local">local</option>
                <option value="huggingface">huggingface</option>
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">ID (sha256, optional)</span>
              <input
                value={form.id}
                onChange={(e) => setForm({ ...form, id: e.target.value })}
                placeholder="sha256:…"
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              />
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Filename</span>
              <input
                value={form.filename}
                onChange={(e) => setForm({ ...form, filename: e.target.value })}
                placeholder="model.gguf"
                className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              />
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted">Size bytes</span>
              <input
                value={form.size_bytes}
                onChange={(e) => setForm({ ...form, size_bytes: e.target.value })}
                type="number"
                placeholder="0"
                className="w-32 rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
              />
            </label>
          </div>
          {form.source === "huggingface" && (
            <div className="flex flex-wrap gap-3">
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">HF repo</span>
                <input
                  value={form.hf_repo}
                  onChange={(e) => setForm({ ...form, hf_repo: e.target.value })}
                  placeholder="user/repo"
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                />
              </label>
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-muted">HF file</span>
                <input
                  value={form.hf_file}
                  onChange={(e) => setForm({ ...form, hf_file: e.target.value })}
                  placeholder="model-q4_k_m.gguf"
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                />
              </label>
            </div>
          )}
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted">Metadata JSON</span>
            <input
              value={form.metadata}
              onChange={(e) => setForm({ ...form, metadata: e.target.value })}
              placeholder='{"arch":"llama","quant":"q4_k_m"}'
              className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
            />
          </label>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="self-start rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-accent-fg"
            >
              Register
            </button>
            {addMsg && <span className="text-sm text-muted">{addMsg}</span>}
          </div>
        </form>
      </details>
    </div>
  );
}
