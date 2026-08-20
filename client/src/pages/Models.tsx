import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
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
  // Identifies this download's worker_jobs row -- lets the poll effect match
  // this exact job in worker.activeDownloads (rather than string-matching
  // detail against repoId/file.path, which used to be the only way to tell
  // downloads apart when a worker could only ever run one job at a time) and
  // lets the Pause button target the right job.
  jobId: string;
  // Set once the user pauses this download -- worker.activeDownloads
  // legitimately stops including it forever at that point (same as a real
  // completion/failure), so this flag is what tells the poll effect to stop
  // waiting for it to reappear and show a Resume affordance instead of
  // finalizing the row away.
  paused?: boolean;
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

// Remembers the manual-add panel's collapsed/expanded state across visits --
// defaults to collapsed (the common case is browsing/downloading from
// Hugging Face, not hand-entering a model).
const MANUAL_ADD_OPEN_KEY = "llamatoaster:models-manual-add-open";

function loadManualAddOpen(): boolean {
  try {
    return localStorage.getItem(MANUAL_ADD_OPEN_KEY) === "1";
  } catch {
    return false;
  }
}

function persistManualAddOpen(open: boolean): void {
  try {
    localStorage.setItem(MANUAL_ADD_OPEN_KEY, open ? "1" : "0");
  } catch {
    /* localStorage unavailable -- state just won't survive a refresh */
  }
}

// `downloading`/`progress`/`speeds` below used to be keyed by bare filename
// alone, which collided whenever two downloads shared a filename -- common
// for GGUF quants, since many uploaders reuse fixed names like
// "model.Q4_K_M.gguf" across different repos -- or when the same repo/file
// was sent to two different workers at once. Either case silently merged two
// unrelated transfers into one dictionary slot. Matches the worker's own
// `${hf_repo}/${hf_file}` progressKey (see worker/src/index.ts's
// executeDownloadModelJob), prefixed with the worker id.
function downloadKey(workerId: string, repoId: string, path: string): string {
  return `${workerId}::${repoId}/${path}`;
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

// "Name (A-Z)" and "Parameters" (the old client-side-only sort options) are
// gone -- HF's search API has no equivalent for either, and now that results
// are real paginated pages (server/src/hf.ts), a client-only re-sort would
// only ever see the current page's 15 items, not the true global order,
// which would be actively misleading rather than just a lesser feature.
type HfSortField = "relevance" | "downloads" | "likes" | "newest" | "updated";

const HF_SORT_OPTIONS: { value: HfSortField; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "downloads", label: "Downloads" },
  { value: "likes", label: "Likes" },
  { value: "newest", label: "Newest" },
  { value: "updated", label: "Recently updated" },
];

// Maps the client's sort dropdown to HF's own API sort field names --
// "relevance" sends no sort param at all (HF's default search-relevance
// ordering).
const HF_SORT_SERVER_FIELD: Partial<Record<HfSortField, "downloads" | "likes" | "createdAt" | "lastModified">> = {
  downloads: "downloads",
  likes: "likes",
  newest: "createdAt",
  updated: "lastModified",
};

interface HfPage {
  results: HfRepoSearchResult[];
  nextCursor: string | null;
}

// Session-scoped (not localStorage) -- a search is "current" for as long as
// the tab lives, but stale HF result counts/likes shouldn't linger forever
// across separate visits the way a remembered sweep should.
const HF_SEARCH_STORAGE_KEY = "llamatoaster:hf-search";

interface PersistedHfSearch {
  query: string;
  pages: HfPage[];
  pageIndex: number;
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
  const [manualAddOpen, setManualAddOpen] = useState(loadManualAddOpen);

  const [hfQuery, setHfQuery] = useState(() => loadPersistedHfSearch()?.query ?? "");
  // One entry per fetched page, indexed by page number -- pageIndex points
  // at which one is currently shown. "Prev" just moves the index back (no
  // refetch, already cached); "Next" moves forward if that page is already
  // cached, otherwise fetches using the current page's nextCursor. HF's
  // search API paginates via a forward-only opaque cursor, not an
  // offset/page number, so there's no way to jump directly to an arbitrary
  // page -- see server/src/hf.ts.
  const [hfPages, setHfPages] = useState<HfPage[]>(() => loadPersistedHfSearch()?.pages ?? []);
  const [hfPageIndex, setHfPageIndex] = useState(() => loadPersistedHfSearch()?.pageIndex ?? 0);
  const hfResults = hfPages[hfPageIndex]?.results ?? [];
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

  // Sort is applied server-side now (server/src/hf.ts) -- this only
  // client-side-filters the current page's 15 results by parameter count, a
  // refinement that doesn't need to be globally correct the way a sort does.
  const visibleHfResults = useMemo(
    () => hfResults.filter((r) => paramsInRange(paramsBFromText(r.id), hfParamsLoIndex, hfParamsHiIndex)),
    [hfResults, hfParamsLoIndex, hfParamsHiIndex]
  );

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

  // Shows a default "top 15" HF listing on first load instead of leaving the
  // section blank until the user types something -- but only when there's
  // nothing already restored from sessionStorage (a real prior search, or
  // this same default listing from earlier in the tab's life), so revisiting
  // the page mid-session doesn't stomp on what the user was looking at.
  useEffect(() => {
    if (hfPages.length === 0) void handleSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    persistDownloads(downloading);
  }, [downloading]);

  useEffect(() => {
    persistHfSearch({
      query: hfQuery,
      pages: hfPages,
      pageIndex: hfPageIndex,
      expandedRepo,
      filesByRepo,
      paramsLoIndex: hfParamsLoIndex,
      paramsHiIndex: hfParamsHiIndex,
      sortField: hfSortField,
      sortDir: hfSortDir,
    });
  }, [hfQuery, hfPages, hfPageIndex, expandedRepo, filesByRepo, hfParamsLoIndex, hfParamsHiIndex, hfSortField, hfSortDir]);

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
  // Shared by finalizeDownload (genuine completion/failure) and
  // handleCancelDownload (user-discarded) -- just the bookkeeping common to
  // both, since only the messaging/loadModels behavior after differs.
  function removeDownloadState(key: string) {
    setDownloading((d) => {
      const next = { ...d };
      delete next[key];
      return next;
    });
    setProgress((p) => {
      const next = { ...p };
      delete next[key];
      return next;
    });
    setSpeeds((s) => {
      const next = { ...s };
      delete next[key];
      return next;
    });
    delete prevSampleRef.current[key];
    delete smoothedSpeedRef.current[key];
    delete seenRef.current[key];
    delete missesRef.current[key];
  }

  function finalizeDownload(key: string) {
    const label = downloading[key]?.file.path ?? key;
    removeDownloadState(key);
    setHfMsg(`${label} is no longer downloading -- refreshed the model list.`);
    // Both calls matter: loadModels() brings in the new registry row, but
    // presentModels/modelsByWorker gate visibility on `locations` (which
    // worker actually has the file) -- see handleDeleteModelFile above for
    // the same pairing. Without loadLocations() too, a freshly-completed
    // download stays invisible until a manual page reload.
    void loadModels();
    void loadLocations();
    // Safety net: the worker's progress entry disappearing only means the
    // byte transfer is done, not that the server has finished registering
    // the model (that still needs the worker's callback to reach
    // POST /api/models/download-callback and hit repo.registerModel -- see
    // worker/src/index.ts's runModelDownload). The pass above can race ahead
    // of that and miss the new row/location; this second pass catches it.
    setTimeout(() => {
      void loadModels();
      void loadLocations();
    }, 2000);
  }

  // Reads progress off the owning worker's activeDownloads (one
  // GET /api/workers read per tick, not one call per file) instead of a
  // dedicated per-file progress endpoint -- that endpoint doesn't exist
  // anymore (MULTIUSER_PLAN.md §1.5/§1.11: workers push progress via their
  // heartbeat, roughly every 10s, not continuously). Polling faster than
  // that heartbeat cadence wouldn't see any new data, so this ticks every 2s
  // (matching RunDetail's own live-poll interval) rather than the old 750ms.
  //
  // Each entry is matched by jobId now (worker/src/index.ts's downloadJobPool
  // reports every concurrently-active download's own progress, keyed by its
  // own job id) rather than string-matching a single shared activeJobProgress
  // -- several downloads on the SAME worker genuinely run and report progress
  // at once now, not just one at a time.
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

      for (const [key, state] of entries) {
        if (state.paused) continue; // legitimately gone from activeDownloads forever -- nothing to poll for
        const worker = byId.get(state.workerId);
        const active = worker?.activeDownloads?.find((d) => d.job_id === state.jobId);

        if (active) {
          seenRef.current[key] = true;
          missesRef.current[key] = 0;
          const bytes = active.bytes ?? 0;
          const total = active.total_bytes ?? null;
          setProgress((prev) => ({ ...prev, [key]: { bytes, total } }));
          const prevSample = prevSampleRef.current[key];
          const now = Date.now();
          if (prevSample && now > prevSample.time) {
            const instantBytesPerSec = Math.max(0, ((bytes - prevSample.bytes) / (now - prevSample.time)) * 1000);
            // EMA with alpha ~0.25 -- smooths poll-to-poll jitter without
            // lagging too far behind a real step change.
            const SPEED_SMOOTHING_ALPHA = 0.25;
            const prevSmoothed = smoothedSpeedRef.current[key];
            const smoothed =
              prevSmoothed == null
                ? instantBytesPerSec
                : prevSmoothed + SPEED_SMOOTHING_ALPHA * (instantBytesPerSec - prevSmoothed);
            smoothedSpeedRef.current[key] = smoothed;
            setSpeeds((s) => ({ ...s, [key]: smoothed }));
          }
          prevSampleRef.current[key] = { bytes, time: now };
        } else if (seenRef.current[key]) {
          // Was actively downloading and just dropped out of activeDownloads
          // -- a genuine finish (success or failure), not a queued job, so no
          // need to consult job status. A short grace period since the
          // worker's heartbeat cadence means "just changed" and "genuinely
          // done" look identical for a tick or two.
          const misses = (missesRef.current[key] ?? 0) + 1;
          missesRef.current[key] = misses;
          const SEEN_GRACE_MISSES = 3;
          if (misses >= SEEN_GRACE_MISSES) finalizeDownload(key);
        } else {
          // Never seen active yet -- could be queued behind the worker's
          // max_concurrent_downloads cap (a pending job never appears in
          // activeDownloads until claimed, and can legitimately wait there
          // for a long time) rather than actually gone. Ask the server which
          // it is instead of guessing from a timeout, so a merely-queued
          // download doesn't get wrongly finalized as "dropped".
          try {
            const { status } = await api.getDownloadStatus(state.workerId, state.jobId);
            if (status === "pending" || status === "claimed") continue; // still queued/just claimed -- keep waiting
            finalizeDownload(key); // completed/failed/cancelled
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) {
              finalizeDownload(key); // job/worker genuinely gone
            }
            // otherwise transient (network hiccup, server restart) -- try again next tick
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

  // Fetches page 0 fresh and replaces hfPages wholesale -- shared by the
  // search form's submit, and by the sort field/direction controls (which
  // must restart from page 0 since previously-cached pages are in the wrong
  // order once the sort changes). Takes field/dir as explicit params rather
  // than reading hfSortField/hfSortDir off state so a control's own onChange
  // can pass its *new* value without waiting a render for state to catch up.
  async function runFreshHfSearch(field: HfSortField, dir: SortDir) {
    setHfMsg("");
    setExpandedRepo(null);
    setSearching(true);
    try {
      const { results, nextCursor } = await api.searchHf({
        q: hfQuery,
        sort: HF_SORT_SERVER_FIELD[field],
        direction: field === "relevance" ? undefined : dir === "asc" ? 1 : -1,
      });
      setHfPages([{ results, nextCursor }]);
      setHfPageIndex(0);
      if (!results.length) setHfMsg("No repos found.");
    } catch (err) {
      setHfMsg(`Search error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSearching(false);
    }
  }

  // query may be empty (that's the default "top 15 by downloads" listing
  // shown before the user types anything). e is undefined when triggered
  // programmatically (on mount) rather than from the form's onSubmit.
  async function handleSearch(e?: FormEvent) {
    e?.preventDefault();
    await runFreshHfSearch(hfSortField, hfSortDir);
  }

  async function goToNextHfPage() {
    if (hfPageIndex + 1 < hfPages.length) {
      setHfPageIndex(hfPageIndex + 1);
      return;
    }
    const cursor = hfPages[hfPageIndex]?.nextCursor;
    if (!cursor) return;
    setSearching(true);
    setHfMsg("");
    try {
      const { results, nextCursor } = await api.searchHf({
        q: hfQuery,
        sort: HF_SORT_SERVER_FIELD[hfSortField],
        direction: hfSortField === "relevance" ? undefined : hfSortDir === "asc" ? 1 : -1,
        cursor,
      });
      setHfPages((pages) => [...pages, { results, nextCursor }]);
      setHfPageIndex(hfPageIndex + 1);
    } catch (err) {
      setHfMsg(`Search error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSearching(false);
    }
  }

  function goToPrevHfPage() {
    setHfPageIndex((i) => Math.max(0, i - 1));
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
      // is the one place that ever clears `downloading` now. Also doubles as
      // "Resume" (Models page's active-downloads section below) -- calling
      // this again for the same worker/repo/file just re-queues a fresh
      // download_model job, which the worker's own .part-file detection
      // (worker/src/index.ts's executeDownloadModelJob) picks up and
      // continues rather than restarting from byte 0.
      const { job_id } = await api.downloadHfFile(workerId, repoId, file.path);
      setDownloading((d) => ({
        ...d,
        [downloadKey(workerId, repoId, file.path)]: { repoId, workerId, startedAt: Date.now(), file, jobId: job_id },
      }));
    } catch (err) {
      setHfMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handlePauseDownload(key: string) {
    const state = downloading[key];
    if (!state) return;
    try {
      await api.pauseDownload(state.workerId, state.jobId);
      setDownloading((d) => (d[key] ? { ...d, [key]: { ...d[key], paused: true } } : d));
    } catch (err) {
      setHfMsg(`Error pausing: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleResumeDownload(key: string) {
    const state = downloading[key];
    if (!state) return;
    // handleDownload overwrites this same key (downloadKey is deterministic
    // off worker/repo/path, not jobId) with a fresh, unpaused entry.
    await handleDownload(state.repoId, state.file);
  }

  async function handleCancelDownload(key: string) {
    const state = downloading[key];
    if (!state) return;
    const label = state.file.path;
    try {
      await api.cancelDownload(state.workerId, state.jobId);
      // Unlike a pause->finalize, there's genuinely nothing left (the
      // worker deletes the .part file too), so this skips finalizeDownload's
      // loadModels() calls -- nothing new could have been registered.
      removeDownloadState(key);
      setHfMsg(`${label} cancelled -- the partial download was deleted.`);
    } catch (err) {
      setHfMsg(`Error cancelling: ${err instanceof Error ? err.message : String(err)}`);
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
            {Object.entries(downloading).map(([key, state]) => {
              const prog = progress[key];
              const speed = speeds[key];
              const pct = prog?.total != null ? Math.min(100, Math.round((prog.bytes / prog.total) * 100)) : null;
              const totalBytes = prog?.total ?? state.file.size_bytes ?? null;
              return (
                <div
                  key={key}
                  className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <IconDownload width={14} height={14} className="text-accent" />
                      <span className="text-fg">{state.file.path}</span>
                      <span className="text-xs text-muted">
                        {state.repoId} · {workers.find((w) => w.id === state.workerId)?.displayName ?? state.workerId}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted">
                        {state.paused ? "Paused" : pct !== null ? `${pct}%` : "Starting…"}
                      </span>
                      {state.paused ? (
                        <button
                          type="button"
                          onClick={() => void handleResumeDownload(key)}
                          className="rounded-md border border-border px-2 py-0.5 text-xs text-fg hover:border-accent/40 hover:text-accent"
                        >
                          Resume
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => void handlePauseDownload(key)}
                          className="rounded-md border border-border px-2 py-0.5 text-xs text-muted hover:border-accent/40 hover:text-accent"
                        >
                          Pause
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          if (window.confirm(`Cancel downloading ${state.file.path}? The partial file will be deleted.`)) {
                            void handleCancelDownload(key);
                          }
                        }}
                        className="rounded-md border border-border px-2 py-0.5 text-xs text-muted hover:border-danger/40 hover:text-danger"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
                    {state.paused ? (
                      <div className="h-full w-full rounded-full bg-surface-raised" />
                    ) : pct !== null ? (
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
                    {!state.paused && speed !== undefined && <span>· {formatBytes(speed)}/s</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <details
        className="group mt-8 rounded-xl border border-border bg-surface"
        open={manualAddOpen}
        onToggle={(e) => {
          const open = e.currentTarget.open;
          setManualAddOpen(open);
          persistManualAddOpen(open);
        }}
      >
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
                    onChange={(e) => {
                      const field = e.target.value as HfSortField;
                      setHfSortField(field);
                      void runFreshHfSearch(field, hfSortDir);
                    }}
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
                      onClick={() => {
                        const dir: SortDir = hfSortDir === "asc" ? "desc" : "asc";
                        setHfSortDir(dir);
                        void runFreshHfSearch(hfSortField, dir);
                      }}
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
                          const dlKey = downloadKey(downloadWorker, r.id, f.path);
                          const dl = downloading[dlKey];
                          const prog = progress[dlKey];
                          const speed = speeds[dlKey];
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
          <div className="mt-3 flex items-center justify-center gap-3 text-sm">
            <button
              type="button"
              onClick={goToPrevHfPage}
              disabled={hfPageIndex === 0 || searching}
              className="rounded-lg border border-border px-3 py-1.5 text-muted hover:border-accent/40 hover:text-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-muted"
            >
              ← Prev
            </button>
            <span className="text-muted">Page {hfPageIndex + 1}</span>
            <button
              type="button"
              onClick={() => void goToNextHfPage()}
              disabled={searching || (hfPageIndex + 1 >= hfPages.length && !hfPages[hfPageIndex]?.nextCursor)}
              className="rounded-lg border border-border px-3 py-1.5 text-muted hover:border-accent/40 hover:text-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-muted"
            >
              Next →
            </button>
          </div>
          </>
        )}
        {hfMsg && <p className="mt-3 text-sm text-muted">{hfMsg}</p>}
      </section>
    </div>
  );
}
