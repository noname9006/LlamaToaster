import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import { StatusPill } from "../components/StatusPill";
import {
  IconAlertTriangle,
  IconCheck,
  IconChevronDown,
  IconDash,
  IconDownload,
  IconPause,
  IconRefreshCw,
  IconX,
} from "../components/icons";
import { ParamsRangeSlider, PARAMS_STOPS, paramsInRange } from "../components/ParamsRangeSlider";
import { buildModelGroups, resolveQuant, groupQuantsByTier, type ModelGroup } from "../modelGrouping";
import {
  formatBytes,
  formatParamsB,
  formatRelativeTime,
  formatShortRelativeTime,
  hfFileUrl,
  hfRepoUrl,
  modelAuthor,
  modelFamily,
  modelParamsB,
  paramsBFromText,
} from "../utils";
import { isMtpDraftModel } from "../types";
import type {
  Model,
  ModelDirFile,
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

// Which worker the user has told THIS BROWSER is running on the same
// physical machine -- there's no way for a web page to detect that on its
// own (the browser and the worker process are unrelated to each other), so
// it's a one-time manual pairing per browser/device rather than anything
// derived from server data. Stored in localStorage (not sessionStorage) so
// it survives across visits on the same machine -- a laptop's browser
// should only need to be told once which worker is "this laptop".
const THIS_MACHINE_WORKER_KEY = "llamatoaster:this-machine-worker-id";

function loadThisMachineWorkerId(): string | null {
  try {
    return localStorage.getItem(THIS_MACHINE_WORKER_KEY);
  } catch {
    return null;
  }
}

function persistThisMachineWorkerId(workerId: string | null): void {
  try {
    if (workerId) localStorage.setItem(THIS_MACHINE_WORKER_KEY, workerId);
    else localStorage.removeItem(THIS_MACHINE_WORKER_KEY);
  } catch {
    /* localStorage unavailable -- pairing just won't survive a refresh */
  }
}

// Every worker's models section starts collapsed -- which ones the user has
// expanded is purely a per-browser display preference, remembered the same
// way THIS_MACHINE_WORKER_KEY is (localStorage, not sessionStorage) so it
// survives across visits on the same machine.
const OPEN_WORKER_SECTIONS_KEY = "llamatoaster:models-open-worker-ids";

function loadOpenWorkerIds(): string[] {
  try {
    const raw = localStorage.getItem(OPEN_WORKER_SECTIONS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === "string") : [];
  } catch {
    return [];
  }
}

function persistOpenWorkerIds(ids: string[]): void {
  try {
    localStorage.setItem(OPEN_WORKER_SECTIONS_KEY, JSON.stringify(ids));
  } catch {
    /* localStorage unavailable -- open/collapsed state just won't survive a refresh */
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

// Display page size for the params-filtered results view -- matches HF's own
// raw per-request page size (server/src/hf.ts's "top 15"), just applied to
// the filtered list instead of the raw one. See filteredHfResults/
// visibleHfResults/goToNextHfPage below.
const HF_PAGE_SIZE = 15;

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

// Finds the ModelDirFile a worker actually reports for a given registered
// model. Path match first (the common case), then a SHA-256 fallback for
// hash-keyed models (id == the file's content hash, see deriveModelId/
// registerModel in server/src/db/repo.ts) -- mirrors server/src/routes/
// models.ts's /api/models/locations ownership check, which is how a worker
// ends up counted as an "owner" in the first place once the on-disk file has
// been manually renamed/moved (same content, different path). Without this
// fallback, callers that only matched by path would silently miss a renamed
// file's real entry -- see renderQuantBadge (used to show "no state
// reported") and handleDeleteModelFile (used to send a stale path that no
// longer exists on disk, making the delete job a silent no-op).
function findWorkerModelFile(m: Model, worker: Worker): ModelDirFile | undefined {
  const filename = m.source === "local" ? m.filename : m.hf_file;
  return (
    worker.modelFiles?.find((f) => f.path === filename) ??
    (/^[0-9a-f]{64}$/.test(m.id) ? worker.modelFiles?.find((f) => f.sha256?.toLowerCase() === m.id) : undefined)
  );
}

// Maps a worker-reported local model state (or its absence) to a small icon
// + tone for a compact quant badge -- same states worker/src's local-cache
// state machine defines (shared/types.ts's LocalModelState), just an icon
// glyph instead of a StatusPill/dot, since a badge only has room for one.
function badgeState(state: string | undefined): { Icon: typeof IconCheck; className: string; label: string } {
  switch (state) {
    case "verified":
      return { Icon: IconCheck, className: "text-success", label: "verified" };
    case "hashing":
      return { Icon: IconRefreshCw, className: "text-accent animate-spin", label: "hashing" };
    case "unknown":
    case "modified":
      return { Icon: IconDash, className: "text-warning", label: state };
    case "corrupted":
    case "missing":
      return { Icon: IconAlertTriangle, className: "text-danger", label: state };
    case "detected":
      return { Icon: IconDash, className: "text-muted", label: "detected" };
    default:
      return { Icon: IconDash, className: "text-muted/50", label: "no state reported" };
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
  const [thisMachineWorkerId, setThisMachineWorkerId] = useState<string | null>(loadThisMachineWorkerId);
  const [openWorkerIds, setOpenWorkerIds] = useState<string[]>(loadOpenWorkerIds);

  const [hfQuery, setHfQuery] = useState(() => loadPersistedHfSearch()?.query ?? "");
  // hfPages holds every raw HF search page fetched so far (each with its own
  // nextCursor), purely as a fetch cache -- it is NOT what's displayed.
  // hfPageIndex instead indexes into the FILTERED, flattened, re-chunked view
  // built below (filteredHfResults/visibleHfResults): "Prev" just moves the
  // index back (already-fetched raw pages are re-sliced, never refetched);
  // "Next" (goToNextHfPage) fetches more raw pages on demand, via HF's
  // forward-only opaque cursor (server/src/hf.ts), until the filtered buffer
  // covers the next display page or the cursor runs out.
  const [hfPages, setHfPages] = useState<HfPage[]>(() => loadPersistedHfSearch()?.pages ?? []);
  const [hfPageIndex, setHfPageIndex] = useState(() => loadPersistedHfSearch()?.pageIndex ?? 0);
  // Monotonic "search world" counter. Any change that invalidates already-fetched
  // HF pages -- a fresh query/sort (runFreshHfSearch) or a params-filter change
  // that re-chunks the filtered buffer -- bumps it. Every async fetch/fill path
  // captures it before awaiting and refuses to commit its (now stale) result
  // afterwards, so a slow in-flight pagination can never overwrite a newer
  // search's pages or advance the page index past what the current filter
  // actually matches (the "empty page" bug). See ensurePageFilled,
  // goToNextHfPage and runFreshHfSearch.
  const hfSearchEpochRef = useRef(0);
  const bumpHfSearchEpoch = () => {
    hfSearchEpochRef.current += 1;
  };
  // Count of user-triggered HF operations (fresh search / Next-page fill) still
  // in flight. Only the LAST one to finish clears the `searching` flag -- if a
  // superseded request drew its finally-run first, it mustn't re-enable the Next
  // button while a newer search is still placing its page-0: a Next clicked into
  // that gap would build on the half-replaced pages the epoch guard hasn't seen
  // yet. With Next kept disabled for the whole overlap, fills can only ever
  // start from a stable, current search world.
  const hfBusyRef = useRef(0);
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
  // True while a download job sits behind the worker's max_concurrent_downloads
  // cap (worker_jobs.status still 'pending', never claimed) -- set/cleared by
  // the poll effect below. Drives showing "Queued" with a static bar instead
  // of "Starting…" with the indeterminate animation, which used to show for
  // both cases identically even though "queued, worker hasn't touched it yet"
  // and "claimed, about to report its first byte" are different states.
  const [queuedKeys, setQueuedKeys] = useState<Record<string, boolean>>({});
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
  // below.
  const [locations, setLocations] = useState<Record<string, string[]>>({});
  const [, setUnreachableLocationWorkers] = useState<string[]>([]);

  const [authorFilter, setAuthorFilter] = useState("");
  const [familyFilter, setFamilyFilter] = useState("");
  const [paramsLoIndex, setParamsLoIndex] = useState(0);
  const [paramsHiIndex, setParamsHiIndex] = useState(PARAMS_STOPS.length - 1);
  const [sortField, setSortField] = useState<SortField>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [refreshingModels, setRefreshingModels] = useState(false);
  const [refreshModelsMsg, setRefreshModelsMsg] = useState("");

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

  // Sort is applied server-side now (server/src/hf.ts) -- params filtering
  // stays client-side (HF's search API has no equivalent). Pages are built
  // FROM the filtered results, not the other way around: every fetched raw
  // HF page (hfPages) is flattened and filtered first, then re-chunked into
  // display pages of HF_PAGE_SIZE -- so a display page the user actually
  // navigates to always holds only matches, never a raw page's worth of
  // results that then turn out empty once filtered. goToNextHfPage below is
  // what keeps this buffer topped up as the user pages forward.
  const filteredHfResults = useMemo(
    () => hfPages.flatMap((p) => p.results).filter((r) => paramsInRange(paramsBFromText(r.id), hfParamsLoIndex, hfParamsHiIndex)),
    [hfPages, hfParamsLoIndex, hfParamsHiIndex]
  );
  const visibleHfResults = useMemo(
    () => filteredHfResults.slice(hfPageIndex * HF_PAGE_SIZE, (hfPageIndex + 1) * HF_PAGE_SIZE),
    [filteredHfResults, hfPageIndex]
  );
  // Whether Next could land on *something* -- either the filtered buffer
  // already holds enough for another page, or there's an unfetched raw
  // cursor that might. Shared by the Next button's disabled state and the
  // "no results" message, so that message never tells the user to "try
  // Next" when Next is actually a dead end.
  const hasMoreHfPages =
    filteredHfResults.length > (hfPageIndex + 1) * HF_PAGE_SIZE || !!hfPages[hfPages.length - 1]?.nextCursor;
  // Highest index the currently-filtered buffer can actually back a full page
  // with -- a hard floor against structurally-empty pages. hfPageIndex only
  // ever advances to a position the fill logic proved non-empty, but stale
  // sessionStorage restores and the narrow window between a filter change and
  // an in-flight commit could still point past the data; clamping here makes an
  // empty display page impossible regardless of how the index got there.
  const maxHfPageIndex = Math.max(0, Math.ceil(filteredHfResults.length / HF_PAGE_SIZE) - 1);

  useEffect(() => {
    setHfPageIndex((i) => Math.min(i, maxHfPageIndex));
  }, [maxHfPageIndex]);

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

  async function handleRefreshModels() {
    if (!downloadWorker) {
      setRefreshModelsMsg("Please select a worker first");
      return;
    }
    setRefreshingModels(true);
    setRefreshModelsMsg("");
    try {
      const res = await api.refreshModels(downloadWorker);
      setRefreshModelsMsg(res.message);
      // Poll job status until refresh finishes (may take minutes for large files),
      // then reload models/locations. Fallback to fixed interval if job endpoint fails.
      const jobId = res.job_id;
      const workerId = downloadWorker;
      const maxPolls = 60; // ~60 * 2s = 2 minutes max
      let polls = 0;
      const poll = async () => {
        polls++;
        try {
          const job = await api.getJobStatus(workerId, jobId);
          if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
            await Promise.all([loadModels(), loadLocations()]);
            setRefreshingModels(false);
            setRefreshModelsMsg(
              job.status === "completed" ? "Refresh complete" : `Refresh ended with: ${job.status}`
            );
            return;
          }
        } catch {
          // Job not found or transient -- fall back to reloading anyway after a few polls
        }
        if (polls >= maxPolls) {
          await Promise.all([loadModels(), loadLocations()]);
          setRefreshingModels(false);
          setRefreshModelsMsg("Refresh polling timed out — reloaded available data");
          return;
        }
        setTimeout(poll, 2000);
      };
      setTimeout(poll, 2000);
    } catch (err) {
      setRefreshModelsMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
      setRefreshingModels(false);
    }
  }

  // Deletes the file from exactly one worker's model_dir -- the registry row
  // (and every other worker's copy) is left untouched, so this never hits
  // the "N run result(s) reference it" conflict the old whole-registry
  // delete could -- there's no DB data being removed here at all.
  async function handleDeleteModelFile(m: Model, worker: Worker) {
    // Prefer the worker's own reported path over the model's registered
    // filename -- for a file manually renamed/moved on disk after download,
    // those differ (see findWorkerModelFile), and sending the stale
    // registered filename here would have the worker look for a path that
    // no longer exists, silently no-opping the delete instead of removing
    // the file that's actually there.
    const filename = findWorkerModelFile(m, worker)?.path ?? (m.source === "local" ? m.filename : m.hf_file);
    if (!filename) return;
    if (
      !window.confirm(
        `Delete ${m.filename} from ${worker.displayName}? This removes it from that machine and from this list -- you can re-download or re-import it later.`
      )
    ) {
      return;
    }
    const key = `${m.id}:${worker.id}`;
    setDeletingKey(key);
    setModelsMsg("");
    try {
      const res = await api.deleteModelFileFromWorker(worker.id, filename);
      setModelsMsg(`Queued: delete ${m.filename} from ${worker.displayName}`);
      // Poll job status until the worker actually claims and finishes the
      // delete (same pattern as handleRefreshModels above) before refreshing
      // locations -- an immediate reload here used to race the job: the
      // worker hasn't deleted the file yet (it may not even have claimed the
      // job), so /api/models/locations still reported this worker as an
      // owner and the row never disappeared until an unrelated page reload
      // happened to land after the next heartbeat.
      const jobId = res.job_id;
      const workerId = worker.id;
      const maxPolls = 30; // ~30 * 2s = 1 minute max
      let polls = 0;
      const poll = async () => {
        polls++;
        try {
          const job = await api.getJobStatus(workerId, jobId);
          if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
            await Promise.all([loadModels(), loadLocations()]);
            setDeletingKey((k) => (k === key ? null : k));
            return;
          }
        } catch {
          // Job not found or transient -- fall back to reloading anyway after a few polls
        }
        if (polls >= maxPolls) {
          await Promise.all([loadModels(), loadLocations()]);
          setDeletingKey((k) => (k === key ? null : k));
          return;
        }
        setTimeout(poll, 2000);
      };
      setTimeout(poll, 2000);
    } catch (err) {
      setModelsMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
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
    persistOpenWorkerIds(openWorkerIds);
  }, [openWorkerIds]);

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
    setQueuedKeys((q) => {
      if (!(key in q)) return q;
      const next = { ...q };
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
          // Real progress data exists -- definitely not merely queued anymore
          // (covers a job that was pending a moment ago and just got claimed).
          setQueuedKeys((q) => (q[key] ? { ...q, [key]: false } : q));
          const bytes = active.bytes ?? 0;
          const total = active.total_bytes ?? null;
          setProgress((prev) => ({ ...prev, [key]: { bytes, total } }));
          const prevSample = prevSampleRef.current[key];
          const now = Date.now();
          // The server-side number only moves once per worker heartbeat
          // (worker/src/index.ts's HEARTBEAT_INTERVAL_MS/_ACTIVE_DOWNLOAD_MS),
          // which can still be slower than this 2s poll. Only recompute speed
          // when bytes actually advanced since the last real sample --
          // treating an unchanged poll as "0 B/s just now" (the old
          // behavior) fed a false near-zero reading into the EMA below on
          // every poll that re-read stale data, then a compensating spike
          // (delta measured over just the last ~2s) the moment fresh data
          // landed, which is exactly the "speed drops then jumps" jitter.
          // Leaving prevSample untouched on a no-change poll instead keeps
          // the window anchored to the last genuine reading, so the next
          // real delta is measured over its true elapsed time.
          if (prevSample && now > prevSample.time && bytes !== prevSample.bytes) {
            const instantBytesPerSec = Math.max(0, ((bytes - prevSample.bytes) / (now - prevSample.time)) * 1000);
            // EMA with alpha ~0.25 -- smooths sample-to-sample jitter without
            // lagging too far behind a real step change.
            const SPEED_SMOOTHING_ALPHA = 0.25;
            const prevSmoothed = smoothedSpeedRef.current[key];
            const smoothed =
              prevSmoothed == null
                ? instantBytesPerSec
                : prevSmoothed + SPEED_SMOOTHING_ALPHA * (instantBytesPerSec - prevSmoothed);
            smoothedSpeedRef.current[key] = smoothed;
            setSpeeds((s) => ({ ...s, [key]: smoothed }));
            prevSampleRef.current[key] = { bytes, time: now };
          } else if (!prevSample) {
            prevSampleRef.current[key] = { bytes, time: now };
          }
        } else if (seenRef.current[key]) {
          // Was actively downloading and just dropped out of activeDownloads.
          // A short grace period first, since the worker's heartbeat cadence
          // means "just changed" and "genuinely done" look identical for a
          // tick or two -- but past that grace period, confirm with the
          // server (same getDownloadStatus check the "never seen" branch
          // below always used) instead of trusting the miss count alone. A
          // job can legitimately vanish from activeDownloads while still
          // genuinely in progress -- e.g. its lease got reaped back to
          // 'pending' for retry (server/src/reaper.ts) while the worker
          // itself is still actually mid-transfer -- and blindly finalizing
          // after 3 misses used to drop a real in-progress download from the
          // UI (most visible right after a page reload, when seenRef has
          // just been freshly re-established and hasn't built up any slack).
          const misses = (missesRef.current[key] ?? 0) + 1;
          missesRef.current[key] = misses;
          const SEEN_GRACE_MISSES = 3;
          if (misses < SEEN_GRACE_MISSES) continue;
          try {
            const { status } = await api.getDownloadStatus(state.workerId, state.jobId);
            if (status === "pending" || status === "claimed") {
              missesRef.current[key] = 0; // confirmed still alive -- a future real gap gets its own fresh grace window
              continue;
            }
            finalizeDownload(key); // completed/failed/cancelled
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) {
              finalizeDownload(key); // job/worker genuinely gone
            }
            // otherwise transient (network hiccup, server restart) -- try again next tick, still missing
          }
        } else {
          // Never seen active yet -- could be queued behind the worker's
          // max_concurrent_downloads cap (a pending job never appears in
          // activeDownloads until claimed, and can legitimately wait there
          // for a long time) rather than actually gone. Ask the server which
          // it is instead of guessing from a timeout, so a merely-queued
          // download doesn't get wrongly finalized as "dropped".
          try {
            const { status } = await api.getDownloadStatus(state.workerId, state.jobId);
            if (status === "pending") {
              setQueuedKeys((q) => (q[key] ? q : { ...q, [key]: true }));
              continue;
            }
            if (status === "claimed") {
              // Claimed but still absent from activeDownloads (e.g. it hasn't
              // reported its first byte yet) -- no longer merely queued.
              setQueuedKeys((q) => (q[key] ? { ...q, [key]: false } : q));
              continue;
            }
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
    // This replaces the entire search world -- any ensurePageFilled/Next fill
    // still in flight belongs to the old one and must not commit afterwards.
    bumpHfSearchEpoch();
    const epoch = hfSearchEpochRef.current;
    hfBusyRef.current += 1;
    setHfMsg("");
    setExpandedRepo(null);
    setSearching(true);
    try {
      const { results, nextCursor } = await api.searchHf({
        q: hfQuery,
        sort: HF_SORT_SERVER_FIELD[field],
        direction: field === "relevance" ? undefined : dir === "asc" ? 1 : -1,
      });
      // A newer query/sort/filter took over while we were fetching -- don't
      // stomp on its pages with this stale result.
      if (hfSearchEpochRef.current !== epoch) return;
      setHfPages([{ results, nextCursor }]);
      setHfPageIndex(0);
      if (!results.length) setHfMsg("No repos found.");
    } catch (err) {
      if (hfSearchEpochRef.current !== epoch) return;
      setHfMsg(`Search error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      hfBusyRef.current -= 1;
      if (hfBusyRef.current <= 0) setSearching(false);
    }
  }

  // query may be empty (that's the default "top 15 by downloads" listing
  // shown before the user types anything). e is undefined when triggered
  // programmatically (on mount) rather than from the form's onSubmit.
  async function handleSearch(e?: FormEvent) {
    e?.preventDefault();
    await runFreshHfSearch(hfSortField, hfSortDir);
  }

  // Safety cap on how many raw HF pages one fill attempt will fetch through
  // -- a filter that matches almost nothing in the whole result set should
  // eventually give up rather than hammering HF's API in a loop.
  const MAX_AUTO_FETCH_PAGES = 8;

  // Fetches further raw HF pages (via the last page's cursor) until the
  // filtered buffer covers a FULL display page at targetIndex, or the cursor
  // runs out, or the fetch cap is hit. This is what actually "groups items
  // onto pages according to the filter": a display page is built FROM the
  // filtered set (filteredHfResults above), so as long as this keeps that
  // set topped up, a page can only ever be non-full at the true end of the
  // results, never because a raw HF page happened to filter down to nothing.
  //
  // `epoch` is the search world the caller started under; if it moves on
  // mid-loop (fresh query/sort, or a filter change), the remaining fetches are
  // abandoned -- they'd only be discarded by the caller's epoch guard anyway --
  // so no more HF API calls are spent on a result set nobody is viewing.
  //
  // Called from two places: goToNextHfPage (targeting the page it's about
  // to advance to) and the effect below (targeting whatever page is
  // *already* current -- covers landing on an under-filled page without any
  // Next click at all, e.g. right after a fresh search or a filter change,
  // which is exactly the "empty page" bug this whole redesign exists to fix).
  async function ensurePageFilled(
    targetIndex: number,
    epoch: number
  ): Promise<{ pages: HfPage[]; filteredCount: number; capped: boolean }> {
    let pages = hfPages;
    const filterFn = (r: HfRepoSearchResult) => paramsInRange(paramsBFromText(r.id), hfParamsLoIndex, hfParamsHiIndex);
    let filteredCount = pages.flatMap((p) => p.results).filter(filterFn).length;
    const needed = (targetIndex + 1) * HF_PAGE_SIZE;
    let attempts = 0;
    while (filteredCount < needed && pages[pages.length - 1]?.nextCursor && attempts < MAX_AUTO_FETCH_PAGES) {
      if (epoch !== hfSearchEpochRef.current) break;
      const cursor = pages[pages.length - 1].nextCursor!;
      const { results, nextCursor } = await api.searchHf({
        q: hfQuery,
        sort: HF_SORT_SERVER_FIELD[hfSortField],
        direction: hfSortField === "relevance" ? undefined : hfSortDir === "asc" ? 1 : -1,
        cursor,
      });
      pages = [...pages, { results, nextCursor }];
      filteredCount = pages.flatMap((p) => p.results).filter(filterFn).length;
      attempts++;
    }
    // True only when the MAX_AUTO_FETCH_PAGES cap stopped the loop with the
    // cursor still live (vs. the cursor genuinely running out) -- lets callers
    // distinguish "there is genuinely nothing further" from "didn't dig far
    // enough yet, keep clicking Next".
    const capped =
      attempts >= MAX_AUTO_FETCH_PAGES && filteredCount < needed && !!pages[pages.length - 1]?.nextCursor;
    return { pages, filteredCount, capped };
  }

  async function goToNextHfPage() {
    setHfMsg("");
    setSearching(true);
    hfBusyRef.current += 1;
    const fromIndex = hfPageIndex;
    const epoch = hfSearchEpochRef.current;
    try {
      const { pages, filteredCount, capped } = await ensurePageFilled(fromIndex + 1, epoch);
      // The query/sort/filter changed while the fill was fetching -- these
      // pages belong to an old search world. Drop them rather than overwriting
      // newer state (which could also advance the index past what the new,
      // narrower filter matches, landing on a fully empty page).
      if (hfSearchEpochRef.current !== epoch) return;
      if (pages !== hfPages) setHfPages(pages);
      if (filteredCount > (fromIndex + 1) * HF_PAGE_SIZE) {
        setHfPageIndex(fromIndex + 1);
      } else if (capped) {
        // Reached MAX_AUTO_FETCH_PAGES with the cursor still live -- more
        // matches are plausibly a fetch or two away, so don't declare the
        // result set exhausted.
        setHfMsg(
          "Haven't pulled enough matching results yet -- click Next to keep looking, or widen the filter above."
        );
      } else {
        setHfMsg("No further results match your parameter filter.");
      }
    } catch (err) {
      if (hfSearchEpochRef.current !== epoch) return;
      setHfMsg(`Search error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      hfBusyRef.current -= 1;
      if (hfBusyRef.current <= 0) setSearching(false);
    }
  }

  // Proactively tops up whatever page is *currently* shown -- without this,
  // only clicking Next ever filled a page, so the page you land on right
  // after a fresh search or a filter change (still index 0, or wherever a
  // sort/filter change reset to) could show a sparse or fully empty page
  // despite plenty of matches being just one more fetch away.
  useEffect(() => {
    let cancelled = false;
    const epoch = hfSearchEpochRef.current;
    void (async () => {
      try {
        const { pages } = await ensurePageFilled(hfPageIndex, epoch);
        // `cancelled` covers this effect being torn down by a newer run; the
        // epoch check additionally covers the search world changing without a
        // re-run of this effect (e.g. a fresh query landing via runFreshHfSearch).
        if (!cancelled && hfSearchEpochRef.current === epoch) {
          setHfPages((prev) => (pages === prev ? prev : pages));
        }
      } catch {
        // transient -- the user can still hit Next/Prev manually, which retries
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hfPageIndex, hfParamsLoIndex, hfParamsHiIndex, hfPages]);

  // No skip logic needed backward: filteredHfResults only ever grows (more
  // raw pages fetched, same filter), and Next above only ever advances
  // hfPageIndex to a position it already proved non-empty -- so re-slicing
  // an earlier window is always safe without touching the network.
  function goToPrevHfPage() {
    setHfMsg("");
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

  // Compact quant badge: state icon + quant code + size, nothing else --
  // matches Hugging Face's own quant-table style rather than a filename-first
  // row. The whole badge is the link to that exact file's HF page when one
  // exists; a manually-registered local file (no hf_repo, nothing to link to)
  // renders as a plain dashed-border span instead. Delete lives in a small
  // hover-revealed corner button, since a badge has no room for a persistent
  // one.
  function renderQuantBadge(m: Model, worker: Worker, isFirstInTier: boolean) {
    const key = `${m.id}:${worker.id}`;
    const workerModelFile = findWorkerModelFile(m, worker);
    const modelState = workerModelFile?.state;
    const hfMatch = workerModelFile?.hf_match;
    const quant = resolveQuant(m) ?? "?";
    // The worker's live hash-verified match takes priority over the model's
    // own stored hf_repo/hf_file (which can point at a since-renamed/moved
    // file the hash lookup would catch).
    const hfLinkRepo = hfMatch && !hfMatch.deleted ? hfMatch.repo_id : !hfMatch ? m.hf_repo : undefined;
    const hfLinkFile = hfMatch && !hfMatch.deleted ? hfMatch.filename : !hfMatch ? m.hf_file : undefined;
    // A confirmed-gone HF source overrides whatever the worker's own local
    // state says -- "the exact file HF served this hash from is gone" is the
    // more important thing to surface than "still verified locally".
    const { Icon: StateIcon, className: stateClassName, label: stateLabel } = hfMatch?.deleted
      ? { Icon: IconAlertTriangle, className: "text-warning", label: "no longer available on Hugging Face" }
      : badgeState(modelState);
    const title = `${m.filename} · ${formatBytes(m.size_bytes)} · ${stateLabel} · added ${formatShortRelativeTime(m.created_at)}`;

    const content = (
      <>
        <StateIcon width={11} height={11} className={`shrink-0 ${stateClassName}`} />
        <span className="font-mono text-xs font-semibold">{quant}</span>
        <span className="font-mono text-[11px] tabular-nums text-muted">{formatBytes(m.size_bytes)}</span>
      </>
    );
    const badgeClassName =
      hfLinkRepo && hfLinkFile
        ? "inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-1 text-xs text-fg no-underline transition-colors hover:border-accent/30 hover:bg-surface-raised hover:text-accent"
        : "inline-flex items-center gap-1.5 rounded-md border border-dashed border-border bg-surface px-2 py-1 text-xs text-fg";

    return (
      <div
        key={key}
        className={`group/badge relative inline-flex items-stretch ${isFirstInTier ? "" : "border-l border-border pl-2.5"}`}
      >
        {hfLinkRepo && hfLinkFile ? (
          <a href={hfFileUrl(hfLinkRepo, hfLinkFile)} target="_blank" rel="noreferrer" className={badgeClassName} title={title}>
            {content}
          </a>
        ) : (
          <span className={badgeClassName} title={`${title} · no Hugging Face page to link to`}>
            {content}
          </span>
        )}
        <button
          type="button"
          disabled={deletingKey === key}
          onClick={() => handleDeleteModelFile(m, worker)}
          className="absolute -right-1.5 -top-1.5 flex h-4 w-4 items-center justify-center rounded-full border border-bg bg-danger-bg text-danger opacity-0 transition-opacity group-hover/badge:opacity-100 disabled:opacity-50"
          aria-label={`Delete ${m.filename} from ${worker.displayName}`}
          title={`Delete this file from ${worker.displayName} -- you can re-download or re-import it later`}
        >
          <IconX width={9} height={9} />
        </button>
      </div>
    );
  }

  // One tier's worth of badges: a bit-width band (see modelGrouping.ts's
  // groupQuantsByTier) or the "MTP" band for draft/companion files, which
  // don't get bit-width-sorted alongside real quants.
  function renderTierRow(label: string, subtitle: string, quants: Model[], worker: Worker) {
    return (
      <div key={label} className="flex items-start gap-3.5 border-t border-border/60 px-4 py-2 first:border-t-0">
        <div className="w-14 shrink-0 pt-0.5 text-[11px] font-semibold text-muted">
          {label}
          <span className="block text-[9.5px] font-medium uppercase tracking-wide text-muted/60">{subtitle}</span>
        </div>
        <div className="flex flex-1 flex-wrap content-start gap-y-2">
          {quants.map((m, i) => renderQuantBadge(m, worker, i === 0))}
        </div>
      </div>
    );
  }

  function renderModelGroup(group: ModelGroup, worker: Worker) {
    // A companion MTP draft file that shares its hf_repo with more than one
    // quant in this group is attached to each of those quants' own
    // ModelQuantEntry.drafts (buildModelGroups) so ModelPicker.tsx can offer
    // it as a pairing for any of them -- but it's still just one physical
    // file on disk, so de-dupe by id before listing/counting it here. Without
    // this, a 2-quant group sharing 1 draft rendered that draft's row twice
    // (with a duplicate React key) and double-counted it in the aggregate.
    const seenDraftIds = new Set<string>();
    const uniqueDrafts = group.quants
      .flatMap((q) => q.drafts)
      .filter((d) => (seenDraftIds.has(d.id) ? false : (seenDraftIds.add(d.id), true)));
    const totalFiles = group.quants.length + uniqueDrafts.length;
    const totalBytes =
      group.quants.reduce((n, q) => n + q.base.size_bytes, 0) + uniqueDrafts.reduce((n, d) => n + d.size_bytes, 0);
    // Params is the same across every quant sibling (same underlying model,
    // just quantized differently) -- shown once here instead of repeating it
    // on every badge. The smallest quant (quants[0], after buildModelGroups'
    // own sort) stands in as the group's representative value, same
    // convention sortModelGroups already uses for author/family/size sorts --
    // its own param-count lookup/backfill button lives here too, for the
    // same reason.
    const repModel = group.quants[0].base;
    const groupParamsB = modelParamsB(repModel);
    const isEstimatedGroupParams = groupParamsB !== null && typeof repModel.metadata.param_count !== "number";
    const tiers = groupQuantsByTier(group.quants);

    return (
      // border-t (not just the individual rows' own hairline) is what
      // actually separates one group from the next -- without it here, a
      // group header sat flush against the previous group's last row with
      // nothing visually marking the boundary between them.
      <div key={`${group.key}:${worker.id}`} className="border-t-2 border-border/80 first:border-t-0">
        <div className="flex flex-wrap items-center gap-2 px-4 pt-3 pb-2">
          <span className="text-[13.5px] font-semibold text-fg">{group.label}</span>
          <span className="text-xs text-muted">· {group.author} · {group.family}</span>
          {groupParamsB !== null && (
            <span className="inline-flex items-center rounded border border-border bg-surface-raised px-1.5 py-0.5 font-mono text-xs font-bold text-fg">
              {formatParamsB(groupParamsB)}
            </span>
          )}
          {isEstimatedGroupParams && repModel.hf_repo && (
            <button
              type="button"
              disabled={paramLookupBusyId === repModel.id}
              onClick={() => lookupParamCount(repModel)}
              className="flex h-5 w-5 items-center justify-center rounded text-muted hover:bg-surface hover:text-accent disabled:opacity-50"
              title="Look up the real parameter count from Hugging Face"
              aria-label={`Look up real parameter count for ${group.label}`}
            >
              <IconRefreshCw width={12} height={12} className={paramLookupBusyId === repModel.id ? "animate-spin" : undefined} />
            </button>
          )}
          <span className="flex-1" />
          {repModel.hf_repo && (
            <a
              href={hfRepoUrl(repModel.hf_repo)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 rounded-md border border-transparent px-2 py-1 text-xs text-accent hover:border-accent/30 hover:bg-accent/10"
            >
              View on HF ↗
            </a>
          )}
          <span className="font-mono text-[10.5px] text-muted/70">
            {totalFiles} file{totalFiles === 1 ? "" : "s"} · {formatBytes(totalBytes)}
          </span>
        </div>
        {paramLookupErr[repModel.id] && <div className="px-4 pb-1.5 text-xs text-danger">{paramLookupErr[repModel.id]}</div>}
        {tiers.map((tier) =>
          renderTierRow(
            tier.label,
            `${tier.quants.length} quant${tier.quants.length === 1 ? "" : "s"}`,
            tier.quants.map((q) => q.base),
            worker
          )
        )}
        {uniqueDrafts.length > 0 &&
          renderTierRow("MTP", `${uniqueDrafts.length} file${uniqueDrafts.length === 1 ? "" : "s"}`, uniqueDrafts, worker)}
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
              {workers.length > 1 && (
                // Refresh Models below acts on whichever worker downloadWorker
                // points at -- with 2+ workers and no prior interaction with
                // the "Download to worker" picker further down the page (the
                // only other control for this same state), downloadWorker
                // stayed "" and the button silently did nothing. Surfacing the
                // picker right here makes the target explicit instead of
                // depending on an unrelated section of the page.
                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="text-muted">Worker to refresh</span>
                  <select
                    value={downloadWorker}
                    onChange={(e) => setDownloadWorker(e.target.value)}
                    className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                  >
                    <option value="">select…</option>
                    {workers.map((w) => (
                      <option key={w.id} value={w.id}>
                        {w.displayName}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <button
                type="button"
                onClick={handleRefreshModels}
                disabled={refreshingModels}
                className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-accent disabled:opacity-50"
              >
                {refreshingModels ? "Refreshing..." : "Refresh Models"}
              </button>
            </div>

            {workers.length > 1 && (
              <label className="mt-3 flex flex-wrap items-center gap-2 text-sm">
                <span className="text-muted">This machine is</span>
                <select
                  value={thisMachineWorkerId && workers.some((w) => w.id === thisMachineWorkerId) ? thisMachineWorkerId : ""}
                  onChange={(e) => {
                    const id = e.target.value || null;
                    setThisMachineWorkerId(id);
                    persistThisMachineWorkerId(id);
                  }}
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-fg outline-none focus:border-accent"
                >
                  <option value="">not set -- guess for me</option>
                  {workers.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.displayName}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <div className="mt-3 flex flex-col gap-3">
              {modelsByWorker.map(({ worker, subset, groups, orphans }) => (
                <details
                  key={worker.id}
                  open={openWorkerIds.includes(worker.id)}
                  onToggle={(e) => {
                    const isOpen = e.currentTarget.open;
                    setOpenWorkerIds((ids) => {
                      if (isOpen) return ids.includes(worker.id) ? ids : [...ids, worker.id];
                      return ids.includes(worker.id) ? ids.filter((id) => id !== worker.id) : ids;
                    });
                  }}
                  className="group rounded-xl border border-border bg-surface"
                >
                  <summary className="flex cursor-pointer items-center justify-between px-4 py-2.5 text-sm font-semibold text-fg">
                    <span className="flex items-center gap-2">
                      {worker.displayName}
                      {worker.id === thisMachineWorkerId && <StatusPill label="this machine" tone="accent" />}
                      <span className="text-xs font-normal text-muted">
                        ({subset.length} file{subset.length === 1 ? "" : "s"}
                        {subset.length > 0 ? ` · ${formatBytes(subset.reduce((n, m) => n + m.size_bytes, 0))}` : ""})
                      </span>
                    </span>
                    <IconChevronDown width={16} height={16} className="transition-transform group-open:rotate-180" />
                  </summary>
                  <div className="border-t border-border">
                    {subset.length === 0 ? (
                      <p className="px-4 py-3 text-sm text-muted">No models on this worker.</p>
                    ) : (
                      <>
                        {groups.map((group) => renderModelGroup(group, worker))}
                        {orphans.length > 0 && (
                          // Every entry here is an MTP draft whose paired base isn't
                          // in this worker's subset (orphanDrafts) -- most often
                          // because its hf_repo doesn't match any registered base's,
                          // so buildModelGroups never nested it under a group at all.
                          // Still gets its own labeled subsection rather than
                          // rendering as anonymous standalone badges indistinguishable
                          // from a normal quant file.
                          <div className="border-t-2 border-border/80">
                            <div className="px-4 pt-3 pb-1 text-[11px] text-muted">
                              <span className="font-semibold text-fg">MTP companions</span>
                              <span className="text-muted/70"> — no matching base model on this worker</span>
                            </div>
                            {renderTierRow("MTP", `${orphans.length} file${orphans.length === 1 ? "" : "s"}`, orphans, worker)}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </details>
              ))}
            </div>
            {modelsMsg && <p className="mt-3 text-sm text-muted">{modelsMsg}</p>}
            {refreshModelsMsg && <p className="mt-3 text-sm text-muted">{refreshModelsMsg}</p>}
          </>
        )}
      </section>

      {Object.keys(downloading).length > 0 && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Active downloads</h2>
          <div className="mt-3 flex flex-col overflow-hidden rounded-lg border border-border bg-surface">
            {Object.entries(downloading).map(([key, state], i) => {
              const prog = progress[key];
              const speed = speeds[key];
              const isQueued = !!queuedKeys[key];
              const pct = prog?.total != null ? Math.min(100, Math.round((prog.bytes / prog.total) * 100)) : null;
              const totalBytes = prog?.total ?? state.file.size_bytes ?? null;
              const statLabel = state.paused
                ? "Paused"
                : isQueued
                  ? "Queued"
                  : pct !== null
                    ? `${pct}% · ${formatBytes(prog?.bytes ?? 0)}${totalBytes != null ? ` / ${formatBytes(totalBytes)}` : ""}`
                    : "Starting…";
              return (
                <div
                  key={key}
                  className={`grid grid-cols-[minmax(200px,1fr)_auto_auto_auto] items-center gap-x-4 px-4 py-2 ${i > 0 ? "border-t border-border" : ""}`}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <IconDownload width={13} height={13} className="shrink-0 text-accent" />
                      <span className="truncate text-[13px] text-fg">{state.file.path}</span>
                      <span className="shrink-0 truncate text-[11px] text-muted">
                        {state.repoId} → {workers.find((w) => w.id === state.workerId)?.displayName ?? state.workerId}
                      </span>
                    </div>
                    <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-surface-raised">
                      {state.paused || isQueued ? (
                        <div className="h-full w-full rounded-full bg-surface-raised" />
                      ) : pct !== null ? (
                        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
                      ) : (
                        <div className="progress-indeterminate h-full w-full rounded-full" />
                      )}
                    </div>
                  </div>
                  <div className="whitespace-nowrap font-mono text-xs tabular-nums text-muted">{statLabel}</div>
                  <div className="whitespace-nowrap font-mono text-xs tabular-nums text-muted/70">
                    {!state.paused && speed !== undefined ? `${formatBytes(speed)}/s` : ""}
                  </div>
                  <div className="flex items-center justify-end gap-1">
                    {state.paused ? (
                      <button
                        type="button"
                        onClick={() => void handleResumeDownload(key)}
                        className="whitespace-nowrap rounded border border-border px-1.5 py-1 text-[11px] text-fg hover:border-accent/40 hover:text-accent"
                      >
                        Resume
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void handlePauseDownload(key)}
                        className="flex h-6 w-6 items-center justify-center rounded text-muted hover:bg-surface hover:text-accent"
                        title="Pause"
                        aria-label={`Pause downloading ${state.file.path}`}
                      >
                        <IconPause width={13} height={13} />
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => {
                        if (window.confirm(`Cancel downloading ${state.file.path}? The partial file will be deleted.`)) {
                          void handleCancelDownload(key);
                        }
                      }}
                      className="flex h-6 w-6 items-center justify-center rounded text-muted hover:bg-danger-bg hover:text-danger"
                      title="Cancel"
                      aria-label={`Cancel downloading ${state.file.path}`}
                    >
                      <IconX width={13} height={13} />
                    </button>
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

        {hfPages.length > 0 && (
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
                    // A page index chosen under the old filter can point past
                    // what the new, narrower filter leaves in the buffer --
                    // back to page 1, same as changing sort already does.
                    setHfPageIndex(0);
                    // Bump the epoch too so any fill/advance that was mid-flight
                    // under the previous filter can't commit stale pages or
                    // re-advance the index afterwards.
                    bumpHfSearchEpoch();
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
                    setHfPageIndex(0);
                    bumpHfSearchEpoch();
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-accent"
                >
                  Clear filters
                </button>
              )}
            </div>
          {/* Since pages are now built FROM the filtered results (see filteredHfResults above),
              this can now only be empty right at the start -- nothing fetched yet has matched.
              hfMsg covers the "Next looked and found nothing further" case with a more precise
              message, so this is skipped whenever hfMsg already has something more specific up. */}
          {visibleHfResults.length === 0 && !hfMsg && (
            <p className="mt-4 text-sm text-muted">
              No results match your parameter filter{hasMoreHfPages ? " yet -- try Next, or widen the filter above." : " -- widen the filter above."}
            </p>
          )}
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
              disabled={searching || !hasMoreHfPages}
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
