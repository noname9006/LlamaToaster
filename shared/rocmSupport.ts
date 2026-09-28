// Which AMD GPUs ROCm officially supports, and the matching logic that turns a
// detected GPU name into a yes/no. Consumed by detectBackend (shared/types.ts)
// on the worker at startup and in the client's backend-mismatch check.
//
// The list itself is refreshed daily by the server from AMD's own docs (see
// server/src/rocm-support.ts) and served at GET /api/rocm-support. Everything
// here is pure and has no I/O, so worker, client and server share one matcher.
//
// Support is tracked per platform because AMD publishes it per platform and the
// two genuinely differ (e.g. RDNA2 cards are unsupported on Windows but the
// Radeon PRO W6800 is supported on Linux).

export interface RocmPlatformSupport {
  // Normalized GPU names (see normalizeGpuName) AMD marks supported / not
  // supported. A marker matches a detected GPU when its tokens appear as a
  // contiguous run in that GPU's normalized name -- so "rx 7900 xt" matches
  // "rx 7900 xt" but not "rx 7900 xtx" (a different card with its own row).
  supported: string[];
  unsupported: string[];
}

export interface RocmSupportList {
  win32: RocmPlatformSupport;
  linux: RocmPlatformSupport;
}

// What GET /api/rocm-support returns. `source` is "builtin" until the server
// has fetched AMD's pages at least once (or whenever a fetch never succeeded),
// in which case fetchedAt is null.
export interface RocmSupportSnapshot {
  list: RocmSupportList;
  source: "live" | "builtin";
  fetchedAt: string | null;
}

// Fallback used whenever no fetched list is available (server not yet
// refreshed, offline, or a worker that can't reach the server). A hand-copied
// snapshot of AMD's docs from Aug 2026 -- the live list supersedes it, so this
// only has to be reasonable, not current. The same union is used for both
// platforms, matching what this app did before the list was fetched.
const BUILTIN_SUPPORTED = [
  // RDNA3 / RDNA4 consumer (Radeon RX)
  "rx 7600", "rx 7650 gre", "rx 7700 xt", "rx 7800 xt", "rx 7900 xt", "rx 7900 xtx",
  "rx 9060", "rx 9070",
  // Radeon PRO / workstation
  "pro w6800", "pro w7700", "pro w7800", "pro w7900", "pro v620", "pro v710", "ai pro r9700", "ai pro r9600d",
  // Instinct (data center)
  "mi100", "mi210", "mi250", "mi250x", "mi300x", "mi300a", "mi325x", "mi350x", "mi355x",
];

export const BUILTIN_ROCM_SUPPORT: RocmSupportList = {
  win32: { supported: BUILTIN_SUPPORTED, unsupported: [] },
  linux: { supported: BUILTIN_SUPPORTED, unsupported: [] },
};

// Words that carry no model identity, dropped from both AMD's table names
// ("AMD Radeon RX 7900 XTX", "AMD RX 9070 GRE") and detected names
// ("AMD Radeon(TM) RX 7900 XTX") so the two forms compare equal.
const NOISE_TOKENS = new Set(["amd", "radeon", "instinct", "graphics", "advanced", "micro", "devices", "inc"]);

export function normalizeGpuName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((t) => t !== "" && !NOISE_TOKENS.has(t))
    // "(tm)" / "(r)" survive the punctuation strip as bare "tm" / "r" tokens.
    .filter((t) => t !== "tm" && t !== "r")
    .join(" ");
}

function containsRun(haystack: string[], needle: string[]): boolean {
  if (needle.length === 0 || needle.length > haystack.length) return false;
  for (let i = 0; i + needle.length <= haystack.length; i++) {
    let ok = true;
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
  }
  return false;
}

// The longest matching marker decides, so a specific row beats a shorter,
// more general one ("rx 6600 xt" unsupported beats a hypothetical "rx 6600"
// supported). Unmatched -> false: a GPU AMD doesn't list is treated as
// unsupported, same as AMD's own "not listed = not officially supported".
export function isRocmSupportedGpu(model: string, platform: string, list: RocmSupportList = BUILTIN_ROCM_SUPPORT): boolean {
  const tokens = normalizeGpuName(model).split(" ").filter(Boolean);
  if (tokens.length === 0) return false;
  const p = platform === "win32" ? list.win32 : list.linux;

  let best = { len: 0, chars: 0, supported: false };
  const consider = (markers: string[], supported: boolean) => {
    for (const m of markers) {
      const mt = m.split(" ").filter(Boolean);
      if (!containsRun(tokens, mt)) continue;
      const chars = m.length;
      if (mt.length > best.len || (mt.length === best.len && chars > best.chars)) {
        best = { len: mt.length, chars, supported };
      } else if (mt.length === best.len && chars === best.chars && supported) {
        // Identical marker in both lists: the supported claim wins.
        best = { len: mt.length, chars, supported: true };
      }
    }
  };
  consider(p.unsupported, false);
  consider(p.supported, true);
  return best.supported;
}

// GPUs AMD does not officially support but that the community reports running
// llama.cpp's ROCm/HIP builds fine, per platform (Windows uses AMD's HIP SDK,
// Linux uses ROCm proper, and the set of working cards differs). A curated list,
// not derived from AMD's docs -- it drives the informational notice on the
// Workers page and deliberately covers only these cards, not every unlisted
// AMD GPU.
export const COMMUNITY_ROCM_GPUS: { win32: string[]; linux: string[] } = {
  win32: ["rx 6600", "rx 6600 xt", "rx 6650 xt", "rx 6700", "rx 6700 xt", "rx 6750 xt"],
  linux: [
    "rx 6600", "rx 6600 xt", "rx 6650 xt", "rx 6700", "rx 6700 xt", "rx 6750 xt",
    "rx 6500 xt", "rx 5700", "rx 5700 xt",
  ],
};

// Whether a detected GPU name is one of `markers` (same contiguous-token match
// as isRocmSupportedGpu, so "rx 6700" also covers "AMD Radeon RX 6700 XT").
export function gpuNameInList(model: string, markers: string[]): boolean {
  const tokens = normalizeGpuName(model).split(" ").filter(Boolean);
  return markers.some((m) => containsRun(tokens, m.split(" ").filter(Boolean)));
}

// Cheap structural check for anything that crossed a trust boundary (HTTP
// response on the worker/client, JSON stored in the DB).
export function isRocmSupportList(v: unknown): v is RocmSupportList {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  const okPlatform = (p: unknown): boolean => {
    if (!p || typeof p !== "object") return false;
    const q = p as Record<string, unknown>;
    return (
      Array.isArray(q.supported) &&
      Array.isArray(q.unsupported) &&
      q.supported.every((s) => typeof s === "string") &&
      q.unsupported.every((s) => typeof s === "string")
    );
  };
  return okPlatform(o.win32) && okPlatform(o.linux);
}
