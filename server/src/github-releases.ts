import type { Backend, LlamaCppRelease } from "../../shared/types.js";
import { log } from "./log.js";

const GITHUB_REPO = "ggml-org/llama.cpp";
const RELEASES_URL = `https://api.github.com/repos/${GITHUB_REPO}/releases?per_page=15`;
const CACHE_TTL_MS = 5 * 60 * 1000;

interface GithubReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GithubReleaseApi {
  tag_name: string;
  published_at: string;
  draft: boolean;
  prerelease: boolean;
  assets: GithubReleaseAsset[];
}

let cache: { fetchedAt: number; releases: LlamaCppRelease[] } | null = null;

async function fetchReleases(): Promise<LlamaCppRelease[]> {
  const headers: Record<string, string> = {
    accept: "application/vnd.github+json",
    "user-agent": "llamatoaster-orchestrator",
  };
  if (process.env.GITHUB_TOKEN) {
    headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  const res = await fetch(RELEASES_URL, { headers });
  if (!res.ok) {
    throw new Error(`GitHub releases fetch failed: ${res.status} ${res.statusText}`);
  }
  const data = (await res.json()) as GithubReleaseApi[];
  return data
    // NOT filtering out prerelease here: verified live against the real repo
    // (checked 2026-08-23) that ggml-org/llama.cpp marks *every* one of its
    // per-commit build releases (the only kind it publishes -- there's no
    // separate "stable" GitHub release for this repo) as prerelease=true.
    // Excluding those excluded every release that exists, for every
    // platform/backend, which is exactly the "nothing available to install"
    // bug this comment is here to prevent regressing. Actual drafts (r.draft)
    // are still excluded -- those are genuinely unfinished/unpublished.
    .filter((r) => !r.draft)
    .map((r) => ({
      tag: r.tag_name,
      published_at: r.published_at,
      assets: r.assets
        .filter((a) => isBinaryAsset(a.name))
        .map((a) => ({
          name: a.name,
          download_url: a.browser_download_url,
          size_bytes: a.size,
        })),
    }));
}

// Verified against the actual ggml-org/llama.cpp release assets (checked
// live, not assumed): Windows and the xcframework ship as .zip, but
// Linux/macOS/Android ship as .tar.gz -- an earlier .zip-only filter here
// silently hid every Linux asset, including the ubuntu-x64 build this whole
// project needs for the CPU VPS worker. A handful of asset names are also
// not a llama-bench binary at all and would otherwise install "successfully"
// into a build directory with nothing runnable in it:
//   - `cudart-llama-bin-*` is the CUDA runtime redistributable, not llama.cpp
//   - `llama-*-ui.tar.gz` is the web UI bundle
//   - `llama-*-xcframework.zip` is an Apple embed framework, not a CLI binary
//   - android/s390x are platforms this app's Node-based worker doesn't
//     realistically ever run on, unlike openvino/opencl which are real
//     selectable backends now (see assetMatchesWorker's generic matching)
const NON_BINARY_ASSET_MARKERS = [
  "-ui.tar.gz",
  "xcframework",
  "android",
  "s390x",
];

function isBinaryAsset(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.startsWith("cudart-")) return false;
  if (!(lower.endsWith(".zip") || lower.endsWith(".tar.gz") || lower.endsWith(".tgz"))) {
    return false;
  }
  return !NON_BINARY_ASSET_MARKERS.some((m) => lower.includes(m));
}

// Checked every time a worker's llama-cpp info is requested (i.e. every time
// the Workers page is opened) but only actually hits GitHub once per
// CACHE_TTL_MS -- unauthenticated GitHub API calls are rate-limited to
// 60/hour, and there's no reason to spend that budget on repeat page loads
// within a few minutes of each other.
export async function getReleases(): Promise<LlamaCppRelease[]> {
  const now = Date.now();
  if (cache && now - cache.fetchedAt < CACHE_TTL_MS) {
    return cache.releases;
  }
  try {
    const releases = await fetchReleases();
    cache = { fetchedAt: now, releases };
    return releases;
  } catch (err) {
    if (cache) {
      log.error(
        `[github-releases] refresh failed, serving stale cache from ${new Date(cache.fetchedAt).toISOString()}:`,
        err
      );
      return cache.releases;
    }
    throw err;
  }
}

// Tokens actually seen in real llama.cpp release asset names for some kind
// of accelerated (non-plain-CPU) build -- used only to decide what counts as
// *not* the plain "cpu" bucket below. "hip" is included alongside "rocm"
// since older llama.cpp releases used the pre-rebrand "hipblas" naming;
// current ones say "rocm" (see the real asset list checked this session,
// e.g. llama-b10361-bin-win-rocm-7.14-x64.zip). "openvino" is its own
// distinct token, not a substring of "opencl" -- caught live: without it, a
// cpu-backend worker's install list wrongly included Ubuntu's openvino
// builds instead of its plain ubuntu-x64 one.
const KNOWN_NON_CPU_ASSET_TOKENS = ["vulkan", "cuda", "rocm", "hip", "sycl", "opencl", "openvino"];

// Real asset names embed the backend as a literal substring for every value
// except "opencl-adreno", whose actual release asset just says "opencl"
// (llama-*-bin-win-opencl-adreno-*.zip) -- everything else, known to this
// app or not, is matched by its own value directly.
function backendToken(backend: Backend): string {
  return backend === "opencl-adreno" ? "opencl" : backend.toLowerCase();
}

// Best-effort match against llama.cpp's actual release asset naming, which
// has changed before (AVX/AVX2/AVX512-tiered CPU builds existed once and
// don't anymore -- current releases ship exactly one generic CPU build per
// platform+arch) and will again -- this is a suggestion aid, not a
// guarantee. If it stops matching anything after a naming change, the UI
// falls back to showing the full unfiltered list rather than an empty one.
//
// Backend matching is deliberately generic (substring match on the backend
// value itself) rather than a hardcoded branch per known backend -- so an
// arbitrary backend string this app has never heard of (a future llama.cpp
// release variant, or just a typo) still does something sensible: it either
// matches real assets containing that literal word, or matches nothing and
// the UI falls back to the unfiltered list, same as the "naming changed"
// case above. See shared/types.ts's KNOWN_BACKENDS for the values this is
// actually expected to work well for.
export function assetMatchesWorker(
  assetName: string,
  platform: string,
  arch: string,
  backend: Backend
): boolean {
  const name = assetName.toLowerCase();
  const platformOk =
    platform === "win32"
      ? name.includes("win")
      : platform === "linux"
        ? name.includes("ubuntu") || name.includes("linux")
        : platform === "darwin"
          ? name.includes("macos") || name.includes("osx")
          : true;
  if (!platformOk) return false;

  // Node's os.arch() ("x64", "arm64", ...) already matches llama.cpp's own
  // asset naming convention directly. Only filter on arch tokens we
  // recognize in the name -- an asset that doesn't mention one isn't
  // penalized for it.
  if (name.includes("arm64") && arch !== "arm64") return false;
  if (name.includes("x64") && arch !== "x64") return false;

  if (backend === "cpu") {
    return !KNOWN_NON_CPU_ASSET_TOKENS.some((t) => name.includes(t));
  }
  return name.includes(backendToken(backend));
}

export function filterReleasesForWorker(
  releases: LlamaCppRelease[],
  platform: string,
  arch: string,
  backend: Backend
): LlamaCppRelease[] {
  return releases.map((r) => ({
    ...r,
    assets: r.assets.filter((a) => assetMatchesWorker(a.name, platform, arch, backend)),
  }));
}
