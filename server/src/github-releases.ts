import type { Backend, InstallBuildJobPayload, LlamaCppAsset, LlamaCppRelease } from "../../shared/types.js";
import { sortAssetsForWorker } from "../../shared/types.js";
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
    .map((r) => {
      // First pass: bucket cudart zips by their "-bin-" suffix so the second
      // pass can pair each CUDA binary with its runtime by exact key lookup.
      const cudartBySuffix = new Map<string, LlamaCppAsset>();
      for (const a of r.assets) {
        if (!isCudartAsset(a.name)) continue;
        const suffix = assetBinSuffix(a.name);
        if (suffix) {
          cudartBySuffix.set(suffix, {
            name: a.name,
            download_url: a.browser_download_url,
            size_bytes: a.size,
          });
        }
      }
      const assets: LlamaCppAsset[] = [];
      const cudartAssets: Record<string, LlamaCppAsset> = {};
      for (const a of r.assets) {
        if (isCudartAsset(a.name)) continue; // handled above -- never an installable binary itself
        if (!isBinaryAsset(a.name)) continue;
        const mapped: LlamaCppAsset = {
          name: a.name,
          download_url: a.browser_download_url,
          size_bytes: a.size,
        };
        assets.push(mapped);
        const suffix = assetBinSuffix(a.name);
        const cudart = suffix ? cudartBySuffix.get(suffix) : undefined;
        // Keyed by the BINARY's name -- that's what the install flow starts
        // from ("install this cuda zip" -> "+ its cudart zip"), not vice versa.
        if (cudart) cudartAssets[a.name] = cudart;
      }
      return {
        tag: r.tag_name,
        published_at: r.published_at,
        assets,
        ...(Object.keys(cudartAssets).length > 0 ? { cudart_assets: cudartAssets } : {}),
      };
    });
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
  if (!(lower.endsWith(".zip") || lower.endsWith(".tar.gz") || lower.endsWith(".tgz"))) {
    return false;
  }
  return !NON_BINARY_ASSET_MARKERS.some((m) => lower.includes(m));
}

// cudart zips ("cudart-llama-bin-win-cuda-12.4-x64.zip") are classified by
// prefix, not by isBinaryAsset -- they'd otherwise pass the extension/marker
// filters and pollute the installable-assets lists with a ~390 MB non-llama
// download. They're kept (paired onto their binary, not dropped) because a
// Windows CUDA build is unusable without them: the DLLs inside must sit next
// to llama-bench.exe or it can't load its CUDA backend at all.
function isCudartAsset(name: string): boolean {
  return name.toLowerCase().startsWith("cudart-");
}

// The cudart naming convention (verified against real releases, e.g. b10612):
//   binary:  llama-<tag>-bin-win-cuda-12.4-x64.zip
//   cudart:  cudart-llama-bin-win-cuda-12.4-x64.zip
// i.e. the cudart name is the binary name with the tag dropped from after
// "llama-" -- so everything from "-bin-" on is identical between the pair,
// which is exactly what this extracts as the pairing key (null when the name
// has no "-bin-" at all).
function assetBinSuffix(name: string): string | null {
  const idx = name.toLowerCase().indexOf("-bin-");
  return idx < 0 ? null : name.slice(idx + "-bin-".length);
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
  backend: Backend,
  driverCudaVersion?: string | null
): LlamaCppRelease[] {
  return releases.map((r) => ({
    ...r,
    assets: sortAssetsForWorker(
      r.assets.filter((a) => assetMatchesWorker(a.name, platform, arch, backend)),
      driverCudaVersion
    ),
  }));
}

// The single place an installable asset becomes an install_build job payload,
// used by BOTH entry points that can queue an install (the Workers page's
// install route and runs.ts's auto-install-at-trigger) so a CUDA build always
// carries its cudart pairing along -- whichever path queued it.
export function buildInstallPayload(release: LlamaCppRelease, asset: LlamaCppAsset): InstallBuildJobPayload {
  const cudart = release.cudart_assets?.[asset.name];
  return {
    tag: release.tag,
    asset_name: asset.name,
    download_url: asset.download_url,
    ...(asset.size_bytes > 0 ? { size_bytes: asset.size_bytes } : {}),
    ...(cudart
      ? {
          cudart_name: cudart.name,
          cudart_url: cudart.download_url,
          cudart_size_bytes: cudart.size_bytes,
        }
      : {}),
  };
}
