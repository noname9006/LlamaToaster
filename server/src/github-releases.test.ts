import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

// Real b10612 asset names/shape (trimmed) -- proves the cudart pairing against
// llama.cpp's actual naming convention rather than a made-up one.
const RELEASE_JSON = {
  tag_name: "b10612",
  published_at: "2026-08-20T00:00:00Z",
  draft: false,
  prerelease: true,
  assets: [
    {
      name: "cudart-llama-bin-win-cuda-12.4-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/cudart-llama-bin-win-cuda-12.4-x64.zip",
      size: 391_443_627,
    },
    {
      name: "cudart-llama-bin-win-cuda-13.3-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/cudart-llama-bin-win-cuda-13.3-x64.zip",
      size: 390_970_417,
    },
    {
      name: "llama-b10612-bin-win-cpu-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/llama-b10612-bin-win-cpu-x64.zip",
      size: 18_067_753,
    },
    {
      name: "llama-b10612-bin-win-cuda-12.4-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/llama-b10612-bin-win-cuda-12.4-x64.zip",
      size: 250_464_246,
    },
    {
      name: "llama-b10612-bin-win-cuda-13.3-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/llama-b10612-bin-win-cuda-13.3-x64.zip",
      size: 146_446_415,
    },
    {
      name: "llama-b10612-bin-win-vulkan-x64.zip",
      browser_download_url: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/llama-b10612-bin-win-vulkan-x64.zip",
      size: 34_403_266,
    },
  ],
};

beforeAll(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify([RELEASE_JSON]), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  );
});

afterAll(() => {
  vi.unstubAllGlobals();
});

describe("getReleases cudart pairing", () => {
  it("pairs each CUDA binary zip with its cudart redistributable and keeps both out of each other's way", async () => {
    const { getReleases } = await import("./github-releases.js");
    const releases = await getReleases();
    expect(releases).toHaveLength(1);
    const rel = releases[0]!;

    // Installable list never contains a cudart zip...
    expect(rel.assets.map((a) => a.name)).not.toContain("cudart-llama-bin-win-cuda-12.4-x64.zip");
    // ...but carries it as the cuda-12.4 build's runtime companion
    const cudart = rel.cudart_assets?.["llama-b10612-bin-win-cuda-12.4-x64.zip"];
    expect(cudart?.name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");
    expect(cudart?.size_bytes).toBe(391_443_627);
    expect(rel.cudart_assets?.["llama-b10612-bin-win-cuda-13.3-x64.zip"]?.name).toBe(
      "cudart-llama-bin-win-cuda-13.3-x64.zip"
    );
    // Non-CUDA builds have no companion
    expect(rel.cudart_assets?.["llama-b10612-bin-win-cpu-x64.zip"]).toBeUndefined();
    expect(rel.cudart_assets?.["llama-b10612-bin-win-vulkan-x64.zip"]).toBeUndefined();
  });

  it("buildInstallPayload attaches the cudart download only to CUDA builds", async () => {
    const { getReleases, buildInstallPayload } = await import("./github-releases.js");
    const rel = (await getReleases())[0]!;
    const cuda = rel.assets.find((a) => a.name.includes("cuda-12.4"))!;
    const cpu = rel.assets.find((a) => a.name.includes("cpu"))!;

    const cudaPayload = buildInstallPayload(rel, cuda);
    expect(cudaPayload.download_url).toBe(cuda.download_url);
    expect(cudaPayload.size_bytes).toBe(250_464_246);
    expect(cudaPayload.cudart_name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");
    expect(cudaPayload.cudart_url).toContain("/b10612/cudart-llama-bin-win-cuda-12.4-x64.zip");
    expect(cudaPayload.cudart_size_bytes).toBe(391_443_627);

    const cpuPayload = buildInstallPayload(rel, cpu);
    expect(cpuPayload.cudart_name).toBeUndefined();
    expect(cpuPayload.cudart_url).toBeUndefined();
  });
});

describe("filterReleasesForWorker driver-aware ordering", () => {
  it("orders cuda variants best-first for the worker's reported driver", async () => {
    const { getReleases, filterReleasesForWorker } = await import("./github-releases.js");
    const releases = await getReleases();

    // A CUDA-12.7-era driver can't load the cuda-13.x build -- the 12.4
    // variant must come first so install pickers' assets[0] is runnable.
    const forOldDriver = filterReleasesForWorker(releases, "win32", "x64", "cuda", "12.7")[0]!.assets;
    expect(forOldDriver[0]!.name).toContain("cuda-12.4");

    // A current driver gets the newest toolkit first.
    const forNewDriver = filterReleasesForWorker(releases, "win32", "x64", "cuda", "13.3")[0]!.assets;
    expect(forNewDriver[0]!.name).toContain("cuda-13.3");

    // No driver info at all -> conservative oldest-first default.
    const forUnknownDriver = filterReleasesForWorker(releases, "win32", "x64", "cuda", null)[0]!.assets;
    expect(forUnknownDriver[0]!.name).toContain("cuda-12.4");

    // Other backends are untouched by all of this.
    const vulkan = filterReleasesForWorker(releases, "win32", "x64", "vulkan", "12.7")[0]!.assets;
    expect(vulkan.map((a) => a.name)).toEqual(["llama-b10612-bin-win-vulkan-x64.zip"]);
  });
});

describe("filterOtherBuildsForWorker", () => {
  const a = (name: string) => ({ name, download_url: `https://x/${name}`, size_bytes: 1 });
  const newer = {
    tag: "b2",
    published_at: "2026-08-21T00:00:00Z",
    assets: [
      a("llama-b2-bin-win-cpu-x64.zip"),
      a("llama-b2-bin-win-cuda-12.4-x64.zip"),
      a("llama-b2-bin-win-vulkan-x64.zip"),
      a("llama-b2-bin-ubuntu-x64.tar.gz"),
      a("llama-b2-bin-win-cpu-arm64.zip"),
    ],
    cudart_assets: { "llama-b2-bin-win-cuda-12.4-x64.zip": a("cudart-llama-bin-win-cuda-12.4-x64.zip") },
  };
  const older = {
    tag: "b1",
    published_at: "2026-08-20T00:00:00Z",
    assets: [a("llama-b1-bin-win-vulkan-x64.zip"), a("llama-b1-bin-win-sycl-x64.zip")],
  };

  it("returns same-OS/arch builds of other backends, newest release per variant only", async () => {
    const { filterOtherBuildsForWorker } = await import("./github-releases.js");
    const other = filterOtherBuildsForWorker([newer, older], "win32", "x64", "cuda");
    const rows = other.flatMap((r) => r.assets.map((x) => x.name));
    // cuda (the worker's own backend), linux and arm64 are excluded; the older
    // vulkan zip is a duplicate variant of the newer one and is dropped.
    expect(rows).toEqual(["llama-b2-bin-win-cpu-x64.zip", "llama-b2-bin-win-vulkan-x64.zip", "llama-b1-bin-win-sycl-x64.zip"]);
  });

  it("carries cudart pairings for CUDA builds offered on a non-CUDA worker", async () => {
    const { filterOtherBuildsForWorker } = await import("./github-releases.js");
    const other = filterOtherBuildsForWorker([newer], "win32", "x64", "vulkan");
    expect(other[0]!.cudart_assets?.["llama-b2-bin-win-cuda-12.4-x64.zip"]?.name).toBe(
      "cudart-llama-bin-win-cuda-12.4-x64.zip"
    );
  });
});
