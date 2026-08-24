import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import AdmZip from "adm-zip";

// The real fetch is never allowed to fire here: installBuild only accepts
// https:// github.com URLs, so tests stub global.fetch to serve in-memory
// zips -- same shape the real GitHub CDN returns (200 + content-length).
const fetchedUrls: string[] = [];
const MAIN_ZIP = new AdmZip();
MAIN_ZIP.addFile("llama-bench.exe", Buffer.from("fake llama-bench binary"));
MAIN_ZIP.addFile("llama-server.exe", Buffer.from("fake llama-server binary"));
const CUDART_ZIP = new AdmZip();
CUDART_ZIP.addFile("cudart64_12.dll", Buffer.from("fake cudart dll"));
CUDART_ZIP.addFile("cublas64_12.dll", Buffer.from("fake cublas dll"));
const MAIN_BYTES = MAIN_ZIP.toBuffer().length;
const CUDART_BYTES = CUDART_ZIP.toBuffer().length;

function zipResponse(buf: Buffer): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(buf));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-length": String(buf.length) },
  });
}

beforeAll(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      fetchedUrls.push(url);
      // NB: check the dead link BEFORE the live cudart one -- includes()
      // would otherwise match both.
      if (url.endsWith("dead-cudart-link")) {
        return new Response("nope", { status: 404, statusText: "Not Found" });
      }
      if (url.includes("cudart")) return zipResponse(CUDART_ZIP.toBuffer());
      if (url.includes("llama-build")) return zipResponse(MAIN_ZIP.toBuffer());
      throw new Error(`unexpected url: ${url}`);
    })
  );
});

afterAll(() => {
  vi.unstubAllGlobals();
});

const tmpRoot = mkdtempSync(join(tmpdir(), "llamatoaster-builds-test-"));

afterAll(() => {
  try {
    rmSync(tmpRoot, { recursive: true, force: true });
  } catch {
    /* open handle on Windows -- harmless */
  }
});

async function importModule() {
  return await import("./llama-builds.js");
}

describe("installBuild with CUDA runtime pairing", () => {
  it("downloads and extracts the cudart archive alongside the binaries, recording it in the manifest", async () => {
    const { installBuild } = await importModule();
    const buildsDir = join(tmpRoot, "with-cudart");
    const phases: string[] = [];
    const details: string[] = [];
    let lastTotal: number | undefined;
    let lastBytes = 0;

    const installed = await installBuild({
      buildsDir,
      tag: "b10612",
      assetName: "llama-b10612-bin-win-cuda-12.4-x64.zip",
      downloadUrl: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/llama-build.zip",
      sizeBytes: 250_000,
      cudartName: "cudart-llama-bin-win-cuda-12.4-x64.zip",
      cudartUrl: "https://github.com/ggml-org/llama.cpp/releases/download/b10612/cudart.zip",
      cudartSizeBytes: 391_000,
      onProgress: (p) => {
        phases.push(p.phase);
        details.push(p.detail);
        if (typeof p.total_bytes === "number") lastTotal = p.total_bytes;
        if (typeof p.bytes === "number") lastBytes = Math.max(lastBytes, p.bytes);
      },
    });

    expect(installed.cudart_name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");
    expect(fetchedUrls).toHaveLength(2);

    const manifest = JSON.parse(readFileSync(join(buildsDir, "b10612", "manifest.json"), "utf8")) as Record<string, unknown>;
    expect(manifest.cudart_name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");

    // DLLs land NEXT TO llama-bench.exe so Windows' loader finds them
    expect(existsSync(join(buildsDir, "b10612", "cudart64_12.dll"))).toBe(true);
    expect(existsSync(join(buildsDir, "b10612", "cublas64_12.dll"))).toBe(true);
    expect(existsSync(join(buildsDir, "b10612", "llama-bench.exe"))).toBe(true);

    // Cumulative across BOTH archives -- one bar covers the whole job.
    // Totals come from the responses' content-length once headers arrive
    // (the payload sizes only seed the bar before that).
    expect(phases[0]).toBe("downloading");
    expect(phases).toContain("extracting");
    expect(lastTotal).toBe(MAIN_BYTES + CUDART_BYTES);
    expect(lastBytes).toBe(MAIN_BYTES + CUDART_BYTES);
    expect(details).toContain("cudart-llama-bin-win-cuda-12.4-x64.zip");
  });

  it("skips the cudart step entirely for non-CUDA builds", async () => {
    const { installBuild } = await importModule();
    fetchedUrls.length = 0;
    await installBuild({
      buildsDir: join(tmpRoot, "plain"),
      tag: "b10613",
      assetName: "llama-b10613-bin-win-cpu-x64.zip",
      downloadUrl: "https://github.com/ggml-org/llama.cpp/releases/download/b10613/llama-build.zip",
    });
    expect(fetchedUrls).toHaveLength(1);
  });

  it("fails the WHOLE install (and cleans up) when the cudart download dies -- a CUDA build without its DLLs can't run at all", async () => {
    const { installBuild } = await importModule();
    const buildsDir = join(tmpRoot, "broken");
    await expect(
      installBuild({
        buildsDir,
        tag: "b10614",
        assetName: "llama-b10614-bin-win-cuda-12.4-x64.zip",
        downloadUrl: "https://github.com/ggml-org/llama.cpp/releases/download/b10614/llama-build.zip",
        cudartName: "cudart-llama-bin-win-cuda-12.4-x64.zip",
        cudartUrl: "https://github.com/ggml-org/llama.cpp/releases/download/b10614/dead-cudart-link",
      })
    ).rejects.toThrow(/download failed/);
    expect(existsSync(join(buildsDir, "b10614"))).toBe(false);
  });
});

describe("reconcileBuildsDir (hand-dropped build adoption)", () => {
  it("gives every directory containing a llama-bench binary a synthesized manifest and lists it", async () => {
    const { reconcileBuildsDir, listInstalledBuilds } = await importModule();
    const buildsDir = join(tmpRoot, "dropped");
    mkdirSync(join(buildsDir, "b10000"), { recursive: true });
    writeFileSync(join(buildsDir, "b10000", "llama-bench"), "fake binary");
    // No manifest.json -- exactly what a hand-unzipped release looks like

    const imported = reconcileBuildsDir(buildsDir);
    expect(imported.map((b) => b.tag)).toEqual(["b10000"]);

    // Now a regular listing sees it (and a second reconcile is a no-op)
    expect(listInstalledBuilds(buildsDir).map((b) => b.tag)).toContain("b10000");
    expect(reconcileBuildsDir(buildsDir)).toHaveLength(0);
  });

  it("ignores dot-prefixed temp dirs, invalid tags, and dirs without a bench binary", async () => {
    const { reconcileBuildsDir } = await importModule();
    const buildsDir = join(tmpRoot, "ignored");
    mkdirSync(join(buildsDir, ".b10615.download"), { recursive: true });
    writeFileSync(join(buildsDir, ".b10615.download", "llama-bench"), "x");
    mkdirSync(join(buildsDir, "..\\escape"), { recursive: true });
    mkdirSync(join(buildsDir, "empty-dir"), { recursive: true });

    expect(reconcileBuildsDir(buildsDir)).toEqual([]);
    expect(existsSync(join(buildsDir, "..\\escape", "manifest.json"))).toBe(false);
  });
});
