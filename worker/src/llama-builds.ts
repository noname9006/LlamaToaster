import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
  chmodSync,
} from "node:fs";
import { join, relative } from "node:path";
import { platform as osPlatform } from "node:os";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { spawn } from "node:child_process";
import AdmZip from "adm-zip";

// Checked against the real ggml-org/llama.cpp releases: Windows and the
// xcframework ship as .zip, but Linux/macOS/Android ship as .tar.gz -- both
// need to be supported or the Linux (ubuntu-x64) asset this whole project
// needs for a CPU VPS worker can never actually be installed.
function archiveExtension(assetName: string): ".zip" | ".tar.gz" | ".tgz" {
  const lower = assetName.toLowerCase();
  if (lower.endsWith(".tar.gz")) return ".tar.gz";
  if (lower.endsWith(".tgz")) return ".tgz";
  return ".zip";
}

// Returns captured stderr (empty on a clean run) rather than throwing on a
// non-zero exit -- see the note above the tar branch for why.
function extractArchive(archivePath: string, destDir: string): Promise<string> {
  const ext = archiveExtension(archivePath);
  if (ext === ".zip") {
    new AdmZip(archivePath).extractAllTo(destDir, true);
    return Promise.resolve("");
  }
  // Shell out to the system `tar` rather than adding a JS gzip/tar
  // dependency -- bsdtar has shipped built into Windows 10/11 since 2018,
  // and every Linux/macOS box already has one. cwd into destDir and pass a
  // *relative* archive path with no `-C` at all: GNU tar (as shipped by
  // Git-for-Windows/MSYS, which is what ends up on PATH for a worker
  // started from a Git Bash shell) parses a path containing a colon not at
  // position 0 as [user@]host:path remote-archive syntax, which a Windows
  // absolute path (`C:\...`) matches by pure coincidence ("C" as a
  // hostname). Confirmed by testing this exact call on Windows: passing
  // `C:\...` as either the archive or the `-C` target failed with "Cannot
  // connect to C: resolve failed" / "Cannot open". A relative path never
  // contains that colon, so it's unambiguous on every tar/OS combination.
  return new Promise((resolvePromise, reject) => {
    const relArchive = relative(destDir, archivePath);
    const proc = spawn("tar", ["-xzf", relArchive], { cwd: destDir, windowsHide: true });
    let stderr = "";
    proc.stderr?.on("data", (d) => (stderr += d.toString()));
    proc.on("error", (err) =>
      reject(new Error(`failed to spawn tar (is it on PATH?): ${err.message}`))
    );
    proc.on("close", () => {
      // Deliberately not gating on tar's exit code: a real archive can
      // contain entries (e.g. Unix symlinks alongside .so files in Linux
      // release tarballs) some filesystems refuse to recreate even though
      // the actual llama-bench binary extracted fine -- confirmed hitting
      // exactly this extracting a real ubuntu-x64 tarball on an NTFS temp
      // dir without symlink privileges. The caller checks for that binary
      // directly right after this resolves; that's the check that actually
      // matters, not tar's own summary exit status.
      resolvePromise(stderr);
    });
  });
}

export interface InstalledBuildInfo {
  tag: string;
  asset_name: string;
  installed_at: number;
  bench_path: string;
  // llama-server ships as a sibling binary in the same release archive as
  // llama-bench, but is only needed for MTP benchmarking (see
  // worker/src/serverBench.ts) -- optional/fail-soft rather than required at
  // install time, since older archives or an unexpected platform layout
  // might lack it while llama-bench itself is still perfectly usable.
  // Unavailability is surfaced at run-trigger time instead (see
  // server/src/routes/runs.ts).
  server_path?: string;
  // Set when the matching cudart redistributable (CUDA runtime DLLs) was
  // downloaded and extracted alongside the binaries -- see installBuild's
  // opts.cudartUrl. Informational (surfaced on the Workers page); absence
  // just means the build wasn't a CUDA one, or its release shipped no cudart.
  cudart_name?: string;
}

// Live byte-level progress of an in-flight install, reported through the
// heartbeat's ActiveJobReport (phase "downloading"/"extracting") so the
// Workers page can render the same progress-bar/speed treatment the Models
// page gives model downloads. bytes/total_bytes are CUMULATIVE across every
// archive the install pulls (a CUDA install is two: the llama zip plus its
// ~390 MB cudart zip) so one bar covers the whole job.
export interface InstallProgress {
  phase: "downloading" | "extracting";
  bytes: number;
  total_bytes?: number;
  detail: string;
}

const MANIFEST_NAME = "manifest.json";
const BENCH_BASENAMES = new Set(["llama-bench", "llama-bench.exe"]);
const SERVER_BASENAMES = new Set(["llama-server", "llama-server.exe"]);

// llama.cpp tags look like "b1234" or "v0.1.2" -- alnum/./-/_ only, and never
// ".." anywhere, so a value that reaches installBuild/deleteBuild can't walk
// out of buildsDir when joined into a path (mirrors the containment check
// resolveModelPath already does for model files, applied here for builds).
const TAG_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

export function validateTag(tag: string): void {
  if (!TAG_PATTERN.test(tag) || tag.includes("..")) {
    throw new Error(`invalid build tag: ${tag}`);
  }
}

// The download URL is supposed to come from the orchestrator's cached GitHub
// API response, never straight from a browser -- but the worker is the
// process that will download, extract, and later exec whatever's at this
// URL, so it re-checks the host itself rather than trusting the caller.
const ALLOWED_DOWNLOAD_HOSTS = new Set([
  "github.com",
  "objects.githubusercontent.com",
  "codeload.github.com",
]);

function assertAllowedDownloadUrl(url: string): void {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`invalid download url: ${url}`);
  }
  if (parsed.protocol !== "https:" || !ALLOWED_DOWNLOAD_HOSTS.has(parsed.hostname)) {
    throw new Error(`download url host not allowed: ${parsed.hostname}`);
  }
}

export function detectPlatform(): string {
  return osPlatform(); // 'win32' | 'linux' | 'darwin'
}

function findBinary(dir: string, basenames: Set<string>): string | null {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findBinary(full, basenames);
      if (found) return found;
    } else if (basenames.has(entry.name)) {
      return full;
    }
  }
  return null;
}

function findBenchBinary(dir: string): string | null {
  return findBinary(dir, BENCH_BASENAMES);
}

function findServerBinary(dir: string): string | null {
  return findBinary(dir, SERVER_BASENAMES);
}

export function listInstalledBuilds(buildsDir: string): InstalledBuildInfo[] {
  if (!existsSync(buildsDir)) return [];
  const out: InstalledBuildInfo[] = [];
  for (const entry of readdirSync(buildsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const dir = join(buildsDir, entry.name);
    const manifestPath = join(dir, MANIFEST_NAME);
    if (!existsSync(manifestPath)) continue;
    try {
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
        tag: string;
        asset_name: string;
        installed_at: number;
        cudart_name?: string;
      };
      const benchPath = findBenchBinary(dir);
      if (!benchPath) continue;
      const serverPath = findServerBinary(dir);
      out.push({ ...manifest, bench_path: benchPath, server_path: serverPath ?? undefined });
    } catch {
      /* skip a corrupt/partial install rather than fail the whole listing */
    }
  }
  return out.sort((a, b) => b.installed_at - a.installed_at);
}

export function getInstalledBuild(
  buildsDir: string,
  tag: string
): InstalledBuildInfo | null {
  return listInstalledBuilds(buildsDir).find((b) => b.tag === tag) ?? null;
}

// Streams one archive to dest, counting bytes per chunk. Returns the byte
// count written (and the server-claimed total when the response had a
// content-length). No retry/resume here on purpose: build archives are
// fetched from GitHub's CDN which is fast and reliable, and an install that
// dies mid-transfer is simply re-run from the Workers page -- unlike model
// downloads there's no catalog row waiting on it.
async function downloadArchive(
  url: string,
  dest: string,
  label: string,
  expectedSize: number | undefined,
  signal: AbortSignal | undefined,
  onChunk: (fileBytes: number, fileTotal: number | undefined) => void
): Promise<void> {
  const res = await fetch(url, {
    headers: { "user-agent": "llamatoaster-worker" },
    redirect: "follow",
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`download failed (${label}): ${res.status} ${res.statusText}`);
  }
  const contentLength = Number(res.headers.get("content-length"));
  // Header total wins when present (GitHub always sends one for release
  // assets); the payload's GitHub-reported size is the fallback so the very
  // first chunks already have a denominator for the progress bar.
  const fileTotal =
    Number.isFinite(contentLength) && contentLength > 0 ? contentLength : expectedSize;
  let bytes = 0;
  const tracker = new Transform({
    transform(chunk: Buffer, _enc, cb) {
      bytes += chunk.length;
      onChunk(bytes, fileTotal);
      cb(null, chunk);
    },
  });
  await pipeline(Readable.fromWeb(res.body as any), tracker, createWriteStream(dest));
}

export async function installBuild(opts: {
  buildsDir: string;
  tag: string;
  assetName: string;
  downloadUrl: string;
  // GitHub-reported size of the main archive (optional -- only used to show
  // a total before the response header arrives).
  sizeBytes?: number;
  // The matching cudart redistributable for CUDA builds (resolved
  // server-side from its cached GitHub data; re-validated against this
  // file's host allowlist below like any other URL). Extracted into the
  // same directory as the main archive so llama-bench.exe finds the DLLs
  // at load time. A CUDA install without these DLLs doesn't gracefully fall
  // back to CPU -- the binary fails with missing-DLL errors -- so a cudart
  // download/extract failure fails the whole install rather than leaving a
  // build behind that can never run.
  cudartName?: string;
  cudartUrl?: string;
  cudartSizeBytes?: number;
  onProgress?: (p: InstallProgress) => void;
  signal?: AbortSignal;
}): Promise<InstalledBuildInfo> {
  validateTag(opts.tag);
  assertAllowedDownloadUrl(opts.downloadUrl);
  if (opts.cudartUrl) assertAllowedDownloadUrl(opts.cudartUrl);
  if (!existsSync(opts.buildsDir)) mkdirSync(opts.buildsDir, { recursive: true });
  const targetDir = join(opts.buildsDir, opts.tag);
  if (existsSync(targetDir)) {
    throw new Error(`build ${opts.tag} is already installed`);
  }

  const report = opts.onProgress ?? (() => {});
  // Cumulative accounting across both archives so ONE bar covers the whole
  // job: totals come from the payload's GitHub-reported sizes until each
  // response's own content-length confirms them.
  let finishedBytes = 0; // bytes of archives fully downloaded
  let mainTotal: number | undefined = opts.sizeBytes;
  let cudartTotal: number | undefined = opts.cudartSizeBytes;
  const hasCudart = Boolean(opts.cudartUrl);
  const grandTotal = (): number | undefined => {
    if (mainTotal === undefined) return undefined;
    if (hasCudart && cudartTotal === undefined) return undefined;
    return mainTotal + (cudartTotal ?? 0);
  };

  const tmpArchive = join(
    opts.buildsDir,
    `.${opts.tag}.download${archiveExtension(opts.assetName)}`
  );
  const tmpCudartArchive = opts.cudartName
    ? join(opts.buildsDir, `.${opts.tag}.download-cudart${archiveExtension(opts.cudartName)}`)
    : null;

  try {
    report({ phase: "downloading", bytes: 0, total_bytes: grandTotal(), detail: opts.assetName });
    await downloadArchive(
      opts.downloadUrl,
      tmpArchive,
      opts.assetName,
      opts.sizeBytes,
      opts.signal,
      (fileBytes, fileTotal) => {
        if (fileTotal !== undefined) mainTotal = fileTotal;
        report({
          phase: "downloading",
          bytes: finishedBytes + fileBytes,
          total_bytes: grandTotal(),
          detail: opts.assetName,
        });
      }
    );
    finishedBytes += mainTotal ?? 0;

    if (opts.cudartUrl && tmpCudartArchive && opts.cudartName) {
      await downloadArchive(
        opts.cudartUrl,
        tmpCudartArchive,
        opts.cudartName,
        opts.cudartSizeBytes,
        opts.signal,
        (fileBytes, fileTotal) => {
          if (fileTotal !== undefined) cudartTotal = fileTotal;
          report({
            phase: "downloading",
            bytes: finishedBytes + fileBytes,
            total_bytes: grandTotal(),
            detail: opts.cudartName!,
          });
        }
      );
      finishedBytes += cudartTotal ?? 0;
    }

    mkdirSync(targetDir, { recursive: true });
    report({
      phase: "extracting",
      bytes: grandTotal() ?? finishedBytes,
      detail: `extracting ${opts.assetName}`,
    });
    const extractStderr = await extractArchive(tmpArchive, targetDir);

    if (tmpCudartArchive && existsSync(tmpCudartArchive)) {
      report({
        phase: "extracting",
        bytes: grandTotal() ?? finishedBytes,
        detail: `extracting ${opts.cudartName}`,
      });
      await extractArchive(tmpCudartArchive, targetDir);
    }

    const benchPath = findBenchBinary(targetDir);
    if (!benchPath) {
      throw new Error(
        `downloaded archive ${opts.assetName} did not contain a llama-bench binary` +
          (extractStderr ? ` (extractor output: ${extractStderr.slice(0, 500)})` : "")
      );
    }
    const serverPath = findServerBinary(targetDir);
    if (osPlatform() !== "win32") {
      try {
        chmodSync(benchPath, 0o755);
        if (serverPath) chmodSync(serverPath, 0o755);
      } catch {
        /* best effort -- extraction usually preserves the exec bit already */
      }
    }

    const installedAt = Date.now();
    writeFileSync(
      join(targetDir, MANIFEST_NAME),
      JSON.stringify(
        {
          tag: opts.tag,
          asset_name: opts.assetName,
          installed_at: installedAt,
          ...(opts.cudartName ? { cudart_name: opts.cudartName } : {}),
        },
        null,
        2
      ),
      "utf8"
    );
    return {
      tag: opts.tag,
      asset_name: opts.assetName,
      installed_at: installedAt,
      bench_path: benchPath,
      server_path: serverPath ?? undefined,
      ...(opts.cudartName ? { cudart_name: opts.cudartName } : {}),
    };
  } catch (err) {
    rmSync(targetDir, { recursive: true, force: true });
    throw err;
  } finally {
    rmSync(tmpArchive, { force: true });
    if (tmpCudartArchive) rmSync(tmpCudartArchive, { force: true });
  }
}

export function deleteBuild(buildsDir: string, tag: string): void {
  validateTag(tag);
  const dir = join(buildsDir, tag);
  if (!existsSync(dir)) {
    throw new Error(`build ${tag} is not installed`);
  }
  rmSync(dir, { recursive: true, force: true });
}

// Startup reconciliation for builds that arrived outside installBuild --
// someone unzips a llama.cpp release into llama-builds/<tag>/ by hand and
// restarts the worker. listInstalledBuilds requires a manifest.json, so such
// a build was invisible on the Workers page forever. This gives every
// subdirectory that actually contains a llama-bench binary a synthesized
// manifest (asset_name "(imported)", installed_at = dir mtime) so it shows
// up -- and is activatable/benchmarkable -- like any downloaded build.
// Dot-prefixed dirs (installBuild's own .<tag>.download temp files) are
// skipped, as are dirs whose name can't be a valid tag.
export function reconcileBuildsDir(buildsDir: string): InstalledBuildInfo[] {
  if (!existsSync(buildsDir)) return [];
  const imported: InstalledBuildInfo[] = [];
  for (const entry of readdirSync(buildsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    const dir = join(buildsDir, entry.name);
    if (existsSync(join(dir, MANIFEST_NAME))) continue; // already a real/known build
    try {
      validateTag(entry.name);
    } catch {
      continue;
    }
    const benchPath = findBenchBinary(dir);
    if (!benchPath) continue; // not a build (or still being unpacked by hand)
    try {
      const installedAt = statSync(dir).mtimeMs;
      writeFileSync(
        join(dir, MANIFEST_NAME),
        JSON.stringify(
          { tag: entry.name, asset_name: "(imported)", installed_at: installedAt },
          null,
          2
        ),
        "utf8"
      );
      imported.push({
        tag: entry.name,
        asset_name: "(imported)",
        installed_at: installedAt,
        bench_path: benchPath,
        server_path: findServerBinary(dir) ?? undefined,
      });
    } catch {
      // unwritable/partial dir -- skip rather than fail startup over it
    }
  }
  return imported;
}
