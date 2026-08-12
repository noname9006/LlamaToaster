import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
  chmodSync,
} from "node:fs";
import { join, relative } from "node:path";
import { platform as osPlatform } from "node:os";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
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

export async function installBuild(opts: {
  buildsDir: string;
  tag: string;
  assetName: string;
  downloadUrl: string;
}): Promise<InstalledBuildInfo> {
  validateTag(opts.tag);
  assertAllowedDownloadUrl(opts.downloadUrl);
  if (!existsSync(opts.buildsDir)) mkdirSync(opts.buildsDir, { recursive: true });
  const targetDir = join(opts.buildsDir, opts.tag);
  if (existsSync(targetDir)) {
    throw new Error(`build ${opts.tag} is already installed`);
  }

  const res = await fetch(opts.downloadUrl, {
    headers: { "user-agent": "llamatoaster-worker" },
    redirect: "follow",
  });
  if (!res.ok || !res.body) {
    throw new Error(`download failed: ${res.status} ${res.statusText}`);
  }

  const tmpArchive = join(
    opts.buildsDir,
    `.${opts.tag}.download${archiveExtension(opts.assetName)}`
  );
  await pipeline(Readable.fromWeb(res.body as any), createWriteStream(tmpArchive));

  try {
    mkdirSync(targetDir, { recursive: true });
    const extractStderr = await extractArchive(tmpArchive, targetDir);

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
        { tag: opts.tag, asset_name: opts.assetName, installed_at: installedAt },
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
    };
  } catch (err) {
    rmSync(targetDir, { recursive: true, force: true });
    throw err;
  } finally {
    rmSync(tmpArchive, { force: true });
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
