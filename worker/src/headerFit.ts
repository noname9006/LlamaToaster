// A model llama-fit-params can size without the model: the real GGUF header
// (metadata + tensor list) of every shard, fetched from Hugging Face with
// range reads, written into a sparse file extended to the shard's full size.
// llama.cpp checks that every tensor lies inside the file but fit never reads
// weight data (meta allocations), so its answer equals the real file's --
// verified byte-for-byte on Qwen3.5-4B, Gemma-4 26B-A4B at 256k and a 3-shard
// GLM-5.3-Flash (docs/plans/HEADER_FIT_PLAN.md).
//
// The placeholder must never be taken for a model: it lives under the
// worker's own header-cache dir, never model_dir, and is removed after use.
// A volume that cannot hold sparse files (FAT/exFAT/HFS+) would really
// allocate the full size, so the space actually taken is checked right after
// the file is extended, and the attempt is abandoned there.

import { execFile } from "node:child_process";
import { mkdir, open, rm, stat, statfs } from "node:fs/promises";
import { basename, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const HEADER_FIRST_READ_BYTES = 4 * 1024 * 1024;
// Same ceiling as shared/gguf.ts's MAX_READ_BYTES: a header past this is not
// one we can size.
export const HEADER_MAX_BYTES = 64 * 1024 * 1024;
// Slack for filesystem metadata when judging whether the extend allocated.
const SPARSE_SLACK_BYTES = 64 * 1024 * 1024;

const GGUF_TYPE_SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const MAX_KV = 1_000_000;
const MAX_TENSORS = 10_000_000;
const MAX_DIMS = 8;

class NeedMore extends Error {}

/** Byte length of a GGUF header (metadata + tensor info, padded to the
 * file's alignment), "more" when `buf` ends inside it, null when it is not
 * a GGUF or its counts are implausible. */
export function ggufHeaderEnd(buf: Buffer): number | "more" | null {
  if (buf.length < 4) return "more";
  if (buf.toString("ascii", 0, 4) !== "GGUF") return null;
  let o = 4;
  const need = (n: number) => {
    if (o + n > buf.length) throw new NeedMore();
  };
  const u32 = () => {
    need(4);
    const v = buf.readUInt32LE(o);
    o += 4;
    return v;
  };
  const u64 = () => {
    need(8);
    const v = buf.readBigUInt64LE(o);
    o += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new RangeError("u64 out of range");
    return Number(v);
  };
  const skip = (n: number) => {
    need(n);
    o += n;
  };
  const str = () => skip(u64());
  try {
    u32(); // version
    const nTensors = u64();
    const nKv = u64();
    if (nTensors > MAX_TENSORS || nKv > MAX_KV) return null;
    let alignment = 32;
    for (let i = 0; i < nKv; i++) {
      const keyLen = u64();
      need(keyLen);
      const key = buf.toString("utf8", o, o + keyLen);
      o += keyLen;
      const t = u32();
      if (t === 8) str();
      else if (t === 9) {
        const et = u32();
        const n = u64();
        if (et === 8) for (let j = 0; j < n; j++) str();
        else {
          const sz = GGUF_TYPE_SIZE[et];
          if (sz == null) return null;
          skip(sz * n);
        }
      } else {
        const sz = GGUF_TYPE_SIZE[t];
        if (sz == null) return null;
        need(sz);
        if (key === "general.alignment" && t === 4) alignment = buf.readUInt32LE(o) || 32;
        o += sz;
      }
    }
    for (let i = 0; i < nTensors; i++) {
      str();
      const nd = u32();
      if (nd > MAX_DIMS) return null;
      skip(8 * nd + 4 + 8);
    }
    return Math.ceil(o / alignment) * alignment;
  } catch (err) {
    if (err instanceof NeedMore) return "more";
    return null;
  }
}

/** Every shard of a split GGUF ("name-00001-of-00003.gguf"), else the file. */
export function shardNames(file: string): string[] {
  const m = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i.exec(file);
  if (!m) return [file];
  const total = Number(m[3]);
  if (!(total >= 1 && total <= 1000)) return [file];
  return Array.from({ length: total }, (_, i) => `${m[1]}-${String(i + 1).padStart(5, "0")}-of-${m[3]}.gguf`);
}

export type FetchFn = (url: string, init: { headers: Record<string, string> }) => Promise<{
  status: number;
  headers: { get(name: string): string | null };
  arrayBuffer(): Promise<ArrayBuffer>;
}>;

export class HeaderFitError extends Error {
  constructor(
    message: string,
    readonly reason: "gated" | "not_found" | "not_gguf" | "too_large" | "http" | "not_sparse" | "io"
  ) {
    super(message);
  }
}

/** The header bytes of one remote GGUF and its full size, by range reads that
 * double until the header ends (never past HEADER_MAX_BYTES). */
export async function fetchGgufHeader(
  url: string,
  opts: { fetchFn?: FetchFn; headers?: Record<string, string> } = {}
): Promise<{ header: Buffer; size: number; fetchedBytes: number }> {
  const fetchFn = opts.fetchFn ?? (fetch as unknown as FetchFn);
  let want = HEADER_FIRST_READ_BYTES;
  let fetched = 0;
  for (;;) {
    const res = await fetchFn(url, { headers: { ...(opts.headers ?? {}), range: `bytes=0-${want - 1}` } });
    if (res.status === 401 || res.status === 403) throw new HeaderFitError(`${res.status} from ${url} (gated or private repo)`, "gated");
    if (res.status === 404) throw new HeaderFitError(`404 from ${url}`, "not_found");
    if (res.status !== 206 && res.status !== 200) throw new HeaderFitError(`${res.status} from ${url}`, "http");
    const body = Buffer.from(await res.arrayBuffer());
    fetched += body.length;
    // 206: "bytes 0-N/TOTAL"; 200: the whole (small) file.
    const range = res.headers.get("content-range");
    const total = range ? Number(/\/(\d+)\s*$/.exec(range)?.[1]) : res.status === 200 ? body.length : NaN;
    if (!Number.isFinite(total) || total <= 0) throw new HeaderFitError(`no size for ${url}`, "http");
    const end = ggufHeaderEnd(body);
    if (end === null) throw new HeaderFitError(`${basename(url)} is not a GGUF this reader can walk`, "not_gguf");
    if (end !== "more") return { header: body.subarray(0, Math.min(end, total)), size: total, fetchedBytes: fetched };
    if (body.length >= total) throw new HeaderFitError(`${basename(url)} ends inside its own header`, "not_gguf");
    if (want >= HEADER_MAX_BYTES) throw new HeaderFitError(`header of ${basename(url)} is over ${HEADER_MAX_BYTES >> 20} MiB`, "too_large");
    want = Math.min(want * 2, HEADER_MAX_BYTES);
  }
}

export type SetSparseFn = (path: string) => Promise<void>;

/** Windows: mark a file sparse before extending it, so the extend leaves a
 * hole instead of reserving the space. Elsewhere holes are the default. */
export const windowsSetSparse: SetSparseFn = async (path) => {
  await execFileAsync("fsutil", ["sparse", "setflag", path], { windowsHide: true, timeout: 15_000 });
};

/** Writes `header` and extends the file to `size` as a hole; throws (and
 * removes the file) when the volume really allocated the extension. */
export async function writeSparsePlaceholder(
  path: string,
  header: Buffer,
  size: number,
  opts: { platform?: NodeJS.Platform; setSparse?: SetSparseFn; freeBytes?: (dir: string) => Promise<number> } = {}
): Promise<void> {
  const platform = opts.platform ?? process.platform;
  const freeBytes = opts.freeBytes ?? (async (dir: string) => {
    const s = await statfs(dir);
    return Number(s.bavail) * Number(s.bsize);
  });
  const dir = join(path, "..");
  const fh = await open(path, "w");
  try {
    await fh.write(header, 0, header.length, 0);
  } finally {
    await fh.close();
  }
  try {
    if (platform === "win32") await (opts.setSparse ?? windowsSetSparse)(path);
    const before = await freeBytes(dir);
    const fh2 = await open(path, "r+");
    try {
      await fh2.truncate(size);
    } finally {
      await fh2.close();
    }
    const after = await freeBytes(dir);
    const st = await stat(path);
    const blocksBytes = typeof st.blocks === "number" && st.blocks > 0 ? st.blocks * 512 : null;
    const taken = Math.max(before - after, blocksBytes ?? 0);
    if (taken > header.length + SPARSE_SLACK_BYTES) {
      throw new HeaderFitError(`this volume does not keep sparse files (extending took ${Math.round(taken / 1048576)} MiB)`, "not_sparse");
    }
  } catch (err) {
    await rm(path, { force: true }).catch(() => {});
    if (err instanceof HeaderFitError) throw err;
    throw new HeaderFitError(`could not create the sparse placeholder: ${err instanceof Error ? err.message : String(err)}`, "not_sparse");
  }
}

export interface HeaderModel {
  /** Path of shard 1 (or the single file) -- what llama.cpp is given. */
  path: string;
  fetchedBytes: number;
  totalBytes: number;
  cleanup: () => Promise<void>;
}

/** Fetches every shard's header and lays the placeholders out under `dir`. */
export async function prepareHeaderModel(input: {
  dir: string;
  file: string;
  urlFor: (file: string) => string;
  headers?: Record<string, string>;
  fetchFn?: FetchFn;
  platform?: NodeJS.Platform;
  setSparse?: SetSparseFn;
  freeBytes?: (dir: string) => Promise<number>;
}): Promise<HeaderModel> {
  await rm(input.dir, { recursive: true, force: true });
  await mkdir(input.dir, { recursive: true });
  const cleanup = () => rm(input.dir, { recursive: true, force: true });
  try {
    let fetched = 0;
    let total = 0;
    const shards = shardNames(input.file);
    for (const shard of shards) {
      const h = await fetchGgufHeader(input.urlFor(shard), { fetchFn: input.fetchFn, headers: input.headers });
      fetched += h.fetchedBytes;
      total += h.size;
      await writeSparsePlaceholder(join(input.dir, basename(shard)), h.header, h.size, {
        platform: input.platform,
        setSparse: input.setSparse,
        freeBytes: input.freeBytes,
      });
    }
    return { path: join(input.dir, basename(shards[0])), fetchedBytes: fetched, totalBytes: total, cleanup };
  } catch (err) {
    await cleanup().catch(() => {});
    throw err;
  }
}
