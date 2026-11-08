// Reads a GGUF's header from Hugging Face with HTTP range requests, 2 MiB at a
// time, and turns it into the facts the fit simulation needs -- the same
// facts the LlamaToaster server stores per file (shared/gguf.ts), so the same
// header gives the same answer here as in the web UI.
//
// A split model ("name-00001-of-00003.gguf") keeps its metadata in shard 1 and
// a tensor list in every shard: every shard's header is read and the tensor
// sizes are summed over all of them. Only header bytes are downloaded, never
// weights: the read stops at the end of the tensor list.

import { buildTensorLayerBreakdown, parseGgufInfo } from "../../shared/gguf";
import type { KvLayout, TensorLayerBreakdown } from "../../shared/vramEstimate";

export const CHUNK_BYTES = 2 * 1024 * 1024;
// Same ceiling the worker uses (worker/src/headerFit.ts HEADER_MAX_BYTES).
const HEADER_MAX_BYTES = 64 * 1024 * 1024;

export interface ModelFacts {
  repo: string;
  file: string;
  shards: string[];
  /** Sum of every shard's size, bytes. */
  totalBytes: number;
  /** What reading the headers cost, bytes. */
  fetchedBytes: number;
  architecture: string | null;
  nLayer: number;
  trainedCtx: number | null;
  /** Routed experts and how many each token uses; null for a dense model. */
  expertCount: number | null;
  expertUsedCount: number | null;
  paramCount: number | null;
  quant: string | null;
  breakdown: TensorLayerBreakdown;
  layout: KvLayout;
}

export class HeaderError extends Error {
  constructor(message: string, readonly reason: "gated" | "not_found" | "not_gguf" | "too_large" | "http" | "unsupported") {
    super(message);
  }
}

const GGUF_TYPE_SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

class NeedMore extends Error {}

interface TensorWalk {
  /** Byte length of the header (metadata + tensor info, padded to alignment). */
  headerEnd: number;
  architecture: string | null;
  tensors: { name: string; offset: number }[];
}

/** Walks the header in `buf`; "more" when the buffer ends inside it. */
function walkHeader(buf: Uint8Array): TensorWalk | "more" | null {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const utf8 = new TextDecoder();
  let o = 0;
  const need = (n: number) => {
    if (o + n > buf.length) throw new NeedMore();
  };
  const u32 = () => {
    need(4);
    const v = view.getUint32(o, true);
    o += 4;
    return v;
  };
  const u64 = () => {
    need(8);
    const v = view.getBigUint64(o, true);
    o += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new RangeError("u64 out of range");
    return Number(v);
  };
  const skip = (n: number) => {
    need(n);
    o += n;
  };
  const str = () => {
    const len = u64();
    need(len);
    const s = utf8.decode(buf.subarray(o, o + len));
    o += len;
    return s;
  };
  try {
    if (u32() !== 0x46554747) return null; // "GGUF"
    const version = u32();
    if (version < 2) return null;
    const nTensors = u64();
    const nKv = u64();
    if (nTensors > 10_000_000 || nKv > 1_000_000) return null;
    let alignment = 32;
    let architecture: string | null = null;
    for (let i = 0; i < nKv; i++) {
      const key = str();
      const t = u32();
      if (t === 8) {
        const v = str();
        if (key === "general.architecture") architecture = v;
      } else if (t === 9) {
        const et = u32();
        const n = u64();
        if (et === 8) for (let j = 0; j < n; j++) skip(u64());
        else {
          const sz = GGUF_TYPE_SIZE[et];
          if (sz == null) return null;
          skip(sz * n);
        }
      } else {
        const sz = GGUF_TYPE_SIZE[t];
        if (sz == null) return null;
        need(sz);
        if (key === "general.alignment" && t === 4) alignment = view.getUint32(o, true) || 32;
        o += sz;
      }
    }
    const tensors: { name: string; offset: number }[] = [];
    for (let i = 0; i < nTensors; i++) {
      const name = str();
      const nd = u32();
      if (nd > 8) return null;
      skip(8 * nd + 4);
      tensors.push({ name, offset: u64() });
    }
    return { headerEnd: Math.ceil(o / alignment) * alignment, architecture, tensors };
  } catch (err) {
    if (err instanceof NeedMore) return "more";
    return null;
  }
}

/** Every shard of a split GGUF, else the file itself. */
export function shardNames(file: string): string[] {
  const m = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i.exec(file);
  if (!m) return [file];
  const total = Number(m[3]);
  if (!(total >= 1 && total <= 1000)) return [file];
  return Array.from({ length: total }, (_, i) => `${m[1]}-${String(i + 1).padStart(5, "0")}-of-${m[3]}.gguf`);
}

export function resolveUrl(repo: string, file: string, revision = "main"): string {
  const enc = (p: string) => p.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${enc(repo)}/resolve/${encodeURIComponent(revision)}/${enc(file)}`;
}

interface ShardHeader {
  bytes: Uint8Array;
  walk: TensorWalk;
  size: number;
  fetched: number;
}

/** Range-reads one shard 2 MiB at a time until its header ends. */
async function fetchShardHeader(url: string, onBytes: (n: number) => void, signal?: AbortSignal): Promise<ShardHeader> {
  let held = new Uint8Array(0);
  let size = NaN;
  let fetched = 0;
  // Hugging Face redirects /resolve/ to its CDN (us.aws.cdn.hf.co), and the
  // CDN's CORS reply does not allow credentialed requests: with cookies the
  // browser blocks the redirect and fetch() throws "Failed to fetch". Public
  // files need no cookies, so go without; only a 401/403 (gated or private
  // repo) is retried with the user's Hugging Face session.
  let credentials: RequestCredentials = "omit";
  for (;;) {
    const start = held.length;
    const end = start + CHUNK_BYTES - 1;
    const get = () => fetch(url, { headers: { range: `bytes=${start}-${end}` }, credentials, signal });
    let res = await get();
    if ((res.status === 401 || res.status === 403) && credentials === "omit") {
      credentials = "include";
      try {
        res = await get();
      } catch (err) {
        if (err instanceof TypeError) throw new HeaderError("Gated or private repo: accept its terms on Hugging Face while signed in", "gated");
        throw err;
      }
    }
    if (res.status === 401 || res.status === 403) throw new HeaderError("Gated or private repo: accept its terms on Hugging Face while signed in", "gated");
    if (res.status === 404) throw new HeaderError("File not found", "not_found");
    if (res.status !== 206 && res.status !== 200) throw new HeaderError(`HTTP ${res.status}`, "http");
    if (res.status === 200) {
      // The server ignored the range and is sending the whole file: stop it.
      await res.body?.cancel().catch(() => {});
      throw new HeaderError("The server does not accept range requests", "http");
    }
    const range = res.headers.get("content-range");
    const total = range ? Number(/\/(\d+)\s*$/.exec(range)?.[1]) : NaN;
    if (Number.isFinite(total) && total > 0) size = total;
    const body = new Uint8Array(await res.arrayBuffer());
    fetched += body.length;
    onBytes(body.length);
    const next = new Uint8Array(held.length + body.length);
    next.set(held, 0);
    next.set(body, held.length);
    held = next;
    const walk = walkHeader(held);
    if (walk === null) throw new HeaderError("Not a GGUF this reader can walk", "not_gguf");
    if (walk !== "more") {
      if (!Number.isFinite(size)) throw new HeaderError("No file size in the response", "http");
      return { bytes: held, walk, size, fetched };
    }
    if (body.length === 0 || (Number.isFinite(size) && held.length >= size)) throw new HeaderError("File ends inside its own header", "not_gguf");
    if (held.length >= HEADER_MAX_BYTES) throw new HeaderError(`Header is over ${HEADER_MAX_BYTES >> 20} MiB`, "too_large");
  }
}

export interface FactsProgress {
  shard: number;
  shards: number;
  fetchedBytes: number;
}

export async function readModelFacts(
  repo: string,
  file: string,
  opts: { revision?: string; onProgress?: (p: FactsProgress) => void; signal?: AbortSignal } = {}
): Promise<ModelFacts> {
  const shards = shardNames(file);
  let fetched = 0;
  let totalBytes = 0;
  const allTensors: { name: string; size: number }[] = [];
  let first: ShardHeader | null = null;

  for (const [i, shard] of shards.entries()) {
    opts.onProgress?.({ shard: i + 1, shards: shards.length, fetchedBytes: fetched });
    const h = await fetchShardHeader(
      resolveUrl(repo, shard, opts.revision),
      (n) => {
        fetched += n;
        opts.onProgress?.({ shard: i + 1, shards: shards.length, fetchedBytes: fetched });
      },
      opts.signal
    );
    if (i === 0) first = h;
    totalBytes += h.size;
    // Each tensor's stored length is the gap to the next offset; the last one
    // runs to the end of this shard's data section.
    const dataBytes = h.size - h.walk.headerEnd;
    const sorted = [...h.walk.tensors].sort((a, b) => a.offset - b.offset);
    for (let k = 0; k < sorted.length; k++) {
      const next = k + 1 < sorted.length ? sorted[k + 1].offset : dataBytes;
      allTensors.push({ name: sorted[k].name, size: Math.max(0, next - sorted[k].offset) });
    }
  }
  if (!first) throw new HeaderError("No shards", "not_gguf");

  // The app's own parser on shard 1's bytes, served from memory.
  let pos = 0;
  const bytes = first.bytes;
  const info = await parseGgufInfo(
    {
      size: async () => first!.size,
      read: async (into, length) => {
        if (pos >= bytes.length) return 0;
        const n = Math.min(length, bytes.length - pos);
        into.set(bytes.subarray(pos, pos + n), 0);
        pos += n;
        return n;
      },
    },
    CHUNK_BYTES
  );
  if (info.n_layer == null || info.n_layer <= 0) throw new HeaderError(`Cannot size this model (${info.debugReason ?? "no layer count"})`, "unsupported");
  if (!info.kv_layout) throw new HeaderError("Cannot read this model's attention layout", "unsupported");
  const breakdown = shards.length > 1 ? buildTensorLayerBreakdown(allTensors, info.n_layer) : info.tensor_layer_bytes;
  if (!breakdown) throw new HeaderError("Cannot read this model's tensor list", "unsupported");

  return {
    repo,
    file,
    shards,
    totalBytes,
    fetchedBytes: fetched,
    architecture: first.walk.architecture,
    nLayer: info.n_layer,
    trainedCtx: info.trained_ctx,
    expertCount: info.expert_count ?? (breakdown.moe?.some((b) => b > 0) ? 0 : null),
    expertUsedCount: info.expert_used_count,
    paramCount: info.param_count,
    quant: info.quant,
    breakdown,
    layout: info.kv_layout,
  };
}
