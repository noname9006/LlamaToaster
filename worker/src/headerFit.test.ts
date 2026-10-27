import { afterAll, describe, expect, it } from "vitest";
import { closeSync, existsSync, mkdtempSync, openSync, readSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  fetchGgufHeader,
  ggufHeaderEnd,
  HEADER_FIRST_READ_BYTES,
  HeaderFitError,
  prepareHeaderModel,
  shardNames,
  writeSparsePlaceholder,
  type FetchFn,
} from "./headerFit.js";

const u32 = (n: number) => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n);
  return b;
};
const u64 = (n: number) => {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
};
const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);

// A GGUF header: one u32 KV, one string-array KV of `vocab` tokens, `tensors`
// tensors of 2 dims. Returns the header and its expected aligned end.
function gguf(opts: { vocab?: number; tensors?: number; alignment?: number } = {}): { buf: Buffer; end: number } {
  const kv: Buffer[] = [];
  let nKv = 0;
  if (opts.alignment) {
    kv.push(str("general.alignment"), u32(4), u32(opts.alignment));
    nKv++;
  }
  kv.push(str("llama.block_count"), u32(4), u32(2));
  nKv++;
  const tokens = Array.from({ length: opts.vocab ?? 3 }, (_, i) => str(`tok${i}`));
  kv.push(str("tokenizer.ggml.tokens"), u32(9), u32(8), u64(tokens.length), Buffer.concat(tokens));
  nKv++;
  const tens: Buffer[] = [];
  for (let i = 0; i < (opts.tensors ?? 2); i++) tens.push(str(`blk.${i}.w`), u32(2), u64(4), u64(4), u32(0), u64(i * 64));
  const body = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(opts.tensors ?? 2), u64(nKv), ...kv, ...tens]);
  const align = opts.alignment ?? 32;
  return { buf: body, end: Math.ceil(body.length / align) * align };
}

const dirs: string[] = [];
const tmp = () => {
  const d = mkdtempSync(join(tmpdir(), "llamatoaster-headerfit-test-"));
  dirs.push(d);
  return d;
};
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

describe("ggufHeaderEnd", () => {
  it("finds the aligned end of metadata + tensor info", () => {
    const g = gguf();
    expect(ggufHeaderEnd(g.buf)).toBe(g.end);
  });

  it("honours general.alignment", () => {
    const g = gguf({ alignment: 64 });
    expect(ggufHeaderEnd(g.buf)).toBe(g.end);
    expect(g.end % 64).toBe(0);
  });

  it("asks for more bytes when the buffer stops inside the header", () => {
    const g = gguf({ vocab: 50 });
    expect(ggufHeaderEnd(g.buf.subarray(0, 40))).toBe("more");
    expect(ggufHeaderEnd(g.buf.subarray(0, g.buf.length - 3))).toBe("more");
  });

  it("rejects what is not a GGUF", () => {
    expect(ggufHeaderEnd(Buffer.from("NOPE and more bytes"))).toBeNull();
  });
});

describe("shardNames", () => {
  it("expands a split model from any shard name, folder kept", () => {
    expect(shardNames("UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf")).toEqual([
      "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf",
      "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00002-of-00003.gguf",
      "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00003-of-00003.gguf",
    ]);
  });
  it("leaves a single file alone", () => {
    expect(shardNames("model-Q4_K_M.gguf")).toEqual(["model-Q4_K_M.gguf"]);
  });
});

// Serves `file` (padded to `size`) the way the HF CDN answers a range read.
function cdn(files: Record<string, { header: Buffer; size: number }>, status?: number): { fetchFn: FetchFn; ranges: string[] } {
  const ranges: string[] = [];
  const fetchFn: FetchFn = async (url, init) => {
    const name = url.slice(url.lastIndexOf("/") + 1);
    const f = files[name];
    if (status) return { status, headers: { get: () => null }, arrayBuffer: async () => new ArrayBuffer(0) };
    if (!f) return { status: 404, headers: { get: () => null }, arrayBuffer: async () => new ArrayBuffer(0) };
    const range = init.headers.range;
    ranges.push(range);
    const end = Number(/bytes=0-(\d+)/.exec(range)![1]);
    const body = Buffer.alloc(Math.min(end + 1, f.size));
    f.header.copy(body, 0, 0, Math.min(f.header.length, body.length));
    return {
      status: 206,
      headers: { get: (n: string) => (n.toLowerCase() === "content-range" ? `bytes 0-${body.length - 1}/${f.size}` : null) },
      arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.length),
    };
  };
  return { fetchFn, ranges };
}

describe("fetchGgufHeader", () => {
  it("doubles the range until the header ends, and reports the full size", async () => {
    // A vocabulary large enough to spill past the first 4 MiB read.
    const g = gguf({ vocab: 300_000 });
    expect(g.buf.length).toBeGreaterThan(HEADER_FIRST_READ_BYTES);
    const { fetchFn, ranges } = cdn({ "m.gguf": { header: g.buf, size: 5_000_000_000 } });
    const h = await fetchGgufHeader("https://x/m.gguf", { fetchFn });
    expect(h.header.length).toBe(g.end);
    expect(h.size).toBe(5_000_000_000);
    expect(ranges).toEqual([`bytes=0-${HEADER_FIRST_READ_BYTES - 1}`, `bytes=0-${2 * HEADER_FIRST_READ_BYTES - 1}`]);
  });

  it("names a gated repo", async () => {
    const { fetchFn } = cdn({}, 401);
    await expect(fetchGgufHeader("https://x/m.gguf", { fetchFn })).rejects.toMatchObject({ reason: "gated" });
  });
});

describe("writeSparsePlaceholder", () => {
  it("keeps the header and the full size without allocating it (real volume)", async () => {
    const d = tmp();
    const path = join(d, "m.gguf");
    const header = gguf().buf;
    const size = 4 * 1024 * 1024 * 1024;
    await writeSparsePlaceholder(path, header, size);
    expect(statSync(path).size).toBe(size);
    const head = Buffer.alloc(header.length);
    const fd = openSync(path, "r");
    try {
      readSync(fd, head, 0, head.length, 0);
    } finally {
      closeSync(fd);
    }
    expect(head.equals(header)).toBe(true);
  }, 60_000);

  it("gives up and removes the file when the volume allocated the extension", async () => {
    const d = tmp();
    const path = join(d, "m.gguf");
    let calls = 0;
    // Free space drops by the whole extension: not sparse.
    const freeBytes = async () => (calls++ === 0 ? 10e9 : 10e9 - 4e9);
    await expect(writeSparsePlaceholder(path, gguf().buf, 4e9, { platform: "linux", freeBytes })).rejects.toBeInstanceOf(HeaderFitError);
    expect(existsSync(path)).toBe(false);
  });
});

describe("prepareHeaderModel", () => {
  it("lays out every shard and returns shard 1", async () => {
    const d = tmp();
    const files = {
      "s-00001-of-00002.gguf": { header: gguf({ tensors: 0 }).buf, size: 9_000_000 },
      "s-00002-of-00002.gguf": { header: gguf({ tensors: 5 }).buf, size: 2_000_000_000 },
    };
    const { fetchFn } = cdn(files);
    const m = await prepareHeaderModel({ dir: join(d, "h"), file: "s-00001-of-00002.gguf", urlFor: (f) => `https://x/${f}`, fetchFn });
    expect(m.path).toBe(join(d, "h", "s-00001-of-00002.gguf"));
    expect(statSync(join(d, "h", "s-00002-of-00002.gguf")).size).toBe(2_000_000_000);
    expect(m.totalBytes).toBe(2_009_000_000);
    await m.cleanup();
    expect(existsSync(join(d, "h"))).toBe(false);
  }, 60_000);

  it("cleans up when a shard fails", async () => {
    const d = tmp();
    const { fetchFn } = cdn({ "s-00001-of-00002.gguf": { header: gguf().buf, size: 1000 } });
    await expect(
      prepareHeaderModel({ dir: join(d, "h"), file: "s-00001-of-00002.gguf", urlFor: (f) => `https://x/${f}`, fetchFn })
    ).rejects.toMatchObject({ reason: "not_found" });
    expect(existsSync(join(d, "h"))).toBe(false);
  });
});
