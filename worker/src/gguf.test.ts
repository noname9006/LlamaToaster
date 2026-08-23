import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readGgufInfo, quantLabelFromFileType } from "./gguf.js";

// Minimal GGUF byte-level writer -- just enough to exercise readGgufInfo's
// own reader, independent of any real model file. Type ids mirror gguf.ts's
// own GGUF_TYPE map (not exported, so duplicated here deliberately).
const T = { UINT32: 4, STRING: 8 } as const;

type Kv = [key: string, type: number, value: number | string];

function u32(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n, 0);
  return b;
}
function u64(n: number): Buffer {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n), 0);
  return b;
}
function ggufString(s: string): Buffer {
  const body = Buffer.from(s, "utf8");
  return Buffer.concat([u64(body.length), body]);
}
function kvBuffer([key, type, value]: Kv): Buffer {
  const parts = [ggufString(key), u32(type)];
  if (type === T.STRING) parts.push(ggufString(value as string));
  else if (type === T.UINT32) parts.push(u32(value as number));
  else throw new Error(`test helper doesn't support type ${type}`);
  return Buffer.concat(parts);
}

// tensorCount is irrelevant to readGgufInfo (consumed only to advance the
// cursor) -- always 0 here.
function buildGguf(kvs: Kv[]): Buffer {
  return Buffer.concat([
    u32(0x46554747), // magic "GGUF"
    u32(3), // version
    u64(0), // tensor_count
    u64(kvs.length), // kv_count
    ...kvs.map(kvBuffer),
  ]);
}

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-gguf-test-"));
let counter = 0;
function writeTempGguf(kvs: Kv[]): string {
  const path = join(tmpDir, `test-${counter++}.gguf`);
  writeFileSync(path, buildGguf(kvs));
  return path;
}

describe("readGgufInfo", () => {
  it("resolves n_layer/expert_count for a normal, well-ordered file", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen3moe"],
      ["qwen3moe.block_count", T.UINT32, 48],
      ["qwen3moe.expert_count", T.UINT32, 128],
      ["tokenizer.ggml.model", T.STRING, "gpt2"],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBe(48);
    expect(info.expert_count).toBe(128);
    expect(info.debugReason).toBeUndefined();
  });

  // Regression test: the original implementation stopped at the first
  // "tokenizer."-prefixed key on the assumption hyperparameter keys always
  // come first in a llama.cpp-produced GGUF. Real files from other
  // conversion pipelines (observed live: NVIDIA Nemotron GGUFs) don't
  // reliably follow that ordering -- this reproduces that ordering and
  // confirms n_layer is still found instead of silently coming back
  // "unknown" for an otherwise perfectly normal file.
  it("still resolves n_layer when tokenizer keys precede the architecture/block_count keys", async () => {
    const path = writeTempGguf([
      ["tokenizer.ggml.model", T.STRING, "gpt2"],
      ["tokenizer.ggml.bos_token_id", T.UINT32, 1],
      ["general.architecture", T.STRING, "nemotron"],
      ["nemotron.block_count", T.UINT32, 32],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBe(32);
    expect(info.expert_count).toBeNull();
  });

  // Regression coverage for a file whose name carries no quant token at all
  // (real-world case: unsloth's "mtp-gemma-4-E2B-it.gguf" drafter) -- the
  // Models page fell back to filename parsing alone and showed "?" for it
  // even though the file's own header says exactly what quant it is.
  it("resolves quant from general.file_type independent of any filename", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gemma3"],
      ["general.file_type", T.UINT32, 7], // LLAMA_FTYPE_MOSTLY_Q8_0
      ["gemma3.block_count", T.UINT32, 4],
    ]);
    const info = await readGgufInfo(path);
    expect(info.quant).toBe("Q8_0");
  });

  it("returns null quant for a file_type value with no known mapping", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gemma3"],
      ["general.file_type", T.UINT32, 1024], // LLAMA_FTYPE_GUESSED
      ["gemma3.block_count", T.UINT32, 4],
    ]);
    const info = await readGgufInfo(path);
    expect(info.quant).toBeNull();
  });

  it("returns null quant when general.file_type is absent", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gemma3"],
      ["gemma3.block_count", T.UINT32, 4],
    ]);
    const info = await readGgufInfo(path);
    expect(info.quant).toBeNull();
  });

  it("returns null with a debugReason when no general.architecture key exists at all", async () => {
    const path = writeTempGguf([["some.other.key", T.UINT32, 1]]);
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBeNull();
    expect(info.debugReason).toMatch(/no general\.architecture key/);
  });

  it("returns null with a debugReason when architecture is found but its block_count key is missing", async () => {
    const path = writeTempGguf([["general.architecture", T.STRING, "mystery-arch"]]);
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBeNull();
    expect(info.debugReason).toMatch(/mystery-arch.*no mystery-arch\.block_count/);
  });

  it("returns null with a debugReason for a file with a bad magic number", async () => {
    const path = join(tmpDir, `bad-magic-${counter++}.gguf`);
    writeFileSync(path, Buffer.from([0, 0, 0, 0]));
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBeNull();
    expect(info.debugReason).toMatch(/bad magic/);
  });

  it("returns null with a debugReason for a nonexistent file", async () => {
    const info = await readGgufInfo(join(tmpDir, "does-not-exist.gguf"));
    expect(info.n_layer).toBeNull();
    expect(info.debugReason).toMatch(/parse threw/);
  });
});

describe("quantLabelFromFileType", () => {
  it("maps known llama_ftype values to their filename-style quant codes", () => {
    expect(quantLabelFromFileType(0)).toBe("F32");
    expect(quantLabelFromFileType(1)).toBe("F16");
    expect(quantLabelFromFileType(15)).toBe("Q4_K_M");
    expect(quantLabelFromFileType(32)).toBe("BF16");
  });

  it("returns null for retired/unrecognized ftype ids", () => {
    expect(quantLabelFromFileType(4)).toBeNull();
    expect(quantLabelFromFileType(1024)).toBeNull();
  });
});

afterAll(() => {
  try {
    rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* best-effort cleanup */
  }
});
