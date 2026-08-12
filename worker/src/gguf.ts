import { open, type FileHandle } from "node:fs/promises";

// Minimal sequential reader over a GGUF file's metadata section -- just
// enough to resolve <architecture>.block_count (the model's transformer
// layer count) without loading tensor data or the tokenizer vocab arrays
// (often multi-MB) that typically follow it in the file.
// Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

const GGUF_MAGIC = 0x46554747; // ASCII "GGUF" read as a little-endian uint32
const READ_CHUNK_BYTES = 64 * 1024;
// Safety cap against corrupt/pathological files -- real llama.cpp-produced
// files resolve architecture + block_count within the first few KB, long
// before this matters.
const MAX_READ_BYTES = 64 * 1024 * 1024;
const MAX_STRING_BYTES = 8 * 1024 * 1024;
const MAX_ARRAY_LEN = 50_000_000;
const MAX_KV_COUNT = 1_000_000;

const GGUF_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;

const FIXED_WIDTH: Record<number, number> = {
  [GGUF_TYPE.UINT8]: 1,
  [GGUF_TYPE.INT8]: 1,
  [GGUF_TYPE.BOOL]: 1,
  [GGUF_TYPE.UINT16]: 2,
  [GGUF_TYPE.INT16]: 2,
  [GGUF_TYPE.UINT32]: 4,
  [GGUF_TYPE.INT32]: 4,
  [GGUF_TYPE.FLOAT32]: 4,
  [GGUF_TYPE.UINT64]: 8,
  [GGUF_TYPE.INT64]: 8,
  [GGUF_TYPE.FLOAT64]: 8,
};

class GgufReader {
  private buf = Buffer.alloc(0);
  private bufPos = 0;
  private totalRead = 0;

  constructor(private fh: FileHandle) {}

  private async ensure(n: number): Promise<void> {
    const need = n - (this.buf.length - this.bufPos);
    if (need <= 0) return;
    if (this.totalRead > MAX_READ_BYTES) throw new Error("gguf: exceeded read budget");
    const toRead = Math.max(READ_CHUNK_BYTES, need);
    const chunk = Buffer.alloc(toRead);
    const { bytesRead } = await this.fh.read(chunk, 0, toRead, null);
    if (bytesRead === 0) throw new Error("gguf: unexpected end of file");
    this.totalRead += bytesRead;
    this.buf = Buffer.concat([this.buf.subarray(this.bufPos), chunk.subarray(0, bytesRead)]);
    this.bufPos = 0;
    if (this.buf.length < n) await this.ensure(n);
  }

  private async bytes(n: number): Promise<Buffer> {
    await this.ensure(n);
    const out = this.buf.subarray(this.bufPos, this.bufPos + n);
    this.bufPos += n;
    return out;
  }

  async u8(): Promise<number> {
    return (await this.bytes(1)).readUInt8(0);
  }
  async u16(): Promise<number> {
    return (await this.bytes(2)).readUInt16LE(0);
  }
  async u32(): Promise<number> {
    return (await this.bytes(4)).readUInt32LE(0);
  }
  async u64(): Promise<number> {
    return Number((await this.bytes(8)).readBigUInt64LE(0));
  }
  async i8(): Promise<number> {
    return (await this.bytes(1)).readInt8(0);
  }
  async i16(): Promise<number> {
    return (await this.bytes(2)).readInt16LE(0);
  }
  async i32(): Promise<number> {
    return (await this.bytes(4)).readInt32LE(0);
  }
  async i64(): Promise<number> {
    return Number((await this.bytes(8)).readBigInt64LE(0));
  }
  async f32(): Promise<number> {
    return (await this.bytes(4)).readFloatLE(0);
  }
  async f64(): Promise<number> {
    return (await this.bytes(8)).readDoubleLE(0);
  }

  async string(): Promise<string> {
    const len = await this.u64();
    if (len < 0 || len > MAX_STRING_BYTES) throw new Error(`gguf: implausible string length ${len}`);
    return (await this.bytes(len)).toString("utf8");
  }

  async numeric(type: number): Promise<number> {
    switch (type) {
      case GGUF_TYPE.UINT8:
        return this.u8();
      case GGUF_TYPE.INT8:
        return this.i8();
      case GGUF_TYPE.UINT16:
        return this.u16();
      case GGUF_TYPE.INT16:
        return this.i16();
      case GGUF_TYPE.UINT32:
        return this.u32();
      case GGUF_TYPE.INT32:
        return this.i32();
      case GGUF_TYPE.FLOAT32:
        return this.f32();
      case GGUF_TYPE.UINT64:
        return this.u64();
      case GGUF_TYPE.INT64:
        return this.i64();
      case GGUF_TYPE.FLOAT64:
        return this.f64();
      default:
        throw new Error(`gguf: value type ${type} is not numeric`);
    }
  }

  async skip(type: number): Promise<void> {
    const width = FIXED_WIDTH[type];
    if (width !== undefined) {
      await this.bytes(width);
      return;
    }
    if (type === GGUF_TYPE.STRING) {
      await this.string();
      return;
    }
    if (type === GGUF_TYPE.ARRAY) {
      const elemType = await this.u32();
      const len = await this.u64();
      if (len < 0 || len > MAX_ARRAY_LEN) throw new Error(`gguf: implausible array length ${len}`);
      for (let i = 0; i < len; i++) await this.skip(elemType);
      return;
    }
    throw new Error(`gguf: unknown value type ${type}`);
  }
}

export interface GgufInfo {
  n_layer: number | null;
  // GGUF's <architecture>.nextn_predict_layers -- >0 means this file has a
  // usable MTP (multi-token-prediction) head: either baked into a normal
  // base model (Qwen/DeepSeek/GLM-style, directly benchmarkable with
  // --spec-type draft-mtp and no companion file), or, for a standalone
  // drafter file (Gemma-4-style, detected separately via filename -- see
  // server/src/routes/workers.ts), this key describes the drafter itself.
  // See shared/types.ts's ModelMetadata.mtp_layers/mtp_role.
  mtp_layers: number | null;
}

const EMPTY_GGUF_INFO: GgufInfo = { n_layer: null, mtp_layers: null };

// Reads a GGUF file's metadata header to find the model's transformer layer
// count and (if present) its MTP/nextn layer count. Returns nulls on any
// parse failure or unrecognized file rather than throwing -- callers treat a
// missing value as "unknown", same fail-soft posture as the rest of
// hardware/model detection in this worker.
export async function readGgufInfo(filePath: string): Promise<GgufInfo> {
  let fh: FileHandle | undefined;
  try {
    fh = await open(filePath, "r");
    const reader = new GgufReader(fh);

    const magic = await reader.u32();
    if (magic !== GGUF_MAGIC) return EMPTY_GGUF_INFO;
    const version = await reader.u32();
    if (version < 2) return EMPTY_GGUF_INFO; // v1 used uint32 counts; no current build produces it

    await reader.u64(); // tensor_count -- unused here, consumed only to advance the cursor
    const kvCount = await reader.u64();
    if (kvCount < 0 || kvCount > MAX_KV_COUNT) return EMPTY_GGUF_INFO;

    let architecture: string | undefined;
    const blockCounts = new Map<string, number>();
    const mtpLayerCounts = new Map<string, number>();

    for (let i = 0; i < kvCount; i++) {
      const key = await reader.string();
      // Hyperparameter keys (general.*, <arch>.*, including block_count and
      // nextn_predict_layers) always precede tokenizer.* keys in a
      // llama.cpp-produced GGUF -- stop here rather than walking the (often
      // multi-MB) vocab/merges arrays that follow, which hold nothing this
      // function needs. A file that somehow doesn't follow that convention
      // just falls through to the post-loop lookup below with whatever was
      // found so far, rather than getting a wrong answer.
      if (key.startsWith("tokenizer.")) break;

      const valueType = await reader.u32();
      if (key === "general.architecture" && valueType === GGUF_TYPE.STRING) {
        architecture = await reader.string();
      } else if (key.endsWith(".block_count") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        blockCounts.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".nextn_predict_layers") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        mtpLayerCounts.set(key, await reader.numeric(valueType));
      } else {
        await reader.skip(valueType);
      }
    }
    return {
      n_layer: architecture ? blockCounts.get(`${architecture}.block_count`) ?? null : null,
      mtp_layers: architecture ? mtpLayerCounts.get(`${architecture}.nextn_predict_layers`) ?? null : null,
    };
  } catch {
    return EMPTY_GGUF_INFO;
  } finally {
    await fh?.close().catch(() => {});
  }
}
