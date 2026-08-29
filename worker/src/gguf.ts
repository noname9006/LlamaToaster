import { open, type FileHandle } from "node:fs/promises";
import type { TensorLayerBreakdown } from "../../shared/vramEstimate.js";

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
// Bounds for the tensor_info walk's own counters -- generous enough for any
// real model (the largest current checkpoints stay in the tens of thousands
// of tensors, each with at most 4-5 dims) but enough to reject a corrupt
// length field that would otherwise drive a wild, potentially multi-GB read.
const MAX_TENSOR_COUNT = 50_000_000;
const MAX_TENSOR_DIMS = 8;

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
  // Logical stream position (bytes actually consumed via bytes(), independent
  // of how far ensure()'s read-ahead has buffered past that point) -- used to
  // locate the tensor data section's start once the tensor_info walk
  // finishes, since GGUF stores each tensor's offset relative to that point
  // rather than to the start of the file.
  pos = 0;

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
    this.pos += n;
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

// Maps GGUF's general.file_type integer (llama.cpp's llama_ftype enum,
// baked into every GGUF file at conversion time by the tool that produced
// it -- independent of whatever the file happens to be named) to the same
// quant-code strings HF filenames use (Q4_K_M, Q8_0, F16, ...). This is the
// authoritative source: a file's *name* carrying no quant token (e.g. an
// unsloth MTP drafter like "mtp-gemma-4-E2B-it.gguf") doesn't mean the file
// has no quantization, just that the filename doesn't advertise it.
// See llama.cpp's include/llama.h llama_ftype enum. Only covers currently-
// live values -- a handful of historic IDs (4,5,6,33,34,35) were retired
// quant formats and are deliberately absent, as is 1024
// (LLAMA_FTYPE_GUESSED, meaning llama.cpp itself couldn't determine one).
const FTYPE_QUANT_LABEL: Record<number, string> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  7: "Q8_0",
  8: "Q5_0",
  9: "Q5_1",
  10: "Q2_K",
  11: "Q3_K_S",
  12: "Q3_K_M",
  13: "Q3_K_L",
  14: "Q4_K_S",
  15: "Q4_K_M",
  16: "Q5_K_S",
  17: "Q5_K_M",
  18: "Q6_K",
  19: "IQ2_XXS",
  20: "IQ2_XS",
  21: "Q2_K_S",
  22: "IQ3_XS",
  23: "IQ3_XXS",
  24: "IQ1_S",
  25: "IQ4_NL",
  26: "IQ3_S",
  27: "IQ3_M",
  28: "IQ2_S",
  29: "IQ2_M",
  30: "IQ4_XS",
  31: "IQ1_M",
  32: "BF16",
  36: "TQ1_0",
  37: "TQ2_0",
};

export function quantLabelFromFileType(fileType: number): string | null {
  return FTYPE_QUANT_LABEL[fileType] ?? null;
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
  // Quant code derived from general.file_type (see FTYPE_QUANT_LABEL above),
  // not from the filename -- null when the key is absent (very old/foreign
  // conversion tools) or its value isn't one of the currently-live ftypes.
  quant: string | null;
  // Total element count across every tensor (sum of product(dimensions) over
  // the file's tensor_info section) -- the real parameter count, read from
  // the file itself. This is the same number HF's own API computes from the
  // tensor shapes; unlike HF's repo-level gguf.total it is per-FILE, so a
  // multi-model repo (e.g. Ex0bit's PRISM-DQ repos shipping 0.8B/2B/4B/9B
  // files under one repo id) can't report one file's count for another's.
  // null on any parse failure -- fail-soft, same as every other field here.
  param_count: number | null;
  // GGUF's <architecture>.context_length -- the sequence length the model was
  // trained on ("trained context"). Anchors M2's target-context clamp,
  // N1's sizing-ladder ceiling, and the Benchmark page's model card.
  trained_ctx: number | null;
  // KV-cache geometry consumed by shared/vramEstimate.ts's
  // maxAffordableContext (M1) / kvBytesPerToken -- <architecture>.embedding_length
  // and <architecture>.attention.* hyperparameters. n_embd/n_head are the
  // both-or-neither fallback pair kvBytesPerToken derives head dims from when
  // the explicit key/value lengths are absent.
  n_head_kv: number | null;
  head_dim_k: number | null;
  head_dim_v: number | null;
  n_embd: number | null;
  n_head: number | null;
  // <architecture>.attention.sliding_window -- present only for architectures
  // with sliding-window attention (Mistral/Gemma-style); null means plain
  // full attention, where KV cache grows linearly across the whole context.
  sliding_window: number | null;
  // Real per-tensor weight-byte breakdown -- see TensorLayerBreakdown. Null
  // whenever it can't be built: n_layer unresolved (nothing to size the
  // per-block arrays against), the tensor_info walk failed, or the file has
  // no tensors at all.
  tensor_layer_bytes: TensorLayerBreakdown | null;
  // Local-only diagnostic for why n_layer/mtp_layers came back
  // null -- never sent over the wire (worker/src/index.ts logs it, nothing
  // else reads it), purely so a failed lookup says WHY instead of leaving
  // "unknown" to guess at (bad magic? unreadable file? architecture key
  // never found? found but no matching block_count key?).
  debugReason?: string;
}

function emptyGgufInfo(debugReason: string): GgufInfo {
  return {
    n_layer: null,
    mtp_layers: null,
    quant: null,
    param_count: null,
    trained_ctx: null,
    n_head_kv: null,
    head_dim_k: null,
    head_dim_v: null,
    n_embd: null,
    n_head: null,
    sliding_window: null,
    tensor_layer_bytes: null,
    debugReason,
  };
}

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
    if (magic !== GGUF_MAGIC) return emptyGgufInfo(`bad magic 0x${magic.toString(16)}`);
    const version = await reader.u32();
    if (version < 2) return emptyGgufInfo(`gguf version ${version} < 2`); // v1 used uint32 counts; no current build produces it

    const tensorCount = await reader.u64(); // tensor_count -- needed only to bound the tensor_info walk below
    const kvCount = await reader.u64();
    if (kvCount < 0 || kvCount > MAX_KV_COUNT) return emptyGgufInfo(`implausible kv_count ${kvCount}`);

    let architecture: string | undefined;
    let fileType: number | undefined;
    // GGUF's general.alignment -- each tensor's stored offset is a multiple
    // of this, and the tensor data section itself starts at the first such
    // multiple after tensor_info. Spec default is 32 when the key is absent.
    let alignment = 32;
    const blockCounts = new Map<string, number>();
    const mtpLayerCounts = new Map<string, number>();
    // KV-cache geometry + trained context -- captured exactly like the maps
    // above (architecture-prefixed keys resolved after the walk), so a
    // hyperparameter that happens to appear before the architecture key is
    // still picked up. Suffix style matches block_count's handling.
    const contextLengths = new Map<string, number>();
    const headCountKv = new Map<string, number>();
    const keyLengths = new Map<string, number>();
    const valueLengths = new Map<string, number>();
    const embeddingLengths = new Map<string, number>();
    const headCounts = new Map<string, number>();
    const slidingWindows = new Map<string, number>();

    // Deliberately walks every KV pair rather than stopping at the first
    // "tokenizer."-prefixed key: a llama.cpp-produced GGUF always puts
    // hyperparameter keys (general.*, <arch>.*, including block_count)
    // before tokenizer.* ones, but files from other conversion pipelines
    // (observed live: NVIDIA Nemotron GGUFs) don't reliably follow that
    // ordering -- stopping early there silently reported every field
    // "unknown" for an otherwise perfectly normal file. This does mean
    // reading through the (often multi-MB) vocab/merges arrays that follow,
    // via skip()'s cheap discard-without-allocating path -- an acceptable
    // cost since both call sites (a download's one-time metadata read, and
    // the on-demand backfill button) are already one-off actions, not
    // something run per file on every heartbeat.
    for (let i = 0; i < kvCount; i++) {
      const key = await reader.string();
      const valueType = await reader.u32();
      if (key === "general.architecture" && valueType === GGUF_TYPE.STRING) {
        architecture = await reader.string();
      } else if (key === "general.file_type" && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        fileType = await reader.numeric(valueType);
      } else if (key === "general.alignment" && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        const a = await reader.numeric(valueType);
        if (a > 0) alignment = a;
      } else if (key.endsWith(".block_count") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        blockCounts.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".nextn_predict_layers") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        mtpLayerCounts.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".context_length") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        contextLengths.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".attention.head_count_kv") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        headCountKv.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.key_length") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        keyLengths.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".attention.value_length") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        valueLengths.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".embedding_length") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        embeddingLengths.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.head_count") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        // NB: this can't collide with .head_count_kv above -- a key ending in
        // ".head_count_kv" doesn't end in ".attention.head_count".
        headCounts.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.sliding_window") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        slidingWindows.set(key, await reader.numeric(valueType));
      } else {
        await reader.skip(valueType);
      }
    }
    if (!architecture) return emptyGgufInfo("no general.architecture key found in the whole file");
    const n_layer = blockCounts.get(`${architecture}.block_count`) ?? null;

    // Real per-file parameter count from the tensor_info section that
    // directly follows the KV metadata. Best-effort and deliberately
    // isolated from the KV parsing above: a pathological/corrupt file (or a
    // reader bug) that trips this walk must not nuke the n_layer/quant/etc.
    // already resolved -- it just leaves param_count null, same fail-soft
    // posture as every other field here.
    let param_count: number | null = null;
    // Name + declared offset for every tensor -- gathered in the same walk as
    // param_count (tensor_info can only be streamed through once) and turned
    // into tensor_layer_bytes below via offset deltas, once the data
    // section's start is known.
    const tensorOffsets: { name: string; offset: number }[] = [];
    try {
      // No tensor_info_count field exists: the tensor_infos array follows the
      // KV metadata directly and is counted by the header's tensor_count
      // (spec's gguf_file_t: tensor_infos[header.tensor_count]). Reading an
      // extra u64 here (as an earlier version did, assuming the count was
      // duplicated) silently misaligned the whole walk on every real file --
      // the "count" read was actually the first tensor's name-length prefix.
      if (tensorCount < 0 || tensorCount > MAX_TENSOR_COUNT) {
        throw new Error(`implausible tensor_count ${tensorCount}`);
      }
      let total = 0n;
      for (let i = 0; i < tensorCount; i++) {
        const name = await reader.string();
        const nDims = await reader.u32();
        if (nDims > MAX_TENSOR_DIMS) throw new Error(`implausible tensor dim count ${nDims}`);
        let elements = 1n;
        for (let d = 0; d < nDims; d++) {
          elements *= BigInt(await reader.u64());
        }
        await reader.u32(); // tensor type -- not needed for the element count or the byte-size-by-offset-delta below
        const offset = await reader.u64();
        tensorOffsets.push({ name, offset });
        total += elements;
      }
      param_count = total <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(total) : null;
    } catch (tensorErr) {
      // Leave param_count null; the fields resolved above still stand.
      // n_layer==null is not the trigger for logging this -- the KV walk's own
      // debugReason already covers that -- so just swallow the detail here.
      void tensorErr;
    }

    // Real per-tensor byte sizes, independent of quantization format: each
    // tensor's stored length is the gap to the next tensor's offset (sorted
    // by offset, since nothing guarantees tensor_info's declaration order
    // matches on-disk order, even though real writers keep them the same),
    // with the last tensor's length filled in from the file's own total size.
    // This needs no per-ggml-type block-size table at all -- a brand new
    // quantization format is sized correctly the moment llama.cpp can write
    // it, with zero changes here. Isolated in its own try/catch for the same
    // reason as param_count above: a corrupt/truncated data section must not
    // disturb any already-resolved field.
    let tensor_layer_bytes: TensorLayerBreakdown | null = null;
    try {
      if (n_layer != null && n_layer > 0 && tensorOffsets.length > 0) {
        const dataSectionStart = Math.ceil(reader.pos / alignment) * alignment;
        const fileSize = (await fh.stat()).size;
        const sorted = [...tensorOffsets].sort((a, b) => a.offset - b.offset);
        const sizes = new Map<string, number>();
        for (let i = 0; i < sorted.length; i++) {
          const next = i + 1 < sorted.length ? sorted[i + 1].offset : fileSize - dataSectionStart;
          sizes.set(sorted[i].name, Math.max(0, next - sorted[i].offset));
        }
        tensor_layer_bytes = buildTensorLayerBreakdown(tensorOffsets.map((t) => ({ name: t.name, size: sizes.get(t.name) ?? 0 })), n_layer);
      }
    } catch (sizeErr) {
      void sizeErr;
    }

    return {
      n_layer,
      mtp_layers: mtpLayerCounts.get(`${architecture}.nextn_predict_layers`) ?? null,
      quant: fileType != null ? quantLabelFromFileType(fileType) : null,
      param_count,
      trained_ctx: contextLengths.get(`${architecture}.context_length`) ?? null,
      // attention.head_count_kv is OPTIONAL in the GGUF spec: a model that
      // doesn't use grouped-query attention (plain multi-head) often has no
      // such key at all, and llama.cpp's own hparam loader defaults it to
      // head_count in that case. Mirror that default here -- otherwise every
      // non-GQA model reads as "unknown KV geometry" even though the true
      // value is fully derivable from head_count alone.
      n_head_kv:
        headCountKv.get(`${architecture}.attention.head_count_kv`) ??
        headCounts.get(`${architecture}.attention.head_count`) ??
        null,
      head_dim_k: keyLengths.get(`${architecture}.attention.key_length`) ?? null,
      head_dim_v: valueLengths.get(`${architecture}.attention.value_length`) ?? null,
      n_embd: embeddingLengths.get(`${architecture}.embedding_length`) ?? null,
      n_head: headCounts.get(`${architecture}.attention.head_count`) ?? null,
      sliding_window: slidingWindows.get(`${architecture}.attention.sliding_window`) ?? null,
      tensor_layer_bytes,
      debugReason:
        n_layer === null ? `architecture "${architecture}" found but no ${architecture}.block_count key` : undefined,
    };
  } catch (err) {
    return emptyGgufInfo(`parse threw: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    await fh?.close().catch(() => {});
  }
}

// A block's transformer-block tensors, e.g. "blk.12.attn_q.weight" -- capture
// group 1 is the block index, group 2 is everything after it.
const BLOCK_TENSOR = /^blk\.(\d+)\.(.+)$/;
// MoE expert-weight tensors, both real-world naming conventions: the modern
// stacked 3D tensor llama.cpp itself writes ("ffn_gate_exps.weight") and the
// older per-expert convention some conversion pipelines still produce
// ("ffn_gate.3.weight", one tensor per expert index). Deliberately excludes
// plain dense FFN tensors ("ffn_gate.weight", no _exps suffix or numeric
// index) -- those are ordinary per-layer weights, not subject to
// --n-cpu-moe.
const MOE_EXPERT_TENSOR = /^ffn_(?:gate|up|down)(_exps)?(?:\.\d+)?\.(?:weight|bias)$/;

function isMoeExpertTensorName(rest: string): boolean {
  const m = MOE_EXPERT_TENSOR.exec(rest);
  if (!m) return false;
  return m[1] === "_exps" || /^ffn_(?:gate|up|down)\.\d+\./.test(rest);
}

// Buckets every tensor's real on-disk byte size (see readGgufInfo's caller
// above) into the shape shared/vramEstimate.ts's placeWeightBytes consumes.
// A block index found in a tensor name but outside [0, nLayer) (a malformed
// or non-standard file) falls back to the "other" bucket rather than
// crashing or silently growing the per-block arrays past the real layer
// count.
function buildTensorLayerBreakdown(tensors: { name: string; size: number }[], nLayer: number): TensorLayerBreakdown {
  const dense = new Array(nLayer).fill(0);
  const moe = new Array(nLayer).fill(0);
  let embed = 0;
  let output = 0;
  let other = 0;

  for (const t of tensors) {
    const blockMatch = BLOCK_TENSOR.exec(t.name);
    if (blockMatch) {
      const idx = Number(blockMatch[1]);
      if (Number.isInteger(idx) && idx >= 0 && idx < nLayer) {
        if (isMoeExpertTensorName(blockMatch[2])) moe[idx] += t.size;
        else dense[idx] += t.size;
        continue;
      }
      other += t.size;
      continue;
    }
    if (t.name.startsWith("token_embd.")) {
      embed += t.size;
    } else if (t.name.startsWith("output_norm.") || t.name.startsWith("output.")) {
      output += t.size;
    } else {
      other += t.size;
    }
  }

  return { dense, moe, embed, output, other };
}
