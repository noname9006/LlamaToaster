import type { KvLayout, TensorLayerBreakdown } from "./vramEstimate.js";

// Where the parser's bytes come from -- a local file on a worker
// (worker/src/gguf.ts), or HTTP Range reads of a Hugging Face file on the
// server (docs/PLAN_MODEL_METADATA_TRUST.md step 2). Read strictly
// sequentially from byte 0; size() is the whole file's length (needed for
// the last tensor's byte size). Deliberately Node-free (Uint8Array/DataView/
// TextDecoder only) since shared/ is also compiled for the browser.
export interface GgufByteSource {
  // Fills `into` with up to `length` next bytes; returns how many (0 = EOF).
  read(into: Uint8Array, length: number): Promise<number>;
  size(): Promise<number>;
}

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
// Upper bound on a model's transformer block count. Everything sized per layer
// (tensor_layer_bytes, the KV walk, the client's placement loops) allocates or
// loops this many times, and the value comes straight from the file -- a
// 240-byte header claiming 300M layers took the process down with an
// uncatchable heap OOM (audit 2026-10-18 A1). Real models are in the low
// hundreds; 4096 matches the caps on kv_layout.kinds, probe ngl and fit
// layers_total.
export const MAX_GGUF_LAYERS = 4096;
// Per-layer arrays (head_count_kv, key_length, ...) longer than this can't be
// one value per layer; they're skipped instead of materialised.
const MAX_PER_LAYER_ARRAY = MAX_GGUF_LAYERS;

/** A plausible transformer block count: an integer in [1, MAX_GGUF_LAYERS]. */
export function isPlausibleLayerCount(n: unknown): n is number {
  return typeof n === "number" && Number.isInteger(n) && n >= 1 && n <= MAX_GGUF_LAYERS;
}

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
  private buf = new Uint8Array(0);
  private bufPos = 0;
  private totalRead = 0;
  private readonly utf8 = new TextDecoder("utf-8");
  // Logical stream position (bytes actually consumed via bytes(), independent
  // of how far ensure()'s read-ahead has buffered past that point) -- used to
  // locate the tensor data section's start once the tensor_info walk
  // finishes, since GGUF stores each tensor's offset relative to that point
  // rather than to the start of the file.
  pos = 0;

  constructor(
    private source: GgufByteSource,
    private chunkBytes = READ_CHUNK_BYTES
  ) {}

  private async ensure(n: number): Promise<void> {
    const need = n - (this.buf.length - this.bufPos);
    if (need <= 0) return;
    if (this.totalRead > MAX_READ_BYTES) throw new Error("gguf: exceeded read budget");
    const toRead = Math.max(this.chunkBytes, need);
    const chunk = new Uint8Array(toRead);
    const bytesRead = await this.source.read(chunk, toRead);
    if (bytesRead === 0) throw new Error("gguf: unexpected end of file");
    this.totalRead += bytesRead;
    const rest = this.buf.subarray(this.bufPos);
    const merged = new Uint8Array(rest.length + bytesRead);
    merged.set(rest, 0);
    merged.set(chunk.subarray(0, bytesRead), rest.length);
    this.buf = merged;
    this.bufPos = 0;
    if (this.buf.length < n) await this.ensure(n);
  }

  private async bytes(n: number): Promise<Uint8Array> {
    await this.ensure(n);
    const out = this.buf.subarray(this.bufPos, this.bufPos + n);
    this.bufPos += n;
    this.pos += n;
    return out;
  }

  private async view(n: number): Promise<DataView> {
    const b = await this.bytes(n);
    return new DataView(b.buffer, b.byteOffset, n);
  }

  async u8(): Promise<number> {
    return (await this.view(1)).getUint8(0);
  }
  async u16(): Promise<number> {
    return (await this.view(2)).getUint16(0, true);
  }
  async u32(): Promise<number> {
    return (await this.view(4)).getUint32(0, true);
  }
  async u64(): Promise<number> {
    return Number((await this.view(8)).getBigUint64(0, true));
  }
  async i8(): Promise<number> {
    return (await this.view(1)).getInt8(0);
  }
  async i16(): Promise<number> {
    return (await this.view(2)).getInt16(0, true);
  }
  async i32(): Promise<number> {
    return (await this.view(4)).getInt32(0, true);
  }
  async i64(): Promise<number> {
    return Number((await this.view(8)).getBigInt64(0, true));
  }
  async f32(): Promise<number> {
    return (await this.view(4)).getFloat32(0, true);
  }
  async f64(): Promise<number> {
    return (await this.view(8)).getFloat64(0, true);
  }

  async string(): Promise<string> {
    const len = await this.u64();
    if (len < 0 || len > MAX_STRING_BYTES) throw new Error(`gguf: implausible string length ${len}`);
    return this.utf8.decode(await this.bytes(len));
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

  // Reads a scalar numeric value normally, or -- when the value is
  // ARRAY-typed -- consumes the whole array and returns its LAST element.
  // Real per-layer hyperparameter arrays (confirmed live on a real Gemma-4
  // GGUF: gemma4.attention.head_count_kv varies 8-per-local-layer/
  // 1-per-global-layer, not the uniform value every other architecture
  // writes as a scalar) can't be discarded the way sliding_window_pattern's
  // per-layer array deliberately is -- unlike that boolean is-local FLAG
  // array, these are the actual per-layer count/dim values
  // shared/vramEstimate.ts's single-scalar-per-model KV formula needs one
  // honest representative of. The array's last element is exactly that
  // representative: llama.cpp's own hybrid-attention layouts (see
  // shared/vramEstimate.ts's swaGlobalLayersInSuffix doc comment) always
  // anchor the global/full-attention layer at the model's LAST transformer
  // block, so index [len-1] is that layer's real, undiscarded value -- not a
  // guess. For a homogeneous (non-SWA) array every element is identical
  // anyway, so this reduces to the same answer a scalar would have given.
  async numericScalarOrArrayLast(type: number): Promise<number | undefined> {
    const all = await this.numericScalarOrArray(type);
    return all.length > 0 ? all[all.length - 1] : undefined;
  }

  // Same read, keeping every element: [v] for a scalar, the whole array
  // otherwise (BOOL elements as 0/1). Feeds KvLayout's per-layer walk.
  async numericScalarOrArray(type: number): Promise<number[]> {
    if (type !== GGUF_TYPE.ARRAY) return [await this.numeric(type)];
    const elemType = await this.u32();
    const len = await this.u64();
    if (len < 0 || len > MAX_ARRAY_LEN) throw new Error(`gguf: implausible array length ${len}`);
    if (len > MAX_PER_LAYER_ARRAY) {
      for (let i = 0; i < len; i++) await this.skip(elemType);
      return [];
    }
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
      if (elemType === GGUF_TYPE.STRING) {
        await this.string(); // not a real case for any numeric hparam key; just don't crash
        continue;
      }
      // numeric() has no BOOL case (nothing scalar needs it) -- read it
      // directly rather than growing that method's contract for this one
      // array-only caller.
      out.push(elemType === GGUF_TYPE.BOOL ? await this.u8() : await this.numeric(elemType));
    }
    return out;
  }

  // Length of an ARRAY value, consuming it (tokenizer.ggml.tokens -> n_vocab).
  async arrayLength(type: number): Promise<number | undefined> {
    if (type !== GGUF_TYPE.ARRAY) {
      await this.skip(type);
      return undefined;
    }
    const elemType = await this.u32();
    const len = await this.u64();
    if (len < 0 || len > MAX_ARRAY_LEN) throw new Error(`gguf: implausible array length ${len}`);
    for (let i = 0; i < len; i++) await this.skip(elemType);
    return len;
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
  //
  // n_head_kv/head_dim_k/head_dim_v/n_head's own header keys can each be
  // written as a full per-layer ARRAY instead of one model-wide scalar
  // (confirmed live on a real Gemma-4 GGUF: attention.head_count_kv varies
  // 8-per-local-layer/1-per-global-layer -- a plain multi-head vs. an
  // extreme-GQA global layer are genuinely different hardware costs, not a
  // conversion quirk). shared/vramEstimate.ts still models the whole KV
  // system with ONE scalar per field, so an ARRAY-typed key here resolves to
  // that array's LAST element -- llama.cpp's own hybrid-attention layouts
  // always anchor the global/full-attention layer at the model's LAST
  // transformer block (see shared/vramEstimate.ts's swaGlobalLayersInSuffix
  // doc comment), so index [len-1] is that layer's real value, the same one
  // key_length/value_length already resolve to when THOSE happen to stay
  // scalar-only (as on the real Gemma-4 file this was confirmed against) --
  // not an arbitrary pick. See GgufReader.numericScalarOrArrayLast.
  n_head_kv: number | null;
  head_dim_k: number | null;
  head_dim_v: number | null;
  n_embd: number | null;
  n_head: number | null;
  // <architecture>.attention.sliding_window -- present only for architectures
  // with sliding-window attention (Mistral/Gemma-style); null means plain
  // full attention, where KV cache grows linearly across the whole context.
  sliding_window: number | null;
  // <architecture>.attention.sliding_window_pattern -- the model's own
  // global:local interleave period (llama.cpp's hparams.set_swa_pattern(N):
  // every Nth layer, counting backward from the model's last block, is
  // global/full-attention, the other N-1 are local/window-bound). Real
  // per-architecture defaults vary meaningfully -- confirmed against
  // llama.cpp's own hparam loaders: gemma2/openai-moe use 2, cohere2/llama4/
  // olmo2 use 4, gemma3 uses 6 -- so shared/vramEstimate.ts's hardcoded
  // "~1-in-6" fallback is only exactly right for Gemma-3-style models. Null
  // when the key is absent, OR when the architecture instead writes this key
  // as a full per-layer array rather than a scalar (confirmed live against a
  // real Gemma-4 GGUF, whose own loader reads it via llama.cpp's
  // get_key_or_arr array overload) -- the scalar-only walk below
  // deliberately skips ARRAY-typed values here, same as it already does for
  // every other numeric field in this file, rather than growing this
  // fixed-shape type a variable-length per-layer array.
  sliding_window_pattern: number | null;
  // <architecture>.attention.shared_kv_layers -- count of TRAILING layers
  // (the model's LAST N transformer blocks) that reuse an earlier layer's KV
  // cache instead of allocating their own -- confirmed against llama.cpp's
  // own gemma3n/gemma4 loader source (src/models/gemma4.cpp:
  // hparams.n_layer_kv_from_start = n_layer_all - shared_kv_layers; layers at
  // index >= n_layer_kv_from_start carry no independent KV allocation at
  // all). Null/0 (the common case, and every non-Gemma3n/4 architecture)
  // means every layer allocates its own KV cache, same as before this field
  // existed.
  shared_kv_layers: number | null;
  // Per-layer KV-cache layout (which layers cache the full context, which
  // only a sliding window, which none) -- see KvLayout and buildKvLayout.
  // Null when the geometry can't be resolved.
  kv_layout: KvLayout | null;
  // The header values kv_layout was built from -- local only, never sent over
  // the wire (ggufMetadataPatch and the worker pick fields by name), so a fit
  // simulation can walk blocks individually via kvLayerWalk.
  kv_inputs?: KvHeaderInputs | null;
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
    sliding_window_pattern: null,
    shared_kv_layers: null,
    kv_layout: null,
    tensor_layer_bytes: null,
    debugReason,
  };
}

// Reads a GGUF file's metadata header to find the model's transformer layer
// count and (if present) its MTP/nextn layer count. Returns nulls on any
// parse failure or unrecognized file rather than throwing -- callers treat a
// missing value as "unknown", same fail-soft posture as the rest of
// hardware/model detection in this worker.
//
// `chunkBytes` is the read-ahead size: small for a local file, large for a
// network source where every read is a round trip.
export async function parseGgufInfo(source: GgufByteSource, chunkBytes = READ_CHUNK_BYTES): Promise<GgufInfo> {
  try {
    const reader = new GgufReader(source, chunkBytes);

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
    const slidingWindowPatterns = new Map<string, number>();
    const sharedKvLayerCounts = new Map<string, number>();
    // Full per-layer arrays (or [scalar]) for buildKvLayout.
    const headCountKvArrays = new Map<string, number[]>();
    const keyLengthArrays = new Map<string, number[]>();
    const valueLengthArrays = new Map<string, number[]>();
    const keyLengthSwa = new Map<string, number>();
    const valueLengthSwa = new Map<string, number>();
    const swaPatternArrays = new Map<string, number[]>();
    const fullAttentionIntervals = new Map<string, number>();
    // Facts that decide whether llama.cpp keeps a per-token cache at all, and
    // where (docs/FIT_VALIDATION_2026-10-14.md): explicit per-layer recurrent
    // flags, and the two inputs of llama_model::create_memory's LFM2 rule.
    const recurrentLayerArrays = new Map<string, number[]>();
    const causalFlags = new Map<string, boolean>();
    const decisionBlockCounts = new Map<string, number>();
    // <arch>.ssm.* -- recurrent state geometry (llama.cpp hparams ssm_d_*).
    const ssmKeys = new Map<string, number>();
    let nVocab: number | undefined;

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
      } else if (key.endsWith(".attention.head_count_kv") && valueType !== GGUF_TYPE.STRING) {
        // ARRAY-typed for some architectures (real Gemma-4 files: one
        // n_head_kv per layer, not a single model-wide value) -- see
        // GgufReader.numericScalarOrArrayLast's doc comment for why reading
        // the array's last element, rather than skipping it like the
        // pre-fix code did, is the correct representative scalar here.
        const arr = await reader.numericScalarOrArray(valueType);
        if (arr.length > 0) {
          headCountKv.set(key, arr[arr.length - 1]);
          headCountKvArrays.set(key, arr);
        }
      } else if (key.endsWith(".attention.key_length") && valueType !== GGUF_TYPE.STRING) {
        const arr = await reader.numericScalarOrArray(valueType);
        if (arr.length > 0) {
          keyLengths.set(key, arr[arr.length - 1]);
          keyLengthArrays.set(key, arr);
        }
      } else if (key.endsWith(".attention.value_length") && valueType !== GGUF_TYPE.STRING) {
        const arr = await reader.numericScalarOrArray(valueType);
        if (arr.length > 0) {
          valueLengths.set(key, arr[arr.length - 1]);
          valueLengthArrays.set(key, arr);
        }
      } else if (key.endsWith(".attention.key_length_swa") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        keyLengthSwa.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.value_length_swa") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        valueLengthSwa.set(key, await reader.numeric(valueType));
      } else if (key === "tokenizer.ggml.tokens") {
        nVocab = await reader.arrayLength(valueType);
      } else if (
        /\.ssm\.(conv_kernel|inner_size|state_size|group_count)$/.test(key) &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        ssmKeys.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".full_attention_interval") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        fullAttentionIntervals.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.recurrent_layers") && valueType === GGUF_TYPE.ARRAY) {
        recurrentLayerArrays.set(key, await reader.numericScalarOrArray(valueType));
      } else if (key.endsWith(".attention.causal") && valueType === GGUF_TYPE.BOOL) {
        causalFlags.set(key, (await reader.u8()) !== 0);
      } else if (key.endsWith(".decision.block_count") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        decisionBlockCounts.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.sliding_window_pattern") && valueType === GGUF_TYPE.ARRAY) {
        // Per-layer is-SWA flags (Gemma-4) -- kept for buildKvLayout only.
        swaPatternArrays.set(key, await reader.numericScalarOrArray(valueType));
      } else if (key.endsWith(".embedding_length") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        embeddingLengths.set(key, await reader.numeric(valueType));
      } else if (key.endsWith(".attention.head_count") && valueType !== GGUF_TYPE.STRING) {
        // NB: this can't collide with .head_count_kv above -- a key ending in
        // ".head_count_kv" doesn't end in ".attention.head_count".
        const v = await reader.numericScalarOrArrayLast(valueType);
        if (v != null) headCounts.set(key, v);
      } else if (key.endsWith(".attention.sliding_window") && valueType !== GGUF_TYPE.ARRAY && valueType !== GGUF_TYPE.STRING) {
        slidingWindows.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".attention.sliding_window_pattern") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        // Array-typed for some architectures (real Gemma-4 files) -- see
        // GgufInfo.sliding_window_pattern's doc comment for why that case is
        // deliberately left unread here rather than handled.
        slidingWindowPatterns.set(key, await reader.numeric(valueType));
      } else if (
        key.endsWith(".attention.shared_kv_layers") &&
        valueType !== GGUF_TYPE.ARRAY &&
        valueType !== GGUF_TYPE.STRING
      ) {
        sharedKvLayerCounts.set(key, await reader.numeric(valueType));
      } else {
        await reader.skip(valueType);
      }
    }
    if (!architecture) return emptyGgufInfo("no general.architecture key found in the whole file");
    const n_layer = blockCounts.get(`${architecture}.block_count`) ?? null;
    // 0 keeps its old meaning (no per-layer data is built for it below).
    if (n_layer !== null && n_layer !== 0 && !isPlausibleLayerCount(n_layer)) {
      return emptyGgufInfo(`implausible ${architecture}.block_count ${n_layer} (max ${MAX_GGUF_LAYERS})`);
    }

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
        const fileSize = await source.size();
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

    const kvInputs: KvHeaderInputs | null =
      n_layer != null && n_layer > 0
        ? {
            architecture,
            nLayer: n_layer,
            nextnLayers: mtpLayerCounts.get(`${architecture}.nextn_predict_layers`) ?? 0,
            headCountKv:
              headCountKvArrays.get(`${architecture}.attention.head_count_kv`) ??
              (headCounts.has(`${architecture}.attention.head_count`)
                ? [headCounts.get(`${architecture}.attention.head_count`)!]
                : null),
            keyLength: keyLengthArrays.get(`${architecture}.attention.key_length`) ?? null,
            valueLength: valueLengthArrays.get(`${architecture}.attention.value_length`) ?? null,
            keyLengthSwa: keyLengthSwa.get(`${architecture}.attention.key_length_swa`) ?? null,
            valueLengthSwa: valueLengthSwa.get(`${architecture}.attention.value_length_swa`) ?? null,
            nEmbd: embeddingLengths.get(`${architecture}.embedding_length`) ?? null,
            nHead: headCounts.get(`${architecture}.attention.head_count`) ?? null,
            slidingWindow: slidingWindows.get(`${architecture}.attention.sliding_window`) ?? null,
            swaPatternScalar: slidingWindowPatterns.get(`${architecture}.attention.sliding_window_pattern`) ?? null,
            swaPatternArray: swaPatternArrays.get(`${architecture}.attention.sliding_window_pattern`) ?? null,
            sharedKvLayers: sharedKvLayerCounts.get(`${architecture}.attention.shared_kv_layers`) ?? 0,
            fullAttentionInterval: fullAttentionIntervals.get(`${architecture}.full_attention_interval`) ?? null,
            recurrentLayers: recurrentLayerArrays.get(`${architecture}.attention.recurrent_layers`) ?? null,
            noCache:
              NO_MEMORY_ARCHES.has(architecture) ||
              (architecture === "lfm2" &&
                causalFlags.get(`${architecture}.attention.causal`) === false &&
                ((decisionBlockCounts.get(`${architecture}.decision.block_count`) ?? blockCounts.get(`${architecture}.decision.block_count`)) ?? 0) > 0),
            ssmConvKernel: ssmKeys.get(`${architecture}.ssm.conv_kernel`) ?? null,
            ssmInnerSize: ssmKeys.get(`${architecture}.ssm.inner_size`) ?? null,
            ssmStateSize: ssmKeys.get(`${architecture}.ssm.state_size`) ?? null,
            ssmGroupCount: ssmKeys.get(`${architecture}.ssm.group_count`) ?? null,
            vocab: nVocab ?? null,
          }
        : null;

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
      sliding_window_pattern:
        slidingWindowPatterns.get(`${architecture}.attention.sliding_window_pattern`) ?? null,
      shared_kv_layers: sharedKvLayerCounts.get(`${architecture}.attention.shared_kv_layers`) ?? null,
      kv_layout: kvInputs ? buildKvLayout(kvInputs) : null,
      kv_inputs: kvInputs,
      tensor_layer_bytes,
      debugReason:
        n_layer === null ? `architecture "${architecture}" found but no ${architecture}.block_count key` : undefined,
    };
  } catch (err) {
    return emptyGgufInfo(`parse threw: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// Hybrid attention/recurrent architectures whose llama.cpp loader marks
// layer i recurrent unless (i + 1) % full_attention_interval == 0, defaulting
// the interval to 4 when the key is absent (src/llama-model.cpp).
const INTERVAL_HYBRID_ARCHES = new Set(["qwen3next", "qwen35", "qwen35moe"]);

// Per-layer walk mirroring llama.cpp's own KV-cache allocation:
// - a layer with n_head_kv == 0, or marked recurrent by the interval rule,
//   has no KV cache (recurrent state is small and fixed; not counted);
// - layers at index >= n_layer - shared_kv_layers reuse an earlier layer's
//   cache (Gemma-3n/4);
// - NextN/MTP blocks counted in block_count get no KV in normal runs;
// - SWA layers come from the per-layer flag array, else the scalar period
//   (llama.cpp set_swa_pattern: il % N < N - 1 is SWA), else -- when only a
//   window is declared -- the period-6 fallback the rest of the estimator
//   uses.
// Null when head counts or head dims can't be resolved.
export interface KvHeaderInputs {
  architecture: string;
  nLayer: number;
  nextnLayers: number;
  headCountKv: number[] | null;
  keyLength: number[] | null;
  valueLength: number[] | null;
  keyLengthSwa: number | null;
  valueLengthSwa: number | null;
  nEmbd: number | null;
  nHead: number | null;
  slidingWindow: number | null;
  swaPatternScalar: number | null;
  swaPatternArray: number[] | null;
  sharedKvLayers: number;
  /** <arch>.attention.recurrent_layers: per-layer flags, 1 = recurrent.
   * Overrides full_attention_interval when present (llama.cpp does). */
  recurrentLayers?: number[] | null;
  /** llama.cpp allocates no KV and no recurrent state for this model
   * (NO_MEMORY_ARCHES, or a non-causal LFM2 with decision layers). */
  noCache?: boolean;
  fullAttentionInterval: number | null;
  ssmConvKernel?: number | null;
  ssmInnerSize?: number | null;
  ssmStateSize?: number | null;
  ssmGroupCount?: number | null;
  vocab?: number | null;
}

// What one transformer block holds in the context cache: "full" and "swa"
// layers cache kElems + vElems values per token (the window for swa),
// "recurrent" layers a fixed stateBytes, "none" layers (shared-KV or MTP
// blocks) nothing.
export interface KvLayerEntry {
  kind: "full" | "swa" | "recurrent" | "none";
  kElems: number;
  vElems: number;
  stateBytes: number;
}

// Per-block walk behind buildKvLayout, kept separate so a fit simulation can
// ask which blocks' caches land on the GPU. Null when head counts or head
// dims can't be resolved.
export function kvLayerWalk(h: KvHeaderInputs): KvLayerEntry[] | null {
  if (!h.headCountKv || h.headCountKv.length === 0) return null;
  if (!isPlausibleLayerCount(h.nLayer)) return null;
  const at = (arr: number[] | null, i: number): number | null =>
    arr == null || arr.length === 0 ? null : arr.length === 1 ? arr[0] : (arr[i] ?? null);
  const fallbackDim = h.nEmbd != null && h.nHead != null && h.nHead > 0 ? h.nEmbd / h.nHead : null;
  const window = h.slidingWindow != null && h.slidingWindow > 0 ? h.slidingWindow : 0;
  const interval =
    h.fullAttentionInterval != null && h.fullAttentionInterval > 0
      ? h.fullAttentionInterval
      : INTERVAL_HYBRID_ARCHES.has(h.architecture)
        ? 4
        : null;
  const kvLayers = Math.max(0, h.nLayer - Math.max(0, h.nextnLayers) - Math.max(0, h.sharedKvLayers));
  // Per recurrent layer, f32 (llama.cpp hparams n_embd_r / n_embd_s):
  //   R = (d_conv - 1) * (d_inner + 2 * n_group * d_state), S = d_state * d_inner
  const dInner = h.ssmInnerSize ?? 0;
  const dState = h.ssmStateSize ?? 0;
  const perRecurrentLayer =
    dInner > 0 && dState > 0
      ? (Math.max(0, (h.ssmConvKernel ?? 0) - 1) * (dInner + 2 * (h.ssmGroupCount ?? 0) * dState) + dState * dInner) * 4
      : 0;
  const out: KvLayerEntry[] = [];
  for (let il = 0; il < h.nLayer; il++) {
    if (il >= kvLayers) {
      out.push({ kind: "none", kElems: 0, vElems: 0, stateBytes: 0 });
      continue;
    }
    const heads = at(h.headCountKv, il);
    const flagged = h.recurrentLayers && h.recurrentLayers.length > il ? h.recurrentLayers[il] !== 0 : null;
    const recurrent = (flagged ?? (interval != null && (il + 1) % interval !== 0)) || heads === 0;
    if (recurrent) {
      out.push({ kind: "recurrent", kElems: 0, vElems: 0, stateBytes: perRecurrentLayer });
      continue;
    }
    if (heads == null) return null;
    let isSwa = false;
    if (window > 0) {
      if (h.swaPatternArray && h.swaPatternArray.length > 0) isSwa = (h.swaPatternArray[il] ?? 0) !== 0;
      else {
        const period = h.swaPatternScalar != null && h.swaPatternScalar > 0 ? h.swaPatternScalar : 6;
        isSwa = il % period < period - 1;
      }
    }
    const dK = (isSwa ? h.keyLengthSwa : null) ?? at(h.keyLength, il) ?? fallbackDim;
    const dV = (isSwa ? h.valueLengthSwa : null) ?? at(h.valueLength, il) ?? dK;
    if (dK == null || dV == null) return null;
    out.push({ kind: isSwa ? "swa" : "full", kElems: heads * dK, vElems: heads * dV, stateBytes: 0 });
  }
  return out;
}

export function buildKvLayout(h: KvHeaderInputs): KvLayout | null {
  const walk = kvLayerWalk(h);
  if (!walk) return null;
  const window = h.slidingWindow != null && h.slidingWindow > 0 ? h.slidingWindow : 0;
  const out: KvLayout = { full_k: 0, full_v: 0, swa_k: 0, swa_v: 0, window };
  let stateBytes = 0;
  let fullMax = 0;
  for (const e of walk) {
    if (e.kind === "full") {
      out.full_k += e.kElems;
      out.full_v += e.vElems;
      fullMax = Math.max(fullMax, e.kElems + e.vElems);
    } else if (e.kind === "swa") {
      out.swa_k += e.kElems;
      out.swa_v += e.vElems;
    } else stateBytes += e.stateBytes;
  }
  if (out.swa_k === 0) out.window = 0;
  if (stateBytes > 0) out.state_bytes = stateBytes;
  if (h.vocab != null && h.vocab > 0) out.vocab = h.vocab;
  if (h.noCache) out.no_cache = true;
  out.kinds = walk.map((e) => (e.kind === "full" ? "f" : e.kind === "swa" ? "s" : e.kind === "recurrent" ? "r" : "n")).join("");
  out.full_max = fullMax;
  return out;
}

// Architectures llama_model::create_memory gives no memory at all (b11609,
// src/llama-model.cpp): encoders, embedding models, diffusion LMs and the
// Clef classifier. Pooling or non-causal attention alone does not mean this --
// Qwen3-VL-Embedding and the LFM2.5 encoders keep a KV cache.
export const NO_MEMORY_ARCHES = new Set([
  "bert", "jina-bert-v2", "jina-bert-v3", "nomic-bert", "nomic-bert-moe", "neo-bert", "eurobert",
  "wavtokenizer-dec", "modern-bert", "gemma-embedding", "gemma-embedding2",
  "dream", "llada", "llada-moe", "rnd1", "clef",
]);

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
//
// Covers the same routed-expert names llama.cpp's own fit uses
// (ffn_(up|down|gate_up|gate)_(ch|)exps): the fused gate+up stack some
// architectures write (gemma-4 26B-A4B's "ffn_gate_up_exps") and the
// "_chexps" variant. Shared experts ("_shexp") and the router
// ("ffn_gate_inp") are dense -- used on every token -- and stay out.
const MOE_EXPERT_TENSOR = /^ffn_(?:gate_up|gate|up|down)(_exps|_chexps)?(?:\.\d+)?\.(?:weight|bias|scale)$/;

function isMoeExpertTensorName(rest: string): boolean {
  const m = MOE_EXPERT_TENSOR.exec(rest);
  if (!m) return false;
  return m[1] === "_exps" || m[1] === "_chexps" || /^ffn_(?:gate_up|gate|up|down)\.\d+\./.test(rest);
}

// Buckets every tensor's real on-disk byte size (see readGgufInfo's caller
// above) into the shape shared/vramEstimate.ts's placeWeightBytes consumes.
// A block index found in a tensor name but outside [0, nLayer) (a malformed
// or non-standard file) falls back to the "other" bucket rather than
// crashing or silently growing the per-block arrays past the real layer
// count.
export function buildTensorLayerBreakdown(tensors: { name: string; size: number }[], nLayer: number): TensorLayerBreakdown {
  if (!isPlausibleLayerCount(nLayer)) throw new RangeError(`implausible layer count ${nLayer}`);
  const dense = new Array(nLayer).fill(0);
  const moe = new Array(nLayer).fill(0);
  let embed = 0;
  let output = 0;
  let other = 0;
  let maxDense = 0;
  let maxMoe = 0;
  let maxEmbed = 0;
  let maxOutput = 0;

  for (const t of tensors) {
    const blockMatch = BLOCK_TENSOR.exec(t.name);
    if (blockMatch) {
      const idx = Number(blockMatch[1]);
      if (Number.isInteger(idx) && idx >= 0 && idx < nLayer) {
        if (isMoeExpertTensorName(blockMatch[2])) {
          moe[idx] += t.size;
          maxMoe = Math.max(maxMoe, t.size);
        } else {
          dense[idx] += t.size;
          maxDense = Math.max(maxDense, t.size);
        }
        continue;
      }
      other += t.size;
      continue;
    }
    if (t.name.startsWith("token_embd.")) {
      embed += t.size;
      maxEmbed = Math.max(maxEmbed, t.size);
    } else if (t.name.startsWith("output_norm.") || t.name.startsWith("output.")) {
      output += t.size;
      maxOutput = Math.max(maxOutput, t.size);
    } else {
      other += t.size;
    }
  }

  return { dense, moe, embed, output, other, max_dense: maxDense, max_moe: maxMoe, max_embed: maxEmbed, max_output: maxOutput };
}
