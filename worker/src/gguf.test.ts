import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readGgufInfo, quantLabelFromFileType } from "./gguf.js";

// Minimal GGUF byte-level writer -- just enough to exercise readGgufInfo's
// own reader, independent of any real model file. Type ids mirror gguf.ts's
// own GGUF_TYPE map (not exported, so duplicated here deliberately).
const T = { UINT32: 4, STRING: 8, ARRAY: 9 } as const;

// An ARRAY-typed value -- real Gemma-4 GGUFs write
// gemma4.attention.sliding_window_pattern this way (one uint32 per layer)
// rather than as a scalar period, confirmed against a real file. elemType is
// always UINT32 here since that's the only array element type this test file
// needs to exercise.
type ArrayValue = { arr: number[] };
type Kv = [key: string, type: number, value: number | string | ArrayValue];

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
  else if (type === T.ARRAY) {
    const { arr } = value as ArrayValue;
    parts.push(u32(T.UINT32), u64(arr.length), ...arr.map(u32));
  } else throw new Error(`test helper doesn't support type ${type}`);
  return Buffer.concat(parts);
}

// tensorCount is irrelevant to readGgufInfo's KV parsing (consumed only to
// advance the cursor); the tensor_info walk below uses it to know how many
// entries to read, and the test builder writes tensor infos only when the
// caller supplies them.
function buildGguf(
  kvs: Kv[],
  tensors: { name: string; dims: number[]; type?: number; offset?: number }[] = []
): Buffer {
  const tensorInfos = tensors.map((t) =>
    Buffer.concat([
      ggufString(t.name),
      u32(t.dims.length),
      ...t.dims.map(u64),
      u32(t.type ?? 0),
      u64(t.offset ?? 0),
    ])
  );
  return Buffer.concat([
    u32(0x46554747), // magic "GGUF"
    u32(3), // version
    u64(tensors.length), // tensor_count
    u64(kvs.length), // kv_count
    ...kvs.map(kvBuffer),
    ...tensorInfos,
  ]);
}

const tmpDir = mkdtempSync(join(tmpdir(), "llamatoaster-gguf-test-"));
let counter = 0;
function writeTempGguf(
  kvs: Kv[],
  tensors: { name: string; dims: number[]; type?: number; offset?: number }[] = []
): string {
  const path = join(tmpDir, `test-${counter++}.gguf`);
  writeFileSync(path, buildGguf(kvs, tensors));
  return path;
}

describe("readGgufInfo", () => {
  it("resolves n_layer for a normal, well-ordered file", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen3moe"],
      ["qwen3moe.block_count", T.UINT32, 48],
      ["tokenizer.ggml.model", T.STRING, "gpt2"],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_layer).toBe(48);
    expect(info.debugReason).toBeUndefined();
  });

  // Trained context + KV geometry -- resolved via the same suffix-matching
  // walk as block_count, keyed off the file's own architecture,
  // so keys that appear before general.architecture are still found.
  it("resolves trained context and KV-cache geometry from architecture-prefixed keys", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen3"],
      ["qwen3.block_count", T.UINT32, 36],
      ["qwen3.context_length", T.UINT32, 40960],
      ["qwen3.attention.head_count", T.UINT32, 32],
      ["qwen3.attention.head_count_kv", T.UINT32, 8],
      ["qwen3.attention.key_length", T.UINT32, 128],
      ["qwen3.attention.value_length", T.UINT32, 128],
      ["qwen3.embedding_length", T.UINT32, 2560],
      ["tokenizer.ggml.model", T.STRING, "gpt2"],
    ]);
    const info = await readGgufInfo(path);
    expect(info.trained_ctx).toBe(40960);
    expect(info.n_head).toBe(32);
    expect(info.n_head_kv).toBe(8);
    expect(info.head_dim_k).toBe(128);
    expect(info.head_dim_v).toBe(128);
    expect(info.n_embd).toBe(2560);
    // Absence paths -- an architecture without sliding-window attention.
    expect(info.sliding_window).toBeNull();
    expect(info.n_layer).toBe(36);
  });

  // attention.head_count_kv is optional in the GGUF spec -- a non-GQA model
  // (KV heads == attention heads) commonly omits it entirely, and llama.cpp's
  // own loader defaults it to head_count in that case. readGgufInfo must do
  // the same, or every such model reads as "unknown KV geometry" despite the
  // value being fully derivable.
  it("defaults n_head_kv to head_count when head_count_kv is absent (non-GQA model)", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gpt2"],
      ["gpt2.block_count", T.UINT32, 24],
      ["gpt2.attention.head_count", T.UINT32, 16],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head).toBe(16);
    expect(info.n_head_kv).toBe(16);
  });

  it("prefers an explicit head_count_kv over the head_count default when both are present", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen3"],
      ["qwen3.block_count", T.UINT32, 36],
      ["qwen3.attention.head_count", T.UINT32, 32],
      ["qwen3.attention.head_count_kv", T.UINT32, 8],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head_kv).toBe(8);
  });

  it("reads sliding_window only for architectures that declare one", async () => {
    const swPath = writeTempGguf([
      ["general.architecture", T.STRING, "mistral"],
      ["mistral.block_count", T.UINT32, 48],
      ["mistral.context_length", T.UINT32, 131072],
      ["mistral.attention.sliding_window", T.UINT32, 4096],
    ]);
    const swInfo = await readGgufInfo(swPath);
    expect(swInfo.trained_ctx).toBe(131072);
    expect(swInfo.sliding_window).toBe(4096);

    const plainPath = writeTempGguf([
      ["general.architecture", T.STRING, "llama"],
      ["llama.block_count", T.UINT32, 32],
      ["llama.context_length", T.UINT32, 8192],
    ]);
    const plainInfo = await readGgufInfo(plainPath);
    expect(plainInfo.trained_ctx).toBe(8192);
    expect(plainInfo.sliding_window).toBeNull();
  });

  // Real per-architecture periods vary (confirmed against llama.cpp's own
  // hparam loaders): gemma2/openai-moe default to 2, cohere2/llama4/olmo2 to
  // 4, gemma3 to 6 -- a non-6 value here is exactly the case
  // shared/vramEstimate.ts's hardcoded "~1-in-6" fallback gets wrong.
  it("reads sliding_window_pattern and shared_kv_layers when the header declares them as scalars", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "cohere2"],
      ["cohere2.block_count", T.UINT32, 32],
      ["cohere2.attention.sliding_window", T.UINT32, 4096],
      ["cohere2.attention.sliding_window_pattern", T.UINT32, 4],
      ["cohere2.attention.shared_kv_layers", T.UINT32, 2],
    ]);
    const info = await readGgufInfo(path);
    expect(info.sliding_window_pattern).toBe(4);
    expect(info.shared_kv_layers).toBe(2);
  });

  // Confirmed live against a real gemma-4-12b GGUF: its own
  // gemma4.attention.sliding_window_pattern key is written as a full
  // per-layer ARRAY (llama.cpp's gemma4 loader reads it via get_key_or_arr's
  // array overload), not the scalar period other SWA architectures use. The
  // scalar-only walk must skip it -- same convention as every other numeric
  // field in this file -- and fall back to null rather than misreading the
  // array's length/first element as a period, or throwing.
  it("leaves sliding_window_pattern null (not misread) when the header writes it as a per-layer array", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gemma4"],
      ["gemma4.block_count", T.UINT32, 6],
      ["gemma4.attention.sliding_window", T.UINT32, 1024],
      ["gemma4.attention.sliding_window_pattern", T.ARRAY, { arr: [1, 1, 1, 1, 1, 0] }],
    ]);
    const info = await readGgufInfo(path);
    expect(info.sliding_window).toBe(1024);
    expect(info.sliding_window_pattern).toBeNull();
    expect(info.n_layer).toBe(6);
  });

  // Confirmed live against the same real gemma-4-12b GGUF as the test above:
  // unlike sliding_window_pattern (a per-layer boolean is-local FLAG, wrong
  // to misread as a scalar period), attention.head_count_kv's real per-layer
  // array holds genuine per-layer KV-head counts -- 8 for each local/
  // sliding-window layer, 1 for the global/full-attention layer every 6th
  // block (index 5 of every group of 6, matching sliding_window_pattern's
  // own period). Before this fix, the ARRAY guard skipped it entirely and
  // n_head_kv silently fell back to head_count (16 here) -- a ~16x
  // overestimate of the real global-layer value that, compounded across
  // shared/vramEstimate.ts's context-scaling KV formula, was the dominant
  // cause of a reported real-world VRAM estimate growing from ~13.6 GiB to
  // ~78.9 GiB as candidate context size grew, while the real measured peak
  // stayed flat around 7.5-7.7 GiB. The array's LAST element is read as the
  // representative value -- llama.cpp's own hybrid-attention layouts always
  // anchor the global/full-attention layer at the model's last transformer
  // block (see shared/vramEstimate.ts's swaGlobalLayersInSuffix), so index
  // [len-1] is that layer's real value, consistent with key_length/
  // value_length (512 here) already resolving to that same global layer's
  // dims when those keys stay scalar-only, as they do on the real file.
  it("resolves n_head_kv from a per-layer ARRAY as the array's last element instead of silently falling back to head_count (real Gemma-4-style head_count_kv)", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "gemma4"],
      ["gemma4.block_count", T.UINT32, 6],
      ["gemma4.attention.head_count", T.UINT32, 16],
      ["gemma4.attention.head_count_kv", T.ARRAY, { arr: [8, 8, 8, 8, 8, 1] }],
      ["gemma4.attention.key_length", T.UINT32, 512],
      ["gemma4.attention.value_length", T.UINT32, 512],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head_kv).toBe(1);
    expect(info.n_head).toBe(16);
    expect(info.head_dim_k).toBe(512);
    expect(info.head_dim_v).toBe(512);
  });

  // Same array-handling extended to head_count/key_length/value_length --
  // not observed array-typed on the real Gemma-4 file (those three stayed
  // scalar there), but the ARRAY guard was symmetric across all four
  // attention.* fields before this fix, so any architecture that DOES write
  // one of these as a per-layer array must resolve it the same
  // last-element way, not silently drop it.
  it("resolves head_count/key_length/value_length from a per-layer ARRAY the same way, for an architecture that writes those as arrays too", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "hypothetical-arch"],
      ["hypothetical-arch.block_count", T.UINT32, 4],
      ["hypothetical-arch.attention.head_count", T.ARRAY, { arr: [32, 32, 32, 16] }],
      ["hypothetical-arch.attention.head_count_kv", T.ARRAY, { arr: [8, 8, 8, 2] }],
      ["hypothetical-arch.attention.key_length", T.ARRAY, { arr: [128, 128, 128, 64] }],
      ["hypothetical-arch.attention.value_length", T.ARRAY, { arr: [128, 128, 128, 64] }],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head).toBe(16);
    expect(info.n_head_kv).toBe(2);
    expect(info.head_dim_k).toBe(64);
    expect(info.head_dim_v).toBe(64);
  });

  // A homogeneous per-layer array (every element identical) must resolve to
  // that same value -- the array-vs-scalar distinction shouldn't change the
  // answer when there's nothing heterogeneous to lose.
  it("resolves n_head_kv from a homogeneous per-layer ARRAY as that single repeated value", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "uniform-arr-arch"],
      ["uniform-arr-arch.block_count", T.UINT32, 3],
      ["uniform-arr-arch.attention.head_count_kv", T.ARRAY, { arr: [4, 4, 4] }],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head_kv).toBe(4);
  });

  // Regression guard for the common case (the vast majority of real
  // architectures): a plain SCALAR head_count_kv must resolve exactly as
  // before -- the array-handling addition must not perturb the scalar path
  // at all. Duplicates part of "prefers an explicit head_count_kv..." above
  // deliberately, as an explicit before/after marker for this fix.
  it("still resolves n_head_kv directly from a plain SCALAR head_count_kv, unaffected by the array-handling addition", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen3"],
      ["qwen3.block_count", T.UINT32, 36],
      ["qwen3.attention.head_count", T.UINT32, 32],
      ["qwen3.attention.head_count_kv", T.UINT32, 8],
    ]);
    const info = await readGgufInfo(path);
    expect(info.n_head_kv).toBe(8);
  });

  // Same architecture-prefix discipline as block_count's map: a header whose
  // hyperparameters precede general.architecture must still resolve them.
  it("resolves context/KV keys written before the architecture key", async () => {
    const path = writeTempGguf([
      ["llama.context_length", T.UINT32, 32768],
      ["llama.attention.head_count_kv", T.UINT32, 4],
      ["llama.attention.head_count", T.UINT32, 28],
      ["general.architecture", T.STRING, "llama"],
      ["llama.block_count", T.UINT32, 28],
    ]);
    const info = await readGgufInfo(path);
    expect(info.trained_ctx).toBe(32768);
    expect(info.n_head_kv).toBe(4);
    expect(info.n_head).toBe(28);
  });

  it("leaves trained context / KV fields null when their header keys are absent", async () => {
    const path = writeTempGguf([
      ["general.architecture", T.STRING, "qwen35"],
      ["qwen35.block_count", T.UINT32, 12],
    ]);
    const info = await readGgufInfo(path);
    expect(info.trained_ctx).toBeNull();
    expect(info.n_head_kv).toBeNull();
    expect(info.n_head).toBeNull();
    expect(info.head_dim_k).toBeNull();
    expect(info.head_dim_v).toBeNull();
    expect(info.n_embd).toBeNull();
    expect(info.sliding_window).toBeNull();
    expect(info.sliding_window_pattern).toBeNull();
    expect(info.shared_kv_layers).toBeNull();
    expect(info.n_layer).toBe(12);
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

  // The tensor_info section directly follows the KV metadata (there is NO
  // tensor_info_count field -- reading one as an earlier implementation did
  // misaligned the whole walk, consuming the first tensor's name-length as a
  // count). param_count is the sum of product(dims) over every tensor, and
  // must be per-file rather than the repo-level guess HF's own API reports.
  it("computes param_count as the total element count across every tensor", async () => {
    const path = writeTempGguf(
      [
        ["general.architecture", T.STRING, "qwen35"],
        ["general.file_type", T.UINT32, 12], // LLAMA_FTYPE_MOSTLY_Q3_K_M
        ["qwen35.block_count", T.UINT32, 32],
      ],
      [
        { name: "token_embd.weight", dims: [16, 1000] }, // 16_000
        { name: "blk.0.attn_q.weight", dims: [16, 16] }, // 256
        { name: "output.weight", dims: [16, 1000] }, // 16_000
      ]
    );
    const info = await readGgufInfo(path);
    expect(info.param_count).toBe(16_000 + 256 + 16_000);
    expect(info.quant).toBe("Q3_K_M");
    expect(info.n_layer).toBe(32);
  });

  // tensor_layer_bytes must come from each tensor's REAL on-disk byte span
  // (offset deltas + final file size), not from dims/param-count -- the whole
  // point is it stays correct across quantization formats without a
  // per-type size table. general.alignment is pinned to 1 here purely to
  // keep the expected offsets/sizes in this test arithmetic-free; real files
  // default to 32 (exercised implicitly by every other test in this file,
  // which never sets it).
  it("computes tensor_layer_bytes from real per-tensor byte offsets, independent of dims/quant", async () => {
    const tensors = [
      { name: "token_embd.weight", dims: [1], offset: 0, size: 100 }, // embed
      { name: "blk.0.attn_q.weight", dims: [1], offset: 100, size: 50 }, // dense, layer 0
      { name: "blk.0.ffn_gate_exps.weight", dims: [1], offset: 150, size: 300 }, // moe, layer 0 (modern stacked convention)
      { name: "blk.1.attn_q.weight", dims: [1], offset: 450, size: 50 }, // dense, layer 1
      { name: "blk.1.ffn_gate.2.weight", dims: [1], offset: 500, size: 70 }, // moe, layer 1 (legacy per-expert convention)
      { name: "output_norm.weight", dims: [1], offset: 570, size: 10 }, // output
      { name: "output.weight", dims: [1], offset: 580, size: 90 }, // output
      { name: "rope_freqs.weight", dims: [1], offset: 670, size: 20 }, // other
    ];
    const totalDataBytes = tensors.reduce((sum, t) => sum + t.size, 0);
    const header = buildGguf(
      [
        ["general.architecture", T.STRING, "qwen3moe"],
        ["general.alignment", T.UINT32, 1],
        ["qwen3moe.block_count", T.UINT32, 2],
      ],
      tensors.map(({ name, dims, offset }) => ({ name, dims, offset }))
    );
    const path = join(tmpDir, `test-tensor-layer-bytes-${counter++}.gguf`);
    writeFileSync(path, Buffer.concat([header, Buffer.alloc(totalDataBytes)]));

    const info = await readGgufInfo(path);
    expect(info.tensor_layer_bytes).toEqual({
      dense: [50, 50],
      moe: [300, 70],
      embed: 100,
      output: 100,
      other: 20,
    });
  });

  it("leaves tensor_layer_bytes null when n_layer is unresolved", async () => {
    const path = writeTempGguf(
      [["general.architecture", T.STRING, "mystery-arch"]],
      [{ name: "token_embd.weight", dims: [16, 1000] }]
    );
    const info = await readGgufInfo(path);
    expect(info.tensor_layer_bytes).toBeNull();
  });

  it("returns null param_count when the tensor_info walk runs off the end of the file", async () => {
    // tensor_count advertises 2 tensors but only 1 is written -- the walk
    // throws on the second, and must leave the already-resolved fields intact
    // rather than wiping them (param_count is best-effort, isolated from the
    // KV parsing).
    const path = writeTempGguf(
      [
        ["general.architecture", T.STRING, "qwen35"],
        ["qwen35.block_count", T.UINT32, 32],
      ],
      [{ name: "token_embd.weight", dims: [16, 1000] }]
    );
    // Overwrite tensor_count to 2 in the header (bytes 8..16).
    const data = Buffer.from(await import("node:fs/promises").then((m) => m.readFile(path)));
    u64(2).copy(data, 8);
    await import("node:fs/promises").then((m) => m.writeFile(path, data));

    const info = await readGgufInfo(path);
    expect(info.param_count).toBeNull();
    expect(info.n_layer).toBe(32);
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
