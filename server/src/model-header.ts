import type { ModelDirFile, ModelMetadata } from "../../shared/types.js";
import type { GgufInfo } from "../../shared/gguf.js";

// Maps a verified model file's heartbeat-reported GGUF header fields onto the
// ModelMetadata shape (see ModelDirFile / worker/src/gguf.ts). Only fields
// with a real value are included, so calling it with a file whose header
// couldn't be parsed produces an empty patch (a no-op for the caller).
// Also used for the server's own parse of a Hugging Face copy (hf-header.ts).
export function ggufMetadataPatch(f: ModelDirFile | GgufInfo): Partial<ModelMetadata> {
  const patch: Partial<ModelMetadata> = {};
  if (typeof f.n_layer === "number") patch.n_layer = f.n_layer;
  if (typeof f.mtp_layers === "number" && f.mtp_layers > 0) patch.mtp_layers = f.mtp_layers;
  if (typeof f.param_count === "number" && f.param_count > 0) patch.param_count = f.param_count;
  if (typeof f.quant === "string" && f.quant) patch.quant = f.quant;
  // Trained context + KV geometry -- adopted like the fields above so a
  // hand-dropped file gets the same catalog metadata as an in-app download.
  if (typeof f.trained_ctx === "number" && f.trained_ctx > 0) patch.trained_ctx = f.trained_ctx;
  if (typeof f.n_head_kv === "number" && f.n_head_kv > 0) patch.n_head_kv = f.n_head_kv;
  if (typeof f.head_dim_k === "number" && f.head_dim_k > 0) patch.head_dim_k = f.head_dim_k;
  if (typeof f.head_dim_v === "number" && f.head_dim_v > 0) patch.head_dim_v = f.head_dim_v;
  if (typeof f.n_embd === "number" && f.n_embd > 0) patch.n_embd = f.n_embd;
  if (typeof f.n_head === "number" && f.n_head > 0) patch.n_head = f.n_head;
  if (typeof f.sliding_window === "number" && f.sliding_window > 0) patch.sliding_window = f.sliding_window;
  if (f.tensor_layer_bytes) patch.tensor_layer_bytes = f.tensor_layer_bytes;
  return patch;
}

// Every field ggufMetadataPatch can produce.
const HEADER_KEYS = [
  "n_layer",
  "mtp_layers",
  "param_count",
  "quant",
  "trained_ctx",
  "n_head_kv",
  "head_dim_k",
  "head_dim_v",
  "n_embd",
  "n_head",
  "sliding_window",
  "tensor_layer_bytes",
] as const satisfies readonly (keyof ModelMetadata)[];

// Puts a resolved header reading onto a stored metadata object by REPLACING
// every header field, not merging: a key the resolved reading doesn't have
// (e.g. mtp_layers on a model with no MTP head) must not survive from an
// earlier, possibly false report. param_count is the one exception -- it can
// also come from HF's repo-level API (a legitimate fallback), so it's only
// replaced when the reading carries one. An empty reading changes nothing.
export function withHeaderFields(metadata: ModelMetadata, header: Partial<ModelMetadata>): ModelMetadata {
  if (Object.keys(header).length === 0) return metadata;
  const out: ModelMetadata = { ...metadata };
  for (const key of HEADER_KEYS) {
    if (key === "param_count" && header.param_count === undefined) continue;
    delete out[key];
  }
  return { ...out, ...header };
}
