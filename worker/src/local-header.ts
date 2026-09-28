import type { Model, ModelMetadata } from "../../shared/types.js";
import { readGgufInfo, type GgufInfo } from "./gguf.js";
import { log } from "./log.js";

// The header fields a job's decisions depend on (ngl ceiling, VRAM fit,
// context ceiling, KV sizing). A job arrives with the server's copy of
// these in model.metadata -- a shared record other tenants' workers feed
// (docs/PLAN_MODEL_METADATA_TRUST.md) -- but this machine has the actual file,
// so its own read of that file's header is the only thing a benchmark here
// should trust.
const HEADER_FIELDS = [
  "n_layer",
  "mtp_layers",
  "quant",
  "param_count",
  "trained_ctx",
  "n_head_kv",
  "head_dim_k",
  "head_dim_v",
  "n_embd",
  "n_head",
  "sliding_window",
  "tensor_layer_bytes",
] as const satisfies readonly (keyof GgufInfo & keyof ModelMetadata)[];

// Compared for the mismatch warning -- the ones that actually drive a
// decision. param_count/quant are display-only and legitimately differ on
// legacy rows (e.g. HF's repo-level param total), so they'd only be noise.
const DECISION_FIELDS: readonly (typeof HEADER_FIELDS)[number][] = [
  "n_layer",
  "trained_ctx",
  "n_head_kv",
  "head_dim_k",
  "head_dim_v",
  "sliding_window",
  "tensor_layer_bytes",
];

export type HeaderReader = (path: string) => Promise<GgufInfo>;

// Returns a copy of `model` whose header fields come from the file at
// `path` (read now, straight from disk -- no cache, so a file replaced in
// place can never be described by stale facts). Fields the local read
// couldn't produce keep the server's value; a failed read keeps them all.
export async function applyLocalHeader(
  model: Model,
  path: string,
  read: HeaderReader = readGgufInfo
): Promise<Model> {
  let info: GgufInfo;
  try {
    info = await read(path);
  } catch (err) {
    log.warn(
      `[local-header] couldn't read ${path}'s header, using the server's metadata: ${err instanceof Error ? err.message : String(err)}`
    );
    return model;
  }

  const metadata: ModelMetadata = { ...model.metadata };
  const differs: string[] = [];
  for (const key of HEADER_FIELDS) {
    const local = info[key];
    if (local === null || local === undefined) continue;
    const server = model.metadata[key];
    if (
      DECISION_FIELDS.includes(key) &&
      server !== undefined &&
      server !== null &&
      JSON.stringify(server) !== JSON.stringify(local)
    ) {
      differs.push(key === "tensor_layer_bytes" ? key : `${key} ${JSON.stringify(server)}→${JSON.stringify(local)}`);
    }
    (metadata as Record<string, unknown>)[key] = local;
  }
  if (differs.length > 0) {
    log.warn(`[local-header] server metadata for ${model.id} disagrees with the local file (using local): ${differs.join(", ")}`);
  }
  return { ...model, metadata };
}
