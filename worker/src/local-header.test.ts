import { describe, expect, it } from "vitest";
import { applyLocalHeader } from "./local-header.js";
import type { GgufInfo } from "./gguf.js";
import type { Model } from "../../shared/types.js";

function model(metadata: Model["metadata"]): Model {
  return {
    id: "m1",
    filename: "m.gguf",
    size_bytes: 1,
    source: "huggingface",
    hf_repo: "org/r",
    hf_file: "m.gguf",
    metadata,
    created_at: 0,
  } as Model;
}

function info(partial: Partial<GgufInfo>): GgufInfo {
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
    ...partial,
  } as GgufInfo;
}

describe("applyLocalHeader", () => {
  it("replaces the server's (possibly poisoned) header fields with the local file's", async () => {
    const poisoned = model({ n_layer: 2, trained_ctx: 512, arch: "llama" } as Model["metadata"]);
    const out = await applyLocalHeader(poisoned, "x.gguf", async () => info({ n_layer: 16, trained_ctx: 131072 }));
    expect(out.metadata.n_layer).toBe(16);
    expect(out.metadata.trained_ctx).toBe(131072);
    // Non-header fields untouched; input not mutated.
    expect((out.metadata as Record<string, unknown>).arch).toBe("llama");
    expect(poisoned.metadata.n_layer).toBe(2);
  });

  it("keeps the server's value for anything the local read couldn't produce", async () => {
    const out = await applyLocalHeader(model({ n_layer: 16, quant: "Q4_K_M" }), "x.gguf", async () => info({ n_layer: 16 }));
    expect(out.metadata.quant).toBe("Q4_K_M");
  });

  it("falls back to the server's metadata entirely when the file can't be read", async () => {
    const m = model({ n_layer: 16 });
    const out = await applyLocalHeader(m, "x.gguf", async () => {
      throw new Error("EACCES");
    });
    expect(out).toBe(m);
  });
});
