import { describe, it, expect } from "vitest";
import {
  extractQuant,
  resolveQuant,
  modelBaseLabel,
  quantTierLabel,
  buildModelGroups,
  groupHfFilesByTier,
  groupQuantsByTier,
} from "./modelGrouping";
import type { Model } from "./types";

function model(patch: Partial<Model> = {}): Model {
  return {
    id: Math.random().toString(36).slice(2),
    filename: "model.gguf",
    size_bytes: 1_000,
    source: "huggingface",
    metadata: {},
    created_at: 0,
    ...patch,
  };
}

describe("groupHfFilesByTier", () => {
  // unsloth/gemma-4-26B-A4B-it-GGUF's real file list, trimmed.
  const f = (path: string, size_bytes: number) => ({ path, size_bytes, quant: null });
  const tiers = groupHfFilesByTier([
    f("BF16/gemma-BF16-00001-of-00002.gguf", 49_920),
    f("BF16/gemma-BF16-00002-of-00002.gguf", 580),
    f("MTP/mtp-gemma-Q8_0.gguf", 460),
    f("MTP/mtp-gemma-BF16.gguf", 860),
    f("gemma-MXFP4_MOE.gguf", 16_550),
    f("gemma-Q8_0.gguf", 26_860),
    f("gemma-UD-IQ2_M.gguf", 10_010),
    f("gemma-UD-Q4_K_M.gguf", 16_950),
    f("gemma-UD-IQ4_XS.gguf", 13_600),
    f("mmproj-F32.gguf", 2_290),
    f("mtp-gemma.gguf", 460),
  ]);

  it("rows are bit widths in ascending order, with what has no bit width last", () => {
    expect(tiers.map((t) => t.label)).toEqual(["2-bit", "4-bit", "8-bit", "16-bit", "Other"]);
  });

  it("counts MXFP4 as 4-bit, sorted by size inside the row", () => {
    expect(tiers[1].chips.map((c) => c.label)).toEqual(["UD-IQ4_XS", "MXFP4_MOE", "UD-Q4_K_M"]);
  });

  it("merges a split model into one chip pointing at shard 1", () => {
    const bf16 = tiers[3].chips.find((c) => c.kind === "quant")!;
    expect(bf16).toMatchObject({ label: "BF16", file: "BF16/gemma-BF16-00001-of-00002.gguf", sizeBytes: 50_500 });
    expect(bf16.files).toHaveLength(2);
  });

  it("keeps MTP drafts in their bit row, tagged", () => {
    expect(tiers[2].chips.map((c) => [c.label, c.kind])).toEqual([["Q8_0", "mtp"], ["Q8_0", "quant"]]);
  });

  it("drops nothing: mmproj and an unlabelled MTP land in Other", () => {
    expect(tiers[4].chips.map((c) => c.kind).sort()).toEqual(["mmproj", "mtp"]);
  });
});

describe("extractQuant", () => {
  it("pulls a standard quant token out of a filename", () => {
    expect(extractQuant("Llama-3.1-8B-Q4_K_M.gguf")).toBe("Q4_K_M");
    expect(extractQuant("Llama-3.1-8B-Q8_0.gguf")).toBe("Q8_0");
    expect(extractQuant("model-IQ2_XXS.gguf")).toBe("IQ2_XXS");
  });

  it("reassembles unsloth's Dynamic prefix as part of the label", () => {
    expect(extractQuant("gemma-3-27b-UD-Q4_K_XL.gguf")).toBe("UD-Q4_K_XL");
    expect(extractQuant("gemma-3-27b-UD_IQ2_M.gguf")).toBe("UD-IQ2_M");
  });

  it("recognises the unquantised float formats", () => {
    expect(extractQuant("model-F16.gguf")).toBe("F16");
    expect(extractQuant("model-bf16.gguf")).toBe("BF16");
    expect(extractQuant("model-f32.gguf")).toBe("F32");
  });

  it("upper-cases whatever case the filename used", () => {
    expect(extractQuant("model-q4_k_m.gguf")).toBe("Q4_K_M");
  });

  it("returns null when the filename carries no quant token", () => {
    expect(extractQuant("model.gguf")).toBeNull();
    expect(extractQuant("mtp-gemma-4-E2B-it.gguf")).toBeNull();
  });

  it("requires a separator before the token, so a parameter count is never mistaken for one", () => {
    expect(extractQuant("Qwen2.5-14B-Instruct.gguf")).toBeNull();
  });
});

describe("resolveQuant", () => {
  it("prefers the authoritative GGUF header metadata over the filename", () => {
    const m = model({ filename: "model-Q4_K_M.gguf", metadata: { quant: "Q6_K" } });
    expect(resolveQuant(m)).toBe("Q6_K");
  });

  it("parses hf_file rather than filename when both are present", () => {
    const m = model({ filename: "local-copy.gguf", hf_file: "model-Q5_K_M.gguf" });
    expect(resolveQuant(m)).toBe("Q5_K_M");
  });

  it("falls back to filename parsing for a model registered before metadata.quant existed", () => {
    expect(resolveQuant(model({ filename: "model-Q3_K_S.gguf" }))).toBe("Q3_K_S");
  });

  it("is null when neither source yields a quant", () => {
    expect(resolveQuant(model({ filename: "model.gguf" }))).toBeNull();
  });
});

describe("modelBaseLabel", () => {
  it("uses the repo name and strips the -GGUF convention suffix", () => {
    expect(modelBaseLabel(model({ hf_repo: "bartowski/Llama-3.1-8B-GGUF" }))).toBe("Llama-3.1-8B");
    expect(modelBaseLabel(model({ hf_repo: "unsloth/gemma-3-27b-it_gguf" }))).toBe("gemma-3-27b-it");
  });

  it("falls back to the whole repo id when there is no owner segment", () => {
    expect(modelBaseLabel(model({ hf_repo: "solo-repo" }))).toBe("solo-repo");
  });

  it("strips extension, quant token and trailing separators from a local filename", () => {
    expect(modelBaseLabel(model({ hf_repo: undefined, filename: "Llama-3.1-8B-Q4_K_M.gguf" }))).toBe("Llama-3.1-8B");
    expect(modelBaseLabel(model({ hf_repo: undefined, filename: "my-model.bin" }))).toBe("my-model");
  });
});

describe("quantTierLabel", () => {
  it("labels by leading bit count", () => {
    expect(quantTierLabel(model({ metadata: { quant: "Q4_K_M" } }))).toBe("4-bit");
    expect(quantTierLabel(model({ metadata: { quant: "IQ2_XXS" } }))).toBe("2-bit");
    expect(quantTierLabel(model({ metadata: { quant: "Q8_0" } }))).toBe("8-bit");
  });

  it("treats the float formats as their true width", () => {
    expect(quantTierLabel(model({ metadata: { quant: "F16" } }))).toBe("16-bit");
    expect(quantTierLabel(model({ metadata: { quant: "BF16" } }))).toBe("16-bit");
    expect(quantTierLabel(model({ metadata: { quant: "F32" } }))).toBe("32-bit");
  });

  it("ranks a Dynamic quant by its underlying code, not the UD- prefix", () => {
    expect(quantTierLabel(model({ metadata: { quant: "UD-Q4_K_XL" } }))).toBe("4-bit");
  });

  it("buckets an unparseable file as Other", () => {
    expect(quantTierLabel(model({ filename: "model.gguf" }))).toBe("Other");
  });
});

describe("buildModelGroups", () => {
  it("groups quant siblings from one repo into a single entry", () => {
    const groups = buildModelGroups([
      model({ id: "a", hf_repo: "bartowski/Llama-3.1-8B-GGUF", hf_file: "Llama-3.1-8B-Q4_K_M.gguf" }),
      model({ id: "b", hf_repo: "bartowski/Llama-3.1-8B-GGUF", hf_file: "Llama-3.1-8B-Q8_0.gguf" }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("Llama-3.1-8B");
    expect(groups[0].author).toBe("bartowski");
    expect(groups[0].family).toBe("Llama 3");
    expect(groups[0].quants.map((q) => q.base.id)).toEqual(["a", "b"]);
  });

  it("keeps the same model from different authors in separate groups", () => {
    const groups = buildModelGroups([
      model({ id: "a", hf_repo: "bartowski/Llama-3.1-8B-GGUF" }),
      model({ id: "b", hf_repo: "unsloth/Llama-3.1-8B-GGUF" }),
    ]);
    expect(groups).toHaveLength(2);
  });

  it("sorts quants most-compressed first, breaking ties on file size", () => {
    const groups = buildModelGroups([
      model({ id: "f16", hf_repo: "r/M-GGUF", metadata: { quant: "F16" }, size_bytes: 900 }),
      model({ id: "q4-big", hf_repo: "r/M-GGUF", metadata: { quant: "Q4_K_M" }, size_bytes: 500 }),
      model({ id: "q2", hf_repo: "r/M-GGUF", metadata: { quant: "IQ2_XXS" }, size_bytes: 100 }),
      model({ id: "q4-small", hf_repo: "r/M-GGUF", metadata: { quant: "Q4_K_S" }, size_bytes: 400 }),
    ]);

    expect(groups[0].quants.map((q) => q.base.id)).toEqual(["q2", "q4-small", "q4-big", "f16"]);
  });

  it("sorts a file with no recognisable quant last", () => {
    const groups = buildModelGroups([
      model({ id: "none", hf_repo: "r/M-GGUF", filename: "plain.gguf" }),
      model({ id: "q4", hf_repo: "r/M-GGUF", metadata: { quant: "Q4_K_M" } }),
    ]);
    expect(groups[0].quants.map((q) => q.base.id)).toEqual(["q4", "none"]);
  });

  it("attaches an MTP draft to its repo siblings instead of listing it as a base model", () => {
    const groups = buildModelGroups([
      model({ id: "base", hf_repo: "unsloth/gemma-4-GGUF", hf_file: "gemma-4-12B-it-Q4_K_M.gguf" }),
      model({ id: "draft", hf_repo: "unsloth/gemma-4-GGUF", hf_file: "MTP/mtp-gemma-4-12B-it.gguf" }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].quants).toHaveLength(1);
    expect(groups[0].quants[0].base.id).toBe("base");
    expect(groups[0].quants[0].drafts.map((d) => d.id)).toEqual(["draft"]);
  });

  it("never attaches a draft to a local model, which has no repo to match on", () => {
    const groups = buildModelGroups([
      model({ id: "local", hf_repo: undefined, filename: "local-model-Q4_K_M.gguf" }),
      model({ id: "draft", hf_repo: "unsloth/x", hf_file: "mtp-x.gguf" }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].quants[0].drafts).toEqual([]);
  });

  it("returns no groups for an empty catalogue", () => {
    expect(buildModelGroups([])).toEqual([]);
  });

  it("returns no groups when every model is a draft", () => {
    expect(buildModelGroups([model({ hf_repo: "r/x", hf_file: "mtp-x.gguf" })])).toEqual([]);
  });
});

describe("groupQuantsByTier", () => {
  it("collapses consecutive same-tier entries into one bucket", () => {
    const groups = buildModelGroups([
      model({ id: "q4a", hf_repo: "r/M-GGUF", metadata: { quant: "Q4_K_S" }, size_bytes: 100 }),
      model({ id: "q4b", hf_repo: "r/M-GGUF", metadata: { quant: "Q4_K_M" }, size_bytes: 200 }),
      model({ id: "q8", hf_repo: "r/M-GGUF", metadata: { quant: "Q8_0" }, size_bytes: 300 }),
    ]);

    const tiers = groupQuantsByTier(groups[0].quants);
    expect(tiers.map((t) => t.label)).toEqual(["4-bit", "8-bit"]);
    expect(tiers[0].quants.map((q) => q.base.id)).toEqual(["q4a", "q4b"]);
    expect(tiers[1].quants.map((q) => q.base.id)).toEqual(["q8"]);
  });

  it("returns an empty list for no quants", () => {
    expect(groupQuantsByTier([])).toEqual([]);
  });

  it("splits a tier that appears twice non-consecutively, documenting the sort dependency", () => {
    // groupQuantsByTier only merges ADJACENT entries -- it relies on
    // buildModelGroups having already clustered same-tier files together.
    // Feeding it an unsorted array produces two "4-bit" buckets, which is the
    // failure mode resolveQuant's doc comment warns about.
    const q = (id: string, quant: string) => ({ base: model({ id, metadata: { quant } }), drafts: [] });
    const tiers = groupQuantsByTier([q("a", "Q4_K_M"), q("b", "Q8_0"), q("c", "Q4_K_S")]);
    expect(tiers.map((t) => t.label)).toEqual(["4-bit", "8-bit", "4-bit"]);
  });
});
