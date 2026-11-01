import { describe, expect, it } from "vitest";
import {
  applySampling,
  buildLoadArgs,
  defaultAlias,
  LOAD_SETTINGS,
  SAMPLING_SETTINGS,
  unsupportedSettings,
  validateGateway,
  validateLoadSettings,
  validateSampling,
} from "./daemonSettings.js";

describe("load settings", () => {
  it("has unique keys and flags", () => {
    expect(new Set(LOAD_SETTINGS.map((d) => d.key)).size).toBe(LOAD_SETTINGS.length);
    expect(new Set(LOAD_SETTINGS.map((d) => d.flag)).size).toBe(LOAD_SETTINGS.length);
  });

  it("offers nothing that runs tools, writes prompts to disk or moves the endpoint", () => {
    const flags = LOAD_SETTINGS.flatMap((d) => [d.flag, ...(d.kind === "toggle" && d.offFlag ? [d.offFlag] : [])]);
    for (const banned of [
      "--tools", "--agent", "--mcp-servers-config", "--mcp-servers-json", "--ui-mcp-proxy", "--log-prompts-dir",
      "--slot-save-path", "--media-path", "--path", "--host", "--port", "--api-key", "--api-key-file", "--cors-origins",
      "--props", "--rpc", "--hf-repo", "--model-url", "--verbosity", "--log-file", "--grammar-file", "--lora",
    ]) {
      expect(flags).not.toContain(banned);
    }
  });

  it("rejects unknown keys instead of dropping them", () => {
    const { errors } = validateLoadSettings({ ctx_size: 4096, tools: "all" });
    expect(errors).toEqual(["unknown setting: tools"]);
  });

  it("does not let a value become another flag", () => {
    for (const [key, value] of [
      ["n_gpu_layers", "--tools"],
      ["device", "--log-prompts-dir=/x"],
      ["override_tensor", "--tools=all"],
      ["chat_template", "-m"],
      ["tensor_split", "-1"],
    ] as const) {
      expect(validateLoadSettings({ [key]: value }).errors).toHaveLength(1);
    }
  });

  it("checks ranges and types", () => {
    expect(validateLoadSettings({ ctx_size: -1 }).errors).toHaveLength(1);
    expect(validateLoadSettings({ ctx_size: 1.5 }).errors).toHaveLength(1);
    expect(validateLoadSettings({ flash_attn: "maybe" }).errors).toHaveLength(1);
    expect(validateLoadSettings({ kv_offload: "yes" }).errors).toHaveLength(1);
    expect(validateLoadSettings({ chat_template_kwargs: "[1,2]" }).errors).toHaveLength(1);
    expect(validateLoadSettings({ chat_template_kwargs: '{"enable_thinking":false}' }).errors).toEqual([]);
  });

  it("leaves unset settings out, so the build's default stands", () => {
    const { values } = validateLoadSettings({ ctx_size: "", parallel: null, threads: undefined });
    expect(values).toEqual({});
    expect(buildLoadArgs(values)).toEqual([]);
  });

  it("builds flags in a stable order, using the off form of a toggle", () => {
    expect(buildLoadArgs({ threads: 6, ctx_size: 8192, kv_offload: false, cache_prompt: true, cpu_moe: false })).toEqual([
      "--ctx-size", "8192", "--no-kv-offload", "--threads", "6", "--cache-prompt",
    ]);
  });

  it("knows a one-way switch set to off needs no support from the build", () => {
    const supported = new Set(["ctx_size"]);
    expect(unsupportedSettings({ ctx_size: 1, cpu_moe: false, kv_offload: false, numa: "isolate" }, supported)).toEqual(["kv_offload", "numa"]);
  });

  it("derives an API identifier from a file name", () => {
    expect(defaultAlias("lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf")).toBe("gemma-4-E2B-it-Q4_K_M");
    expect(defaultAlias("weird name (1).gguf")).toBe("weird-name-1-");
  });
});

describe("sampling", () => {
  it("every setting names a known request field and a default", () => {
    for (const def of SAMPLING_SETTINGS) {
      expect(def.fields.chat ?? def.fields.messages).toBeTruthy();
      expect(def.defaultField).toBeTruthy();
    }
  });

  it("rejects out-of-range and unknown values", () => {
    expect(validateSampling({ values: { temperature: 9 } }).errors).toHaveLength(1);
    expect(validateSampling({ values: { made_up: 1 } }).errors).toHaveLength(1);
    expect(validateSampling({ values: { temperature: 0.3 } }).sampling.values).toEqual({ temperature: 0.3 });
    expect(validateSampling({ stop: ["", "x"] }).errors).toHaveLength(1);
  });

  it("fills only what the client left unset, keeping its own values", () => {
    const body = { messages: [{ role: "user", content: "hi" }], temperature: 0.1 };
    const out = applySampling(body, "chat", { values: { temperature: 0.9, top_k: 7 } });
    expect(out.temperature).toBe(0.1);
    expect(out.top_k).toBe(7);
    expect(body).not.toHaveProperty("top_k");
  });

  it("wins over the client when told to", () => {
    const out = applySampling({ temperature: 0.1 }, "chat", { values: { temperature: 0.9 }, override_client: true });
    expect(out.temperature).toBe(0.9);
  });

  it("uses each endpoint's own field and skips what it does not take", () => {
    expect(applySampling({}, "responses", { values: { max_tokens: 5, top_k: 7, min_p: 0.1 } })).toEqual({ max_output_tokens: 5, top_k: 7, min_p: 0.1 });
    expect(applySampling({}, "messages", { values: { top_k: 7, min_p: 0.1 }, stop: ["x"] })).toEqual({ top_k: 7, stop_sequences: ["x"] });
    expect(applySampling({}, "responses", { values: {}, stop: ["x"] })).toEqual({});
  });

  it("adds a system prompt only when the request has none", () => {
    const user = { role: "user", content: "hi" };
    const added = applySampling({ messages: [user] }, "chat", { values: {}, system_prompt: "be terse" });
    expect(added.messages).toEqual([{ role: "system", content: "be terse" }, user]);
    const own = { role: "system", content: "mine" };
    expect(applySampling({ messages: [own, user] }, "chat", { values: {}, system_prompt: "be terse" }).messages).toEqual([own, user]);
    const forced = applySampling({ messages: [own, user] }, "chat", { values: {}, system_prompt: "be terse", override_client: true });
    expect(forced.messages).toEqual([{ role: "system", content: "be terse" }, user]);
    expect(applySampling({ system: "mine" }, "messages", { values: {}, system_prompt: "x" }).system).toBe("mine");
    expect(applySampling({}, "responses", { values: {}, system_prompt: "x" }).instructions).toBe("x");
  });

  it("returns the body as it came with nothing to apply", () => {
    const body = { a: 1 };
    expect(applySampling(body, "chat", null)).toBe(body);
  });
});

describe("gateway settings", () => {
  const good = { host: "127.0.0.1", port: 8642, cors_origins: [], autostart: false };

  it("refuses to open beyond this machine without a key", () => {
    expect(validateGateway({ ...good, host: "0.0.0.0" }, 0).errors).toHaveLength(1);
    expect(validateGateway({ ...good, host: "100.64.0.5" }, 0).errors).toHaveLength(1);
    expect(validateGateway({ ...good, host: "0.0.0.0" }, 1).errors).toEqual([]);
    expect(validateGateway(good, 0).errors).toEqual([]);
  });

  it("checks the port, the address and the origins", () => {
    expect(validateGateway({ ...good, port: 80 }, 1).errors).toHaveLength(1);
    expect(validateGateway({ ...good, host: "evil.example" }, 1).errors).toHaveLength(1);
    expect(validateGateway({ ...good, cors_origins: ["*"] }, 1).errors).toHaveLength(1);
    expect(validateGateway({ ...good, cors_origins: ["http://localhost:3000"] }, 1).errors).toEqual([]);
  });
});
