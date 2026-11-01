import { describe, expect, it } from "vitest";
import { baseUrl, formatMib, loadSummary, modelsOnMachine, projectorFiles, settingsDiffer, withRemainder } from "./serveData";
import type { DaemonStatus } from "../../../../shared/daemonTypes";
import type { Model, Worker } from "../../types";

const model = (over: Partial<Model>) => ({ id: "x", source: "huggingface", hf_file: "a.gguf", filename: "a.gguf", ...over }) as Model;
const worker = (paths: { path: string; sha256?: string }[]) => ({ modelFiles: paths }) as unknown as Worker;

describe("modelsOnMachine", () => {
  it("keeps what is on disk, matched by path or hash, and hides projectors", () => {
    const sha = "a".repeat(64);
    const models = [
      model({ id: "1", hf_file: "on-disk.gguf" }),
      model({ id: "2", hf_file: "elsewhere.gguf" }),
      model({ id: sha, hf_file: "renamed.gguf" }),
      model({ id: "4", hf_file: "mmproj-x.gguf" }),
    ];
    const w = worker([{ path: "on-disk.gguf" }, { path: "local-name.gguf", sha256: sha }, { path: "mmproj-x.gguf" }]);
    expect(modelsOnMachine(models, w).map((m) => m.id).sort()).toEqual(["1", sha].sort());
    expect(projectorFiles(w)).toEqual(["mmproj-x.gguf"]);
  });

  it("is empty with no machine", () => {
    expect(modelsOnMachine([model({})], undefined)).toEqual([]);
  });
});

describe("baseUrl", () => {
  it("names a real address for an all-interfaces bind", () => {
    const status = { interfaces: [{ name: "Ethernet", address: "192.168.1.5" }] } as DaemonStatus;
    expect(baseUrl("127.0.0.1", 8642, status)).toBe("http://127.0.0.1:8642/v1");
    expect(baseUrl("0.0.0.0", 8642, status)).toBe("http://192.168.1.5:8642/v1");
    expect(baseUrl("0.0.0.0", 8642, null)).toContain("this machine's address");
  });
});

describe("settingsDiffer", () => {
  it("ignores key order and treats nothing as empty", () => {
    expect(settingsDiffer({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(false);
    expect(settingsDiffer(null, {})).toBe(false);
    expect(settingsDiffer({ a: 1 }, { a: 2 })).toBe(true);
    expect(settingsDiffer({ a: 1 }, {})).toBe(true);
  });
});

describe("withRemainder", () => {
  it("accounts for what llama.cpp did not itemize", () => {
    expect(withRemainder([{ label: "weights", mib: 863 }, { label: "kv", mib: 192 }, { label: "x", mib: null }], 1110, "runtime")).toEqual([
      { label: "weights", mib: 863 },
      { label: "kv", mib: 192 },
      { label: "runtime", mib: 55 },
    ]);
  });

  it("does not invent a remainder from counter noise or a negative", () => {
    expect(withRemainder([{ label: "weights", mib: 100 }], 104, "runtime")).toHaveLength(1);
    expect(withRemainder([{ label: "weights", mib: 100 }], 50, "runtime")).toHaveLength(1);
    expect(withRemainder([{ label: "weights", mib: 100 }], null, "runtime")).toHaveLength(1);
  });
});

describe("formatting", () => {
  it("formats memory", () => {
    expect(formatMib(null)).toBe("n/a");
    expect(formatMib(812)).toBe("812 MiB");
    expect(formatMib(8176)).toBe("8.0 GiB");
  });

  it("summarizes a load", () => {
    expect(loadSummary({ settings: { ctx_size: 8192 }, actual: { gpu_layers: 49, total_layers: 49, n_ctx: 8192, n_slots: 1 }, build_tag: "b11609", requests: 2 })).toBe("49/49 layers · ctx 8,192 · 1 slot · b11609 · 2 requests");
  });
});
