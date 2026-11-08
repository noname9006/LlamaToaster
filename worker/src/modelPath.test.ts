import { describe, expect, it } from "vitest";
import { chooseModelPath, type ModelPathEntry } from "./modelPath.js";

const SHA = "a".repeat(64);
const OTHER = "b".repeat(64);
const onDisk = (...paths: string[]) => (p: string) => paths.includes(p);

describe("chooseModelPath", () => {
  it("finds a hand-dropped file by hash wherever it is, ahead of the name", () => {
    const entries: ModelPathEntry[] = [
      { path: "name.gguf", sha256: OTHER },
      { path: "elsewhere/renamed.gguf", sha256: SHA },
    ];
    expect(chooseModelPath({ id: SHA }, "name.gguf", entries, onDisk("name.gguf"))).toEqual({
      kind: "found",
      path: "elsewhere/renamed.gguf",
      proven: true,
    });
  });

  it("refuses a file at the name whose hash is known to differ", () => {
    const choice = chooseModelPath({ id: SHA }, "name.gguf", [{ path: "name.gguf", sha256: OTHER }], onDisk("name.gguf"));
    expect(choice).toEqual({ kind: "different", path: "name.gguf", actual: OTHER });
  });

  it("uses the file at the name while its hash isn't known yet", () => {
    const choice = chooseModelPath({ id: SHA }, "name.gguf", [{ path: "name.gguf", state: "hashing" }], onDisk("name.gguf"));
    expect(choice).toEqual({ kind: "found", path: "name.gguf", proven: true });
  });

  it("ignores a missing entry's hash", () => {
    const entries: ModelPathEntry[] = [{ path: "gone.gguf", sha256: SHA, state: "missing" }];
    expect(chooseModelPath({ id: SHA }, "name.gguf", entries, onDisk())).toEqual({ kind: "none" });
  });

  it("matches by Hugging Face provenance", () => {
    const model = { id: "hf:r/f.gguf", hf_repo: "r/n", hf_file: "f.gguf" };
    const entries: ModelPathEntry[] = [{ path: "x/y.gguf", hf_model_id: "r/n/f.gguf" }];
    expect(chooseModelPath(model, "f.gguf", entries, onDisk())).toEqual({ kind: "found", path: "x/y.gguf", proven: true });
  });

  it("doesn't take a provenance match whose hash contradicts the model", () => {
    const model = { id: SHA, hf_repo: "r/n", hf_file: "f.gguf" };
    const entries: ModelPathEntry[] = [{ path: "f.gguf", hf_model_id: "r/n/f.gguf", sha256: OTHER }];
    expect(chooseModelPath(model, "f.gguf", entries, onDisk("f.gguf")).kind).toBe("different");
  });

  describe("exactOnly (Will it fit?)", () => {
    const model = { id: "hf:wanted", hf_repo: "wanted/repo", hf_file: "Model-Q4.gguf" };

    it("another repo's same-named file isn't this one", () => {
      const entries: ModelPathEntry[] = [{ path: "Model-Q4.gguf", hf_model_id: "other/repo/Model-Q4.gguf" }];
      expect(chooseModelPath(model, "Model-Q4.gguf", entries, onDisk("Model-Q4.gguf"), { exactOnly: true }).kind).toBe("different");
    });

    it("a same-named file of unknown origin is found, but not proven", () => {
      expect(chooseModelPath(model, "Model-Q4.gguf", [], onDisk("Model-Q4.gguf"), { exactOnly: true })).toEqual({
        kind: "found",
        path: "Model-Q4.gguf",
        proven: false,
      });
    });

    it("never falls back to a basename match", () => {
      const entries: ModelPathEntry[] = [{ path: "sub/Model-Q4.gguf" }];
      expect(chooseModelPath(model, "Model-Q4.gguf", entries, onDisk(), { exactOnly: true })).toEqual({ kind: "none" });
    });

    it("a local model's name is proof enough: nothing else could answer", () => {
      expect(chooseModelPath({ id: "local-1", source: "local" }, "m.gguf", [], onDisk("m.gguf"), { exactOnly: true })).toEqual({
        kind: "found",
        path: "m.gguf",
        proven: true,
      });
    });
  });

  it("falls back to a same-named file, unless its hash says otherwise", () => {
    expect(chooseModelPath({ id: "derived" }, "sub/m.gguf", [{ path: "m.gguf" }], onDisk())).toEqual({
      kind: "found",
      path: "m.gguf",
      proven: false,
    });
    expect(chooseModelPath({ id: SHA }, "sub/m.gguf", [{ path: "m.gguf", sha256: OTHER }], onDisk())).toEqual({ kind: "none" });
  });
});
