import { describe, expect, it } from "vitest";
import { parseWorkerState, parseActiveJobReport, parseDeviceStart } from "./validate-worker-state.js";
import { BadRequestError } from "./errors.js";

function validBody(overrides: Record<string, unknown> = {}) {
  return {
    machine_id: "machine-1",
    capabilities: ["benchmark"],
    hostname: "gpu-tower",
    backend: "cuda",
    hardware: {
      platform: "win32",
      arch: "x64",
      cpu: { manufacturer: "AMD", brand: "Ryzen 5 5600X", flags: ["avx2"], cores: 6 },
      gpu: [{ vendor: "NVIDIA", model: "RTX 4090", vram_mb: 24576, vram_dynamic: false }],
      mem_total_bytes: 34_000_000_000,
    },
    installed_builds: [{ tag: "b10362", asset_name: "win-cuda.zip", installed_at: 1000, active: true }],
    model_files: [{ path: "model.gguf", size_bytes: 5_000_000_000 }],
    status: "idle",
    ...overrides,
  };
}

describe("parseWorkerState", () => {
  it("accepts a well-formed body and passes values through", () => {
    const state = parseWorkerState(validBody());
    expect(state.machine_id).toBe("machine-1");
    expect(state.hardware.gpu[0]?.model).toBe("RTX 4090");
    expect(state.installed_builds[0]?.tag).toBe("b10362");
    expect(state.status).toBe("idle");
  });

  it("rejects a non-object body", () => {
    expect(() => parseWorkerState(null)).toThrow(BadRequestError);
    expect(() => parseWorkerState("nope")).toThrow(BadRequestError);
    expect(() => parseWorkerState(42)).toThrow(BadRequestError);
  });

  it("rejects a status outside idle/busy rather than coercing it", () => {
    expect(() => parseWorkerState(validBody({ status: "running" }))).toThrow(BadRequestError);
  });

  it("rejects a wrong-typed field instead of coercing it", () => {
    expect(() => parseWorkerState(validBody({ hostname: 12345 }))).toThrow(BadRequestError);
    expect(() => parseWorkerState(validBody({ machine_id: { not: "a string" } }))).toThrow(BadRequestError);
  });

  it("strips control characters from free-text fields", () => {
    const state = parseWorkerState(validBody({ hostname: "gpu\x00tower\x1b[31m\x7f" }));
    expect(state.hostname).toBe("gputower[31m");
  });

  it("truncates an overlong string rather than rejecting the whole request", () => {
    const long = "x".repeat(1000);
    const state = parseWorkerState(validBody({ hostname: long }));
    expect(state.hostname).toHaveLength(256);
  });

  it("caps model_files to maxModelFiles instead of accepting an unbounded array", () => {
    const files = Array.from({ length: 3000 }, (_, i) => ({ path: `f${i}.gguf`, size_bytes: 1 }));
    const state = parseWorkerState(validBody({ model_files: files }));
    expect(state.model_files).toHaveLength(2000);
  });

  it("preserves sha256, state, and hf_match from model_files instead of dropping them", () => {
    const sha = "a".repeat(64);
    const state = parseWorkerState(
      validBody({
        model_files: [
          {
            path: "m.gguf",
            size_bytes: 1,
            sha256: sha.toUpperCase(),
            state: "verified",
            hf_match: { repo_id: "org/repo", filename: "m.gguf", revision: "main", deleted: false },
          },
        ],
      })
    );
    expect(state.model_files[0]?.sha256).toBe(sha); // normalized to lowercase
    expect(state.model_files[0]?.state).toBe("verified");
    expect(state.model_files[0]?.hf_match).toEqual({
      repo_id: "org/repo",
      filename: "m.gguf",
      revision: "main",
      deleted: false,
    });
  });

  it("rejects a malformed model file sha256 or state rather than silently dropping it", () => {
    expect(() =>
      parseWorkerState(validBody({ model_files: [{ path: "m.gguf", size_bytes: 1, sha256: "not-a-digest" }] }))
    ).toThrow(BadRequestError);
    expect(() =>
      parseWorkerState(validBody({ model_files: [{ path: "m.gguf", size_bytes: 1, state: "bogus" }] }))
    ).toThrow(BadRequestError);
  });

  it("still accepts model files without hash fields (old worker binaries)", () => {
    const state = parseWorkerState(validBody());
    expect(state.model_files[0]).toMatchObject({ path: "model.gguf", size_bytes: 5_000_000_000 });
    expect(state.model_files[0]?.sha256).toBeUndefined();
    expect(state.model_files[0]?.state).toBeUndefined();
    expect(state.model_files[0]?.hf_match).toBeNull();
  });

  it("caps installed_builds to maxBuilds", () => {
    const builds = Array.from({ length: 500 }, (_, i) => ({
      tag: `b${i}`,
      asset_name: "a.zip",
      installed_at: 1,
      active: false,
    }));
    const state = parseWorkerState(validBody({ installed_builds: builds }));
    expect(state.installed_builds).toHaveLength(100);
  });

  it("caps gpu array to maxGpus", () => {
    const gpu = Array.from({ length: 50 }, () => ({ vendor: "NVIDIA", model: "RTX 4090" }));
    const state = parseWorkerState(validBody({ hardware: { ...validBody().hardware as object, gpu } }));
    expect(state.hardware.gpu).toHaveLength(16);
  });

  it("rejects missing hardware.cpu", () => {
    const body = validBody();
    const hardware = { ...(body.hardware as Record<string, unknown>) };
    delete hardware.cpu;
    expect(() => parseWorkerState({ ...body, hardware })).toThrow(BadRequestError);
  });

  it("passes the NVIDIA driver probe through (and keeps it optional for old workers)", () => {
    const without = parseWorkerState(validBody());
    expect(without.hardware.nvidia_driver).toBeUndefined();

    const body = validBody({
      hardware: {
        ...(validBody().hardware as object),
        nvidia_driver: { version: "566.36", cuda_version: "12.7" },
      },
    });
    const withProbe = parseWorkerState(body);
    expect(withProbe.hardware.nvidia_driver).toEqual({ version: "566.36", cuda_version: "12.7" });
  });

  it("rejects a non-object nvidia_driver rather than coercing it", () => {
    expect(() =>
      parseWorkerState(validBody({ hardware: { ...(validBody().hardware as object), nvidia_driver: "566.36" } }))
    ).toThrow(BadRequestError);
  });

  it("passes cudart_name through on installed_builds", () => {
    const state = parseWorkerState(
      validBody({
        installed_builds: [
          {
            tag: "b10612",
            asset_name: "llama-b10612-bin-win-cuda-12.4-x64.zip",
            installed_at: 1000,
            active: true,
            cudart_name: "cudart-llama-bin-win-cuda-12.4-x64.zip",
          },
        ],
      })
    );
    expect(state.installed_builds[0]?.cudart_name).toBe("cudart-llama-bin-win-cuda-12.4-x64.zip");
  });
});

describe("parseActiveJobReport", () => {
  it("returns undefined when active_job is absent or null", () => {
    expect(parseActiveJobReport({})).toBeUndefined();
    expect(parseActiveJobReport({ active_job: null })).toBeUndefined();
  });

  it("parses a well-formed active_job", () => {
    const report = parseActiveJobReport({
      active_job: { job_id: "job-1", phase: "benchmarking", item_idx: 2, items_total: 10 },
    });
    expect(report).toMatchObject({ job_id: "job-1", phase: "benchmarking", item_idx: 2, items_total: 10 });
  });

  it("rejects an invalid phase", () => {
    expect(() => parseActiveJobReport({ active_job: { job_id: "job-1", phase: "sleeping" } })).toThrow(
      BadRequestError
    );
  });
});

describe("parseDeviceStart", () => {
  function body(overrides: Record<string, unknown> = {}) {
    return {
      machine_id: "machine-1",
      hostname: "gpu-tower",
      platform: "linux",
      arch: "x64",
      ...overrides,
    };
  }

  it("accepts a body with no hardware field at all (a not-yet-updated worker binary)", () => {
    const input = parseDeviceStart(body());
    expect(input.hardware).toBeUndefined();
  });

  it("parses hardware via the same validator the heartbeat path uses, when present", () => {
    const input = parseDeviceStart(
      body({
        hardware: {
          platform: "linux",
          arch: "x64",
          cpu: { manufacturer: "AMD", brand: "Ryzen 5 5600X", flags: ["avx2"], cores: 6 },
          gpu: [{ vendor: "NVIDIA", model: "RTX 4090", vram_mb: 24576, vram_dynamic: false }],
          mem_total_bytes: 34_000_000_000,
        },
      })
    );
    expect(input.hardware?.cpu.brand).toBe("Ryzen 5 5600X");
    expect(input.hardware?.gpu[0]?.model).toBe("RTX 4090");
  });

  it("rejects a malformed hardware object rather than silently dropping it", () => {
    expect(() => parseDeviceStart(body({ hardware: { platform: "linux" } }))).toThrow(BadRequestError);
  });
});
