import { describe, it, expect, afterEach, vi } from "vitest";
import {
  formatBytes,
  formatGpuLabel,
  formatRelativeTime,
  formatShortRelativeTime,
  formatElapsed,
  formatFlashAttn,
  shortId,
  hfRepoUrl,
  hfFileUrl,
  modelAuthor,
  modelFamily,
  paramsBFromText,
  modelParamsB,
  formatParamsB,
  migrateLegacyStorageKeys,
} from "./utils";
import type { Model } from "./types";

function model(patch: Partial<Model> = {}): Model {
  return {
    id: "m1",
    filename: "model.gguf",
    size_bytes: 1_000,
    source: "huggingface",
    metadata: {},
    created_at: 0,
    ...patch,
  };
}

describe("formatBytes", () => {
  it("renders 0 without a unit step", () => {
    expect(formatBytes(0)).toBe("0 B");
  });

  it("keeps whole bytes undecimated but decimates every larger unit", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1024 ** 3)).toBe("1.0 GB");
  });

  it("stops climbing at TB rather than running off the unit list", () => {
    expect(formatBytes(1024 ** 5)).toBe("1024.0 TB");
  });
});

describe("formatGpuLabel", () => {
  it("falls back to the vendor, then to 'unknown', when the model name is blank", () => {
    expect(formatGpuLabel({ model: "", vendor: "NVIDIA" })).toBe("NVIDIA");
    expect(formatGpuLabel({ model: "", vendor: "" })).toBe("unknown");
  });

  it("omits the VRAM figure entirely when the driver reported none", () => {
    expect(formatGpuLabel({ model: "RTX 4090", vendor: "NVIDIA" })).toBe("RTX 4090");
    expect(formatGpuLabel({ model: "RTX 4090", vendor: "NVIDIA", vram_mb: null })).toBe("RTX 4090");
  });

  it("marks unified/shared memory so it is not read as dedicated VRAM", () => {
    expect(formatGpuLabel({ model: "RTX 4090", vendor: "NVIDIA", vram_mb: 24_576 })).toBe("RTX 4090 (24.0 GB)");
    expect(formatGpuLabel({ model: "Iris Xe", vendor: "Intel", vram_mb: 1024, vram_dynamic: true })).toBe(
      "Iris Xe (1.0 GB shared)"
    );
  });
});

describe("formatRelativeTime", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  function at(now: string) {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(now));
  }

  it("returns null for a missing or unparseable timestamp so callers can drop the label", () => {
    expect(formatRelativeTime(null)).toBeNull();
    expect(formatRelativeTime(undefined)).toBeNull();
    expect(formatRelativeTime("")).toBeNull();
    expect(formatRelativeTime("not a date")).toBeNull();
  });

  it("walks day -> month -> year granularity", () => {
    at("2026-09-06T12:00:00Z");
    expect(formatRelativeTime("2026-09-06T11:00:00Z")).toBe("today");
    expect(formatRelativeTime("2026-09-05T11:00:00Z")).toBe("yesterday");
    expect(formatRelativeTime("2026-09-01T12:00:00Z")).toBe("5 days ago");
    expect(formatRelativeTime("2026-07-06T12:00:00Z")).toBe("2 months ago");
    expect(formatRelativeTime("2024-09-06T12:00:00Z")).toBe("2 years ago");
  });

  it("singularises the one-month and one-year cases", () => {
    at("2026-09-06T12:00:00Z");
    expect(formatRelativeTime("2026-08-01T12:00:00Z")).toBe("1 month ago");
    expect(formatRelativeTime("2025-08-01T12:00:00Z")).toBe("1 year ago");
  });

  it("reads a future timestamp as 'today' rather than emitting a negative count", () => {
    at("2026-09-06T12:00:00Z");
    expect(formatRelativeTime("2027-01-01T00:00:00Z")).toBe("today");
  });
});

describe("formatShortRelativeTime", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("steps through every unit boundary", () => {
    vi.useFakeTimers();
    const now = new Date("2026-09-06T12:00:00Z").getTime();
    vi.setSystemTime(now);
    const ago = (ms: number) => formatShortRelativeTime(now - ms);
    const s = 1000;
    const min = 60 * s;
    const h = 60 * min;
    const d = 24 * h;

    expect(ago(0)).toBe("now");
    expect(ago(59 * s)).toBe("now");
    expect(ago(90 * s)).toBe("1m");
    expect(ago(3 * h)).toBe("3h");
    expect(ago(3 * d)).toBe("3d");
    // 7-29 days reports in weeks; 30+ switches to months.
    expect(ago(10 * d)).toBe("1w");
    expect(ago(29 * d)).toBe("4w");
    expect(ago(45 * d)).toBe("1mo");
    expect(ago(400 * d)).toBe("1y");
  });

  it("clamps a future timestamp to 'now' instead of going negative", () => {
    vi.useFakeTimers();
    const now = Date.now();
    vi.setSystemTime(now);
    expect(formatShortRelativeTime(now + 60_000)).toBe("now");
  });
});

describe("formatElapsed", () => {
  it("drops seconds once the duration reaches an hour", () => {
    expect(formatElapsed(0)).toBe("0s");
    expect(formatElapsed(45_000)).toBe("45s");
    expect(formatElapsed(65_000)).toBe("1m 5s");
    expect(formatElapsed(3_661_000)).toBe("1h 1m");
  });

  it("floors a negative duration to zero rather than rendering a negative clock", () => {
    expect(formatElapsed(-5_000)).toBe("0s");
  });
});

describe("formatFlashAttn", () => {
  it("normalises every historical encoding of the same value", () => {
    expect(formatFlashAttn("true")).toBe("on");
    expect(formatFlashAttn("1")).toBe("on");
    expect(formatFlashAttn("false")).toBe("off");
    expect(formatFlashAttn("0")).toBe("off");
  });

  it("passes an already-normalised or unrecognised value straight through", () => {
    expect(formatFlashAttn("on")).toBe("on");
    expect(formatFlashAttn("auto")).toBe("auto");
  });
});

describe("shortId and HF URL builders", () => {
  it("truncates to the requested length without padding a short id", () => {
    expect(shortId("abcdefghijklmnop")).toBe("abcdefghijkl");
    expect(shortId("abcdefghijklmnop", 4)).toBe("abcd");
    expect(shortId("abc")).toBe("abc");
  });

  it("builds repo and blob URLs", () => {
    expect(hfRepoUrl("bartowski/Llama-3.1-8B-GGUF")).toBe("https://huggingface.co/bartowski/Llama-3.1-8B-GGUF");
    expect(hfFileUrl("unsloth/gemma-4", "MTP/mtp-gemma-4.gguf")).toBe(
      "https://huggingface.co/unsloth/gemma-4/blob/main/MTP/mtp-gemma-4.gguf"
    );
  });
});

describe("modelAuthor", () => {
  it("takes the HF namespace, and labels a locally-registered model 'local'", () => {
    expect(modelAuthor(model({ hf_repo: "bartowski/Llama-3.1-8B-GGUF" }))).toBe("bartowski");
    expect(modelAuthor(model({ hf_repo: undefined }))).toBe("local");
  });
});

describe("modelFamily", () => {
  it("prefers stored metadata.arch over any filename guess", () => {
    expect(modelFamily(model({ metadata: { arch: "qwen3moe" }, hf_repo: "x/Llama-3.1-8B" }))).toBe("qwen3moe");
  });

  it("ignores a blank arch and falls through to pattern matching", () => {
    expect(modelFamily(model({ metadata: { arch: "   " }, filename: "gemma-3-4b.gguf" }))).toBe("Gemma");
  });

  it("matches the more specific generation before the bare family name", () => {
    expect(modelFamily(model({ filename: "Llama-4-Scout.gguf" }))).toBe("Llama 4");
    expect(modelFamily(model({ filename: "Llama-3.1-8B.gguf" }))).toBe("Llama 3");
    expect(modelFamily(model({ filename: "llama-7b.gguf" }))).toBe("Llama");
  });

  it("falls back to 'Other' rather than guessing", () => {
    expect(modelFamily(model({ filename: "totally-novel-arch.gguf" }))).toBe("Other");
  });
});

describe("parameter-count parsing", () => {
  it("reads the conventional -8B- token in either case", () => {
    expect(paramsBFromText("bartowski/Llama-3.1-8B-GGUF")).toBe(8);
    expect(paramsBFromText("gemma-3-27b-it")).toBe(27);
    expect(paramsBFromText("model-1.5B-chat")).toBe(1.5);
  });

  it("tolerates a single letter prefix (Gemma's E2B/E4B) without capturing it", () => {
    expect(paramsBFromText("gemma-3n-E4B-it")).toBe(4);
  });

  it("returns null when nothing matches, rather than guessing", () => {
    expect(paramsBFromText("some-model-name")).toBeNull();
    // A bare "8B" with no separator before it must not match.
    expect(paramsBFromText("abc8B")).toBeNull();
  });

  it("prefers authoritative metadata.param_count over the filename guess", () => {
    const m = model({ hf_repo: "x/Llama-3.1-8B-GGUF", metadata: { param_count: 8_030_000_000 } });
    expect(modelParamsB(m)).toBe(8);
  });

  it("rounds a metadata count to one decimal place", () => {
    expect(modelParamsB(model({ metadata: { param_count: 1_777_000_000 } }))).toBe(1.8);
  });

  it("falls back to the filename when no metadata count is stored", () => {
    expect(modelParamsB(model({ filename: "Qwen2.5-14B-Instruct.gguf", metadata: {} }))).toBe(14);
    expect(modelParamsB(model({ filename: "mystery.gguf", metadata: {} }))).toBeNull();
  });

  it("renders an em dash for an unknown count", () => {
    expect(formatParamsB(null)).toBe("—");
    expect(formatParamsB(8)).toBe("8B");
  });
});

describe("migrateLegacyStorageKeys", () => {
  it("renames old-prefixed keys in both localStorage and sessionStorage", () => {
    localStorage.setItem("llama-bench:panel-width", "320");
    sessionStorage.setItem("llama-bench:draft", "hello");

    migrateLegacyStorageKeys();

    expect(localStorage.getItem("llamatoaster:panel-width")).toBe("320");
    expect(localStorage.getItem("llama-bench:panel-width")).toBeNull();
    expect(sessionStorage.getItem("llamatoaster:draft")).toBe("hello");
    expect(sessionStorage.getItem("llama-bench:draft")).toBeNull();
  });

  it("keeps the newer value when both the old and new key exist, and still drops the old one", () => {
    localStorage.setItem("llama-bench:theme", "old");
    localStorage.setItem("llamatoaster:theme", "new");

    migrateLegacyStorageKeys();

    expect(localStorage.getItem("llamatoaster:theme")).toBe("new");
    expect(localStorage.getItem("llama-bench:theme")).toBeNull();
  });

  it("leaves unrelated keys untouched and is a no-op on a second run", () => {
    localStorage.setItem("unrelated", "keep");
    localStorage.setItem("llama-bench:a", "1");

    migrateLegacyStorageKeys();
    migrateLegacyStorageKeys();

    expect(localStorage.getItem("unrelated")).toBe("keep");
    expect(localStorage.getItem("llamatoaster:a")).toBe("1");
    expect(localStorage.length).toBe(2);
  });

  it("migrates every matching key when several are present", () => {
    // Guards the index-based scan in the implementation: removing keys while
    // iterating by index is exactly where a partial migration would hide.
    for (let i = 0; i < 5; i++) localStorage.setItem(`llama-bench:k${i}`, String(i));

    migrateLegacyStorageKeys();

    for (let i = 0; i < 5; i++) {
      expect(localStorage.getItem(`llamatoaster:k${i}`)).toBe(String(i));
      expect(localStorage.getItem(`llama-bench:k${i}`)).toBeNull();
    }
  });
});
