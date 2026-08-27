export type EngineKind = "bench" | "server";
export type SpecMode = "off" | "mtp" | "draft";

export interface EnginePair {
  engine: EngineKind;
  spec: SpecMode;
}

export const LEGAL_ENGINE_PAIRS: readonly EnginePair[] = [
  { engine: "bench", spec: "off" },
  { engine: "server", spec: "off" },
  { engine: "server", spec: "mtp" },
  { engine: "server", spec: "draft" },
];

export function isLegalEnginePair(pair: EnginePair): boolean {
  return LEGAL_ENGINE_PAIRS.some((p) => p.engine === pair.engine && p.spec === pair.spec);
}

export const CACHE_TYPE_VALUES = [
  "f32",
  "bf16",
  "f16",
  "q8_0",
  "q5_0",
  "q5_1",
  "q4_1",
  "q4_0",
  "iq4_nl",
] as const;

export type CacheType = (typeof CACHE_TYPE_VALUES)[number];

export function isKnownCacheType(v: unknown): v is CacheType {
  return typeof v === "string" && (CACHE_TYPE_VALUES as readonly string[]).includes(v);
}

export const DEPTH_RULE_REJECTION_MESSAGE =
  "n_depth is only supported on the llama-bench engine — llama-server has no KV-prefill flag. Set n_depth to 0 for llama-server configurations.";

export function engineFromItem(item: { mtp: string }): EngineKind {
  return item.mtp === "on" ? "server" : "bench";
}

export function pairFromItem(item: { mtp: string }): EnginePair {
  if (item.mtp === "on") return { engine: "server", spec: "mtp" };
  return { engine: "bench", spec: "off" };
}

export function specTypeFor(pair: EnginePair): string | null {
  if (pair.engine !== "server") return null;
  if (pair.spec === "mtp") return "draft-mtp";
  if (pair.spec === "draft") return "draft-simple";
  return null;
}
