export type GoalKind = "balanced" | "max_speed" | "max_context";
export type WorkloadShape = "chat" | "docs" | "even";
export type KvTolerance = "q4_0_ok" | "q8_0_ok" | "f16_only";

export interface GoalsConfig {
  goal: GoalKind;
  target_ctx?: number | null;
  speed_floor_frac?: number;
  workload: WorkloadShape;
  kv_tolerance?: KvTolerance;
}

export const GOAL_KINDS: readonly GoalKind[] = ["balanced", "max_speed", "max_context"];
export const WORKLOAD_SHAPES: readonly WorkloadShape[] = ["chat", "docs", "even"];
export const KV_TOLERANCES: readonly KvTolerance[] = ["q4_0_ok", "q8_0_ok", "f16_only"];
export const SPEED_FLOOR_CHOICES = [0.4, 0.5, 0.6] as const;

export const WORKLOAD_WEIGHTS: Record<WorkloadShape, { wPP: number; wTG: number }> = {
  chat: { wPP: 0.25, wTG: 0.75 },
  docs: { wPP: 0.7, wTG: 0.3 },
  even: { wPP: 0.5, wTG: 0.5 },
};

export function defaultGoals(): GoalsConfig {
  return { goal: "balanced", target_ctx: null, speed_floor_frac: 0.5, workload: "even", kv_tolerance: "q4_0_ok" };
}

function oneOf<T extends string>(raw: unknown, allowed: readonly T[]): T | undefined {
  return typeof raw === "string" && (allowed as readonly string[]).includes(raw) ? (raw as T) : undefined;
}

export function normalizeGoals(raw: unknown): GoalsConfig | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const goal = oneOf(r.goal, GOAL_KINDS) ?? defaultGoals().goal;
  const workload = oneOf(r.workload, WORKLOAD_SHAPES) ?? defaultGoals().workload;
  const kv_tolerance = oneOf(r.kv_tolerance, KV_TOLERANCES) ?? defaultGoals().kv_tolerance;
  let target_ctx: number | null | undefined;
  if (r.target_ctx === null || r.target_ctx === "unverified") target_ctx = null;
  else if (typeof r.target_ctx === "number" && Number.isFinite(r.target_ctx) && r.target_ctx > 0) {
    target_ctx = Math.floor(r.target_ctx);
  } else target_ctx = defaultGoals().target_ctx;
  const floorRaw = typeof r.speed_floor_frac === "number" ? r.speed_floor_frac : undefined;
  const speed_floor_frac =
    floorRaw != null && (SPEED_FLOOR_CHOICES as readonly number[]).includes(floorRaw)
      ? floorRaw
      : defaultGoals().speed_floor_frac;
  return { goal, target_ctx, speed_floor_frac, workload, kv_tolerance };
}

export function goalsEqualDefaults(goals: GoalsConfig | undefined): boolean {
  if (!goals) return true;
  const d = defaultGoals();
  return (
    goals.goal === d.goal &&
    (goals.target_ctx == null || goals.target_ctx === d.target_ctx) &&
    (goals.speed_floor_frac ?? d.speed_floor_frac) === d.speed_floor_frac &&
    goals.workload === d.workload &&
    (goals.kv_tolerance ?? d.kv_tolerance) === d.kv_tolerance
  );
}

const QUANTIZED_KV_TYPES = new Set(["q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]);
const UNQUANTIZED_KV_TYPES = new Set(["f32", "bf16", "f16"]);

function isQuantized(t: string): boolean {
  return QUANTIZED_KV_TYPES.has(t);
}

function minQuantBitsRequired(tolerance: KvTolerance): number {
  switch (tolerance) {
    case "q4_0_ok":
      return 0;
    case "q8_0_ok":
      return 5;
    case "f16_only":
      return Number.POSITIVE_INFINITY;
  }
}

function quantBits(t: string): number {
  switch (t) {
    case "q8_0":
      return 8;
    case "q5_1":
    case "q5_0":
      return 5;
    case "q4_1":
    case "q4_0":
    case "iq4_nl":
      return 4;
    default:
      return 16;
  }
}

export function pairAllowedUnderTolerance(ck: string, cv: string, tolerance: KvTolerance): boolean {
  const min = minQuantBitsRequired(tolerance);
  for (const t of [ck, cv]) {
    if (!isQuantized(t)) continue;
    if (quantBits(t) < min) return false;
  }
  return true;
}

export function pruneCacheTypes(types: string[], tolerance: KvTolerance): string[] {
  // If every value in this axis is a quantized type the tolerance forbids,
  // the axis prunes to empty -- NEVER back to the original, unquantized
  // types are never pruned by tolerance in the first place, so an empty
  // result here means the caller reintroducing the originals would be
  // reintroducing exactly the forbidden quantized values (a silent
  // tolerance violation). ensureUnquantizedPairSurvives is what adds a
  // real fallback pair back in.
  return types.filter((t) => !isQuantized(t) || quantBits(t) >= minQuantBitsRequired(tolerance));
}

// One unquantized pair always survives so the flash-attention-off axis keeps
// something to vary against (M4's inviolable rule). Expert mode prunes
// quantized sides but retains other unquantized pairs; only the recommended
// set collapses to the single f16/f16 value.
export function ensureUnquantizedPairSurvives(
  cache_type_k: string[],
  cache_type_v: string[],
  recommendedPairs: ReadonlyArray<readonly [string, string]>,
  expertMode: boolean
): { cache_type_k: string[]; cache_type_v: string[] } {
  const hasUnquantizedPair = cache_type_k.some((k) => UNQUANTIZED_KV_TYPES.has(k)) &&
    cache_type_v.some((v) => UNQUANTIZED_KV_TYPES.has(v));
  if (hasUnquantizedPair) return { cache_type_k, cache_type_v };
  const fallbackPair = expertMode
    ? (recommendedPairs.find(([k, v]) => UNQUANTIZED_KV_TYPES.has(k) && UNQUANTIZED_KV_TYPES.has(v)) ??
      (["f32", "bf16", "f16"] as const)
        .map((t) => [t, t] as const)
        .find(([k, v]) => UNQUANTIZED_KV_TYPES.has(k) && UNQUANTIZED_KV_TYPES.has(v)))!
    : (["f16", "f16"] as const);
  return {
    cache_type_k: [...new Set([...cache_type_k, fallbackPair[0]])],
    cache_type_v: [...new Set([...cache_type_v, fallbackPair[1]])],
  };
}

export const DEFAULT_RECOMMENDED_KV_PAIRS: ReadonlyArray<readonly [string, string]> = [
  ["f16", "f16"],
  ["f16", "q8_0"],
  ["q8_0", "q8_0"],
  ["q8_0", "q4_0"],
];

export function recommendedKvGrid(tolerance: KvTolerance): Array<readonly [string, string]> {
  const pruned = DEFAULT_RECOMMENDED_KV_PAIRS.filter(([k, v]) => pairAllowedUnderTolerance(k, v, tolerance));
  if (pruned.length > 0) return pruned;
  return [["f16", "f16"]];
}
