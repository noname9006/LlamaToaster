export type GoalKind = "balanced" | "max_speed" | "max_context";
export type WorkloadShape = "chat" | "docs" | "even";
export type KvPreset = "compact" | "basic" | "extended" | "comprehensive";

export interface GoalsConfig {
  goal: GoalKind;
  target_ctx?: number | null;
  speed_floor_frac?: number;
  workload: WorkloadShape;
  kv_preset?: KvPreset;
}

export const GOAL_KINDS: readonly GoalKind[] = ["balanced", "max_speed", "max_context"];
export const WORKLOAD_SHAPES: readonly WorkloadShape[] = ["chat", "docs", "even"];
export const KV_PRESETS: readonly KvPreset[] = ["compact", "basic", "extended", "comprehensive"];
export const SPEED_FLOOR_CHOICES = [0.4, 0.5, 0.6] as const;

export const WORKLOAD_WEIGHTS: Record<WorkloadShape, { wPP: number; wTG: number }> = {
  chat: { wPP: 0.25, wTG: 0.75 },
  docs: { wPP: 0.7, wTG: 0.3 },
  even: { wPP: 0.5, wTG: 0.5 },
};

export function defaultGoals(): GoalsConfig {
  return { goal: "balanced", target_ctx: null, speed_floor_frac: 0.5, workload: "even", kv_preset: "extended" };
}

function oneOf<T extends string>(raw: unknown, allowed: readonly T[]): T | undefined {
  return typeof raw === "string" && (allowed as readonly string[]).includes(raw) ? (raw as T) : undefined;
}

export function normalizeGoals(raw: unknown): GoalsConfig | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const goal = oneOf(r.goal, GOAL_KINDS) ?? defaultGoals().goal;
  const workload = oneOf(r.workload, WORKLOAD_SHAPES) ?? defaultGoals().workload;
  const kv_preset = oneOf(r.kv_preset, KV_PRESETS) ?? defaultGoals().kv_preset;
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
  return { goal, target_ctx, speed_floor_frac, workload, kv_preset };
}

export function goalsEqualDefaults(goals: GoalsConfig | undefined): boolean {
  if (!goals) return true;
  const d = defaultGoals();
  return (
    goals.goal === d.goal &&
    (goals.target_ctx == null || goals.target_ctx === d.target_ctx) &&
    (goals.speed_floor_frac ?? d.speed_floor_frac) === d.speed_floor_frac &&
    goals.workload === d.workload &&
    (goals.kv_preset ?? d.kv_preset) === d.kv_preset
  );
}

// Four curated (K,V) grids, each a strict superset of the one before it --
// Compact ⊂ Basic ⊂ Extended ⊂ Comprehensive -- so picking a bigger tier is
// always strictly "more", never a lateral trade against what a smaller tier
// already covered. Every tier keeps f16/f16 as its baseline, which is why no
// separate "ensure an unquantized pair survives" fallback is needed here the
// way the old tolerance-pruning machinery required: there is nothing left to
// prune at run time, these are exactly the pairs that run.
export const KV_PRESET_PAIRS: Record<KvPreset, ReadonlyArray<readonly [string, string]>> = {
  // Fast sanity check: baseline, mild (both q8_0), and the popular
  // long-context setup (q8_0 K / q4_0 V).
  compact: [
    ["f16", "f16"],
    ["q8_0", "q8_0"],
    ["q8_0", "q4_0"],
  ],
  // Compact + bf16/bf16, so it also answers "does bf16 behave like f16".
  basic: [
    ["f16", "f16"],
    ["bf16", "bf16"],
    ["q8_0", "q8_0"],
    ["q8_0", "q4_0"],
  ],
  // Basic + K-only and V-only isolation at the mild (q8_0) level, plus both
  // sides pushed to the aggressive (q4_0) level together.
  extended: [
    ["f16", "f16"],
    ["bf16", "bf16"],
    ["q8_0", "f16"],
    ["f16", "q8_0"],
    ["q8_0", "q8_0"],
    ["q8_0", "q4_0"],
    ["q4_0", "q4_0"],
  ],
  // Extended + the asymmetric-quantized corner (K aggressive / V mild), the
  // two full-range corners (one side untouched, the other pushed straight to
  // q4_0), a middle bit-width rung (q5_0, between q8_0 and q4_0), and an
  // alternative 4-bit codec (iq4_nl) compared against q4_0 at the same size.
  comprehensive: [
    ["f16", "f16"],
    ["bf16", "bf16"],
    ["q8_0", "f16"],
    ["f16", "q8_0"],
    ["q8_0", "q8_0"],
    ["q5_0", "q5_0"],
    ["q8_0", "q4_0"],
    ["q4_0", "q8_0"],
    ["q4_0", "q4_0"],
    ["f16", "q4_0"],
    ["q4_0", "f16"],
    ["iq4_nl", "iq4_nl"],
  ],
};

export function kvPresetPairs(preset: KvPreset | undefined): ReadonlyArray<readonly [string, string]> {
  return KV_PRESET_PAIRS[preset ?? "extended"];
}
