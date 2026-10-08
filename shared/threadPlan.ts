// Step T of the optimization flow (docs/plans/OPTIMIZATION_FLOW_REDESIGN.md):
// CPU thread defaults from the machine's topology, plus the optional sweep.
//
// Only physical cores are ever used -- one logical CPU per core, SMT siblings
// left out (user decision 2026-10-04). Defaults:
//   single CCD          -t = -tb = all physical cores
//   multi-CCD           -t = -tb = one CCD (the larger-L3 one on X3D parts)
//   hybrid P/E          -t = P-cores, -tb = all physical (P+E)
//   macOS               no affinity masks; -t = P-cores, -tb = all physical
// --cpu-strict 1 pins thread i to the i-th set bit of the mask, so a run is
// deployed with the same placement it was measured with.

export interface CpuCore {
  /** Physical core index (OS order). */
  id: number;
  /** Logical CPU numbers on this core (SMT siblings), ascending. */
  logical: number[];
  /** L3 domain id (cores sharing one L3). */
  l3: number | null;
  /** Die / CCD id when the OS reports one. */
  die: number | null;
  /** Higher = faster core (Windows EfficiencyClass, Linux cpu_core vs atom). */
  effClass: number;
  /** Windows processor group. */
  group?: number;
}

export interface CpuTopology {
  source: "windows" | "linux" | "macos" | "fallback";
  cores: CpuCore[];
  logicalCount: number;
  /** L3 size per domain id, KiB, when known. */
  l3SizesKib?: Record<string, number>;
  /** Whether -C/-Cb masks can be passed on this OS. */
  masksSupported: boolean;
  /** CCD boundaries were inferred from core count and L3 size, not read. */
  estimated?: boolean;
  /** macOS: P/E counts from sysctl hw.perflevel*. */
  perfCores?: number;
  effCores?: number;
}

export type ThreadCandidateKind = "default" | "one_core" | "half" | "two_ccd" | "p_plus_e" | "p_only";

export interface ThreadSetting {
  threads: number;
  /** Hex affinity mask, one logical CPU per used core; null = no mask. */
  mask: string | null;
  kind: ThreadCandidateKind;
  label: string;
}

export interface ThreadPlan {
  t: ThreadSetting;
  tb: ThreadSetting;
  strict: boolean;
  /** One line explaining the default. */
  reason: string;
  /** "Ryzen 5 5600X · 1 CCD · 6 cores / 12 threads"-style summary minus the brand. */
  summary: string;
  domainLabel: "CCD" | "CCX";
  domains: number;
  hybrid: boolean;
  note: string | null;
}

interface Domain {
  id: number;
  cores: CpuCore[];
  l3Kib: number | null;
}

function domainsOf(topo: CpuTopology): { domains: Domain[]; label: "CCD" | "CCX" } {
  const dies = new Set(topo.cores.map((c) => c.die).filter((d): d is number => d != null));
  const l3s = new Set(topo.cores.map((c) => c.l3).filter((d): d is number => d != null));
  let key: "die" | "l3" | null = null;
  let label: "CCD" | "CCX" = "CCD";
  if (dies.size > 1) key = "die";
  else if (l3s.size > 1) {
    key = "l3";
    // A single reported die that still splits its L3 is a Zen 2-style CCX.
    if (dies.size === 1) label = "CCX";
  }
  if (!key) return { domains: [{ id: 0, cores: topo.cores, l3Kib: null }], label };
  const by = new Map<number, CpuCore[]>();
  for (const c of topo.cores) {
    const id = (key === "die" ? c.die : c.l3) ?? -1;
    by.set(id, [...(by.get(id) ?? []), c]);
  }
  const domains = [...by.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([id, cores]) => {
      const l3 = cores[0].l3;
      const kib = l3 != null ? topo.l3SizesKib?.[String(l3)] ?? null : null;
      return { id, cores, l3Kib: kib };
    });
  return { domains, label };
}

/** Hex mask with one bit per core (its first logical CPU). */
export function coreMask(cores: readonly CpuCore[]): string {
  let mask = 0n;
  for (const c of cores) if (c.logical.length) mask |= 1n << BigInt(c.logical[0]);
  return "0x" + mask.toString(16);
}

function masksUsable(topo: CpuTopology, cores: readonly CpuCore[]): boolean {
  if (!topo.masksSupported) return false;
  // Windows processor groups (> 64 logical CPUs): a mask can only address one
  // group, so only pass one when every core lies in the same group.
  // How llama.cpp maps mask bits beyond 64 onto Windows groups is unverified,
  // so masks are only passed when every core sits in the first 64 CPUs.
  return cores.every((c) => (c.group ?? 0) === 0 && c.logical.every((l) => l < 64));
}

function setting(topo: CpuTopology, cores: readonly CpuCore[], kind: ThreadCandidateKind, label: string): ThreadSetting {
  const n = Math.max(1, cores.length);
  return { threads: n, mask: masksUsable(topo, cores) ? coreMask(cores) : null, kind, label };
}

function pCoresOf(topo: CpuTopology): { p: CpuCore[]; e: CpuCore[] } {
  const classes = [...new Set(topo.cores.map((c) => c.effClass))].sort((a, b) => b - a);
  if (classes.length < 2) return { p: topo.cores, e: [] };
  return { p: topo.cores.filter((c) => c.effClass === classes[0]), e: topo.cores.filter((c) => c.effClass !== classes[0]) };
}

function preferredDomain(domains: Domain[]): Domain {
  const sized = domains.filter((d) => d.l3Kib != null);
  if (sized.length === domains.length && new Set(sized.map((d) => d.l3Kib)).size > 1) {
    return [...domains].sort((a, b) => (b.l3Kib ?? 0) - (a.l3Kib ?? 0) || a.id - b.id)[0];
  }
  return domains[0];
}

export function defaultThreadPlan(topo: CpuTopology): ThreadPlan {
  const physical = topo.cores.length;
  const summaryBase = `${physical} cores / ${topo.logicalCount} threads`;
  if (topo.source === "macos") {
    const p = topo.perfCores && topo.perfCores > 0 ? topo.perfCores : physical;
    return {
      t: { threads: p, mask: null, kind: "default", label: topo.perfCores ? "P-cores" : "all physical cores" },
      tb: { threads: physical, mask: null, kind: "default", label: "all physical cores" },
      strict: false,
      reason: topo.perfCores
        ? "Generation on the performance cores, prompt processing on every physical core. macOS has no thread-affinity API, so no masks are passed."
        : "Every physical core. macOS has no thread-affinity API, so no masks are passed.",
      summary: topo.perfCores ? `${topo.perfCores} P + ${topo.effCores ?? 0} E cores` : summaryBase,
      domainLabel: "CCD",
      domains: 1,
      hybrid: !!topo.effCores,
      note: null,
    };
  }

  const { p, e } = pCoresOf(topo);
  const hybrid = e.length > 0;
  const { domains, label } = domainsOf(topo);
  const note = topo.logicalCount > 64 && !topo.cores.every((c) => (c.group ?? 0) === (topo.cores[0].group ?? 0))
    ? "More than 64 logical CPUs across processor groups: thread counts are set, affinity masks are not."
    : topo.estimated
      ? "The CCD split is estimated from the core count and L3 size, not read from the OS."
      : null;

  if (hybrid) {
    return {
      t: setting(topo, p, "default", "P-cores"),
      tb: setting(topo, topo.cores, "default", "all physical cores (P+E)"),
      strict: true,
      reason: `Generation runs on the ${p.length} performance cores; prompt processing uses all ${physical} physical cores.`,
      summary: `${p.length} P + ${e.length} E cores · ${topo.logicalCount} threads`,
      domainLabel: label,
      domains: 1,
      hybrid,
      note,
    };
  }
  if (domains.length > 1) {
    const d = preferredDomain(domains);
    const why = d.l3Kib != null && domains.some((x) => x.l3Kib !== d.l3Kib) ? " (the one with the larger L3 cache)" : "";
    return {
      t: setting(topo, d.cores, "default", `one ${label}`),
      tb: setting(topo, d.cores, "default", `one ${label}`),
      strict: true,
      reason: `Both run on the ${d.cores.length} cores of one ${label}${why}: crossing ${label}s adds memory latency that usually costs more than the extra cores give.`,
      summary: `${domains.length} ${label}s · ${summaryBase}`,
      domainLabel: label,
      domains: domains.length,
      hybrid,
      note,
    };
  }
  return {
    t: setting(topo, topo.cores, "default", "all physical cores"),
    tb: setting(topo, topo.cores, "default", "all physical cores"),
    strict: true,
    reason: `Both use all ${physical} physical cores, one thread per core (SMT siblings left out).`,
    summary: `1 ${label} · ${summaryBase}`,
    domainLabel: label,
    domains: 1,
    hybrid,
    note,
  };
}

export interface ThreadCandidate extends ThreadSetting {
  /** Pre-ticked in the sweep card (the "next step" candidate). */
  preselected: boolean;
}

/**
 * Sweep candidates for -t (role "t") or -tb (role "tb"). The "next step"
 * candidate comes first and is pre-ticked; 1 core and half follow; the
 * default is always measured, it is the reference.
 */
export function threadCandidates(topo: CpuTopology, role: "t" | "tb"): ThreadCandidate[] {
  const plan = defaultThreadPlan(topo);
  const def = role === "t" ? plan.t : plan.tb;
  const out: ThreadCandidate[] = [];
  const push = (s: ThreadSetting, preselected: boolean) => {
    if (!out.some((c) => c.threads === s.threads && c.mask === s.mask)) out.push({ ...s, preselected });
  };
  // Cores of the default set, in preference order -- candidates are prefixes.
  let defCores: CpuCore[] = topo.cores;
  if (topo.source !== "macos") {
    const { p, e } = pCoresOf(topo);
    const { domains, label } = domainsOf(topo);
    if (e.length > 0) {
      defCores = role === "t" ? p : topo.cores;
      if (role === "t") push(setting(topo, [...p, ...e], "p_plus_e", "P + E cores"), true);
      else push(setting(topo, p, "p_only", "P-cores only"), true);
    } else if (domains.length > 1) {
      const d = preferredDomain(domains);
      defCores = d.cores;
      const next = domains.find((x) => x.id !== d.id)!;
      push(setting(topo, [...d.cores, ...next.cores], "two_ccd", `2 ${label}s`), true);
    }
  }
  const take = (n: number, kind: ThreadCandidateKind, label: string) =>
    topo.source === "macos"
      ? ({ threads: n, mask: null, kind, label } as ThreadSetting)
      : setting(topo, defCores.slice(0, n), kind, label);
  push(take(1, "one_core", "1 core"), false);
  const half = Math.max(1, Math.floor(def.threads / 2));
  if (half > 1) push(take(half, "half", `half (${half})`), false);
  push(def, true);
  return out;
}

export interface MeasuredCandidate {
  kind: ThreadCandidateKind;
  mean: number;
  stddev: number;
  samples: number;
}

export type WinnerVerdict =
  | { kind: "winner"; winner: ThreadCandidateKind }
  | { kind: "default_kept"; reason: "no_clear_winner" | "no_spread" | "no_default" };

/**
 * A candidate replaces the default only when its mean beats the default's by
 * more than both standard deviations combined. With a single repeat there is
 * no spread, so the default is never replaced.
 */
export function pickThreadWinner(rows: readonly MeasuredCandidate[]): WinnerVerdict {
  const def = rows.find((r) => r.kind === "default");
  if (!def) return { kind: "default_kept", reason: "no_default" };
  if (def.samples < 2) return { kind: "default_kept", reason: "no_spread" };
  let best: MeasuredCandidate | null = null;
  for (const r of rows) {
    if (r.kind === "default" || r.samples < 2) continue;
    if (r.mean - def.mean > r.stddev + def.stddev && (!best || r.mean > best.mean)) best = r;
  }
  return best ? { kind: "winner", winner: best.kind } : { kind: "default_kept", reason: "no_clear_winner" };
}

/** A topology for machines whose worker predates detection: core count only. */
export function fallbackTopology(cores: number): CpuTopology {
  const n = Math.max(1, Math.floor(cores));
  return {
    source: "fallback",
    cores: Array.from({ length: n }, (_, i) => ({ id: i, logical: [i], l3: null, die: null, effClass: 0 })),
    logicalCount: n,
    masksSupported: false,
  };
}
