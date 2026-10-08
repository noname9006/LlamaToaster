// Per-core CPU topology for the optimization flow's thread step
// (shared/threadPlan.ts): which logical CPUs share a physical core, an L3
// cache and a die, and which cores are efficiency cores. Read once at worker
// startup and sent with the hardware report. Best-effort everywhere: any
// failure returns null and the flow falls back to the plain core count.
//
//   Windows  systeminformation counts (no script compilation -- see below)
//   Linux    /sys/devices/system/cpu/cpu*/topology + cache/index3, hybrid
//            via /sys/devices/cpu_core|cpu_atom/cpus or cpu_capacity
//   macOS    sysctl hw.perflevel*.physicalcpu (no affinity API, no masks)

import { execFile } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { promisify } from "node:util";
import type { CpuCore, CpuTopology } from "../../shared/threadPlan.js";

const execFileAsync = promisify(execFile);

// --- Windows -----------------------------------------------------------------

// Built from systeminformation's counts (the same data the worker's hardware
// report already reads) rather than from GetLogicalProcessorInformationEx:
// calling that API means compiling C# through PowerShell Add-Type, which
// endpoint protection (Symantec SONAR, observed 2026-10-08) blocks as a
// suspicious script. Windows numbers logical CPUs core by core -- SMT
// siblings adjacent, P-cores before E-cores -- so the per-core masks follow
// from the counts. What the counts can't say is where one CCD ends; on AMD
// parts with more L3 than one CCD carries it is estimated (8 cores and 32 MiB
// per CCD on Zen 3 and later) and the topology is marked as estimated.

export interface WindowsCpuCounts {
  brand: string;
  manufacturer: string;
  logical: number;
  physical: number;
  efficiencyCores: number;
  l3Bytes: number | null;
}

const CCD_L3_KIB = 32 * 1024;
const CCD_CORES = 8;

export function synthesizeWindowsTopology(c: WindowsCpuCounts): CpuTopology | null {
  const physical = Math.floor(c.physical);
  const logical = Math.floor(c.logical);
  if (!(physical > 0) || !(logical >= physical)) return null;
  let eRaw = Math.floor(c.efficiencyCores || 0);
  // Not every systeminformation path reports E-cores on Windows. An Intel
  // part with more threads than cores but fewer than two per core is hybrid:
  // the P-cores have SMT, the E-cores don't (16 cores / 24 threads = 8P + 8E).
  if (eRaw === 0 && /intel/i.test(`${c.manufacturer} ${c.brand}`) && logical > physical && logical < 2 * physical) {
    eRaw = physical - (logical - physical);
  }
  const e = Math.max(0, Math.min(physical - 1, eRaw));
  const p = physical - e;
  // P-cores carry the SMT threads; E-cores have one each.
  const smtThreads = logical - e;
  const perP = p > 0 ? Math.max(1, Math.round(smtThreads / p)) : 1;
  const cores: CpuCore[] = [];
  let next = 0;
  for (let i = 0; i < p; i++) {
    cores.push({ id: i, logical: Array.from({ length: perP }, (_, k) => next + k), l3: 0, die: null, effClass: e > 0 ? 1 : 0, group: Math.floor(next / 64) });
    next += perP;
  }
  for (let i = 0; i < e; i++) {
    cores.push({ id: p + i, logical: [next], l3: 0, die: null, effClass: 0, group: Math.floor(next / 64) });
    next += 1;
  }
  let estimated = false;
  const l3Kib = c.l3Bytes && c.l3Bytes > 0 ? Math.round(c.l3Bytes / 1024) : null;
  const l3SizesKib: Record<string, number> = {};
  const amd = /amd|ryzen|threadripper|epyc/i.test(`${c.manufacturer} ${c.brand}`);
  if (amd && e === 0 && physical > CCD_CORES && l3Kib != null && l3Kib > CCD_L3_KIB * 1.25) {
    const domains = Math.max(2, Math.ceil(physical / CCD_CORES));
    const per = Math.ceil(physical / domains);
    cores.forEach((core, i) => (core.die = Math.floor(i / per)));
    estimated = true;
  } else if (l3Kib != null) {
    l3SizesKib["0"] = l3Kib;
  }
  return {
    source: "windows",
    cores,
    logicalCount: logical,
    ...(Object.keys(l3SizesKib).length ? { l3SizesKib } : {}),
    masksSupported: true,
    ...(estimated ? { estimated: true } : {}),
  };
}

export interface SiCpuLike {
  brand?: string;
  manufacturer?: string;
  cores: number;
  physicalCores: number;
  efficiencyCores?: number;
  cache?: { l3?: number };
}

async function detectWindows(siCpu?: SiCpuLike): Promise<CpuTopology | null> {
  const cpu: SiCpuLike = siCpu ?? (await (await import("systeminformation")).default.cpu());
  return synthesizeWindowsTopology({
    brand: cpu.brand ?? "",
    manufacturer: cpu.manufacturer ?? "",
    logical: cpu.cores,
    physical: cpu.physicalCores,
    efficiencyCores: typeof cpu.efficiencyCores === "number" ? cpu.efficiencyCores : 0,
    l3Bytes: typeof cpu.cache?.l3 === "number" ? cpu.cache.l3 : null,
  });
}

// --- Linux -------------------------------------------------------------------

/** "0-3,8,10-11" -> [0,1,2,3,8,10,11] */
export function parseCpuList(text: string): number[] {
  const out: number[] = [];
  for (const part of text.trim().split(",")) {
    if (!part) continue;
    const [a, b] = part.split("-").map(Number);
    if (!Number.isFinite(a)) continue;
    for (let i = a; i <= (Number.isFinite(b) ? b : a); i++) out.push(i);
  }
  return out;
}

function readText(path: string): string | null {
  try {
    return readFileSync(path, "utf8").trim();
  } catch {
    return null;
  }
}

function parseCacheSize(text: string | null): number | null {
  if (!text) return null;
  const m = /^(\d+)\s*([KMG]?)/i.exec(text);
  if (!m) return null;
  const n = Number(m[1]);
  const unit = m[2].toUpperCase();
  return unit === "M" ? n * 1024 : unit === "G" ? n * 1024 * 1024 : n;
}

export interface LinuxCpuEntry {
  cpu: number;
  coreKey: string;
  die: number | null;
  l3List: string | null;
  l3SizeKib: number | null;
  effClass: number;
}

export function buildLinuxTopology(entries: readonly LinuxCpuEntry[]): CpuTopology | null {
  if (entries.length === 0) return null;
  const byCore = new Map<string, LinuxCpuEntry[]>();
  for (const e of entries) byCore.set(e.coreKey, [...(byCore.get(e.coreKey) ?? []), e]);
  const l3Ids = new Map<string, number>();
  const l3SizesKib: Record<string, number> = {};
  const cores: CpuCore[] = [...byCore.values()]
    .map((list) => list.sort((a, b) => a.cpu - b.cpu))
    .sort((a, b) => a[0].cpu - b[0].cpu)
    .map((list, id) => {
      const first = list[0];
      let l3: number | null = null;
      if (first.l3List) {
        if (!l3Ids.has(first.l3List)) {
          l3Ids.set(first.l3List, l3Ids.size);
          if (first.l3SizeKib != null) l3SizesKib[String(l3Ids.size - 1)] = first.l3SizeKib;
        }
        l3 = l3Ids.get(first.l3List)!;
      }
      return { id, logical: list.map((e) => e.cpu), l3, die: first.die, effClass: first.effClass };
    });
  return { source: "linux", cores, logicalCount: entries.length, l3SizesKib, masksSupported: true };
}

function detectLinux(): CpuTopology | null {
  const base = "/sys/devices/system/cpu";
  if (!existsSync(base)) return null;
  const cpus = readdirSync(base)
    .filter((n) => /^cpu\d+$/.test(n))
    .map((n) => Number(n.slice(3)))
    .filter((n) => readText(join(base, `cpu${n}`, "online")) !== "0");
  const pCores = new Set(parseCpuList(readText("/sys/devices/cpu_core/cpus") ?? ""));
  const eCores = new Set(parseCpuList(readText("/sys/devices/cpu_atom/cpus") ?? ""));
  const capacities = new Map<number, number>();
  for (const c of cpus) {
    const cap = Number(readText(join(base, `cpu${c}`, "cpu_capacity")));
    if (Number.isFinite(cap) && cap > 0) capacities.set(c, cap);
  }
  const entries: LinuxCpuEntry[] = cpus.map((cpu) => {
    const topo = join(base, `cpu${cpu}`, "topology");
    const pkg = readText(join(topo, "physical_package_id")) ?? "0";
    const coreId = readText(join(topo, "core_id")) ?? String(cpu);
    const dieRaw = readText(join(topo, "die_id"));
    const die = dieRaw != null && dieRaw !== "-1" ? Number(pkg) * 1000 + Number(dieRaw) : null;
    let effClass = 0;
    if (pCores.size || eCores.size) effClass = pCores.has(cpu) ? 1 : 0;
    else if (capacities.size) effClass = capacities.get(cpu) ?? 0;
    return {
      cpu,
      coreKey: `${pkg}:${dieRaw ?? 0}:${coreId}`,
      die,
      l3List: readText(join(base, `cpu${cpu}`, "cache", "index3", "shared_cpu_list")),
      l3SizeKib: parseCacheSize(readText(join(base, `cpu${cpu}`, "cache", "index3", "size"))),
      effClass,
    };
  });
  return buildLinuxTopology(entries);
}

// --- macOS -------------------------------------------------------------------

async function sysctlNumber(key: string): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("sysctl", ["-n", key], { timeout: 5000 });
    const n = Number(stdout.trim());
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

async function detectMac(): Promise<CpuTopology | null> {
  const physical = await sysctlNumber("hw.physicalcpu");
  const logical = await sysctlNumber("hw.logicalcpu");
  if (!physical) return null;
  const perf = await sysctlNumber("hw.perflevel0.physicalcpu");
  const eff = await sysctlNumber("hw.perflevel1.physicalcpu");
  const cores: CpuCore[] = Array.from({ length: physical }, (_, i) => ({
    id: i,
    logical: [i],
    l3: null,
    die: null,
    effClass: perf != null && i < perf ? 1 : 0,
  }));
  return {
    source: "macos",
    cores,
    logicalCount: logical ?? physical,
    masksSupported: false,
    perfCores: perf ?? undefined,
    effCores: eff ?? undefined,
  };
}

export async function detectCpuTopology(
  platform: NodeJS.Platform = process.platform,
  siCpu?: SiCpuLike
): Promise<CpuTopology | null> {
  try {
    if (platform === "win32") return await detectWindows(siCpu);
    if (platform === "linux") return detectLinux();
    if (platform === "darwin") return await detectMac();
  } catch {
    return null;
  }
  return null;
}
