// LlamaToaster VRAM host (Chrome native messaging, Linux and macOS).
// Registered by the worker installer/updater (register.sh) so the LlamaToaster
// Chrome extension can read VRAM. Node built-ins only, read-only.
//   Linux: AMD and Intel (amdgpu, xe, i915) via DRM sysfs, NVIDIA via nvidia-smi.
//   macOS: unified memory via sysctl + vm_stat.
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, realpathSync } from "node:fs";
import { basename, join } from "node:path";

const MIB = 1024 * 1024;
const VENDORS = { 0x10de: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel" };
const rd = (p) => { try { return readFileSync(p, "utf8").trim(); } catch { return null; } };
const num = (p) => { const v = rd(p); if (!v) return null; const n = v.startsWith("0x") ? parseInt(v, 16) : Number(v); return Number.isFinite(n) ? n : null; };
const run = (cmd, args) => { try { return execFileSync(cmd, args, { encoding: "utf8", timeout: 10_000, stdio: ["ignore", "pipe", "ignore"] }); } catch { return ""; } };
const glob1 = (dir, re) => { try { return readdirSync(dir).filter((n) => re.test(n)).map((n) => join(dir, n)); } catch { return []; } };

function linuxDrm() {
  const out = [];
  for (const card of glob1("/sys/class/drm", /^card\d+$/)) {
    const dev = join(card, "device");
    const vid = parseInt(rd(join(dev, "vendor")) ?? "0", 16);
    if (vid === 0x10de) continue; // nvidia-smi below
    let driver = "?";
    try { driver = basename(realpathSync(join(dev, "driver"))); } catch {}
    const did = parseInt(rd(join(dev, "device")) ?? "0", 16);
    let total = num(join(dev, "mem_info_vram_total"));
    let used = num(join(dev, "mem_info_vram_used"));
    if (total == null) {
      // xe: per-tile VRAM regions
      for (const tile of glob1(dev, /^tile\d+$/)) {
        for (const p of [...glob1(tile, /^vram\d+$/).map((v) => join(v, "io_size")), join(tile, "physical_vram_size_bytes")]) {
          const v = num(p);
          if (v) total = (total ?? 0) + v;
        }
      }
    }
    if (total == null) {
      // i915 discrete (newer kernels)
      total = num(join(card, "lmem_total_bytes"));
      const avail = num(join(card, "lmem_avail_bytes"));
      if (total != null && avail != null) used = total - avail;
    }
    out.push({
      name: `${VENDORS[vid] ?? "0x" + vid.toString(16)} GPU ${did.toString(16).padStart(4, "0")} (${driver})`,
      vendor: VENDORS[vid] ?? "0x" + vid.toString(16), vendorId: vid, deviceId: did,
      integrated: total == null || total <= 512 * MIB,
      dedicatedTotalMib: total ? Math.round(total / MIB) : 0, sharedTotalMib: null,
      dedicatedUsedMib: used != null ? Math.round(used / MIB) : null, sharedUsedMib: null,
    });
  }
  return out;
}

function nvidia() {
  const csv = run("nvidia-smi", ["--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]);
  return csv.split("\n").map((l) => l.split(",").map((s) => s.trim())).filter((p) => p.length >= 3 && p[0]).map((p) => ({
    name: p[0], vendor: "NVIDIA", vendorId: 0x10de, deviceId: 0, integrated: false,
    dedicatedTotalMib: Math.round(Number(p[1])) || 0, sharedTotalMib: null,
    dedicatedUsedMib: Number.isFinite(Number(p[2])) ? Math.round(Number(p[2])) : null, sharedUsedMib: null,
  }));
}

function mac() {
  const total = Number(run("sysctl", ["-n", "hw.memsize"]).trim()) || 0;
  const vm = run("vm_stat", []);
  const page = Number(/page size of (\d+)/.exec(vm)?.[1] ?? 16384);
  const pages = (k) => Number(new RegExp(`${k}:\s+(\d+)`).exec(vm)?.[1] ?? 0);
  const free = (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")) * page;
  const chip = run("sysctl", ["-n", "machdep.cpu.brand_string"]).trim();
  return [{ name: chip || "Apple GPU", vendor: "Apple", vendorId: 0x106b, deviceId: 0, integrated: true,
    dedicatedTotalMib: 0, sharedTotalMib: Math.round(total / MIB), dedicatedUsedMib: null, sharedUsedMib: Math.round((total - free) / MIB) }];
}

// What llama.cpp processes hold: resident RAM from ps; VRAM only where the driver names processes (nvidia-smi).
function llamaUsage() {
  const rows = run("ps", ["-axo", "pid=,rss=,comm="]).split("\n").map((l) => l.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/)).filter(Boolean);
  const mine = rows.filter((m) => /(^|\/)llama-/.test(m[3]));
  const pids = new Set(mine.map((m) => m[1]));
  const ramMib = Math.round(mine.reduce((a, m) => a + Number(m[2]), 0) / 1024);
  let nvMib = 0;
  for (const l of run("nvidia-smi", ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).split("\n")) {
    const [pid, mem] = l.split(",").map((s) => s.trim());
    if (pids.has(pid)) nvMib += Number(mem) || 0;
  }
  return { ramMib, nvMib };
}

// The apps holding the most memory, summed per app name: resident RAM from ps
// (above 256 MiB, top 8) and, on NVIDIA, VRAM from nvidia-smi's compute apps.
function topApps() {
  const rows = run("ps", ["-axo", "pid=,rss=,comm="]).split("\n").map((l) => l.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/)).filter(Boolean);
  const nameOf = new Map();
  const ram = new Map();
  for (const m of rows) {
    const name = basename(m[3]);
    nameOf.set(m[1], name);
    ram.set(name, (ram.get(name) ?? 0) + Number(m[2]) * 1024);
  }
  const vram = new Map();
  for (const l of run("nvidia-smi", ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).split("\n")) {
    const [pid, mem] = l.split(",").map((s) => s.trim());
    if (!pid || !Number(mem)) continue;
    const name = nameOf.get(pid) ?? `pid ${pid}`;
    vram.set(name, (vram.get(name) ?? 0) + Number(mem) * MIB);
  }
  const top = (m, minBytes) => [...m].filter(([, v]) => v >= minBytes).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([name, v]) => ({ name, mib: Math.round(v / MIB) }));
  return { ram: top(ram, 256 * MIB), nvidia: top(vram, 128 * MIB) };
}

// Installed RAM: type and speed. macOS: system_profiler. Linux: dmidecode,
// which needs root on most systems (then this is null).
function readRam() {
  if (process.platform === "darwin") {
    try {
      const j = JSON.parse(run("system_profiler", ["SPMemoryDataType", "-json"]) || "{}");
      const top = j.SPMemoryDataType?.[0] ?? {};
      const items = top._items ?? [];
      const type = top.dimm_type ?? items[0]?.dimm_type ?? null;
      const speeds = items.map((i) => parseInt(i.dimm_speed, 10)).filter((n) => n > 0);
      return type || speeds.length ? { type, mts: speeds.length ? Math.min(...speeds) : null, ratedMts: null, modules: items.length || 1 } : null;
    } catch { return null; }
  }
  const out = run("dmidecode", ["-t", "17"]);
  const mods = out.split(/\n(?=Memory Device)/).filter((b) => /Size:\s*\d/.test(b));
  if (!mods.length) return null;
  const pick = (b, re) => re.exec(b)?.[1] ?? null;
  const type = mods.map((b) => pick(b, /\n\s*Type:\s*(\S+)/)).find((t) => t && t !== "Unknown") ?? null;
  const run_ = mods.map((b) => parseInt(pick(b, /Configured (?:Memory|Clock) Speed:\s*(\d+)/) ?? pick(b, /\n\s*Speed:\s*(\d+)/) ?? "", 10)).filter((n) => n > 0);
  const rated = mods.map((b) => parseInt(pick(b, /\n\s*Speed:\s*(\d+)/) ?? "", 10)).filter((n) => n > 0);
  const mts = run_.length ? Math.min(...run_) : null;
  const r = rated.length ? Math.min(...rated) : null;
  return { type, mts, ratedMts: r && mts && r > mts ? r : null, modules: mods.length };
}

const readGpus = () => (process.platform === "darwin" ? ["macos", mac()] : ["linux", [...linuxDrm(), ...nvidia()]]);

function send(obj) {
  const b = Buffer.from(JSON.stringify(obj));
  const h = Buffer.alloc(4);
  h.writeUInt32LE(b.length);
  process.stdout.write(Buffer.concat([h, b]));
}

if (process.argv.includes("--print")) {
  console.log(JSON.stringify(readGpus()[1], null, 2));
  console.log(JSON.stringify(readRam()));
} else {
  let acc = Buffer.alloc(0);
  process.stdin.on("data", (d) => {
    acc = Buffer.concat([acc, d]);
    while (acc.length >= 4 && acc.length >= 4 + acc.readUInt32LE(0)) {
      const len = acc.readUInt32LE(0);
      let req = {};
      try { req = JSON.parse(acc.subarray(4, 4 + len).toString("utf8")); } catch {}
      acc = acc.subarray(4 + len);
      try {
        const [os, gpus] = readGpus();
        const llama = llamaUsage();
        const apps = req.apps ? topApps() : null;
        for (const g of gpus) {
          g.llamaDedicatedMib = g.vendor === "NVIDIA" ? llama.nvMib : 0;
          g.llamaSharedMib = 0;
          // nvidia-smi names processes; AMD/Intel DRM sysfs does not.
          if (apps) g.apps = g.vendor === "NVIDIA" && gpus.filter((x) => x.vendor === "NVIDIA").length === 1 ? apps.nvidia : null;
        }
        send({ ok: true, helper: 1, os, gpus, ram: req.ram ? readRam() : null, llamaRamMib: llama.ramMib, topRam: apps ? apps.ram : null });
      } catch (e) {
        send({ ok: false, helper: 1, error: String(e?.message ?? e) });
      }
    }
  });
}
