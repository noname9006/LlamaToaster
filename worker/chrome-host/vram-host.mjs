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

const readGpus = () => (process.platform === "darwin" ? ["macos", mac()] : ["linux", [...linuxDrm(), ...nvidia()]]);

function send(obj) {
  const b = Buffer.from(JSON.stringify(obj));
  const h = Buffer.alloc(4);
  h.writeUInt32LE(b.length);
  process.stdout.write(Buffer.concat([h, b]));
}

if (process.argv.includes("--print")) {
  console.log(JSON.stringify(readGpus()[1], null, 2));
} else {
  let acc = Buffer.alloc(0);
  process.stdin.on("data", (d) => {
    acc = Buffer.concat([acc, d]);
    while (acc.length >= 4 && acc.length >= 4 + acc.readUInt32LE(0)) {
      acc = acc.subarray(4 + acc.readUInt32LE(0));
      try {
        const [os, gpus] = readGpus();
        send({ ok: true, helper: 1, os, gpus });
      } catch (e) {
        send({ ok: false, helper: 1, error: String(e?.message ?? e) });
      }
    }
  });
}
