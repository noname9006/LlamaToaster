#!/usr/bin/env python3
"""RAM / VRAM bandwidth benchmark.

  python scripts/membench.py fast          # ~10 s, one 1 GiB buffer each, coarse numbers
  python scripts/membench.py deep          # fills (almost) all free RAM / VRAM in chunks,
                                           # measures every chunk, flags slow regions
  python scripts/membench.py deep --no-vram   (or --no-ram)
  python scripts/membench.py deep --no-ram --gpu 1      # pick a GPU (default: all GPUs)

Needs numpy + psutil; VRAM needs pyopencl (pip install pyopencl).

Every run starts with a system block: CPU, DIMMs and memory clock (from Windows),
plus FCLK / UCLK / timings when HWiNFO is running with "Shared Memory Support" enabled
(Settings -> General). Windows itself can't report FCLK; it needs a kernel-level tool.

Deep mode notes:
  * Per-chunk speed differences in RAM usually come from asymmetric channel population
    (e.g. 8+16 GB: the upper 8 GB runs single-channel), not from DIMMs at different
    clocks - the memory controller runs all DIMMs at the slowest module's speed.
    The OS maps pages from both regions, so a slow region shows up as a band of slow chunks.
  * A slow tail in VRAM usually means the driver spilled the allocation into shared
    system memory (WDDM), i.e. you've reached the real usable VRAM.
"""
import argparse
import mmap
import os
import re
import struct
import subprocess
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

GiB = 1024 ** 3
MiB = 1024 ** 2
THREADS = os.cpu_count() or 4
POOL = ThreadPoolExecutor(THREADS)


# ---------------------------------------------------------------- system info

def _wmi_dimms():
    """[(slot, GiB, MT/s, part)] from Win32_PhysicalMemory, [] off Windows / on failure."""
    if os.name != "nt":
        return []
    ps = ("Get-CimInstance Win32_PhysicalMemory | ForEach-Object { "
          "'{0}|{1}|{2}|{3}|{4}' -f $_.BankLabel,$_.DeviceLocator,$_.Capacity,$_.ConfiguredClockSpeed,$_.PartNumber }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        f = line.strip().split("|")
        if len(f) == 5:
            rows.append((f"{f[0]} {f[1]}".strip(), int(f[2] or 0) / GiB, f[3], f[4].strip()))
    return rows


def _cpu_name():
    if os.name == "nt":
        try:
            return subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


HWINFO_SM = "Global\\HWiNFO_SENS_SM2"
HWINFO_KEYS = re.compile(r"fclk|uclk|mclk|fabric|memory controller clock|memory clock|"
                         r"^t(cas|cl|rcd|rp|ras|rc|rfc|faw|rrd|wr)\b|command rate|gear", re.I)


def _open_shared(name):
    """Read-only view of an existing named mapping as bytes, or None.

    mmap.mmap(tagname=...) calls CreateFileMapping, which a non-admin process is denied on
    HWiNFO's Global\\ object, so open it with OpenFileMapping + MapViewOfFile instead."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenFileMappingW.restype = wintypes.HANDLE
    k.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    FILE_MAP_READ = 4
    h = k.OpenFileMappingW(FILE_MAP_READ, False, name)
    if not h:
        return None
    try:
        view = k.MapViewOfFile(h, FILE_MAP_READ, 0, 0, 0)
        if not view:
            return None
        try:
            hdr = ctypes.string_at(view, 44)
            r_off, r_size, r_num = struct.unpack("<III", hdr[32:44])
            return hdr + ctypes.string_at(view + 44, max(0, r_off + r_size * r_num - 44))
        finally:
            k.UnmapViewOfFile(view)
    finally:
        k.CloseHandle(h)


def _hwinfo_readings():
    """FCLK/UCLK/MCLK/timing readings from HWiNFO shared memory, or None if HWiNFO isn't sharing."""
    if os.name != "nt":
        return None
    m = _open_shared(HWINFO_SM)
    if not m or len(m) < 44:
        return None
    sig, _ver, _rev, _t, _so, _ss, _sn, r_off, r_size, r_num = struct.unpack("<4sIIqIIIIII", m[:44])
    if sig not in (b"HWiS", b"SiWH") or not r_num:
        return None
    found = {}
    for i in range(r_num):
        b = m[r_off + i * r_size: r_off + (i + 1) * r_size]
        label = b[12:140].split(b"\0", 1)[0].decode("latin-1")
        unit = b[268:284].split(b"\0", 1)[0].decode("latin-1")
        value = struct.unpack("<d", b[284:292])[0]
        if HWINFO_KEYS.search(label) and "GPU" not in label and label not in found:
            found[label] = (value, unit)
    return found


def system_info():
    print(f"CPU: {_cpu_name() or 'unknown'}   RAM total: {psutil.virtual_memory().total / GiB:.1f} GiB")
    dimms = _wmi_dimms()
    for slot, gib, speed, part in dimms:
        print(f"  {slot:<22} {gib:4.0f} GiB  {speed:>5} MT/s  {part}")
    hw = _hwinfo_readings()
    if hw is None:
        print("  FCLK/UCLK/timings: not available (start HWiNFO with Shared Memory Support enabled)")
    elif not hw:
        print("  HWiNFO is sharing, but reported no FCLK/UCLK/timing sensors")
    else:
        print("  " + "   ".join(f"{k}: {v:g} {u}".rstrip() for k, (v, u) in hw.items()))
    print()


# ---------------------------------------------------------------- RAM

def _split(a):
    return np.array_split(a, THREADS)


def _par(fn, *arrs):
    parts = [ _split(a) for a in arrs ]
    list(POOL.map(lambda p: fn(*p), zip(*parts)))


def ram_measure(buf, scratch, reps):
    """Returns (read, write, copy) GB/s, best of reps. buf/scratch are uint64 arrays."""
    nbytes = buf.nbytes
    best = {"read": 0.0, "write": 0.0, "copy": 0.0}
    for _ in range(reps):
        t = time.perf_counter(); _par(lambda a: a.sum(), buf)
        best["read"] = max(best["read"], nbytes / (time.perf_counter() - t) / 1e9)
        t = time.perf_counter(); _par(lambda a: a.fill(7), buf)
        best["write"] = max(best["write"], nbytes / (time.perf_counter() - t) / 1e9)
        t = time.perf_counter(); _par(lambda d, s: np.copyto(d, s), scratch, buf)
        # copy moves 2x bytes (read + write)
        best["copy"] = max(best["copy"], 2 * nbytes / (time.perf_counter() - t) / 1e9)
    return best


def ram_fast():
    n = GiB // 8
    buf = np.ones(n, dtype=np.uint64)
    scratch = np.empty_like(buf)
    r = ram_measure(buf, scratch, reps=5)
    print(f"RAM  (1 GiB, {THREADS} threads)  read {r['read']:6.1f} GB/s   "
          f"write {r['write']:6.1f} GB/s   copy {r['copy']:6.1f} GB/s")


def ram_deep(chunk_mib, reserve_gib):
    avail = psutil.virtual_memory().available
    target = int(avail - reserve_gib * GiB)
    chunk = chunk_mib * MiB
    count = max(1, target // chunk)
    print(f"RAM deep: {psutil.virtual_memory().total / GiB:.1f} GiB total, "
          f"{avail / GiB:.1f} GiB free -> testing {count} x {chunk_mib} MiB "
          f"({count * chunk / GiB:.1f} GiB), leaving {reserve_gib} GiB free")
    chunks = []
    try:
        for i in range(count):
            c = np.empty(chunk // 8, dtype=np.uint64)
            c.fill(1)  # commit the pages
            chunks.append(c)
            print(f"\r  allocated {i + 1}/{count}", end="", flush=True)
    except MemoryError:
        print(f"\n  MemoryError after {len(chunks)} chunks, testing what we got")
    print()
    scratch = np.empty(chunk // 8, dtype=np.uint64)
    rows = []
    for i, c in enumerate(chunks):
        r = ram_measure(c, scratch, reps=3)
        rows.append(r)
        print(f"\r  measured {i + 1}/{len(chunks)}", end="", flush=True)
    print()
    _report("RAM", rows, chunk_mib)
    del chunks


# ---------------------------------------------------------------- VRAM

def _gpus():
    """All OpenCL GPUs, in a stable order (platform, then device)."""
    try:
        import pyopencl as cl
    except ImportError:
        print("VRAM: pyopencl not installed (pip install pyopencl) - skipped")
        return []
    devs = []
    for p in cl.get_platforms():
        try:
            devs += p.get_devices(cl.device_type.GPU)
        except cl.Error:
            pass
    if not devs:
        print("VRAM: no OpenCL GPU found - skipped")
    return devs


def _gpu_name(dev):
    # AMD reports the chip (gfx1032) as name; the marketing name is in board_name_amd
    try:
        return dev.board_name_amd.strip()
    except Exception:
        return dev.name.strip()


def _cl(index):
    import pyopencl as cl
    dev = _gpus()[index]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, """
    __kernel void rd(__global const ulong4* a, __global ulong* out, ulong n) {
        ulong4 s = 0; for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) s ^= a[i];
        if ((s.x ^ s.y ^ s.z ^ s.w) == 0x123456789abcdefUL) out[0] = 1; }
    __kernel void wr(__global ulong4* a, ulong n) {
        for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) a[i] = (ulong4)(i); }
    """).build()
    prg = {k: cl.Kernel(prg, k) for k in ("rd", "wr")}
    return cl, dev, ctx, q, prg


def vram_measure(env, buf, nbytes, scratch, reps):
    cl, dev, ctx, q, prg = env
    n4 = np.uint64(nbytes // 32)
    gsize = (dev.max_compute_units * 2048,)
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 8)
    best = {"read": 0.0, "write": 0.0, "copy": 0.0}

    def timed(f):
        q.finish(); t = time.perf_counter(); f(); q.finish()
        return time.perf_counter() - t

    for _ in range(reps):
        best["read"] = max(best["read"], nbytes / timed(lambda: cl.enqueue_nd_range_kernel(q, prg["rd"].set_args(buf, out, n4) or prg["rd"], gsize, None)) / 1e9)
        best["write"] = max(best["write"], nbytes / timed(lambda: cl.enqueue_nd_range_kernel(q, prg["wr"].set_args(buf, n4) or prg["wr"], gsize, None)) / 1e9)
        best["copy"] = max(best["copy"], 2 * nbytes / timed(
            lambda: cl.enqueue_copy(q, scratch, buf, byte_count=nbytes)) / 1e9)
    return best


def vram_fast(index):
    env = _cl(index)
    cl, dev, ctx, q, _ = env
    nbytes = GiB
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nbytes)
    scratch = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nbytes)
    r = vram_measure(env, buf, nbytes, scratch, reps=5)
    host = np.ones(nbytes // 8, dtype=np.uint64)
    t = time.perf_counter(); cl.enqueue_copy(q, buf, host); q.finish()
    h2d = nbytes / (time.perf_counter() - t) / 1e9
    t = time.perf_counter(); cl.enqueue_copy(q, host, buf); q.finish()
    d2h = nbytes / (time.perf_counter() - t) / 1e9
    print(f"VRAM [GPU {index}] ({_gpu_name(dev)}, 1 GiB)  read {r['read']:6.1f} GB/s   "
          f"write {r['write']:6.1f} GB/s   copy {r['copy']:6.1f} GB/s")
    print(f"PCIe host->GPU {h2d:5.1f} GB/s   GPU->host {d2h:5.1f} GB/s")


def vram_deep(index, chunk_mib, reserve_mib):
    env = _cl(index)
    cl, dev, ctx, q, _ = env
    chunk = chunk_mib * MiB
    total = dev.global_mem_size
    # one chunk is the copy scratch; try to fill the rest (may spill past real VRAM - that's the point)
    count = max(1, (total - reserve_mib * MiB) // chunk - 1)
    print(f"VRAM deep [GPU {index}]: {_gpu_name(dev)}, {total / GiB:.1f} GiB reported -> "
          f"{count} x {chunk_mib} MiB chunks + 1 scratch")
    scratch = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, chunk)
    chunks = []
    for i in range(count):
        try:
            b = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, chunk)
            cl.enqueue_fill_buffer(q, b, np.uint8(1), 0, chunk); q.finish()  # force residency
            chunks.append(b)
        except cl.Error as e:
            print(f"\n  allocation stopped at {len(chunks)} chunks ({e.code})")
            break
        print(f"\r  allocated {i + 1}/{count}", end="", flush=True)
    print()
    rows = []
    for i, b in enumerate(chunks):
        rows.append(vram_measure(env, b, chunk, scratch, reps=3))
        print(f"\r  measured {i + 1}/{len(chunks)}", end="", flush=True)
    print()
    _report(f"VRAM [GPU {index}] {_gpu_name(dev)}", rows, chunk_mib)


# ---------------------------------------------------------------- report

def _report(label, rows, chunk_mib, slow=0.85):
    if not rows:
        return
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    print(f"\n{label} per-chunk ({chunk_mib} MiB each)   median: read {med['read']:.1f}  "
          f"write {med['write']:.1f}  copy {med['copy']:.1f} GB/s")
    print(f"{'chunk':>6} {'offset':>9} {'read':>8} {'write':>8} {'copy':>8}")
    flagged = []
    for i, r in enumerate(rows):
        bad = [k for k in r if r[k] < slow * med[k]]
        if bad:
            flagged.append(i)
        mark = "  <-- slow " + ",".join(bad) if bad else ""
        print(f"{i:>6} {i * chunk_mib / 1024:>7.2f}G {r['read']:>8.1f} {r['write']:>8.1f} {r['copy']:>8.1f}{mark}")
    for k in med:
        vals = [r[k] for r in rows]
        print(f"  {k:5}: min {min(vals):6.1f}  max {max(vals):6.1f}  spread {100 * (max(vals) - min(vals)) / med[k]:5.1f}%")
    if flagged:
        gb = len(flagged) * chunk_mib / 1024
        print(f"  {len(flagged)} chunk(s) (~{gb:.1f} GiB) below {int(slow * 100)}% of median.")
    else:
        print(f"  all chunks within {int(slow * 100)}% of median - uniform speed.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fast", "deep"])
    ap.add_argument("--no-ram", action="store_true")
    ap.add_argument("--no-vram", action="store_true")
    ap.add_argument("--chunk-mib", type=int, default=512, help="deep RAM chunk size (default 512)")
    ap.add_argument("--vram-chunk-mib", type=int, default=256, help="deep VRAM chunk size (default 256)")
    ap.add_argument("--reserve-gib", type=float, default=2.0, help="RAM to leave free in deep mode (default 2)")
    ap.add_argument("--gpu", type=int, default=None, help="OpenCL GPU index to test (default: all)")
    ap.add_argument("--vram-reserve-mib", type=int, default=0,
                    help="VRAM to leave free in deep mode (default 0 = try to fill it all)")
    a = ap.parse_args()
    t0 = time.time()
    system_info()
    gpus = [] if a.no_vram else _gpus()
    for i, d in enumerate(gpus):
        print(f"  GPU {i}: {_gpu_name(d)}  {d.global_mem_size / GiB:.1f} GiB")
    sel = range(len(gpus)) if a.gpu is None else [a.gpu]
    if a.mode == "fast":
        if not a.no_ram: ram_fast()
        for i in sel: vram_fast(i)
    else:
        if not a.no_ram: ram_deep(a.chunk_mib, a.reserve_gib)
        for i in sel: vram_deep(i, a.vram_chunk_mib, a.vram_reserve_mib)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
