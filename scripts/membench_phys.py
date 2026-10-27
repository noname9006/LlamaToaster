"""
membench_phys.py - RAM bandwidth along PHYSICAL addresses, with the memory that could not be tested marked.

How it works
  1. Reads the physical memory map Windows manages (HKLM\\HARDWARE\\RESOURCEMAP\\System Resources\\Physical Memory),
     i.e. every address range that is real RAM, holes included.
  2. Takes as many physical pages as Windows will give (AWE: AllocateUserPhysicalPagesNuma) and learns their real
     physical addresses from the page frame numbers. Whatever is NOT obtained is "occupied": in use by the OS, drivers
     or other processes, or kept back by --reserve-gib. That part cannot be tested.
  3. Maps the obtained pages sorted by physical address, splits the address space into bins (--bin-mib) and measures
     read / write / copy bandwidth of the pages that lie in each bin. --runs repeats the measurement without
     reallocating (the occupied map stays the same between runs of one launch).
  4. Writes a csv: '#' lines on top carry the system info, then one row per (run, bin) for the WHOLE address space
     (also bins that are only holes or only occupied memory), so a chart can show tested, occupied and holes separately.
  5. BEFORE all of that (fresh memory state, as an application would see it) it runs the "system order" test: ordinary
     memory (numpy arrays) is allocated and touched in --order-chunk-mib chunks, the way any program gets memory, and the
     chunks are measured in the order the system handed them out, from the first to the last. Written to
     <name>_system.csv (one row per run and chunk) so the speed can be charted from the start to the end of what a
     program gets. Skip it with --no-system-order. This is the same method as membench.py deep.
  6. Optional (--awe-order): maps the AWE pages again in the order AllocateUserPhysicalPagesNuma returned them and
     measures per chunk, written to <name>_order.csv together with the physical address spread of each chunk. That order
     is NOT the order ordinary memory is handed out in (measured: AWE gives a longer slow start and is more repeatable).

Run it in Safe Mode (minimal) so that almost nothing occupies memory:
  - Settings > Recovery > Advanced startup > Restart now > Troubleshoot > Advanced options > Startup Settings >
    Restart > press 4 (Safe Mode).  Or: msconfig > Boot > Safe boot > Minimal, reboot (untick it afterwards!).
  - Open a console as the same user that has "Lock pages in memory", run:  python membench_phys.py
  - HWiNFO and WMI are usually not available in Safe Mode; DIMM slots and MT/s are still read from SMBIOS (no driver needed).
    FCLK and timings need a driver, so pass them by hand:  --note "CL16-19-19-39 1T tRC87 tRFC653 FCLK1867"
Normal mode works too, only more memory is occupied.
Note: with --max-gib the pages come scattered over the whole address space, so a trial needs --min-tested-mib 16:
  python membench_phys.py --max-gib 3 --runs 1 --min-tested-mib 16

Needs: pip install numpy. The account needs the "Lock pages in memory" right (secpol.msc > Local Policies >
User Rights Assignment); log off and on after granting it. The memory is held until the script exits.

csv columns: run, bin, start_gib, end_gib, range_mib (RAM present in the bin), tested_mib (pages obtained),
occupied_mib (range_mib - tested_mib), hole_mib (bin size - range_mib: address space that is not RAM),
tested_pct, node (NUMA node most of the tested pages came from, -1 if none), threads,
read_gbs, write_gbs, copy_gbs (empty when the bin has less than --min-tested-mib tested). Copy counts read + write bytes.

system csv columns: run, chunk, order_start_gib, order_end_gib (position in the order the system handed the chunks out),
threads, read_gbs, write_gbs, copy_gbs.

order csv (--awe-order) columns: run, chunk, order_start_gib, order_end_gib (position in the allocation order), tested_mib, node (the
node argument of the call that returned most of the chunk's pages: only a preference, not the real node of the pages),
phys_min_gib, phys_p10_gib, phys_median_gib, phys_p90_gib, phys_max_gib (physical addresses of the chunk's pages),
pct_below_boundary (share of the pages below --boundary-gib), threads, read_gbs, write_gbs, copy_gbs.
"""
import argparse
import ctypes
import datetime
import os
import statistics
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

import numpy as np

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
adv = ctypes.WinDLL("advapi32", use_last_error=True)
u32 = ctypes.WinDLL("user32", use_last_error=True)

MEM_RESERVE = 0x2000
MEM_PHYSICAL = 0x400000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
PAGE = 4096
WORDS_PER_PAGE = PAGE // 8
GiB = 2 ** 30
MiB = 2 ** 20

k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.AllocateUserPhysicalPagesNuma.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.c_void_p, wintypes.DWORD]
k32.AllocateUserPhysicalPagesNuma.restype = wintypes.BOOL
k32.FreeUserPhysicalPages.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
k32.FreeUserPhysicalPages.restype = wintypes.BOOL
k32.MapUserPhysicalPages.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
k32.MapUserPhysicalPages.restype = wintypes.BOOL
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
k32.VirtualFree.restype = wintypes.BOOL
k32.GetNumaAvailableMemoryNodeEx.argtypes = [ctypes.c_ushort, ctypes.POINTER(ctypes.c_ulonglong)]
k32.GetNumaHighestNodeNumber.argtypes = [ctypes.POINTER(ctypes.c_ulong)]
u32.GetSystemMetrics.argtypes = [ctypes.c_int]
adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
adv.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p]
adv.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.c_void_p,
                                      wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]


class TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def enable_lock_privilege():
    tok = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x28, ctypes.byref(tok)):
        return False
    luid = LUID()
    if not adv.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
        return False
    tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(LUID_AND_ATTRIBUTES(luid, 2)))
    ctypes.set_last_error(0)
    adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None)
    return ctypes.get_last_error() == 0


def mem_status():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys, m.ullAvailPhys


def node_avail(node):
    b = ctypes.c_ulonglong(0)
    if not k32.GetNumaAvailableMemoryNodeEx(node, ctypes.byref(b)):
        return 0
    return b.value


def physical_ranges():
    """[(start, length)] in bytes of every RAM range Windows manages, from the registry resource map."""
    import winreg
    k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\RESOURCEMAP\System Resources\Physical Memory")
    v, _ = winreg.QueryValueEx(k, ".Translated")
    off = 0
    count, = struct.unpack_from("<I", v, off)
    off += 4
    out = []
    for _ in range(count):
        off += 8                                  # InterfaceType, BusNumber
        _ver, _rev, n = struct.unpack_from("<HHI", v, off)
        off += 8
        for _ in range(n):
            typ, _share, flags = struct.unpack_from("<BBH", v, off)
            start, length = struct.unpack_from("<QI", v, off + 4)
            off += 20
            if typ not in (3, 7):                 # CmResourceTypeMemory / MemoryLarge
                continue
            length <<= 8 if flags & 0x200 else 16 if flags & 0x400 else 32 if flags & 0x800 else 0
            out.append((start, length))
    return sorted(out)


def alloc_node(node, max_pages, reserve_bytes, log_every=4 * GiB, chunk=65536):
    """Take pages preferring `node` until its available memory (or the global one) is down to the reserve."""
    proc = k32.GetCurrentProcess()
    parts, got, err, last_log = [], 0, None, 0
    while got < max_pages:
        n = min(chunk, max_pages - got)
        if node_avail(node) < reserve_bytes + n * PAGE or mem_status()[1] < reserve_bytes + n * PAGE:
            err = "reserve reached"
            break
        arr = np.empty(n, dtype=np.uintp)
        cnt = ctypes.c_size_t(n)
        if not k32.AllocateUserPhysicalPagesNuma(proc, ctypes.byref(cnt), arr.ctypes.data, node):
            err = f"Windows error {ctypes.get_last_error()}"
            break
        if cnt.value == 0:
            err = "no pages returned"
            break
        parts.append(arr[:cnt.value].copy())
        got += cnt.value
        if got * PAGE - last_log >= log_every:
            last_log = got * PAGE
            print(f"    node {node}: {got * PAGE / GiB:.1f} GiB so far", flush=True)
    pfns = np.concatenate(parts) if parts else np.empty(0, dtype=np.uintp)
    return pfns, parts, err


def map_pages(pfns, chunk=262144):
    """Map the pages into one contiguous view in the given order (sorted by address, or in allocation order)."""
    total = len(pfns)
    base = k32.VirtualAlloc(None, total * PAGE, MEM_RESERVE | MEM_PHYSICAL, PAGE_READWRITE)
    if not base:
        sys.exit(f"VirtualAlloc(MEM_PHYSICAL) failed, error {ctypes.get_last_error()}")
    for off in range(0, total, chunk):
        sub = np.ascontiguousarray(pfns[off:off + chunk])
        if not k32.MapUserPhysicalPages(base + off * PAGE, len(sub), sub.ctypes.data):
            sys.exit(f"MapUserPhysicalPages failed at page {off}, error {ctypes.get_last_error()}")
    buf = (ctypes.c_uint64 * (total * WORDS_PER_PAGE)).from_address(base)
    return np.frombuffer(buf, dtype=np.uint64)


def _timed(pool, fns):
    t0 = time.perf_counter()
    list(pool.map(lambda f: f(), fns))
    return time.perf_counter() - t0


def measure(pool, view, threads, passes):
    """(read, write, copy) GB/s of one slice of the view, best of `passes`."""
    n = view.size
    step = n // threads
    slices = [view[i * step:(i + 1) * step] for i in range(threads)]
    half = step // 2
    read_b, write_b, copy_b = step * threads * 8, step * threads * 8, half * threads * 8 * 2
    best = [None, None, None]
    for _ in range(passes):
        times = (
            _timed(pool, [lambda s=s: s.sum() for s in slices]),
            _timed(pool, [lambda s=s: s.fill(1) for s in slices]),
            _timed(pool, [lambda s=s: np.copyto(s[half:2 * half], s[:half]) for s in slices]),
        )
        for j, t in enumerate(times):
            if best[j] is None or t < best[j]:
                best[j] = t
    return read_b / best[0] / 1e9, write_b / best[1] / 1e9, copy_b / best[2] / 1e9


def system_order(chunk_bytes, reserve_bytes, cap_bytes, threads, passes, runs):
    """Ordinary memory handed out by the system in allocation order: [(run, chunk, (threads, read, write, copy))], GiB got."""
    chunks = []
    try:
        while mem_status()[1] > reserve_bytes + chunk_bytes and len(chunks) * chunk_bytes < cap_bytes:
            c = np.empty(chunk_bytes // 8, dtype=np.uint64)
            c.fill(1)                                   # commit the pages
            chunks.append(c)
    except MemoryError:
        print(f"  MemoryError after {len(chunks)} chunks, testing what we got")
    got = len(chunks) * chunk_bytes
    print(f"system order: {len(chunks)} chunks of {chunk_bytes // MiB} MiB = {got / GiB:.2f} GiB handed out; {runs} runs ...", flush=True)
    th = max(1, min(threads, (chunk_bytes // MiB) // 16))
    rows = []
    with ThreadPoolExecutor(threads) as pool:
        for run in range(1, runs + 1):
            t1 = time.perf_counter()
            sp = []
            for i, c in enumerate(chunks):
                r = (th,) + measure(pool, c, th, passes)
                sp.append(r)
                rows.append((run, i, r))
            med = [statistics.median(r[j] for r in sp) for j in (1, 2, 3)] if sp else [0, 0, 0]
            print(f"  system run {run}/{runs}: {time.perf_counter() - t1:.0f} s, median read {med[0]:.1f} "
                  f"write {med[1]:.1f} copy {med[2]:.1f} GB/s", flush=True)
    chunks.clear()                                      # give everything back before the AWE test takes it
    return rows, got


def smbios_dimms():
    """DIMM slots from the SMBIOS table (type 17): works without WMI, HWiNFO or any driver, also in Safe Mode."""
    k32.GetSystemFirmwareTable.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    k32.GetSystemFirmwareTable.restype = ctypes.c_uint32
    sig = struct.unpack(">I", b"RSMB")[0]
    n = k32.GetSystemFirmwareTable(sig, 0, None, 0)
    if not n:
        return []
    buf = ctypes.create_string_buffer(n)
    k32.GetSystemFirmwareTable(sig, 0, buf, n)
    tlen = struct.unpack_from("<I", buf.raw, 4)[0]
    t = buf.raw[8:8 + tlen]
    out, off = [], 0
    while off + 4 <= len(t):
        typ, length, _h = struct.unpack_from("<BBH", t, off)
        if typ == 127:
            break
        tail = t[off + length:].split(bytes(2))[0]
        strs = tail.split(bytes(1))
        if typ == 17 and length >= 0x22:
            size, = struct.unpack_from("<H", t, off + 12)
            speed, = struct.unpack_from("<H", t, off + 0x15)
            conf, = struct.unpack_from("<H", t, off + 0x20)

            def g(i):
                return strs[i - 1].decode(errors="ignore").strip() if 0 < i <= len(strs) else ""
            if size:
                out.append((g(t[off + 0x11]) or g(t[off + 0x10]), g(t[off + 0x10]), size, speed, conf, g(t[off + 0x1A])))
        off += length + len(tail) + 2
    return out


def system_info():
    info = {}
    try:
        for i, (bank, loc, mib, speed, conf, part) in enumerate(smbios_dimms()):
            info[f"smbios_dimm{i}"] = f"{bank} {loc} | {mib / 1024:g} GiB | rated {speed} MT/s | configured {conf} MT/s | {part}"
    except Exception as e:
        info["smbios"] = f"not available ({e.__class__.__name__})"
    try:
        import platform
        info["cpu"] = platform.processor()
    except Exception:
        pass
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import membench as mb          # reuses the WMI / HWiNFO readers of membench.py
        try:
            info["cpu"] = mb._cpu_name() or info.get("cpu", "")
        except Exception:
            pass
        try:
            for i, (slot, gib, speed, part) in enumerate(mb._wmi_dimms()):
                info[f"dimm{i}"] = f"{slot} | {gib:.0f} GiB | {speed} MT/s | {part}"
        except Exception as e:
            info["dimms"] = f"not available ({e.__class__.__name__})"
        try:
            hw = mb._hwinfo_readings()
            if hw is None:
                info["hwinfo"] = "not available"
            else:
                for k, (v, u) in hw.items():
                    info["hwinfo " + k] = f"{v:g} {u}".strip()
        except Exception as e:
            info["hwinfo"] = f"not available ({e.__class__.__name__})"
    except Exception as e:
        info["membench_helpers"] = f"not available ({e.__class__.__name__})"
    return info


def main():
    if sys.platform != "win32":
        sys.exit("Windows only")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin-mib", type=int, default=256, help="size of one physical-address bin (default 256)")
    ap.add_argument("--runs", type=int, default=5, help="measurement repeats over the same pages (default 5)")
    ap.add_argument("--passes", type=int, default=3, help="best-of passes per bin and run (default 3)")
    ap.add_argument("--reserve-gib", type=float, default=0.25, help="memory left available to the system (default 0.25)")
    ap.add_argument("--max-gib", type=float, default=None, help="take at most this much (for a trial run)")
    ap.add_argument("--min-tested-mib", type=int, default=128,
                    help="measure a bin only if this much of it was obtained (default 128; smaller slices fit the CPU cache and read too fast)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--node-order", choices=("low-first", "high-first"), default="low-first",
                    help="which NUMA node is asked for pages first (default low-first: node 0, where the CPUs are, like an ordinary process)")
    ap.add_argument("--order-chunk-mib", type=int, default=256, help="chunk size of the allocation-order tests (default 256)")
    ap.add_argument("--boundary-gib", type=float, default=17.0, help="physical slow/fast boundary for the AWE order csv (default 17)")
    ap.add_argument("--no-system-order", action="store_true", help="skip the ordinary-memory (system order) test and its csv")
    ap.add_argument("--awe-order", action="store_true", help="also measure the AWE allocation order (second csv, _order.csv)")
    ap.add_argument("--note", default="", help='free text for the csv header, e.g. "CL16-19-19-39 1T tRC87 tRFC653 FCLK1867 DRAM Map Inversion=Auto SRAT=...": '
                    "timings and FCLK cannot be read without a driver, so write them here")
    ap.add_argument("--out", default=None, help="csv path (default membench_phys_<time>.csv next to the script)")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"membench_phys_{stamp}.csv")
    safe = u32.GetSystemMetrics(67)             # SM_CLEANBOOT: 0 normal, 1 safe, 2 safe with network
    print(f"boot mode: {['normal', 'SAFE MODE', 'SAFE MODE with network'][safe] if safe in (0, 1, 2) else safe}")
    if not enable_lock_privilege():
        print("WARNING: 'Lock pages in memory' is not enabled for this account; allocation will likely fail.")

    ranges = physical_ranges()
    top = max(s + l for s, l in ranges)
    total, avail0 = mem_status()
    print(f"RAM visible {total / GiB:.2f} GiB, available {avail0 / GiB:.2f} GiB, address space up to {top / GiB:.2f} GiB")
    print("ranges:", ", ".join(f"{s / GiB:.3f}-{(s + l) / GiB:.3f}" for s, l in ranges))

    hi = ctypes.c_ulong(0)
    k32.GetNumaHighestNodeNumber(ctypes.byref(hi))
    nodes = list(range(hi.value + 1)) if a.node_order == "low-first" else list(range(hi.value, -1, -1))
    reserve = int(a.reserve_gib * GiB)
    cap = int(a.max_gib * GiB) // PAGE if a.max_gib else 1 << 60
    print(f"NUMA nodes: {len(nodes)}; reserve {a.reserve_gib} GiB per node; taking pages ...", flush=True)

    system_rows, system_got = [], 0
    if not a.no_system_order:
        system_rows, system_got = system_order(a.order_chunk_mib * MiB, reserve, int(a.max_gib * GiB) if a.max_gib else 1 << 62,
                                                a.threads, a.passes, a.runs)
        time.sleep(2)                                   # let the freed memory settle before the AWE allocation
        total, avail0 = mem_status()

    all_parts, pfn_list, node_list = [], [], []
    t0 = time.perf_counter()
    for node in nodes:
        want = cap - sum(len(p) for p in pfn_list)
        if want <= 0:
            break
        pfns, parts, err = alloc_node(node, want, reserve)
        all_parts += parts
        pfn_list.append(pfns)
        node_list.append(np.full(len(pfns), node, dtype=np.int16))
        print(f"  node {node}: got {len(pfns) * PAGE / GiB:.2f} GiB" + (f" (stopped: {err})" if err else ""), flush=True)
    pfns = np.concatenate(pfn_list) if pfn_list else np.empty(0, dtype=np.uintp)
    nodes_arr = np.concatenate(node_list) if node_list else np.empty(0, dtype=np.int16)
    if len(pfns) == 0:
        sys.exit("No pages obtained. Is 'Lock pages in memory' granted (and did you log off/on afterwards)?")
    alloc_pfns, alloc_nodes = pfns, nodes_arr     # in the order Windows handed the pages out
    order = np.argsort(pfns, kind="stable")
    pfns, nodes_arr = pfns[order], nodes_arr[order]
    print(f"obtained {len(pfns) * PAGE / GiB:.2f} GiB in {time.perf_counter() - t0:.0f} s; "
          f"available now {mem_status()[1] / GiB:.2f} GiB; mapping ...", flush=True)

    try:
        view = map_pages(pfns)
        bin_bytes = a.bin_mib * MiB
        ppb = bin_bytes // PAGE
        nbins = -(-top // bin_bytes)
        bins = []
        for b in range(nbins):
            lo_b, hi_b = b * bin_bytes, (b + 1) * bin_bytes
            in_range = sum(max(0, min(hi_b, s + l) - max(lo_b, s)) for s, l in ranges)
            lo = int(np.searchsorted(pfns, b * ppb, "left"))
            hi = int(np.searchsorted(pfns, (b + 1) * ppb, "left"))
            tested = (hi - lo) * PAGE
            node = -1
            if hi > lo:
                vals, cnts = np.unique(nodes_arr[lo:hi], return_counts=True)
                node = int(vals[np.argmax(cnts)])
            bins.append(dict(b=b, lo=lo, hi=hi, range=in_range, tested=tested, node=node))

        rows = []
        measurable = [x for x in bins if x["tested"] >= a.min_tested_mib * MiB]
        print(f"{len(measurable)} of {nbins} bins are measurable; {a.runs} runs ...", flush=True)
        with ThreadPoolExecutor(a.threads) as pool:
            for run in range(1, a.runs + 1):
                t1 = time.perf_counter()
                speeds = {}
                for x in measurable:
                    th = max(1, min(a.threads, (x["tested"] // MiB) // 16))
                    seg = view[x["lo"] * WORDS_PER_PAGE:x["hi"] * WORDS_PER_PAGE]
                    speeds[x["b"]] = (th,) + measure(pool, seg, th, a.passes)
                for x in bins:
                    s = speeds.get(x["b"])
                    rows.append((run, x, s))
                med = [statistics.median(v[j] for v in speeds.values()) for j in (1, 2, 3)] if speeds else [0, 0, 0]
                print(f"  run {run}/{a.runs}: {time.perf_counter() - t1:.0f} s, median read {med[0]:.1f} "
                      f"write {med[1]:.1f} copy {med[2]:.1f} GB/s", flush=True)

        order_rows = []
        if a.awe_order:
            k32.VirtualFree(view.ctypes.data, 0, MEM_RELEASE)     # unmap; the pages stay ours until they are freed below
            del view
            oview = map_pages(alloc_pfns)
            cpages = a.order_chunk_mib * MiB // PAGE
            chunks = []
            for lo in range(0, len(alloc_pfns), cpages):
                hi_c = min(lo + cpages, len(alloc_pfns))
                if (hi_c - lo) * PAGE < a.min_tested_mib * MiB:
                    continue                                       # a short tail would fit the CPU cache
                phys = alloc_pfns[lo:hi_c].astype(np.float64) * PAGE / GiB
                vals, cnts = np.unique(alloc_nodes[lo:hi_c], return_counts=True)
                chunks.append(dict(lo=lo, hi=hi_c, node=int(vals[np.argmax(cnts)]),
                                   q=np.percentile(phys, [0, 10, 50, 90, 100]),
                                   below=100.0 * float(np.mean(phys < a.boundary_gib))))
            print(f"allocation-order view: {len(chunks)} chunks of {a.order_chunk_mib} MiB; {a.runs} runs ...", flush=True)
            with ThreadPoolExecutor(a.threads) as pool:
                for run in range(1, a.runs + 1):
                    t1 = time.perf_counter()
                    got = []
                    for i, c in enumerate(chunks):
                        th = max(1, min(a.threads, ((c["hi"] - c["lo"]) * PAGE // MiB) // 16))
                        seg = oview[c["lo"] * WORDS_PER_PAGE:c["hi"] * WORDS_PER_PAGE]
                        sp = (th,) + measure(pool, seg, th, a.passes)
                        got.append(sp)
                        order_rows.append((run, i, c, sp))
                    med = [statistics.median(g[j] for g in got) for j in (1, 2, 3)] if got else [0, 0, 0]
                    print(f"  order run {run}/{a.runs}: {time.perf_counter() - t1:.0f} s, median read {med[0]:.1f} "
                          f"write {med[1]:.1f} copy {med[2]:.1f} GB/s", flush=True)
    finally:
        proc = k32.GetCurrentProcess()
        for p in all_parts:
            cnt = ctypes.c_size_t(len(p))
            k32.FreeUserPhysicalPages(proc, ctypes.byref(cnt), p.ctypes.data)

    import threading
    box = {}
    th = threading.Thread(target=lambda: box.update(system_info()), daemon=True)
    th.start()
    th.join(30)                                   # WMI / HWiNFO can hang in Safe Mode; never lose the measurement for it
    info = dict(box) if not th.is_alive() else {**box, "system_info": "timed out"}
    tested_total = sum(x["tested"] for x in bins)
    range_total = sum(x["range"] for x in bins)
    with open(out, "w", encoding="utf8", newline="\n") as f:
        meta = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "boot_mode": ["normal", "safe", "safe_network"][safe] if safe in (0, 1, 2) else str(safe),
            "script": "membench_phys.py",
            "ram_visible_gib": f"{total / GiB:.3f}",
            "available_before_gib": f"{avail0 / GiB:.3f}",
            "obtained_gib": f"{tested_total / GiB:.3f}",
            "occupied_gib": f"{(range_total - tested_total) / GiB:.3f}",
            "reserve_gib": a.reserve_gib,
            "bin_mib": a.bin_mib,
            "runs": a.runs,
            "passes": a.passes,
            "min_tested_mib": a.min_tested_mib,
            "numa_nodes": len(nodes),
            "ranges_gib": ";".join(f"{s / GiB:.4f}-{(s + l) / GiB:.4f}" for s, l in ranges),
        }
        meta.update(info)
        if a.note:
            meta["note"] = " ".join(a.note.splitlines())
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("run,bin,start_gib,end_gib,range_mib,tested_mib,occupied_mib,hole_mib,tested_pct,node,threads,"
                "read_gbs,write_gbs,copy_gbs\n")
        for run, x, s in rows:
            start = x["b"] * a.bin_mib / 1024
            rng, tst = x["range"] / MiB, x["tested"] / MiB
            occ = max(0.0, rng - tst)
            hole = a.bin_mib - rng
            pct = 100.0 * tst / rng if rng else 0.0
            sp = ("", "", "", "") if s is None else (s[0], f"{s[1]:.2f}", f"{s[2]:.2f}", f"{s[3]:.2f}")
            f.write(f"{run},{x['b']},{start:.4f},{start + a.bin_mib / 1024:.4f},{rng:.1f},{tst:.1f},{occ:.1f},"
                    f"{hole:.1f},{pct:.1f},{x['node']},{sp[0]},{sp[1]},{sp[2]},{sp[3]}\n")
    if system_rows:
        out_system = os.path.splitext(out)[0] + "_system.csv"
        with open(out_system, "w", encoding="utf8", newline="\n") as f:
            sysmeta = dict(meta)
            sysmeta.update({"view": "system order (ordinary memory)", "obtained_gib": f"{system_got / GiB:.3f}",
                            "occupied_gib": f"{total / GiB - system_got / GiB:.3f}", "order_chunk_mib": a.order_chunk_mib})
            sysmeta.pop("bin_mib", None)
            for k, v in sysmeta.items():
                f.write(f"# {k}: {v}\n")
            f.write("run,chunk,order_start_gib,order_end_gib,threads,read_gbs,write_gbs,copy_gbs\n")
            for run, i, sp in system_rows:
                s0 = i * a.order_chunk_mib / 1024
                f.write(f"{run},{i},{s0:.4f},{s0 + a.order_chunk_mib / 1024:.4f},{sp[0]},{sp[1]:.2f},{sp[2]:.2f},{sp[3]:.2f}\n")
        print(f"system-order csv written: {out_system}")
    if order_rows:
        out_order = os.path.splitext(out)[0] + "_order.csv"
        with open(out_order, "w", encoding="utf8", newline="\n") as f:
            meta.update({"view": "allocation order", "node_order": a.node_order,
                         "order_chunk_mib": a.order_chunk_mib, "boundary_gib": a.boundary_gib})
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("run,chunk,order_start_gib,order_end_gib,tested_mib,node,phys_min_gib,phys_p10_gib,phys_median_gib,"
                    "phys_p90_gib,phys_max_gib,pct_below_boundary,threads,read_gbs,write_gbs,copy_gbs\n")
            for run, i, c, sp in order_rows:
                start = c["lo"] * PAGE / GiB
                q = c["q"]
                f.write(f"{run},{i},{start:.4f},{c['hi'] * PAGE / GiB:.4f},{(c['hi'] - c['lo']) * PAGE / MiB:.1f},{c['node']},"
                        f"{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f},{q[4]:.4f},{c['below']:.1f},"
                        f"{sp[0]},{sp[1]:.2f},{sp[2]:.2f},{sp[3]:.2f}\n")
        print(f"order csv written: {out_order}")
    print(f"\nobtained {tested_total / GiB:.2f} GiB of {range_total / GiB:.2f} GiB of RAM; "
          f"{(range_total - tested_total) / GiB:.2f} GiB occupied (not testable)")
    print(f"csv written: {out}")


if __name__ == "__main__":
    main()
