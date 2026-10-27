# membench_phys.py

RAM bandwidth measured along **physical addresses** and along the **order in which the system hands memory out**, with the memory that could not be tested (occupied by the OS, drivers, other processes, or kept back as reserve) reported separately. Windows only.

Related files in this folder:

- `membench_phys.py` - the benchmark (this document).
- `membench_phys_order_plot.py` - chart of the `*_system.csv` / `*_order.csv` files (see [Charting](#charting)).
- `membench.py` - the older virtual-memory benchmark (fast / deep modes, RAM and VRAM). `membench_phys.py` reuses its WMI / HWiNFO readers for the CSV header when they are available.

## What it measures

One launch runs up to three measurements, in this order:

1. **System order** (default, `--no-system-order` skips it). Ordinary memory (numpy arrays, demand-zero pages) is allocated and touched in chunks (`--order-chunk-mib`, default 256 MiB), the way any application gets memory. The chunks are then measured in the order the system handed them out, from the first to the last. This shows what a program actually sees when it asks the system for memory. It runs first, on a fresh memory state, and everything is released before the next step. Output: `<name>_system.csv`.
2. **Physical map** (always). The script reads the physical memory map Windows manages (`HKLM\HARDWARE\RESOURCEMAP\System Resources\Physical Memory`), then takes as many physical pages as Windows will give through AWE (Address Windowing Extensions: `AllocateUserPhysicalPagesNuma`, `MapUserPhysicalPages`) and learns their physical addresses from the page frame numbers (PFN x 4096). Whatever it cannot obtain is reported as **occupied**: in use by the OS, drivers or other processes, or held back by `--reserve-gib`. The obtained pages are mapped sorted by physical address, the address space is split into bins (`--bin-mib`, default 256 MiB), and read / write / copy bandwidth is measured for the pages in each bin. The measurement is repeated `--runs` times over the same pages, so the occupied map is identical between the runs of one launch. Output: `<name>.csv` (one row per run and bin, for the **whole** address space, holes and occupied bins included).
3. **AWE order** (optional, `--awe-order`). The AWE pages are mapped again in the order `AllocateUserPhysicalPagesNuma` returned them and measured per chunk, together with the physical address spread of every chunk. Output: `<name>_order.csv`. Note that this order is **not** the order ordinary memory is handed out in (measured on this setup: AWE gives a longer slow start and is more repeatable), so use the system-order test for "what does an application get".

Measurement kernels (numpy, multi-threaded): read = `sum`, write = `fill`, copy = `copyto` of one half of the slice onto the other half. Copy counts read + write bytes (2x). Each bin or chunk is measured `--passes` times and the best pass is kept; the CSV `run` column holds the repeat.

SMBIOS type 17 (DIMM slots, rated and configured MT/s, part numbers) is read through `GetSystemFirmwareTable`, which needs no driver and also works in Safe Mode. Timings and FCLK need a driver; pass them by hand with `--note`.

## Requirements

- Windows, Python 3, `pip install numpy` (`matplotlib` for the chart script).
- The account needs the **"Lock pages in memory"** right (`secpol.msc` > Local Policies > User Rights Assignment). Log off and on after granting it. The script enables the privilege itself; without the right the AWE allocation fails.
- The memory is held until the script exits.

## Running

Run it in **Safe Mode (minimal)** so that almost nothing occupies memory:

- Settings > Recovery > Advanced startup > Restart now > Troubleshoot > Advanced options > Startup Settings > Restart > press `4`. Or `msconfig` > Boot > Safe boot > Minimal, reboot (untick it afterwards).
- Open a console as the same user that has "Lock pages in memory" and run:

```
python membench_phys.py --runs 10 --note "SRAT=on DRAM Map Inversion=Auto 3733 CL16-19-19-39 FCLK1867"
```

Normal mode works too, only more memory is occupied. HWiNFO and WMI are usually not available in Safe Mode; the header then says so and the SMBIOS lines still describe the DIMMs.

Quick trial (about a minute). With `--max-gib` the pages come scattered over the whole address space, so a trial needs `--min-tested-mib 16`:

```
python membench_phys.py --max-gib 3 --runs 1 --min-tested-mib 16
```

Time: one run takes roughly 12-18 s per 40 GiB for each of the two default measurements; with the default 5 runs a launch takes a few minutes.

## Flags

All flags are optional.

| Flag | Default | Meaning |
|---|---|---|
| `--runs N` | 5 | Measurement repeats over the same pages (applies to every measurement). |
| `--passes N` | 3 | Best-of passes per bin or chunk and run. |
| `--reserve-gib X` | 0.25 | Memory left available to the system. The script stops taking memory at this level (per NUMA node for AWE); the untaken part is reported as occupied. |
| `--max-gib X` | none | Take at most X GiB (for a trial run). |
| `--threads N` | CPU count | Worker threads for the measurement (a chunk or bin uses at most 1 thread per 16 MiB). |
| `--bin-mib N` | 256 | Size of one physical-address bin of the physical map. |
| `--min-tested-mib N` | 128 | Measure a bin (or a chunk) only if at least this much of it was obtained; smaller slices fit the CPU cache and read too fast. |
| `--order-chunk-mib N` | 256 | Chunk size of the system-order and AWE-order measurements. |
| `--no-system-order` | off | Skip the system-order measurement and its `_system.csv`. |
| `--awe-order` | off | Also measure the AWE allocation order (`_order.csv`). |
| `--node-order low-first / high-first` | `low-first` | Which NUMA node AWE asks for pages first. `low-first` starts at node 0 (where the CPUs are), like an ordinary process. |
| `--boundary-gib X` | 17 | Physical slow/fast boundary (GiB) used for the `pct_below_boundary` column of `_order.csv`. It is a property of the machine (here: end of the single-channel region); the benchmark does not detect it. |
| `--note "text"` | empty | Free text for the CSV header, e.g. timings, FCLK, BIOS options. |
| `--out path.csv` | `membench_phys_<YYYYMMDD_HHMMSS>.csv` next to the script | Path of the main CSV. `_system.csv` and `_order.csv` are written next to it with the same base name. |
| `-h`, `--help` | | Show the help. |

## Output files

Every CSV starts with `#` header lines: `created`, `boot_mode` (normal / safe / safe_network), `script`, `ram_visible_gib`, `available_before_gib`, `obtained_gib`, `occupied_gib`, `reserve_gib`, `bin_mib`, `runs`, `passes`, `min_tested_mib`, `numa_nodes`, `ranges_gib` (the physical RAM ranges), `smbios_dimmN`, `cpu`, WMI / HWiNFO readings when available, and `note`. `_system.csv` and `_order.csv` additionally carry `view` and `order_chunk_mib` (and the order csv `node_order`, `boundary_gib`).

**`<name>.csv`** - physical map, one row per (run, bin):

| Column | Meaning |
|---|---|
| `run`, `bin` | Repeat number and bin index. |
| `start_gib`, `end_gib` | Physical address range of the bin. |
| `range_mib` | RAM present in the bin. |
| `tested_mib` | Pages obtained and measured. |
| `occupied_mib` | `range_mib - tested_mib`: RAM that could not be obtained (OS, drivers, processes, reserve). |
| `hole_mib` | Bin size minus `range_mib`: address space that is not RAM (e.g. the MMIO hole below 4 GiB). |
| `tested_pct` | `tested_mib / range_mib`. |
| `node` | NUMA node of the allocation call that returned most of the bin's pages (-1 if none). It is only a preferred node, not necessarily the real node of the pages; do not use it to infer the SRAT node layout. |
| `threads` | Threads used. |
| `read_gbs`, `write_gbs`, `copy_gbs` | Bandwidth in GB/s; empty when the bin has less than `--min-tested-mib` tested. |

**`<name>_system.csv`** - ordinary memory in the order the system handed it out, one row per (run, chunk): `run, chunk, order_start_gib, order_end_gib, threads, read_gbs, write_gbs, copy_gbs`. There are no physical addresses here (they cannot be read without a driver or administrator-only undocumented calls); the zone of a chunk is recognised by its speed, e.g. on a 1 + 2 DIMM layout the write speed separates the single-channel zone from the dual-channel one.

**`<name>_order.csv`** (only with `--awe-order`) - one row per (run, chunk): `run, chunk, order_start_gib, order_end_gib, tested_mib, node, phys_min_gib, phys_p10_gib, phys_median_gib, phys_p90_gib, phys_max_gib, pct_below_boundary, threads, read_gbs, write_gbs, copy_gbs`.

## Charting

`membench_phys_order_plot.py` draws read and write along the allocation order (line = median over the runs, pale band = min-max across runs), one panel per file. For `_system.csv` a chunk is shaded grey when its median write speed is below `--slow-write`; for `_order.csv` the shading follows `pct_below_boundary` and a strip under the panel shows the physical address of every chunk's pages.

```
python membench_phys_order_plot.py a_system.csv b_system.csv --labels "A" "B" --out compare.png
```

| Flag | Default | Meaning |
|---|---|---|
| `csv ...` | required | One or more `_system.csv` / `_order.csv` files; one panel each. |
| `--labels L1 L2 ...` | file names | Panel titles. |
| `--out file.png` | `membench-phys-order.png` | Output image. |
| `--title "text"` | built-in title | Figure title. |
| `--slow-write X` | 28.3 | `_system.csv`: shade a chunk when its median write speed (GB/s) is below X. |
| `--slow-pct X` | 50 | `_order.csv`: shade a chunk when more than X % of its pages lie below the boundary. |

The chart of the physical map itself (speed per physical-address bin, with occupied memory and holes) is built from the main `<name>.csv`; it is not part of this script.

## Caveats

- AWE pages and ordinary pages are drawn from the same pool through different code paths, so the AWE order differs from the order an application gets. Use `_system.csv` for the latter.
- `occupied_gib` includes the part kept back by `--reserve-gib` (and, with several NUMA nodes, allocation granularity), not only memory in use by the system; compare `ram_visible_gib - available_before_gib` for the memory really in use before the run.
- The slow/fast boundary and the speeds of the zones depend on the machine (DIMM population, BIOS options, FCLK); the numbers quoted in `membench-results.md` are for the machine this was developed on.
