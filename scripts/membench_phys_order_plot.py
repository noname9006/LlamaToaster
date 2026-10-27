"""
membench_phys_order_plot.py - chart of the *_system.csv (ordinary memory, in the order the system hands it out) and the
optional *_order.csv (AWE allocation order) files written by membench_phys.py: read / write speed along the allocation
order, one panel per file. Lines are medians over the runs, the pale band is min-max across runs.
For *_order.csv files a strip below shows the physical address of every chunk's pages and chunks mostly below the
boundary (the slow single-channel zone) are shaded grey. *_system.csv has no physical addresses: a chunk is shaded
grey there when its median write speed is below --slow-write (the write speed tells the zones apart: ~26.8 in the slow
single-channel zone, ~29.8 in the dual-channel one on this machine).

Usage: python membench_phys_order_plot.py a_system.csv b_system.csv --labels "1 node" "2 nodes (SRAT)" --out compare.png
Needs: numpy, matplotlib.
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#2f78db", "#e8662b", "#d9d9d9"


def load(path):
    hdr, lines = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(":")
                hdr[k.strip()] = v.strip()
            else:
                lines.append(line)
    cols = {}
    for r in csv.DictReader(lines):
        for k, v in r.items():
            cols.setdefault(k, []).append(float(v) if v != "" else np.nan)
    cols = {k: np.array(v) for k, v in cols.items()}
    nr, nc = int(cols["run"].max()), int(cols["chunk"].max()) + 1
    out = {"hdr": hdr, "nr": nr, "nc": nc}
    for k, v in cols.items():
        out[k] = v.reshape(nr, nc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="membench-phys-order.png")
    ap.add_argument("--title", default="RAM speed from the first to the last chunk the system hands out")
    ap.add_argument("--slow-pct", type=float, default=50.0, help="AWE order csv: shade a chunk when more than this %% of its pages lie below the boundary")
    ap.add_argument("--slow-write", type=float, default=28.3, help="system csv: shade a chunk when its median write speed is below this GB/s")
    a = ap.parse_args()
    data = [load(p) for p in a.csv]
    labels = a.labels or list(a.csv)
    ratios, nrows = [], 0
    for d in data:
        d["strip"] = "phys_median_gib" in d
        ratios += [3, 1.3] if d["strip"] else [3]
        nrows += 2 if d["strip"] else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 3.9 * nrows), sharex=True,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.2})
    axes = np.atleast_1d(axes)
    ymax = max(np.nanmax(d["read_gbs"]) for d in data) * 1.12
    k = 0
    for i, d in enumerate(data):
        ax = axes[k]
        strip = axes[k + 1] if d["strip"] else None
        k += 2 if d["strip"] else 1
        x = (d["order_start_gib"][0] + d["order_end_gib"][0]) / 2
        if d["strip"]:
            boundary = float(d["hdr"].get("boundary_gib", 17))
            slow = d["pct_below_boundary"].mean(axis=0) > a.slow_pct
            note = "grey = pages mostly below %g GiB (slow zone)" % boundary
        else:
            slow = np.nanmedian(d["write_gbs"], axis=0) < a.slow_write
            note = "grey = write speed below %g GB/s (slow single-channel zone)" % a.slow_write
        for j in np.where(slow)[0]:
            for p in (ax, strip):
                if p is not None:
                    p.axvspan(d["order_start_gib"][0, j], d["order_end_gib"][0, j], color=GREY, alpha=.6, lw=0, zorder=0)
        for key, c, lab in (("read_gbs", BLUE, "Read"), ("write_gbs", ORANGE, "Write")):
            v = d[key]
            ax.fill_between(x, np.nanmin(v, axis=0), np.nanmax(v, axis=0), color=c, alpha=.18, lw=0, zorder=2)
            ax.plot(x, np.nanmedian(v, axis=0), color=c, lw=2, label=f"{lab} (median of {d['nr']} runs)", zorder=3)
        h = d["hdr"]
        vis, got = float(h["ram_visible_gib"]), float(h["obtained_gib"])
        ax.set_title(f"{labels[i]}  |  handed out {got:.2f} GiB of {vis:.2f} GiB visible "
                     f"({vis - got:.2f} GiB occupied or kept as reserve), boot: {h.get('boot_mode', '?')}",
                     fontsize=11.5, loc="left")
        ax.set_ylabel("GB/s")
        ax.set_ylim(0, ymax)
        ax.grid(color="#e3e3e3", lw=.8)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=10)
        ax.text(0.995, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#555")
        if strip is not None:
            strip.vlines(x, d["phys_p10_gib"].mean(axis=0), d["phys_p90_gib"].mean(axis=0), color="#7f8c8d", lw=1.2, zorder=2)
            strip.plot(x, d["phys_median_gib"].mean(axis=0), ".", color="#34495e", ms=3.5, zorder=3)
            strip.axhline(boundary, color="#444", ls="--", lw=1.1)
            strip.set_ylabel("phys. GiB")
            strip.grid(color="#e3e3e3", lw=.8)
            strip.set_axisbelow(True)
            strip.text(0.995, 0.06, "physical address of the chunk's pages (dot = median, bar = 10-90%)",
                       transform=strip.transAxes, fontsize=8.5, ha="right", va="bottom", color="#555")
    axes[-1].set_xlabel("Position in the order the system handed the memory out, GiB (first chunk to last)")
    axes[-1].set_xlim(0, max(float(d["order_end_gib"].max()) for d in data) * 1.005)
    fig.suptitle(a.title, fontsize=14, y=0.94 if len(data) > 1 else 1.04)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
