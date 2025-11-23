#!/usr/bin/env python3
import argparse, re, sys
from pathlib import Path
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np

Record = namedtuple("Record", "fmt mode run_size_mb run_gen_s merge_s total_s source")

# --------- Regex patterns ----------
RE_RUN_SIZE     = re.compile(r"Run size \(MB\):\s*([0-9]+(?:\.[0-9]+)?)")
RE_OVC_ENABLED  = re.compile(r"OVC enabled:\s*(true|false)", re.IGNORECASE)
RE_R_TIME       = re.compile(r"^\s*\(R\)\s*time:\s*(\d+)\s*ms\s*$", re.MULTILINE)
RE_M_TIME       = re.compile(r"^\s*\(M\)\s*time:\s*(\d+)\s*ms\s*$", re.MULTILINE)
RE_RUNGEN_GEN   = re.compile(r"Generated\s+\d+\s+runs\s+in\s+(\d+)\s+ms")
RE_MERGE_LINE   = re.compile(r"Merge phase took\s+(\d+)\s+ms")
RE_TOTAL_S      = re.compile(r"^\s*([0-9]+\.[0-9]+)s\s*$", re.MULTILINE)

# --------- Helpers ----------
def infer_format_and_mode(path: Path, text: str):
    name = path.name.lower()
    # Format
    if "csv_" in name or "csv file:" in text:
        fmt = "CSV"
    elif "kvbin" in name or "kv binary" in text.lower() or "kv bin" in text.lower():
        fmt = "KVBin"
    else:
        parent = path.parent.name.lower()
        if "csv" in parent:
            fmt = "CSV"
        elif "kvbin" in parent:
            fmt = "KVBin"
        else:
            fmt = "UNKNOWN"

    # Mode
    m = RE_OVC_ENABLED.search(text)
    if m:
        mode = "OVC" if m.group(1).lower() == "true" else "Baseline"
    else:
        if "no_ovc" in name or "baseline" in name:
            mode = "Baseline"
        elif "ovc" in name:
            mode = "OVC"
        else:
            mode = "Baseline"
    return fmt, mode

def parse_one_log(path: Path):
    try:
        txt = path.read_text(errors="ignore")
    except Exception as e:
        print(f"warn: cannot read {path}: {e}", file=sys.stderr)
        return None

    fmt, mode = infer_format_and_mode(path, txt)
    m_size = RE_RUN_SIZE.search(txt)
    if not m_size:
        m_size = re.search(r"Run Size:\s*([0-9]+(?:\.[0-9]+)?)\s*MB", txt)
    run_size_mb = float(m_size.group(1)) if m_size else None

    m_r, m_m = None, None
    for m in RE_R_TIME.finditer(txt):
        m_r = m
    for m in RE_M_TIME.finditer(txt):
        m_m = m

    if m_r and m_m:
        run_gen_ms = int(m_r.group(1))
        merge_ms   = int(m_m.group(1))
    else:
        run_gen_ms = None
        for m in RE_RUNGEN_GEN.finditer(txt):
            run_gen_ms = int(m.group(1))
        merge_ms = sum(int(m.group(1)) for m in RE_MERGE_LINE.finditer(txt))

    total_s = None
    for m in RE_TOTAL_S.finditer(txt):
        total_s = float(m.group(1))

    if run_size_mb is None or run_gen_ms is None or merge_ms is None:
        return None

    return Record(
        fmt=fmt,
        mode=mode,
        run_size_mb=run_size_mb,
        run_gen_s=run_gen_ms / 1000.0,
        merge_s=merge_ms / 1000.0,
        total_s=total_s,
        source=str(path),
    )

def scan_root(root: Path):
    txts = list(root.rglob("*.txt"))
    if not txts:
        print(f"No .txt logs found under {root}", file=sys.stderr)
        sys.exit(1)
    recs = []
    for p in txts:
        r = parse_one_log(p)
        if r:
            recs.append(r)
    if not recs:
        print("No parseable logs matched expected patterns.", file=sys.stderr)
        sys.exit(2)
    return recs

def write_csv(records, out_dir: Path):
    out = out_dir / "parsed_times.csv"
    with out.open("w") as f:
        f.write("Format,RunSizeMB,Mode,RunGen_s,Merge_s,Total_s,Source\n")
        for r in sorted(records, key=lambda t: (t.fmt, t.run_size_mb, t.mode, t.source)):
            f.write(f"{r.fmt},{r.run_size_mb},{r.mode},{r.run_gen_s},{r.merge_s},{'' if r.total_s is None else r.total_s},{r.source}\n")
    print(f"Wrote: {out}")

# ===================================================
# Shared-axis two-row plot
# ===================================================
def plot(records, out_path: Path):
    from matplotlib.patches import Patch

    BASELINE = "#4C78A8"   # blue
    OVC      = "#F58518"   # orange
    MERGE_BG = "#E0E0E0"   # light grey

    groups = defaultdict(lambda: {"Baseline": None, "OVC": None})
    sizes_by_fmt = defaultdict(set)
    for r in records:
        groups[(r.fmt, r.run_size_mb)][r.mode] = r
        sizes_by_fmt[r.fmt].add(r.run_size_mb)

    fmts = [fmt for fmt in ("CSV", "KVBin") if fmt in sizes_by_fmt]
    if not fmts:
        print("Nothing to plot.", file=sys.stderr)
        return

    # Find global Y max for shared axis
    y_max = max((r.run_gen_s + r.merge_s) for r in records) * 1.15

    fig, axes = plt.subplots(
        nrows=len(fmts),
        ncols=1,
        figsize=(max(9, 1 + 2.2 * len(records) / len(fmts)), 6 * len(fmts)),
        sharex=False,
        sharey=True,
    )
    if len(fmts) == 1:
        axes = [axes]

    width = 0.36

    for ax, fmt in zip(axes, fmts):
        x_positions = []
        x_labels = []
        x = 0.0
        sizes = sorted(sizes_by_fmt[fmt])

        for size in sizes:
            pair = groups[(fmt, size)]
            base, ovc = pair["Baseline"], pair["OVC"]

            if base:
                ax.bar(x - width/2, base.run_gen_s, width,
                       color=BASELINE, edgecolor="black", hatch="/", linewidth=0.7)
                ax.bar(x - width/2, base.merge_s, width, bottom=base.run_gen_s,
                       color=MERGE_BG, edgecolor="black", hatch=".", linewidth=0.7)
                ax.text(x - width/2, base.run_gen_s + base.merge_s + 0.25,
                        f"{base.run_gen_s + base.merge_s:.2f}s", ha="center", va="bottom", fontsize=9)

            if ovc:
                ax.bar(x + width/2, ovc.run_gen_s, width,
                       color=OVC, edgecolor="black", hatch="/", linewidth=0.7)
                ax.bar(x + width/2, ovc.merge_s, width, bottom=ovc.run_gen_s,
                       color=MERGE_BG, edgecolor="black", hatch=".", linewidth=0.7)
                ax.text(x + width/2, ovc.run_gen_s + ovc.merge_s + 0.25,
                        f"{ovc.run_gen_s + ovc.merge_s:.2f}s", ha="center", va="bottom", fontsize=9)

            x_positions.append(x)
            x_labels.append(str(int(size)) if float(size).is_integer() else str(size))
            x += 1.0

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("Run Size (MB)", fontsize=11)
        ax.set_ylabel("Time (s)", fontsize=11)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_title(f"{fmt} Sorting Time (Baseline vs OVC)", fontsize=14, pad=10)

        if fmt == fmts[0]:
            legend = [
                Patch(facecolor=BASELINE, edgecolor="black", label="Baseline"),
                Patch(facecolor=OVC, edgecolor="black", label="OVC"),
                Patch(facecolor="white", edgecolor="black", hatch="/", label="Run Gen"),
                Patch(facecolor="white", edgecolor="black", hatch=".", label="Merge"),
            ]
            ax.legend(handles=legend, ncols=2, loc="upper right", frameon=False)

    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_path, dpi=220)
    print(f"Wrote: {out_path}")

# ===================================================
# Entry point
# ===================================================
def main():
    ap = argparse.ArgumentParser(description="Parse ES benchmark logs and plot Baseline vs OVC stacked times.")
    ap.add_argument("--root", type=Path, required=True, help="Directory containing *_no_ovc.txt / *_ovc.txt logs.")
    args = ap.parse_args()

    recs = scan_root(args.root)
    write_csv(recs, args.root)
    plot(recs, args.root / "sorting_time_stacked.png")

if __name__ == "__main__":
    main()