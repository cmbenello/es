#!/usr/bin/env python3
import os
import glob
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Your 4 experiments mapped to device + OVC
# -----------------------------------------------------------------------------
TARGET_DIR_BASENAMES = [
    "kvbin_thread_sweep_yes_20251112_224857",  # NVMe OVC
    "kvbin_thread_sweep_no_20251114_140717",   # NVMe Baseline
    "kvbin_thread_sweep_no_20251115_174603",   # HDD Baseline
    "kvbin_thread_sweep_yes_20251116_223628",  # HDD OVC
]

DIR_DEVICE_MAP = {
    "kvbin_thread_sweep_yes_20251112_224857": "nvme",
    "kvbin_thread_sweep_no_20251114_140717": "nvme",
    "kvbin_thread_sweep_no_20251115_174603": "hdd",
    "kvbin_thread_sweep_yes_20251116_223628": "hdd",
}

DIR_OVC_MAP = {
    "kvbin_thread_sweep_yes_20251112_224857": True,
    "kvbin_thread_sweep_no_20251114_140717": False,
    "kvbin_thread_sweep_no_20251115_174603": False,
    "kvbin_thread_sweep_yes_20251116_223628": True,
}

# PROFESSIONAL PAPER COLORS (same as your plots)
BASELINE_COLOR = "#4C72B0"   # blue
OVC_COLOR      = "#DD8452"   # orange


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
def parse_benchmark_summary(lines):
    for i, line in enumerate(lines):
        if "Benchmark Results Summary" in line:
            start_idx = i
            break
    else:
        raise RuntimeError("No summary section")

    for i in range(start_idx, len(lines)):
        if lines[i].strip().startswith("Config") and "Total (s)" in lines[i]:
            header_idx = i
            break
    else:
        raise RuntimeError("No header")

    for i in range(header_idx + 1, len(lines)):
        if lines[i].strip().startswith("no_name[avg]"):
            avg_idx = i
            break
    else:
        raise RuntimeError("Avg row not found")

    row = lines[avg_idx].split()
    total_s = float(row[5])
    run_gen_s = float(row[6])
    merge_s = float(row[7])
    return total_s, run_gen_s, merge_s


def parse_dir(dir_path):
    dir_base = os.path.basename(dir_path)
    device = DIR_DEVICE_MAP[dir_base]
    ovc = DIR_OVC_MAP[dir_base]

    rows = []
    for path in glob.glob(os.path.join(dir_path, "KVBin_*_thr*_rs*.txt")):
        fname = os.path.basename(path)
        m = re.search(r"_thr(\d+)_", fname)
        if not m:
            continue
        threads = int(m.group(1))

        with open(path, "r") as f:
            lines = f.readlines()

        total_s, run_gen_s, merge_s = parse_benchmark_summary(lines)

        rows.append({
            "Threads": threads,
            "Device": device,
            "OVC": ovc,
            "RunGen_s": run_gen_s,
            "Merge_s": merge_s,
            "Total_s": total_s,
        })

    return pd.DataFrame(rows)


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
    dfs = [parse_dir(os.path.join(base_dir, bn)) for bn in TARGET_DIR_BASENAMES]
    full = pd.concat(dfs, ignore_index=True)

    devices = ["nvme", "hdd"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, dev in zip(axes, devices):
        df = full[full["Device"] == dev]
        threads_sorted = sorted(df["Threads"].unique())
        x = np.arange(len(threads_sorted))
        width = 0.33

        # Baseline
        base = df[df["OVC"] == False].set_index("Threads")
        # OVC
        ovc = df[df["OVC"] == True].set_index("Threads")

        rg_base = [base.loc[t]["RunGen_s"] if t in base.index else 0 for t in threads_sorted]
        m_base  = [base.loc[t]["Merge_s"]  if t in base.index else 0 for t in threads_sorted]

        rg_ovc  = [ovc.loc[t]["RunGen_s"] if t in ovc.index else 0 for t in threads_sorted]
        m_ovc   = [ovc.loc[t]["Merge_s"]  if t in ovc.index else 0 for t in threads_sorted]

        # Baseline bar — blue
        ax.bar(x - width/2, rg_base, width,
               color=BASELINE_COLOR, alpha=0.9,
               label="Baseline RunGen")
        ax.bar(x - width/2, m_base, width,
               bottom=rg_base,
               color=BASELINE_COLOR, alpha=0.9,
               hatch="///", label="Baseline Merge")

        # OVC bar — orange
        ax.bar(x + width/2, rg_ovc, width,
               color=OVC_COLOR, alpha=0.9,
               label="OVC RunGen")
        ax.bar(x + width/2, m_ovc, width,
               bottom=rg_ovc,
               color=OVC_COLOR, alpha=0.9,
               hatch="\\\\\\", label="OVC Merge")

        ax.set_ylabel("Time (s)")
        ax.set_title(f"{dev.upper()} — Sorting Time Breakdown")
        ax.set_xticks(x)
        ax.set_xticklabels(threads_sorted)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize="small")

    axes[-1].set_xlabel("Number of Threads")

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_file = os.path.join(base_dir, "figure_breakdown_grouped_clean.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()