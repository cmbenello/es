#!/usr/bin/env python3
import os
import glob
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Explicitly use these four experiments
TARGET_DIR_BASENAMES = [
    "kvbin_thread_sweep_yes_20251112_224857",  # NVMe, OVC
    "kvbin_thread_sweep_no_20251114_140717",   # NVMe, Baseline
    "kvbin_thread_sweep_no_20251115_174603",   # HDD, Baseline
    "kvbin_thread_sweep_yes_20251116_223628",  # HDD, OVC
]

# Manual mapping from dir → device + OVC, to avoid any ambiguity
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


def parse_device_from_path(lines):
    """Infer device from the KVBin filepath (as a sanity check only)."""
    for line in lines:
        if "KVBin file:" in line:
            path = line.split("KVBin file:")[1].strip()
            if "/nvme" in path:
                return "nvme"
            if "/tank" in path or "/local" in path:
                return "hdd"
            return "other"
    return "unknown"


def parse_ovc_from_log(lines):
    """Infer OVC enabled/disabled from the log (sanity check)."""
    for line in lines:
        if "OVC enabled:" in line:
            tail = line.split("OVC enabled:")[1].strip().lower()
            return "true" in tail
    return None


def parse_benchmark_summary(lines):
    """
    From one .txt log, extract:
      - Total_s (end-to-end)
      - Throughput (M entries/s)
      - RunGen_s
      - Merge_s
    using the no_name[avg] row in 'Benchmark Results Summary'.
    """
    start_idx = None
    for i, line in enumerate(lines):
        if "Benchmark Results Summary" in line:
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError("Benchmark Results Summary section not found")

    header_idx = None
    for i in range(start_idx, len(lines)):
        if lines[i].strip().startswith("Config") and "Total (s)" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Summary header line not found")

    avg_idx = None
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("no_name[avg]"):
            avg_idx = i
            break
    if avg_idx is None:
        raise RuntimeError("no_name[avg] row not found in summary table")

    row = lines[avg_idx].split()
    # Expected columns:
    # 0 Config, 1 RunSize, 2 RGRuns, 3 GenThr, 4 MergeThr,
    # 5 Total(s), 6 RunGen(s), 7 Merge(s),
    # 8 Entries, 9 Throughput, 10 ReadMB, 11 WriteMB
    if len(row) < 10:
        raise RuntimeError(f"Unexpected summary row format: {row}")

    total_s = float(row[5])
    run_gen_s = float(row[6])
    merge_s = float(row[7])
    throughput = float(row[9])
    return total_s, throughput, run_gen_s, merge_s


def parse_dir(dir_path):
    """
    For a kvbin_thread_sweep_* directory, parse each KVBin_*.txt log.

    Returns a DataFrame with:
      Dir, File, Device, OVC (bool), Threads,
      Total_s, Throughput_MB_s, RunGen_s, Merge_s
    """
    dir_base = os.path.basename(dir_path)
    records = []

    for path in glob.glob(os.path.join(dir_path, "KVBin_*_thr*_rs*.txt")):
        base = os.path.basename(path)
        m = re.search(r"_thr(\d+)_", base)
        if not m:
            continue
        threads = int(m.group(1))

        with open(path, "r") as f:
            lines = f.readlines()

        total_s, throughput, run_gen_s, merge_s = parse_benchmark_summary(lines)

        # Start from manual mapping (trusted)
        device = DIR_DEVICE_MAP.get(dir_base, parse_device_from_path(lines))
        ovc = DIR_OVC_MAP.get(dir_base, parse_ovc_from_log(lines))

        records.append(
            {
                "Dir": dir_base,
                "File": base,
                "Device": device,          # nvme / hdd / other
                "OVC": ovc,                # True / False / None
                "Threads": threads,
                "Total_s": total_s,
                "Throughput_MB_s": throughput,
                "RunGen_s": run_gen_s,
                "Merge_s": merge_s,
            }
        )

    if not records:
        raise RuntimeError(f"No KVBin_*_thr*_rs*.txt logs parsed in {dir_path}")

    return pd.DataFrame.from_records(records)


def label_for(device, ovc):
    # Pretty device name
    dev_label = "NVMe" if device == "nvme" else ("HDD" if device == "hdd" else device.upper())
    if ovc is True:
        return f"{dev_label} – OVC"
    if ovc is False:
        return f"{dev_label} – Baseline"
    return f"{dev_label} – Unknown"


def main():
    # base_dir is where benchmark_results live
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
    print(f"Using base_dir={base_dir}")

    target_dirs = [os.path.join(base_dir, d) for d in TARGET_DIR_BASENAMES]

    print("Using these 4 experiments explicitly:")
    for d in target_dirs:
        print("  ", d)

    dfs = []
    for d in target_dirs:
        if not os.path.isdir(d):
            print(f"Warning: directory not found, skipping: {d}")
            continue
        try:
            df = parse_dir(d)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to parse {d}: {e}")

    if not dfs:
        raise RuntimeError("No data parsed from any of the target directories.")

    full = pd.concat(dfs, ignore_index=True)

    # Map OVC to human labels for later if needed
    full["OVC_label"] = full["OVC"].map({True: "OVC", False: "Baseline"}).fillna("Unknown")

    # ---------- Figure 1: Throughput vs Threads (NVMe vs HDD) ----------
    fig1, axes1 = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    devices = ["nvme", "hdd"]

    for ax, device in zip(axes1, devices):
        sub = full[full["Device"] == device]
        if sub.empty:
            ax.set_visible(False)
            continue

        for ovc_val, group in sub.groupby("OVC"):
            label = label_for(device, ovc_val)
            g = group.groupby("Threads")["Throughput_MB_s"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=label)

        ax.set_ylabel("Throughput (M entries/s)")
        title_dev = "NVMe" if device == "nvme" else "HDD"
        ax.set_title(f"{title_dev} – Sorting throughput vs threads")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small", loc="upper left")

    axes1[-1].set_xlabel("Number of threads")

    fig1.tight_layout()
    out1 = os.path.join(base_dir, "figure_threads_vs_throughput_nvme_hdd.png")
    fig1.savefig(out1, bbox_inches="tight")
    print(f"Saved throughput figure to {out1}")

    # ---------- Figure 2: Total time vs Threads (NVMe vs HDD) ----------
    fig2, axes2 = plt.subplots(2, 1, sharex=True, figsize=(7, 7))

    for ax, device in zip(axes2, devices):
        sub = full[full["Device"] == device]
        if sub.empty:
            ax.set_visible(False)
            continue

        for ovc_val, group in sub.groupby("OVC"):
            label = label_for(device, ovc_val)
            g = group.groupby("Threads")["Total_s"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=label)

        ax.set_ylabel("Total time (s)")
        title_dev = "NVMe" if device == "nvme" else "HDD"
        ax.set_title(f"{title_dev} – End-to-end sort time vs threads")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small", loc="upper right")

    axes2[-1].set_xlabel("Number of threads")

    fig2.tight_layout()
    out2 = os.path.join(base_dir, "figure_threads_vs_total_time_nvme_hdd.png")
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved total-time figure to {out2}")


if __name__ == "__main__":
    main()