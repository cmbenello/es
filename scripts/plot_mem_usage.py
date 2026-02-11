#!/usr/bin/env python3
"""
Plot memory usage over time from *_mem.log files produced by the benchmark scripts.

Each mem log contains lines of the form:
    elapsed_ms rss_kb

Usage:
    python3 plot_mem_usage.py <log_dir_or_mem_log_files...> [-o output.png]

Examples:
    python3 plot_mem_usage.py logs/resource_bench_2026-01-01_00-00-00/
    python3 plot_mem_usage.py logs/resource_bench_*/Exp1_*_mem.log -o exp1_mem.png
"""

import sys
import os
import glob
import argparse
import matplotlib.pyplot as plt


def parse_mem_log(path):
    times_s, rss_mb = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    times_s.append(int(parts[0]) / 1000.0)   # ms -> s
                    rss_mb.append(int(parts[1]) / 1024.0)     # KB -> MB
                except ValueError:
                    continue
    return times_s, rss_mb


def collect_files(args):
    files = []
    for arg in args:
        if os.path.isdir(arg):
            files.extend(sorted(glob.glob(os.path.join(arg, "*_mem.log"))))
        else:
            files.extend(sorted(glob.glob(arg)))
    return files


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark memory usage over time")
    parser.add_argument("inputs", nargs="+", help="Log dir(s) or *_mem.log file(s)")
    parser.add_argument("-o", "--output", default="mem_usage.png", help="Output image path")
    opts = parser.parse_args()

    files = collect_files(opts.inputs)
    if not files:
        print("No *_mem.log files found.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for path in files:
        times_s, rss_mb = parse_mem_log(path)
        if not times_s:
            continue
        label = os.path.basename(path).replace("_mem.log", "")
        ax.plot(times_s, rss_mb, label=label, linewidth=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSS (MB)")
    ax.set_title("Memory Usage Over Time")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(opts.output, dpi=150)
    print(f"Saved: {opts.output}")


if __name__ == "__main__":
    main()
