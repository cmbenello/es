#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate rs_sweep_kvbin* KVBin_OVC_rs*.txt files and plot run_size vs total_time (no log scale)."
    )
    p.add_argument(
        "--root",
        default="benchmark_results",
        help="Root directory containing rs_sweep_kvbin* dirs (default: benchmark_results)",
    )
    p.add_argument(
        "--prefix",
        default="rs_sweep_kvbin",
        help="Directory prefix to include (default: rs_sweep_kvbin)",
    )
    p.add_argument(
        "--pattern",
        default="KVBin_OVC_rs",
        help="File prefix to include inside each dir (default: KVBin_OVC_rs)",
    )
    p.add_argument(
        "--out",
        default="rs_sweep_kvbin_all.png",
        help="Output PNG filename (default: rs_sweep_kvbin_all.png)",
    )
    return p.parse_args()


def extract_last_time_seconds(text: str):
    """
    Extract last line that looks like:
         152.09s
         186.73s
    """
    time_re = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*$")
    last = None
    for line in text.splitlines():
        m = time_re.match(line)
        if m:
            last = float(m.group(1))
    return last


def main():
    args = parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root dir {root} does not exist or is not a directory.")

    # Find dirs
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(args.prefix)
    )
    if not dirs:
        raise SystemExit(f"No directories starting with '{args.prefix}' found under {root}")

    print("Found result directories:")
    for d in dirs:
        print("  ", d.name)

    # Collect data
    times_by_rs = defaultdict(list)
    fname_re = re.compile(r"_rs(\d+)\.txt$")

    for d in dirs:
        for f in d.iterdir():
            if not f.is_file():
                continue
            if not f.name.startswith(args.pattern):
                continue

            m = fname_re.search(f.name)
            if not m:
                print(f"[WARN] Skipping {f} (no _rsXX suffix).")
                continue

            rs_mb = int(m.group(1))

            text = f.read_text()
            t = extract_last_time_seconds(text)
            if t is None:
                print(f"[WARN] Could not find time in {f}, skipping.")
                continue

            print(f"Parsed {f.name}: rs={rs_mb} MB, time={t:.2f} s")
            times_by_rs[rs_mb].append(t)

    if not times_by_rs:
        raise SystemExit("No usable data found in any KVBin_OVC_rs*.txt.")

    # Aggregate
    run_sizes = sorted(times_by_rs.keys())
    avg_times = []
    print("\nAggregated per run size:")
    for rs in run_sizes:
        vals = times_by_rs[rs]
        avg = sum(vals) / len(vals)
        avg_times.append(avg)
        print(f"  rs={rs:4d} MB -> n={len(vals)} runs, avg_time={avg:.3f} s")

    # Plot (no log scale!)
    plt.figure(figsize=(10, 6))
    plt.plot(run_sizes, avg_times, marker="o", linewidth=2)

    plt.xlabel("Run size (MB)")
    plt.ylabel("Total sort time (s)")
    plt.title(f"KVBin run-size sweep ({args.pattern}*, aggregated)")

    # Force ticks for each run-size
    plt.xticks(run_sizes, [str(rs) for rs in run_sizes], rotation=45)

    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()