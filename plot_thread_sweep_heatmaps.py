#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_grid(df, metric_col):
    x_vals = sorted(df["RunGenThreads"].unique())
    y_vals = sorted(df["MergeThreads"].unique())
    Z = np.full((len(y_vals), len(x_vals)), np.nan)

    for _, row in df.iterrows():
        xi = x_vals.index(int(row["RunGenThreads"]))
        yi = y_vals.index(int(row["MergeThreads"]))
        Z[yi, xi] = row[metric_col]

    return x_vals, y_vals, Z


def plot_heatmap(df_rs, run_size, metric_col, metric_label,
                 vmin, vmax, out_path):
    df = df_rs.copy()

    fig, ax = plt.subplots(figsize=(6, 5))

    x_vals, y_vals, Z = build_grid(df, metric_col)

    im = ax.imshow(
        Z,
        origin="lower",
        cmap="viridis",    # or your blue→red cmap if you change it
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    ax.set_xlabel("Run generation threads")
    ax.set_ylabel("Merge threads")
    ax.set_title(
        f"KVBin OVC thread sweep — run_size={int(run_size)} MB\n"
        f"color = {metric_label}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--out-prefix",
        default="thread_sweep_kvbin_ovc_shared",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)

    # Force numeric
    for col in [
        "RunGenThreads",
        "MergeThreads",
        "RunSizeMB",
        "Sort_Time_s",
        "RunGen_Time_s",
        "Merge_Time_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep rows where we at least have total sort time
    df = df.dropna(subset=["RunGenThreads", "MergeThreads", "RunSizeMB", "Sort_Time_s"])

    # List of metrics to plot: (column, human label, suffix)
    metrics = [
        ("Sort_Time_s", "Total sort time (s)", "total"),
        ("RunGen_Time_s", "Run-generation time (s)", "rungen"),
        ("Merge_Time_s", "Merge time (s)", "merge"),
    ]

    for metric_col, metric_label, suffix in metrics:
        if metric_col not in df.columns:
            print(f"[WARN] Column '{metric_col}' not in summary; skipping {metric_label}")
            continue

        df_metric = df.dropna(subset=[metric_col])
        if df_metric.empty:
            print(f"[WARN] No data for metric '{metric_col}', skipping")
            continue

        # Global color scale for this metric across all run sizes
        global_min = df_metric[metric_col].min()
        global_max = df_metric[metric_col].max()
        print(
            f"Global color scale for {metric_col}: "
            f"{global_min:.2f}s → {global_max:.2f}s"
        )

        # One plot per run size for this metric
        for rs in sorted(df_metric["RunSizeMB"].unique()):
            df_rs = df_metric[df_metric["RunSizeMB"] == rs]
            out_path = out_dir / f"{args.out_prefix}_rs{int(rs)}_{suffix}.png"
            plot_heatmap(df_rs, rs, metric_col, metric_label,
                         global_min, global_max, out_path)


if __name__ == "__main__":
    main()