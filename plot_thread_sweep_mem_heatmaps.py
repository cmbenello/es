#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_grid(df, xcol, ycol, zcol):
    x_vals = sorted(df[xcol].unique())
    y_vals = sorted(df[ycol].unique())
    Z = np.full((len(y_vals), len(x_vals)), np.nan)

    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}

    for _, row in df.iterrows():
        xi = x_index[int(row[xcol])]
        yi = y_index[int(row[ycol])]
        Z[yi, xi] = float(row[zcol])

    return x_vals, y_vals, Z

def plot_one(df_sub, title, xcol, ycol, zcol, cmap, vmin, vmax, out_path):
    x_vals, y_vals, Z = build_grid(df_sub, xcol, ycol, zcol)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    ax.set_xlabel("Run generation threads")
    ax.set_ylabel("Merge threads")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(zcol)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", default="thread_sweep_kvbin_ovc_mem")
    ap.add_argument("--cmap", default="coolwarm")  # blue(low) -> red(high)
    ap.add_argument("--facet", choices=["MemBudgetMB", "RunSizePerThreadMB"], default="MemBudgetMB")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)

    needed = ["RunGenThreads", "MergeThreads", args.facet, "Sort_Time_s", "RunGen_Time_s", "Merge_Time_s"]
    for col in needed:
        if col not in df.columns:
            raise SystemExit(f"[ERR] Missing column '{col}' in {args.summary}. Have: {list(df.columns)}")

    # numeric
    for col in ["RunGenThreads", "MergeThreads", "Sort_Time_s", "RunGen_Time_s", "Merge_Time_s", args.facet]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep rows with at least total time
    df = df.dropna(subset=["RunGenThreads", "MergeThreads", args.facet, "Sort_Time_s"])

    metrics = [
        ("Sort_Time_s", "Total sort time (s)", "total"),
        ("RunGen_Time_s", "Run-generation time (s)", "rungen"),
        ("Merge_Time_s", "Merge time (s)", "merge"),
    ]

    for metric_col, metric_label, suffix in metrics:
        d = df.dropna(subset=[metric_col])
        if d.empty:
            print(f"[WARN] No rows for {metric_col}, skipping")
            continue

        vmin = float(d[metric_col].min())
        vmax = float(d[metric_col].max())
        print(f"Global color scale for {metric_col}: {vmin:.2f} → {vmax:.2f}")

        for facet_val in sorted(d[args.facet].unique()):
            sub = d[d[args.facet] == facet_val]
            if sub.empty:
                continue

            facet_str = f"{args.facet}={int(facet_val) if float(facet_val).is_integer() else facet_val}"
            title = f"KVBin OVC thread sweep — {facet_str}\ncolor = {metric_label}"

            out_path = out_dir / f"{args.out_prefix}_{suffix}_{args.facet}_{facet_val}.png"
            plot_one(
                sub,
                title=title,
                xcol="RunGenThreads",
                ycol="MergeThreads",
                zcol=metric_col,
                cmap=args.cmap,
                vmin=vmin,
                vmax=vmax,
                out_path=out_path,
            )

if __name__ == "__main__":
    main()