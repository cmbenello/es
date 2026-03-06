#!/usr/bin/env python3
"""
plot_thread_sweep_mem_heatmaps.py

Rewritten from scratch to be robust against:
- duplicate columns (the "Grouper for 'MergeThreads' not 1-dimensional" error)
- weird CSV widths / extra columns (we ignore unknown columns safely)
- missing metrics (we only plot what exists)
- lots of metrics (auto-generates many heatmaps)

Key behavior you asked for:
- X axis = RunGenThreads
- Y axis = MergeThreads
- (rg=1, mg=1) is bottom-left
- Annotate ALL non-empty cells
- Special combo heatmap: color = total sort time, annotation = # merge passes

Usage:
  python3 plot_thread_sweep_mem_heatmaps.py --summary path/to/summary.csv
  python3 plot_thread_sweep_mem_heatmaps.py --summary ... --out-dir heatmaps_all_v3
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def _dedupe_columns(cols: List[str]) -> List[str]:
    """
    If the CSV has duplicate headers, pandas can produce a DataFrame where selecting
    df['MergeThreads'] returns a DataFrame (2D) -> groupby/pivot explode.

    This makes column names unique by suffixing: col, col__2, col__3, ...
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def _first_present_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _auto_metric_cols(df: pd.DataFrame) -> List[str]:
    """
    Return numeric-ish columns worth plotting, excluding identifiers.
    We try to be conservative: must have at least 1 numeric value.
    """
    exclude = {
        "Config",
        "Label",
        "Name",
        "Notes",
    }
    key_cols = {"MemBudgetMB", "RunGenThreads", "MergeThreads", "RunSizePerThreadMB"}
    exclude |= key_cols

    metrics: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        # ignore dedupe variants of keys too
        if c.startswith("RunGenThreads__") or c.startswith("MergeThreads__") or c.startswith("MemBudgetMB__"):
            continue
        # keep only if any numeric exists
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            metrics.append(c)
    return metrics


def _pivot_grid(
    sub: pd.DataFrame,
    value_col: str,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """
    Pivot so:
      rows = MergeThreads (Y)
      cols = RunGenThreads (X)

    Sorted ascending so 1 is the smallest label.
    When plotted with origin='lower', (1,1) becomes bottom-left.
    """
    d = sub.copy()
    for c in ["RunGenThreads", "MergeThreads", value_col]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["RunGenThreads", "MergeThreads"])

    piv = d.pivot_table(
        index="MergeThreads",
        columns="RunGenThreads",
        values=value_col,
        aggfunc="mean",
        dropna=False,
    )

    piv = piv.sort_index(axis=0).sort_index(axis=1)

    rg_vals = [int(x) for x in piv.columns.tolist()]
    mg_vals = [int(x) for x in piv.index.tolist()]
    return piv, rg_vals, mg_vals


def _format_annot(val: float, kind: str) -> str:
    """
    kind: 'float', 'int', 'seconds', 'mb', 'throughput', etc.
    Keep it compact (heatmaps get dense).
    """
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return ""

    if kind == "int":
        return f"{int(round(val))}"
    if kind == "seconds":
        # show integer seconds if large, else 1 decimal
        if abs(val) >= 100:
            return f"{val:.0f}"
        return f"{val:.1f}"
    if kind == "mb":
        if abs(val) >= 1000:
            return f"{val:.0f}"
        return f"{val:.1f}"
    if kind == "throughput":
        # usually M entries/s or MB/s etc
        if abs(val) >= 10:
            return f"{val:.1f}"
        return f"{val:.2f}"
    # default
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def _infer_kind(col: str) -> str:
    c = col.lower()
    if "pass" in c and "merge" in c:
        return "int"
    if c.endswith("_s") or "time" in c or "total (s" in c or "rungen (s" in c or "merge (s" in c:
        return "seconds"
    if "mb" in c or "mib" in c or "bytes" in c:
        return "mb"
    if "throughput" in c:
        return "throughput"
    if "entries" in c and ("m" in c or "per" in c):
        return "throughput"
    if "imb" in c or "factor" in c:
        return "float"
    return "float"


def _plot_heatmap(
    grid: pd.DataFrame,
    title: str,
    out_path: Path,
    xlabel: str = "RunGenThreads",
    ylabel: str = "MergeThreads",
    annot_grid: Optional[pd.DataFrame] = None,
    annot_kind: str = "float",

    cmap: str = "viridis",
) -> None:
    """
    grid: numeric values for color
    annot_grid: values for text annotations (if None, annotate using grid)
    """
    # Convert to numpy with NaNs
    data = grid.to_numpy(dtype=float)

    # Build annotation strings
    if annot_grid is None:
        annot_vals = data
    else:
        annot_vals = annot_grid.to_numpy(dtype=float)

    ann_text = np.empty_like(annot_vals, dtype=object)
    for i in range(annot_vals.shape[0]):
        for j in range(annot_vals.shape[1]):
            v = annot_vals[i, j]
            if np.isnan(v) or np.isinf(v):
                ann_text[i, j] = ""
            else:
                ann_text[i, j] = _format_annot(float(v), annot_kind)

    # Plot
    fig_w = max(8.0, 0.55 * max(6, data.shape[1]))
    fig_h = max(6.0, 0.55 * max(6, data.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)

    # Ticks / labels
    xlabels = [str(int(x)) for x in grid.columns.tolist()]
    ylabels = [str(int(y)) for y in grid.index.tolist()]

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate all non-empty cells
    # choose contrasting text color based on colormap intensity:
    # (simple heuristic: use white on darker cells, black otherwise)
    # We'll compute normalized data; NaNs skip.
    finite = np.isfinite(data)
    if finite.any():
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        norm = (data - vmin) / denom
    else:
        norm = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isfinite(data[i, j]):
                continue
            txt = ann_text[i, j]
            if txt == "":
                continue
            color = "white" if norm[i, j] >= 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.csv")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: alongside summary as heatmaps_all_v3)",
    )
    ap.add_argument(
        "--only-mem",
        default=None,
        help="Optional: only plot one MemBudgetMB (e.g., 2048)",
    )
    ap.add_argument(
        "--no-combo",
        action="store_true",
        help="Disable the special combo heatmap (sort time w/ merge-pass annotations).",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"[ERR] summary not found: {summary_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (summary_path.parent / "heatmaps_all_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] writing plots to: {out_dir}")

    # Read CSV robustly:
    # - engine='python' handles uneven fields better
    # - on_bad_lines='skip' avoids blowing up on malformed rows
    df = pd.read_csv(
        summary_path,
        engine="python",
        on_bad_lines="skip",
    )

    # Dedupe columns to avoid 2D groupers
    df.columns = _dedupe_columns([str(c) for c in df.columns])

    # Find canonical key columns (handle accidental duplicates)
    mem_col = _first_present_col(df, ["MemBudgetMB", "MemBudgetMB__2", "MemBudgetMB__3"])
    rg_col  = _first_present_col(df, ["RunGenThreads", "RunGenThreads__2", "RunGenThreads__3"])
    mg_col  = _first_present_col(df, ["MergeThreads", "MergeThreads__2", "MergeThreads__3"])

    if mem_col is None or rg_col is None or mg_col is None:
        raise SystemExit(
            "[ERR] summary.csv must contain MemBudgetMB, RunGenThreads, MergeThreads (possibly duplicated headers)."
        )

    # Rename into stable names for downstream
    df = df.rename(columns={mem_col: "MemBudgetMB", rg_col: "RunGenThreads", mg_col: "MergeThreads"})

    # Ensure numeric for keys
    df = _normalize_numeric(df, ["MemBudgetMB", "RunGenThreads", "MergeThreads"])

    # Optionally filter mem
    if args.only_mem is not None:
        want = float(args.only_mem)
        df = df[df["MemBudgetMB"] == want].copy()

    if df.empty:
        raise SystemExit("[ERR] no rows after filtering; check --only-mem or summary contents")

    # Infer special columns for combo plot
    sort_time_col = _first_present_col(df, ["Sort_Time_s", "Total_s", "Total (s)", "Total_s__2"])
    merge_passes_col = None
    # try common names / patterns
    for c in df.columns:
        cl = c.lower()
        if ("merge" in cl and "pass" in cl) or cl in {"nummergepasses", "mergepasses", "merge_passes"}:
            merge_passes_col = c
            break

    # Choose all numeric metrics available
    metric_cols = _auto_metric_cols(df)


    # Make sure Sort_Time is included if present (even if name is weird)
    if sort_time_col and sort_time_col not in metric_cols:
        metric_cols.insert(0, sort_time_col)

    # Group per mem budget
    mem_vals = sorted([m for m in df["MemBudgetMB"].dropna().unique().tolist()])
    if not mem_vals:
        raise SystemExit("[ERR] no MemBudgetMB values found")

    # Plot each metric per mem
    for mem in mem_vals:
        sub = df[df["MemBudgetMB"] == mem].copy()
        if sub.empty:
            continue

        mem_dir = out_dir / f"mem_{int(mem)}"
        mem_dir.mkdir(parents=True, exist_ok=True)

        # Per metric heatmaps
        for metric in metric_cols:
            # Skip the merge passes heatmap if it's non-numeric / junk; we handle it too, but safe
            s = pd.to_numeric(sub[metric], errors="coerce")
            if not s.notna().any():
                continue

            grid, rg_vals, mg_vals = _pivot_grid(sub, metric)

            title = f"mem={int(mem)} | {metric} (Y=MergeThreads, X=RunGenThreads)"
            out_path = mem_dir / f"{metric}.png"

            kind = _infer_kind(metric)
            _plot_heatmap(
                grid=grid,
                title=title,
                out_path=out_path,
                xlabel="RunGenThreads",
                ylabel="MergeThreads",
                annot_grid=None,          # annotate with the same values
                annot_kind=kind,
            )

        # Special combo: color=sort time, annotation=#merge passes
        if (not args.no_combo) and sort_time_col and merge_passes_col:
            # numeric conversions
            sub2 = sub.copy()
            sub2[sort_time_col] = pd.to_numeric(sub2[sort_time_col], errors="coerce")
            sub2[merge_passes_col] = pd.to_numeric(sub2[merge_passes_col], errors="coerce")

            sort_grid, _, _ = _pivot_grid(sub2, sort_time_col)
            pass_grid, _, _ = _pivot_grid(sub2, merge_passes_col)

            out_path = mem_dir / "COMBO_sort_time_with_merge_passes.png"
            title = f"mem={int(mem)} | color=Sort_Time_s, text=#merge passes (Y=MergeThreads, X=RunGenThreads)"

            _plot_heatmap(
                grid=sort_grid,
                title=title,
                out_path=out_path,
                xlabel="RunGenThreads",
                ylabel="MergeThreads",
                annot_grid=pass_grid,
                annot_kind="int",
            )

        elif not args.no_combo:
            print(
                f"[WARN] mem={int(mem)}: combo plot skipped (need sort-time col + merge-pass col). "
                f"found sort_time={sort_time_col}, merge_passes={merge_passes_col}"
            )

    print(f"[DONE] wrote heatmaps under: {out_dir}")


if __name__ == "__main__":
    main()