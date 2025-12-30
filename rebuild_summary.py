#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# -----------------------------
# Helpers
# -----------------------------

LABEL_RE = re.compile(r"^mem(?P<mem>\d+)_rg(?P<rg>\d+)_mg(?P<mg>\d+)$")


def parse_label(label: str) -> Optional[Tuple[int, int, int]]:
    """
    label: mem2048_rg32_mg8
    returns: (mem_budget_mb, run_gen_threads, merge_threads)
    """
    m = LABEL_RE.match(label.strip())
    if not m:
        return None
    return int(m.group("mem")), int(m.group("rg")), int(m.group("mg"))


def parse_run_size_per_thread_mb(text: str) -> Optional[float]:
    """
    Extract from bench output:
      Run size (MB): 114.00
    """
    m = re.search(r"^\s*Run size \(MB\):\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_avg_row_from_benchmark_summary(text: str, label: str) -> Optional[Dict[str, float]]:
    """
    Parse the [avg] row if present, else [1] row as fallback.

    We extract:
      total_s, run_gen_s, merge_s, throughput_M_entries_per_s, read_mb, write_mb

    Assumes your table row is whitespace-aligned and contains at least:
      Config ... Total(s) RunGen(s) Merge(s) ... Throughput ... ReadMB WriteMB
    """
    best = None
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith(label + "["):
            continue
        if s.startswith(label + "[avg]"):
            best = s
            break
        if best is None and s.startswith(label + "[1]"):
            best = s

    if best is None:
        return None

    parts = best.split()
    # Expected minimal length: 12 columns (see your original script)
    if len(parts) < 12:
        return None

    try:
        return {
            "total_s": float(parts[5]),
            "run_gen_s": float(parts[6]),
            "merge_s": float(parts[7]),
            "throughput_m_entries_s": float(parts[9]),
            "read_mb": float(parts[10]),
            "write_mb": float(parts[11]),
        }
    except ValueError:
        return None


def parse_log_file(log_path: Path, label: str) -> Dict[str, Any]:
    """
    Parse one log file into a row dict for summary.csv.
    Returns {} if label doesn't match expected mem*_rg*_mg* format.
    """
    text = log_path.read_text(errors="replace")

    parsed_label = parse_label(label)
    if not parsed_label:
        return {}

    mem_budget_mb, rg, mg = parsed_label

    run_size_per_thread = parse_run_size_per_thread_mb(text)
    if run_size_per_thread is None:
        run_size_per_thread = mem_budget_mb / rg

    metrics = parse_avg_row_from_benchmark_summary(text, label)

    row: Dict[str, Any] = {
        "Config": label,
        "MemBudgetMB": mem_budget_mb,
        "RunSizePerThreadMB": f"{run_size_per_thread:.2f}",
        "RunGenThreads": rg,
        "MergeThreads": mg,
        "Sort_Time_s": "",
        "RunGen_Time_s": "",
        "Merge_Time_s": "",
        "Throughput_M_entries_s": "",
        "Read_MB": "",
        "Write_MB": "",
        "Total_IO_MB": "",
    }

    if metrics:
        total_s = metrics["total_s"]
        run_gen_s = metrics["run_gen_s"]
        merge_s = metrics["merge_s"]
        thr = metrics["throughput_m_entries_s"]
        read_mb = metrics["read_mb"]
        write_mb = metrics["write_mb"]

        row["Sort_Time_s"] = f"{total_s:.2f}"
        row["RunGen_Time_s"] = f"{run_gen_s:.2f}"
        row["Merge_Time_s"] = f"{merge_s:.2f}"
        row["Throughput_M_entries_s"] = f"{thr:.4f}"
        row["Read_MB"] = f"{read_mb:.1f}"
        row["Write_MB"] = f"{write_mb:.1f}"
        row["Total_IO_MB"] = f"{(read_mb + write_mb):.1f}"

    return row


def build_summary_for_dir(logs_dir: Path, glob_pat: str, out_name: str) -> bool:
    """
    Build logs_dir/out_name from logs_dir/glob_pat.
    Returns True if wrote a CSV, False if skipped.
    """
    if not logs_dir.is_dir():
        print(f"[SKIP] not a directory: {logs_dir}")
        return False

    log_files = sorted(logs_dir.glob(glob_pat))
    if not log_files:
        print(f"[SKIP] no logs in {logs_dir} matching {glob_pat}")
        return False

    rows: List[Dict[str, Any]] = []
    for lf in log_files:
        label = lf.stem
        row = parse_log_file(lf, label)
        if row:
            rows.append(row)

    if not rows:
        print(f"[SKIP] parsed 0 rows in {logs_dir} (labels didn't match mem*_rg*_mg*)")
        return False

    # Sort: mem asc, rg desc, mg desc
    rows.sort(key=lambda r: (int(r["MemBudgetMB"]), -int(r["RunGenThreads"]), -int(r["MergeThreads"])))

    fieldnames = [
        "Config",
        "MemBudgetMB",
        "RunSizePerThreadMB",
        "RunGenThreads",
        "MergeThreads",
        "Sort_Time_s",
        "RunGen_Time_s",
        "Merge_Time_s",
        "Throughput_M_entries_s",
        "Read_MB",
        "Write_MB",
        "Total_IO_MB",
    ]

    summary_path = logs_dir / out_name
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_filled = sum(1 for r in rows if r["Sort_Time_s"] != "")
    print(f"[OK] {logs_dir}: wrote {len(rows)} rows → {summary_path}")
    print(f"     {n_filled}/{len(rows)} rows had parsed metrics")
    return True


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Option A: For each given directory, rebuild directory/summary.csv from mem*_rg*_mg*.txt logs."
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        help="One or more directories containing mem*_rg*_mg*.txt log files",
    )
    ap.add_argument(
        "--glob",
        default="mem*_rg*_mg*.txt",
        help="Glob pattern for log files (default: mem*_rg*_mg*.txt)",
    )
    ap.add_argument(
        "--out-name",
        default="summary.csv",
        help="Output CSV name inside each directory (default: summary.csv)",
    )
    args = ap.parse_args()

    wrote_any = False
    for d in args.dirs:
        wrote_any = build_summary_for_dir(Path(d), args.glob, args.out_name) or wrote_any

    if not wrote_any:
        raise SystemExit("[ERR] wrote 0 summaries (no dirs had parseable logs)")


if __name__ == "__main__":
    main()