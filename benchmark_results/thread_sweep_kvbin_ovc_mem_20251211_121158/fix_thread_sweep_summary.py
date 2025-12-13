#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

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
    Your Rust CLI prints a big "Benchmark Results Summary" table that includes:
      mem2048_rg32_mg8[1] ...
      mem2048_rg32_mg8[avg] ...

    We parse the [avg] row (preferred), else [1] fallback.

    We extract:
      total_s, run_gen_s, merge_s, throughput_M_entries_per_s, read_mb, write_mb

    Table columns in your output appear as:

    Config  Run Size (MB)  RG Runs  Gen Thr  Merge Thr  Total (s)  RunGen (s)  Merge (s)
    Entries  Throughput (M entries/s)  Read MB  Write MB
    """
    # find candidate lines that begin with the label + [avg] or [1]
    # We'll scan all lines and keep best match.
    best = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(label + "["):
            continue
        if line.startswith(label + "[avg]"):
            best = line
            break
        if best is None and line.startswith(label + "[1]"):
            best = line

    if best is None:
        return None

    # Split on whitespace; robust enough for your aligned table.
    parts = best.split()
    # Expected minimal length: config + run_size + rg_runs + gen_thr + merge_thr + total + run_gen + merge + entries + throughput + read + write
    if len(parts) < 12:
        return None

    # parts[0] = label[avg] / label[1]
    # parts[1] = run_size
    # parts[5] = total_s
    # parts[6] = run_gen_s
    # parts[7] = merge_s
    # parts[9] = throughput (M entries/s)
    # parts[10] = read_mb
    # parts[11] = write_mb
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
    Always returns a dict with required keys; metrics may be blank if missing.
    """
    text = log_path.read_text(errors="replace")

    parsed_label = parse_label(label)
    if not parsed_label:
        # Unknown label format; skip
        return {}

    mem_budget_mb, rg, mg = parsed_label

    # run size per thread is printed near top, but we also compute fallback = mem/rg
    run_size_per_thread = parse_run_size_per_thread_mb(text)
    if run_size_per_thread is None:
        run_size_per_thread = mem_budget_mb / rg

    metrics = parse_avg_row_from_benchmark_summary(text, label)

    # Fill row
    row: Dict[str, Any] = {
        "Config": label,
        "MemBudgetMB": mem_budget_mb,
        "RunSizePerThreadMB": f"{run_size_per_thread:.2f}",
        "RunGenThreads": rg,
        "MergeThreads": mg,
        # Metrics (may be blank)
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

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild thread_sweep_kvbin_ovc_mem summary.csv from mem*_rg*_mg*.txt logs."
    )
    ap.add_argument("summary_csv", help="Path to summary.csv (will be overwritten)")
    ap.add_argument(
        "--logs-dir",
        default=None,
        help="Directory containing the .txt logs (default: parent of summary.csv)",
    )
    ap.add_argument(
        "--glob",
        default="mem*_rg*_mg*.txt",
        help="Glob pattern for log files (default: mem*_rg*_mg*.txt)",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    logs_dir = Path(args.logs_dir) if args.logs_dir else summary_path.parent

    if not logs_dir.is_dir():
        raise SystemExit(f"[ERR] logs dir not found: {logs_dir}")

    log_files = sorted(logs_dir.glob(args.glob))
    if not log_files:
        raise SystemExit(f"[ERR] no logs found in {logs_dir} matching {args.glob}")

    rows = []
    for lf in log_files:
        label = lf.stem  # filename without .txt
        row = parse_log_file(lf, label)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit("[ERR] parsed 0 rows (labels didn't match mem*_rg*_mg*?)")

    # Sort for nicer reading: mem asc, rg desc, mg desc
    def sort_key(r: Dict[str, Any]):
        return (int(r["MemBudgetMB"]), -int(r["RunGenThreads"]), -int(r["MergeThreads"]))
    rows.sort(key=sort_key)

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

    # Overwrite summary.csv
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_filled = sum(1 for r in rows if r["Sort_Time_s"] != "")
    print(f"[OK] Wrote {len(rows)} rows to {summary_path}")
    print(f"[OK] {n_filled}/{len(rows)} rows had parsed metrics (Sort_Time_s present)")
    print(f"[OK] Logs dir: {logs_dir}")

if __name__ == "__main__":
    main()