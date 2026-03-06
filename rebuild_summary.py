#!/usr/bin/env python3
"""
rebuild_summary.py

Rebuild summary.csv from benchmark log files named like:

  mem8192_rg96_mg80.txt
  mem4096_rg8_mg48_F1365.txt   (variable merge fan-in suffix)

Key features:
- Accepts optional _F<fanin> suffix in filename stem.
- Also parses "Merge fan-in:" from the log body (preferred, if present).
- Parses metrics from:
  - Benchmark Results Summary
  - Merge Operations Summary
  - Detailed I/O Statistics
- Supports dynamic number of merge passes (M1, M2, ...).
- Emits a CSV with stable base columns + dynamic per-pass columns.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


# --------------------------
# Filename label parsing
# --------------------------
# Accept:
#   mem<MB>_rg<threads>_mg<threads>
#   mem<MB>_rg<threads>_mg<threads>_F<fanin>
LABEL_RE = re.compile(
    r"^mem(?P<mem>\d+)_rg(?P<rg>\d+)_mg(?P<mg>\d+)(?:_F(?P<fanin>\d+))?$"
)

def parse_label(label: str) -> Optional[Tuple[int, int, int, Optional[int]]]:
    m = LABEL_RE.match(label.strip())
    if not m:
        return None
    fanin = m.group("fanin")
    return (
        int(m.group("mem")),
        int(m.group("rg")),
        int(m.group("mg")),
        int(fanin) if fanin is not None else None,
    )


# --------------------------
# Single-value parsers (from log body)
# --------------------------
RUN_SIZE_RE = re.compile(r"^\s*Run size \(MB\):\s*(?P<x>[0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
EST_SIZE_RE = re.compile(r"^\s*Estimated data size:\s*(?P<x>[0-9]+(?:\.[0-9]+)?)\s*MB\s*$", re.MULTILINE)
MERGE_FANIN_RE = re.compile(r"^\s*Merge fan-in:\s*(?P<x>\d+)\s*$", re.MULTILINE)

def parse_run_size_per_thread_mb(text: str) -> Optional[float]:
    m = RUN_SIZE_RE.search(text)
    return float(m.group("x")) if m else None

def parse_estimated_data_mb(text: str) -> Optional[float]:
    m = EST_SIZE_RE.search(text)
    return float(m.group("x")) if m else None

def parse_merge_fanin(text: str) -> Optional[int]:
    m = MERGE_FANIN_RE.search(text)
    return int(m.group("x")) if m else None

def parse_estimated_data_mb(text: str) -> Optional[float]:
    m = EST_SIZE_RE.search(text)
    return float(m.group("x")) if m else None

# --------------------------
# Section anchors
# --------------------------
MERGE_OPS_ANCHOR = "Merge Operations Summary"
DETAIL_IO_ANCHOR = "Detailed I/O Statistics"
BENCH_SUMMARY_ANCHOR = "Benchmark Results Summary"


def _extract_section_tail(text: str, anchor: str) -> Optional[str]:
    """
    Return the text starting from the first occurrence of `anchor` to EOF.
    We keep it simple and just scan lines from there.
    """
    idx = text.find(anchor)
    if idx < 0:
        return None
    return text[idx:]


def _pick_best_config_line(block: str, label: str) -> Optional[str]:
    """
    In a table block, pick the best row for label:
      prefer label[avg], else label[1], else first label[*] row.
    Returns the stripped line.
    """
    best_line = None
    fallback_line = None

    for line in block.splitlines():
        s = line.strip()
        if not s.startswith(label + "["):
            continue

        # strongest preference
        if s.startswith(label + "[avg]"):
            return s

        # next preference
        if best_line is None and s.startswith(label + "[1]"):
            best_line = s

        # fallback: first occurrence of any run index
        if fallback_line is None:
            fallback_line = s

    return best_line or fallback_line


# --------------------------
# Table parsers
# --------------------------
def parse_benchmark_results_summary(text: str, label: str) -> Optional[Dict[str, float]]:
    """
    Parse the label[avg] row if present, else label[1], from "Benchmark Results Summary".
    Expected columns (space separated):
      Config RunSize RGRuns GenThr MergeThr Total(s) RunGen(s) Merge(s) Entries Throughput ReadMB WriteMB

    Returns dict keys:
      total_s, run_gen_s, merge_s, entries, throughput_m_entries_s, read_mb, write_mb
    """
    block = _extract_section_tail(text, BENCH_SUMMARY_ANCHOR)
    if block is None:
        return None

    line = _pick_best_config_line(block, label)
    if line is None:
        return None

    parts = line.split()
    # parts[0] = "label[avg]" etc.
    # Then:
    # 1 RunSize
    # 2 RGRuns
    # 3 GenThr
    # 4 MergeThr
    # 5 Total(s)
    # 6 RunGen(s)
    # 7 Merge(s)
    # 8 Entries
    # 9 Throughput
    # 10 ReadMB
    # 11 WriteMB
    if len(parts) < 12:
        return None

    try:
        return {
            "total_s": float(parts[5]),
            "run_gen_s": float(parts[6]),
            "merge_s": float(parts[7]),
            "entries": float(parts[8]),
            "throughput_m_entries_s": float(parts[9]),
            "read_mb": float(parts[10]),
            "write_mb": float(parts[11]),
        }
    except ValueError:
        return None


def parse_merge_operations_summary(text: str, label: str) -> Dict[str, Any]:
    """
    Parse dynamic merge passes from "Merge Operations Summary".

    The row (after the config token) is numeric in groups of 5 per pass:
      PAvg, PMax, Imbal, Slow, Fast

    Returns keys:
      MergePasses
      Imbal_pK, Slow_s_pK, Fast_s_pK  for K=1..MergePasses
      MaxMergeImbal, MeanMergeImbal, LastMergeImbal
    """
    block = _extract_section_tail(text, MERGE_OPS_ANCHOR)
    if block is None:
        return {}

    line = _pick_best_config_line(block, label)
    if line is None:
        return {}

    parts = line.split()
    if len(parts) < 2:
        return {}

    nums: List[float] = []
    for tok in parts[1:]:
        try:
            nums.append(float(tok))
        except ValueError:
            return {}

    if not nums:
        return {}

    # Groups of 5 per pass
    n_passes = len(nums) // 5
    if n_passes <= 0:
        return {}

    out: Dict[str, Any] = {"MergePasses": n_passes}

    imbal_list: List[float] = []
    last_imbal: Optional[float] = None

    for p in range(1, n_passes + 1):
        base = (p - 1) * 5
        # pavg = nums[base + 0]
        # pmax = nums[base + 1]
        imbal = nums[base + 2]
        slow = nums[base + 3]
        fast = nums[base + 4]

        out[f"Imbal_p{p}"] = imbal
        out[f"Slow_s_p{p}"] = slow
        out[f"Fast_s_p{p}"] = fast

        imbal_list.append(imbal)
        last_imbal = imbal

    out["MaxMergeImbal"] = max(imbal_list) if imbal_list else ""
    out["MeanMergeImbal"] = (sum(imbal_list) / len(imbal_list)) if imbal_list else ""
    out["LastMergeImbal"] = last_imbal if last_imbal is not None else ""
    return out


def parse_detailed_io_stats(text: str, label: str) -> Dict[str, Any]:
    """
    Parse dynamic per-pass IO + RG IO from "Detailed I/O Statistics".

    Row:
      label[avg] RGRead RGWrite M1Read M1Write M2Read M2Write ...

    Returns keys:
      RG_Read_MB, RG_Write_MB
      ReadMB_pK, WriteMB_pK
      TotalMergeRead_MB, TotalMergeWrite_MB
    """
    block = _extract_section_tail(text, DETAIL_IO_ANCHOR)
    if block is None:
        return {}

    line = _pick_best_config_line(block, label)
    if line is None:
        return {}

    parts = line.split()
    if len(parts) < 3:
        return {}

    nums: List[float] = []
    for tok in parts[1:]:
        try:
            nums.append(float(tok))
        except ValueError:
            return {}

    if len(nums) < 2:
        return {}

    out: Dict[str, Any] = {}
    out["RG_Read_MB"] = nums[0]
    out["RG_Write_MB"] = nums[1]

    merge_pairs = nums[2:]
    pass_idx = 1
    total_r = 0.0
    total_w = 0.0
    for i in range(0, len(merge_pairs), 2):
        if i + 1 >= len(merge_pairs):
            break
        r = merge_pairs[i]
        w = merge_pairs[i + 1]
        out[f"ReadMB_p{pass_idx}"] = r
        out[f"WriteMB_p{pass_idx}"] = w
        total_r += r
        total_w += w
        pass_idx += 1

    if pass_idx > 1:
        out["TotalMergeRead_MB"] = total_r
        out["TotalMergeWrite_MB"] = total_w

    return out


# --------------------------
# Main file parse
# --------------------------
def parse_log_file(log_path: Path) -> Dict[str, Any]:
    label = log_path.stem
    parsed = parse_label(label)
    if not parsed:
        return {}

    mem_budget_mb, rg, mg, fanin_from_name = parsed

    text = log_path.read_text(errors="replace")

    # Prefer fan-in from log body; fall back to filename suffix.
    fanin_from_log = parse_merge_fanin(text)
    merge_fanin = fanin_from_log if fanin_from_log is not None else fanin_from_name

    run_size = parse_run_size_per_thread_mb(text)
    if run_size is None:
        # historical fallback used in your old script
        run_size = mem_budget_mb / rg

    est_mb = parse_estimated_data_mb(text)

    bench = parse_benchmark_results_summary(text, label) or {}
    merge_ops = parse_merge_operations_summary(text, label) or {}
    io = parse_detailed_io_stats(text, label) or {}

    row: Dict[str, Any] = {
        "Config": label,
        "MemBudgetMB": mem_budget_mb,
        "RunSizePerThreadMB": f"{run_size:.2f}",
        "RunGenThreads": rg,
        "MergeThreads": mg,
        "MergeFanin": str(merge_fanin) if merge_fanin is not None else "",
        "EstimatedDataMB": f"{est_mb:.2f}" if est_mb is not None else "",

        # Benchmark summary fields
        "Sort_Time_s": "",
        "RunGen_Time_s": "",
        "Merge_Time_s": "",
        "Entries": "",
        "Throughput_M_entries_s": "",
        "Read_MB": "",
        "Write_MB": "",
        "Total_IO_MB": "",

        # I/O breakdown
        "RG_Read_MB": "",
        "RG_Write_MB": "",
        "MergePasses": "",
        "MaxMergeImbal": "",
        "MeanMergeImbal": "",
        "LastMergeImbal": "",
        "TotalMergeRead_MB": "",
        "TotalMergeWrite_MB": "",
    }

    if bench:
        row["Sort_Time_s"] = f"{bench['total_s']:.2f}"
        row["RunGen_Time_s"] = f"{bench['run_gen_s']:.2f}"
        row["Merge_Time_s"] = f"{bench['merge_s']:.2f}"
        row["Entries"] = f"{bench['entries']:.0f}"
        row["Throughput_M_entries_s"] = f"{bench['throughput_m_entries_s']:.4f}"
        row["Read_MB"] = f"{bench['read_mb']:.1f}"
        row["Write_MB"] = f"{bench['write_mb']:.1f}"
        row["Total_IO_MB"] = f"{(bench['read_mb'] + bench['write_mb']):.1f}"

    # Merge ops: dynamic imbalances + times
    for k, v in merge_ops.items():
        if k == "MergePasses":
            row["MergePasses"] = str(v)
        elif k in ("MaxMergeImbal", "MeanMergeImbal", "LastMergeImbal"):
            row[k] = f"{float(v):.3f}" if v != "" else ""
        elif k.startswith("Imbal_p"):
            row[k] = f"{float(v):.3f}"
        elif k.startswith("Slow_s_p") or k.startswith("Fast_s_p"):
            row[k] = f"{float(v):.2f}"

    # IO breakdown: dynamic per-pass IO
    if "RG_Read_MB" in io:
        row["RG_Read_MB"] = f"{io['RG_Read_MB']:.1f}"
    if "RG_Write_MB" in io:
        row["RG_Write_MB"] = f"{io['RG_Write_MB']:.1f}"
    if "TotalMergeRead_MB" in io:
        row["TotalMergeRead_MB"] = f"{io['TotalMergeRead_MB']:.1f}"
    if "TotalMergeWrite_MB" in io:
        row["TotalMergeWrite_MB"] = f"{io['TotalMergeWrite_MB']:.1f}"

    for k, v in io.items():
        if k.startswith("ReadMB_p") or k.startswith("WriteMB_p"):
            row[k] = f"{float(v):.1f}"

    return row


# --------------------------
# CSV building
# --------------------------
def build_summary_for_dir(logs_dir: Path, glob_pat: str, out_name: str) -> bool:
    if not logs_dir.is_dir():
        print(f"[SKIP] not a directory: {logs_dir}")
        return False

    log_files = sorted(logs_dir.glob(glob_pat))
    if not log_files:
        print(f"[SKIP] no logs in {logs_dir} matching {glob_pat}")
        return False

    rows: List[Dict[str, Any]] = []
    for lf in log_files:
        row = parse_log_file(lf)
        if row:
            rows.append(row)

    if not rows:
        print(f"[SKIP] parsed 0 rows in {logs_dir}")
        return False

    base_order = [
        "Config",
        "MemBudgetMB",
        "RunSizePerThreadMB",
        "RunGenThreads",
        "MergeThreads",
        "MergeFanin",
        "EstimatedDataMB",

        "Sort_Time_s",
        "RunGen_Time_s",
        "Merge_Time_s",
        "Entries",
        "Throughput_M_entries_s",
        "Read_MB",
        "Write_MB",
        "Total_IO_MB",

        "RG_Read_MB",
        "RG_Write_MB",

        "MergePasses",
        "MaxMergeImbal",
        "MeanMergeImbal",
        "LastMergeImbal",

        "TotalMergeRead_MB",
        "TotalMergeWrite_MB",
    ]

    # dynamic columns: Imbal/Slow/Fast/ReadMB/WriteMB per pass
    dyn_cols = set()
    for r in rows:
        for k in r.keys():
            if k.startswith(("Imbal_p", "Slow_s_p", "Fast_s_p", "ReadMB_p", "WriteMB_p")):
                dyn_cols.add(k)

    def dyn_sort_key(col: str):
        m = re.search(r"_p(\d+)$", col)
        p = int(m.group(1)) if m else 10**9
        # group ordering per pass
        if col.startswith("Imbal_p"):
            g = 0
        elif col.startswith("Slow_s_p"):
            g = 1
        elif col.startswith("Fast_s_p"):
            g = 2
        elif col.startswith("ReadMB_p"):
            g = 3
        else:
            g = 4
        return (p, g, col)

    fieldnames = base_order[:]
    for c in sorted(dyn_cols, key=dyn_sort_key):
        if c not in fieldnames:
            fieldnames.append(c)

    # any extras (should be none, but keep robust)
    extras = set()
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                extras.add(k)
    for c in sorted(extras):
        fieldnames.append(c)

    # Sort: mem asc, rg desc, mg desc, fanin asc (blank fanin last)
    def safe_int(x: str, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    rows.sort(key=lambda r: (
        safe_int(str(r.get("MemBudgetMB", "")), 10**18),
        -safe_int(str(r.get("RunGenThreads", "")), -10**18),
        -safe_int(str(r.get("MergeThreads", "")), -10**18),
        safe_int(str(r.get("MergeFanin", "")), 10**18),
    ))

    summary_path = logs_dir / out_name
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_filled = sum(1 for r in rows if r.get("Sort_Time_s", "") != "")
    print(f"[OK] {logs_dir}: wrote {len(rows)} rows → {summary_path}")
    print(f"     {n_filled}/{len(rows)} rows had parsed benchmark summary metrics")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild summary.csv from mem*_rg*_mg*.txt logs (supports optional _F<fanin> suffix)."
    )
    ap.add_argument("dirs", nargs="+", help="Dirs containing log files")
    ap.add_argument("--glob", default="mem*_rg*_mg*.txt", help="Glob for logs (default: mem*_rg*_mg*.txt)")
    ap.add_argument("--out-name", default="summary.csv", help="Output CSV name (default: summary.csv)")
    args = ap.parse_args()

    wrote_any = False
    for d in args.dirs:
        wrote_any = build_summary_for_dir(Path(d), args.glob, args.out_name) or wrote_any

    if not wrote_any:
        raise SystemExit("[ERR] wrote 0 summaries (no dirs had parseable logs)")

if __name__ == "__main__":
    main()