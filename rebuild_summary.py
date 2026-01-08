#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

LABEL_RE = re.compile(r"^mem(?P<mem>\d+)_rg(?P<rg>\d+)_mg(?P<mg>\d+)$")

RUN_SIZE_RE = re.compile(r"^\s*Run size \(MB\):\s*(?P<x>[0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
EST_SIZE_RE = re.compile(r"^\s*Estimated data size:\s*(?P<x>[0-9]+(?:\.[0-9]+)?)\s*MB\s*$", re.MULTILINE)

# ---- Section anchors
MERGE_OPS_ANCHOR = "Merge Operations Summary"
DETAIL_IO_ANCHOR = "Detailed I/O Statistics"
BENCH_SUMMARY_ANCHOR = "Benchmark Results Summary"

def parse_label(label: str) -> Optional[Tuple[int, int, int]]:
    m = LABEL_RE.match(label.strip())
    if not m:
        return None
    return int(m.group("mem")), int(m.group("rg")), int(m.group("mg"))

def parse_run_size_per_thread_mb(text: str) -> Optional[float]:
    m = RUN_SIZE_RE.search(text)
    return float(m.group("x")) if m else None

def parse_estimated_data_mb(text: str) -> Optional[float]:
    m = EST_SIZE_RE.search(text)
    return float(m.group("x")) if m else None

def _extract_section_block(text: str, anchor: str) -> Optional[str]:
    """
    Return text from anchor line until the next huge separator block or EOF.
    This is tolerant of the repeated '=' / '-' separators your logs print.
    """
    idx = text.find(anchor)
    if idx < 0:
        return None
    tail = text[idx:]
    # Heuristic: end at the next "\n====" that starts a NEW major section AFTER some content,
    # but keep it simple: just return the tail; parsers below scan for rows they care about.
    return tail

def parse_benchmark_results_summary(text: str, label: str) -> Optional[Dict[str, float]]:
    """
    Parse the [avg] row if present, else [1] row from the "Benchmark Results Summary" table.
    Columns:
      Config RunSize RGRuns GenThr MergeThr Total(s) RunGen(s) Merge(s) Entries Throughput ReadMB WriteMB
    """
    block = _extract_section_block(text, BENCH_SUMMARY_ANCHOR)
    if block is None:
        return None

    best_line = None
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith(label + "["):
            continue
        if s.startswith(label + "[avg]"):
            best_line = s
            break
        if best_line is None and s.startswith(label + "[1]"):
            best_line = s

    if best_line is None:
        return None

    parts = best_line.split()
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
    Row looks like:
      mem2048_rg96_mg96[avg] 745427 1771070 2.38 20.60 20.60 5809788 10210507 1.76 150.80 150.80 ...
    That is groups of 5 per pass:
      PAvg, PMax, Imbal, Slow, Fast
    We return:
      MergePasses
      Imbal_pK, Slow_s_pK, Fast_s_pK
      MaxMergeImbal, MeanMergeImbal, LastMergeImbal
    """
    block = _extract_section_block(text, MERGE_OPS_ANCHOR)
    if block is None:
        return {}

    best_line = None
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith(label + "["):
            continue
        if s.startswith(label + "[avg]"):
            best_line = s
            break
        if best_line is None and s.startswith(label + "[1]"):
            best_line = s

    if best_line is None:
        return {}

    parts = best_line.split()
    if len(parts) < 2:
        return {}

    # After config token, everything should be numeric
    nums: List[float] = []
    for x in parts[1:]:
        try:
            nums.append(float(x))
        except ValueError:
            # bail if row is weird
            return {}

    # groups of 5: PAvg, PMax, Imbal, Slow, Fast
    if len(nums) % 5 != 0:
        # still try: floor
        n_passes = len(nums) // 5
    else:
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
    """
    block = _extract_section_block(text, DETAIL_IO_ANCHOR)
    if block is None:
        return {}

    best_line = None
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith(label + "["):
            continue
        if s.startswith(label + "[avg]"):
            best_line = s
            break
        if best_line is None and s.startswith(label + "[1]"):
            best_line = s

    if best_line is None:
        return {}

    parts = best_line.split()
    if len(parts) < 3:
        return {}

    nums: List[float] = []
    for x in parts[1:]:
        try:
            nums.append(float(x))
        except ValueError:
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

def parse_log_file(log_path: Path, label: str) -> Dict[str, Any]:
    text = log_path.read_text(errors="replace")

    parsed = parse_label(label)
    if not parsed:
        return {}

    mem_budget_mb, rg, mg = parsed

    run_size = parse_run_size_per_thread_mb(text)
    if run_size is None:
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
        "EstimatedDataMB": f"{est_mb:.2f}" if est_mb is not None else "",

        "Sort_Time_s": "",
        "RunGen_Time_s": "",
        "Merge_Time_s": "",
        "Entries": "",
        "Throughput_M_entries_s": "",
        "Read_MB": "",
        "Write_MB": "",
        "Total_IO_MB": "",

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

    # Merge ops (dynamic imbalances + times)
    for k, v in merge_ops.items():
        if k == "MergePasses":
            row["MergePasses"] = str(v)
        elif k in ("MaxMergeImbal", "MeanMergeImbal", "LastMergeImbal"):
            row[k] = f"{float(v):.3f}" if v != "" else ""
        elif k.startswith("Imbal_p"):
            row[k] = f"{float(v):.3f}"
        elif k.startswith("Slow_s_p") or k.startswith("Fast_s_p"):
            row[k] = f"{float(v):.2f}"

    # IO (dynamic per-pass IO)
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
        label = lf.stem
        row = parse_log_file(lf, label)
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

    # dynamic cols: Imbal/Slow/Fast/ReadMB/WriteMB per pass
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

    dyn_cols_sorted = sorted(dyn_cols, key=dyn_sort_key)

    fieldnames = base_order[:]
    for c in dyn_cols_sorted:
        if c not in fieldnames:
            fieldnames.append(c)

    # any extras (should be none)
    extras = set()
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                extras.add(k)
    for c in sorted(extras):
        fieldnames.append(c)

    # Sort: mem asc, rg desc, mg desc
    rows.sort(key=lambda r: (int(r["MemBudgetMB"]), -int(r["RunGenThreads"]), -int(r["MergeThreads"])))

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

def main():
    ap = argparse.ArgumentParser(description="Rebuild summary.csv from mem*_rg*_mg*.txt logs (dynamic merge passes).")
    ap.add_argument("dirs", nargs="+", help="Dirs containing mem*_rg*_mg*.txt logs")
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