#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

def parse_log(log_path: Path, label: str):
    """
    Parse one log file and return:
      run_size, total_s, run_gen_s, merge_s, throughput, read_mb, write_mb
    using the '... [avg]' row, or '[1]' as fallback.
    """
    if not log_path.is_file():
        print(f"[WARN] Log not found for {label}: {log_path}")
        return None

    avg_line = None
    first_line = None
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(label + "[avg]"):
                avg_line = line
                break
            if first_line is None and line.startswith(label + "[1]"):
                first_line = line

    line = avg_line or first_line
    if line is None:
        print(f"[WARN] No summary row for {label} in {log_path.name}")
        return None

    parts = line.split()
    if len(parts) < 12:
        print(f"[WARN] Unexpected summary format for {label}: {line}")
        return None

    # Columns (from Benchmark Results Summary row):
    # 0: Config
    # 1: Run Size (MB)
    # 2: RG Runs
    # 3: Gen Thr
    # 4: Merge Thr
    # 5: Total (s)
    # 6: RunGen (s)
    # 7: Merge (s)
    # 8: Entries
    # 9: Throughput (M entries/s)
    # 10: Read MB
    # 11: Write MB
    run_size   = float(parts[1])
    total_s    = float(parts[5])
    run_gen_s  = float(parts[6])
    merge_s    = float(parts[7])
    throughput = float(parts[9])
    read_mb    = float(parts[10])
    write_mb   = float(parts[11])

    return run_size, total_s, run_gen_s, merge_s, throughput, read_mb, write_mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="Path to summary.csv to rebuild in-place")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = summary_path.parent

    # Read existing rows
    rows = []
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    new_rows = []
    for row in rows:
        label = row.get("Config", "")
        if not label:
            continue

        log_path = out_dir / f"{label}.txt"
        parsed = parse_log(log_path, label)
        if parsed is None:
            # keep row but leave metrics blank
            new_rows.append(row)
            continue

        run_size, total_s, run_gen_s, merge_s, throughput, read_mb, write_mb = parsed
        total_io = read_mb + write_mb

        # Update / add fields
        row["RunSizeMB"]        = f"{run_size:.0f}"
        row["Sort_Time_s"]      = f"{total_s:.2f}"
        row["RunGen_Time_s"]    = f"{run_gen_s:.2f}"
        row["Merge_Time_s"]     = f"{merge_s:.2f}"
        # Name kept for backwards-compat, but this is actually M entries/s
        row["Throughput_MB_s"]  = f"{throughput:.4f}"
        row["Read_MB"]          = f"{read_mb:.1f}"
        row["Write_MB"]         = f"{write_mb:.1f}"
        row["Total_IO_MB"]      = f"{total_io:.1f}"
        # leave Peak_Memory_MB, Avg_Key_Size, Avg_Value_Size as-is if present

        new_rows.append(row)

    # Define full header (old + new)
    fieldnames = [
        "Config",
        "RunGenThreads",
        "MergeThreads",
        "RunSizeMB",
        "Sort_Time_s",
        "RunGen_Time_s",
        "Merge_Time_s",
        "Throughput_MB_s",
        "Peak_Memory_MB",
        "Total_IO_MB",
        "Read_MB",
        "Write_MB",
        "Avg_Key_Size",
        "Avg_Value_Size",
    ]

    # Ensure every row has all fields
    for row in new_rows:
        for fn in fieldnames:
            row.setdefault(fn, "")

    # Overwrite summary.csv with updated values
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print(f"Rebuilt summary with metrics: {summary_path}")

if __name__ == "__main__":
    main()