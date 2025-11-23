#!/usr/bin/env bash
set -euo pipefail

# Where your benchmark results live (same root your scripts used)
BASE="${1:-/mnt/nvme1/cmbenello/es/benchmark_results}"

# Find the most recent summary.csv under $BASE
latest_summary() {
  # prefer newest file named summary.csv anywhere under BASE
  find "$BASE" -type f -name summary.csv -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1{print $2}'
}

SUM="$(latest_summary || true)"
if [[ -z "${SUM:-}" || ! -f "$SUM" ]]; then
  echo "No summary.csv found under: $BASE"
  exit 1
fi

DIR="$(dirname "$SUM")"
OUT="${DIR}/kvbin_only_summary_$(date +%Y%m%d_%H%M%S).csv"

echo "Using summary: $SUM"
echo "Writing KVBin-only CSV -> $OUT"
echo

# 1) Emit KVBin-only CSV
# Keep header, filter rows with Mode beginning with 'KVBin_'
awk -F',' 'NR==1 || $1 ~ /^KVBin_/ {print $0}' "$SUM" > "$OUT"

# 2) Pretty print to terminal as a small table:
#    Extract Variant (OVC/NoOVC) and RS (MB) from Mode like "KVBin_OVC_rs114"
#    Then print: RS_MB, Variant, Sort_Time_s, Throughput_MB_s, Peak_Memory_MB, Total_IO_MB
echo "KVBin results (latest run):"
echo "RS_MB,Variant,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB"
awk -F',' '
  NR==1 { next }                                # skip header here
  $1 ~ /^KVBin_/ {
    mode=$1
    # mode looks like KVBin_OVC_rs114 or KVBin_NoOVC_rs114
    variant = (mode ~ /KVBin_OVC/) ? "OVC" : "NoOVC"
    rs = mode
    sub(/^.*_rs/, "", rs)                        # strip up to _rs
    # Fields: Mode,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size
    printf "%s,%s,%s,%s,%s,%s\n", rs, variant, $2, $3, $4, $5
  }
' "$OUT" \
| sort -n -t',' -k1,1 -k2,2

echo
echo "Tip: pass a base directory to scan a different tree, e.g.:"
echo "  ./show_kvbin_results.sh /tank/local/cb/benchmark_results"
