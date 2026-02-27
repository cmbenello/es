#!/usr/bin/env bash
set -euo pipefail

# Local multi-merge test: 50 GiB gensort data, 14 threads, 384 MiB memory.
# Expects Nmerge ≈ 2-3 (with replacement selection) or 4-5 (without).
#
# Usage: ./local_multi_merge_test.sh <path/to/gensort.data> [output_dir]

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <path/to/gensort.data> [output_dir]" >&2
  exit 1
fi

INPUT_DATA=$1
if [[ ! -f "$INPUT_DATA" ]]; then
  echo "Input file not found: $INPUT_DATA" >&2
  exit 1
fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/multi_merge_test_${TS}"}
mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/multi_merge.log"

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
THREADS=14
MEM_MB=384
PAGE_SIZE_KB=64

rg_buf_mb=$(echo "scale=2; $MEM_MB / $THREADS" | bc)       # 27.43
FANIN=$(echo "$MEM_MB * 1024 / ($THREADS * $PAGE_SIZE_KB)" | bc)  # 439

echo "Building gen_sort_cli example (release)..."
cargo build --release --example gen_sort_cli >/dev/null
BINARY=./target/release/examples/gen_sort_cli

echo "================================================================"
echo "Total Memory:   ${MEM_MB} MiB"
echo "Threads:        $THREADS"
echo "Run Size:       ${rg_buf_mb} MiB"
echo "Fan-In:         $FANIN"
echo "Log:            $LOG_FILE"
echo "================================================================"

"$BINARY" \
  -n "MultiMerge_${MEM_MB}MB_T${THREADS}" \
  -i "$INPUT_DATA" \
  --run-gen-threads "$THREADS" \
  --merge-threads "$THREADS" \
  --rg-buf-mb "$rg_buf_mb" \
  --merge-fanin "$FANIN" \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --cooldown-seconds 0 \
  --dir "${OUT_DIR}/tmp" \
  2>&1 | tee "$LOG_FILE"

rm -rf "${OUT_DIR}/tmp"
echo "Log saved to: $LOG_FILE"
