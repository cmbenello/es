#!/usr/bin/env bash
set -euo pipefail

# GenSort benchmark runner mirroring yellow_taxi_sort_bench.sh.
# Usage: ./gensort_sort_bench.sh <path/to/gensort.data> [output_dir]

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
OUT_DIR=${2:-"logs/gensort_bench_${TS}"}
mkdir -p "$OUT_DIR"

# Fixed parameters (kept consistent with yellow_taxi_sort_bench.sh)
RUN_GEN_THREADS=32
MERGE_THREADS=32

# Build once to avoid repeated compiles
echo "Building gen_sort_cli example (release)..."
cargo build --release --example gen_sort_cli >/dev/null

run_case() {
  local name="$1"; shift
  local run_size_mb="$1"; shift
  local merge_fanin="$1"; shift
  # Any extra flags (e.g., --ovc) are passed through
  local extra_flags=("$@")

  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  # Label if OVC is enabled for this run
  local ovc_label="no"
  for f in "${extra_flags[@]}"; do
    if [[ "$f" == "--ovc" ]]; then
      ovc_label="yes"
      break
    fi
  done

  echo "[$name]: Run Size: ${run_size_mb} MB, Run Gen Threads: ${RUN_GEN_THREADS}, Merge Threads: ${MERGE_THREADS}, Run Gen Mem: ${run_gen_mem} MB, Merge Mem: ${merge_mem} MB, OVC: ${ovc_label}" | tee -a "${OUT_DIR}/${name}.log"

  cargo run --release --example gen_sort_cli -- \
    -n "$name" \
    -i "$INPUT_DATA" \
    --run-gen-threads "$RUN_GEN_THREADS" \
    --merge-threads "$MERGE_THREADS" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$merge_fanin" \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --dir "$temp_dir" \
    "${extra_flags[@]}" 2>&1 | tee -a "${OUT_DIR}/${name}.log"
}

# Baseline (no OVC) — fixed merge fan-in for single-step merge
run_case "Log_0.0"       "12.50"   20000
run_case "Log_0.25"      "37.61"   20000
run_case "Log_0.5"       "113.14"  20000
run_case "Log_0.75"      "340.37"  20000
run_case "Log_1.0"       "1024.00" 20000

# With OVC — fixed merge fan-in for single-step merge
run_case "Log_0.0_ovc"   "12.50"   20000 --ovc
run_case "Log_0.25_ovc"  "37.61"   20000 --ovc
run_case "Log_0.5_ovc"   "113.14"  20000 --ovc
run_case "Log_0.75_ovc"  "340.37"  20000 --ovc
run_case "Log_1.0_ovc"   "1024.00" 20000 --ovc

echo "All runs completed. Logs at: ${OUT_DIR}"
