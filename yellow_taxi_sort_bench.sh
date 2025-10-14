#!/usr/bin/env bash
set -euo pipefail

# Yellow Taxi benchmark runner for specified configurations.
# Usage: ./yellow_taxi_sort_bench.sh <path/to/yellow_taxi.csv> [output_dir]

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <path/to/yellow_taxi.csv> [output_dir]" >&2
  exit 1
fi

INPUT_CSV=$1
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/yellow_taxi_bench_${TS}"}
mkdir -p "$OUT_DIR"

# Fixed parameters from the request
RUN_GEN_THREADS=32
MERGE_THREADS=32

# Build once to avoid repeated compiles
echo "Building yellow_taxi_benchmark_cli example (release)..."
cargo build --release --example yellow_taxi_benchmark_cli >/dev/null

# Cooldown between runs to prevent SSD thermal throttling
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-60}
cooldown() {
  echo "Cooling down for ${COOLDOWN_SECONDS}s to let SSD rest..."
  sleep "$COOLDOWN_SECONDS"
}

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

  # Derived memory figures for context (MiB)
  echo "[$name]: Run Size: ${run_size_mb} MB, Run Gen Threads: ${RUN_GEN_THREADS}, Merge Threads: ${MERGE_THREADS}, OVC: ${ovc_label}" | tee -a "${OUT_DIR}/${name}.log"

  # Note: --headers flag assumes the input CSV has a header row.
  # Key columns default to "2,3"; value column default to "0".
  cargo run --release --example yellow_taxi_benchmark_cli -- \
    -n "$name" \
    -i "$INPUT_CSV" \
    --run-gen-threads "$RUN_GEN_THREADS" \
    --merge-threads "$MERGE_THREADS" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$merge_fanin" \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --cooldown-seconds 60 \
    --dir "$temp_dir" \
    "${extra_flags[@]}" 2>&1 | tee -a "${OUT_DIR}/${name}.log"
}

# Requested configurations
# Baseline (no OVC)
run_case "Log_0.0"       "3.91"    20000
cooldown
run_case "Log_0.25"      "15.72"   20000
cooldown
run_case "Log_0.5"       "63.25"   20000
cooldown
run_case "Log_0.75"      "254.49"  20000
cooldown
run_case "Log_1.0"       "1024.00" 20000
cooldown

# With OVC
run_case "Log_0.0_ovc"   "3.91"    20000 --ovc
cooldown
run_case "Log_0.25_ovc"  "15.72"   20000 --ovc
cooldown
run_case "Log_0.5_ovc"   "63.25"   20000 --ovc
cooldown
run_case "Log_0.75_ovc"  "254.49"  20000 --ovc
cooldown
run_case "Log_1.0_ovc"   "1024.00" 20000 --ovc
cooldown

echo "All runs completed. Logs at: ${OUT_DIR}"
