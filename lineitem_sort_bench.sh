#!/usr/bin/env bash
set -euo pipefail

# Lineitem benchmark runner modeled after yellow_taxi_sort_bench.sh.
# Usage: ./lineitem_sort_bench.sh <path/to/lineitem.csv> [output_dir]

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <path/to/lineitem.csv> [output_dir]" >&2
  exit 1
fi

INPUT_CSV=$1
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/lineitem_bench_${TS}"}
mkdir -p "$OUT_DIR"

# Fixed parameters derived from common TPC-H SF500 studies
RUN_GEN_THREADS=10
MERGE_THREADS=10

echo "Building lineitem_benchmark_cli example (release)..."
cargo build --release --example lineitem_benchmark_cli >/dev/null

# Cooldown between runs to keep SSD temperatures in check
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-60}
cooldown() {
  echo "Cooling down for ${COOLDOWN_SECONDS}s to let SSD rest..."
  sleep "$COOLDOWN_SECONDS"
}

run_case() {
  local name="$1"; shift
  local run_size_mb="$1"; shift
  local merge_fanin="$1"; shift
  # Any extra flags (e.g., --ovc) are passed through untouched
  local extra_flags=("$@")

  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  local ovc_label="no"
  for f in ${extra_flags[@]+"${extra_flags[@]}"}; do
    if [[ "$f" == "--ovc" ]]; then
      ovc_label="yes"
      break
    fi
  done

  echo "[$name]: Run Size: ${run_size_mb} MB, Run Gen Threads: ${RUN_GEN_THREADS}, Merge Threads: ${MERGE_THREADS}, OVC: ${ovc_label}" | tee -a "${OUT_DIR}/${name}.log"

  # Uses default key columns (8,9,13,14,15) and value columns (0,3) from lineitem_benchmark_cli.
  cargo run --release --example lineitem_benchmark_cli -- \
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
    ${extra_flags[@]+"${extra_flags[@]}"} 2>&1 | tee -a "${OUT_DIR}/${name}.log"
}

# Requested configurations (tuned for ~200 GB SF500 lineitem table)
# Baseline (no OVC)
# run_case "Log_0.0"       "12.74"   20000
# cooldown
# run_case "Log_0.25"      "38.14"   20000
# cooldown
# run_case "Log_0.5"       "114.20"  20000
# cooldown
# run_case "Log_0.75"      "341.97"  20000
# cooldown
# run_case "Log_1.0"       "1024.00" 20000
# cooldown
# 
# # With OVC enabled
run_case "Log_0.0_ovc"   "12.74"   20000 --ovc
cooldown
# run_case "Log_0.25_ovc"  "38.14"   20000 --ovc
# cooldown
# run_case "Log_0.5_ovc"   "114.20"  20000 --ovc
# cooldown
# run_case "Log_0.75_ovc"  "341.97"  20000 --ovc
# cooldown
# run_case "Log_1.0_ovc"   "1024.00" 20000 --ovc
# cooldown

echo "All runs completed. Logs at: ${OUT_DIR}"
