#!/usr/bin/env bash
set -euo pipefail

# Regime 2 Lineitem benchmark runner for comparing different thread allocation strategies.
# Usage: ./regime2_lineitem.sh <path/to/lineitem.csv> [output_dir]
#
# This script tests 4 different strategies for handling the regime 2 scenario:
# Strategy 1: Throttle Run Generation Threads (12.6 threads gen, 32 threads merge)
# Strategy 2: Throttle Merge Threads (32 threads gen, 12.6 threads merge)
# Strategy 3: Throttle Both Phases - Crossover Point (20.1 threads each)
# Strategy 4: Multiple Merge Operations - Full Parallelism (32 threads each, 3 passes)

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
OUT_DIR=${2:-"logs/lineitem_regime2_${TS}"}
mkdir -p "$OUT_DIR"

# Build once to avoid repeated compiles
echo "Building lineitem_benchmark_cli example (release)..."
cargo build --release --example lineitem_benchmark_cli >/dev/null

# Cooldown between runs to prevent SSD thermal throttling
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-60}
cooldown() {
  echo "Cooling down for ${COOLDOWN_SECONDS}s to let SSD rest..."
  sleep "$COOLDOWN_SECONDS"
}

run_case() {
  local name="$1"; shift
  local run_gen_threads="$1"; shift
  local merge_threads="$1"; shift
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

  echo "[$name]: Run Gen Threads: ${run_gen_threads}, Merge Threads: ${merge_threads}, Run Size: ${run_size_mb} MB, Merge Fan-in: ${merge_fanin}, OVC: ${ovc_label}" | tee -a "${OUT_DIR}/${name}.log"

  # Note: Using default key columns (8,9,13,14,15) and value columns (0,3) for TPC-H lineitem
  cargo run --release --example lineitem_benchmark_cli -- \
    -n "$name" \
    -i "$INPUT_CSV" \
    --run-gen-threads "$run_gen_threads" \
    --merge-threads "$merge_threads" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$merge_fanin" \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --cooldown-seconds 60 \
    --dir "$temp_dir" \
    "${extra_flags[@]}" 2>&1 | tee -a "${OUT_DIR}/${name}.log"
}

echo ""
echo "================================================================================"
echo "Strategy 1: Throttle Run Generation Threads"
echo "  Run Generation: 12.6 threads, RS = 81.05 MiB"
echo "  Merge: 32 threads, 512 runs, 1 pass"
echo "  Memory: Gen = 1.00 GiB, Merge = 1.00 GiB"
echo "================================================================================"
echo ""

# Strategy 1: Use 12 threads for run generation (rounding down 12.6), 32 for merge
# Run size = 81 MiB
run_case "Strategy1_baseline" 12 32 81 20000
cooldown
run_case "Strategy1_ovc" 12 32 81 20000 --ovc
cooldown

echo ""
echo "================================================================================"
echo "Strategy 2: Throttle Merge Threads"
echo "  Run Generation: 32 threads, RS = 32.00 MiB"
echo "  Merge: 12.6 threads, 1297 runs, 1 pass"
echo "  Memory: Gen = 1.00 GiB, Merge = 1.00 GiB"
echo "================================================================================"
echo ""

# Strategy 2: Use 32 threads for run generation, 12 threads for merge (rounding down 12.6)
# Run size = 32 MiB, merge fan-in calculated to give ~1297 runs in 1 pass
run_case "Strategy2_baseline" 32 12 32 20000
cooldown
run_case "Strategy2_ovc" 32 12 32 20000 --ovc
cooldown

echo ""
echo "================================================================================"
echo "Strategy 3: Throttle Both Phases (Crossover Point)"
echo "  Run Generation: 20.1 threads, RS = 50.93 MiB"
echo "  Merge: 20.1 threads, 815 runs, 1 pass"
echo "  Memory: Gen = 1.00 GiB, Merge = 1.00 GiB"
echo "================================================================================"
echo ""

# Strategy 3: Use 20 threads for both phases (rounding 20.1)
# Run size = 51 MiB, merge fan-in calculated to give ~815 runs in 1 pass
run_case "Strategy3_baseline" 20 20 51 20000
cooldown
run_case "Strategy3_ovc" 20 20 51 20000 --ovc
cooldown

echo ""
echo "================================================================================"
echo "Strategy 4: Multiple Merge Operations (Full Parallelism)"
echo "  Run Generation: 32 threads, RS = 32.00 MiB"
echo "  Merge: 32 threads, 1297 runs, 3 passes"
echo "  Memory: Gen = 1.00 GiB, Merge/op = 0.84 GiB"
echo "================================================================================"
echo ""

# Strategy 4: Use 32 threads for both phases, but allow multiple merge passes
# Run size = 32 MiB, merge fan-in kept at 512 to give 3 operations
# 1GiB / (32 threads * 64KiB) = 512-way merge, so 1297/512 = ~3 passes
run_case "Strategy4_baseline" 32 32 32 512
cooldown
run_case "Strategy4_ovc" 32 32 32 512 --ovc
cooldown

echo ""
echo "All runs completed. Logs at: ${OUT_DIR}"
