#!/usr/bin/env bash
set -euo pipefail

# Usage: ./lineitem_sort_bench_new.sh <path/to/lineitem.csv> [output_dir]

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <path/to/lineitem.csv> [output_dir]" >&2
  exit 1
fi

INPUT_CSV=$1
# if [[ ! -f "$INPUT_CSV" ]]; then
#   echo "Input CSV not found: $INPUT_CSV" >&2
#   exit 1
# fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/lineitem_bench_${TS}"}
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAGE_SIZE_KB=64

echo "Building lineitem_benchmark_cli example (release)..."
cargo build --release --example lineitem_benchmark_cli >/dev/null

cooldown() {
  sleep 30
}

run_calculated_case() {
  local exp_prefix="$1"
  local active_threads="$2"
  local mem_gb="$3"
  local extra_flags="${4-}"

  # 1. Calculate Buffer Size (MB) per thread
  # run_size_mb = (MemGB * 1024) / Threads
  local run_size_mb=$(echo "scale=2; ($mem_gb * 1024) / $active_threads" | bc)

  # 2. Calculate Max Feasible Fan-In (Physical Limit)
  # FanIn <= (MemGB * 1024^2) / (Threads * PageKB)
  local max_fanin=$(echo "($mem_gb * 1024 * 1024) / ($active_threads * $PAGE_SIZE_KB)" | bc)

  local name="${exp_prefix}_Thr${active_threads}_Mem${mem_gb}GB"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo "----------------------------------------------------------------"
  echo "BENCHMARK: $name"
  echo "  Mem Constraint: ${mem_gb} GB"
  echo "  Threads:        $active_threads"
  echo "  Buffer/Thread:  $run_size_mb MB"
  echo "  Max Fan-In:     $max_fanin"
  echo "----------------------------------------------------------------"

  # Uses default key columns (8,9,13,14,15) and value columns (0,3) from lineitem_benchmark_cli.
  cargo run --release --example lineitem_benchmark_cli -- \
    -n "$name" \
    -i "$INPUT_CSV" \
    --run-gen-threads "$active_threads" \
    --merge-threads "$active_threads" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$max_fanin" \
    --warmup-runs 1 \
    --benchmark-runs 3 \
    --cooldown-seconds 30 \
    --dir "$temp_dir" \
    $extra_flags 2>&1 | tee -a "${OUT_DIR}/${name}.log"

  rm -rf "$temp_dir"
}

# ==============================================================================
# EXPERIMENT 1: SCALABILITY TRAP (Fixed 2GB RAM)
# ==============================================================================
echo "=== EXP 1: SCALABILITY (2GB RAM) ==="
# 4, 8, 16 should be Safe. 24 Borderline. 32, 40 Fail.
for t in 4 8 16 24 32 40 44; do
  run_calculated_case "Exp1" "$t" "2"
  cooldown
done

# ==============================================================================
# EXPERIMENT 2: MEMORY CLIFF (Fixed 40 Threads)
# ==============================================================================
echo "=== EXP 2: MEMORY CLIFF (40 THREADS) ==="
# 8, 6, 4 should be Safe. 2, 1 Fail.
# We skip 2 to avoid overlap with Exp 1.
for m in 32 24 16 8 6 4 2 1; do
  # Skip overlap with Exp 1
  if [[ "$m" == "2" ]]; then continue; fi

  run_calculated_case "Exp2" "40" "$m"
  cooldown
done


# ==============================================================================
# EXPERIMENT 3: OVC VS NO-OVC (40 Threads)
# ==============================================================================
echo "=== EXP 3: NO-OVC (40 THREADS) ==="
for m in 32 24 16 8 6 4 2 1; do
  run_calculated_case "Exp3" "40" "$m" "--ovc=false"
  cooldown
done


# ==============================================================================
# EXPERIMENT 4: Reservoir Sampling vs KLL (Fixed 2GB RAM)
# ==============================================================================
echo "=== EXP 4: SKETCHING IMPACT (2GB RAM) ==="
for t in 4 8 16 24 32 40 44; do
  run_calculated_case "Exp4" "$t" "2" "--sketch-type reservoir-sampling"
  cooldown
done

# ==============================================================================
# EXPERIMENT 5: Imbalance Impact (Fixed 2GB RAM, 24 Threads)
# ==============================================================================
echo "=== EXP 5: IMBALANCE FACTOR IMPACT (24 THREADS, 2GB RAM) ==="
for i in 1.0 1.5 2.0 3.0 4.0; do
  run_calculated_case "Exp5_Imbalance${i}" "24" "2" "--imbalance-factor ${i}"
  cooldown
done


echo "Done. Results in ${OUT_DIR}"
