#!/usr/bin/env bash
set -euo pipefail

# Usage: ./gensort_resource_bench.sh <path/to/gensort.data> [output_dir]

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
OUT_DIR=${2:-"logs/resource_bench_${TS}"}
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAGE_SIZE_KB=64

echo "Building gen_sort_cli example (release)..."
cargo build --release --example gen_sort_cli >/dev/null

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

  cargo run --release --example gen_sort_cli -- \
    -n "$name" \
    -i "$INPUT_DATA" \
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
  run_calculated_case "Exp1" "$t" "2" "--ovc"
  cooldown
done

# ==============================================================================
# EXPERIMENT 2: MEMORY CLIFF (Fixed 40 Threads)
# ==============================================================================
echo "=== EXP 2: MEMORY CLIFF (40 THREADS) ==="
# 8, 6, 4 should be Safe. 2, 1 Fail. 
# We skip 2 to avoid overlap with Exp 1. 
for m in 8 6 4 2 1; do
  # Skip overlap with Exp 1
  if [[ "$m" == "2" ]]; then continue; fi
  
  run_calculated_case "Exp2" "40" "$m" "--ovc"
  cooldown
done

echo "Done. Results in ${OUT_DIR}"
