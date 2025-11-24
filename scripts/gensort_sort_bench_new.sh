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

echo "Building gen_sort_cli example (release)..."
cargo build --release --example gen_sort_cli >/dev/null

# Cooldown to prevent NVMe thermal throttling
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-30}
cooldown() {
  echo "Cooling down for ${COOLDOWN_SECONDS}s..."
  sleep "$COOLDOWN_SECONDS"
}

# ---------------------------------------------------------
# CORE RUN FUNCTION
# ---------------------------------------------------------
run_case() {
  local name="$1"
  local active_threads="$2"
  local run_size_mb="$3"
  local merge_fanin="$4"
  # Capture OVC flag if present
  local extra_flags="${5-}"

  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo "----------------------------------------------------------------"
  echo "BENCHMARK: $name"
  echo "  Threads:      $active_threads"
  echo "  Run Size:     $run_size_mb MB"
  echo "  Merge Fan-In: $merge_fanin"
  echo "  OVC:          ${extra_flags:-no}"
  echo "----------------------------------------------------------------"

  # Run the binary
  # Note: benchmark-runs set to 1 for long-running tests (100GB+)
  cargo run --release --example gen_sort_cli -- \
    -n "$name" \
    -i "$INPUT_DATA" \
    --run-gen-threads "$active_threads" \
    --merge-threads "$active_threads" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$merge_fanin" \
    --warmup-runs 0 \
    --benchmark-runs 1 \
    --cooldown-seconds 30 \
    --dir "$temp_dir" \
    $extra_flags 2>&1 | tee -a "${OUT_DIR}/${name}.log"

  rm -rf "$temp_dir"
}

# ==============================================================================
# EXPERIMENT 1: SCALABILITY TRAP (Fixed 2GB RAM)
# Goal: Show that increasing threads shrinks run size until we hit multi-pass.
# ==============================================================================
echo "=== STARTING EXPERIMENT 1: SCALABILITY (FIXED 2GB MEMORY) ==="

# 1. Safe Zone (Regime 1) - High Fan-In (20000) to force single pass
# 4 Threads -> 512MB Runs
run_case "Exp1_Thr04_Mem2GB" 4 "512.00" 20000 "--ovc"
cooldown

# 8 Threads -> 256MB Runs
run_case "Exp1_Thr08_Mem2GB" 8 "256.00" 20000 "--ovc"
cooldown

# 16 Threads -> 128MB Runs
run_case "Exp1_Thr16_Mem2GB" 16 "128.00" 20000 "--ovc"
cooldown

# 2. Danger Zone (Regime 2) - Realistic Fan-In (128) for Multi-Pass
# Note: At 2GB RAM, we cannot handle the 2000+ runs generated here in one pass.
# We switch to realistic fan-in (128) to measure the penalty of the multi-pass merge.

# 24 Threads -> ~85MB Runs (Borderline)
run_case "Exp1_Thr24_Mem2GB" 24 "85.33" 128 "--ovc"
cooldown

# 32 Threads -> 64MB Runs (Failure)
run_case "Exp1_Thr32_Mem2GB" 32 "64.00" 128 "--ovc"
cooldown

# 40 Threads -> 51MB Runs (Deep Failure)
run_case "Exp1_Thr40_Mem2GB" 40 "51.20" 128 "--ovc"
cooldown


# ==============================================================================
# EXPERIMENT 2: MEMORY CLIFF (Fixed 40 Threads)
# Goal: Show 40 threads is fine at high memory, but crashes/slows at low memory.
# ==============================================================================
echo "=== STARTING EXPERIMENT 2: MEMORY CLIFF (FIXED 40 THREADS) ==="

# 1. Safe Zone (Regime 1) - High Fan-In
# 8 GB -> ~205MB Runs
run_case "Exp2_Thr40_Mem8GB" 40 "204.80" 20000 "--ovc"
cooldown

# 6 GB -> ~154MB Runs
run_case "Exp2_Thr40_Mem6GB" 40 "153.60" 20000 "--ovc"
cooldown

# 4 GB -> ~102MB Runs
run_case "Exp2_Thr40_Mem4GB" 40 "102.40" 20000 "--ovc"
cooldown

# 2. Danger Zone (Regime 2) - Realistic Fan-In
# 2 GB -> 51MB Runs (Same as Exp1_Thr40, re-running for consistency or skip if desired)
run_case "Exp2_Thr40_Mem2GB" 40 "51.20" 128 "--ovc"
cooldown

# 1 GB -> 25MB Runs (Extreme Stress)
run_case "Exp2_Thr40_Mem1GB" 40 "25.60" 128 "--ovc"
cooldown

echo "All benchmarks completed. Results saved in ${OUT_DIR}"