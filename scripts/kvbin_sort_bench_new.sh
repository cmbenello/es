#!/usr/bin/env bash
set -euo pipefail

# Usage: ./kvbin_sort_bench_new.sh <datasets_dir> [output_dir]
#
# Expects these files in <datasets_dir>:
#   freq_key.kvbin        freq_key.kvbin.idx
#   heavy_key.kvbin       heavy_key.kvbin.idx
#   heavy_range.kvbin     heavy_range.kvbin.idx

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <datasets_dir> [output_dir]" >&2
  exit 1
fi

DATASETS_DIR=$1
if [[ ! -d "$DATASETS_DIR" ]]; then
  echo "Datasets dir not found: $DATASETS_DIR" >&2
  exit 1
fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/kvbin_bench_${TS}"}
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAGE_SIZE_KB=64
WARMUP_RUNS=0
BENCHMARK_RUNS=1
CLI_COOLDOWN_SECONDS=30
SLEEP_BETWEEN_CONFIGS_SECONDS=30
RUN_GEN_THREADS=44
MERGE_THREADS=44
MEM_GB=10
KEEP_TEMP_DIRS=false

DATASETS=("freq_key" "heavy_key" "heavy_range")
PARTITIONS=("key-only" "count-balanced" "size-balanced")

echo "Building kvbin_benchmark_cli example (release)..."
cargo build --release --example kvbin_benchmark_cli >/dev/null
BINARY=./target/release/examples/kvbin_benchmark_cli

cooldown() {
  sleep "$SLEEP_BETWEEN_CONFIGS_SECONDS"
}

clear_cache_if_available() {
  if [[ -x /usr/local/sbin/clearcache3.sh ]]; then
    if [[ $(id -u) -eq 0 ]]; then
      /usr/local/sbin/clearcache3.sh || echo "Warning: clearcache3.sh failed" >&2
    elif command -v sudo >/dev/null 2>&1; then
      sudo /usr/local/sbin/clearcache3.sh || echo "Warning: clearcache3.sh failed (sudo)" >&2
    else
      echo "Warning: clearcache3.sh found but sudo not available" >&2
    fi
  fi
}

require_dataset_files() {
  local base="$1"
  local kvbin="${DATASETS_DIR}/${base}.kvbin"
  local idx="${DATASETS_DIR}/${base}.kvbin.idx"
  if [[ ! -f "$kvbin" ]]; then
    echo "Missing dataset file: $kvbin" >&2
    exit 1
  fi
  if [[ ! -f "$idx" ]]; then
    echo "Missing index file: $idx" >&2
    exit 1
  fi
}

run_bench() {
  local dataset="$1"
  local partition_type="$2"

  local kvbin="${DATASETS_DIR}/${dataset}.kvbin"
  local idx="${DATASETS_DIR}/${dataset}.kvbin.idx"

  # run_size_mb = (MemGB * 1024) / RunGenThreads
  local run_size_mb
  run_size_mb=$(echo "scale=2; ($MEM_GB * 1024) / $RUN_GEN_THREADS" | bc)

  # 5% of memory reserved for sparse index; remaining 95% for I/O buffers.
  # Each thread needs fanin input buffers + 1 output buffer.
  # fanin = total * 0.95 / (threads * page_size) - 1
  local max_fanin
  max_fanin=$(echo "($MEM_GB * 1024 * 1024 * 95 / 100) / ($MERGE_THREADS * $PAGE_SIZE_KB) - 1" | bc)

  local name="${dataset}_${partition_type}_RunGen${RUN_GEN_THREADS}_Merge${MERGE_THREADS}_Mem${MEM_GB}GB"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo "----------------------------------------------------------------"
  echo "BENCHMARK: $name"
  echo "  Dataset:         $kvbin"
  echo "  Index:           $idx"
  echo "  Mem Constraint:  ${MEM_GB} GB"
  echo "  Run Gen Threads: $RUN_GEN_THREADS"
  echo "  Merge Threads:   $MERGE_THREADS"
  echo "  Buffer/Thread:   $run_size_mb MB"
  echo "  Max Fan-In:      $max_fanin"
  echo "  Partition Type:  $partition_type"
  echo "  Warmup Runs:     $WARMUP_RUNS"
  echo "  Bench Runs:      $BENCHMARK_RUNS"
  echo "  Cooldown (s):    $CLI_COOLDOWN_SECONDS"
  echo "  Temp Dir:        $temp_dir"
  echo "----------------------------------------------------------------"

  local log_file="${OUT_DIR}/${name}.log"

  local status="OK"
  if "$BINARY" \
    -n "$name" \
    -i "$kvbin" \
    --index "$idx" \
    --run-gen-threads "$RUN_GEN_THREADS" \
    --merge-threads "$MERGE_THREADS" \
    --run-size-mb "$run_size_mb" \
    --merge-fanin "$max_fanin" \
    --warmup-runs "$WARMUP_RUNS" \
    --benchmark-runs "$BENCHMARK_RUNS" \
    --cooldown-seconds "$CLI_COOLDOWN_SECONDS" \
    --partition-type "$partition_type" \
    --dir "$temp_dir" \
    2>&1 | tee -a "$log_file"; then
    status="OK"
  else
    status="FAIL"
  fi

  echo -e "${dataset}\t${partition_type}\t${status}\t${log_file}" | tee -a "$RESULTS_FILE"

  if [[ "$KEEP_TEMP_DIRS" != "true" ]]; then
    rm -rf "$temp_dir"
  fi
  clear_cache_if_available
}

# Validate inputs
for d in "${DATASETS[@]}"; do
  require_dataset_files "$d"
done

RESULTS_FILE="${OUT_DIR}/summary.tsv"
echo -e "dataset\tpartition\tstatus\tlog" > "$RESULTS_FILE"

echo "=== KVBin Partitioning Matrix (Mem ${MEM_GB} GiB, Threads ${RUN_GEN_THREADS}) ==="
for dataset in "${DATASETS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    run_bench "$dataset" "$partition"
    cooldown
  done
done

echo "Summary written to: $RESULTS_FILE"
