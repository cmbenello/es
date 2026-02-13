#!/usr/bin/env bash
set -euo pipefail

# Local KVBin sort test: generate 50 GiB freq_key / heavy_key / heavy_range
# datasets, then benchmark all 3 partition types (key-only, count-balanced,
# size-balanced).  Designed for a local machine with ~16 cores and ~60 GiB RAM.
#
# Usage: ./local_kvbin_test.sh [datasets_dir] [output_dir]

DATASETS_DIR=${1:-"datasets"}
TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/local_kvbin_test_${TS}"}
mkdir -p "$DATASETS_DIR" "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
THREADS=14
MEM_GB=2
PAGE_SIZE_KB=64
WARMUP_RUNS=0
BENCHMARK_RUNS=1
CLI_COOLDOWN_SECONDS=5
SLEEP_BETWEEN_CONFIGS=5

# 50 GiB row counts (derived from per-record sizes)
#   freq_key:    528 B/row  → 50×1024³ / 528  ≈ 101_680_097
#   heavy_key:   avg 16460 B/row → 50×1024³ / 16460 ≈ 3_260_878
#   heavy_range: avg 6672 B/row  → 50×1024³ / 6672  ≈ 8_046_626
FREQ_KEY_ROWS=101680097
HEAVY_KEY_ROWS=3260878
HEAVY_RANGE_ROWS=8046626

DATASETS=("freq_key" "heavy_key" "heavy_range")
PARTITIONS=("key-only" "count-balanced" "size-balanced")

# ---------------------------------------------------------
# BUILD
# ---------------------------------------------------------
echo "Building generators + benchmark CLI (release)..."
cargo build --release \
  --example gen_freq_key_kvbin \
  --example gen_heavy_key_kvbin \
  --example gen_heavy_range_kvbin \
  --example kvbin_benchmark_cli

GEN_FREQ=./target/release/examples/gen_freq_key_kvbin
GEN_HEAVY_KEY=./target/release/examples/gen_heavy_key_kvbin
GEN_HEAVY_RANGE=./target/release/examples/gen_heavy_range_kvbin
BENCH_BIN=./target/release/examples/kvbin_benchmark_cli

# ---------------------------------------------------------
# GENERATE DATASETS (skip if already present)
# ---------------------------------------------------------
generate_if_missing() {
  local name="$1"
  local kvbin="${DATASETS_DIR}/${name}.kvbin"
  local idx="${DATASETS_DIR}/${name}.kvbin.idx"

  if [[ -f "$kvbin" && -f "$idx" ]]; then
    echo "Dataset $name already exists, skipping generation."
    return 0
  fi

  echo "================================================================"
  echo "Generating: $name"
  echo "================================================================"

  case "$name" in
    freq_key)
      "$GEN_FREQ" --out "$kvbin" --idx "$idx" --rows "$FREQ_KEY_ROWS"
      ;;
    heavy_key)
      "$GEN_HEAVY_KEY" --out "$kvbin" --idx "$idx" --rows "$HEAVY_KEY_ROWS"
      ;;
    heavy_range)
      "$GEN_HEAVY_RANGE" --out "$kvbin" --idx "$idx" --rows "$HEAVY_RANGE_ROWS"
      ;;
    *)
      echo "Unknown dataset: $name" >&2
      exit 1
      ;;
  esac

  echo "Done generating $name."
  ls -lh "$kvbin" "$idx"
}

for d in "${DATASETS[@]}"; do
  generate_if_missing "$d"
done

# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------
run_bench() {
  local dataset="$1"
  local partition_type="$2"

  local kvbin="${DATASETS_DIR}/${dataset}.kvbin"
  local idx="${DATASETS_DIR}/${dataset}.kvbin.idx"

  # run_size_mb = (MemGB * 1024) / Threads
  local run_size_mb
  run_size_mb=$(echo "scale=2; ($MEM_GB * 1024) / $THREADS" | bc)

  # fanin = (mem * 0.95) / (threads * page_size) - 1
  local max_fanin
  max_fanin=$(echo "($MEM_GB * 1024 * 1024 * 95 / 100) / ($THREADS * $PAGE_SIZE_KB) - 1" | bc)

  local name="${dataset}_${partition_type}_T${THREADS}_Mem${MEM_GB}GB"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo "----------------------------------------------------------------"
  echo "BENCHMARK: $name"
  echo "  Dataset:         $kvbin"
  echo "  Index:           $idx"
  echo "  Mem Constraint:  ${MEM_GB} GB"
  echo "  Threads:         $THREADS"
  echo "  Buffer/Thread:   $run_size_mb MB"
  echo "  Max Fan-In:      $max_fanin"
  echo "  Partition Type:  $partition_type"
  echo "----------------------------------------------------------------"

  local log_file="${OUT_DIR}/${name}.log"

  local status="OK"
  if "$BENCH_BIN" \
    -n "$name" \
    -i "$kvbin" \
    --index "$idx" \
    --run-gen-threads "$THREADS" \
    --merge-threads "$THREADS" \
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

  rm -rf "$temp_dir"
}

RESULTS_FILE="${OUT_DIR}/summary.tsv"
echo -e "dataset\tpartition\tstatus\tlog" > "$RESULTS_FILE"

echo ""
echo "=== KVBin Local Test (Mem ${MEM_GB} GiB, Threads ${THREADS}) ==="
echo ""

for dataset in "${DATASETS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    run_bench "$dataset" "$partition"
    sleep "$SLEEP_BETWEEN_CONFIGS"
  done
done

echo ""
echo "================================================================"
echo "ALL DONE.  Summary:"
echo "================================================================"
cat "$RESULTS_FILE"
echo ""
echo "Logs saved to: $OUT_DIR"
