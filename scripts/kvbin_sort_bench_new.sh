#!/usr/bin/env bash
set -euo pipefail

# KVBin server benchmark: 3 datasets (freq_key, heavy_key, heavy_range) ×
# 3 partition types (key-only, count-balanced, size-balanced).
#
# Ported from local_kvbin_test.sh with 44 threads, 2 GiB memory, and the
# server harness (rclone upload, cache clearing) from lineitem/gensort scripts.
#
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

# ---------------------------------------------------------
# RCLONE CONFIG (upload logs to Google Drive after benchmark)
# Set RCLONE_REMOTE to your rclone remote:path, e.g. "gdrive:bench_results"
# Leave empty or unset to skip upload. Silently skipped if rclone not found.
# ---------------------------------------------------------
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:bench_results/kvbin}"

upload_logs() {
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found, skipping upload." >&2
    return 0
  fi
  if [[ -z "${RCLONE_REMOTE:-}" ]]; then
    echo "RCLONE_REMOTE not set, skipping upload." >&2
    return 0
  fi
  local dest="${RCLONE_REMOTE}/$(basename "$OUT_DIR")"
  echo "Uploading $OUT_DIR -> $dest ..."
  if rclone copy "$OUT_DIR" "$dest" --progress; then
    echo "Upload complete: $dest"
  else
    echo "Warning: rclone upload failed (exit $?)" >&2
  fi
}

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/kvbin_bench_${TS}"}
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
THREADS=44
MEM_GB=2
PAGE_SIZE_KB=64
WARMUP_RUNS=0
BENCHMARK_RUNS=1
CLI_COOLDOWN_SECONDS=30
SLEEP_BETWEEN_CONFIGS=30

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
  --example kvbin_benchmark_cli >/dev/null

GEN_FREQ=./target/release/examples/gen_freq_key_kvbin
GEN_HEAVY_KEY=./target/release/examples/gen_heavy_key_kvbin
GEN_HEAVY_RANGE=./target/release/examples/gen_heavy_range_kvbin
BINARY=./target/release/examples/kvbin_benchmark_cli

cooldown() {
  sleep "$SLEEP_BETWEEN_CONFIGS"
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

# ---------------------------------------------------------
# GENERATE DATASETS (skip if already present)
# ---------------------------------------------------------
generate_if_missing() {
  local name="$1"
  local kvbin="${DATASETS_DIR}/${name}.kvbin"
  local idx="${DATASETS_DIR}/${name}.kvbin.idx"

  if [[ -f "$kvbin" && -f "$idx" ]]; then
    local size
    size=$(du -sh "$kvbin" 2>/dev/null | cut -f1)
    echo "✓ Dataset $name already exists ($size), skipping generation."
    return 0
  fi

  local rows rec_desc
  case "$name" in
    freq_key)    rows=$FREQ_KEY_ROWS;    rec_desc="528 B/row (fixed)" ;;
    heavy_key)   rows=$HEAVY_KEY_ROWS;   rec_desc="~16,460 B/row (variable)" ;;
    heavy_range) rows=$HEAVY_RANGE_ROWS; rec_desc="~6,672 B/row (variable)" ;;
    *)
      echo "Unknown dataset: $name" >&2
      exit 1
      ;;
  esac

  echo ""
  echo "================================================================"
  echo "  Generating: $name"
  echo "  Target size:  ~50 GiB"
  echo "  Rows:         $(printf "%'d" "$rows")"
  echo "  Record size:  $rec_desc"
  echo "  Output:       $kvbin"
  echo "  Index:        $idx"
  echo "================================================================"

  local start_ts
  start_ts=$(date +%s)

  case "$name" in
    freq_key)
      "$GEN_FREQ" --out "$kvbin" --idx "$idx" --rows "$rows"
      ;;
    heavy_key)
      "$GEN_HEAVY_KEY" --out "$kvbin" --idx "$idx" --rows "$rows"
      ;;
    heavy_range)
      "$GEN_HEAVY_RANGE" --out "$kvbin" --idx "$idx" --rows "$rows"
      ;;
  esac

  local end_ts elapsed_s
  end_ts=$(date +%s)
  elapsed_s=$(( end_ts - start_ts ))
  local mins=$(( elapsed_s / 60 ))
  local secs=$(( elapsed_s % 60 ))

  echo ""
  echo "✓ Done generating $name in ${mins}m ${secs}s"
  ls -lh "$kvbin" "$idx"
  echo ""
}

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
  if "$BINARY" \
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
  upload_logs
  clear_cache_if_available
}

# Generate datasets if missing
for d in "${DATASETS[@]}"; do
  generate_if_missing "$d"
done

RESULTS_FILE="${OUT_DIR}/summary.tsv"
echo -e "dataset\tpartition\tstatus\tlog" > "$RESULTS_FILE"

echo ""
echo "=== KVBin Bench (Mem ${MEM_GB} GiB, Threads ${THREADS}) ==="
echo ""

for dataset in "${DATASETS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    run_bench "$dataset" "$partition"
    cooldown
  done
done

echo ""
echo "================================================================"
echo "ALL DONE.  Summary:"
echo "================================================================"
cat "$RESULTS_FILE"
echo ""
echo "Logs saved to: $OUT_DIR"
