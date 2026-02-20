#!/usr/bin/env bash
set -euo pipefail

# Usage: ./lineitem_sort_bench_new.sh <path/to/lineitem.csv> [output_dir]

if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <path/to/lineitem.csv> [output_dir]" >&2
  exit 1
fi

# ---------------------------------------------------------
# RCLONE CONFIG (upload logs to Google Drive after benchmark)
# Set RCLONE_REMOTE to your rclone remote:path, e.g. "gdrive:bench_results"
# Leave empty or unset to skip upload. Silently skipped if rclone not found.
# ---------------------------------------------------------
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:bench_results/lineitem}"

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

INPUT_CSV=$1
DATASET_NAME_SAFE=$(basename "$INPUT_CSV" | tr -cs 'a-zA-Z0-9_-' '_')

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/lineitem_bench_${TS}"}
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PAGE_SIZE_KB=64
WARMUP_RUNS=0
BENCHMARK_RUNS=3
CLI_COOLDOWN_SECONDS=30
SLEEP_BETWEEN_CONFIGS_SECONDS=30
# cgroup memory enforcement: "on" (apply limits, warn on failure), "strict" (abort on failure), "off" (disabled)
CGROUP_MODE="${CGROUP_MODE:-on}"

echo "Building lineitem_benchmark_cli example (release)..."
cargo build --release --example lineitem_benchmark_cli >/dev/null
BINARY=./target/release/examples/lineitem_benchmark_cli

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

cgroup_memory_limit_bytes() {
  local spec="$1"
  if [[ "$spec" =~ ^([0-9]+(\.[0-9]+)?)[[:space:]]*(GiB|GB|G)$ ]]; then
    local num="${BASH_REMATCH[1]}"
    printf '%d\n' "$(echo "scale=0; $num * 1073741824 / 1" | bc)"
  elif [[ "$spec" =~ ^([0-9]+(\.[0-9]+)?)[[:space:]]*(MiB|MB|M)$ ]]; then
    local num="${BASH_REMATCH[1]}"
    printf '%d\n' "$(echo "scale=0; $num * 1048576 / 1" | bc)"
  else
    echo "[cgroup] Cannot parse memory spec: '$spec'" >&2
    return 1
  fi
}

run_with_cgroup_limits() {
  local memory_limit="$1"
  local scope_name="$2"
  shift 2

  if [[ "$CGROUP_MODE" == "off" ]]; then
    "$@"
    return $?
  fi

  local limit_bytes
  if ! limit_bytes=$(cgroup_memory_limit_bytes "$memory_limit"); then
    echo "[cgroup] Warning: could not parse memory limit '$memory_limit'; running without cgroup." >&2
    if [[ "$CGROUP_MODE" == "strict" ]]; then
      return 1
    fi
    "$@"
    return $?
  fi

  if [[ "$(uname -s)" != "Linux" ]] || ! command -v systemd-run >/dev/null 2>&1; then
    echo "[cgroup] Warning: systemd-run unavailable; running without cgroup enforcement." >&2
    if [[ "$CGROUP_MODE" == "strict" ]]; then
      return 1
    fi
    "$@"
    return $?
  fi

  local unit_name="es-duck-${scope_name}-${DATASET_NAME_SAFE}-$$-$(date +%s)"
  echo "[cgroup] Applying limits for ${scope_name}: memory.high=${limit_bytes}, memory.max=${limit_bytes}, memory.swap.max=0"

  if systemd-run --user --scope --quiet --collect \
      --unit "${unit_name}" \
      --property "MemoryAccounting=yes" \
      --property "MemoryHigh=${limit_bytes}" \
      --property "MemoryMax=${limit_bytes}" \
      --property "MemorySwapMax=0" \
      -- "$@"; then
    return 0
  fi

  local status=$?
  echo "[cgroup] Run exited with status ${status} under cgroup limits (may be OOM kill or binary failure)." >&2
  return "$status"
}

run_bench() {
  local exp_prefix="$1"
  local run_gen_threads="$2"
  local merge_threads="$3"
  local mem_gb="$4"
  local extra_flags="${5-}"
  local track_mem="${6-false}"
  local cgroup_limit="${7-}"

  local discard_output="false"
  local ovc="true"
  local partition_type="size-balanced"
  local imbalance_factor="1.0"

  if [[ "$extra_flags" == *"--discard-final-output"* ]]; then
    discard_output="true"
  fi
  if [[ "$extra_flags" == *"--ovc=false"* ]]; then
    ovc="false"
  elif [[ "$extra_flags" == *"--ovc=true"* ]]; then
    ovc="true"
  fi
  if [[ "$extra_flags" =~ --partition-type[=\ ]([^[:space:]]+) ]]; then
    partition_type="${BASH_REMATCH[1]}"
  fi
  if [[ "$extra_flags" =~ --imbalance-factor[=\ ]([^[:space:]]+) ]]; then
    imbalance_factor="${BASH_REMATCH[1]}"
  fi

  # run_size_mb = (MemGB * 1024) / RunGenThreads
  local run_size_mb=$(echo "scale=2; ($mem_gb * 1024) / $run_gen_threads" | bc)

  # 5% of memory reserved for sparse index; remaining 95% for I/O buffers.
  # Each thread needs fanin input buffers + 1 output buffer.
  # fanin = total * 0.95 / (threads * page_size) - 1
  local max_fanin=$(echo "($mem_gb * 1024 * 1024 * 95 / 100) / ($merge_threads * $PAGE_SIZE_KB) - 1" | bc)

  local name="${exp_prefix}_RunGen${run_gen_threads}_Merge${merge_threads}_Mem${mem_gb}GB"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo "----------------------------------------------------------------"
  echo "BENCHMARK: $name"
  echo "  Mem Constraint:  ${mem_gb} GB"
  echo "  Run Gen Threads: $run_gen_threads"
  echo "  Merge Threads:   $merge_threads"
  echo "  Buffer/Thread:   $run_size_mb MB"
  echo "  Max Fan-In:      $max_fanin"
  echo "  Discard Output:  $discard_output"
  echo "  OVC:             $ovc"
  echo "  Partition Type:  $partition_type"
  echo "  Imbalance Factor:$imbalance_factor"
  echo "  Warmup Runs:     $WARMUP_RUNS"
  echo "  Bench Runs:      $BENCHMARK_RUNS"
  echo "  Cooldown (s):    $CLI_COOLDOWN_SECONDS"
  echo "  Temp Dir:        $temp_dir"
  if [[ -n "$extra_flags" ]]; then
    echo "  Extra Flags:     $extra_flags"
  fi
  echo "----------------------------------------------------------------"

  local log_file="${OUT_DIR}/${name}.log"

  if [[ "$track_mem" == "true" ]]; then
    local mem_log="${OUT_DIR}/${name}_mem.log"
    local pid_file="${temp_dir}/bin.pid"
    local start_ms
    start_ms=$(date +%s%3N)

    # Subshell writes its own PID before exec-ing so we can track RSS
    # while keeping stdout piped through tee.
    # Uses default key columns (8,9,13,14,15) and value columns (0,3) from lineitem_benchmark_cli.
    (
      echo $BASHPID > "$pid_file"
      exec "$BINARY" \
        -n "$name" \
        -i "$INPUT_CSV" \
        --run-gen-threads "$run_gen_threads" \
        --merge-threads "$merge_threads" \
        --run-size-mb "$run_size_mb" \
        --merge-fanin "$max_fanin" \
        --warmup-runs "$WARMUP_RUNS" \
        --benchmark-runs "$BENCHMARK_RUNS" \
        --cooldown-seconds "$CLI_COOLDOWN_SECONDS" \
        --partition-type "$partition_type" \
        --imbalance-factor "$imbalance_factor" \
        --dir "$temp_dir" \
        $extra_flags
    ) 2>&1 | tee -a "$log_file" &
    local pipe_bg=$!

    while [[ ! -s "$pid_file" ]]; do sleep 0.01; done
    local bin_pid
    bin_pid=$(cat "$pid_file")

    # Poll RSS every 1s; write "elapsed_ms rss_kb" to mem log
    (
      while kill -0 "$bin_pid" 2>/dev/null; do
        local rss elapsed_ms now_ms
        rss=$(ps -o rss= -p "$bin_pid" 2>/dev/null | tr -d ' ')
        now_ms=$(date +%s%3N)
        elapsed_ms=$(( now_ms - start_ms ))
        [[ -n "$rss" ]] && echo "$elapsed_ms $rss"
        sleep 1
      done
    ) > "$mem_log" &
    local poll_pid=$!

    wait "$pipe_bg"
    wait "$poll_pid" 2>/dev/null || true
  else
    # Uses default key columns (8,9,13,14,15) and value columns (0,3) from lineitem_benchmark_cli.
    local bin_cmd=("$BINARY"
      -n "$name"
      -i "$INPUT_CSV"
      --run-gen-threads "$run_gen_threads"
      --merge-threads "$merge_threads"
      --run-size-mb "$run_size_mb"
      --merge-fanin "$max_fanin"
      --warmup-runs "$WARMUP_RUNS"
      --benchmark-runs "$BENCHMARK_RUNS"
      --cooldown-seconds "$CLI_COOLDOWN_SECONDS"
      --partition-type "$partition_type"
      --imbalance-factor "$imbalance_factor"
      --dir "$temp_dir"
    )
    # shellcheck disable=SC2206
    [[ -n "$extra_flags" ]] && bin_cmd+=($extra_flags)

    if [[ -n "$cgroup_limit" ]]; then
      run_with_cgroup_limits "$cgroup_limit" "$name" "${bin_cmd[@]}" 2>&1 | tee -a "$log_file"
    else
      "${bin_cmd[@]}" 2>&1 | tee -a "$log_file"
    fi
  fi

  rm -rf "$temp_dir"
  upload_logs
  clear_cache_if_available
}

# ==============================================================================
# EXPERIMENT 0: SINGLE RUN WITH MEMORY TRACKING (2GB RAM, 44 Threads)
# ==============================================================================
echo "=== EXP 0: MEMORY TRACKING (2GB RAM, 44 THREADS) ==="
SAVED_BENCHMARK_RUNS=$BENCHMARK_RUNS
BENCHMARK_RUNS=1
run_bench "Exp0" "44" "44" "2" "" "true"
BENCHMARK_RUNS=$SAVED_BENCHMARK_RUNS
cooldown

# ==============================================================================
# EXPERIMENT 1: CGROUP MEMORY SWEEP (Fixed 16 Threads, Memory = 80% of cgroup)
# ==============================================================================
echo "=== EXP 1: CGROUP MEMORY SWEEP (16 THREADS, Memory = 80% of cgroup) ==="
for cgroup in 48 32 24 16 8 4; do
  mem=$(echo "scale=1; $cgroup * 0.8" | bc)
  run_bench "Exp1" "16" "16" "$mem" "--discard-final-output" "false" "${cgroup}GiB"
  cooldown
done

# ==============================================================================
# EXPERIMENT 2: MEMORY CLIFF (Fixed 44 Threads)
# ==============================================================================
echo "=== EXP 2: MEMORY CLIFF (44 THREADS) ==="
for m in 32 24 16 8 6 4 2 1; do
  run_bench "Exp2" "44" "44" "$m"
  cooldown
done

# ==============================================================================
# EXPERIMENT 3: OVC VS NO-OVC (2GB RAM)
# ==============================================================================
echo "=== EXP 3: NO-OVC (Scalability, 2GB RAM) ==="
for t in 4 8 16 24 32 40 44; do
  run_bench "Exp3" "$t" "$t" "2" "--ovc=false"
  cooldown
done

# ==============================================================================
# EXPERIMENT 6: THREAD CONFIGURATION GRID SEARCH (RunGen × Merge)
# ==============================================================================
# Purpose: Systematic exploration of thread configuration space to identify
#          optimal (RunGen, Merge) combinations and phase interactions
# ==============================================================================
echo "=== EXP 6: THREAD CONFIGURATION GRID (RunGen × Merge, 2GB RAM) ==="

RUNGEN_GRID=(4 8 16 24 32 40 44)
MERGE_GRID=(4 8 16 24 32 40 44)

config_count=0
total_grid_configs=$((${#RUNGEN_GRID[@]} * ${#MERGE_GRID[@]}))

echo "Grid dimensions: ${#RUNGEN_GRID[@]} × ${#MERGE_GRID[@]} = $total_grid_configs configs"

SAVED_BENCHMARK_RUNS=$BENCHMARK_RUNS
BENCHMARK_RUNS=1
for rg in "${RUNGEN_GRID[@]}"; do
  for mg in "${MERGE_GRID[@]}"; do
    config_count=$((config_count + 1))
    echo ">>> Grid Progress: $config_count / $total_grid_configs (asymmetric only) <<<"

    run_bench "Exp6" "$rg" "$mg" "2"
    cooldown
  done
done
BENCHMARK_RUNS=$SAVED_BENCHMARK_RUNS

echo "All benchmarks complete. Results in: $OUT_DIR"
