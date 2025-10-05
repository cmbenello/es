#!/usr/bin/env bash

set -euo pipefail

# Compare sorting performance with DuckDB 1.3 vs 1.4 (pure DuckDB)
#
# Usage:
#   ./compare_sort_duckdb_versions.sh <csv_file> [threads] [memory] [options]
#   ./compare_sort_duckdb_versions.sh <csv_file> [options]
#
# Positional (backwards compatible):
#   threads               Number of threads (default: 4)
#   memory                Memory limit (default: 4GB)
#
# Options:
#   -t, --threads <n>     Number of threads
#   -m, --memory <mem>    Memory limit, e.g. 4GB, 16GB
#       --bin-dir <dir>   Directory with duckdb-<ver> binaries (default: current dir)
#       --version-13 <v>  DuckDB 1.3.x version (default: 1.3.2)
#       --version-14 <v>  DuckDB 1.4.x version (default: 1.4.0)
#   -w, --warmup <n>      Number of warmup runs per version (default: 0)
#   -r, --runs <n>        Number of measured runs per version (default: 1)
#       --db-path <path>  Use on-disk database file instead of :memory:
#       --skip-13         Skip DuckDB 1.3 version testing
#   -h, --help            Show this help and exit
#
# Notes:
# - Expects DuckDB CLIs installed as duckdb-<version> in current dir by default
#   (use scripts/duckdb_setup.sh or pass --bin-dir)

usage() {
  cat <<USAGE
Usage: $0 <csv_file> [threads] [memory] [options]

Positional (optional):
  threads               Number of threads (default: 4)
  memory                Memory limit (default: 4GB)

Options:
  -t, --threads <n>     Number of threads
  -m, --memory <mem>    Memory limit, e.g. 4GB, 16GB
      --bin-dir <dir>   Directory with duckdb-<ver> binaries (default: current dir)
      --version-13 <v>  DuckDB 1.3.x version (default: 1.3.2)
      --version-14 <v>  DuckDB 1.4.x version (default: 1.4.0)
  -w, --warmup <n>      Number of warmup runs per version (default: 0)
  -r, --runs <n>        Number of measured runs per version (default: 1)
      --db-path <path>  Use on-disk database file instead of :memory:
      --skip-13         Skip DuckDB 1.3 version testing
  -h, --help            Show this help and exit

Example:
  $0 lineitem_sf1.csv 4 4GB --bin-dir . --version-13 1.3.2 --version-14 1.4.0 --warmup 1 --runs 3
  $0 lineitem_sf10.csv --runs 3 --threads 8 --memory 16GB
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage; exit 0
fi

if [ $# -lt 1 ]; then
    usage; exit 1
fi

CSV_FILE="$1"; shift || true

# Defaults
THREADS=4
MEMORY=4GB

# Backward-compatible optional positionals for threads and memory
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  THREADS="$1"; shift
fi
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  MEMORY="$1"; shift
fi

BIN_DIR="${PWD}"
VER13="1.3.2"
VER14="1.4.0"
WARMUP_RUNS=0
MEASURE_RUNS=1
DB_PATH=""
SKIP_13=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    -t|--threads)
      THREADS="$2"; shift 2 ;;
    --threads=*)
      THREADS="${1#*=}"; shift ;;
    -m|--memory|--mem)
      MEMORY="$2"; shift 2 ;;
    --memory=*|--mem=*)
      MEMORY="${1#*=}"; shift ;;
    --db-path)
      DB_PATH="$2"; shift 2 ;;
    --db-path=*)
      DB_PATH="${1#*=}"; shift ;;
    --bin-dir)
      BIN_DIR="$2"; shift 2 ;;
    --bin-dir=*)
      BIN_DIR="${1#*=}"; shift ;;
    --version-13)
      VER13="$2"; shift 2 ;;
    --version-13=*)
      VER13="${1#*=}"; shift ;;
    --version-14)
      VER14="$2"; shift 2 ;;
    --version-14=*)
      VER14="${1#*=}"; shift ;;
    -w|--warmup)
      WARMUP_RUNS="$2"; shift 2 ;;
    --warmup=*)
      WARMUP_RUNS="${1#*=}"; shift ;;
    -r|--runs)
      MEASURE_RUNS="$2"; shift 2 ;;
    --runs=*)
      MEASURE_RUNS="${1#*=}"; shift ;;
    --skip-13)
      SKIP_13=true; shift ;;
    --)
      shift; break ;;
    -*)
      echo "Unknown option: $1"; echo; usage; exit 1 ;;
    *)
      # Ignore stray positionals (already handled for threads/memory)
      shift ;;
  esac
done

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File '$CSV_FILE' not found"; exit 1
fi

DUCKDB13_BIN="${BIN_DIR}/duckdb-${VER13}"
DUCKDB14_BIN="${BIN_DIR}/duckdb-${VER14}"

if [ "$SKIP_13" = false ]; then
  if [ ! -x "$DUCKDB13_BIN" ]; then
    echo "Error: DuckDB 1.3 binary not found at: $DUCKDB13_BIN"; echo "Run scripts/duckdb_setup.sh or pass --bin-dir"; exit 1
  fi
fi
if [ ! -x "$DUCKDB14_BIN" ]; then
  echo "Error: DuckDB 1.4 binary not found at: $DUCKDB14_BIN"; echo "Run scripts/duckdb_setup.sh or pass --bin-dir"; exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="comparison_duckdb_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/comparison_results.txt"

# Set default DB paths - separate database for each version
DB_PATH_13="${OUTPUT_DIR}/benchmark_13.duckdb"
DB_PATH_14="${OUTPUT_DIR}/benchmark_14.duckdb"

# Handle user-provided --db-path
if [ -n "$DB_PATH" ] && [ "$DB_PATH" != ":memory:" ]; then
  # If user provided a path, use it as base and add version suffix
  DB_DIR=$(dirname "$DB_PATH")
  DB_BASE=$(basename "$DB_PATH" .duckdb)
  DB_PATH_13="${DB_DIR}/${DB_BASE}_13.duckdb"
  DB_PATH_14="${DB_DIR}/${DB_BASE}_14.duckdb"
  mkdir -p "$DB_DIR"
elif [ "$DB_PATH" = ":memory:" ]; then
  DB_PATH_13=":memory:"
  DB_PATH_14=":memory:"
fi

# Basic validations
if ! [[ "$WARMUP_RUNS" =~ ^[0-9]+$ ]]; then echo "Error: --warmup must be a non-negative integer"; exit 1; fi
if ! [[ "$MEASURE_RUNS" =~ ^[0-9]+$ ]]; then echo "Error: --runs must be a non-negative integer"; exit 1; fi
if [ "$MEASURE_RUNS" -lt 1 ]; then echo "Error: --runs must be at least 1"; exit 1; fi
if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then echo "Error: threads must be an integer"; exit 1; fi

# File size
FILE_SIZE_BYTES=$(stat -c%s "$CSV_FILE" 2>/dev/null || stat -f%z "$CSV_FILE" 2>/dev/null)
FILE_SIZE_GB=$(echo "scale=2; $FILE_SIZE_BYTES / 1024 / 1024 / 1024" | bc)

echo "======================================================================"  | tee "$LOG_FILE"
echo "DUCKDB COMPARISON (1.3 vs 1.4)"                                             | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "Date: $(date)"                                                           | tee -a "$LOG_FILE"
echo "File: $CSV_FILE"                                                         | tee -a "$LOG_FILE"
echo "File size: ${FILE_SIZE_GB} GB"                                          | tee -a "$LOG_FILE"
echo "Threads: $THREADS"                                                       | tee -a "$LOG_FILE"
echo "Memory: $MEMORY"                                                         | tee -a "$LOG_FILE"
echo "DuckDB 1.3 bin: $DUCKDB13_BIN"                                          | tee -a "$LOG_FILE"
echo "DuckDB 1.4 bin: $DUCKDB14_BIN"                                          | tee -a "$LOG_FILE"
echo "Database 1.3: ${DB_PATH_13}"                                            | tee -a "$LOG_FILE"
echo "Database 1.4: ${DB_PATH_14}"                                            | tee -a "$LOG_FILE"
echo "Warmup runs: $WARMUP_RUNS"                                              | tee -a "$LOG_FILE"
echo "Measured runs: $MEASURE_RUNS"                                           | tee -a "$LOG_FILE"
echo ""                                                                        | tee -a "$LOG_FILE"

DELIMITER=","  # CSV

# Helper to build SQL file for DuckDB run
build_duckdb_sql() {
  local sql_path="$1"
  cat > "$sql_path" << 'EOF'
SET threads=${THREADS};
SET memory_limit='${MEMORY}';
.timer on

EXPLAIN
SELECT l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, l_comment, l_orderkey, l_linenumber
FROM read_csv_auto('${CSV_FILE}', header=true, sample_size=1000)
ORDER BY l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, l_comment;

CREATE TABLE __sorted_result AS
SELECT l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, l_comment, l_orderkey, l_linenumber
FROM read_csv_auto('${CSV_FILE}', header=true, sample_size=1000)
ORDER BY l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, l_comment;
EOF
  # Replace variables
  sed -i "s|\${THREADS}|${THREADS}|g; s|\${MEMORY}|${MEMORY}|g; s|\${CSV_FILE}|${CSV_FILE}|g" "$sql_path"
}

# Test 1: DuckDB 1.3
if [ "$SKIP_13" = false ]; then
  echo "======================================================================"  | tee -a "$LOG_FILE"
  echo "TEST 1: DuckDB ${VER13}"                                                | tee -a "$LOG_FILE"
  echo "======================================================================"  | tee -a "$LOG_FILE"
  DUCKDB_SQL_13="${OUTPUT_DIR}/duckdb_sort_13.sql"

  build_duckdb_sql "$DUCKDB_SQL_13"

  DUCKDB_13_LOG="${OUTPUT_DIR}/duckdb_13_output.log"
  : > "$DUCKDB_13_LOG"
  DUCKDB_13_CRASHED=false

  # Warmups 1.3
  if [ "$WARMUP_RUNS" -gt 0 ]; then
    echo "-- Warmup runs (${WARMUP_RUNS})"                                       | tee -a "$LOG_FILE"
    for i in $(seq 1 "$WARMUP_RUNS"); do
      echo "Warmup #$i (v${VER13})..."                                           | tee -a "$LOG_FILE"
      # Clean database before each warmup
      rm -f "$DB_PATH_13" "${DB_PATH_13}.wal"
      rm -rf "${DB_PATH_13}.tmp"
      sync
      set +e
      "$DUCKDB13_BIN" "$DB_PATH_13" < "$DUCKDB_SQL_13" 2>&1 | tee -a "$DUCKDB_13_LOG" >/dev/null
      cmd_status=${PIPESTATUS[0]}
      set -e
      if [ $cmd_status -ne 0 ]; then
        # Decode common signals
        if [ $cmd_status -ge 128 ]; then
          sig=$((cmd_status-128))
          case $sig in
            11) reason="Segmentation fault (signal 11)" ;;
            9)  reason="Killed (signal 9)" ;;
            *)  reason="Terminated by signal ${sig}" ;;
          esac
        else
          reason="Exited with status ${cmd_status}"
        fi
        echo "DuckDB ${VER13} crashed during warmup: ${reason}. Skipping remaining runs for this version." | tee -a "$LOG_FILE"
        DUCKDB_13_CRASHED=true
        break
      fi
    done
  fi

  # Measured runs 1.3
  if [ "$DUCKDB_13_CRASHED" = false ]; then
    echo "-- Measured runs (${MEASURE_RUNS})"                                     | tee -a "$LOG_FILE"
    DUCKDB_13_SUM=0
    DUCKDB_13_MIN=
    DUCKDB_13_MAX=
    DUCKDB_13_COMPLETED=0
    for i in $(seq 1 "$MEASURE_RUNS"); do
      # Clean database files and sync disk before each run
      rm -f "$DB_PATH_13" "${DB_PATH_13}.wal"
      rm -rf "${DB_PATH_13}.tmp"
      sync
      sleep 1

      RUN_LOG="${OUTPUT_DIR}/duckdb_13_run${i}.log"
      set +e
      "$DUCKDB13_BIN" "$DB_PATH_13" < "$DUCKDB_SQL_13" 2>&1 | tee "$RUN_LOG" >> "$DUCKDB_13_LOG"
      cmd_status=${PIPESTATUS[0]}
      set -e
      if [ $cmd_status -ne 0 ]; then
        if [ $cmd_status -ge 128 ]; then
          sig=$((cmd_status-128))
          case $sig in
            11) reason="Segmentation fault (signal 11)" ;;
            9)  reason="Killed (signal 9)" ;;
            *)  reason="Terminated by signal ${sig}" ;;
          esac
        else
          reason="Exited with status ${cmd_status}"
        fi
        echo "DuckDB ${VER13} crashed during measured run #$i: ${reason}. Skipping remaining runs for this version." | tee -a "$LOG_FILE"
        DUCKDB_13_CRASHED=true
        break
      fi

      # Extract CREATE TABLE time only (skip EXPLAIN)
      # Line 1: EXPLAIN, Line 2: CREATE TABLE
      CREATE_TIME=$(grep "^Run Time" "$RUN_LOG" | sed -n '2p' | grep -oP 'real \K[0-9.]+' || echo "0")
      DUCKDB_13_DURATION="$CREATE_TIME"

      echo "Run #$i time (v${VER13}): $(printf '%.2f' "$DUCKDB_13_DURATION") s (CREATE: ${CREATE_TIME}s)" | tee -a "$LOG_FILE"
      DUCKDB_13_SUM=$(echo "$DUCKDB_13_SUM + $DUCKDB_13_DURATION" | bc)
      if [ -z "$DUCKDB_13_MIN" ] || (( $(echo "$DUCKDB_13_DURATION < $DUCKDB_13_MIN" | bc -l) )); then DUCKDB_13_MIN="$DUCKDB_13_DURATION"; fi
      if [ -z "$DUCKDB_13_MAX" ] || (( $(echo "$DUCKDB_13_DURATION > $DUCKDB_13_MAX" | bc -l) )); then DUCKDB_13_MAX="$DUCKDB_13_DURATION"; fi
      DUCKDB_13_COMPLETED=$((DUCKDB_13_COMPLETED + 1))
    done
    if [ "${DUCKDB_13_COMPLETED:-0}" -gt 0 ]; then
      DUCKDB_13_AVG=$(echo "scale=6; $DUCKDB_13_SUM / $DUCKDB_13_COMPLETED" | bc)
      echo "DuckDB ${VER13} avg time: $(printf '%.2f' "$DUCKDB_13_AVG") seconds"   | tee -a "$LOG_FILE"
    else
      DUCKDB_13_AVG=0
      echo "DuckDB ${VER13} had no successful measured runs."                     | tee -a "$LOG_FILE"
    fi
    echo ""                                                                        | tee -a "$LOG_FILE"
  else
    DUCKDB_13_AVG=0
  fi

  # Clean up 1.3 database and sync disk before next experiment
  echo "Cleaning up and syncing disk..."                                        | tee -a "$LOG_FILE"
  rm -f "$DB_PATH_13" "${DB_PATH_13}.wal"
  rm -rf "${DB_PATH_13}.tmp"
  sync
  sleep 2
  echo ""                                                                        | tee -a "$LOG_FILE"
else
  echo "======================================================================"  | tee -a "$LOG_FILE"
  echo "TEST 1: DuckDB ${VER13} - SKIPPED"                                      | tee -a "$LOG_FILE"
  echo "======================================================================"  | tee -a "$LOG_FILE"
  echo ""                                                                        | tee -a "$LOG_FILE"
  DUCKDB_13_AVG=0
fi

# Test 2: DuckDB 1.4
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "TEST 2: DuckDB ${VER14}"                                                | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
DUCKDB_SQL_14="${OUTPUT_DIR}/duckdb_sort_14.sql"
build_duckdb_sql "$DUCKDB_SQL_14"

DUCKDB_14_LOG="${OUTPUT_DIR}/duckdb_14_output.log"
: > "$DUCKDB_14_LOG"
DUCKDB_14_CRASHED=false

# Warmups 1.4
if [ "$WARMUP_RUNS" -gt 0 ]; then
  echo "-- Warmup runs (${WARMUP_RUNS})"                                       | tee -a "$LOG_FILE"
  for i in $(seq 1 "$WARMUP_RUNS"); do
    echo "Warmup #$i (v${VER14})..."                                           | tee -a "$LOG_FILE"
    # Clean database before each warmup
    rm -f "$DB_PATH_14" "${DB_PATH_14}.wal"
    rm -rf "${DB_PATH_14}.tmp"
    sync
    set +e
    "$DUCKDB14_BIN" "$DB_PATH_14" < "$DUCKDB_SQL_14" 2>&1 | tee -a "$DUCKDB_14_LOG" >/dev/null
    cmd_status=${PIPESTATUS[0]}
    set -e
    if [ $cmd_status -ne 0 ]; then
      if [ $cmd_status -ge 128 ]; then
        sig=$((cmd_status-128))
        case $sig in
          11) reason="Segmentation fault (signal 11)" ;;
          9)  reason="Killed (signal 9)" ;;
          *)  reason="Terminated by signal ${sig}" ;;
        esac
      else
        reason="Exited with status ${cmd_status}"
      fi
      echo "DuckDB ${VER14} crashed during warmup: ${reason}. Skipping remaining runs for this version." | tee -a "$LOG_FILE"
      DUCKDB_14_CRASHED=true
      break
    fi
  done
fi

# Measured runs 1.4
if [ "$DUCKDB_14_CRASHED" = false ]; then
  echo "-- Measured runs (${MEASURE_RUNS})"                                     | tee -a "$LOG_FILE"
  DUCKDB_14_SUM=0
  DUCKDB_14_MIN=
  DUCKDB_14_MAX=
  DUCKDB_14_COMPLETED=0
  for i in $(seq 1 "$MEASURE_RUNS"); do
    # Clean database files and sync disk before each run
    rm -f "$DB_PATH_14" "${DB_PATH_14}.wal"
    rm -rf "${DB_PATH_14}.tmp"
    sync
    sleep 1

    RUN_LOG="${OUTPUT_DIR}/duckdb_14_run${i}.log"
    set +e
    "$DUCKDB14_BIN" "$DB_PATH_14" < "$DUCKDB_SQL_14" 2>&1 | tee "$RUN_LOG" >> "$DUCKDB_14_LOG"
    cmd_status=${PIPESTATUS[0]}
    set -e
    if [ $cmd_status -ne 0 ]; then
      if [ $cmd_status -ge 128 ]; then
        sig=$((cmd_status-128))
        case $sig in
          11) reason="Segmentation fault (signal 11)" ;;
          9)  reason="Killed (signal 9)" ;;
          *)  reason="Terminated by signal ${sig}" ;;
        esac
      else
        reason="Exited with status ${cmd_status}"
      fi
      echo "DuckDB ${VER14} crashed during measured run #$i: ${reason}. Skipping remaining runs for this version." | tee -a "$LOG_FILE"
      DUCKDB_14_CRASHED=true
      break
    fi

    # Extract CREATE TABLE time only (skip EXPLAIN)
    # Line 1: EXPLAIN, Line 2: CREATE TABLE
    CREATE_TIME=$(grep "^Run Time" "$RUN_LOG" | sed -n '2p' | grep -oP 'real \K[0-9.]+' || echo "0")
    DUCKDB_14_DURATION="$CREATE_TIME"

    echo "Run #$i time (v${VER14}): $(printf '%.2f' "$DUCKDB_14_DURATION") s (CREATE: ${CREATE_TIME}s)" | tee -a "$LOG_FILE"
    DUCKDB_14_SUM=$(echo "$DUCKDB_14_SUM + $DUCKDB_14_DURATION" | bc)
    if [ -z "$DUCKDB_14_MIN" ] || (( $(echo "$DUCKDB_14_DURATION < $DUCKDB_14_MIN" | bc -l) )); then DUCKDB_14_MIN="$DUCKDB_14_DURATION"; fi
    if [ -z "$DUCKDB_14_MAX" ] || (( $(echo "$DUCKDB_14_DURATION > $DUCKDB_14_MAX" | bc -l) )); then DUCKDB_14_MAX="$DUCKDB_14_DURATION"; fi
    DUCKDB_14_COMPLETED=$((DUCKDB_14_COMPLETED + 1))
  done
  if [ "${DUCKDB_14_COMPLETED:-0}" -gt 0 ]; then
    DUCKDB_14_AVG=$(echo "scale=6; $DUCKDB_14_SUM / $DUCKDB_14_COMPLETED" | bc)
    echo "DuckDB ${VER14} avg time: $(printf '%.2f' "$DUCKDB_14_AVG") seconds"   | tee -a "$LOG_FILE"
  else
    DUCKDB_14_AVG=0
    echo "DuckDB ${VER14} had no successful measured runs."                     | tee -a "$LOG_FILE"
  fi
  echo ""                                                                        | tee -a "$LOG_FILE"
else
  DUCKDB_14_AVG=0
fi

# Summary
echo ""                                                                        | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "SUMMARY"                                                                | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
if [ "$SKIP_13" = false ]; then
  if [ "${DUCKDB_13_COMPLETED:-0}" -gt 0 ]; then
    echo "DuckDB ${VER13} avg time (${DUCKDB_13_COMPLETED}/${MEASURE_RUNS} runs): $(printf "%.2f" "$DUCKDB_13_AVG") seconds" | tee -a "$LOG_FILE"
  elif [ "${DUCKDB_13_CRASHED:-false}" = true ]; then
    echo "DuckDB ${VER13}: crashed; 0/${MEASURE_RUNS} runs completed"            | tee -a "$LOG_FILE"
  else
    echo "DuckDB ${VER13}: no measured runs executed"                             | tee -a "$LOG_FILE"
  fi
fi
if [ "${DUCKDB_14_COMPLETED:-0}" -gt 0 ]; then
  echo "DuckDB ${VER14} avg time (${DUCKDB_14_COMPLETED}/${MEASURE_RUNS} runs): $(printf "%.2f" "$DUCKDB_14_AVG") seconds" | tee -a "$LOG_FILE"
elif [ "${DUCKDB_14_CRASHED:-false}" = true ]; then
  echo "DuckDB ${VER14}: crashed; 0/${MEASURE_RUNS} runs completed"            | tee -a "$LOG_FILE"
else
  echo "DuckDB ${VER14}: no measured runs executed"                             | tee -a "$LOG_FILE"
fi

if [ "$SKIP_13" = false ] && (( $(echo "${DUCKDB_13_AVG:-0} > 0" | bc -l) )) && (( $(echo "${DUCKDB_14_AVG:-0} > 0" | bc -l) )); then
  REL=$(echo "scale=2; $DUCKDB_13_AVG / $DUCKDB_14_AVG" | bc)
  echo "Relative (1.3 / 1.4) avg: ${REL}x"                                    | tee -a "$LOG_FILE"
fi

echo ""                                                                        | tee -a "$LOG_FILE"
echo "Output files:"                                                          | tee -a "$LOG_FILE"
if [ "$SKIP_13" = false ]; then
  echo "  - DuckDB ${VER13} log: ${OUTPUT_DIR}/duckdb_13_output.log"            | tee -a "$LOG_FILE"
fi
echo "  - DuckDB ${VER14} log: ${OUTPUT_DIR}/duckdb_14_output.log"            | tee -a "$LOG_FILE"
echo ""                                                                        | tee -a "$LOG_FILE"
echo "SQL scripts:"                                                            | tee -a "$LOG_FILE"
if [ "$SKIP_13" = false ]; then
  echo "  - ${OUTPUT_DIR}/duckdb_sort_13.sql"                                    | tee -a "$LOG_FILE"
fi
echo "  - ${OUTPUT_DIR}/duckdb_sort_14.sql"                                    | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
