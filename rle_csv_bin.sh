#!/usr/bin/env bash
set -euo pipefail

# ================================
# Config (SF10 only)
# ================================
THREADS=40
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT="experiments/run_length_sf10_${TIMESTAMP}"

# Memory sweep for SF10
MEM_SF10="512MB,1GB,2GB,4GB,8GB"

# CSV key/value column indices
CSV_KEYS="10,5,1"   # l_shipdate, l_extendedprice, l_partkey
CSV_VALS="0,3,11"   # l_orderkey, l_linenumber, l_commitdate

# Inputs
CSV_SF10="sf10/lineitem.csv"
KVBIN_SF10="sf10/lineitem.kvbin"

# Optional passthrough
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ================================
# Candidate flags (we'll probe)
# ================================
MEM_FLAGS=(--memory --mem -m --run-mem --run_memory --run-size --run_size --buffersize --buffer)
THREAD_FLAGS=(--threads -t --num-threads)
INPUT_FLAGS=(--input "")   # prefer --input, else positional
KEY_FLAGS=(--key --keys --key-cols --keycols -k)
VAL_FLAGS=(--value --values --value-cols --valuecols -v)
KVBIN_FLAGS=(--kvbin "--input-format=kvbin" "--format=kvbin" "")  # last = rely on extension

EXE_PREFIX=(cargo run --release --example lineitem_benchmark_cli --)

# Run a trial command and return 0 if flags weren't rejected
try_cmd() {
  # We expect the program to either run or fail for reasons unrelated to
  # "unexpected argument". If stderr contains "unexpected argument", we reject.
  set +e
  out="$("${EXE_PREFIX[@]}" "$@" 2>&1 </dev/null)"
  rc=$?
  set -e
  if grep -qiE 'unexpected (argument|option) .*(--|-)[A-Za-z0-9_-]+' <<<"$out"; then
    return 1
  fi
  # If it complains about missing file, missing required options, etc., that still
  # means the flags themselves are accepted → treat as success.
  return 0
}

pick_flag() {
  # $1 = array name of candidates, $2... = base args to include for the probe
  local -n CANDS="$1"
  shift
  local base_args=( "$@" )

  for cand in "${CANDS[@]}"; do
    if [[ -z "$cand" ]]; then
      # positional/no flag case → skip adding the name
      if try_cmd "${base_args[@]}"; then
        echo ""
        return 0
      fi
    else
      if try_cmd "${base_args[@]}" "$cand" "DUMMY"; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

pick_input_style() {
  local infile="$1"
  # Try --input FILE first
  if try_cmd --input "$infile" --help; then
    echo "--input"
    return 0
  fi
  # Positional input (no flag)
  if try_cmd "$infile" --help; then
    echo ""
    return 0
  fi
  echo ""   # fallback positional
  return 0
}

detect_all_flags() {
  local csv_in="$1"

  # INPUT style
  INPUT_FLAG="$(pick_input_style "$csv_in")"

  # MEMORY flag (probe against csv_in; pass a harmless memory token)
  MEMORY_FLAG=""
  for cand in "${MEM_FLAGS[@]}"; do
    if [[ -z "$cand" ]]; then continue; fi
    if try_cmd ${INPUT_FLAG:+$INPUT_FLAG} "$csv_in" "$cand" "512MB" --help; then
      MEMORY_FLAG="$cand"
      break
    fi
  done
  if [[ -z "$MEMORY_FLAG" ]]; then
    echo "WARN: Could not auto-detect memory flag; will run WITHOUT a memory sweep flag." >&2
  fi

  # THREADS flag (optional)
  THREADS_FLAG=""
  for cand in "${THREAD_FLAGS[@]}"; do
    if [[ -z "$cand" ]]; then continue; fi
    if try_cmd ${INPUT_FLAG:+$INPUT_FLAG} "$csv_in" "$cand" "2" --help; then
      THREADS_FLAG="$cand"
      break
    fi
  done

  # CSV key/value flags (optional; only for CSV path)
  KEYS_FLAG=""
  for cand in "${KEY_FLAGS[@]}"; do
    if try_cmd ${INPUT_FLAG:+$INPUT_FLAG} "$csv_in" "$cand" "$CSV_KEYS" --help; then
      KEYS_FLAG="$cand"; break
    fi
  done
  VALS_FLAG=""
  for cand in "${VAL_FLAGS[@]}"; do
    if try_cmd ${INPUT_FLAG:+$INPUT_FLAG} "$csv_in" "$cand" "$CSV_VALS" --help; then
      VALS_FLAG="$cand"; break
    fi
  done

  # KVBin indicator (optional)
  KVBIN_FLAG=""
  for cand in "${KVBIN_FLAGS[@]}"; do
    if [[ -z "$cand" ]]; then KVBIN_FLAG=""; break; fi
    if try_cmd ${INPUT_FLAG:+$INPUT_FLAG} "$csv_in" "$cand" --help; then
      KVBIN_FLAG="$cand"; break
    fi
  done
}

header() {
  local msg="$1"
  echo "=================================================================================="
  echo "$msg"
  echo "=================================================================================="
}

run_one() {
  local label="$1"   # csv | kvbin
  local infile="$2"
  local mems="$3"
  local outdir="$4"
  local summary="$5"

  mkdir -p "$outdir"
  local log="${outdir}/${label}_sf10_experiment.log"

  header "SF10 (${label^^}) - ${infile}" | tee "$log"
  if [[ ! -f "$infile" ]]; then
    echo "WARNING: input not found, skipping: $infile" | tee -a "$log" | tee -a "$summary"
    return
  fi

  echo "Running SF10 ${label^^}..." | tee -a "$summary"
  echo "Log file: $log" | tee -a "$summary"
  echo "" | tee -a "$summary"

  # Assemble fixed args
  ARGS=()
  [[ -n "${THREADS_FLAG:-}" ]] && ARGS+=("$THREADS_FLAG" "$THREADS")

  # For CSV, add key/value flags if available
  if [[ "$label" == "csv" ]]; then
    [[ -n "${KEYS_FLAG:-}" ]] && ARGS+=("$KEYS_FLAG" "$CSV_KEYS")
    [[ -n "${VALS_FLAG:-}" ]] && ARGS+=("$VALS_FLAG" "$CSV_VALS")
  else
    # KVBin hint if needed and not empty
    [[ -n "${KVBIN_FLAG:-}" ]] && ARGS+=($KVBIN_FLAG)
  fi

  # Memory sweep: if we found a memory flag, pass the sweep; else run once without it
  if [[ -n "${MEMORY_FLAG:-}" ]]; then
    ARGS+=("$MEMORY_FLAG" "$mems")
  else
    echo "NOTE: No memory flag detected; running a single pass without sweep." | tee -a "$summary" | tee -a "$log"
  fi

  # Input position
  if [[ -n "${INPUT_FLAG:-}" ]]; then
    ARGS+=("$INPUT_FLAG" "$infile")
  else
    ARGS+=("$infile")
  fi

  "${EXE_PREFIX[@]}" \
    "${ARGS[@]}" \
    ${EXTRA_FLAGS} 2>&1 | tee -a "$log"

  echo "SF10 ${label^^} test completed" | tee -a "$summary"
  echo "" | tee -a "$summary"
}

# ================================
# Main
# ================================
mkdir -p "$ROOT_OUT"
SUMMARY="${ROOT_OUT}/experiment_summary.txt"

{
  echo "====================================="
  echo "RUN LENGTH EXPERIMENT SUMMARY (SF10, CSV vs KVBin)"
  echo "====================================="
  echo "Date: $(date)"
  echo "Threads: $THREADS"
  echo "Output dir: $ROOT_OUT"
  echo ""
} | tee "$SUMMARY"

OUT_CSV="${ROOT_OUT}/csv"
OUT_KVBIN="${ROOT_OUT}/kvbin"
mkdir -p "$OUT_CSV" "$OUT_KVBIN"

# Probe flags once using the CSV file (exists and simplest)
detect_all_flags "$CSV_SF10"

# Run CSV and KVBin
run_one "csv"   "$CSV_SF10"   "$MEM_SF10" "$OUT_CSV"   "$SUMMARY"
run_one "kvbin" "$KVBIN_SF10" "$MEM_SF10" "$OUT_KVBIN" "$SUMMARY"

{
  echo ""
  echo "=================================================================================="
  echo "EXPERIMENT COMPLETE (SF10)"
  echo "=================================================================================="
  echo ""
  echo "Results saved in: $ROOT_OUT"
  echo "Logs:"
  echo "  CSV:   ${OUT_CSV}/csv_sf10_experiment.log"
  echo "  KVBin: ${OUT_KVBIN}/kvbin_sf10_experiment.log"
  echo ""
  echo "Key metrics to analyze per memory step:"
  echo "  - Number of runs"
  echo "  - Avg/median run size (MB)"
  echo "  - Run generation time (ms)"
  echo "  - Merge time (ms)"
  echo "  - Total sort time (ms)"
  echo "  - Throughput (M entries/sec)"
} | tee -a "$SUMMARY"

# Optional stitched CSV
if command -v python3 >/dev/null 2>&1 && [[ -f "parse_run_length_experiment.py" ]]; then
  COMBINED="${ROOT_OUT}/combined_sf10_summary.csv"
  echo "Generating combined CSV via parse_run_length_experiment.py ..." | tee -a "$SUMMARY"
  python3 parse_run_length_experiment.py \
    "${OUT_CSV}/csv_sf10_experiment.log" \
    "${OUT_KVBIN}/kvbin_sf10_experiment.log" > "$COMBINED" || true
  echo "Combined CSV: $COMBINED" | tee -a "$SUMMARY"
fi