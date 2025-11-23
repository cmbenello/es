#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (defaults; overridable by flags)
# =========================
INPUT_CSV="sf500/lineitem.csv"     # CSV source
KVBIN_OUT="sf500/lineitem.kvbin"   # KVBin target (auto-built if needed)
THREADS=32
BENCHMARK_RUNS=1
WARMUP_RUNS=1
# CSV column indices for key/value (adjust if needed)
KEYS="10,5,1"
VALS="0,3,11"

# Merge fan-in during merge phase
MERGE_FANIN=512

# Write an index entry every N records when converting CSV -> KVBin
IDX_EVERY=100000

# Sweep of *per-thread* run sizes (MB)
RUN_SIZE_SWEEP=(12 114 1024)
# RUN_SIZE_SWEEP=(1024 341 114 38 12)

# Mode & variant controls
MODE="kvbin"            # csv | kvbin | both
OVC_VARIANT="both"     # no | yes | both
SKIP_BUILD="no"        # yes | no

# =========================
# Styling
# =========================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --mode=csv|kvbin|both      Which input format(s) to run (default: both)
  --ovc=no|yes|both          Run without OVC, with OVC, or both (default: both)
  --threads=N                Threads for run-gen and merge (default: $THREADS)
  --runs=N                   Benchmark runs (default: $BENCHMARK_RUNS)
  --warmup=N                 Warmup runs (default: $WARMUP_RUNS)
  --rs="a,b,c"               Run-size-per-thread MB sweep (comma-separated)
  --csv=PATH                 CSV input path (default: $INPUT_CSV)
  --kvbin=PATH               KVBin path (default: $KVBIN_OUT)
  --keys="i,j,k"             CSV key columns (default: $KEYS)
  --vals="i,j,k"             CSV value columns (default: $VALS)
  --fanin=N                  Merge fan-in (default: $MERGE_FANIN)
  --idx-every=N              csv2kvbin index stride (default: $IDX_EVERY)
  --skip-build               Use existing binaries; don't build
  -h|--help                  Show this help

Examples:
  $0 --mode=kvbin
  $0 --mode=csv --ovc=no
  $0 --mode=kvbin --rs="12,38,114,341"
EOF
}

# =========================
# Parse args
# =========================
for arg in "$@"; do
  case "$arg" in
    --mode=*) MODE="${arg#*=}";;
    --ovc=*) OVC_VARIANT="${arg#*=}";;
    --threads=*) THREADS="${arg#*=}";;
    --runs=*) BENCHMARK_RUNS="${arg#*=}";;
    --warmup=*) WARMUP_RUNS="${arg#*=}";;
    --rs=*) IFS=',' read -r -a RUN_SIZE_SWEEP <<< "${arg#*=}";;
    --csv=*) INPUT_CSV="${arg#*=}";;
    --kvbin=*) KVBIN_OUT="${arg#*=}";;
    --keys=*) KEYS="${arg#*=}";;
    --vals=*) VALS="${arg#*=}";;
    --fanin=*) MERGE_FANIN="${arg#*=}";;
    --idx-every=*) IDX_EVERY="${arg#*=}";;
    --skip-build) SKIP_BUILD="yes";;
    -h|--help) usage; exit 0;;
    *) echo -e "${RED}Unknown option: $arg${NC}"; usage; exit 1;;
  esac
done

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}  Lineitem Sort Benchmark: CSV vs KVBin (OVC/no)        ${NC}"
echo -e "${BLUE}  mode=${MODE}, ovc=${OVC_VARIANT}                      ${NC}"
echo -e "${BLUE}========================================================${NC}\n"

# =========================
# Sanity checks
# =========================
if [[ ! -f "$INPUT_CSV" ]]; then
  echo -e "${RED}Error: CSV input '$INPUT_CSV' not found${NC}"; exit 1
fi

# =========================
# Build
# =========================
if [[ "$SKIP_BUILD" != "yes" ]]; then
  echo -e "${YELLOW}Building binaries...${NC}"
  cargo build --release --example lineitem_benchmark_cli
  cargo build --release --example csv2kvbin || true  # if present
fi

BENCH_BIN="./target/release/examples/lineitem_benchmark_cli"
CSV2KVBIN="./target/release/examples/csv2kvbin"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo -e "${RED}Error: benchmark binary missing at $BENCH_BIN${NC}"; exit 1
fi

# =========================
# Ensure KVBin exists (only if kvbin mode is requested)
# =========================
need_kvbin="no"
case "$MODE" in
  kvbin|both) need_kvbin="yes";;
esac

if [[ "$need_kvbin" == "yes" && ! -f "$KVBIN_OUT" ]]; then
  if [[ -x "$CSV2KVBIN" ]]; then
    echo -e "${YELLOW}No KVBin found. Converting CSV -> KVBin (+.idx)...${NC}"
    "$CSV2KVBIN" \
      --input  "$INPUT_CSV" \
      --output "$KVBIN_OUT" \
      --keys   "$KEYS" \
      --vals   "$VALS" \
      --idx-every "$IDX_EVERY"
  else
    echo -e "${RED}KVBin '$KVBIN_OUT' missing and csv2kvbin example not available.${NC}"
    echo -e "${RED}Build or provide KVBin manually, or add the csv2kvbin example.${NC}"
    exit 1
  fi
fi

# =========================
# Results dir + helpers
# =========================
ts="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="benchmark_results/rs_sweep_${MODE}_${OVC_VARIANT}_${ts}"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results directory: $RESULTS_DIR${NC}\n"

extract_metrics() {
  local output="$1"
  local label="$2"
  local logfile="$3"

  local sort_time=$(echo "$output" | grep -m1 "Sort completed in" | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/')
  local throughput=$(echo "$output" | grep -m1 "Throughput:" | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/')
  local peak_memory=$(echo "$output" | grep -m1 "Peak memory usage:" | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/')
  local total_io=$(echo "$output" | grep -m1 "Total I/O:" | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/')
  local avg_key_size=$(echo "$output" | grep -m1 "Average key size:" | sed 's/.*Average key size: \([0-9.]*\) bytes.*/\1/')
  local avg_value_size=$(echo "$output" | grep -m1 "Average value size:" | sed 's/.*Average value size: \([0-9.]*\) bytes.*/\1/')

  echo -e "${YELLOW}=== $label ===${NC}"
  [ -n "${sort_time:-}" ]      && echo "  Sort time: ${sort_time}s"
  [ -n "${throughput:-}" ]     && echo "  Throughput: ${throughput} MB/s"
  [ -n "${peak_memory:-}" ]    && echo "  Peak memory: ${peak_memory} MB"
  [ -n "${total_io:-}" ]       && echo "  Total I/O: ${total_io} MB"
  [ -n "${avg_key_size:-}" ]   && echo "  Avg key size: ${avg_key_size} bytes"
  [ -n "${avg_value_size:-}" ] && echo "  Avg value size: ${avg_value_size} bytes"
  echo

  echo "$label,$sort_time,$throughput,$peak_memory,$total_io,$avg_key_size,$avg_value_size" >> "$RESULTS_DIR/summary.csv"
  echo "  Log: $logfile"
}

# CSV header for combined summary
echo "Mode,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size" > "$RESULTS_DIR/summary.csv"

run_csv_variant() {
  local rs="$1"
  local with_ovc="$2"   # yes|no
  local tag="CSV_$( [[ $with_ovc == yes ]] && echo OVC || echo NoOVC )_rs${rs}"
  local log="$RESULTS_DIR/${tag}.txt"
  echo -e "${BLUE}>>> CSV mode: $( [[ $with_ovc == yes ]] && echo WITH || echo NO ) OVC (run-size-mb=${rs})${NC}"
  "$BENCH_BIN" \
    -i "$INPUT_CSV" \
    -k "$KEYS" \
    -v "$VALS" \
    --benchmark-runs "$BENCHMARK_RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    $([[ $with_ovc == yes ]] && echo --ovc) \
    --run-gen-threads "$THREADS" \
    --merge-threads "$THREADS" \
    --run-size-mb "$rs" \
    --merge-fanin "$MERGE_FANIN" 2>&1 | tee "$log"
  extract_metrics "$(cat "$log")" "$tag" "$log"
}

run_kvbin_variant() {
  local rs="$1"
  local with_ovc="$2"   # yes|no
  local tag="KVBin_$( [[ $with_ovc == yes ]] && echo OVC || echo NoOVC )_rs${rs}"
  local log="$RESULTS_DIR/${tag}.txt"
  echo -e "${BLUE}>>> KVBin mode: $( [[ $with_ovc == yes ]] && echo WITH || echo NO ) OVC (run-size-mb=${rs})${NC}"
  "$BENCH_BIN" \
    -i "$KVBIN_OUT" \
    --benchmark-runs "$BENCHMARK_RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    $([[ $with_ovc == yes ]] && echo --ovc) \
    --run-gen-threads "$THREADS" \
    --merge-threads "$THREADS" \
    --run-size-mb "$rs" \
    --merge-fanin "$MERGE_FANIN" 2>&1 | tee "$log"
  extract_metrics "$(cat "$log")" "$tag" "$log"
}

# =========================
# Sweep
# =========================
for RS in "${RUN_SIZE_SWEEP[@]}"; do
  echo -e "${BLUE}==============================${NC}"
  echo -e "${BLUE} Run-size per thread: ${RS} MB ${NC}"
  echo -e "${BLUE}==============================${NC}"

  if [[ "$MODE" == "csv" || "$MODE" == "both" ]]; then
    case "$OVC_VARIANT" in
      no)   run_csv_variant "$RS" "no" ;;
      yes)  run_csv_variant "$RS" "yes" ;;
      both) run_csv_variant "$RS" "no"; run_csv_variant "$RS" "yes" ;;
      *) echo -e "${RED}Invalid --ovc value: $OVC_VARIANT${NC}"; exit 1;;
    esac
  fi

  if [[ "$MODE" == "kvbin" || "$MODE" == "both" ]]; then
    case "$OVC_VARIANT" in
      no)   run_kvbin_variant "$RS" "no" ;;
      yes)  run_kvbin_variant "$RS" "yes" ;;
      both) run_kvbin_variant "$RS" "no"; run_kvbin_variant "$RS" "yes" ;;
      *) echo -e "${RED}Invalid --ovc value: $OVC_VARIANT${NC}"; exit 1;;
    esac
  fi
  echo
done

echo -e "${GREEN}Summary written to:${NC} $RESULTS_DIR/summary.csv"
echo "All logs in: $RESULTS_DIR"
