#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG — override via env vars if needed
########################################
ES_ROOT="${ES_ROOT:-/home/cc/es}"
DATASETS_DIR="${DATASETS_DIR:-${ES_ROOT}/datasets}"

# PrefixSkew-HeavyPayload dataset files
DATASET_BASENAME="${DATASET_BASENAME:-prefix_skew_heavy_payload}"
KVBIN="${KVBIN:-${DATASETS_DIR}/${DATASET_BASENAME}.kvbin}"
KVBIN_IDX="${KVBIN_IDX:-${KVBIN}.idx}"

# Generator sizing controls (used only when dataset is missing)
PREFIX_SKEW_SF="${PREFIX_SKEW_SF:-1.0}"
PREFIX_SKEW_GIB_PER_SF="${PREFIX_SKEW_GIB_PER_SF:-1.0}"
PREFIX_SKEW_ROWS="${PREFIX_SKEW_ROWS:-}"              # optional explicit row override
PREFIX_SKEW_SEED="${PREFIX_SKEW_SEED:-1}"
PREFIX_SKEW_BURST_PROB="${PREFIX_SKEW_BURST_PROB:-0.28}"
PREFIX_SKEW_HEAVY_PAYLOAD_MIN="${PREFIX_SKEW_HEAVY_PAYLOAD_MIN:-8192}"
# Default cap is sorter-compatible (64KiB extent model with 208-byte heavy keys).
PREFIX_SKEW_HEAVY_PAYLOAD_MAX="${PREFIX_SKEW_HEAVY_PAYLOAD_MAX:-65316}"
FORCE_REGENERATE_DATASET="${FORCE_REGENERATE_DATASET:-0}"

# Thread sweeps
RUN_GEN_THREADS=(64 56 48 40 32 24 16 8 4 1)
MERGE_THREADS=(64 56 48 40 32 24 16 8 4 1)

# Fixed total memory budgets (MB)
MEM_BUDGETS_MB=(256 512 1024 2048)

# Merge fan-in bound parameters
PAGE_KIB=64
PAGE_BYTES=$((PAGE_KIB * 1024))

# Bench config
BENCHMARK_RUNS=1
WARMUP_RUNS=0
TEMP_DIR="${TEMP_DIR:-${ES_ROOT}/tmp}"
mkdir -p "$TEMP_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

########################################
# Paths / build
########################################
cd "${ES_ROOT}"

mkdir -p "$DATASETS_DIR"

BENCH_BIN="${ES_ROOT}/target/release/examples/kvbin_benchmark_cli"
GEN_BIN="${ES_ROOT}/target/release/examples/gen_prefix_skew_heavy_payload_kvbin"

if [[ ! -x "$BENCH_BIN" || ! -x "$GEN_BIN" ]]; then
  echo -e "${YELLOW}Building benchmark + generator binaries...${NC}"
  cargo build --release \
    --example kvbin_benchmark_cli \
    --example gen_prefix_skew_heavy_payload_kvbin
fi

ensure_dataset() {
  if [[ "$FORCE_REGENERATE_DATASET" != "1" && -f "$KVBIN" && -f "$KVBIN_IDX" ]]; then
    echo -e "${GREEN}Using existing dataset:${NC} $KVBIN"
    return
  fi

  if [[ "$FORCE_REGENERATE_DATASET" == "1" ]]; then
    echo -e "${YELLOW}FORCE_REGENERATE_DATASET=1; regenerating dataset.${NC}"
    rm -f "$KVBIN" "$KVBIN_IDX"
  else
    echo -e "${YELLOW}PrefixSkew dataset missing; generating:${NC}"
  fi
  echo "  out:   $KVBIN"
  echo "  idx:   $KVBIN_IDX"
  echo "  sf:    $PREFIX_SKEW_SF"
  echo "  gib/sf:$PREFIX_SKEW_GIB_PER_SF"
  echo "  seed:  $PREFIX_SKEW_SEED"
  echo "  burst: $PREFIX_SKEW_BURST_PROB"
  echo "  heavy-payload-min: $PREFIX_SKEW_HEAVY_PAYLOAD_MIN"
  echo "  heavy-payload-max: $PREFIX_SKEW_HEAVY_PAYLOAD_MAX"
  if [[ -n "${PREFIX_SKEW_ROWS}" ]]; then
    echo "  rows:  $PREFIX_SKEW_ROWS (overrides sf)"
  fi

  local gen_args=(
    --out "$KVBIN"
    --idx "$KVBIN_IDX"
    --sf "$PREFIX_SKEW_SF"
    --gib-per-sf "$PREFIX_SKEW_GIB_PER_SF"
    --seed "$PREFIX_SKEW_SEED"
    --burst-start-prob "$PREFIX_SKEW_BURST_PROB"
    --heavy-payload-min "$PREFIX_SKEW_HEAVY_PAYLOAD_MIN"
    --heavy-payload-max "$PREFIX_SKEW_HEAVY_PAYLOAD_MAX"
  )
  if [[ -n "${PREFIX_SKEW_ROWS}" ]]; then
    gen_args+=(--rows "$PREFIX_SKEW_ROWS")
  fi

  "$GEN_BIN" "${gen_args[@]}"
}

ensure_dataset

if [[ ! -f "$KVBIN" ]]; then
  echo -e "${RED}Error: Expected KVBin not found:${NC} $KVBIN"
  exit 1
fi
if [[ ! -f "$KVBIN_IDX" ]]; then
  echo -e "${RED}Error: Expected KVBin index not found:${NC} $KVBIN_IDX"
  exit 1
fi

########################################
# Results dir + helpers
########################################
ts="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ES_ROOT}/benchmark_results/thread_sweep_prefix_skew_ovc_mem_${ts}"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results directory: ${RESULTS_DIR}${NC}\n"

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "Config,MemBudgetMB,RunSizePerThreadMB,RunGenThreads,MergeThreads,MergeFanin,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size" > "$SUMMARY_CSV"

########################################
# Temp cleanup (after EACH config)
########################################
cleanup_tmp() {
  if [[ -z "${TEMP_DIR}" || "${TEMP_DIR}" == "/" ]]; then
    echo -e "${RED}[FATAL] TEMP_DIR is unsafe ('${TEMP_DIR}'); refusing to delete.${NC}"
    exit 1
  fi
  rm -rf "${TEMP_DIR:?}/"*
  mkdir -p "$TEMP_DIR"
}

# Paper-style fan-in bound: F = floor(M / (Tmerge * P))
compute_merge_fanin() {
  local mem_budget_mb="$1"
  local mt="$2"

  local M_bytes=$((mem_budget_mb * 1024 * 1024))
  local denom=$((mt * PAGE_BYTES))

  if [[ "$denom" -le 0 ]]; then
    echo 2
    return
  fi

  local F=$((M_bytes / denom))
  if [[ "$F" -lt 2 ]]; then F=2; fi
  if [[ "$F" -gt 8192 ]]; then F=8192; fi
  echo "$F"
}

extract_metrics() {
  local output="$1"
  local label="$2"
  local mem_budget="$3"
  local run_size_per_thread="$4"
  local rg="$5"
  local mt="$6"
  local fanin="$7"

  local sort_time throughput peak_memory total_io avg_key_size avg_value_size

  sort_time=$(echo "$output" | grep -m1 "Sort completed in" | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/' || true)
  throughput=$(echo "$output" | grep -m1 "Throughput:" | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/' || true)
  peak_memory=$(echo "$output" | grep -m1 "Peak memory usage:" | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/' || true)
  total_io=$(echo "$output" | grep -m1 "Total I/O:" | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/' || true)
  avg_key_size=$(echo "$output" | grep -m1 "Average key size:" | sed 's/.*Average key size: \([0-9.]*\) bytes.*/\1/' || true)
  avg_value_size=$(echo "$output" | grep -m1 "Average value size:" | sed 's/.*Average value size: \([0-9.]*\) bytes.*/\1/' || true)

  echo "${label},${mem_budget},${run_size_per_thread},${rg},${mt},${fanin},${sort_time},${throughput},${peak_memory},${total_io},${avg_key_size},${avg_value_size}" \
    >> "$SUMMARY_CSV"
}

run_one_config() {
  local mem_budget_mb="$1"
  local rg="$2"
  local mt="$3"

  cleanup_tmp
  trap cleanup_tmp RETURN

  local rg_buf_mb
  rg_buf_mb=$(awk -v total="$mem_budget_mb" -v thr="$rg" 'BEGIN { printf "%.2f", total/thr }')

  local merge_fanin
  merge_fanin="$(compute_merge_fanin "$mem_budget_mb" "$mt")"

  local label="mem${mem_budget_mb}_rg${rg}_mg${mt}_F${merge_fanin}"
  local log="${RESULTS_DIR}/${label}.txt"

  echo -e "${BLUE}>>> START ${label} (dataset=${DATASET_BASENAME}, P=${PAGE_KIB}KiB)${NC}"

  if ! "$BENCH_BIN" \
      -n "$label" \
      -i "$KVBIN" \
      --index "$KVBIN_IDX" \
      --ovc true \
      --run-gen-threads "$rg" \
      --merge-threads "$mt" \
      --rg-buf-mb "$rg_buf_mb" \
      --merge-fanin "$merge_fanin" \
      --benchmark-runs "$BENCHMARK_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --partition-type size-balanced \
      --dir "$TEMP_DIR" 2>&1 | tee "$log"; then
    echo -e "${RED}[WARN] ${label} exited non-zero, skipping metrics.${NC}"
    return 1
  fi

  extract_metrics "$(cat "$log")" "$label" "$mem_budget_mb" "$rg_buf_mb" "$rg" "$mt" "$merge_fanin"
  return 0
}

for mem in "${MEM_BUDGETS_MB[@]}"; do
  echo -e "${GREEN}=== Memory budget: ${mem} MB ===${NC}"
  for rg in "${RUN_GEN_THREADS[@]}"; do
    for mt in "${MERGE_THREADS[@]}"; do
      run_one_config "$mem" "$rg" "$mt" || true
    done
  done
done

echo -e "${GREEN}Done. Summary written to:${NC} ${SUMMARY_CSV}"
echo "All logs in: ${RESULTS_DIR}"
