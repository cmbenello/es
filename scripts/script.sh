#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG — adjust paths as needed
########################################
ES_ROOT="/home/cc/es"            # repo root
SRC_DIR="/home/cc/es/datasets"   # where lineitem.csv lives

CSV="${SRC_DIR}/lineitem.csv"

# --- Paper-faithful TPC-H lineitem key/payload ---
# Sort key: l_returnflag(8), l_linestatus(9), l_shipinstruct(13), l_shipmode(14), l_comment(15)
KEY_COLUMNS="8,9,13,14,15"
# Payload: l_orderkey(0), l_linenumber(3)
VALUE_COLUMNS="0,3"

# The KVBin name is derived from key/value columns
KVBIN="${ES_ROOT}/datasets/lineitem.k-8-9-13-14-15.v-0-3.kvbin"
KVBIN_IDX="${KVBIN}.idx"

# Thread sweeps
RUN_GEN_THREADS=(64 56 48 40 32 24 16 8 4)
MERGE_THREADS=(64 56 48 40 32 24 16 8 4)

# Fixed total memory budgets (MB)
MEM_BUDGETS_MB=(256 512 1024 2048)

# Paper page size P (bytes)
PAGE_KIB=64
PAGE_BYTES=$((PAGE_KIB * 1024))

# Bench config
BENCHMARK_RUNS=1
WARMUP_RUNS=0
TEMP_DIR="${ES_ROOT}/tmp"
mkdir -p "$TEMP_DIR"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

########################################
# Paths / build
########################################
cd "${ES_ROOT}"

BENCH_BIN="${ES_ROOT}/target/release/examples/lineitem_benchmark_cli"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo -e "${YELLOW}Building benchmark binary...${NC}"
  cargo build --release --example lineitem_benchmark_cli
fi

if [[ ! -f "$CSV" ]]; then
  echo -e "${RED}Error: CSV input '$CSV' not found${NC}"
  exit 1
fi

# Hard-require the KVBin + idx to exist (so it NEVER creates new ones)
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
RESULTS_DIR="${ES_ROOT}/benchmark_results/thread_sweep_kvbin_ovc_mem_${ts}"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results directory: ${RESULTS_DIR}${NC}\n"

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "Config,MemBudgetMB,RGBufferMB,RunGenThreads,MergeThreads,MergeFanin,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size" > "$SUMMARY_CSV"

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

# Paper-based fan-in bound: F = floor(M / (Tmerge * P))
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
  local rg_buf_mb="$4"
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

  echo "${label},${mem_budget},${rg_buf_mb},${rg},${mt},${fanin},${sort_time},${throughput},${peak_memory},${total_io},${avg_key_size},${avg_value_size}" >> "$SUMMARY_CSV"
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

  echo -e "${BLUE}>>> START ${label} (k=${KEY_COLUMNS}, v=${VALUE_COLUMNS}, P=${PAGE_KIB}KiB)${NC}"

  if ! "$BENCH_BIN" \
      -n "$label" \
      -i "$CSV" \
      -k "$KEY_COLUMNS" \
      -v "$VALUE_COLUMNS" \
      --ovc true \
      --run-gen-threads "$rg" \
      --merge-threads "$mt" \
      --rg-buf-mb "$rg_buf_mb" \
      --merge-fanin "$merge_fanin" \
      --partition-type size-balanced \
      --benchmark-runs "$BENCHMARK_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
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