#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG — adjust paths as needed
########################################
ES_ROOT="/mnt/nvme1/cmbenello/es"           # repo root
SRC_DIR="/mnt/nvme1/cmbenello/data/sf200"   # where lineitem.{csv,kvbin,idx} live

CSV="${SRC_DIR}/lineitem.csv"
KVBIN="${SRC_DIR}/lineitem.kvbin"

# Thread sweeps
RUN_GEN_THREADS=(32 24 16 8 4 2 1)
MERGE_THREADS=(32 24 16 8 4 2 1)

# Run-size (per-thread MB) sweep = memory allowance factor
RUN_SIZES=(114 341 1024 2048 4096)

# Bench config
BENCHMARK_RUNS=1      # set to 2 if you want two measurements per config
WARMUP_RUNS=1         # warmups disabled for now
MERGE_FANIN=512
TEMP_DIR="${ES_ROOT}" # where temp files go (can be a dedicated tmp dir)

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

########################################
# Paths / build
########################################
cd "${ES_ROOT}"

BENCH_BIN="${ES_ROOT}/target/release/examples/lineitem_benchmark_cli"
CSV2KVBIN="${ES_ROOT}/target/release/examples/csv_thread_scaling"  # DUMMY; only used if you really want csv2kvbin

# Build benchmark binary if missing
if [[ ! -x "$BENCH_BIN" ]]; then
  echo -e "${YELLOW}Building benchmark binary...${NC}"
  cargo build --release --example lineitem_benchmark_cli
fi

if [[ ! -f "$CSV" ]]; then
  echo -e "${RED}Error: CSV input '$CSV' not found${NC}"
  exit 1
fi

# Ensure KVBin exists (but do NOT require csv2kvbin example to exist
# if you’ve already generated KVBIN some other way)
if [[ ! -f "$KVBIN" ]]; then
  echo -e "${RED}KVBin '$KVBIN' is missing, and this script expects it to already exist.${NC}"
  echo -e "${RED}Generate it once (using your preferred tool) and rerun this script.${NC}"
  exit 1
fi

########################################
# Results dir + helpers
########################################
ts="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ES_ROOT}/benchmark_results/thread_sweep_kvbin_ovc_${ts}"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results directory: ${RESULTS_DIR}${NC}\n"

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "Config,RunGenThreads,MergeThreads,RunSizeMB,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size" > "$SUMMARY_CSV"

########################################
# Metric extraction – tolerant to missing lines
########################################
extract_metrics() {
  local output="$1"
  local label="$2"

  local sort_time throughput peak_memory total_io avg_key_size avg_value_size

  # Each grep guarded with `|| true` so lack of match doesn't kill the script
  sort_time=$(echo "$output" | grep -m1 "Sort completed in" \
              | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/' \
              || true)
  throughput=$(echo "$output" | grep -m1 "Throughput:" \
               | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/' \
               || true)
  peak_memory=$(echo "$output" | grep -m1 "Peak memory usage:" \
                | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/' \
                || true)
  total_io=$(echo "$output" | grep -m1 "Total I/O:" \
             | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/' \
             || true)
  avg_key_size=$(echo "$output" | grep -m1 "Average key size:" \
                 | sed 's/.*Average key size: \([0-9.]*\) bytes.*/\1/' \
                 || true)
  avg_value_size=$(echo "$output" | grep -m1 "Average value size:" \
                   | sed 's/.*Average value size: \([0-9.]*\) bytes.*/\1/' \
                   || true)

  echo -e "${YELLOW}=== ${label} ===${NC}"
  [[ -n "${sort_time}"      ]] && echo "  Sort time: ${sort_time}s"
  [[ -n "${throughput}"     ]] && echo "  Throughput: ${throughput} MB/s"
  [[ -n "${peak_memory}"    ]] && echo "  Peak memory: ${peak_memory} MB"
  [[ -n "${total_io}"       ]] && echo "  Total I/O: ${total_io} MB"
  [[ -n "${avg_key_size}"   ]] && echo "  Avg key size: ${avg_key_size} bytes"
  [[ -n "${avg_value_size}" ]] && echo "  Avg value size: ${avg_value_size} bytes"
  echo

  # label format: rg<R>_mg<M>_rs<S>
  local rg mt rs
  rg=$(echo "$label" | sed 's/.*rg\([0-9]\+\).*/\1/')
  mt=$(echo "$label" | sed 's/.*mg\([0-9]\+\).*/\1/')
  rs=$(echo "$label" | sed 's/.*rs\([0-9]\+\).*/\1/')

  echo "${label},${rg},${mt},${rs},${sort_time},${throughput},${peak_memory},${total_io},${avg_key_size},${avg_value_size}" \
    >> "$SUMMARY_CSV"

  # Never fail from here
  return 0
}

########################################
# Run one config (OVC ON)
########################################
run_one_config() {
  local rs="$1"
  local rg="$2"
  local mt="$3"

  local label="rg${rg}_mg${mt}_rs${rs}"
  local log="${RESULTS_DIR}/${label}.txt"

  echo -e "${BLUE}>>> START Config: run_gen_threads=${rg}, merge_threads=${mt}, run_size_mb=${rs} (OVC ON)${NC}"

  # Run benchmark; capture status explicitly so set -e doesn't kill us
  if ! "$BENCH_BIN" \
      -n "$label" \
      -i "$KVBIN" \
      --benchmark-runs "$BENCHMARK_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --ovc \
      --run-gen-threads "$rg" \
      --merge-threads "$mt" \
      --run-size-mb "$rs" \
      --merge-fanin "$MERGE_FANIN" \
      --dir "$TEMP_DIR" 2>&1 | tee "$log"; then
    echo -e "${RED}[WARN] ${label} benchmark binary exited non-zero, skipping metrics extraction.${NC}"
    echo -e "${RED}[WARN] See log:${NC} ${log}"
    return 1
  fi

  # Extract metrics (never fails)
  extract_metrics "$(cat "$log")" "$label"
  echo -e "${BLUE}>>> DONE  Config: ${label}${NC}"
  return 0
}

########################################
# Sweep: (run_gen_threads × merge_threads × run_size_mb) with OVC
########################################

for rs in "${RUN_SIZES[@]}"; do
  echo -e "${GREEN}=== Run-size per-thread: ${rs} MB ===${NC}"
  for rg in "${RUN_GEN_THREADS[@]}"; do
    for mt in "${MERGE_THREADS[@]}"; do
      # Never let a single failed config kill the whole sweep
      run_one_config "$rs" "$rg" "$mt" || true
      # Optional cooldown:
      # sleep 2
    done
  done
done

echo -e "${GREEN}Done. Summary written to:${NC} ${SUMMARY_CSV}"
echo "All logs in: ${RESULTS_DIR}"
