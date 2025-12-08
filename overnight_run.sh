#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG — adjust if your paths differ
# =========================
ES_ROOT="/mnt/nvme1/cmbenello/es"            # repo root (where the bench script lives)
SCRIPT="${ES_ROOT}/lineitem_csv_bin.sh"       # your benchmark driver
SRC_DIR="/mnt/nvme1/cmbenello/data/sf100"       # where the data lives on NVMe
CLEAN_DIR="${ES_ROOT}"                        # directory where warmup*/176* temp dirs live

CSV="${SRC_DIR}/lineitem.csv"
KVBIN="${SRC_DIR}/lineitem.kvbin"
IDX="${SRC_DIR}/lineitem.idx"

# Per-thread run-size sweep; each is run in a separate invocation of the driver.
RUN_SIZES=("12" "38" "114" "341" "1024" "2048")

LOG_DIR="${ES_ROOT}/overnight_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/overnight_${TS}.log"

# Prevent accidental double-runs
LOCKFILE="/tmp/overnight_bench.lock"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "Another overnight_run is already running (lock: ${LOCKFILE}). Exiting." | tee -a "${LOG}"
  exit 1
fi

# Always remove the lock even on kill/error
trap 'rm -f /tmp/overnight_bench.lock' EXIT

echo "==== Overnight run started @ $(date -Is) ====" | tee -a "${LOG}"

# ---- helpers ----
clean_temps() {
  echo "[CLEAN] Removing ${CLEAN_DIR}/warmup* and ${CLEAN_DIR}/176* ..." | tee -a "${LOG}"
  ( cd "${CLEAN_DIR}" && rm -rf warmup* 176* 2>/dev/null || true )
}

run_with_cleanup() {
  # $1: label, $2+: command (already properly quoted by caller)
  local label="$1"; shift
  echo "[RUN] ${label}" | tee -a "${LOG}"

  clean_temps

  set +e
  "$@" 2>&1 | tee -a "${LOG}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "[WARN] ${label} exited with status ${status}. Cleaning temps and continuing..." | tee -a "${LOG}"
    clean_temps
    return 0
  fi

  clean_temps
  return 0
}

# =========================
# 1) KVBin-only benchmarks (per run-size, OVC=both)
# =========================
if [[ ! -x "${SCRIPT}" ]]; then
  echo "ERROR: ${SCRIPT} not found or not executable" | tee -a "${LOG}"
  exit 1
fi

if [[ ! -f "${KVBIN}" && -f "${CSV}" ]]; then
  echo "KVBin missing; the bench script will build it from CSV as needed." | tee -a "${LOG}"
fi

echo "[1/6] KVBin benchmarks, per run-size..." | tee -a "${LOG}"
for rs in "${RUN_SIZES[@]}"; do
  run_with_cleanup \
    "KVBin --ovc=both --rs=${rs}" \
    "${SCRIPT}" \
      --mode=kvbin --ovc=both \
      --csv="${CSV}" \
      --kvbin="${KVBIN}" \
      --rs="${rs}"
done

# =========================
# 2) Cleanup temp artifacts now (extra safety)
# =========================
echo "[2/6] Global cleanup after KVBin phase" | tee -a "${LOG}"
clean_temps

