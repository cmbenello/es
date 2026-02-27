#!/usr/bin/env bash
set -euo pipefail

# Run gensort + lineitem benchmarks back-to-back.
# Reuses dataset files already present on the local SSD; copies only if missing.

# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR="./datasets"

GENSORT_SRC="/tank/local/riki/datasets/gensort_200GiB.data"
GENSORT_LOCAL="${DATASETS_DIR}/gensort_200GiB.data"

LINEITEM_SRC_GLOB="/tank/local/riki/datasets/lineitem_sf500.k-8-9-13-14-15.v-0-3.kvbin*"
LINEITEM_DST_GLOB="${DATASETS_DIR}/lineitem_sf500.k-8-9-13-14-15.v-0-3.kvbin*"
LINEITEM_CSV="${DATASETS_DIR}/lineitem_sf500.csv"

# ── helpers ──────────────────────────────────────────────────────────────────
print_disk_info() {
  echo "=== Local SSD / disk info (${DATASETS_DIR}) ==="
  df -h "${DATASETS_DIR}"
  echo ""
}

glob_exists() {
  compgen -G "$1" > /dev/null 2>&1
}

# ── setup ────────────────────────────────────────────────────────────────────
mkdir -p "${DATASETS_DIR}"
print_disk_info

# ════════════════════════════════════════════════════════════════════════════
# 1. GENSORT BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
echo "=== [1/2] gensort benchmark ==="

if [[ ! -f "${GENSORT_SRC}" ]]; then
  echo "ERROR: gensort source not found: ${GENSORT_SRC}" >&2
  exit 1
fi

if [[ -f "${GENSORT_LOCAL}" ]]; then
  echo "[skip] gensort_200GiB.data already present locally, reusing."
else
  echo "[copy] Copying gensort_200GiB.data from tank..."
  cp "${GENSORT_SRC}" "${GENSORT_LOCAL}"
  echo "[copy] Done."
fi

print_disk_info

./scripts/gensort_sort_bench_new.sh "${GENSORT_LOCAL}"

echo "[cleanup] Removing gensort_200GiB.data..."
rm -f "${GENSORT_LOCAL}"
sync

# ════════════════════════════════════════════════════════════════════════════
# 2. LINEITEM BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
echo "=== [2/2] lineitem benchmark ==="

if ! glob_exists "${LINEITEM_SRC_GLOB}"; then
  echo "ERROR: No lineitem kvbin source files found matching: ${LINEITEM_SRC_GLOB}" >&2
  exit 1
fi

if glob_exists "${LINEITEM_DST_GLOB}"; then
  echo "[skip] lineitem kvbin files already present locally, reusing."
else
  echo "[copy] Copying lineitem kvbin files from tank..."
  # shellcheck disable=SC2086
  cp ${LINEITEM_SRC_GLOB} "${DATASETS_DIR}/"
  echo "[copy] Done."
fi

print_disk_info

./scripts/lineitem_sort_bench_new.sh "${LINEITEM_CSV}"

echo "[cleanup] Removing lineitem kvbin files..."
# shellcheck disable=SC2086
rm -f ${LINEITEM_DST_GLOB}
sync

echo "=== All benchmarks complete ==="
