#!/usr/bin/env bash

set -euo pipefail

# fio I/O bandwidth benchmark over varying thread counts (numjobs)
# - Supports workloads: randread, randwrite, randrw
# - Measures empirical bandwidth using fio JSON output
# - Logs results to a CSV for easy plotting (x: threads, y: bandwidth)
#
# Usage examples:
#   scripts/fio_io_bandwidth_bench.sh \
#     --threads 1,2,4,8,16,32 \
#     --workloads randwrite,randread \
#     --bs 4k --size 4G --runtime 60 \
#     --filename ./fio_random_file
#
#   scripts/fio_io_bandwidth_bench.sh --workloads randrw --threads 1,2,4,8 --filename /mnt/nvme1/fio_test_file
#

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  -t, --threads CSV        Comma-separated thread counts (numjobs). Default: 1,2,4,8,16,32
  -w, --workloads CSV      Comma-separated workloads: randread, randwrite, randrw. Default: randwrite,randread
  -b, --bs SIZE            Block size (e.g., 4k, 128k). Default: 4k
  -s, --size SIZE          File size per job (fio 'size'). Default: 4G
  -r, --runtime SECS       Runtime seconds per run (time_based). Default: 60
  -d, --iodepth N          IO depth (per job). Default: 1
      --ioengine NAME      fio ioengine. Default: psync
  -f, --filename PATH      Target file path for IO. Default: ./fio_random_file
  -o, --outdir DIR         Output directory base. Default: logs/fio_io
      --no-prefill         Skip prefill for read workloads (not recommended)
  -h, --help               Show this help and exit

Notes:
  - For read workloads (randread, randrw), the script will prefill the file via fio sequential write
    unless --no-prefill is supplied. This ensures the file has real data on disk (not sparse).
  - Bandwidth is aggregated across jobs; CSV contains read, write, and total bandwidth in bytes/sec and MiB/s.
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage; exit 0
fi

command -v fio >/dev/null 2>&1 || { echo "Error: fio not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required to parse fio JSON output"; exit 1; }

# Defaults
THREADS_CSV="1,2,4,8,16,32"
WORKLOADS_CSV="randwrite,randread"
BS="64k"
SIZE="32G"
RUNTIME=60
IODEPTH=1
IOENGINE="psync"
FILENAME="./fio_random_file"
OUTDIR_BASE="logs/fio_io"
PREFILL=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--threads) THREADS_CSV="$2"; shift 2;;
    --threads=*) THREADS_CSV="${1#*=}"; shift ;;
    -w|--workloads) WORKLOADS_CSV="$2"; shift 2;;
    --workloads=*) WORKLOADS_CSV="${1#*=}"; shift ;;
    -b|--bs) BS="$2"; shift 2;;
    --bs=*) BS="${1#*=}"; shift ;;
    -s|--size) SIZE="$2"; shift 2;;
    --size=*) SIZE="${1#*=}"; shift ;;
    -r|--runtime) RUNTIME="$2"; shift 2;;
    --runtime=*) RUNTIME="${1#*=}"; shift ;;
    -d|--iodepth) IODEPTH="$2"; shift 2;;
    --iodepth=*) IODEPTH="${1#*=}"; shift ;;
    --ioengine) IOENGINE="$2"; shift 2;;
    --ioengine=*) IOENGINE="${1#*=}"; shift ;;
    -f|--filename) FILENAME="$2"; shift 2;;
    --filename=*) FILENAME="${1#*=}"; shift ;;
    -o|--outdir) OUTDIR_BASE="$2"; shift 2;;
    --outdir=*) OUTDIR_BASE="${1#*=}"; shift ;;
    --no-prefill) PREFILL=false; shift ;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; echo; usage; exit 1;;
  esac
done

# Prepare output dirs and CSV
TS=$(date +"%Y%m%d_%H%M%S")
OUTDIR="${OUTDIR_BASE}/${TS}"
mkdir -p "$OUTDIR"

# Per-experiment CSV (stored inside the timestamped output directory)
CSV_EXPERIMENT="${OUTDIR}/fio_io_bandwidth.csv"
if [ ! -f "$CSV_EXPERIMENT" ]; then
  echo "timestamp,workload,threads,bs,iodepth,ioengine,size,runtime,read_bw_bytes,write_bw_bytes,total_bw_bytes,read_bw_mib_s,write_bw_mib_s,total_bw_mib_s,json_path" > "$CSV_EXPERIMENT"
fi

IFS=',' read -r -a THREADS_ARR <<< "$THREADS_CSV"
IFS=',' read -r -a WORKLOADS_ARR <<< "$WORKLOADS_CSV"

# Prefill file if any read workload present and prefill enabled
needs_prefill=false
for w in "${WORKLOADS_ARR[@]}"; do
  if [[ "$w" == "randread" || "$w" == "randrw" ]]; then
    needs_prefill=true; break
  fi
done

if $PREFILL && $needs_prefill; then
  echo "Prefilling file for read workloads: $FILENAME (${SIZE})"
  fio \
    --name=prefill \
    --ioengine="$IOENGINE" \
    --rw=write \
    --bs=1M \
    --numjobs=1 \
    --size="$SIZE" \
    --iodepth="$IODEPTH" \
    --direct=1 \
    --filename="$FILENAME" \
    --time_based=0 \
    --group_reporting=1 \
    --output-format=json \
    --output="${OUTDIR}/prefill.json" >/dev/null
fi

bytes_to_mib_py='import sys; print(float(sys.argv[1]) / 1048576.0)'

for workload in "${WORKLOADS_ARR[@]}"; do
  echo "--- Workload: ${workload} ---"
  for threads in "${THREADS_ARR[@]}"; do
    echo "Running: workload=${workload} threads=${threads} bs=${BS} iodepth=${IODEPTH} runtime=${RUNTIME}s"
    JSON_OUT="${OUTDIR}/${workload}_threads${threads}.json"

    # Execute fio
    set +e
    fio \
      --name="${workload}-t${threads}" \
      --ioengine="$IOENGINE" \
      --rw="$workload" \
      --bs="$BS" \
      --numjobs="$threads" \
      --size="$SIZE" \
      --iodepth="$IODEPTH" \
      --runtime="$RUNTIME" \
      --time_based=1 \
      --direct=1 \
      --filename="$FILENAME" \
      --group_reporting=1 \
      --eta-newline=1 \
      --output-format=json \
      --output="$JSON_OUT"
    status=$?
    set -e

    if [ $status -ne 0 ]; then
      echo "fio run failed (status=$status) for workload=${workload}, threads=${threads}. See ${JSON_OUT} (may be empty)." >&2
      # Still record a row with zeros to keep the series complete
      echo "${TS},${workload},${threads},${BS},${IODEPTH},${IOENGINE},${SIZE},${RUNTIME},0,0,0,0,0,0,${JSON_OUT}" >> "$CSV_EXPERIMENT"
      continue
    fi

    # Parse JSON and aggregate bandwidth across jobs
    RW_BW=$(python3 - "$JSON_OUT" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    j=json.load(f)
jobs=j.get("jobs", [])
read_bw=sum((job.get("read",{}) or {}).get("bw_bytes",0) for job in jobs)
write_bw=sum((job.get("write",{}) or {}).get("bw_bytes",0) for job in jobs)
print(f"{read_bw},{write_bw}")
PY
)
    READ_BW_BYTES=$(echo "$RW_BW" | cut -d, -f1)
    WRITE_BW_BYTES=$(echo "$RW_BW" | cut -d, -f2)
    READ_BW_BYTES=${READ_BW_BYTES:-0}
    WRITE_BW_BYTES=${WRITE_BW_BYTES:-0}

    case "$workload" in
      randread)  TOTAL_BW_BYTES="$READ_BW_BYTES" ;;
      randwrite) TOTAL_BW_BYTES="$WRITE_BW_BYTES" ;;
      randrw)    TOTAL_BW_BYTES=$(( READ_BW_BYTES + WRITE_BW_BYTES )) ;;
      *)         TOTAL_BW_BYTES=$(( READ_BW_BYTES + WRITE_BW_BYTES )) ;;
    esac

    READ_MIB=$(python3 -c "$bytes_to_mib_py" "$READ_BW_BYTES")
    WRITE_MIB=$(python3 -c "$bytes_to_mib_py" "$WRITE_BW_BYTES")
    TOTAL_MIB=$(python3 -c "$bytes_to_mib_py" "$TOTAL_BW_BYTES")

    printf "Result: threads=%s total=%.2f MiB/s (read=%.2f, write=%.2f)\n" \
      "$threads" "$TOTAL_MIB" "$READ_MIB" "$WRITE_MIB"

    ROW="${TS},${workload},${threads},${BS},${IODEPTH},${IOENGINE},${SIZE},${RUNTIME},${READ_BW_BYTES},${WRITE_BW_BYTES},${TOTAL_BW_BYTES},${READ_MIB},${WRITE_MIB},${TOTAL_MIB},${JSON_OUT}"
    echo "$ROW" >> "$CSV_EXPERIMENT"
  done
done

echo
echo "Experiment CSV: ${CSV_EXPERIMENT}"
echo "Per-run JSON outputs: ${OUTDIR}"
