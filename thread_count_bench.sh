#!/bin/bash
mkdir -p logs
TS=$(date +%F_%H-%M-%S)
BASE="logs/lineitem_thread_count_${TS}_baseline.log"
OVC="logs/lineitem_thread_count_${TS}_ovc.log"

# baseline with thread_count experiment
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 0 --benchmark-runs 1 -i lineitem_sf500.csv -t 32 -m 32768 \
  --sketch-sampling-interval 100 \
  --experiment-type thread_count \
  2>&1 | tee "$BASE"

# with --ovc and thread_count experiment
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 0 --benchmark-runs 1 -i lineitem_sf500.csv -t 32 -m 32768 \
  --sketch-sampling-interval 100 --ovc \
  --experiment-type thread_count \
  2>&1 | tee "$OVC"
