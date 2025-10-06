#!/bin/bash
mkdir -p logs
TS=$(date +%F_%H-%M-%S)
BASE="logs/lineitem_imbalance_factor_${TS}_baseline.log"
OVC="logs/lineitem_imbalance_factor_${TS}_ovc.log"

# baseline with imbalance_factor experiment
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 1 --benchmark-runs 3 -i lineitem_sf500.csv -t 40 -m 32768 \
  --sketch-sampling-interval 100 \
  --experiment-type imbalance_factor \
  2>&1 | tee "$BASE"

# with --ovc and imbalance_factor experiment
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 1 --benchmark-runs 3 -i lineitem_sf500.csv -t 40 -m 32768 \
  --sketch-sampling-interval 100 --ovc \
  --experiment-type imbalance_factor \
  2>&1 | tee "$OVC"
