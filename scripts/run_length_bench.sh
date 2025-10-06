cargo build --release --example lineitem_benchmark_cli
mkdir -p logs
TS=$(date +%F_%H-%M-%S)
BASE="logs/lineitem_${TS}_baseline.log"
OVC="logs/lineitem_${TS}_ovc.log"

# baseline
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 1 --benchmark-runs 3 -i lineitem_sf500.csv -t 32 -m 32768 \
  --sketch-sampling-interval 100 \
  2>&1 | tee "$BASE"

# with --ovc
./target/release/examples/lineitem_benchmark_cli \
  --warmup-runs 1 --benchmark-runs 3 -i lineitem_sf500.csv -t 32 -m 32768 \
  --sketch-sampling-interval 100 --ovc \
  2>&1 | tee "$OVC"