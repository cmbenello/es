#!/bin/bash

# Configurable parameters - modify these as needed
INPUT_FILE="lineitem_sf100.csv"
MEMORY_MB=1024
THREADS=32
BENCHMARK_RUNS=5
WARMUP_RUNS=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lineitem Sort Benchmark: OVC vs Non-OVC${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Input file: $INPUT_FILE"
echo "  Memory: ${MEMORY_MB}MB"
echo "  Threads: $THREADS"
echo "  Benchmark runs: $BENCHMARK_RUNS"
echo "  Warmup runs: $WARMUP_RUNS"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

# Check if binary exists
cargo build --release --example lineitem_benchmark_cli
BINARY="./target/release/examples/lineitem_benchmark_cli"
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}Binary not found, building in release mode...${NC}"
    cargo build --release --example lineitem_benchmark_cli
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed${NC}"
        exit 1
    fi
fi

# Create temporary directory for results
RESULTS_DIR="benchmark_results/ovc_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# Function to extract metrics from output
extract_metrics() {
    local output="$1"
    local label="$2"
    
    # Extract key metrics
    local sort_time=$(echo "$output" | grep "Sort completed in" | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/')
    local throughput=$(echo "$output" | grep "Throughput:" | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/')
    local peak_memory=$(echo "$output" | grep "Peak memory usage:" | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/')
    local total_io=$(echo "$output" | grep "Total I/O:" | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/')
    local avg_key_size=$(echo "$output" | grep "Average key size:" | sed 's/.*Average key size: \([0-9.]*\) bytes.*/\1/')
    local avg_value_size=$(echo "$output" | grep "Average value size:" | sed 's/.*Average value size: \([0-9.]*\) bytes.*/\1/')
    
    echo -e "\n${YELLOW}=== $label Results ===${NC}"
    echo "Sort time: ${sort_time}s"
    echo "Throughput: ${throughput} MB/s"
    echo "Peak memory: ${peak_memory} MB"
    echo "Total I/O: ${total_io} MB"
    if [ ! -z "$avg_key_size" ]; then
        echo "Avg key size: ${avg_key_size} bytes"
        echo "Avg value size: ${avg_value_size} bytes"
    fi
    
    # Save to results file
    echo "$label,$sort_time,$throughput,$peak_memory,$total_io,$avg_key_size,$avg_value_size" >> "$RESULTS_DIR/summary.csv"
}

# Create CSV header
echo "Mode,Sort_Time_s,Throughput_MB_s,Peak_Memory_MB,Total_IO_MB,Avg_Key_Size,Avg_Value_Size" > "$RESULTS_DIR/summary.csv"

# Run without OVC
echo -e "${BLUE}\n>>> Running benchmark WITHOUT OVC...${NC}"

$BINARY \
    -i "$INPUT_FILE" \
    --benchmark-runs "$BENCHMARK_RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --verify \
    --run-gen-threads "$THREADS" \
    --merge-threads "$THREADS" \
    --run-size-mb "$(awk "BEGIN { printf \"%.2f\", $MEMORY_MB / $THREADS }")" \
    --merge-fanin "$(awk -v mem=$MEMORY_MB -v th=$THREADS 'BEGIN { printf "%d", mem/(th*0.0625) }')" \
    2>&1 | tee "$RESULTS_DIR/no_ovc_output.txt"

OUTPUT_NO_OVC=$(cat "$RESULTS_DIR/no_ovc_output.txt")
extract_metrics "$OUTPUT_NO_OVC" "Without OVC"

# Run with OVC
echo -e "${BLUE}\n>>> Running benchmark WITH OVC...${NC}"
$BINARY \
    -i "$INPUT_FILE" \
    --benchmark-runs "$BENCHMARK_RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --verify \
    --ovc \
    --run-gen-threads "$THREADS" \
    --merge-threads "$THREADS" \
    --run-size-mb "$(awk "BEGIN { printf \"%.2f\", $MEMORY_MB / $THREADS }")" \
    --merge-fanin "$(awk -v mem=$MEMORY_MB -v th=$THREADS 'BEGIN { printf "%d", mem/(th*0.0625) }')" \
    2>&1 | tee "$RESULTS_DIR/with_ovc_output.txt"

OUTPUT_WITH_OVC=$(cat "$RESULTS_DIR/with_ovc_output.txt")
extract_metrics "$OUTPUT_WITH_OVC" "With OVC"

# Calculate and display comparison
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}           COMPARISON SUMMARY           ${NC}"
echo -e "${GREEN}========================================${NC}"

# Extract values for comparison
NO_OVC_TIME=$(echo "$OUTPUT_NO_OVC" | grep "Sort completed in" | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/' | head -1)
OVC_TIME=$(echo "$OUTPUT_WITH_OVC" | grep "Sort completed in" | sed 's/.*Sort completed in \([0-9.]*\)s.*/\1/' | head -1)

NO_OVC_THROUGHPUT=$(echo "$OUTPUT_NO_OVC" | grep "Throughput:" | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/' | head -1)
OVC_THROUGHPUT=$(echo "$OUTPUT_WITH_OVC" | grep "Throughput:" | sed 's/.*Throughput: \([0-9.]*\) MB\/s.*/\1/' | head -1)

NO_OVC_MEMORY=$(echo "$OUTPUT_NO_OVC" | grep "Peak memory usage:" | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/' | head -1)
OVC_MEMORY=$(echo "$OUTPUT_WITH_OVC" | grep "Peak memory usage:" | sed 's/.*Peak memory usage: \([0-9.]*\) MB.*/\1/' | head -1)

NO_OVC_IO=$(echo "$OUTPUT_NO_OVC" | grep "Total I/O:" | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/' | head -1)
OVC_IO=$(echo "$OUTPUT_WITH_OVC" | grep "Total I/O:" | sed 's/.*Total I/O: \([0-9.]*\) MB.*/\1/' | head -1)

# Calculate speedup and improvements
if [ ! -z "$NO_OVC_TIME" ] && [ ! -z "$OVC_TIME" ]; then
    SPEEDUP=$(echo "scale=2; $NO_OVC_TIME / $OVC_TIME" | bc)
    TIME_REDUCTION=$(echo "scale=1; (($NO_OVC_TIME - $OVC_TIME) / $NO_OVC_TIME) * 100" | bc)
    
    echo -e "\n${YELLOW}Performance Metrics:${NC}"
    echo "  Speedup: ${SPEEDUP}x"
    echo "  Time reduction: ${TIME_REDUCTION}%"
fi

if [ ! -z "$NO_OVC_THROUGHPUT" ] && [ ! -z "$OVC_THROUGHPUT" ]; then
    THROUGHPUT_INCREASE=$(echo "scale=1; (($OVC_THROUGHPUT - $NO_OVC_THROUGHPUT) / $NO_OVC_THROUGHPUT) * 100" | bc)
    echo "  Throughput increase: ${THROUGHPUT_INCREASE}%"
fi

if [ ! -z "$NO_OVC_MEMORY" ] && [ ! -z "$OVC_MEMORY" ]; then
    MEMORY_DIFF=$(echo "scale=1; $OVC_MEMORY - $NO_OVC_MEMORY" | bc)
    MEMORY_CHANGE=$(echo "scale=1; (($OVC_MEMORY - $NO_OVC_MEMORY) / $NO_OVC_MEMORY) * 100" | bc)
    echo -e "\n${YELLOW}Resource Usage:${NC}"
    echo "  Memory difference: ${MEMORY_DIFF} MB (${MEMORY_CHANGE}%)"
fi

if [ ! -z "$NO_OVC_IO" ] && [ ! -z "$OVC_IO" ]; then
    IO_REDUCTION=$(echo "scale=1; (($NO_OVC_IO - $OVC_IO) / $NO_OVC_IO) * 100" | bc)
    echo "  I/O reduction: ${IO_REDUCTION}%"
fi

# Summary recommendation
echo -e "\n${GREEN}Recommendation:${NC}"
if [ ! -z "$SPEEDUP" ]; then
    if (( $(echo "$SPEEDUP > 1.1" | bc -l) )); then
        echo "  OVC provides significant performance improvement (${SPEEDUP}x faster)"
    elif (( $(echo "$SPEEDUP > 0.9" | bc -l) )); then
        echo "  OVC and non-OVC have similar performance"
    else
        echo "  Non-OVC performs better for this dataset"
    fi
fi

echo -e "\n${BLUE}Full results saved to: $RESULTS_DIR${NC}"
echo "  - summary.csv: Metrics comparison"
echo "  - no_ovc_output.txt: Full output without OVC"
echo "  - with_ovc_output.txt: Full output with OVC"
