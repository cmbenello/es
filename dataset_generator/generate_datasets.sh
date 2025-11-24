#!/bin/bash

# Script to generate GenSort datasets of various sizes
# Each record is 100 bytes

# Create output directory if it doesn't exist
OUTPUT_DIR="./gensort_data"
mkdir -p "$OUTPUT_DIR"

# Path to gensort binary
GENSORT="./dataset_generator/gensort"

# Number of threads to use (default to number of CPU cores)
THREADS=$(nproc)

# Dataset sizes to generate (in GiB)
# DATA_SIZES=(10 20 40 60 80 100)  # Original sizes
DATA_SIZES=(2 4 6 8 10)  # Current sizes

# Function to calculate number of records for given size in GiB
calculate_records() {
    local size_gib=$1
    local bytes=$((size_gib * 1024 * 1024 * 1024))
    local records=$((bytes / 100))
    echo $records
}

# Generate datasets
for size in "${DATA_SIZES[@]}"; do
    records=$(calculate_records $size)
    output_file="$OUTPUT_DIR/gensort_${size}GiB.data"
    
    echo "Generating ${size} GiB dataset..."
    echo "  Records: $records"
    echo "  Output: $output_file"
    echo "  Threads: $THREADS"
    echo "  Command: $GENSORT -t$THREADS $records $output_file"
    echo ""
    
    # Uncomment the line below to actually run the generation
    $GENSORT -t$THREADS $records "$output_file"
done

echo "Dataset generation complete."
echo "Using $THREADS threads for generation (detected from CPU cores)."
echo "Generated datasets: ${DATA_SIZES[*]} GiB"
echo ""
echo "You can also override the thread count by setting THREADS environment variable:"
echo "  THREADS=8 ./generate_datasets.sh"