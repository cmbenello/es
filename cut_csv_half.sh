#!/bin/bash

# Script to cut a CSV file in half based on size and return the top half
# Ensures the last line is complete and valid

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_csv> <output_csv>"
    echo "Example: $0 input.csv output_half.csv"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get the total file size in bytes
FILE_SIZE=$(stat -f%z "$INPUT_FILE" 2>/dev/null || stat -c%s "$INPUT_FILE" 2>/dev/null)
HALF_SIZE=$((FILE_SIZE / 2))

echo "File size: $FILE_SIZE bytes"
echo "Target half size: $HALF_SIZE bytes"

# Read up to approximately half the file, then find the next complete line
# Use head with byte count and ensure we end on a complete line
head -c "$HALF_SIZE" "$INPUT_FILE" > "${OUTPUT_FILE}.tmp"

# Add one more complete line after the cut point to ensure we don't have a partial line
# Then remove the last incomplete line if any
if [ -s "${OUTPUT_FILE}.tmp" ]; then
    # Check if the last character is a newline
    LAST_CHAR=$(tail -c 1 "${OUTPUT_FILE}.tmp" | od -An -tx1 | tr -d ' ')

    if [ "$LAST_CHAR" != "0a" ]; then
        # Last line is incomplete, remove it
        echo "Removing incomplete last line..."
        head -n -1 "${OUTPUT_FILE}.tmp" > "$OUTPUT_FILE"
    else
        # Last line is complete
        mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"
    fi

    rm -f "${OUTPUT_FILE}.tmp"

    # Get actual output size (avoid slow operations like wc on large files)
    OUTPUT_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)

    echo "Output file: $OUTPUT_FILE"
    echo "Output size: $OUTPUT_SIZE bytes ($(awk "BEGIN {printf \"%.2f\", $OUTPUT_SIZE/$FILE_SIZE*100}")% of original)"
else
    echo "Error: Failed to create output file"
    rm -f "${OUTPUT_FILE}.tmp"
    exit 1
fi

echo "Done!"
