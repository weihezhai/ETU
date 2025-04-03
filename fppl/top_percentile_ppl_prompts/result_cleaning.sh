#!/bin/bash

# Process all JSON files in a folder
# Usage: ./process_json_files.sh --input-dir INPUT_DIR --output-dir OUTPUT_DIR [--top-k TOP_K]

# Default value for top-k
TOP_K=15
INPUT_DIR="/data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/evaluation_results/75"
OUTPUT_DIR="/data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/evaluation_results_cleaned/75"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir|-i)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --top-k|-k)
      TOP_K="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: ./process_json_files.sh --input-dir INPUT_DIR --output-dir OUTPUT_DIR [--top-k TOP_K]"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each JSON file in the input directory
for input_file in "$INPUT_DIR"/*.json; do
  # Get the base filename
  filename=$(basename "$input_file")
  
  # Create output file path
  output_file="$OUTPUT_DIR/${filename%.json}_cleaned.json"
  
  echo "Processing $filename..."
  
  # Run the Python script
  python3 /data/home/mpx602/projects/ETU/ETU/similarity/topsim/result_cleaning.py \
    --input "$input_file" \
    --output "$output_file" \
    --top-k "$TOP_K"
done

echo "All files processed."