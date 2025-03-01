#!/bin/bash
# Usage:
#   ./run_calculate_path_ppl.sh [-i input_file] [-o output_file] [-m model_dir]
#
# Example:
#   ./run_calculate_path_ppl.sh -i "no_middle_entity.jsonl" -o "path_ppl_scores.jsonl" -m "/path/to/model"

# Default parameter values
INPUT_FILE="/users/acr24wz/ETU/fppl/no_middle_entity.jsonl"
OUTPUT_FILE="/users/acr24wz/ETU/fppl/path_ppl_scores.jsonl"
MODEL_DIR="$model/qwen2.5/7b_ori"

# Parse command-line arguments
while getopts "i:o:m:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

echo "Launching path perplexity calculation with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"

# Run the script with the provided parameters
python calculate_path_ppl.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR" 