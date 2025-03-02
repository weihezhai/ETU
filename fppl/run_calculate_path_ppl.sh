#!/bin/bash
# Usage:
#   ./run_calculate_path_ppl.sh [-i input_file] [-o output_file] [-m model_dir] [-f prompt_format]
#
# Example:
#   ./run_calculate_path_ppl.sh -i "no_middle_entity.jsonl" -o "path_ppl_scores_integrated.jsonl" -m "/path/to/model" -f "integrated"

# Default parameter values
INPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/fppl/no_middle_entity.jsonl"
OUTPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/fppl/path_ppl_scores.jsonl"
MODEL_DIR="$model/qwen2.5/7b_ori"
PROMPT_FORMAT="path_then_question"  # Default format

# Parse command-line arguments
while getopts "i:o:m:f:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    f) PROMPT_FORMAT="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

echo "Launching path perplexity calculation with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"
echo "  Prompt Format:            $PROMPT_FORMAT"

# Run the script with the provided parameters
python calculate_path_ppl.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR" \
    --prompt_format "$PROMPT_FORMAT" 