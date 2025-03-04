#!/bin/bash
# Usage:
#   ./run_calculate_path_similarity.sh [-i input_file] [-o output_file] [-m model_dir]
#
# Example:
#   ./run_calculate_path_similarity.sh -i "path_ppl_scores.jsonl" -o "path_similarity_scores.jsonl" -m "/path/to/model"

# Default parameter values
INPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/fppl/path_ppl_scores_explicit_reasoning.jsonl"
OUTPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/similarity/path_similarity_scores.jsonl"
MODEL_DIR="/mnt/parscratch/users/acr24wz/etu/qwen2.5/gte_qwen2_7b_instruct"

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

echo "Launching path similarity calculation with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"

# Run the script with the provided parameters
python sim_question_path.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR"