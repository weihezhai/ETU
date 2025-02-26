#!/bin/bash
# Usage:
#   ./run_ppl_detail.sh [-i input_file] [-o output_file] [-m model_dir]
#
# Example:
#   ./run_ppl_detail.sh -i "my_trajectories.jsonl" -o "my_ppl_details.jsonl" -m "/path/to/model"

# Default parameter values
INPUT_FILE="/users/acr24wz/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"
OUTPUT_FILE="/users/acr24wz/ETU/preprocess/ppl/ppl_detail/ppl_details.jsonl"
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

echo "Launching perplexity detail extraction with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"

# Run the script with the provided parameters
python ppl_detail.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR"