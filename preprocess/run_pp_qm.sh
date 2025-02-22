#!/bin/bash
# Usage:
#   ./run_pp.sh [-i input_file] [-o output_file] [-k top_k] [-m model_dir]
#
# Example:
#   ./run_pp.sh -i "my_trajectories.jsonl" -o "my_filtered.jsonl" -k 5 -m "/path/to/model"

# Default parameter values
INPUT_FILE="/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"
OUTPUT_FILE="/data/home/mpx602/projects/ETU/ETU/preprocess/topk_path/topk_ppl_path.jsonl"
TOP_K=5
MODEL_DIR="/data/scratch/mpx602/ETU/qwen2.5/7b_ori"

# Parse command-line arguments
while getopts "i:o:k:m:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    k) TOP_K="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Create output directory if it doesn't exist
# mkdir -p "$(dirname "$OUTPUT_FILE")"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

echo "Launching preprocessing with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Top-k Paths:              $TOP_K"
echo "  Model Directory:          $MODEL_DIR"

# Run the preprocessing script with the provided parameters
python ETU/ETU/preprocess/pp.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --k "$TOP_K" \
    --model_dir "$MODEL_DIR" 