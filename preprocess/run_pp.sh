#!/bin/bash
# Usage:
#   ./run_pp.sh [-i input_file] [-o output_file] [-k top_k]
#
# Example:
#   ./run_pp.sh -i "my_trajectories.jsonl" -o "my_filtered.jsonl" -k 5

# Default parameter values
INPUT_FILE="/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"
OUTPUT_FILE="/data/home/mpx602/projects/ETU/ETU/preprocess/topk_path/topk_ppl_path.jsonl"
TOP_K=5

# Parse command-line arguments
while getopts "i:o:k:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    k) TOP_K="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "Launching preprocessing with the following parameters:"
echo "  Input File:               $INPUT_FILE"
echo "  Output File:              $OUTPUT_FILE"
echo "  Top-k Paths:              $TOP_K"

# Run the preprocessing script with the provided parameters
python ETU/ETU/preprocess/pp.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --k "$TOP_K" 