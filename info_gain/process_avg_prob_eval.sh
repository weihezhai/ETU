#!/bin/bash

# Define input and output files
INPUT_FILE="/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/averaged_results_filtered_by_relation_top20.json"
PROCESSED_FILE="/data/home/mpx602/projects/ETU/ETU/info_gain/results/processed_combined_evaluation_top20.json"
GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
OUTPUT_DIR="/data/home/mpx602/projects/ETU/ETU/info_gain/results/top20"
TOPK=25  # Default value for top-k paths to evaluate

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topk|-k)
      TOPK="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the processing script
echo "Processing averaged_results.json to extract last entities as answers (top-$TOPK)..."
python3 /data/home/mpx602/projects/ETU/ETU/info_gain/process_average_prob_evaluation.py --input "$INPUT_FILE" --output "$PROCESSED_FILE" --topk "$TOPK"

# Run the evaluation script
echo "Evaluating results against ground truth..."
python3 /data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluate_results.py \
  --cleaned "$PROCESSED_FILE" \
  --ground-truth "$GROUND_TRUTH" \
  --output "$OUTPUT_DIR/avg_prob_metrics_top${TOPK}.json"

echo "Evaluation complete. Results saved to $OUTPUT_DIR/avg_prob_metrics_top${TOPK}.json"