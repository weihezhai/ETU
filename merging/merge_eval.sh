#!/bin/bash

# Define file paths
# These should be adapted to your actual file locations
# FILE1="/data/home/mpx602/projects/ETU/ETU/info_gain/results/top20/avg_prob_metrics_top25.json"
# FILE2="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_metrics/15/top15_sim_filtered_paths_llm_results_cleaned_metrics.json"
# MERGED_RAW_OUTPUT_DIR="merged_results" # Directory to store the raw merged output
# MERGED_RAW_FILE="$MERGED_RAW_OUTPUT_DIR/merged_predictions_raw.json"
# FINAL_EVAL_OUTPUT_DIR="$MERGED_RAW_OUTPUT_DIR/evaluation" # Directory for final evaluation output
# FINAL_EVAL_FILE="$FINAL_EVAL_OUTPUT_DIR/merged_evaluated_metrics.json"

# # This should be the path to your ground truth file
# # Update this path accordingly
# GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
# # Example: GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/path_to_ground_truth/webqsp_test.json"

# # Path to the merging script
# MERGE_SCRIPT_PATH="merge.py" # Assuming it's in the same directory or in PATH

# # Path to the evaluation script (from your eval.sh)
# EVAL_SCRIPT_PATH="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluate_results.py"

# # Create output directories if they don't exist
# mkdir -p "$MERGED_RAW_OUTPUT_DIR"
# mkdir -p "$FINAL_EVAL_OUTPUT_DIR"

# echo "Starting merging process..."
# # Step 1: Merge the prediction files
# python3 "$MERGE_SCRIPT_PATH" \
#   --file1 "$FILE1" \
#   --file2 "$FILE2" \
#   --output "$MERGED_RAW_FILE"

# if [ $? -ne 0 ]; then
#   echo "Merging failed. Exiting."
#   exit 1
# fi
# echo "Merging complete. Raw merged results saved to $MERGED_RAW_FILE"

# echo "Starting evaluation of merged results..."
# # Step 2: Evaluate the merged predictions
# # The evaluate_results.py script expects the --cleaned argument for the predictions
# # and --ground-truth for the ground truth.
# python3 "$EVAL_SCRIPT_PATH" \
#   --cleaned "$MERGED_RAW_FILE" \
#   --ground-truth "$GROUND_TRUTH" \
#   --output "$FINAL_EVAL_FILE"

# if [ $? -ne 0 ]; then
#   echo "Evaluation failed. Exiting."
#   exit 1
# fi

# echo "Evaluation complete. Final evaluated results saved to $FINAL_EVAL_FILE"
# echo "Done."
#!/bin/bash

# Set paths - adjust these as needed
FILE1="/data/home/mpx602/projects/ETU/ETU/info_gain/results/processed_combined_evaluation_top20.json"
FILE2="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/last_entity_ans_path_ordered_by_filtered_rel_sim/filter_top0_1/processed_answers_top20_filt0.json"
GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
OUTPUT_DIR="/data/home/mpx602/projects/ETU/ETU/merging/"

# Check if all arguments provided
if [ -z "$FILE1" ] || [ -z "$FILE2" ] || [ -z "$GROUND_TRUTH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: ./merge_eval.sh file1.json file2.json ground_truth.json output_dir"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Merge the prediction files
MERGED_FILE="$OUTPUT_DIR/merged_results.json"
echo "Merging prediction files..."
python3 merge2.py --file1 "$FILE1" --file2 "$FILE2" --output "$MERGED_FILE"

# Step 2: Evaluate the merged results
echo "Evaluating merged results..."
python3 /data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluate_results.py \
  --cleaned "$MERGED_FILE" \
  --ground-truth "$GROUND_TRUTH" \
  --output "$OUTPUT_DIR/merged_metrics.json"

echo "Merge and evaluation complete. Results saved to $OUTPUT_DIR/merged_metrics.json"