#!/bin/bash

# Evaluate cleaned results against ground truth
# Usage: ./run_evaluation.sh --cleaned-dir CLEANED_DIR --ground-truth GROUND_TRUTH --output-dir OUTPUT_DIR

# Default values
CLEANED_DIR="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_results_cleaned"
GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
OUTPUT_DIR="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_metrics"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleaned-dir|-c)
      CLEANED_DIR="$2"
      shift 2
      ;;
    --ground-truth|-g)
      GROUND_TRUTH="$2"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$CLEANED_DIR" ] || [ -z "$GROUND_TRUTH" ]; then
  echo "Usage: ./run_evaluation.sh --cleaned-dir CLEANED_DIR --ground-truth GROUND_TRUTH [--output-dir OUTPUT_DIR]"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if cleaned directory contains subdirectories
if [ -d "$CLEANED_DIR" ] && [ "$(find "$CLEANED_DIR" -mindepth 1 -type d | wc -l)" -gt 0 ]; then
  # Process each subdirectory (for different top-k values)
  for k_dir in "$CLEANED_DIR"/*; do
    if [ -d "$k_dir" ]; then
      k_value=$(basename "$k_dir")
      echo "Processing directory for k=$k_value..."
      
      # Create corresponding output directory
      k_output_dir="$OUTPUT_DIR/$k_value"
      mkdir -p "$k_output_dir"
      
      # Process each JSON file in this k directory
      for cleaned_file in "$k_dir"/*.json; do
        if [ -f "$cleaned_file" ]; then
          # Get the base filename
          filename=$(basename "$cleaned_file")
          base_name="${filename%.json}"
          
          # Create output file path
          output_file="$k_output_dir/${base_name}_metrics.json"
          
          echo "Evaluating $filename..."
          
          # Run the Python evaluation script
          python3 /data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluate_results.py \
            --cleaned "$cleaned_file" \
            --ground-truth "$GROUND_TRUTH" \
            --output "$output_file"
        fi
      done
    fi
  done
else
  # Process all JSON files in the cleaned directory directly
  for cleaned_file in "$CLEANED_DIR"/*.json; do
    if [ -f "$cleaned_file" ]; then
      # Get the base filename
      filename=$(basename "$cleaned_file")
      base_name="${filename%.json}"
      
      # Create output file path
      output_file="$OUTPUT_DIR/${base_name}_metrics.json"
      
      echo "Evaluating $filename..."
      
      # Run the Python evaluation script
      python3 /data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluate_results.py \
        --cleaned "$cleaned_file" \
        --ground-truth "$GROUND_TRUTH" \
        --output "$output_file"
    fi
  done
fi

echo "All evaluations complete. Results saved in $OUTPUT_DIR" 