#!/bin/bash

# Path to the local Llama 3.1 8B model
MODEL_PATH="/path/to/llama-3.1-8b-instruct"  # Replace with your actual model path

# Create a directory for results
mkdir -p /data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_results

# Process all the prompt files
for file in /data/home/mpx602/projects/ETU/ETU/similarity/topsim/topk_paths_prompts/top*_sim_filtered_paths_with_prompts.jsonl; do
  filename=$(basename "$file")
  base_filename="${filename%_with_prompts.jsonl}"
  output="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_results/${base_filename}_llm_results.json"
  
  echo "Evaluating $file -> $output"
  
  # Optional: limit the number of samples for testing
  # MAX_SAMPLES="--max-samples 10"
  MAX_SAMPLES=""
  
  python /data/home/mpx602/projects/ETU/ETU/similarity/topsim/test_prompts_with_llm.py \
    --model "$MODEL_PATH" \
    --input "$file" \
    --output "$output" \
    --batch-size 1 \
    $MAX_SAMPLES
done

echo "All evaluations completed successfully!"