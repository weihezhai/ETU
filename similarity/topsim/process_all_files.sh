#!/bin/bash

# Create a directory for processed files
mkdir -p /data/home/mpx602/projects/ETU/ETU/similarity/topsim/topk_paths_prompts_rog

# Process all the filtered paths jsonl files
for file in /data/home/mpx602/projects/ETU/ETU/similarity/topsim/topk_paths/top*_sim_filtered_paths.jsonl; do
  filename=$(basename "$file")
  output="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/topk_paths_prompts_rog/${filename%.jsonl}_with_prompts.jsonl"
  
  echo "Processing $file -> $output"
  python /data/home/mpx602/projects/ETU/ETU/similarity/topsim/generate_prompts.py --input "$file" --output "$output"
done

echo "All files processed successfully!" 