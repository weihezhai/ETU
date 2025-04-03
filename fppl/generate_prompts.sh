#!/bin/bash

# Create a directory for processed files
mkdir -p /data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/25

# Process all the filtered paths jsonl files in the folder
for file in /data/home/mpx602/projects/ETU/ETU/fppl/res_filtered_percent/25/*.jsonl; do
  filename=$(basename "$file")
  output="/data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/25/${filename%.jsonl}_with_prompts.jsonl"
  
  echo "Processing $file -> $output"
  python /data/home/mpx602/projects/ETU/ETU/fppl/generate_prompts.py --input "$file" --output "$output"
done

echo "All files processed successfully!" 