#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="extract_filtered_path.py"
INPUT_DIR="/Users/rickzhai/Documents/GitHub/ETU/ETU/similarity/topsim/topk_relation_filtered_paths" # Directory containing the topK...jsonl files. Use "." for current directory.
OUTPUT_DIR="extracted_filtered_paths" # Name of the folder to save the output .json files
# --- End Configuration ---

# Check if python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create output directory '$OUTPUT_DIR'."
    exit 1
fi

echo "Starting processing..."

# Find files matching the pattern in the input directory
# Use find for better handling of filenames with spaces or special characters
# -maxdepth 1 ensures we only look in the immediate INPUT_DIR, not subdirectories
find "$INPUT_DIR" -maxdepth 1 -name 'top*_sim_filtered_paths.jsonl' -print0 | while IFS= read -r -d $'\0' file; do
    # Extract filename from the full path
    filename=$(basename "$file")

    # Extract the number (k) between "top" and "_sim" using parameter expansion or sed
    # Using parameter expansion (requires bash)
    temp="${filename#top}"     # Remove "top" prefix
    k_value="${temp%%_sim*}"   # Remove "_sim..." suffix

    # Basic validation: Check if k_value is a number
    if ! [[ "$k_value" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not extract a valid numeric k value from filename '$filename'. Skipping."
        continue
    fi

    echo "Processing '$filename' (k=$k_value)..."

    # Run the python script
    python "$PYTHON_SCRIPT" --k_value "$k_value" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "Error processing file '$filename'. Check Python script output for details."
        # Decide if you want to stop on error or continue with next file
        # exit 1 # Uncomment to stop on first error
    fi
done

echo "Processing finished. Output files are in '$OUTPUT_DIR'."

exit 0