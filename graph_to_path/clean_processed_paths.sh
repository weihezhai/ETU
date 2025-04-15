#!/bin/bash

# === Configuration ===
# Input file: The JSON output from process_subgraph.py
INPUT_JSON="/mnt/parscratch/users/acr24wz/src/src/output/train_output.json"

# Output file: Where to save the cleaned JSON data
OUTPUT_JSON="/mnt/parscratch/users/acr24wz/src/src/output/train_output_cleaned.json"

# Python script to execute
PYTHON_SCRIPT="/mnt/parscratch/users/acr24wz/ETU/graph_to_path/clean_processed_paths.py"

# Optional flags (e.g., add --debug for verbose logging)
EXTRA_ARGS=""
# EXTRA_ARGS="--debug" # Uncomment this line for debug mode

# === Execution ===
echo "Starting path cleaning process..."

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file not found at '$INPUT_JSON'"
    exit 1
fi

# Run the Python script
python "$PYTHON_SCRIPT" \
    --input_file "$INPUT_JSON" \
    --output_file "$OUTPUT_JSON" \
    $EXTRA_ARGS

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "Path cleaning script finished successfully."
    echo "Cleaned data saved to: $OUTPUT_JSON"
else
    echo "Error: Path cleaning script failed."
    exit 1
fi

echo "Done."