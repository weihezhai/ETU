#!/bin/bash

# Define file paths relative to the script's location or workspace root
SCRIPT_DIR=$(dirname "$0")
BASE_DIR="$SCRIPT_DIR/.." # Assumes the script is in graph_to_path, adjust if needed
PYTHON_SCRIPT="$SCRIPT_DIR/process_subgraph.py"
SUBGRAPH_FILE="$SCRIPT_DIR/debug.json"
KB_MAP_FILE="$SCRIPT_DIR/debug_kb_map.json"
GOLDEN_RELS_FILE="$SCRIPT_DIR/debuggold.json"
OUTPUT_FILE="$SCRIPT_DIR/debug_output.json"
MAX_HOPS=4


echo "Running the subgraph processing script..."

# Execute the Python script
python "$PYTHON_SCRIPT" \
  --subgraph_file "$SUBGRAPH_FILE" \
  --kb_map_file "$KB_MAP_FILE" \
  --golden_rels_file "$GOLDEN_RELS_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_hops "$MAX_HOPS" \
  --debug

# Optional: Clean up the temporary map file
# echo "Cleaning up temporary KB ID map file..."
# rm "$KB_MAP_FILE"

echo "Script finished. Output saved to $OUTPUT_FILE" 