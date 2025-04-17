#!/bin/bash
##SBATCH --partition=cpu          # CPU partition should be sufficient
#SBATCH --mem=32G                # Adjust memory if needed, depends on map sizes
#SBATCH --cpus-per-task=8        # Adjust CPU count if needed
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=ground_triples
#SBATCH --output=./logs/ground_triples_%A.out # Log file name
#SBATCH --time=0-01:00:00        # Adjust time limit (should be quick for moderate files)

# Load necessary modules
module load Anaconda3/2024.02-1

# Activate your conda environment
source activate etu

# === Configuration ===
# Define base directories
ETU_DIR="/mnt/parscratch/users/acr24wz/ETU"
SRC_DATA_DIR="/mnt/parscratch/users/acr24wz/src/src" # Or wherever your maps are
OUTPUT_DIR="$SRC_DATA_DIR/output" # Or a different output location

# Python script to execute
PYTHON_SCRIPT="$ETU_DIR/graph_to_path/ground_triples.py"

# --- INPUT/OUTPUT ---
# !!! PLEASE MODIFY THESE TWO LINES with your actual file paths !!!
INPUT_TRIPLES_FILE="$SRC_DATA_DIR/your_input_triples.json" # e.g., $SRC_DATA_DIR/a.json
GROUNDED_TRIPLES_OUTPUT_FILE="$OUTPUT_DIR/your_output_grounded_triples.json"
# --- ---

# Mapping files
ENTITIES_MAP_FILE="$SRC_DATA_DIR/entities.json"
RELATIONS_MAP_FILE="$SRC_DATA_DIR/relations.json"
ENTITY_LABELS_FILE="$SRC_DATA_DIR/entities_names.json" # Assumes same label file as before

# Optional flags (e.g., add --debug for verbose logging)
EXTRA_ARGS=""
# EXTRA_ARGS="--debug" # Uncomment this line for debug mode

# === Execution ===
echo "Starting triple grounding process..."

# Check if input files exist
if [ ! -f "$INPUT_TRIPLES_FILE" ]; then echo "Error: Input triples file not found: $INPUT_TRIPLES_FILE"; exit 1; fi
if [ ! -f "$ENTITIES_MAP_FILE" ]; then echo "Error: Entities map file not found: $ENTITIES_MAP_FILE"; exit 1; fi
if [ ! -f "$RELATIONS_MAP_FILE" ]; then echo "Error: Relations map file not found: $RELATIONS_MAP_FILE"; exit 1; fi
if [ ! -f "$ENTITY_LABELS_FILE" ]; then echo "Error: Entity labels file not found: $ENTITY_LABELS_FILE"; exit 1; fi

# Ensure output directory exists
mkdir -p "$(dirname "$GROUNDED_TRIPLES_OUTPUT_FILE")"

# Run the Python script
echo "Running Python script: $PYTHON_SCRIPT"
python "$PYTHON_SCRIPT" \
    --input_triples_file "$INPUT_TRIPLES_FILE" \
    --entities_map_file "$ENTITIES_MAP_FILE" \
    --relations_map_file "$RELATIONS_MAP_FILE" \
    --entity_labels_file "$ENTITY_LABELS_FILE" \
    --output_file "$GROUNDED_TRIPLES_OUTPUT_FILE" \
    $EXTRA_ARGS

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "Triple grounding script finished successfully."
    echo "Grounded triples saved to: $GROUNDED_TRIPLES_OUTPUT_FILE"
else
    echo "Error: Triple grounding script failed."
    exit 1
fi

echo "Done." 