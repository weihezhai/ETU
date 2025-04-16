#!/bin/bash
##SBATCH --partition=gpu          # Or a suitable CPU partition if preferred
##SBATCH --qos=gpu                # Match partition QOS
##SBATCH --gres=gpu:1             # Might not need GPU, but keeps consistency
#SBATCH --mem=82G                # Adjust memory if needed, grounding might be memory intensive with large maps
#SBATCH --cpus-per-task=12       # Adjust CPU count if needed
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=train_path_grounding
#SBATCH --output=./logs/train_path_grounding_%A.out # Use %A for job ID, no array index needed
#SBATCH --time=0-08:00:00        # Adjust time limit as needed (grounding should be faster)

# Load necessary modules
module load Anaconda3/2024.02-1
# module load cuDNN/8.9.2.26-CUDA-12.1.1 # Likely not needed for this script

# Activate your conda environment
source activate etu

# === Configuration ===
# Define base directories
ETU_DIR="/mnt/parscratch/users/acr24wz/ETU"
SRC_DATA_DIR="/mnt/parscratch/users/acr24wz/src/src"
OUTPUT_DIR="$SRC_DATA_DIR/output"

# Python script to execute
PYTHON_SCRIPT="$ETU_DIR/graph_to_path/ground_paths.py"

# Input files
CLEANED_PATHS_FILE="$OUTPUT_DIR/train_output_cleaned.json"
ENTITIES_MAP_FILE="$SRC_DATA_DIR/entities.json"
RELATIONS_MAP_FILE="$SRC_DATA_DIR/relations.json"
ENTITY_LABELS_FILE="$SRC_DATA_DIR/CWQ_all_label_map.json" # Ensure this path is correct

# Output file
GROUNDED_OUTPUT_FILE="$OUTPUT_DIR/train_output_grounded.json"

# Optional flags (e.g., add --debug for verbose logging)
EXTRA_ARGS=""
# EXTRA_ARGS="--debug" # Uncomment this line for debug mode

# === Execution ===
echo "Starting path grounding process..."

# Check if input files exist
if [ ! -f "$CLEANED_PATHS_FILE" ]; then echo "Error: Cleaned paths file not found: $CLEANED_PATHS_FILE"; exit 1; fi
if [ ! -f "$ENTITIES_MAP_FILE" ]; then echo "Error: Entities map file not found: $ENTITIES_MAP_FILE"; exit 1; fi
if [ ! -f "$RELATIONS_MAP_FILE" ]; then echo "Error: Relations map file not found: $RELATIONS_MAP_FILE"; exit 1; fi
if [ ! -f "$ENTITY_LABELS_FILE" ]; then echo "Error: Entity labels file not found: $ENTITY_LABELS_FILE"; exit 1; fi

# Ensure output directory exists (Python script also does this, but good practice)
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Running Python script: $PYTHON_SCRIPT"
python "$PYTHON_SCRIPT" \
    --cleaned_paths_file "$CLEANED_PATHS_FILE" \
    --entities_map_file "$ENTITIES_MAP_FILE" \
    --relations_map_file "$RELATIONS_MAP_FILE" \
    --entity_labels_file "$ENTITY_LABELS_FILE" \
    --output_file "$GROUNDED_OUTPUT_FILE" \
    $EXTRA_ARGS

# Check the exit status of the Python script
if [ $? -eq 0 ]; then
    echo "Path grounding script finished successfully."
    echo "Grounded data saved to: $GROUNDED_OUTPUT_FILE"
else
    echo "Error: Path grounding script failed."
    exit 1
fi

echo "Done." 