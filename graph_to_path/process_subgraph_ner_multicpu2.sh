#!/bin/bash
##SBATCH --partition=gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1 # NOTE: This script is CPU-bound, GPU resources will likely be unused.
#SBATCH --mem=156G
#SBATCH --cpus-per-task=12  # Requesting 12 CPUs for the task
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=train_subgraph_mp # Changed job name (removed 'ner')
#SBATCH --output=./logs/train_subgraph_mp_%A_%a.out # Changed log name (removed 'ner')
#SBATCH --time=0-24:00:00
##SBATCH --array=0-8  # Process 9 files (indices 0-8) - Keep if you process multiple files

# Load necessary modules
module load Anaconda3/2024.02-1
# module load cuDNN/8.9.2.26-CUDA-12.1.1 # Removed as GPU/cuDNN not needed

# Activate your conda environment
source activate etu

# Define file paths relative to the script's location or workspace root
# *** IMPORTANT: Update PYTHON_SCRIPT if you saved the modified code to a new file ***
# SCRIPT_DIR="/mnt/parscratch/users/acr24wz/ETU/graph_to_path" # Example path
# SRC_DIR="/mnt/parscratch/users/acr24wz/src/src"             # Example path
SCRIPT_DIR="/mnt/parscratch/users/acr24wz/ETU/graph_to_path" # Assuming script is run from workspace root
SRC_DIR="/mnt/parscratch/users/acr24wz/src/src" # Example: Adjust if your data is elsewhere
# --- Using the potentially new script name ---
PYTHON_SCRIPT="$SCRIPT_DIR/process_subgraph_ner_multicpu2.py" # *** ADJUST IF FILENAME IS DIFFERENT ***
# --- Input files ---
SUBGRAPH_FILE="$SRC_DIR/train_subgraph.json"
KB_MAP_FILE="$SRC_DIR/entities.json" # Maps KB ID ('Qxxx') -> integer ID
# ENTITY_LABELS_FILE="$SRC_DIR/entities_names.json" # REMOVED - No longer needed
GOLDEN_RELS_FILE="$SRC_DIR/gold_relations.json"
# --- Output ---
OUTPUT_DIR="$SRC_DIR/output" # Define output directory
OUTPUT_FILE="$OUTPUT_DIR/train_output_mp.json" # Changed output filename (removed 'ner')
# --- Parameters ---
MAX_HOPS=4
# SPACY_MODEL="en_core_web_lg" # REMOVED - No longer needed

# Use the number of CPUs allocated by SLURM for the number of workers
NUM_WORKERS=8 # Default to 4 if SLURM_CPUS_PER_TASK is not set

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Updated echo statements
echo "Running the subgraph processing script (NO NER) using $NUM_WORKERS workers..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM CPUs Per Task: $SLURM_CPUS_PER_TASK"
echo "Number of Workers: $NUM_WORKERS"
echo "Input Subgraph: $SUBGRAPH_FILE"
echo "Output File: $OUTPUT_FILE"
# echo "Spacy Model: $SPACY_MODEL" # REMOVED

# Execute the Python script with updated arguments
python "$PYTHON_SCRIPT" \
  --subgraph_file "$SUBGRAPH_FILE" \
  --kb_map_file "$KB_MAP_FILE" \
  --golden_rels_file "$GOLDEN_RELS_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_hops "$MAX_HOPS" \
  --num_workers "$NUM_WORKERS"
  # --spacy_model "$SPACY_MODEL" \ # REMOVED argument
  # --debug # Uncomment for detailed logging (can be very verbose)

echo "Script finished. Output saved to $OUTPUT_FILE"