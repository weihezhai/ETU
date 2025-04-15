#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=train_subgraph_pruning
#SBATCH --output=./logs/train_subgraph_pruning_%A_%a.out
#SBATCH --time=0-24:00:00
##SBATCH --array=0-8  # Process 9 files (indices 0-8)

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Define file paths relative to the script's location or workspace root
SCRIPT_DIR="/mnt/parscratch/users/acr24wz/ETU/graph_to_path"
SRC_DIR="/mnt/parscratch/users/acr24wz/src/src" # Assumes the script is in graph_to_path, adjust if needed
PYTHON_SCRIPT="$SCRIPT_DIR/process_subgraph.py"
SUBGRAPH_FILE="$SRC_DIR/train_subgraph.json"
KB_MAP_FILE="$SRC_DIR/entities.json"
GOLDEN_RELS_FILE="$SRC_DIR/gold_relations.json"
OUTPUT_FILE="$SRC_DIR/output/train_output.json"
MAX_HOPS=4


echo "Running the subgraph processing script..."

# Execute the Python script
python "$PYTHON_SCRIPT" \
  --subgraph_file "$SUBGRAPH_FILE" \
  --kb_map_file "$KB_MAP_FILE" \
  --golden_rels_file "$GOLDEN_RELS_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_hops "$MAX_HOPS" 
  # --debug

# Optional: Clean up the temporary map file
# echo "Cleaning up temporary KB ID map file..."
# rm "$KB_MAP_FILE"

echo "Script finished. Output saved to $OUTPUT_FILE" 