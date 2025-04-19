#!/bin/bash
##SBATCH --partition=gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1
#SBATCH --mem=156G
#SBATCH --cpus-per-task=12  # Request CPUs for SLURM
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=proc_subgraph_mc # Updated job name slightly
#SBATCH --output=./logs/proc_subgraph_mc_%A_%a.out # Updated log name slightly
#SBATCH --time=0-24:00:00
##SBATCH --array=0-8

# Load necessary modules
module load Anaconda3/2024.02-1
# module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Define file paths relative to the script's location or workspace root
SCRIPT_DIR="/mnt/parscratch/users/acr24wz/ETU/graph_to_path"
SRC_DIR="/mnt/parscratch/users/acr24wz/src/src" # Assumes the script is in graph_to_path, adjust if needed
PYTHON_SCRIPT="$SCRIPT_DIR/process_subgraph.py" # This script is now multicore capable
SUBGRAPH_FILE="$SRC_DIR/train_subgraph.json"
KB_MAP_FILE="$SRC_DIR/entities.json" # Maps KB ID ('Qxxx') -> integer ID
GOLDEN_RELS_FILE="$SRC_DIR/gold_relations.json"
OUTPUT_DIR="$SRC_DIR/output_mc" # Changed output directory name slightly
OUTPUT_FILE="$OUTPUT_DIR/train_output_mc.json" # Changed output filename slightly
MAX_HOPS=4

# --- NEW: Configure number of workers ---
# Let the Python script determine the default based on SLURM_CPUS_PER_TASK or os.cpu_count()
# Or, uncomment and set a specific number:
NUM_WORKERS=8

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running the multicore subgraph processing script..."

# Construct the command
CMD="python \"$PYTHON_SCRIPT\" \
  --subgraph_file \"$SUBGRAPH_FILE\" \
  --kb_map_file \"$KB_MAP_FILE\" \
  --golden_rels_file \"$GOLDEN_RELS_FILE\" \
  --output_file \"$OUTPUT_FILE\" \
  --max_hops \"$MAX_HOPS\""

# --- Add the num_workers argument ---
# Check if NUM_WORKERS variable is set and not empty
if [ -n "$NUM_WORKERS" ]; then
  CMD+=" --num_workers \"$NUM_WORKERS\""
  echo "Using specified number of workers: $NUM_WORKERS"
else
  echo "Using default number of workers (determined by script based on SLURM/CPU count)."
  # No need to add --num_workers if we want the script's default logic to apply
fi

# --- Add debug flag if needed ---
# Uncomment the line below to enable debug mode in the Python script
# CMD+=" --debug"

# Print the command being executed
echo "Executing command:"
echo "$CMD"
echo "---"

# Execute the command
eval "$CMD"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "---"
  echo "Script finished successfully."
else
  echo "---"
  echo "Script finished with errors (Exit Code: ${EXIT_CODE})."
fi

echo "Output saved to $OUTPUT_FILE"
exit $EXIT_CODE