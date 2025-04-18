#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1 # Note: NER/Graph processing is CPU-bound, GPU might be underutilized unless Spacy uses it.
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12  # Requesting 12 CPUs for the task
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=train_subgraph_ground_truth_ner_mp # Changed job name slightly for clarity
#SBATCH --output=./logs/train_subgraph_ground_truth_ner_mp_%A_%a.out # Changed log name slightly
#SBATCH --time=0-24:00:00
##SBATCH --array=0-8  # Process 9 files (indices 0-8)

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1 # May not be strictly necessary if Spacy doesn't use GPU heavily

# Activate your conda environment
source activate etu

# Define file paths relative to the script's location or workspace root
SCRIPT_DIR="/mnt/parscratch/users/acr24wz/ETU/graph_to_path"
SRC_DIR="/mnt/parscratch/users/acr24wz/src/src"
PYTHON_SCRIPT="$SCRIPT_DIR/process_subgraph_ner.py" # Make sure this matches the python script filename
SUBGRAPH_FILE="$SRC_DIR/train_subgraph.json"
KB_MAP_FILE="$SRC_DIR/entities.json" # Maps KB ID ('Qxxx') -> integer ID
ENTITY_LABELS_FILE="$SRC_DIR/entities_names.json" # Maps KB ID 'Qxxx' -> 'Label'
GOLDEN_RELS_FILE="$SRC_DIR/gold_relations.json"
OUTPUT_DIR="$SRC_DIR/output" # Define output directory
OUTPUT_FILE="$OUTPUT_DIR/train_output_with_ner_mp.json" # Changed output filename slightly
MAX_HOPS=4
SPACY_MODEL="en_core_web_lg" # Define the Spacy model name

# Use the number of CPUs allocated by SLURM for the number of workers
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-4} # Default to 4 if SLURM_CPUS_PER_TASK is not set

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running the subgraph processing script with NER using $NUM_WORKERS workers..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM CPUs Per Task: $SLURM_CPUS_PER_TASK"
echo "Number of Workers: $NUM_WORKERS"
echo "Input Subgraph: $SUBGRAPH_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Spacy Model: $SPACY_MODEL"

# Execute the Python script with multiprocessing arguments
python "$PYTHON_SCRIPT" \
  --subgraph_file "$SUBGRAPH_FILE" \
  --kb_map_file "$KB_MAP_FILE" \
  --entity_labels_file "$ENTITY_LABELS_FILE" \
  --golden_rels_file "$GOLDEN_RELS_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_hops "$MAX_HOPS" \
  --num_workers "$NUM_WORKERS" \
  --spacy_model "$SPACY_MODEL"
  # --debug # Uncomment for detailed logging (can be very verbose)

echo "Script finished. Output saved to $OUTPUT_FILE"
