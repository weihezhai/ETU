#!/bin/bash
##SBATCH --partition=gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1 # NOTE: This script is CPU-bound, GPU resources will likely be unused.
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12  # Requesting 12 CPUs for the task
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=clean_processed_paths
#SBATCH --output=./logs/clean_processed_paths_%A_%a.out # Changed log name (removed 'ner')
#SBATCH --time=0-24:00:00
##SBATCH --array=0-8  # Process 9 files (indices 0-8) - Keep if you process multiple files

# Load necessary modules
module load Anaconda3/2024.02-1
# module load cuDNN/8.9.2.26-CUDA-12.1.1 # Removed as GPU/cuDNN not needed

# Activate your conda environment
source activate etu

# === Configuration ===
# Input file: The JSON output from process_subgraph.py
INPUT_JSON="/mnt/parscratch/users/acr24wz/src/src/output/train_output.json"

# Output file: Where to save the cleaned JSON data
OUTPUT_JSON="/mnt/parscratch/users/acr24wz/src/src/output/train_output_cleaned_more.json"

# Python script to execute
PYTHON_SCRIPT="/mnt/parscratch/users/acr24wz/ETU/graph_to_path/clean_processed_paths_more.py"

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