#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=path_sim
#SBATCH --output=./logs/path_sim_%A_%a.out
#SBATCH --time=0-12:00:00


# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Define the input files to process
INPUT_FILES="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/sim/web_filtered_paths_prompt.jsonl"

# Default model directory
MODEL_DIR="/mnt/parscratch/users/acr24wz/etu/llama3/"

# Extract file name for output path

OUTPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/sim/web_filtered_paths_prompt_sim.jsonl"


echo "Output will be saved to: $OUTPUT_FILE"
echo "Model directory: $MODEL_DIR"

# Run the similarity calculation directly without calling another script
echo "Launching path similarity calculation with the following parameters:"
echo "  Input File:               $INPUT_FILES"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"

# Run the python script with the provided parameters
python sim_question_path.py \
    --input_file "$INPUT_FILES" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR"