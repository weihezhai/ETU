#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=path_ppl
#SBATCH --output=./logs/path_ppl_%A_%a.out
#SBATCH --time=0-12:00:00
#SBATCH --array=0-4

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Define the prompt formats to test
PROMPT_FORMATS=(
  "path_then_question"
  "question_then_path"
  "integrated"
  "path_context"
  "explicit_reasoning"
)

# Get the current format based on array task ID
CURRENT_FORMAT=${PROMPT_FORMATS[$SLURM_ARRAY_TASK_ID]}

# Set output file based on format
OUTPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/fppl/path_ppl_scores_${CURRENT_FORMAT}.jsonl"

# Run the path perplexity calculation script with the current format
bash run_calculate_path_ppl.sh -f "$CURRENT_FORMAT" -o "$OUTPUT_FILE" 