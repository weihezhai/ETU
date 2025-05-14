#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=llm_evaluation
#SBATCH --output=./logs/llm_evaluation_%A_%a.out
#SBATCH --time=0-24:00:00
#SBATCH --array=0-8  # Process 9 files (indices 0-8)

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Path to the local Llama 3.1 8B model
MODEL_PATH="/mnt/parscratch/users/acr24wz/etu/llama3/"  # Replace with your actual model path

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create a directory for results
mkdir -p /mnt/parscratch/users/acr24wz/ETU/similarity/topsim/evaluation_results

# Get all prompt files
FILES=($(ls /mnt/parscratch/users/acr24wz/ETU/similarity/topsim/topk_paths_prompts/top*_sim_filtered_paths_with_prompts.jsonl))

# Get the specific file for this array task
file=${FILES[$SLURM_ARRAY_TASK_ID]}
filename=$(basename "$file")
base_filename="${filename%_with_prompts.jsonl}"
output="/mnt/parscratch/users/acr24wz/ETU/similarity/topsim/evaluation_results/${base_filename}_llm_results.json"

echo -e "${BLUE}Task ID: $SLURM_ARRAY_TASK_ID - Evaluating $filename...${NC}"

# Optional: limit the number of samples for testing
# MAX_SAMPLES="--max-samples 10"
MAX_SAMPLES=""

python /mnt/parscratch/users/acr24wz/ETU/similarity/llm/test_prompts_with_llm.py \
  --model "$MODEL_PATH" \
  --input "$file" \
  --output "$output" \
  --batch-size 1 \
  $MAX_SAMPLES
  
echo -e "${GREEN}Completed evaluation of $filename!${NC}"