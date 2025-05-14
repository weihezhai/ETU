#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=llm_evaluation
#SBATCH --output=./logs/llm_evaluation_%A_%a.out
#SBATCH --time=0-10:00:00


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
mkdir -p /mnt/parscratch/users/acr24wz/ETU/__webqsp__/icl/llm_gen_results

# Get all prompt files


# Get the specific file for this array task
input="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/icl/icl_prompt/web_filtered_paths_prompt.jsonl"
output="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/icl/llm_gen_results/llm_gen_results.json"

# Optional: limit the number of samples for testing
# MAX_SAMPLES="--max-samples 10"
MAX_SAMPLES=""

python /mnt/parscratch/users/acr24wz/ETU/__webqsp__/icl/test_prompts_with_llm.py \
  --model "$MODEL_PATH" \
  --input "$input" \
  --output "$output" \
  --batch-size 1 \
  $MAX_SAMPLES
  
echo -e "${GREEN}Completed evaluation of $filename!${NC}"