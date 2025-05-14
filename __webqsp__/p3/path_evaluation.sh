#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8      # 8 cores
#$ -l h_rt=24:00:00  # 12 hours runtime
#$ -l h_vmem=7.5G  # 7.5G RAM per core
#$ -m be
#$ -l gpu=1        # request 1 GPU
#$ -l rocky
#$ -l cluster=andrena   
#$ -N path_eval
#$ -t 1-1        # Adjust array size based on number of files to process

module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set model path - adjust as needed
MODEL_PATH="/data/scratch/mpx602/model/llm-models/Meta-Llama-3.1-8B-Instruct"  # Can be HF model ID or local path

mkdir -p ./logs

# Define system prompts directly in the script for easier editing
# SYSTEM_PROMPTS=(
#   "You are a helpful assistant that answers questions accurately based on the given information."
#   "As a knowledgeable AI, please provide the correct answer to the following question."
#   "I'll provide a question, and your job is to answer it directly and accurately."
#   "You are an expert assistant that provides precise answers to questions."
#   "Answer the following question based on your knowledge."
# )
SYSTEM_PROMPTS=(
  "Answer the following question based on your knowledge and the possibly correct supportive reasoning path."
  )
# Get all input files - adjust the path as needed
INPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/p3/web_filtered_paths.jsonl"

OUTPUT_FILE="/mnt/parscratch/users/acr24wz/ETU/__webqsp__/p3/web_filtered_paths_eval_results_p3.json"

# Build the system prompts arguments
PROMPT_ARGS=""
for prompt in "${SYSTEM_PROMPTS[@]}"; do
  PROMPT_ARGS+=" --system-prompts \"$prompt\""
done

# Run path evaluation script
# Adjust the path to path_evaluation.py if needed
# We use eval to properly handle the quoted arguments
eval python /mnt/parscratch/users/acr24wz/ETU/__webqsp__/p3/path_evaluation.py \
  --model \"$MODEL_PATH\" \
  --input \"$INPUT_FILE\" \
  --output \"$OUTPUT_FILE\" \
  --system-prompts "Answer the following question based on your knowledge and the possibly correct supportive reasoning path."
  
echo -e "${GREEN}Completed evaluation of $INPUT_FILE! Results saved to $OUTPUT_FILE${NC}" 