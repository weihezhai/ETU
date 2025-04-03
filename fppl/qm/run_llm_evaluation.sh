#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8      # 32 cores
#$ -l h_rt=12:0:0  # 240 hours runtime
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -m be
#$ -l gpu=1         # request 4 GPUs
## $ -l h=sbg4
#$ -l rocky
#$ -l cluster=andrena   
#$ -N fppl_llm_prompt
#$ -t 1-2

# module load gcc
source /data/home/mpx602/projects/py311/bin/activate

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Path to the local Llama 3.1 8B model
MODEL_PATH="/data/scratch/mpx602/model/llm-models/Meta-Llama-3.1-8B-Instruct"  # Replace with your actual model path

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create a directory for results
mkdir -p /data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/evaluation_results/75

# Get all input files
INPUT_DIR="/data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/75"
FILES=($(ls ${INPUT_DIR}/*.jsonl))

# Get the specific file for this array task
PROCESSED_FILE="${FILES[$((SGE_TASK_ID-1))]}"

# Extract filename for output
filename=$(basename "$PROCESSED_FILE")
base_filename="${filename%_with_prompts.jsonl}"
output="/data/home/mpx602/projects/ETU/ETU/fppl/top_percentile_ppl_prompts/evaluation_results/75/${base_filename}_llm_results.json"

echo -e "${BLUE}Evaluating $filename (Task ID: $SGE_TASK_ID)...${NC}"

# Optional: limit the number of samples for testing
# MAX_SAMPLES="--max-samples 10"
MAX_SAMPLES=""

python /data/home/mpx602/projects/ETU/ETU/fppl/qm/test_prompts_with_llm.py \
  --model "$MODEL_PATH" \
  --input "$PROCESSED_FILE" \
  --output "$output" \
  --batch-size 1 \
  $MAX_SAMPLES
  
echo -e "${GREEN}Completed evaluation of $filename!${NC}"