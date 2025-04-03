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

# Activate python environment
source /data/home/mpx602/projects/py311/bin/activate

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set model path - adjust as needed
MODEL_PATH="/data/scratch/mpx602/model/llm-models/Meta-Llama-3.1-8B-Instruct"  # Can be HF model ID or local path

# Create directories
RESULTS_DIR="/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation"
mkdir -p $RESULTS_DIR
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
INPUT_DIR="/data/home/mpx602/projects/ETU/ETU/info_gain/input_jsonl"
FILES=($(ls ${INPUT_DIR}/*.jsonl))

# Get the specific file for this array task
# If not running as array job, process the first file
TASK_ID=${SGE_TASK_ID:-1}
PROCESSED_FILE="${FILES[$((TASK_ID-1))]}"

# Extract filename for output
filename=$(basename "$PROCESSED_FILE")
base_filename="${filename%.jsonl}"
OUTPUT_FILE="${RESULTS_DIR}/${base_filename}_path_eval_results_2.json"

echo -e "${BLUE}Evaluating $filename (Task ID: $TASK_ID)...${NC}"
echo -e "${BLUE}Using ${#SYSTEM_PROMPTS[@]} system prompts for robust evaluation${NC}"
echo -e "${BLUE}Testing with first 10 samples only${NC}"

# Build the system prompts arguments
PROMPT_ARGS=""
for prompt in "${SYSTEM_PROMPTS[@]}"; do
  PROMPT_ARGS+=" --system-prompts \"$prompt\""
done

# Run path evaluation script
# Adjust the path to path_evaluation.py if needed
# We use eval to properly handle the quoted arguments
eval python /data/home/mpx602/projects/ETU/ETU/info_gain/path_evaluation.py \
  --model \"$MODEL_PATH\" \
  --input \"$PROCESSED_FILE\" \
  --output \"$OUTPUT_FILE\" \
  --device \"cuda\" \
  $PROMPT_ARGS 
  
echo -e "${GREEN}Completed evaluation of $filename! Results saved to $OUTPUT_FILE${NC}" 