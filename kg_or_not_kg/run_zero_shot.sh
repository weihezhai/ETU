#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=kg_zero_shot
#SBATCH --output=./logs/kg_zero_shot_%A_%a.out
#SBATCH --time=0-12:00:00
#SBATCH --array=0-1

# Define array of models to test (HF model name and local cache path)
declare -a MODEL_SOURCES=(
  "Qwen/Qwen2.5-7B-Instruct" 
  "meta-llama/Llama-3.1-8B-Instruct"
)

declare -a MODEL_CACHE_DIRS=(
  "/mnt/parscratch/users/acr24wz/etu/qwen2.5/7b_ori" 
  "/mnt/parscratch/users/acr24wz/etu/llama3/llama-3.1-8b-instruct"
)

# Default parameters that can be overridden
OUTPUT_DIR="./results"
OUTPUT_FILE="rog_cwq_results.json"
DATASET_NAME="rmanluo/RoG-cwq"
MAX_SAMPLES=-1  # -1 means process all samples
PROMPT="Please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list with the most possible answer at the top position."
TEMPERATURE=0.7
MAX_NEW_TOKENS=100
TOP_P=0.9
BATCH_SIZE=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --mail_user)
      # Update the SBATCH mail-user setting
      sed -i "s/^#SBATCH --mail-user=.*/#SBATCH --mail-user=$2/" $0
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Load necessary modules (adjust as needed for your environment)
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Get the current array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}
CURRENT_MODEL_SOURCE=${MODEL_SOURCES[$TASK_ID]}
CURRENT_MODEL_CACHE=${MODEL_CACHE_DIRS[$TASK_ID]}

# Create logs and output directories if they don't exist
mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CURRENT_MODEL_CACHE"

# Get a simplified model name for the output file
MODEL_IDENTIFIER=$(basename "$CURRENT_MODEL_SOURCE" | tr '/' '_')
MODEL_OUTPUT_FILE="${OUTPUT_FILE%.json}_${MODEL_IDENTIFIER}.json"
FULL_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_OUTPUT_FILE}"

echo "Job ${SLURM_ARRAY_TASK_ID} (${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}): Starting evaluation"
echo "Using model: ${CURRENT_MODEL_SOURCE}"
echo "Model cache directory: ${CURRENT_MODEL_CACHE}"
echo "Dataset: ${DATASET_NAME}"
echo "Output will be saved to: ${FULL_OUTPUT_PATH}"

# Run the Python script with the provided parameters - pass both model name and path
python llm_zero_shot.py \
  --model_name "$CURRENT_MODEL_SOURCE" \
  --model_path "$CURRENT_MODEL_CACHE" \
  --output_file "$FULL_OUTPUT_PATH" \
  --dataset_name "$DATASET_NAME" \
  --max_samples "$MAX_SAMPLES" \
  --prompt "$PROMPT" \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --top_p "$TOP_P" \
  --batch_size "$BATCH_SIZE"

echo "Job ${SLURM_ARRAY_TASK_ID} completed"