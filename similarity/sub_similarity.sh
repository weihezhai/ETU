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
#SBATCH --array=0-5

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Define the input files to process
INPUT_FILES=(
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_path_then_question.jsonl"
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_question_then_path.jsonl"
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_integrated.jsonl"
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_path_context.jsonl"
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
  "$mnt/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores.jsonl"
)

# Default model directory
MODEL_DIR="$mnt/etu/qwen2.5/alibaba_nlp_gte_qwen2_7b_instruct"

# Get the current input file based on array task ID
CURRENT_INPUT=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

# Extract file name for output path
FILE_NAME=$(basename "$CURRENT_INPUT")
OUTPUT_FILE="$mnt/ETU/similarity/results/${FILE_NAME/.jsonl/_with_sim.jsonl}"

# Create output directory if it doesn't exist
mkdir -p "$mnt/ETU/similarity/results"
mkdir -p "$MODEL_DIR"

echo "Processing file $((SLURM_ARRAY_TASK_ID+1)) of ${#INPUT_FILES[@]}: $CURRENT_INPUT"
echo "Output will be saved to: $OUTPUT_FILE"
echo "Model directory: $MODEL_DIR"

# Run the similarity calculation directly without calling another script
echo "Launching path similarity calculation with the following parameters:"
echo "  Input File:               $CURRENT_INPUT"
echo "  Output File:              $OUTPUT_FILE"
echo "  Model Directory:          $MODEL_DIR"

# Run the python script with the provided parameters
python sim_question_path.py \
    --input_file "$CURRENT_INPUT" \
    --output_file "$OUTPUT_FILE" \
    --model_dir "$MODEL_DIR"