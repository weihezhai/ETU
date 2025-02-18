#!/bin/bash
# Usage:
#   ./run_training.sh [-e epochs] [-b batch_size] [-g grad_accum_steps]
#                     [-l learning_rate] [-o output_dir] [-m model_name]
#
# Example:
#   ./run_training.sh -e 5 -b 4 -g 4 -l 3e-5 -o "/data/scratch/mpx602/ETU/qwen2.5/my_run" -m "QwenInc/qwen2.5-7b-hf"

# Default hyperparameter values
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5
OUTPUT_DIR="/data/scratch/mpx602/ETU/qwen2.5/qwen2.5-7b-finetuned-relation"
MODEL_NAME="Qwen/Qwen2.5-7B"

# Parse command-line arguments.
while getopts "e:b:g:l:o:m:" opt; do
  case $opt in
    e) EPOCHS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    g) GRAD_ACCUM="$OPTARG" ;;
    l) LEARNING_RATE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) MODEL_NAME="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "Launching training with the following parameters:"
echo "  Epochs:                   $EPOCHS"
echo "  Per-device Batch Size:    $BATCH_SIZE"
echo "  Gradient Accumulation:    $GRAD_ACCUM"
echo "  Learning Rate:            $LEARNING_RATE"
echo "  Output Directory:         $OUTPUT_DIR"
echo "  Model Name:               $MODEL_NAME"

# Run the training script with the provided parameters.
python training.py \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" 