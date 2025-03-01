#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=path_ppl
#SBATCH --output=./logs/path_ppl.%j.out
#SBATCH --time=0-12:00:00

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Run the path perplexity calculation script using the runner
bash run_calculate_path_ppl.sh 