#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --job-name=train
#SBATCH --output=./logs/output.%j.out
#SBATCH --time=0-12:00:00

module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate etu
bash run_training_shef.sh
