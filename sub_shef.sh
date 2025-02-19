#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --job-name=train
#SBATCH --output=./logs/output.%j.out

module load Anaconda3/2022.05
module load cuDNN/8.7.0.84-CUDA-11.8.0
source activate etu
bash run_training_shef.sh
