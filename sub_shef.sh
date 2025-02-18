#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=82G
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --job-name=train

module load cuDNN/8.7.0.84-CUDA-11.8.0
source activate etu
bash run_training_shef.sh
