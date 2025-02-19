#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --job-name=train
#SBATCH --output=./logs/output.%j.out
export SLURM_EXPORT_ENV=ALL
module load Python/3.9.6-GCCcore-11.2.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
source /users/acr24wz/etu/bin/activate
bash run_training_shef.sh
