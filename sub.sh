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
# $ -l cluster=andrena   
#$ -N train_qwen2.5_1.5b_relation

# module load gcc
source /data/home/mpx602/projects/py311/bin/activate
bash run_training.sh
