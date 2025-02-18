#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12      # 32 cores
#$ -l h_rt=12:0:0  # 240 hours runtime
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -m be
#$ -l gpu=1         # request 4 GPUs
## $ -l h=sbg4
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -N train_csqa_kl_mlp_0.0lambda

module load gcc/12.1.0
source /data/home/mpx602/projects/btlnk/bin/activate
python extract_path.py
