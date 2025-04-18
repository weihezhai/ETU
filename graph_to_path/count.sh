#!/bin/bash
##SBATCH --partition=cpu          # CPU partition should be sufficient
#SBATCH --mem=32G                # Adjust memory if needed, depends on map sizes
#SBATCH --cpus-per-task=8        # Adjust CPU count if needed
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=ground_triples
#SBATCH --output=./logs/count_%A.out # Log file name
#SBATCH --time=0-01:00:00        # Adjust time limit (should be quick for moderate files)

# Load necessary modules
module load Anaconda3/2024.02-1

# Activate your conda environment
source activate etu

ETU_DIR="/mnt/parscratch/users/acr24wz/ETU"
SRC_DATA_DIR="/mnt/parscratch/users/acr24wz/src/src" # Or wherever your maps are
# OUTPUT_DIR="$SRC_DATA_DIR/output" # Or a different output location

# Python script to execute
PYTHON_SCRIPT="$ETU_DIR/graph_to_path/count.py"
python "$PYTHON_SCRIPT" /mnt/parscratch/users/acr24wz/src/src/train_subgraph.json