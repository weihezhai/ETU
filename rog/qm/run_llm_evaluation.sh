# Create logs directory if it doesn't exist
mkdir -p ./logs

# Path to the local Llama 3.1 8B model
MODEL_PATH="/data/scratch/mpx602/model/llm-models/"  # Replace with your actual model path

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create a directory for results
mkdir -p /data/home/mpx602/projects/ETU/ETU/rog/evaluation_results

# Get all prompt files
# FILES=($(ls /data/home/mpx602/projects/ETU/ETU/rog/topk_paths_prompts_rog/top*_sim_filtered_paths_with_prompts.jsonl))
# PROCESSED_FILE="/data/home/mpx602/projects/ETU/ETU/rog/topk_paths_prompts_rog/top15_sim_filtered_paths_with_prompts.jsonl"
PROCESSED_FILE="/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"
# Get the specific file for this array task

filename=$(basename "$PROCESSED_FILE")
base_filename="${filename%_with_prompts.jsonl}"
output="/data/home/mpx602/projects/ETU/ETU/rog/evaluation_results/${base_filename}_llm_results_rog.json"

echo -e "${BLUE}Evaluating $filename...${NC}"

# Optional: limit the number of samples for testing
# MAX_SAMPLES="--max-samples 10"
MAX_SAMPLES=""

python /data/home/mpx602/projects/ETU/ETU/rog/qm/test_prompts_with_llm.py \
  --model "$MODEL_PATH" \
  --input "$PROCESSED_FILE" \
  --output "$output" \
  --batch-size 1 \
  $MAX_SAMPLES
  
echo -e "${GREEN}Completed evaluation of $filename!${NC}"