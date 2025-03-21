
# Create logs directory if it doesn't exist
mkdir -p ./logs

# Path to the local Llama 3.1 8B model
MODEL_PATH="/data/scratch/mpx602/model/llm-models/"  # Replace with your actual model path

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Example 1: Test with a specific sample from a processed file
echo -e "${BLUE}Running test with a sample from a processed file...${NC}"

# Choose one of the processed files
PROCESSED_FILE="/data/home/mpx602/projects/ETU/ETU/rog/topk_paths_prompts_rog/top8_sim_filtered_paths_with_prompts.jsonl"

  # Test the first sample (index 0)
  python /data/home/mpx602/projects/ETU/ETU/rog/test_single_prompt.py \
  --model "$MODEL_PATH" \
  --input-file "$PROCESSED_FILE" \
  --sample-index 0

echo -e "${GREEN}Test completed!${NC}"
echo ""

# Example 2: Test with a different sample index
echo -e "${BLUE}Running test with a different sample index...${NC}"

# Test sample at index 5
python /data/home/mpx602/projects/ETU/ETU/rog/test_single_prompt.py \
  --model "$MODEL_PATH" \
  --input-file "$PROCESSED_FILE" \
  --sample-index 5

echo -e "${GREEN}Test completed!${NC}"
echo ""

# Example 3: Test with a direct prompt
echo -e "${BLUE}Running test with a direct prompt...${NC}"

# Custom prompt
PROMPT="Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list. The answer list is wrapped with [ANS][/ANS], each entry in the answer list can contain nothing but the answer text itself.
Reasoning Paths:
Lou Seal -> mascot.team -> sports_championship_event.champion -> 2014 World Series
Lou Seal -> mascot.team -> sports_team.championships -> 2012 World Series

Question:
Lou Seal is the mascot for the team that last won the World Series when?"

python /data/home/mpx602/projects/ETU/ETU/rog/test_single_prompt.py \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT"

echo -e "${GREEN}Test completed!${NC}"
echo ""

echo "All tests finished successfully!" 