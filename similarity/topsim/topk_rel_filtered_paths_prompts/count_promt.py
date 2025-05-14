import json

def count_average_prompt_tokens(file_path):
    total_tokens = 0
    count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'prompt' in data:
                # Simple token counting by splitting on whitespace
                tokens = len(data['prompt'].split())
                total_tokens += tokens
                count += 1
    
    return total_tokens / count if count > 0 else 0

# Example usage
file_path = "/data/home/mpx602/projects/ETU/ETU/similarity/topsim/topk_rel_filtered_paths_prompts/top15_sim_filtered_paths_with_prompts.jsonl"
avg_tokens = count_average_prompt_tokens(file_path)
print(f"Average tokens per prompt: {avg_tokens}")