import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, prompt, continuation):
    input_text = prompt + " " + " ".join(continuation)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Calculate negative log likelihood
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    
    return math.exp(neg_log_likelihood)

def filter_paths(input_file, output_file, k=3, model_name="gpt2"):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            question = data["question"]
            paths = data["paths"]
            
            # Compute perplexity for each path
            scored_paths = []
            for path in paths:
                perplexity = compute_perplexity(model, tokenizer, question, path)
                scored_paths.append((perplexity, path))
            
            # Select top-k lowest perplexity paths
            scored_paths.sort(key=lambda x: x[0])
            selected_paths = [path for _, path in scored_paths[:k]]
            
            # Update and write result
            data["paths"] = selected_paths
            f_out.write(json.dumps(data) + "\n")

# Usage
filter_paths("trajectories.jsonl", "filtered_trajectories.jsonl", k=3)