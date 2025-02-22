import json
import math
import torch
import argparse
import os
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, model_dir=None):
    """
    Loads the model and tokenizer from a local directory if it contains at least one valid 
    model checkpoint file (such as pytorch_model.bin, model.safetensors, tf_model.h5, 
    model.ckpt.index or flax_model.msgpack). Otherwise, downloads the model 
    from HuggingFace and saves it to the given directory (if provided).

    Args:
        model_name (str): Name of the model to load from HuggingFace.
        model_dir (str or None): Local directory where model is stored (or to save the new model).

    Returns:
        tuple: (model, tokenizer)
    """
    valid_files = [
        "pytorch_model.bin", 
        "model.safetensors", 
        "tf_model.h5", 
        "model.ckpt.index", 
        "flax_model.msgpack"
    ]
    
    load_local = False
    if model_dir is not None and os.path.exists(model_dir):
        # Check for .safetensors files exist in the directory
        for file in os.listdir(model_dir):
            if file.endswith('.safetensors'):
                load_local = True
                break
    
    if load_local:
        print(f"Loading model from local directory: {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        print(f"Downloading model from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_dir:
            print(f"Saving model to: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
    
    return model, tokenizer

def input_to_path_list(input):
    """
    Converts a string input to a list of paths and a list of nodes.
    
    Args:
        input (str): The input string containing paths and nodes.
    
    Returns:
        List of valid paths and relevant lists of nodes.
    """
    # Find the reasoning paths section between "Reasoning Paths:" and "Question:"
    pattern = r'Reasoning Paths:\n(.*?)\n\nQuestion:'
    match = re.search(pattern, input, re.DOTALL)
    if not match:
        return []
    
    paths_text = match.group(1).strip()
    if not paths_text:
        return []
    
    paths = paths_text.split('\n')
    nodes = []
    # Split each path into nodes and clean up whitespace
    for path in paths:
        if '->' in path:
            # Split path into nodes and clean up whitespace
            elements = [elem.strip() for elem in path.split('->')]
            # Only include paths that end with one of the prediction values
            nodes.append(elements)
    return paths, nodes

def compute_perplexity(model, tokenizer, question, path):
    """
    Computes the perplexity for a candidate reasoning path given a question.
    The prompt is formatted as:
        "Question: {question}\nThe_Path: {path}"
    """
    # Create the full prompt.
    input_text = f"Question: {question}\nThe_Path: {path}"
    
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Compute negative log likelihood and return perplexity.
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    return math.exp(neg_log_likelihood)

def filter_paths(input_file, output_file, k=3, model_dir=None):
    """
    Processes the trajectories JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        k: Number of top paths to keep
        model_dir: Directory containing the model (if exists)
    """
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
    
    # Count total lines first for an accurate progress bar.
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    total_candidate_paths = 0
    total_paths_hits = 0
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing entries"):
            data = json.loads(line)
            question = data["question"]
            # paths and nodes are lists of paths and the relevant nodes of the reasoning paths.
            paths, nodes = input_to_path_list(data["input"])
            ground_truths = data["ground_truth"]
            
            # Count all candidate paths for the ratio computing.
            total_candidate_paths += len(paths)
            
            scored_paths = []
            for path, node_list in zip(paths, nodes):
                perplexity = compute_perplexity(model, tokenizer, question, path)
                # Determine if the end node is one of the ground truth answers.
                hit = 1 if node_list[-1] in ground_truths else 0
                total_paths_hits += hit
                scored_paths.append({
                    "path": path,
                    "node_list": node_list,
                    "perplexity": perplexity,
                    "hit": hit
                })
            
            # Sort paths by perplexity (ascending)
            scored_paths.sort(key=lambda x: x["perplexity"])
            
            # Take top-k paths (or all if less than k)
            selected_paths = scored_paths[:min(k, len(scored_paths))]
            
            # Create output entry in desired format
            output_entry = {
                "id": data["id"],
                "question": question,
                "ground_truth": ground_truths,
                "paths": selected_paths
            }
            
            # Write to output file
            f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
    
    # Calculate and display the overall hit ratio over all candidate paths.
    hit_ratio = total_paths_hits / total_candidate_paths if total_candidate_paths > 0 else 0
    print(f"Overall endnode hit ratio: {hit_ratio:.2%}")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess trajectories to compute perplexity and filter top-k paths."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="trajectories.jsonl",
        help="Path to the input trajectories JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="filtered_trajectories.jsonl",
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top paths (lowest perplexity) to select for each entry."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing the pre-trained model (if exists)."
    )
    args = parser.parse_args()
    
    filter_paths(args.input_file, args.output_file, args.k, args.model_dir)

if __name__ == "__main__":
    main()