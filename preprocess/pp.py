import json
import math
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from extract_path import extract_paths

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
        # Check if any expected model file exists in the directory.
        for vf in valid_files:
            if os.path.exists(os.path.join(model_dir, vf)):
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

def compute_perplexity(model, tokenizer, question, path):
    """
    Computes the perplexity for a candidate reasoning path given a question.
    The prompt is formatted as:
        "Question: {question}\nThe_Path: {formatted_path}"
    where formatted_path is obtained by extracting the nodes from 'path' using extract_paths.
    """
    # Use extract_paths to get the nodes of the reasoning path.
    nodes = extract_paths(path)
    # Join the nodes with an arrow to create a readable path string.
    formatted_path = " -> ".join(nodes)
    # Create the full prompt.
    input_text = f"Question: {question}\nThe_Path: {formatted_path}"
    
    # Tokenize the input (with truncation if needed).
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
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
            paths = data["input"]
            ground_truths = data["ground_truth"]
            
            # Count all candidate paths for the ratio computing.
            total_candidate_paths += len(paths)
            
            scored_paths = []
            for path in paths:
                perplexity = compute_perplexity(model, tokenizer, question, path)
                nodes = extract_paths(path)
                # Get the end node of the path (if available)
                endnode = nodes[-1] if nodes else ""
                # Determine if the end node is one of the ground truth answers.
                hit = 1 if endnode in ground_truths else 0
                total_paths_hits += hit
                scored_paths.append({
                    "path": path,
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