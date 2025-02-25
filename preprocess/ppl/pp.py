import json
import math
import torch
import argparse
import os
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np  # Add import for statistics calculations


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
        return [], []
    
    paths_text = match.group(1).strip()
    if not paths_text:
        return [], []
    
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

def compute_question_perplexity(model, tokenizer, question):
    """
    Computes the perplexity for just a question without any path.
    """
    # Create the question-only prompt
    input_text = f"Question: {question}"
    
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Compute negative log likelihood and return perplexity
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    return math.exp(neg_log_likelihood)

def compute_path_perplexity(model, tokenizer, path):
    """
    Computes the perplexity for just a path without the question.
    """
    # Create the path-only prompt
    input_text = f"The_Path: {path}"
    
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Compute negative log likelihood and return perplexity
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    return math.exp(neg_log_likelihood)

def compute_perplexity_reversed(model, tokenizer, question, path):
    """
    Computes the perplexity for a candidate reasoning path given a question,
    but with the path appearing before the question in the prompt.
    The prompt is formatted as:
        "The_Path: {path}\nQuestion: {question}"
    """
    # Create the full prompt with reversed order
    input_text = f"The_Path: {path}\nQuestion: {question}"
    
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Compute negative log likelihood and return perplexity.
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    return math.exp(neg_log_likelihood)

def filter_paths(input_file, output_file, k=3, model_dir=None, stats_file=None):
    """
    Processes the trajectories JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        k: Number of top paths to keep
        model_dir: Directory containing the model (if exists)
        stats_file: Path to output statistics file (if None, print to stdout)
    """
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
    
    # Count total lines first for an accurate progress bar.
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    total_candidate_paths = 0
    total_paths_hits = 0
    
    # Statistics tracking
    hit_paths_ppl = []  # All hit paths perplexity scores (just the path)
    question_hit_path_ppl = []  # All question+hit_path perplexity scores
    questions_only_ppl = []  # All questions-only perplexity scores
    path_question_ppl = []  # Reversed order: path+question perplexity (for hit paths)
    all_paths_ppl = []  # All paths perplexity scores (hit and non-hit)
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing entries"):
            data = json.loads(line)
            question = data["question"]
            
            # Calculate and store perplexity for question-only
            question_ppl = compute_question_perplexity(model, tokenizer, question)
            questions_only_ppl.append(question_ppl)
            
            # Skip entries with empty input
            if not data.get("input"):
                continue
                
            # Handle empty paths case
            result = input_to_path_list(data["input"])
            if not result:  # Empty result
                continue
            paths, nodes = result
            ground_truths = data["ground_truth"]
            
            # Count all candidate paths for the ratio computing.
            total_candidate_paths += len(paths)
            
            scored_paths = []
            for path, node_list in zip(paths, nodes):
                # Get question+path perplexity (original functionality)
                q_path_perplexity = compute_perplexity(model, tokenizer, question, path)
                
                # Calculate path-only perplexity (for all paths)
                path_only_ppl = compute_path_perplexity(model, tokenizer, path)
                all_paths_ppl.append(path_only_ppl)
                
                # Get path+question perplexity (reversed order)
                path_q_perplexity = compute_perplexity_reversed(model, tokenizer, question, path)
                
                # Determine if the end node is one of the ground truth answers.
                hit = 1 if node_list[-1] in ground_truths else 0
                total_paths_hits += hit
                
                # If this is a hit path, store its statistics
                if hit == 1:
                    hit_paths_ppl.append(path_only_ppl)
                    question_hit_path_ppl.append(q_path_perplexity)
                    path_question_ppl.append(path_q_perplexity)
                
                scored_paths.append({
                    "path": path,
                    "node_list": node_list,
                    "perplexity": q_path_perplexity,
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
    
    # Calculate statistics for the three requested metrics
    stats_results = {}
    
    # 1. Statistics for hit paths only
    stats_results["hit_paths"] = {
        "count": len(hit_paths_ppl),
        "mean": float(np.mean(hit_paths_ppl)) if hit_paths_ppl else float('nan'),
        "std": float(np.std(hit_paths_ppl)) if hit_paths_ppl else float('nan'),
        "min": float(min(hit_paths_ppl)) if hit_paths_ppl else float('nan'),
        "max": float(max(hit_paths_ppl)) if hit_paths_ppl else float('nan')
    }
    
    # 2. Statistics for question + hit path
    stats_results["question_hit_path"] = {
        "count": len(question_hit_path_ppl),
        "mean": float(np.mean(question_hit_path_ppl)) if question_hit_path_ppl else float('nan'),
        "std": float(np.std(question_hit_path_ppl)) if question_hit_path_ppl else float('nan'),
        "min": float(min(question_hit_path_ppl)) if question_hit_path_ppl else float('nan'),
        "max": float(max(question_hit_path_ppl)) if question_hit_path_ppl else float('nan')
    }
    
    # 3. Statistics for questions only
    stats_results["questions_only"] = {
        "count": len(questions_only_ppl),
        "mean": float(np.mean(questions_only_ppl)) if questions_only_ppl else float('nan'),
        "std": float(np.std(questions_only_ppl)) if questions_only_ppl else float('nan'),
        "min": float(min(questions_only_ppl)) if questions_only_ppl else float('nan'),
        "max": float(max(questions_only_ppl)) if questions_only_ppl else float('nan')
    }
    
    # 4. Statistics for path + question (reversed order)
    stats_results["path_question_reversed"] = {
        "count": len(path_question_ppl),
        "mean": float(np.mean(path_question_ppl)) if path_question_ppl else float('nan'),
        "std": float(np.std(path_question_ppl)) if path_question_ppl else float('nan'),
        "min": float(min(path_question_ppl)) if path_question_ppl else float('nan'),
        "max": float(max(path_question_ppl)) if path_question_ppl else float('nan')
    }
    
    # 5. Statistics for all paths (hit and non-hit)
    stats_results["all_paths"] = {
        "count": len(all_paths_ppl),
        "mean": float(np.mean(all_paths_ppl)) if all_paths_ppl else float('nan'),
        "std": float(np.std(all_paths_ppl)) if all_paths_ppl else float('nan'),
        "min": float(min(all_paths_ppl)) if all_paths_ppl else float('nan'),
        "max": float(max(all_paths_ppl)) if all_paths_ppl else float('nan')
    }
    
    # Output statistics
    if stats_file:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_results, f, indent=2)
    
    # Always print a summary to stdout
    print("\nPerplexity Statistics:")
    for category, stats in stats_results.items():
        print(f"\n{category.replace('_', ' ').title()} (count: {stats['count']}):")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
    
    return stats_results

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
    parser.add_argument(
        "--stats_file",
        type=str,
        default=None,
        help="Path to output statistics JSON file."
    )
    args = parser.parse_args()
    
    filter_paths(args.input_file, args.output_file, args.k, args.model_dir, args.stats_file)

if __name__ == "__main__":
    main()