import json
import math
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def load_model_and_tokenizer(model_name, model_dir=None):
    """
    Loads the model and tokenizer from a local directory if available,
    otherwise downloads from HuggingFace.

    Args:
        model_name (str): Name of the model to load from HuggingFace.
        model_dir (str or None): Local directory where model is stored or to be saved.

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
        # Check if .safetensors files exist in the directory
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


def compute_perplexity(model, tokenizer, text):
    """
    Computes the perplexity for the given text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        text: Input text to compute perplexity for
    
    Returns:
        float: Perplexity score
    """
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Compute negative log likelihood and return perplexity
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        neg_log_likelihood = outputs.loss.item()
    
    return math.exp(neg_log_likelihood)


def format_path(path_list):
    """
    Format a path list into a string representation.
    
    Args:
        path_list: List of path nodes
    
    Returns:
        string: Formatted path with arrow notation
    """
    if len(path_list) <= 1:
        return "".join(path_list)
    
    formatted = ""
    for i, node in enumerate(path_list):
        formatted += node
        if i < len(path_list) - 1:
            formatted += " -> "
    
    return formatted


def calculate_path_ppl_scores(input_file, output_file, model_dir=None):
    """
    Calculate perplexity scores for path+question combinations in no_middle_entity.jsonl.
    
    Args:
        input_file: Path to the no_middle_entity.jsonl file
        output_file: Path to save the output with perplexity scores
        model_dir: Directory containing the model (if exists)
    """
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
    
    # Count total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # Statistics tracking
    all_path_question_ppl = []
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Calculating perplexity"):
            entry = json.loads(line)
            question = entry["question"]
            
            paths_with_ppl = []
            
            for path_list in entry["paths"]:
                # Format path as a string with arrow notation
                path_str = format_path(path_list)
                
                # Calculate perplexity for path + question
                path_question_prompt = f"The_Path: {path_str}\nQuestion: {question}"
                ppl_score = compute_perplexity(model, tokenizer, path_question_prompt)
                
                # Track statistics
                all_path_question_ppl.append(ppl_score)
                
                # Save path with its perplexity score
                paths_with_ppl.append({
                    "path": path_list,
                    "path_str": path_str,
                    "ppl_score": ppl_score
                })
            
            # Sort paths by perplexity (ascending)
            paths_with_ppl.sort(key=lambda x: x["ppl_score"])
            
            # Create output entry
            output_entry = {
                "id": entry["id"],
                "question": question,
                "prediction": entry["prediction"],
                "paths": paths_with_ppl
            }
            
            # Write to output file
            f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
    
    # Calculate and display statistics
    if all_path_question_ppl:
        print("\nPerplexity Statistics for Path + Question:")
        print(f"  Count: {len(all_path_question_ppl)}")
        print(f"  Mean: {np.mean(all_path_question_ppl):.4f}")
        print(f"  Std Dev: {np.std(all_path_question_ppl):.4f}")
        print(f"  Min: {min(all_path_question_ppl):.4f}")
        print(f"  Max: {max(all_path_question_ppl):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity scores for path+question combinations."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="ETU/ETU/fppl/no_middle_entity.jsonl",
        help="Path to the no_middle_entity.jsonl file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ETU/ETU/fppl/path_ppl_scores.jsonl",
        help="Path to the output JSONL file with perplexity scores."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing the pre-trained model (if exists)."
    )
    args = parser.parse_args()
    
    calculate_path_ppl_scores(args.input_file, args.output_file, args.model_dir)


if __name__ == "__main__":
    main()