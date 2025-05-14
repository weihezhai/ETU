import json
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def load_model_and_tokenizer(model_name, model_dir=None):
    """
    Loads the embedding model and tokenizer from a local directory if available,
    otherwise downloads from HuggingFace.

    Args:
        model_name (str): Name of the model to load from HuggingFace.
        model_dir (str or None): Local directory where model is stored or to be saved.

    Returns:
        tuple: (model, tokenizer)
    """
    load_local = False
    if model_dir is not None and os.path.exists(model_dir):
        # Check if model files exist in the directory
        for file in os.listdir(model_dir):
            if file.endswith('.safetensors') or file == "pytorch_model.bin":
                load_local = True
                break
    
    if load_local:
        print(f"Loading model from local directory: {model_dir}")
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        print(f"Downloading model from HuggingFace: {model_name}")
        model = AutoModel.from_pretrained(
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

def get_embedding(model, tokenizer, text):
    """
    Get embeddings for the given text.
    
    Args:
        model: The embedding model
        tokenizer: The tokenizer for the model
        text: Input text to compute embeddings for
    
    Returns:
        numpy.ndarray: Text embedding
    """
    # Get the device where model is located
    device = next(model.parameters()).device
    
    # Tokenize and move inputs to model's device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # Compute embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Use the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()
    
    return embedding

def compute_similarity(model, tokenizer, question, prompt):
    """
    Computes the cosine similarity between question and prompt.
    
    Args:
        model: The embedding model
        tokenizer: The tokenizer for the model
        question: Question text
        prompt: Prompt text
    
    Returns:
        float: Cosine similarity score
    """
    # Get embeddings
    question_embedding = get_embedding(model, tokenizer, question)
    prompt_embedding = get_embedding(model, tokenizer, prompt)
    
    # Compute cosine similarity
    similarity = cosine_similarity(question_embedding, prompt_embedding)[0][0]
    
    return similarity

def calculate_similarity_scores(input_file, output_file, model_dir=None):
    """
    Calculate similarity scores between questions and path strings.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the output with similarity scores
        model_dir: Directory containing the model (if exists)
    """
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
    
    # Count total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # Track statistics
    all_similarity_scores = []
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Calculating similarity scores"):
            entry = json.loads(line)
            question = entry["question"]
            
            for path_entry in entry["paths"]:
                prompt = path_entry["path_str"]
                
                # Calculate similarity score with prompt
                similarity_score = compute_similarity(model, tokenizer, question, prompt)
                all_similarity_scores.append(similarity_score)
                
                # Add similarity score to path entry
                path_entry["similarity_score"] = float(similarity_score)
            
            # Write to output file
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Calculate and display statistics
    if all_similarity_scores:
        print("\nSimilarity Score Statistics:")
        print(f"  Count: {len(all_similarity_scores)}")
        print(f"  Mean: {np.mean(all_similarity_scores):.4f}")
        print(f"  Std Dev: {np.std(all_similarity_scores):.4f}")
        print(f"  Min: {min(all_similarity_scores):.4f}")
        print(f"  Max: {max(all_similarity_scores):.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate similarity scores between questions and path strings."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file with similarity scores."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing the pre-trained model (if exists)."
    )
    args = parser.parse_args()
    
    calculate_similarity_scores(
        args.input_file, 
        args.output_file, 
        args.model_dir
    )

if __name__ == "__main__":
    main()