import json
import os
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name, model_dir=None):
    """
    Loads the causal language model and tokenizer from a local directory if available,
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
        logger.info(f"Loading model from local directory: {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        logger.info(f"Downloading model from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_dir:
            logger.info(f"Saving model to: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, text, device):
    """Calculate the perplexity of a text using the model."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    # Calculate perplexity from loss
    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()
    return perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on knowledge graph QA tasks")
    # Remove the mutually exclusive group to allow both arguments to be provided
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Model name from Hugging Face Hub to use for evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local model directory to use for evaluation")
    
    parser.add_argument("--output_file", type=str, default="rog_cwq_results.json", 
                        help="Path to save results JSON file")
    parser.add_argument("--dataset_name", type=str, default="rmanluo/RoG-cwq", 
                        help="Hugging Face dataset name")
    parser.add_argument("--max_samples", type=int, default=-1, 
                        help="Maximum number of samples to process. Set to -1 to process all.")
    parser.add_argument("--prompt", type=str, 
                        default="Please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list with the most possible answer at the top position.",
                        help="Prompt template to use for the model")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for processing (currently only batch_size=1 is supported)")
    
    args = parser.parse_args()
    # Check that at least one of model_name or model_path is provided
    if args.model_name is None and args.model_path is None:
        parser.error("At least one of --model_name or --model_path must be specified")
    
    return args

def get_gold_answers(example):
    """
    Extract gold answers from the example, handling different dataset formats.
    
    Args:
        example: Dataset example containing answer information
        
    Returns:
        list: List of gold answers
    """
    # Try different possible keys for answers in the dataset
    if "answer" in example:
        # Handle single answer or list of answers
        answers = example["answer"]
        if isinstance(answers, list):
            return answers
        else:
            return [answers]
    elif "answers" in example:
        # Handle answers key (used in some datasets)
        answers = example["answers"]
        if isinstance(answers, list):
            return answers
        else:
            return [answers]
    else:
        # No answer found
        return []

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load the dataset
    logger.info(f"Loading dataset from Hugging Face: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    test_set = dataset["test"]
    logger.info(f"Test set loaded with {len(test_set)} examples")
    
    # Apply max_samples limit if specified
    if args.max_samples > 0 and args.max_samples < len(test_set):
        logger.info(f"Limiting to {args.max_samples} samples as requested")
        test_set = test_set.select(range(args.max_samples))
    
    # Load model and tokenizer
    if args.model_path:
        # Check if the directory exists
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path, exist_ok=True)
            logger.info(f"Created model directory: {args.model_path}")
        
        # Try to load from local or download to this path
        if args.model_name:
            # If both are provided, use model_name as source and model_path as destination
            model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)
        else:
            # If only model_path is provided, try to load from there or fail
            try:
                model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_path)
            except Exception as e:
                logger.error(f"Failed to load model from path {args.model_path}: {e}")
                logger.error("If this is a download directory, please provide --model_name as well")
                raise
    else:
        # Only model_name provided, load/download without saving
        model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir and output_dir != '.':
        os.makedirs(output_dir, exist_ok=True)
    
    # Construct the prompt template
    prompt_template = args.prompt + "\n\nQuestion: {}\n\nAnswer:"
    
    # Store results
    results = []
    hit_ppls = []
    unhit_ppls = []
    
    # Analyze dataset format
    sample_example = test_set[0] if len(test_set) > 0 else {}
    logger.info(f"Dataset format sample: {list(sample_example.keys())}")
    
    # Process each question in the test set
    for idx, example in enumerate(tqdm(test_set)):
        question_id = example.get("id", f"question_{idx}")
        question = example["question"]
        gold_answers = get_gold_answers(example)
        
        # Create prompt for the model
        prompt = prompt_template.format(question)
        
        # Calculate perplexity
        ppl = calculate_perplexity(model, tokenizer, prompt, device)
        
        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=1,
            )
        
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        answer_part = generated_text[len(prompt):].strip()
        
        # Try to parse answers as a list (simple heuristic)
        try:
            # Extract answers from the generated text
            generated_answers = []
            for line in answer_part.split('\n'):
                # Remove numbering and bullet points
                clean_line = line.strip()
                if clean_line and any(clean_line.startswith(p) for p in ['- ', '• ', '* ', '1. ', '2. ']):
                    clean_line = clean_line[2:].strip()
                if clean_line:
                    generated_answers.append(clean_line)
            
            if not generated_answers:
                generated_answers = [answer_part]
        except:
            generated_answers = [answer_part]
        
        # Check if the top answer is in the gold answers (hit or unhit)
        is_hit = False
        if generated_answers and gold_answers:
            top_answer = generated_answers[0].lower()
            if any(gold.lower() in top_answer or top_answer in gold.lower() for gold in gold_answers):
                is_hit = True
        
        # Store perplexity based on hit/unhit
        if is_hit:
            hit_ppls.append(ppl)
        else:
            unhit_ppls.append(ppl)
        
        # Store result for this question
        result = {
            "id": question_id,
            "question": question,
            "ppl": ppl,
            "generated_answers": generated_answers,
            "gold_answers": gold_answers,
            "is_hit": is_hit
        }
        results.append(result)
        
        # Log progress every 10 questions
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1} questions")
    
    # Calculate statistics
    hit_ppl_avg = np.mean(hit_ppls) if hit_ppls else 0
    hit_ppl_std = np.std(hit_ppls) if hit_ppls else 0
    unhit_ppl_avg = np.mean(unhit_ppls) if unhit_ppls else 0
    unhit_ppl_std = np.std(unhit_ppls) if unhit_ppls else 0
    
    # Create final results
    final_results = {
        "questions": results,
        "statistics": {
            "hit_questions_count": len(hit_ppls),
            "hit_questions_ppl_avg": hit_ppl_avg,
            "hit_questions_ppl_std": hit_ppl_std,
            "unhit_questions_count": len(unhit_ppls),
            "unhit_questions_ppl_avg": unhit_ppl_avg,
            "unhit_questions_ppl_std": unhit_ppl_std,
            "ppl_difference": hit_ppl_avg - unhit_ppl_avg
        }
    }
    
    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Print statistics
    print("\nSTATISTICS:")
    print(f"Hit questions: {len(hit_ppls)}, Avg PPL: {hit_ppl_avg:.4f}, Std: {hit_ppl_std:.4f}")
    print(f"Unhit questions: {len(unhit_ppls)}, Avg PPL: {unhit_ppl_avg:.4f}, Std: {unhit_ppl_std:.4f}")
    print(f"PPL Difference (Hit - Unhit): {hit_ppl_avg - unhit_ppl_avg:.4f}")
    
if __name__ == "__main__":
    main()
