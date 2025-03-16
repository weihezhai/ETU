#!/usr/bin/env python3
import json
import argparse
import os
import torch
from pathlib import Path
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(response):
    """Extract answers from the [ANS][/ANS] tags in the response"""
    pattern = r'\[ANS\](.*?)\[/ANS\]'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # Join multiple answers if found, filtering out empty ones
        answers = [match.strip() for match in matches if match.strip()]
        if answers:
            return '\n'.join(answers)
    
    # If no valid tags found, return the full response
    return response.strip()

def test_prompts_with_llm(model_path, input_file, output_file, batch_size=1, max_samples=None):
    """
    Test prompts with Llama 3.1 8B model and save results
    
    Args:
        model_path: Path to the local model
        input_file: JSON file with prompts
        output_file: Output file to save results
        batch_size: Number of samples to process at once (for efficiency)
        max_samples: Maximum number of samples to process (for testing)
    """
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Read input data
    print(f"Reading input file: {input_file}")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples and len(samples) >= max_samples:
                break
    
    results = []
    
    # Process samples
    print(f"Processing {len(samples)} samples...")
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]
        
        for sample in batch:
            sample_id = sample.get('id', '')
            question = sample.get('question', '')
            prompt = sample.get('prompt', '')
            
            # Generate response from the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from the response
            answer = extract_answer(response)
            
            # Add result to the list
            results.append({
                "id": sample_id,
                "question": question,
                "generated_result": answer
            })
            
            # Save results periodically
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
    
    # Save final results
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Test prompts with Llama 3.1 8B model')
    parser.add_argument('--model', type=str, required=True, help='Path to the local model')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file with prompts')
    parser.add_argument('--output', type=str, required=True, help='Output file to save results')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Import torch here to avoid importing it if the script is not run
    import torch
    
    test_prompts_with_llm(
        args.model,
        args.input,
        args.output,
        args.batch_size,
        args.max_samples
    )
    
    print("Testing completed successfully!")

if __name__ == "__main__":
    main() 