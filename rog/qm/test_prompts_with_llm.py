#!/usr/bin/env python3
import json
import argparse
import os
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    # Set pad_token to eos_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype="auto",
        pad_token_id=tokenizer.pad_token_id,  # Explicitly pass pad_token_id
        use_safetensors=True
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
            
            # Preprocess the prompt: remove instruction markers and add "Answer:"
            # Remove "[INST] <<SYS>>\n<</SYS>>" from the beginning
            prompt = re.sub(r'^\[INST\] <<SYS>>\n<</SYS>>', '', prompt)
            
            # Remove "[/INST]" from the end
            prompt = re.sub(r'\[/INST\]$', '', prompt)
            
            # Add "Answer:" at the end
            prompt = prompt.rstrip() + " Answer:"
            
            # Generate response from the model
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response - use directly as the answer
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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