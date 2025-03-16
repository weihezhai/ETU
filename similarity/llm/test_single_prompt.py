#!/usr/bin/env python3
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


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

def test_single_prompt(model_path, prompt):
    """Test a single prompt with the Llama model and print the result"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype="auto"
    )
    
    print("\nPrompt:")
    print(prompt)
    
    # Generate response from the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nGenerating response...")
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
    
    print("\nFull Response:")
    print(response)
    
    # Extract answer from the response
    answer = extract_answer(response)
    
    print("\nExtracted Answer:")
    print(answer)

def main():
    parser = argparse.ArgumentParser(description='Test a single prompt with Llama model')
    parser.add_argument('--model', type=str, required=True, help='Path to the local model')
    parser.add_argument('--input-file', type=str, help='Path to a JSON file containing a sample with a prompt')
    parser.add_argument('--sample-index', type=int, default=0, help='Index of the sample to use (if file contains multiple)')
    parser.add_argument('--prompt', type=str, help='Direct prompt to test (alternative to file input)')
    
    args = parser.parse_args()
    
    if not args.input_file and not args.prompt:
        print("Error: Either --input-file or --prompt must be provided")
        return
    
    prompt = args.prompt
    
    if args.input_file:
        # Read prompt from file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            try:
                # Try to read as JSONL
                samples = []
                for line in f:
                    samples.append(json.loads(line.strip()))
                
                if args.sample_index >= len(samples):
                    print(f"Error: Sample index {args.sample_index} is out of range (0-{len(samples)-1})")
                    return
                
                sample = samples[args.sample_index]
                prompt = sample.get('prompt', '')
                
            except json.JSONDecodeError:
                # Try to read as single JSON
                f.seek(0)
                sample = json.load(f)
                prompt = sample.get('prompt', '')
    
    if not prompt:
        print("Error: No prompt found")
        return
    
    test_single_prompt(args.model, prompt)

if __name__ == "__main__":
    main() 