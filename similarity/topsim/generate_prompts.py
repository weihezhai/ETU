#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def format_path(path):
    """Format a path by connecting elements with '->'"""
    return " -> ".join(path)

def generate_prompt(question, filtered_paths):
    """Generate a prompt based on paths and question"""
    # Format each path
    formatted_paths = [format_path(path) for path in filtered_paths]
    # Join paths with newlines
    path_str = "\n".join(formatted_paths)
    
    # Create the prompt according to the specified format
    prompt = (
        "Based on the reasoning paths, please answer the given question. "
        "Please keep the answer as simple as possible and return all the possible answers as a list. "
        "The answer list is wrapped with [ANS][/ANS], each entry in the answer list can contain nothing but the answer text itself."
        "\nReasoning Paths:\n" + path_str + 
        "\nQuestion:\n" + question
    )
    
    return prompt

def process_file(input_file, output_file):
    """Process the input JSONL file and generate prompts for each question"""
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Parse the JSON object
            data = json.loads(line.strip())
            
            # Extract the question and filtered paths
            question = data.get('question', '')
            filtered_paths = data.get('filtered_path_by_relation_similarity', [])
            
            # Generate the prompt
            prompt = generate_prompt(question, filtered_paths)
            
            # Add the prompt to the JSON object
            data['prompt'] = prompt
            
            # Write the updated JSON object to the output file
            f_out.write(json.dumps(data) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate prompts for JSONL files')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_path}...")
    process_file(input_path, output_path)
    print(f"Generated prompts saved to {output_path}")

if __name__ == "__main__":
    main() 