"""
Processes KGQA (Knowledge Graph Question Answering) results to extract valid reasoning paths.

This module:
1. Extracts reasoning paths from LLM-generated text
2. Filters predictions to only include valid answers (subset of ground truth)
3. Processes JSONL data to create cleaned datasets of valid question-answer pairs with their reasoning paths

Example usage shown in main block processes input JSONL to create trajectories.jsonl with valid paths.
"""

import json
import re

def extract_paths(input_text, prediction):
    """
    Extracts reasoning paths from input text without filtering.
    
    Args:
        input_text: Raw text containing reasoning paths
        prediction: Model's predicted answer(s) - no longer used for filtering
    
    Returns:
        List of all paths where each path is a list of nodes
    """
    # Find the reasoning paths section between "Reasoning Paths:" and "Question:"
    pattern = r'Reasoning Paths:\n(.*?)\n\nQuestion:'
    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        return []
    
    paths_text = match.group(1).strip()
    if not paths_text:
        return []
    
    # Split into individual paths and process each one
    paths = []
    for path in paths_text.split('\n'):
        if '->' in path:
            # Split path into nodes and clean up whitespace
            elements = [elem.strip() for elem in path.split('->')]
            
            # Process each element - for relation elements (typically at odd indices),
            # keep only the last two parts of the relation name
            for i in range(len(elements)):
                # Assuming relations are at odd indices (1, 3, 5, etc.)
                if i % 2 == 1 and '.' in elements[i]:
                    parts = elements[i].split('.')
                    if len(parts) > 2:
                        # Keep only the last two parts
                        elements[i] = '.'.join(parts[-2:])
            
            # Include all paths regardless of endpoint
            paths.append(elements)
    
    return paths

def is_valid_prediction(prediction, ground_truth):
    """
    Validates if model predictions are subset of ground truth answers.
    
    Args:
        prediction: Newline-separated string of predicted answers
        ground_truth: List of acceptable answers
    
    Returns:
        True if all predictions are valid answers, False otherwise
    """
    # Convert prediction string to set of items (split by newlines)
    pred_set = set(prediction.split('\n'))
    # Convert ground truth list to set
    gt_set = set(ground_truth)
    
    # Check if prediction is a subset of ground truth
    return pred_set.issubset(gt_set)

def process_jsonl(input_file, output_file):
    """
    Processes JSONL file to create dataset with all reasoning paths.
    
    Performs:
    - Extraction of all reasoning paths
    - Includes all entries, regardless of prediction validity
    
    Args:
        input_file: Path to input JSONL with raw QA pairs
        output_file: Path to write JSONL with paths
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            
            # Extract all paths from input without filtering
            paths = extract_paths(entry['input'], entry['prediction'])
            
            # Skip entries with no paths
            if not paths:
                continue
            
            # Create new entry with required fields
            new_entry = {
                'id': entry['id'],
                'question': entry['question'],
                'prediction': entry['prediction'],
                'paths': paths
            }
            
            # Write to output file
            json.dump(new_entry, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == "__main__":
    # Example usage
    input_file = "/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"  # Update with your input file path
    output_file = "all_paths.jsonl"  # Update with your output file path
    process_jsonl(input_file, output_file)
