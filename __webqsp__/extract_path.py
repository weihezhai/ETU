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
    Extracts reasoning paths from input text that end with predicted answers.
    The paths are processed to:
    1. Remove intermediate entities.
    2. For relations, keep only the last two components (e.g., "type.property").
    
    Args:
        input_text: Raw text containing reasoning paths.
        prediction: Model's predicted answer(s) used to filter paths by their endpoint.
    
    Returns:
        List of processed, valid paths. Each path is a list of strings
        (start_entity, processed_relation_1, ..., processed_relation_n, end_entity).
    """
    # Find the reasoning paths section between "Reasoning Paths:" and "Question:"
    pattern = r'Reasoning Paths:\n(.*?)\n\nQuestion:'
    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        return []
    
    paths_text = match.group(1).strip()
    if not paths_text:
        return []
    
    # Split prediction into possible values
    pred_values = set(prediction.split('\n'))
    
    # Process individual paths
    final_processed_paths = []
    for path_str in paths_text.split('\n'):
        if '->' not in path_str: # Ensure it's a chained path
            continue
            
        elements = [elem.strip() for elem in path_str.split('->')]
        
        # Only include paths that are non-empty and end with one of the prediction values
        if not elements or elements[-1] not in pred_values:
            continue
            
        current_path_elements = []
        
        # Add the first entity (guaranteed to exist if elements is not empty)
        current_path_elements.append(elements[0])

        # Process relations (at odd indices: 1, 3, ...)
        # Keep only the last two parts if the relation string has them.
        # Intermediate entities (at even indices: 2, 4, ...) are skipped by the loop's step.
        for i in range(1, len(elements) - 1, 2):
            relation_candidate = elements[i]
            parts = relation_candidate.split('.')
            if len(parts) >= 2:
                current_path_elements.append('.'.join(parts[-2:]))
            # If relation_candidate doesn't have at least two parts, it's omitted from the path.
        
        # Add the last entity if the path originally had more than one element.
        # If len(elements) == 1, elements[0] is also elements[-1] and was already added.
        if len(elements) > 1:
            current_path_elements.append(elements[-1])
        
        # Add the constructed path.
        # This path might be shorter if relations were filtered (e.g., [start_entity, end_entity]).
        final_processed_paths.append(current_path_elements)
    
    return final_processed_paths

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
    Processes JSONL file to create filtered dataset with valid reasoning paths.
    
    Performs:
    - Extraction of relevant reasoning paths
    - Output filtering to only include entries with valid paths
    
    Args:
        input_file: Path to input JSONL with raw QA pairs
        output_file: Path to write filtered JSONL with paths
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            
            # Extract paths from input, passing prediction to filter paths
            paths = extract_paths(entry['input'], entry['prediction'])
            
            # Skip entries with no valid paths
            if not paths:
                continue
            
            # Create new entry with required fields
            new_entry = {
                'id': entry['id'],
                'question': entry['question'],
                'prediction': entry['prediction'],
                'ground_truth': entry['ground_truth'],
                'paths': paths
            }
            
            # Write to output file
            json.dump(new_entry, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == "__main__":
    # Example usage
    input_file = "/data/home/mpx602/projects/ETU/ETU/__webqsp__/predictions.jsonl"  # Update with your input file path
    output_file = "web_trajectories.jsonl"  # Update with your output file path
    process_jsonl(input_file, output_file)
