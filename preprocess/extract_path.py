import json
import re

def extract_paths(input_text, prediction):
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
    
    # Split into individual paths and process each one
    paths = []
    for path in paths_text.split('\n'):
        if '->' in path:
            # Split path into nodes and clean up whitespace
            elements = [elem.strip() for elem in path.split('->')]
            # Only include paths that end with one of the prediction values
            if elements[-1] in pred_values:
                paths.append(elements)
    
    return paths

def is_valid_prediction(prediction, ground_truth):
    # Convert prediction string to set of items (split by newlines)
    pred_set = set(prediction.split('\n'))
    # Convert ground truth list to set
    gt_set = set(ground_truth)
    
    # Check if prediction is a subset of ground truth
    return pred_set.issubset(gt_set)

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            
            # Skip entries where prediction doesn't match ground truth
            if not is_valid_prediction(entry['prediction'], entry['ground_truth']):
                continue
            
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
                'paths': paths
            }
            
            # Write to output file
            json.dump(new_entry, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == "__main__":
    # Example usage
    input_file = "/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"  # Update with your input file path
    output_file = "trajectories.jsonl"  # Update with your output file path
    process_jsonl(input_file, output_file)
