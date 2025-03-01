import json

def create_step_by_step_examples(path):
    """
    Create step-by-step training examples from a single path.
    Each example predicts the next relation given the trajectory so far.
    """
    examples = []
    
    for i in range(1, len(path), 2):  # Step by 2 to get relation positions
        # Input: previous nodes and relations up to current node
        input_trajectory = path[:i]
        # Target: next relation to predict
        target_relation = path[i]
        
        examples.append({
            'input_trajectory': input_trajectory,
            'target_relation': target_relation
        })
    
    return examples

def prepare_training_data(input_file, output_file):
    training_examples = []
    
    # Read the entire file as a single string
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split on closing brace followed by opening brace to get individual JSON objects
    json_strings = content.split('}\n{')
    
    for i, json_str in enumerate(json_strings):
        # Restore the braces except for first and last entries
        if i > 0:
            json_str = '{' + json_str
        if i < len(json_strings) - 1:
            json_str = json_str + '}'
            
        try:
            entry = json.loads(json_str)
            
            # Process each path in the entry
            for path in entry['paths']:
                # Create training examples for this path
                path_examples = create_step_by_step_examples(path)
                
                # Add metadata to each example
                for example in path_examples:
                    training_example = {
                        'id': entry['id'],
                        'question': entry['question'],
                        'full_path': path,
                        'input_trajectory': example['input_trajectory'],
                        'target_relation': example['target_relation']
                    }
                    training_examples.append(training_example)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON entry {i}: {e}")
            continue
    
    # Write training examples to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    input_file = "trajectories.jsonl"
    output_file = "relation_prediction_dataset.jsonl"
    prepare_training_data(input_file, output_file)
