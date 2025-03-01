import json

def transform_paths(input_file, output_file):
    """
    Transforms paths by:
    1. Removing intermediate entities
    2. Removing consecutive duplicate relations
    
    Args:
        input_file: Path to input JSONL with paths
        output_file: Path to write transformed JSONL
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            transformed_paths = []
            
            for path in entry['paths']:
                # Handle empty or invalid paths
                if len(path) < 2:
                    continue
                    
                # Extract head and tail entities
                head_entity = path[0]
                tail_entity = path[-1]
                
                # Extract all relations (at odd indices)
                relations = [path[i] for i in range(1, len(path)-1, 2)]
                
                # Remove consecutive duplicate relations
                unique_relations = []
                for rel in relations:
                    if not unique_relations or rel != unique_relations[-1]:
                        unique_relations.append(rel)
                
                # Create new path with head, unique relations, and tail
                new_path = [head_entity] + unique_relations + [tail_entity]
                transformed_paths.append(new_path)
            
            # Create new entry with transformed paths
            new_entry = {
                'id': entry['id'],
                'question': entry['question'],
                'prediction': entry['prediction'],
                'paths': transformed_paths
            }
            
            # Write to output file
            json.dump(new_entry, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == "__main__":
    input_file = "/data/home/mpx602/projects/ETU/ETU/fppl/all_paths.jsonl"  # Update with your input file path
    output_file = "/data/home/mpx602/projects/ETU/ETU/fppl/no_middle_entity.jsonl"  # Update with your desired output path
    transform_paths(input_file, output_file)
    print(f"Transformed paths written to {output_file}")