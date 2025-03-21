import json
import argparse
from tqdm import tqdm

def preprocess_relation(relation):
    """Extract the last two parts of a relation (separated by dots)."""
    parts = relation.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return relation

def filter_paths_by_top_relations(source_file, sorted_relations_file, output_file, k=10):
    """
    Filter paths for each question based on top k relations from sorted_relation.json.
    
    Args:
        source_file: Path to the source JSONL file containing questions and paths
        sorted_relations_file: Path to the JSON file with sorted relations
        output_file: Path to save the filtered results
        k: Number of top relations to use as filters
    """
    # Load source data
    source_data = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            source_data.append(json.loads(line.strip()))
    
    # Load sorted relations
    with open(sorted_relations_file, 'r', encoding='utf-8') as f:
        sorted_relations = json.load(f)
    
    # Process each question
    results = []
    for item in tqdm(source_data, desc="Processing questions"):
        question_id = item['id']
        
        # Get top k relations for this question
        if question_id in sorted_relations:
            top_relations = sorted_relations[question_id][:k]
            # Preprocess relations to get matching patterns
            relation_patterns = [preprocess_relation(rel) for rel in top_relations]
            
            # Filter paths that contain any of the top k relations
            filtered_paths = []
            for path in item['paths']:
                # Paths are in format [head node, rel, rel, ..., rel, tail node]
                # We need to check all relations in the path (all elements except first and last)
                if len(path) >= 2:  # At minimum, we need [head, tail]
                    path_matches = False
                    # Check all elements except first and last as potential relations
                    for i in range(1, len(path) - 1):
                        if isinstance(path[i], str):
                            relation = path[i]
                            # Check if this relation matches any of our patterns
                            for pattern in relation_patterns:
                                if pattern in relation:
                                    path_matches = True
                                    break
                            if path_matches:
                                break
                    
                    if path_matches:
                        filtered_paths.append(path)
            
            # Add filtered paths to the result
            item_copy = item.copy()
            # If filtered paths is empty, use the original paths instead
            if not filtered_paths:
                item_copy['filtered_path_by_relation_similarity'] = item['paths']
            else:
                item_copy['filtered_path_by_relation_similarity'] = filtered_paths
            results.append(item_copy)
        else:
            # If no top relations found, use the original paths
            item_copy = item.copy()
            item_copy['filtered_path_by_relation_similarity'] = item['paths']
            results.append(item_copy)
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    
    print(f"Filtered paths saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter paths based on top k relations')
    parser.add_argument('--source', required=True, help='Path to source JSONL file')
    parser.add_argument('--relations', required=True, help='Path to sorted relations JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSONL file')
    parser.add_argument('--k', type=int, default=10, help='Number of top relations to use (default: 10)')
    
    args = parser.parse_args()
    
    filter_paths_by_top_relations(args.source, args.relations, args.output, args.k)