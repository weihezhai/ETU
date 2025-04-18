import json
import argparse
import os
from evaluate_results import evaluate_results
from tqdm import tqdm

def preprocess_relation(relation):
    """Extract the last two parts of a relation (separated by dots)."""
    parts = relation.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return relation

def extract_answers_by_similarity(json_file, output_file, top_k=5, sorted_relations_file=None, top_relations=None):
    """
    Process JSON-like file to extract answers based on similarity scores.
    Each path's last entity is used as a candidate answer.
    Paths are sorted by similarity_score in descending order.
    
    If sorted_relations_file is provided, paths are first filtered to only include
    those containing at least one of the top_relations relations.
    """
    processed_data = []
    
    # Load sorted relations if provided
    relation_patterns = {}
    if sorted_relations_file and top_relations:
        print("Loading relation patterns from", sorted_relations_file)
        with open(sorted_relations_file, 'r') as f:
            sorted_relations = json.load(f)
        
        # Preprocess top relations for each question
        for question_id, relations in tqdm(sorted_relations.items(), desc="Preprocessing relations"):
            top_k_relations = relations[:top_relations]
            relation_patterns[question_id] = [preprocess_relation(rel) for rel in top_k_relations]
    
    # Read the entire file content
    print(f"Reading file: {json_file}")
    with open(json_file, 'r') as f:
        content = f.read()
    
    # First, count number of JSON objects to process
    print("Counting JSON objects...")
    obj_count = content.count('{"id":')
    print(f"Found approximately {obj_count} objects to process")
    
    # Find each JSON object by counting braces
    # Start from each opening brace '{'
    i = 0
    count = 0
    objects_processed = 0
    
    with tqdm(total=obj_count, desc="Processing JSON objects") as pbar:
        while i < len(content):
            if content[i] == '{':
                # Found start of a JSON object
                start = i
                count = 1  # Count of unclosed braces
                i += 1
                
                # Find the matching closing brace
                while i < len(content) and count > 0:
                    if content[i] == '{':
                        count += 1
                    elif content[i] == '}':
                        count -= 1
                    i += 1
                
                if count == 0:
                    # We found a complete JSON object
                    json_str = content[start:i]
                    try:
                        entry = json.loads(json_str)
                        
                        # Skip entries without paths
                        if "paths" not in entry or not entry["paths"]:
                            pbar.update(1)
                            objects_processed += 1
                            continue
                        
                        paths_to_consider = entry["paths"]
                        
                        # Filter paths by relations if sorted_relations_file is provided
                        if sorted_relations_file and top_relations and entry.get("id") in relation_patterns:
                            patterns = relation_patterns[entry["id"]]
                            filtered_paths = []
                            
                            for path in entry["paths"]:
                                if "path" in path and path["path"]:
                                    path_data = path["path"]
                                    path_matches = False
                                    
                                    # Check all elements except first and last as potential relations
                                    for j in range(1, len(path_data) - 1):
                                        if isinstance(path_data[j], str):
                                            relation = path_data[j]
                                            # Check if this relation matches any of our patterns
                                            for pattern in patterns:
                                                if pattern in relation:
                                                    path_matches = True
                                                    break
                                            if path_matches:
                                                break
                                    
                                    if path_matches:
                                        filtered_paths.append(path)
                            
                            # If no paths matched the relations, use all paths
                            if filtered_paths:
                                paths_to_consider = filtered_paths
                        
                        # Sort paths by similarity score (descending)
                        sorted_paths = sorted(
                            paths_to_consider, 
                            key=lambda x: x.get("similarity_score", 0), 
                            reverse=True
                        )
                        
                        # Extract the last entity from each path as an answer
                        # Use a set to collect unique answers
                        unique_answers = set()
                        answers = []

                        for path in sorted_paths[:top_k]:
                            if "path" in path and path["path"]:
                                # Get the last entity in the path
                                last_entity = path["path"][-1]
                                if last_entity not in unique_answers:
                                    unique_answers.add(last_entity)
                                    answers.append(last_entity)
                        
                        processed_entry = {
                            "id": entry.get("id"),
                            "processed_results": answers
                        }
                        
                        processed_data.append(processed_entry)
                        objects_processed += 1
                        pbar.update(1)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON object: {e}")
                        print(f"First 100 chars of problematic object: {json_str[:100]}...")
                        pbar.update(1)
                        objects_processed += 1
                        continue
            else:
                i += 1
    
    print(f"Processed {len(processed_data)} questions out of {objects_processed} objects")
    
    # Save processed data
    print(f"Saving processed data to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate answers based on similarity scores')
    parser.add_argument('--input', '-i', required=True, help='Path to JSON file with paths and similarity scores')
    parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth JSONL file path')
    parser.add_argument('--output-dir', '-o', default='./results', help='Directory to save results')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of top paths to consider by similarity score')
    parser.add_argument('--relations-file', '-r', help='Path to sorted relations JSON file for filtering')
    parser.add_argument('--top-relations', '-tr', type=int, default=10, help='Number of top relations to use for filtering')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process input file to extract answers
    filter_suffix = f"_filt{args.top_relations}" if args.relations_file else ""
    processed_file = os.path.join(args.output_dir, f'processed_answers_top{args.top_k}{filter_suffix}.json')
    
    extract_answers_by_similarity(
        args.input, 
        processed_file, 
        args.top_k, 
        args.relations_file, 
        args.top_relations
    )
    
    filter_msg = f" after filtering with top {args.top_relations} relations" if args.relations_file else ""
    print(f"Processed answers saved to {processed_file}")
    print(f"Evaluating with top {args.top_k} paths by similarity score{filter_msg}...")
    
    # Run evaluation
    evaluation_output = os.path.join(args.output_dir, f'evaluation_top{args.top_k}{filter_suffix}_by_similarity.json')
    metrics, _ = evaluate_results(processed_file, args.ground_truth)
    
    # Print metrics
    print(f"\nEvaluation Results (Top {args.top_k} by similarity{filter_msg}):")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Hit@K: {metrics['hit_count']} ({metrics['hit_rate']:.4f})")
    print(f"Hit@1: {metrics['h_at_1_count']} ({metrics['h_at_1_rate']:.4f})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save detailed evaluation results
    with open(evaluation_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation results saved to {evaluation_output}")

if __name__ == "__main__":
    main()