import json
import argparse
from tqdm import tqdm

def preprocess_relation(relation):
    """Extract the last two parts of a relation (separated by dots)."""
    parts = relation.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return relation

def filter_items_by_top_relations(source_file, sorted_relations_file, output_file, k=10):
    """
    Filter items from a JSON list based on whether their 'path' contains top k relations.
    
    Args:
        source_file: Path to the source JSON file (list of objects, each with 'id', 'path').
        sorted_relations_file: Path to the JSON file with sorted relations (maps question_id to list of relations).
        output_file: Path to save the filtered results (JSON list of objects).
        k: Number of top relations to use as filters.
    """
    # Load source data (now a JSON list)
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Load sorted relations
    with open(sorted_relations_file, 'r', encoding='utf-8') as f:
        sorted_relations_map = json.load(f)
    
    # Process each item
    filtered_results = []
    for item in tqdm(source_data, desc="Processing items"):
        # Use the original item ID for lookup
        item_id = item['id']
        
        current_path = item.get('path') # Each item has one 'path'
        if not isinstance(current_path, list): # Skip if path is not a list or missing
            print(f"Skipping item {item.get('id', 'Unknown ID')} due to missing or invalid 'path'.")
            continue

        item_passes_filter = False # Flag for the current item

        # Use item_id (the original ID) for lookup in sorted_relations_map
        if item_id in sorted_relations_map:
            top_relations_for_id = sorted_relations_map[item_id][:k]
            # Preprocess relations to get matching patterns
            relation_patterns = [preprocess_relation(rel) for rel in top_relations_for_id]
            
            # Check relations within current_path
            # current_path format: [head, rel1, rel2, ..., tail]
            # We need to check relations from index 1 to len(current_path)-2
            if len(current_path) >= 3: # Path must have at least one relation: [head, rel, tail]
                for i in range(1, len(current_path) - 1):
                    if isinstance(current_path[i], str):
                        relation_in_path = current_path[i]
                        for pattern in relation_patterns:
                            if pattern in relation_in_path: # Substring match
                                item_passes_filter = True
                                break # Found a matching pattern for this relation_in_path
                        if item_passes_filter:
                            break # This item's path matches, no need to check more relations
            
        if item_passes_filter:
            filtered_results.append(item)
        else:
            print(f"Skipping item {item.get('id', 'Unknown ID')} due to missing or invalid 'path'.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2)
    
    print(f"Filtered items saved to {output_file}")
    print(f"Original items: {len(source_data)}, Filtered items: {len(filtered_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter items from a JSON list based on top k relations in their paths.')
    parser.add_argument('--source', required=True, help='Path to source JSON file (list of dicts with "id" and "path" fields)')
    parser.add_argument('--relations', required=True, help='Path to sorted relations JSON file (maps question_id to list of relations)')
    parser.add_argument('--output', required=True, help='Path to output JSON file (filtered list of dicts)')
    parser.add_argument('--k', type=int, default=10, help='Number of top relations to use (default: 10)')
    
    args = parser.parse_args()
    
    filter_items_by_top_relations(args.source, args.relations, args.output, args.k)