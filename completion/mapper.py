import json
import argparse
import csv

def load_mapping_file(file_path):
    """Load a mapping file where each line number corresponds to an ID."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_id_to_entity_mapping(file_path):
    """Load the JSON file mapping node_ids to entity names."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_freebase_tsv_mapping(file_path):
    """Load the TSV file mapping freebase IDs to entity labels."""
    freebase_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) >= 3:
                freebase_id = row[0]
                label = row[2]
                # Extract the ID part after "/m/"
                if freebase_id.startswith("/m/"):
                    id_part = freebase_id[3:]
                    freebase_mapping[id_part] = label
    return freebase_mapping

def map_triple_to_text(triple, entity_mapping, relation_mapping):
    """Map a triple of IDs to a triple of text."""
    subject_id, relation_id, object_id = triple
    
    # Get text representations (line numbers in mapping files correspond to IDs)
    subject_text = entity_mapping[subject_id] if subject_id < len(entity_mapping) else f"Unknown Entity {subject_id}"
    relation_text = relation_mapping[relation_id] if relation_id < len(relation_mapping) else f"Unknown Relation {relation_id}"
    object_text = entity_mapping[object_id] if object_id < len(entity_mapping) else f"Unknown Entity {object_id}"
    
    return [subject_text, relation_text, object_text]

def map_triple_to_names(text_triple, id_to_entity_mapping, freebase_mapping=None):
    """Map a text triple (with node IDs) to a triple with entity names."""
    subject_id, relation, object_id = text_triple
    
    # Replace node IDs with entity names using multiple mapping sources
    subject_name = get_entity_name(subject_id, id_to_entity_mapping, freebase_mapping)
    object_name = get_entity_name(object_id, id_to_entity_mapping, freebase_mapping)
    
    return [subject_name, relation, object_name]

def get_entity_name(node_id, id_to_entity_mapping, freebase_mapping=None):
    """Get entity name from multiple mapping sources."""
    # First try the primary id_to_entity mapping
    if node_id in id_to_entity_mapping:
        return id_to_entity_mapping[node_id]
    
    # If not found and node_id starts with "m." and we have freebase mapping, try that
    if freebase_mapping and isinstance(node_id, str) and node_id.startswith("m."):
        id_part = node_id[2:]  # Remove "m." prefix
        if id_part in freebase_mapping:
            return freebase_mapping[id_part]
    
    # If still not found, return the original ID
    return node_id

def process_graph_file(graph_file_path, entity_mapping_path, relation_mapping_path, id_to_entity_path, freebase_tsv_path, output_file_path):
    """Process the graph triple file and map IDs to text."""
    # Load the mappings
    entity_mapping = load_mapping_file(entity_mapping_path)
    relation_mapping = load_mapping_file(relation_mapping_path)
    id_to_entity_mapping = load_id_to_entity_mapping(id_to_entity_path)
    
    # Load the additional freebase TSV mapping if provided
    freebase_mapping = None
    if freebase_tsv_path:
        freebase_mapping = load_freebase_tsv_mapping(freebase_tsv_path)
        print(f"Loaded {len(freebase_mapping)} entries from Freebase TSV mapping")
    
    # Process the graph JSON file line by line (JSONL format)
    results = []
    with open(graph_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                graph_entry = json.loads(line)
                
                # Process this entry
                if 'subgraph' in graph_entry and 'tuples' in graph_entry['subgraph']:
                    # Create a new field for text triples (if not already present)
                    if 'text_tuples' not in graph_entry['subgraph']:
                        graph_entry['subgraph']['text_tuples'] = [
                            map_triple_to_text(triple, entity_mapping, relation_mapping)
                            for triple in graph_entry['subgraph']['tuples']
                        ]
                    
                    # Create a new field for named tuples (with entity names instead of IDs)
                    graph_entry['subgraph']['named_tuples'] = [
                        map_triple_to_names(text_triple, id_to_entity_mapping, freebase_mapping)
                        for text_triple in graph_entry['subgraph']['text_tuples']
                    ]
                
                results.append(graph_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}... - {e}")
    
    # Write the results to the output file (in JSONL format)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Map graph triples from IDs to text and entity names.')
    parser.add_argument('graph_file', help='Path to the graph triple JSON file (JSONL format).')
    parser.add_argument('entity_file', help='Path to the entity mapping file.')
    parser.add_argument('relation_file', help='Path to the relation mapping file.')
    parser.add_argument('id_to_entity_file', help='Path to the JSON file mapping node IDs to entity names.')
    parser.add_argument('freebase_tsv', help='Path to the TSV file mapping freebase IDs to entity labels.', default=None)
    parser.add_argument('output_file', help='Path to save the output JSON file.')
    
    args = parser.parse_args()
    
    try:
        results = process_graph_file(
            args.graph_file,
            args.entity_file,
            args.relation_file,
            args.id_to_entity_file,
            args.freebase_tsv,
            args.output_file
        )
        print(f"Mapping complete. Processed {len(results)} entries. Output saved to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()