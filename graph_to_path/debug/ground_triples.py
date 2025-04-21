import json
import argparse
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl_data(filepath):
    """Loads data from a JSON Lines file."""
    logging.info(f"Loading JSON Lines data from: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                    if line_num % 1000 == 0: # Log progress for large files
                        logging.info(f"  Loaded {line_num} lines from {filepath}...")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON line {line_num} in {filepath}: {e}")
                    logging.error(f"Problematic line content: {line[:100]}...")
                    # Decide whether to skip or raise
                    # raise # Option 1: Stop on first error
                    logging.warning(f"Skipping line {line_num} due to JSON decode error.") # Option 2: Skip line
        logging.info(f"Successfully loaded {len(data)} JSON objects from: {filepath} (JSON Lines format)")
        return data
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        raise

def load_json_map(filepath):
    """Loads data from a regular JSON file (expected to be a map/dict)."""
    logging.info(f"Loading JSON map data from: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logging.warning(f"Loaded data from {filepath} is not a dictionary (map). Type: {type(data).__name__}")
        logging.info(f"Successfully loaded map with {len(data)} entries from: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON map in {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading map {filepath}: {e}")
        raise

def create_inverse_map(mapping):
    """Creates an inverse mapping (value -> key) from a dictionary."""
    inverse_map = {v: k for k, v in mapping.items()}
    if len(inverse_map) != len(mapping):
        logging.warning("Original mapping had duplicate values. Inverse map size differs.")
    logging.info(f"Created inverse map with {len(inverse_map)} entries.")
    return inverse_map

def ground_single_triple(triple, int_to_mid_map, int_to_rel_map, mid_to_label_map):
    """Converts a single triple [h, r, t] of integer IDs to a human-readable string."""
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        # Reduce noise by logging only in debug mode for this specific warning
        logging.debug(f"Skipping invalid triple format: {triple}")
        return None

    head_id, rel_id, tail_id = triple

    # Ground Head
    head_mid = int_to_mid_map.get(head_id)
    if head_mid:
        head_label = mid_to_label_map.get(head_mid, f"LabelNotFound({head_mid})")
        head_str = f"{head_label}[{head_mid}]"
    else:
        head_str = f"EntityIDNotFound({head_id})"

    # Ground Relation
    rel_text = int_to_rel_map.get(rel_id, f"RelationIDNotFound({rel_id})")
    rel_str = f"-({rel_text})->"

    # Ground Tail
    tail_mid = int_to_mid_map.get(tail_id)
    if tail_mid:
        tail_label = mid_to_label_map.get(tail_mid, f"LabelNotFound({tail_mid})")
        tail_str = f"{tail_label}[{tail_mid}]"
    else:
        tail_str = f"EntityIDNotFound({tail_id})"

    return f"{head_str} {rel_str} {tail_str}"

def save_grounded_triples(grounded_list, filepath):
    """Saves the list of grounded triples to a JSON file."""
    logging.info(f"Saving {len(grounded_list)} grounded triples to: {filepath}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(grounded_list, f, indent=4)
        logging.info(f"Successfully saved grounded triples to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground integer ID triples from a JSON Lines file to human-readable text.")
    parser.add_argument("--input_subgraph_file", required=True, help="Path to the input JSON Lines (.jsonl) file containing subgraph structures. Each line should be a JSON object with 'subgraph': {'tuples': [[h,r,t],... ] }.")
    parser.add_argument("--entities_map_file", required=True, help="Path to the entity MID to integer ID mapping JSON file (e.g., entities.json).")
    parser.add_argument("--relations_map_file", required=True, help="Path to the relation text to integer ID mapping JSON file (e.g., relations.json).")
    parser.add_argument("--entity_labels_file", required=True, help="Path to the entity MID to human-readable label mapping JSON file (e.g., entities_names.json).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file containing a single list of all grounded triple strings.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    # Use the new JSON Lines loader for the main input
    input_data_list = load_jsonl_data(args.input_subgraph_file)
    # Use a simple JSON loader for the map files
    entities_map = load_json_map(args.entities_map_file)
    relations_map = load_json_map(args.relations_map_file)
    entity_labels_map = load_json_map(args.entity_labels_file)

    # Create inverse maps
    int_to_mid_map = create_inverse_map(entities_map)
    int_to_rel_map = create_inverse_map(relations_map)

    # Ground triples from all lines
    logging.info(f"Starting triple grounding for {len(input_data_list)} entries in the JSON Lines file...")
    all_grounded_triples_list = []
    processed_lines = 0
    skipped_lines_structure = 0

    for input_data in tqdm(input_data_list, desc="Processing lines"):
        # --- Extract triples from the expected structure for this line ---
        input_triples = []
        if isinstance(input_data, dict) and 'subgraph' in input_data and \
           isinstance(input_data['subgraph'], dict) and 'tuples' in input_data['subgraph'] and \
           isinstance(input_data['subgraph']['tuples'], list):
            input_triples = input_data['subgraph']['tuples']
            logging.debug(f"Extracted {len(input_triples)} triples from line {processed_lines + 1}.")
        else:
            logging.warning(f"Line {processed_lines + 1}: Input JSON object does not have the expected structure: {{'subgraph': {{'tuples': [...]}}}}. Skipping triples for this line.")
            skipped_lines_structure += 1
            processed_lines += 1
            continue # Skip to the next line
        # --- ---

        for triple in input_triples:
            grounded_str = ground_single_triple(
                triple, int_to_mid_map, int_to_rel_map, entity_labels_map
            )
            if grounded_str:
                all_grounded_triples_list.append(grounded_str)
        processed_lines += 1

    logging.info(f"Triple grounding complete. Processed {processed_lines} lines.")
    if skipped_lines_structure > 0:
         logging.warning(f"Skipped {skipped_lines_structure} lines due to unexpected structure.")
    logging.info(f"Successfully grounded a total of {len(all_grounded_triples_list)} triples.")


    # Save results (a single list of all grounded triples)
    save_grounded_triples(all_grounded_triples_list, args.output_file)

    logging.info("Script finished.")
