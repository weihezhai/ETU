# graph_to_path/ground_triples.py
import json
import argparse
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(filepath):
    """Loads data from a JSON file."""
    logging.info(f"Loading JSON data from: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        object_type = type(data).__name__
        size = len(data) if hasattr(data, '__len__') else 'N/A'
        logging.info(f"Successfully loaded data (type: {object_type}, size: {size}) from: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
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
        logging.warning(f"Skipping invalid triple format: {triple}")
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
            # Save as a JSON list of strings, one per line for potential easier reading/processing
            json.dump(grounded_list, f, indent=4)
        logging.info(f"Successfully saved grounded triples to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground integer ID triples to human-readable text.")
    parser.add_argument("--input_triples_file", required=True, help="Path to the input JSON file containing a list of triples [h, r, t].")
    parser.add_argument("--entities_map_file", required=True, help="Path to the entity MID to integer ID mapping JSON file (e.g., entities.json).")
    parser.add_argument("--relations_map_file", required=True, help="Path to the relation text to integer ID mapping JSON file (e.g., relations.json).")
    parser.add_argument("--entity_labels_file", required=True, help="Path to the entity MID to human-readable label mapping JSON file (e.g., entities_names.json).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file with grounded triple strings.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    input_triples = load_json_data(args.input_triples_file)
    entities_map = load_json_data(args.entities_map_file)
    relations_map = load_json_data(args.relations_map_file)
    entity_labels_map = load_json_data(args.entity_labels_file)

    # Create inverse maps
    int_to_mid_map = create_inverse_map(entities_map)
    int_to_rel_map = create_inverse_map(relations_map)

    # Ground triples
    logging.info(f"Starting triple grounding for {len(input_triples)} triples...")
    grounded_triples_list = []
    for triple in tqdm(input_triples, desc="Grounding triples"):
        grounded_str = ground_single_triple(
            triple, int_to_mid_map, int_to_rel_map, entity_labels_map
        )
        if grounded_str:
            grounded_triples_list.append(grounded_str)
    logging.info(f"Triple grounding complete. Successfully grounded {len(grounded_triples_list)} triples.")


    # Save results
    save_grounded_triples(grounded_triples_list, args.output_file)

    logging.info("Script finished.") 