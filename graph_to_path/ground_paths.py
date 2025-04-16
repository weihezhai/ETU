import json
import argparse
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(filepath, is_map=False):
    """Loads data from a JSON file (single object/list or map)."""
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

def ground_path(path, int_to_mid_map, int_to_rel_map, mid_to_label_map):
    """Converts a path of integer IDs to a human-readable string."""
    grounded_elements = []
    for i, element_id in enumerate(path):
        if i % 2 == 0: # Node ID
            mid = int_to_mid_map.get(element_id)
            if mid:
                label = mid_to_label_map.get(mid, f"LabelNotFound({mid})")
                grounded_elements.append(f"{label}[{mid}]") # Include MID for clarity
            else:
                grounded_elements.append(f"EntityIDNotFound({element_id})")
        else: # Relation ID
            rel_text = int_to_rel_map.get(element_id, f"RelationIDNotFound({element_id})")
            grounded_elements.append(f"-({rel_text})->")
    return " ".join(grounded_elements)

def ground_paths_in_data(cleaned_data, int_to_mid_map, int_to_rel_map, mid_to_label_map):
    """Adds grounded paths to the cleaned data entries."""
    logging.info("Starting path grounding...")
    grounded_results = []
    for entry in tqdm(cleaned_data, desc="Grounding paths"):
        grounded_paths_dict = {}
        original_paths = entry.get('paths', {})

        for hop_key, paths_list in original_paths.items():
            if paths_list: # Only process hop levels that have paths
                grounded_paths_list = []
                for path in paths_list:
                    grounded_path_str = ground_path(
                        path, int_to_mid_map, int_to_rel_map, mid_to_label_map
                    )
                    grounded_paths_list.append(grounded_path_str)
                grounded_paths_dict[hop_key] = grounded_paths_list
            # else: # Optionally keep empty hop keys
            #    grounded_paths_dict[hop_key] = []

        # Create a new entry or update the existing one
        new_entry = entry.copy() # Avoid modifying original data if reused
        new_entry['grounded_paths'] = grounded_paths_dict
        grounded_results.append(new_entry)

    logging.info("Path grounding complete.")
    return grounded_results

def save_json_data(data, filepath):
    """Saves data to a JSON file."""
    logging.info(f"Saving grounded data to: {filepath}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) # Use indent=4 for readability
        logging.info(f"Successfully saved grounded data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground integer ID paths to human-readable text.")
    parser.add_argument("--cleaned_paths_file", required=True, help="Path to the cleaned paths JSON file (output of clean_processed_paths.py).")
    parser.add_argument("--entities_map_file", required=True, help="Path to the entity MID to integer ID mapping JSON file (e.g., entities.json).")
    parser.add_argument("--relations_map_file", required=True, help="Path to the relation text to integer ID mapping JSON file (e.g., relations.json).")
    parser.add_argument("--entity_labels_file", required=True, help="Path to the entity MID to human-readable label mapping JSON file (e.g., CWQ_all_label_map.json).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file with grounded paths.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    cleaned_data = load_json_data(args.cleaned_paths_file)
    entities_map = load_json_data(args.entities_map_file, is_map=True)
    relations_map = load_json_data(args.relations_map_file, is_map=True)
    entity_labels_map = load_json_data(args.entity_labels_file, is_map=True)

    # Create inverse maps
    int_to_mid_map = create_inverse_map(entities_map)
    int_to_rel_map = create_inverse_map(relations_map)

    # Ground paths
    grounded_data = ground_paths_in_data(
        cleaned_data, int_to_mid_map, int_to_rel_map, entity_labels_map
    )

    # Save results
    save_json_data(grounded_data, args.output_file)

    logging.info("Script finished.") 