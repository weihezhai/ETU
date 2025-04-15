# graph_to_path/clean_processed_paths.py
import json
import argparse
import logging
import os
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(filepath):
    """Loads data from a JSON file (expects a single JSON object/list)."""
    logging.info(f"Loading JSON data from: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        raise

def clean_path_data(data):
    """
    Cleans the processed path data:
    1. Removes entries with no paths found.
    2. For entries with paths, keeps only the shortest hop paths.
    """
    logging.info("Starting path data cleaning...")
    cleaned_data = []
    removed_no_paths_count = 0
    kept_shortest_paths_count = 0

    for entry in data:
        paths_dict = entry.get('paths', {})
        min_hops = None
        shortest_paths = []

        # Determine the maximum hop key present to iterate safely
        # Assumes keys are like '1hop', '2hop', etc.
        hop_keys = sorted([int(k.replace('hop', '')) for k in paths_dict.keys() if k.endswith('hop')])

        # Find the minimum hop count with non-empty paths
        for k in hop_keys:
            hop_key = f'{k}hop'
            if paths_dict.get(hop_key): # Check if list is non-empty
                min_hops = k
                shortest_paths = paths_dict[hop_key]
                break # Found the shortest hop paths

        if min_hops is not None:
            # Keep the entry, but only with the shortest paths
            new_paths_dict = OrderedDict() # Keep order if needed, start with shortest
            new_paths_dict[f'{min_hops}hop'] = shortest_paths
            # Ensure other hop keys are present but empty if needed for schema consistency
            # Or simply keep only the shortest hop key. Let's keep only shortest.
            # for k_other in hop_keys:
            #    if k_other != min_hops:
            #       new_paths_dict[f'{k_other}hop'] = []

            entry['paths'] = new_paths_dict
            cleaned_data.append(entry)
            kept_shortest_paths_count += 1
            logging.debug(f"Entry {entry.get('id', 'N/A')}: Kept shortest paths ({min_hops} hops).")
        else:
            # No paths found for any hop count, remove the entry
            removed_no_paths_count += 1
            logging.debug(f"Entry {entry.get('id', 'N/A')}: Removed (no paths found).")

    logging.info(f"Cleaning complete. Kept {kept_shortest_paths_count} entries (shortest paths). Removed {removed_no_paths_count} entries (no paths).")
    return cleaned_data

def save_json_data(data, filepath):
    """Saves data to a JSON file."""
    logging.info(f"Saving cleaned data to: {filepath}")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) # Use indent=4 for readability
        logging.info(f"Successfully saved cleaned data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean processed path data by removing entries with no paths and keeping only the shortest paths.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file (output of process_subgraph.py).")
    parser.add_argument("--output_file", required=True, help="Path to save the cleaned output JSON file.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load
    raw_data = load_json_data(args.input_file)

    # Clean
    cleaned_results = clean_path_data(raw_data)

    # Save
    save_json_data(cleaned_results, args.output_file)

    logging.info("Script finished.")