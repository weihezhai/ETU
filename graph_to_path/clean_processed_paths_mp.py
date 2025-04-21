# graph_to_path/clean_processed_paths_mp.py
import json
import argparse
import logging
import os
from collections import OrderedDict, defaultdict
import time # Added for timing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... load_json_data remains the same ...
def load_json_data(filepath):
    """Loads data from a JSON file (expects a single JSON object/list)."""
    logging.info(f"Loading JSON data from: {filepath}")
    start_time = time.time()
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Assuming the output of process_subgraph_ner_multicpu2.py is a single JSON list
            data = json.load(f)
        duration = time.time() - start_time
        logging.info(f"Successfully loaded {len(data)} entries from: {filepath} in {duration:.2f}s")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        raise

def clean_path_data(data):
    """
    Cleans the processed path data based on specified criteria:
    1. Removes entries with no paths found at all.
    2. For each source entity in an entry, keeps its shortest hop paths.
    3. Keeps longer paths (not the shortest for a source) if they contain more than one golden relation.
    """
    logging.info("Starting path data cleaning...")
    start_time = time.time()
    cleaned_data = []
    # Entry level stats
    initial_entry_count = len(data)
    removed_no_paths_count = 0
    removed_empty_after_filter_count = 0
    kept_entry_count = 0
    # Path level stats
    total_paths_processed = 0
    paths_kept_shortest_count = 0
    paths_kept_multi_gold_count = 0
    paths_removed_wrong_start = 0
    paths_removed_filter_criteria = 0
    paths_removed_as_duplicates = 0 # Count duplicates removed at the end

    for entry in data:
        entry_id = entry.get('id', 'N/A')
        paths_dict = entry.get('paths', {})
        source_entities = set(entry.get('source_entities', [])) # Use set for quick lookup
        gold_relations = set(entry.get('gold_relations', []))

        # Check if there are any paths at all in the input entry
        has_any_paths_initially = False
        entry_initial_path_count = 0
        for hop_paths in paths_dict.values():
            entry_initial_path_count += len(hop_paths)
            if hop_paths:
                has_any_paths_initially = True
        total_paths_processed += entry_initial_path_count # Count all paths initially present

        if not has_any_paths_initially:
            removed_no_paths_count += 1
            logging.debug(f"Entry {entry_id}: Removed (no paths found initially).")
            continue

        # Determine the maximum hop key present to iterate safely
        hop_keys_int = sorted([int(k.replace('hop', '')) for k in paths_dict.keys() if k.endswith('hop')])
        if not hop_keys_int: # Should not happen if has_any_paths_initially is True, but safe check
             removed_no_paths_count += 1 # Count this unlikely case here too
             logging.debug(f"Entry {entry_id}: Removed (no valid hop keys found despite paths present?).")
             continue

        # Store paths to keep, organized by hop
        temp_kept_paths = defaultdict(list)
        # Track the minimum hop level found for each source entity
        min_hop_found_for_source = defaultdict(lambda: float('inf'))

        # --- Iterate through paths to apply filtering rules ---
        for hop_level in hop_keys_int:
            hop_key = f'{hop_level}hop'
            current_hop_paths = paths_dict.get(hop_key, [])

            for path in current_hop_paths:
                if not path: continue # Skip empty paths just in case
                start_node = path[0]

                # Ensure path starts with one of the designated source entities for this entry
                if start_node not in source_entities:
                    paths_removed_wrong_start += 1
                    logging.warning(f"Entry {entry_id}: Path {path} starts with node {start_node} which is not in source_entities {source_entities}. Skipping this path.")
                    continue

                is_shortest = False
                path_kept = False # Flag to check if path was kept by any rule

                # Rule 2: Keep the shortest path(s) for each source entity
                if hop_level <= min_hop_found_for_source[start_node]:
                    # If it's strictly shorter, update the minimum
                    if hop_level < min_hop_found_for_source[start_node]:
                         min_hop_found_for_source[start_node] = hop_level
                    # Keep the path (either new shortest or same as current shortest)
                    temp_kept_paths[hop_key].append(path)
                    paths_kept_shortest_count += 1
                    is_shortest = True
                    path_kept = True

                # Rule 3: Keep longer paths if they contain > 1 golden relation
                if not is_shortest: # Only apply if it wasn't kept as a shortest path
                    path_relations = set(path[i] for i in range(1, len(path), 2))
                    common_gold_rels = path_relations.intersection(gold_relations)
                    if len(common_gold_rels) > 1:
                        temp_kept_paths[hop_key].append(path)
                        paths_kept_multi_gold_count += 1
                        path_kept = True

                # If the path wasn't kept by any rule, count it as filtered
                if not path_kept:
                    paths_removed_filter_criteria += 1


        # --- Finalize the paths for this entry ---
        final_paths_for_entry = OrderedDict()
        has_kept_paths = False
        max_hop_in_original = max(hop_keys_int) if hop_keys_int else 0
        entry_paths_before_dedup = 0
        entry_paths_after_dedup = 0

        for k in range(1, max_hop_in_original + 1):
            hop_str = f'{k}hop'
            paths_at_this_hop = temp_kept_paths.get(hop_str, [])
            entry_paths_before_dedup += len(paths_at_this_hop)
            if paths_at_this_hop:
                # Deduplicate paths within this hop level
                unique_paths_tuples = set(tuple(p) for p in paths_at_this_hop)
                deduplicated_paths = [list(p) for p in unique_paths_tuples]
                final_paths_for_entry[hop_str] = deduplicated_paths
                entry_paths_after_dedup += len(deduplicated_paths)
                if deduplicated_paths: # Check if list is non-empty after deduplication
                    has_kept_paths = True
            else:
                 final_paths_for_entry[hop_str] = [] # Ensure key exists

        paths_removed_as_duplicates += (entry_paths_before_dedup - entry_paths_after_dedup)

        # --- Add entry to cleaned_data if it has kept paths ---
        if has_kept_paths:
            entry['paths'] = final_paths_for_entry
            cleaned_data.append(entry)
            kept_entry_count += 1
            logging.debug(f"Entry {entry_id}: Kept with {entry_paths_after_dedup} paths after filtering and deduplication.")
        else:
            removed_empty_after_filter_count += 1
            logging.debug(f"Entry {entry_id}: Removed (no paths met keeping criteria or survived deduplication).")

    duration = time.time() - start_time
    logging.info(f"Cleaning complete in {duration:.2f}s.")
    logging.info(f"--- Entry Statistics ---")
    logging.info(f"  Initial Entries: {initial_entry_count}")
    logging.info(f"  Kept Entries: {kept_entry_count}")
    logging.info(f"  Removed (no paths initially): {removed_no_paths_count}")
    logging.info(f"  Removed (empty after filtering): {removed_empty_after_filter_count}")
    logging.info(f"--- Path Statistics ---")
    logging.info(f"  Total Paths Processed (Initial): {total_paths_processed}")
    total_kept = paths_kept_shortest_count + paths_kept_multi_gold_count
    logging.info(f"  Paths Kept (Before Deduplication): {total_kept}")
    logging.info(f"    - Kept as Shortest: {paths_kept_shortest_count}")
    logging.info(f"    - Kept as Multi-Gold: {paths_kept_multi_gold_count}")
    total_removed = paths_removed_wrong_start + paths_removed_filter_criteria + paths_removed_as_duplicates
    logging.info(f"  Paths Removed: {total_removed}")
    logging.info(f"    - Removed (Wrong Start Node): {paths_removed_wrong_start}")
    logging.info(f"    - Removed (Filter Criteria Not Met): {paths_removed_filter_criteria}")
    logging.info(f"    - Removed (Duplicate within entry/hop): {paths_removed_as_duplicates}")
    # Sanity check: initial paths = kept_before_dedup + removed_wrong_start + removed_filter_criteria
    # Note: paths_removed_as_duplicates relates kept_before_dedup to the final count in the output file.
    if total_paths_processed != total_kept + paths_removed_wrong_start + paths_removed_filter_criteria:
         logging.warning(f"Path count mismatch detected! Processed: {total_paths_processed}, "
                         f"Accounted For (Kept + Filtered): {total_kept + paths_removed_wrong_start + paths_removed_filter_criteria}")

    return cleaned_data

# ... save_json_data remains the same ...
def save_json_data(data, filepath):
    """Saves data to a JSON file."""
    logging.info(f"Saving {len(data)} cleaned entries to: {filepath}")
    start_time = time.time()
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             logging.info(f"Created output directory: {output_dir}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) # Use indent=4 for readability
        duration = time.time() - start_time
        logging.info(f"Successfully saved cleaned data to {filepath} in {duration:.2f}s")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean processed path data based on shortest paths per source and multi-golden relations.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file (output of process_subgraph*.py).")
    parser.add_argument("--output_file", required=True, help="Path to save the cleaned output JSON file.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    script_start_time = time.time()
    # Load
    raw_data = load_json_data(args.input_file)

    # Clean
    cleaned_results = clean_path_data(raw_data)

    # Save
    save_json_data(cleaned_results, args.output_file)

    script_duration = time.time() - script_start_time
    logging.info(f"Script finished successfully in {script_duration:.2f}s.")