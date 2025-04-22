import json
import argparse
import logging
import os
from collections import OrderedDict, defaultdict, Counter

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
    2. For each starting entity, keeps all paths of the minimum hop count.
    3. Adds a filter for 4-hop paths: they must contain at least two entities
       that are also source entities within the same entry.
    Logs detailed statistics about the process.
    """
    logging.info("Starting path data cleaning and refinement...")
    cleaned_data = []

    # --- Statistics Initialization ---
    total_entries_in = len(data)
    entries_removed_no_initial_paths = 0
    entries_removed_no_kept_paths = 0
    entries_processed_count = 0
    initial_paths_total = 0
    initial_paths_by_hop = Counter()
    kept_paths_total_unique = 0
    kept_paths_by_hop = Counter()
    four_hop_paths_considered_mention_rule = 0 # New Stat
    four_hop_paths_filtered_mention_rule = 0 # New Stat
    # Removed stats related to multi-gold rule
    # --------------------------------

    for entry in data:
        entry_id = entry.get('id', 'N/A')
        paths_dict = entry.get('paths', {})

        # --- Collect all unique source entities for this entry ---
        all_source_entities_in_entry = set()
        current_entry_initial_paths = 0
        for hop_key, paths_list in paths_dict.items():
             count = len(paths_list)
             initial_paths_total += count
             current_entry_initial_paths += count
             if count > 0:
                 try:
                     hop_count = int(hop_key.replace('hop', ''))
                     initial_paths_by_hop[hop_count] += count
                     # Collect source entities
                     for path in paths_list:
                         if path:
                             all_source_entities_in_entry.add(path[0])
                 except ValueError:
                     pass # Ignore malformed keys for stats

        # --- 1. Check if entry has any paths ---
        if current_entry_initial_paths == 0:
            logging.debug(f"Entry {entry_id}: Removed (no paths found initially).")
            entries_removed_no_initial_paths += 1
            continue

        kept_paths_set = set() # Use set of tuples to store unique paths to keep
        paths_by_source = defaultdict(lambda: defaultdict(list)) # {source: {hop: [paths]}}
        # Removed kept_reason_multi_gold

        # --- Group paths by source and hop ---
        for hop_key, paths_list in paths_dict.items():
            if not paths_list:
                continue
            try:
                hop_count = int(hop_key.replace('hop', ''))
            except ValueError:
                logging.warning(f"Entry {entry_id}: Skipping unexpected path key '{hop_key}'")
                continue

            for path in paths_list:
                if not path: continue
                start_node = path[0]
                path_tuple = tuple(path)
                paths_by_source[start_node][hop_count].append(path_tuple)

        # --- 2. Find and keep shortest paths for each source, applying 4-hop filter ---
        for start_node, hop_paths_dict in paths_by_source.items():
            if not hop_paths_dict: continue

            min_hops = min(hop_paths_dict.keys())
            shortest_paths_for_source = hop_paths_dict[min_hops]

            if min_hops == 4:
                # Apply the new filter for 4-hop paths
                for path_tuple in shortest_paths_for_source:
                    four_hop_paths_considered_mention_rule += 1
                    path_nodes = path_tuple[::2] # Extract nodes (at even indices)
                    # Count intersections with all source entities, excluding the path's own start node
                    # Ensure we compare against the full set of source entities for the *entry*
                    mentioned_source_entities = {node for node in path_nodes[1:] if node in all_source_entities_in_entry}

                    if len(mentioned_source_entities) >= 2:
                        if path_tuple not in kept_paths_set:
                            logging.debug(f"Entry {entry_id}: Keeping 4-hop path {path_tuple} (shortest for {start_node}, >=2 other sources mentioned)")
                            kept_paths_set.add(path_tuple)
                    else:
                        logging.debug(f"Entry {entry_id}: Filtering 4-hop path {path_tuple} (shortest for {start_node}, mentions {len(mentioned_source_entities)} other sources, required >=2)")
                        four_hop_paths_filtered_mention_rule += 1
            else:
                # Keep all shortest paths (non-4-hop)
                for path_tuple in shortest_paths_for_source:
                     if path_tuple not in kept_paths_set:
                         logging.debug(f"Entry {entry_id}: Keeping path {path_tuple} (shortest path for source {start_node} at {min_hops} hops)")
                         kept_paths_set.add(path_tuple)


        # --- Reconstruct the entry if paths were kept ---
        if kept_paths_set:
            new_paths_dict = defaultdict(list)
            max_hop_found = 0
            for path_tuple in kept_paths_set:
                path_list = list(path_tuple)
                num_hops = len(path_list) // 2
                new_paths_dict[f'{num_hops}hop'].append(path_list)
                max_hop_found = max(max_hop_found, num_hops)
                # --- Stat: Count kept paths by hop ---
                kept_paths_by_hop[num_hops] += 1

            original_max_hops = 0
            for k in paths_dict.keys():
                 try: original_max_hops = max(original_max_hops, int(k.replace('hop','')))
                 except: pass

            final_paths_output = OrderedDict()
            # Ensure all original hop keys are present for consistency, even if empty
            for k in range(1, max(original_max_hops, max_hop_found) + 1):
                 hop_key = f'{k}hop'
                 final_paths_output[hop_key] = new_paths_dict.get(hop_key, [])


            entry['paths'] = final_paths_output
            cleaned_data.append(entry)
            entries_processed_count += 1
            # --- Stat: Count total kept paths ---
            num_kept_this_entry = len(kept_paths_set)
            kept_paths_total_unique += num_kept_this_entry
            # Removed multi-gold stats increments

            logging.debug(f"Entry {entry_id}: Kept {num_kept_this_entry} unique paths after shortest-path filtering.")
        else:
            logging.debug(f"Entry {entry_id}: Removed (no paths met shortest-path criteria).")
            entries_removed_no_kept_paths += 1


    # --- Final Statistics Logging (Updated) ---
    logging.info("="*30 + " Cleaning Statistics " + "="*30)
    logging.info(f"Total entries received: {total_entries_in}")
    logging.info(f"Entries removed (had 0 paths initially): {entries_removed_no_initial_paths}")
    logging.info(f"Entries removed (no paths met criteria after filtering): {entries_removed_no_kept_paths}")
    logging.info(f"Total entries removed: {entries_removed_no_initial_paths + entries_removed_no_kept_paths}")
    logging.info(f"Entries kept: {entries_processed_count}")
    logging.info("-" * 70)
    logging.info(f"Total initial paths across all entries: {initial_paths_total}")
    logging.info(f"Initial path distribution by hop: {dict(initial_paths_by_hop)}")
    logging.info("-" * 70)
    logging.info(f"Total unique paths kept across final entries (shortest per source + 4-hop rule): {kept_paths_total_unique}")
    logging.info(f"Kept path distribution by hop: {dict(kept_paths_by_hop)}")
    if entries_processed_count > 0:
        avg_paths_per_kept_entry = kept_paths_total_unique / entries_processed_count
        logging.info(f"Average paths per kept entry: {avg_paths_per_kept_entry:.2f}")
    else:
        logging.info("Average paths per kept entry: N/A (0 entries kept)")
    logging.info("-" * 70)
    # Removed logging specific to multi-gold criteria
    logging.info("--- 4-Hop Path Filter Statistics ---")
    logging.info(f"Total 4-hop paths considered by mention rule: {four_hop_paths_considered_mention_rule}")
    logging.info(f"4-hop paths filtered out by mention rule (<2 source mentions): {four_hop_paths_filtered_mention_rule}")
    kept_4_hop_mention = four_hop_paths_considered_mention_rule - four_hop_paths_filtered_mention_rule
    logging.info(f"4-hop paths kept after mention rule: {kept_4_hop_mention}")
    logging.info(f"Note: Kept paths represent the shortest path for each source,")
    logging.info(f"      with the additional constraint for 4-hop paths.")
    logging.info("="*70)

    logging.info(f"Cleaning complete. Kept {entries_processed_count} entries. Removed {entries_removed_no_initial_paths + entries_removed_no_kept_paths} entries.")
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