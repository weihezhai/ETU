# graph_to_path/process_subgraph_ner_multicpu2.py
import json
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
import os
# import spacy # Removed spacy import
import multiprocessing # Added multiprocessing
import functools # Added functools
import time # For basic timing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s') # Added processName

# Removed Global variable for Spacy model
# Removed init_worker function

# --- Load function remains the same ---
def load_json(filepath, is_json_lines=True):
    """Loads data from a JSON file. Handles JSON Lines or single JSON object."""
    logging.info(f"Loading JSON data from: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if is_json_lines:
                data = []
                line_count = 0 # Keep track even without logging every line
                for line_num, line in enumerate(f, 1):
                    line = line.strip() # Remove leading/trailing whitespace
                    if not line: # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                        line_count += 1
                        # Log progress less frequently for potentially faster loading
                        if line_count % 10000 == 0:
                             logging.info(f"  Loaded {line_count} lines from {filepath}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON line {line_num} in {filepath}: {e}")
                        logging.error(f"Problematic line content: {line[:100]}...") # Log first 100 chars
                        raise # Re-raise after logging details
                logging.info(f"Successfully loaded {len(data)} objects (JSON Lines format) from: {filepath}")
                return data
            else:
                # Load the entire file as a single JSON object
                try:
                    data = json.load(f)
                    object_type = type(data).__name__
                    # Attempt to get a meaningful size (e.g., number of keys for dict)
                    try:
                        size = len(data)
                        logging.info(f"Successfully loaded 1 object (type: {object_type}, size: {size}) from: {filepath}")
                    except TypeError: # Handle types without len() like int, bool
                        logging.info(f"Successfully loaded 1 object (type: {object_type}) from: {filepath}")
                    return data
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding single JSON object in {filepath}: {e}")
                    # Attempt to read a snippet for context if possible (might fail on large files)
                    try:
                        f.seek(0)
                        snippet = f.read(200)
                        logging.error(f"File snippet (first 200 chars): {snippet}...")
                    except Exception:
                        pass
                    raise # Re-raise after logging details

    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        raise

# --- build_graph remains the same ---
def build_graph(triples):
    """Builds an undirected networkx MultiGraph from triples."""
    G = nx.MultiGraph() # Changed from MultiDiGraph to MultiGraph
    # logging.debug(f"Building undirected graph from {len(triples)} triples.") # Reduced verbosity
    for head, rel, tail in triples:
        G.add_edge(head, tail, key=rel, relation_id=rel)
    # logging.debug(f"Undirected graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.") # Reduced verbosity
    return G

# --- find_paths_bfs remains the same ---
def find_paths_bfs(G, source_node, target_node, max_hops):
    """
    Finds simple paths (no repeated nodes) up to max_hops using BFS,
    including relation IDs.
    Returns paths in the format [node1, rel1, node2, rel2, ..., node_k+1].
    """
    # Reduced verbosity for parallel execution
    if source_node not in G or target_node not in G:
        # logging.debug(f"Source ({source_node}) or Target ({target_node}) not in graph. Skipping path finding.")
        return []

    found_paths = []
    queue = [(source_node, [source_node])]
    # logging.debug(f"Starting BFS from {source_node} to {target_node} (max_hops={max_hops})")

    while queue:
        current_node, path = queue.pop(0)
        current_num_hops = len(path) // 2
        if current_num_hops >= max_hops:
            continue
        if current_node not in G:
             continue

        neighbors_data = G.adj.get(current_node, {})
        for neighbor, edge_dict in neighbors_data.items():
            relations = [data['relation_id'] for key, data in edge_dict.items()]
            for rel in relations:
                if neighbor not in path[::2]:
                    new_path = path + [rel, neighbor]
                    if neighbor == target_node:
                        # logging.debug(f"  Found path: {new_path}")
                        found_paths.append(new_path)
                    if current_num_hops + 1 < max_hops:
                         queue.append((neighbor, new_path))

    # logging.debug(f"BFS from {source_node} to {target_node} finished. Found {len(found_paths)} paths.")
    return found_paths


# --- load_data_and_mappings updated ---
# Removed entity_labels_file argument
def load_data_and_mappings(subgraph_file, kb_id_map_file, golden_relations_file):
    """Loads necessary data files and creates required mappings (excluding entity labels)."""
    subgraph_data = load_json(subgraph_file, is_json_lines=True)
    kb_id_to_int_id_map = load_json(kb_id_map_file, is_json_lines=False)
    # Removed loading of entity_labels_file
    golden_relations_data = load_json(golden_relations_file, is_json_lines=False)

    # Removed label_to_int_id_map creation logic

    # Removed label_to_int_id_map from return
    return subgraph_data, kb_id_to_int_id_map, golden_relations_data

# --- Removed extract_ner_entities function ---

# --- map_answers_to_target_ids remains the same ---
def map_answers_to_target_ids(answers_info, kb_id_to_int_id_map, entry_id):
    """Maps answer KB IDs to integer IDs."""
    # (Code is identical to previous version)
    target_nodes = set()
    valid_answers_info = []
    for ans in answers_info:
        kb_id = ans.get('kb_id')
        if kb_id and kb_id in kb_id_to_int_id_map:
            target_id = kb_id_to_int_id_map[kb_id]
            target_nodes.add(target_id)
            valid_answers_info.append(ans) # Keep original info for mapped answers
        else:
            # Reduced verbosity for parallel execution
            # logging.warning(f"KB ID '{kb_id}' for answer in entry {entry_id} not found in kb_id_map.")
            pass # Or log less frequently if needed
    return list(target_nodes), valid_answers_info

# --- find_and_filter_paths remains the same ---
def find_and_filter_paths(G, valid_source_nodes, target_nodes, gold_rels_set, max_hops, entry_id):
    """Finds paths using BFS and filters them based on golden relations."""
    # (Code is identical to previous version, maybe reduce logging verbosity)
    all_paths = []
    # logging.debug(f"Finding paths for entry {entry_id} from sources {valid_source_nodes} to targets {target_nodes}...")
    for source_node in valid_source_nodes:
        for target_node in target_nodes:
            if target_node not in G:
                # logging.warning(f"Entry {entry_id}: Target node {target_node} not found in graph. Skipping paths to it.")
                continue
            paths = find_paths_bfs(G, source_node, target_node, max_hops)
            all_paths.extend(paths)
    # logging.info(f"Entry {entry_id}: Found {len(all_paths)} raw paths before filtering.")

    filtered_paths_by_hop = defaultdict(list)
    # if not gold_rels_set:
    #     logging.warning(f"Entry {entry_id}: No golden relations found. No paths will be kept.")
    # elif not all_paths:
    #     logging.info(f"Entry {entry_id}: No raw paths found between sources and targets.")
    if gold_rels_set and all_paths: # Only filter if necessary
        # logging.debug(f"Filtering {len(all_paths)} paths using {len(gold_rels_set)} golden relations: {gold_rels_set}")
        for path in all_paths:
            path_relations = set(path[i] for i in range(1, len(path), 2))
            if not path_relations.isdisjoint(gold_rels_set):
                num_hops = len(path) // 2
                if 1 <= num_hops <= max_hops:
                    filtered_paths_by_hop[f'{num_hops}hop'].append(path)
        # logging.info(f"Entry {entry_id}: Kept {sum(len(v) for v in filtered_paths_by_hop.values())} paths after filtering.")

    final_filtered_paths = {f'{k}hop': filtered_paths_by_hop.get(f'{k}hop', []) for k in range(1, max_hops + 1)}
    return final_filtered_paths

# --- process_single_entry updated ---
# Removed label_to_int_id_map argument
# Removed NER processing
def process_single_entry(entry, kb_id_to_int_id_map, golden_relations_data, max_hops):
    """Processes a single entry from the subgraph data. Designed to be called by multiprocessing Pool."""
    try: # Add try/except block for better error isolation in workers
        entry_id = entry.get('id')
        question = entry.get('question', '')
        original_source_nodes = entry.get('entities', [])
        answers_info = entry.get('answers', [])
        subgraph_triples = entry.get('subgraph', {}).get('tuples', [])

        # Basic check
        if not entry_id or not answers_info or not subgraph_triples:
            logging.warning(f"Skipping entry due to missing critical data: id={entry_id}...") # Reduce log noise
            return None, True # Indicate skipped

        logging.info(f"Processing entry ID: {entry_id}") # Reduce log noise

        # 1. Use only original source nodes (NER removed)
        final_source_nodes = list(set(original_source_nodes)) # Use original entities directly, ensure uniqueness
        # Removed NER call and combining logic
        # Removed ner_added_count tracking

        base_output_entry = {
            'id': entry_id, 'answers': answers_info, 'question': question,
            'source_entities': final_source_nodes,
            'gold_relations': golden_relations_data.get(entry_id, []),
            'paths': {f'{k}hop': [] for k in range(1, max_hops + 1)}
        }

        if not final_source_nodes:
            # logging.warning(f"Skipping entry {entry_id} as no source entities were found...") # Reduce log noise
            return base_output_entry, True # Indicate skipped (empty paths)

        target_nodes, _ = map_answers_to_target_ids(answers_info, kb_id_to_int_id_map, entry_id)
        if not target_nodes:
            # logging.warning(f"No valid target node IDs found for entry {entry_id}...") # Reduce log noise
            return base_output_entry, True # Indicate skipped (empty paths)

        G = build_graph(subgraph_triples)
        valid_source_nodes = [s for s in final_source_nodes if s in G]
        # if len(valid_source_nodes) != len(final_source_nodes):
            # logging.warning(f"Entry {entry_id}: Some source nodes not found in graph...") # Reduce log noise
        if not valid_source_nodes:
            # logging.warning(f"Entry {entry_id}: No valid source nodes found in graph...") # Reduce log noise
            return base_output_entry, True # Indicate skipped (empty paths)

        gold_rels_list = golden_relations_data.get(entry_id, [])
        gold_rels_set = set(gold_rels_list)
        filtered_paths = find_and_filter_paths(G, valid_source_nodes, target_nodes, gold_rels_set, max_hops, entry_id)

        output_entry = {
            'id': entry_id, 'answers': answers_info, 'question': question,
            'source_entities': final_source_nodes,
            'gold_relations': gold_rels_list, 'paths': filtered_paths
        }

        # Determine if processed or skipped based on paths
        skipped = all(not paths for paths in output_entry['paths'].values())

        # Removed ner_added_count from return
        return output_entry, skipped

    except Exception as e:
        logging.error(f"Error processing entry ID {entry.get('id', 'UNKNOWN')} in worker {os.getpid()}: {e}", exc_info=True)
        # Return something to indicate failure, or re-raise depending on desired behavior
        return None, True # Treat as skipped on error


# --- save_results remains the same ---
def save_results(results, output_file):
    """Saves the processed results to a JSON file."""
    # (Code is identical to previous version)
    logging.info(f"Saving {len(results)} processed entries to {output_file}...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")
        raise


# --- process_subgraphs updated ---
# Removed entity_labels_file and spacy_model_name arguments
def process_subgraphs(subgraph_file, kb_id_map_file, golden_relations_file, output_file, max_hops=4, num_workers=4):
    """
    Main function to process subgraph data using multiprocessing.
    """
    start_time = time.time()
    logging.info(f"Starting subgraph processing with {num_workers} workers...")

    # 1. Load Data and Mappings (Sequentially in main process)
    try:
        # Removed entity_labels_file from call and return
        subgraph_data, kb_id_to_int_id_map, golden_relations_data = load_data_and_mappings(
            subgraph_file, kb_id_map_file, golden_relations_file
        )
        logging.info(f"Data loading and mapping creation complete. Time: {time.time() - start_time:.2f}s")
    except FileNotFoundError:
        logging.error("Exiting due to missing input file.")
        return
    except Exception as e:
        logging.error(f"Exiting due to error during data loading: {e}")
        return

    # 2. Process Entries in Parallel
    results = []
    processed_count = 0
    skipped_count = 0
    # Removed total_ner_added_sources
    total_entries = len(subgraph_data)

    # Prepare partial function with fixed arguments for the worker
    # The 'entry' argument will be supplied by imap
    # Removed label_to_int_id_map argument
    worker_func = functools.partial(
        process_single_entry,
        kb_id_to_int_id_map=kb_id_to_int_id_map,
        golden_relations_data=golden_relations_data,
        max_hops=max_hops
    )

    processing_start_time = time.time()
    logging.info("Starting parallel processing of entries...")

    # Use try/finally to ensure pool cleanup
    pool = None # Initialize pool to None
    try:
        # Create pool without initializer
        pool = multiprocessing.Pool(processes=num_workers) # Removed initializer and initargs

        # Use imap_unordered for potentially better performance if order doesn't matter
        chunksize=100 # Adjust as needed

        # Wrap the imap_unordered iterator with tqdm for progress tracking
        pool_results = pool.imap_unordered(worker_func, subgraph_data, chunksize=chunksize)

        # Updated loop to unpack result without ner_added_count
        for i, result in enumerate(tqdm(pool_results, total=total_entries, desc="Processing entries")):
            if result is not None:
                output_entry, skipped = result # Unpack updated return value
                # Removed total_ner_added_sources update
                if output_entry is not None: # Check if the worker returned a valid entry (not None due to error)
                    results.append(output_entry)
                    if skipped:
                        skipped_count += 1
                    else:
                        processed_count += 1
                else: # Worker encountered an error and returned None
                    skipped_count += 1 # Count as skipped if error occurred
            else: # Handles Nones if worker_func somehow returns None directly
                 skipped_count += 1

            # Optional: Log progress periodically
            # if (i + 1) % 1000 == 0:
            #    logging.info(f"  Processed {i+1}/{total_entries} entries...")

        pool.close() # No more tasks will be submitted
        pool.join()  # Wait for all worker processes to finish
        logging.info("Parallel processing finished.")

    except Exception as e:
         logging.error(f"An error occurred during parallel processing: {e}", exc_info=True)
         if pool:
             pool.terminate() # Terminate pool forcefully on unexpected error
         # Handle partial results or exit
         logging.warning("Attempting to save any partial results collected before the error...")

    finally:
        if pool and pool._state == multiprocessing.pool.RUN: # Ensure pool is closed/joined if exception didn't occur in try
             pool.close()
             pool.join()


    processing_duration = time.time() - processing_start_time
    logging.info(f"Parallel processing duration: {processing_duration:.2f}s")

    # 3. Save Results
    save_results(results, output_file)

    total_duration = time.time() - start_time
    logging.info(f"Processing complete. Total time: {total_duration:.2f}s")
    logging.info(f"Entries processed (paths found/filtered): {processed_count}, Entries skipped/empty paths/errors: {skipped_count}")
    # Removed NER logging
    # logging.info(f"Total source nodes added via NER across all entries: {total_ner_added_sources}")


if __name__ == "__main__":
    # Determine default number of workers
    default_workers = 1
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        try:
            default_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        except ValueError:
            logging.warning("Could not parse SLURM_CPUS_PER_TASK, defaulting to 1 worker.")
            default_workers = 1
    else:
        try:
            cpu_count = os.cpu_count()
            if cpu_count:
                default_workers = cpu_count
            else:
                 logging.warning("Could not determine CPU count, defaulting to 1 worker.")
                 default_workers = 1
        except NotImplementedError:
            logging.warning("os.cpu_count() not implemented, defaulting to 1 worker.")
            default_workers = 1

    parser = argparse.ArgumentParser(description="Find and filter multi-hop paths in subgraphs, using multiprocessing.") # Updated description
    parser.add_argument("--subgraph_file", required=True, help="Path to the input subgraph JSON Lines file.")
    parser.add_argument("--kb_map_file", required=True, help="Path to the KB ID ('Qxxx') to integer ID mapping JSON file.")
    # Removed entity_labels_file argument
    parser.add_argument("--golden_rels_file", required=True, help="Path to the golden relations JSON file (map: entry_id -> [rel_id1, rel_id2]).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file.")
    parser.add_argument("--max_hops", type=int, default=4, help="Maximum number of hops for paths (default: 4).")
    parser.add_argument("--num_workers", type=int, default=default_workers, help=f"Number of worker processes to use (default: {default_workers}, based on SLURM/os.cpu_count()).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging (Note: Can be very verbose with multiprocessing).")
    # Removed spacy_model argument

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.warning("DEBUG logging enabled. Output may be very verbose with multiprocessing.")
    # Removed spacy logging configuration
    # else:
    #     logging.getLogger("spacy").setLevel(logging.ERROR)


    # Removed spacy model loading call

    # Updated process_subgraphs call arguments
    process_subgraphs(
        subgraph_file=args.subgraph_file,
        kb_id_map_file=args.kb_map_file,
        golden_relations_file=args.golden_rels_file,
        output_file=args.output_file,
        max_hops=args.max_hops,
        num_workers=args.num_workers
    )
