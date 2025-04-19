# graph_to_path/process_subgraph_multicpu.py
import json
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
import os
import multiprocessing # Added multiprocessing
import functools     # Added functools
import time          # Added time

# Setup logging - Added processName
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')

# --- load_json remains the same ---
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
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip() # Remove leading/trailing whitespace
                    if not line: # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                        line_count += 1
                        # Log progress less frequently
                        if line_count % 10000 == 0:
                            logging.info(f"  Loaded {line_count} lines from {filepath}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON line {line_num} in {filepath}: {e}")
                        logging.error(f"Problematic line content: {line[:100]}...")
                        raise
                logging.info(f"Successfully loaded {len(data)} objects (JSON Lines format) from: {filepath}")
                return data
            else:
                try:
                    data = json.load(f)
                    object_type = type(data).__name__
                    try:
                        size = len(data)
                        logging.info(f"Successfully loaded 1 object (type: {object_type}, size: {size}) from: {filepath}")
                    except TypeError:
                        logging.info(f"Successfully loaded 1 object (type: {object_type}) from: {filepath}")
                    return data
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding single JSON object in {filepath}: {e}")
                    try:
                        f.seek(0)
                        snippet = f.read(200)
                        logging.error(f"File snippet (first 200 chars): {snippet}...")
                    except Exception:
                        pass
                    raise

    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        raise


# --- build_graph remains the same ---
def build_graph(triples):
    """Builds an undirected networkx MultiGraph from triples."""
    G = nx.MultiDiGraph() # Changed from MultiDiGraph to MultiGraph
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
    if source_node not in G or target_node not in G:
        # logging.debug(f"Source ({source_node}) or Target ({target_node}) not in graph. Skipping path finding.") # Reduced verbosity
        return []

    found_paths = []
    queue = [(source_node, [source_node])]
    # logging.debug(f"Starting BFS from {source_node} to {target_node} (max_hops={max_hops})") # Reduced verbosity

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
                        # logging.debug(f"  Found path: {new_path}") # Reduced verbosity
                        found_paths.append(new_path)
                    if current_num_hops + 1 < max_hops:
                         queue.append((neighbor, new_path))

    # logging.debug(f"BFS from {source_node} to {target_node} finished. Found {len(found_paths)} paths.") # Reduced verbosity
    return found_paths


# --- New helper function to load data ---
def load_data_and_mappings(subgraph_file, kb_id_map_file, golden_relations_file):
    """Loads all necessary data files."""
    logging.info("Loading input data files...")
    subgraph_data = load_json(subgraph_file, is_json_lines=True)
    kb_id_map = load_json(kb_id_map_file, is_json_lines=False) # Map: KB ID ('Qxxx') -> integer ID
    golden_relations_data = load_json(golden_relations_file, is_json_lines=False) # Map: entry_id -> [rel_id1, ...]
    logging.info("Input data loading complete.")
    return subgraph_data, kb_id_map, golden_relations_data


# --- New function to process a single entry ---
def process_single_entry(entry, kb_id_map, golden_relations_data, max_hops):
    """Processes a single entry from the subgraph data. Designed for multiprocessing."""
    try:
        entry_id = entry.get('id')
        question = entry.get('question', 'N/A')
        source_nodes = entry.get('entities', []) # Original source entities (integer IDs)
        answers_info = entry.get('answers', []) # List of answer dicts [{'kb_id': 'Qxxx', ...}]
        subgraph_triples = entry.get('subgraph', {}).get('tuples', [])

        # Basic validation
        if not entry_id or not source_nodes or not answers_info or not subgraph_triples:
            # logging.warning(f"Skipping entry {entry_id} due to missing critical data.") # Reduce log noise
            return None, True # Indicate skipped

        # logging.info(f"Processing entry ID: {entry_id}") # Reduce log noise

        # Base output structure in case of early exit
        base_output_entry = {
            'id': entry_id, 'answers': answers_info, 'question': question,
            'gold_relations': golden_relations_data.get(entry_id, []),
            'paths': {f'{k}hop': [] for k in range(1, max_hops + 1)}
        }

        # Map answer KB IDs to integer IDs
        target_nodes = set()
        for ans in answers_info:
            kb_id = ans.get('kb_id')
            if kb_id and kb_id in kb_id_map:
                target_id = kb_id_map[kb_id]
                target_nodes.add(target_id)
            # else: logging.warning(...) # Reduce log noise

        target_nodes = list(target_nodes)
        if not target_nodes:
            # logging.warning(f"No valid target node IDs found for entry {entry_id}.") # Reduce log noise
            return base_output_entry, True # Indicate skipped

        # Build graph
        G = build_graph(subgraph_triples)

        # Validate source nodes exist in the built graph
        valid_source_nodes = [s for s in source_nodes if s in G]
        # if len(valid_source_nodes) != len(source_nodes): logging.warning(...) # Reduce log noise
        if not valid_source_nodes:
            # logging.warning(f"Entry {entry_id}: No valid source nodes found in graph.") # Reduce log noise
            return base_output_entry, True # Indicate skipped

        # Find paths between all valid source-target pairs
        all_paths = []
        for source_node in valid_source_nodes:
            for target_node in target_nodes:
                # Check target existence (find_paths_bfs also does this, but quick check is fine)
                if target_node not in G:
                    # logging.warning(...) # Reduce log noise
                    continue
                paths = find_paths_bfs(G, source_node, target_node, max_hops)
                all_paths.extend(paths)
        # logging.info(f"Entry {entry_id}: Found {len(all_paths)} raw paths.") # Reduce log noise

        # Filter paths based on golden relations
        gold_rels_list = golden_relations_data.get(entry_id, [])
        gold_rels_set = set(gold_rels_list)
        # if not gold_rels_set: logging.warning(...) # Reduce log noise

        filtered_paths_by_hop = defaultdict(list)
        if gold_rels_set and all_paths: # Only filter if needed
            for path in all_paths:
                path_relations = set(path[i] for i in range(1, len(path), 2))
                if not path_relations.isdisjoint(gold_rels_set):
                    num_hops = len(path) // 2
                    if 1 <= num_hops <= max_hops:
                        filtered_paths_by_hop[f'{num_hops}hop'].append(path)
            # logging.info(f"Entry {entry_id}: Kept {sum(len(v) for v in filtered_paths_by_hop.values())} paths.") # Reduce log noise

        final_filtered_paths = {f'{k}hop': filtered_paths_by_hop.get(f'{k}hop', []) for k in range(1, max_hops + 1)}

        output_entry = {
            'id': entry_id, 'answers': answers_info, 'question': question,
            'gold_relations': gold_rels_list, 'paths': final_filtered_paths
        }

        # Determine if skipped (i.e., no paths found/kept)
        skipped = all(not paths for paths in output_entry['paths'].values())

        return output_entry, skipped

    except Exception as e:
        logging.error(f"Error processing entry ID {entry.get('id', 'UNKNOWN')} in worker {os.getpid()}: {e}", exc_info=True)
        return None, True # Indicate skipped on error


# --- save_results remains the same ---
def save_results(results, output_file):
    """Saves the processed results to a JSON file."""
    logging.info(f"Saving {len(results)} processed entries to {output_file}...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4) # Use indent=4 for readability
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")
        raise


# --- process_subgraphs adapted for multiprocessing ---
def process_subgraphs(subgraph_file, kb_id_map_file, golden_relations_file, output_file, max_hops=4, num_workers=4):
    """
    Processes subgraph data using multiprocessing to find, filter, and store paths.
    """
    start_time = time.time()
    logging.info(f"Starting subgraph processing with {num_workers} workers...")

    # 1. Load Data Sequentially
    try:
        subgraph_data, kb_id_map, golden_relations_data = load_data_and_mappings(
            subgraph_file, kb_id_map_file, golden_relations_file
        )
        logging.info(f"Data loading complete. Time: {time.time() - start_time:.2f}s")
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
    total_entries = len(subgraph_data)

    # Prepare partial function for the worker
    worker_func = functools.partial(
        process_single_entry,
        kb_id_map=kb_id_map,
        golden_relations_data=golden_relations_data,
        max_hops=max_hops
    )

    processing_start_time = time.time()
    logging.info("Starting parallel processing of entries...")

    pool = None # Initialize pool to None
    try:
        # Using 'fork' start method can be faster if safe (no complex global state)
        # Default 'spawn' is generally safer but might be slower due to re-imports.
        # context = multiprocessing.get_context('fork')
        # pool = context.Pool(processes=num_workers)
        pool = multiprocessing.Pool(processes=num_workers) # Use default context for broader compatibility

        # chunksize can impact performance; adjust based on task duration/memory
        chunksize = max(1, total_entries // (num_workers * 4)) # Heuristic
        # chunksize=100 # Or fixed value

        # Use imap_unordered for potential speedup, wrap with tqdm
        pool_results = pool.imap_unordered(worker_func, subgraph_data, chunksize=chunksize)

        for i, result in enumerate(tqdm(pool_results, total=total_entries, desc="Processing entries")):
            if result is not None:
                output_entry, skipped = result
                if output_entry is not None: # Check if worker returned valid data
                    results.append(output_entry)
                    if skipped:
                        skipped_count += 1
                    else:
                        processed_count += 1
                else: # Worker returned None (likely due to error)
                    skipped_count += 1
            else: # Should not happen if worker_func always returns tuple, but handle defensively
                 skipped_count += 1

            # Optional periodic logging
            # if (i + 1) % 5000 == 0:
            #    logging.info(f"  Processed {i+1}/{total_entries} entries...")

        pool.close()
        pool.join()
        logging.info("Parallel processing finished.")

    except Exception as e:
         logging.error(f"An error occurred during parallel processing: {e}", exc_info=True)
         if pool:
             pool.terminate()
         logging.warning("Attempting to save any partial results collected before the error...")

    finally:
        if pool and pool._state == multiprocessing.pool.RUN:
             pool.close()
             pool.join()

    processing_duration = time.time() - processing_start_time
    logging.info(f"Parallel processing duration: {processing_duration:.2f}s")

    # 3. Save Results
    save_results(results, output_file)

    total_duration = time.time() - start_time
    logging.info(f"Processing complete. Total time: {total_duration:.2f}s")
    logging.info(f"Entries processed (paths found/filtered): {processed_count}, Entries skipped/empty paths/errors: {skipped_count}")


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
                # Leave one or two cores free for system processes if many cores available
                default_workers = max(1, cpu_count - 1) if cpu_count > 2 else 1
            else:
                 logging.warning("Could not determine CPU count, defaulting to 1 worker.")
                 default_workers = 1
        except NotImplementedError:
            logging.warning("os.cpu_count() not implemented, defaulting to 1 worker.")
            default_workers = 1

    parser = argparse.ArgumentParser(description="Find and filter multi-hop paths in subgraphs using multiprocessing.")
    parser.add_argument("--subgraph_file", required=True, help="Path to the input subgraph JSON Lines file.")
    parser.add_argument("--kb_map_file", required=True, help="Path to the KB ID to integer ID mapping JSON file.")
    parser.add_argument("--golden_rels_file", required=True, help="Path to the golden relations JSON file (map: entry_id -> [rel_id1, rel_id2]).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file.")
    parser.add_argument("--max_hops", type=int, default=4, help="Maximum number of hops for paths (default: 4).")
    parser.add_argument("--num_workers", type=int, default=default_workers, help=f"Number of worker processes to use (default: {default_workers}, based on SLURM/os.cpu_count()).") # Added num_workers
    parser.add_argument("--debug", action='store_true', help="Enable debug logging (Note: Can be verbose with multiprocessing).")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.warning("DEBUG logging enabled. Output may be very verbose with multiprocessing.")

    process_subgraphs(
        subgraph_file=args.subgraph_file,
        kb_id_map_file=args.kb_map_file,
        golden_relations_file=args.golden_rels_file,
        output_file=args.output_file,
        max_hops=args.max_hops,
        num_workers=args.num_workers # Pass num_workers
    )