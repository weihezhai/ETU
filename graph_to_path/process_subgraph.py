# graph_to_path/process_subgraphs.py
import json
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                for line_num, line in enumerate(f, 1):
                    line = line.strip() # Remove leading/trailing whitespace
                    if not line: # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                        # Log progress every 100 lines
                        if line_num % 1000 == 0:
                            logging.info(f"  Loaded {line_num} lines from {filepath}...")
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

def build_graph(triples):
    """Builds a networkx MultiDiGraph from triples."""
    G = nx.MultiDiGraph()
    logging.debug(f"Building graph from {len(triples)} triples.")
    for head, rel, tail in triples:
        # Add edge with relation_id as an attribute and also as the key
        # This helps distinguish parallel edges and retrieve relation easily
        G.add_edge(head, tail, key=rel, relation_id=rel)
    logging.debug(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def find_paths_bfs(G, source_node, target_node, max_hops):
    """
    Finds simple paths (no repeated nodes) up to max_hops using BFS,
    including relation IDs.
    Returns paths in the format [node1, rel1, node2, rel2, ..., node_k+1].
    """
    if source_node not in G or target_node not in G:
        logging.debug(f"Source ({source_node}) or Target ({target_node}) not in graph. Skipping path finding.")
        return []

    found_paths = []
    # Queue stores tuples: (current_node, path_so_far [n1, r1, n2, ... nk])
    queue = [(source_node, [source_node])] # Start with the source node in the path

    logging.debug(f"Starting BFS from {source_node} to {target_node} (max_hops={max_hops})")

    processed_paths = 0 # Debug counter

    while queue:
        current_node, path = queue.pop(0) # FIFO for BFS
        current_num_hops = len(path) // 2

        # Path length is num_hops. Stop exploring if max_hops is reached.
        if current_num_hops >= max_hops:
            continue

        # Explore neighbors
        if current_node not in G: # Node might exist but have no outgoing edges
             continue

        # G.adj[node] or G[node] gives successors with edge data in MultiDiGraph
        # Format: {neighbor: {key: {attr_dict}}}
        neighbors_data = G.adj.get(current_node, {})
        for neighbor, edge_dict in neighbors_data.items():
            processed_paths +=1 # Debug counter
            # if processed_paths % 10000 == 0: # Log progress for large searches
            #     logging.debug(f"  BFS progress: Explored {processed_paths} path segments from {source_node}...")

            relations = [data['relation_id'] for key, data in edge_dict.items()]

            for rel in relations:
                # Check for cycles: Ensure the neighbor isn't already in the node path
                # Nodes are at even indices: 0, 2, 4, ...
                if neighbor not in path[::2]:
                    new_path = path + [rel, neighbor]
                    if neighbor == target_node:
                        # Found a path to the target
                        logging.debug(f"  Found path: {new_path}")
                        found_paths.append(new_path)
                        # Continue searching for other paths up to max_hops
                        # Do not stop here, as we want all paths up to max_hops

                    # Enqueue the extended path to explore further,
                    # only if we haven't reached max_hops yet.
                    # The next hop will be current_num_hops + 1.
                    if current_num_hops + 1 < max_hops:
                         queue.append((neighbor, new_path))
                # else: cycle detected, prune this path extension

    logging.debug(f"BFS from {source_node} to {target_node} finished. Found {len(found_paths)} paths.")
    return found_paths


def process_subgraphs(subgraph_file, kb_id_map_file, golden_relations_file, output_file, max_hops=4):
    """
    Processes subgraph data to find, filter, and store paths between entities and answers.
    """
    logging.info("Starting subgraph processing...")

    # Load necessary data files
    subgraph_data = load_json(subgraph_file, is_json_lines=True) # Explicitly True (or default)
    kb_id_map = load_json(kb_id_map_file, is_json_lines=False) # This file is a single JSON object
    golden_relations_data = load_json(golden_relations_file, is_json_lines=False) # This should also be a single JSON object

    results = []
    processed_count = 0
    skipped_count = 0

    # Process each entry in the subgraph data
    for entry in tqdm(subgraph_data, desc="Processing entries"):
        entry_id = entry.get('id')
        question = entry.get('question', 'N/A') # Handle potentially missing question
        source_nodes = entry.get('entities', [])
        answers_info = entry.get('answers', [])
        subgraph_triples = entry.get('subgraph', {}).get('tuples', [])

        if not entry_id or not source_nodes or not answers_info or not subgraph_triples:
            logging.warning(f"Skipping entry due to missing critical data: id={entry_id}, "
                            f"has_sources={bool(source_nodes)}, has_answers={bool(answers_info)}, "
                            f"has_triples={bool(subgraph_triples)}")
            skipped_count += 1
            continue

        logging.info(f"Processing entry ID: {entry_id}")

        # Convert answer kb_ids to integer IDs using the map
        target_nodes = []
        valid_answers_info = [] # Keep track of answers we could map
        for ans in answers_info:
            kb_id = ans.get('kb_id')
            if kb_id and kb_id in kb_id_map:
                target_id = kb_id_map[kb_id]
                target_nodes.append(target_id)
                valid_answers_info.append(ans) # Keep original info for mapped answers
            else:
                logging.warning(f"KB ID '{kb_id}' for entry {entry_id} not found in kb_id_map.")

        # Remove duplicate target node IDs if any
        target_nodes = list(set(target_nodes))

        if not target_nodes:
            logging.warning(f"No valid target node IDs found for entry {entry_id} after mapping.")
            # Store entry with empty paths if desired, or skip fully
            output_entry = {
                'id': entry_id,
                'answers': answers_info, # Report original answers
                'question': question,
                'gold_relations': golden_relations_data.get(entry_id, []),
                'paths': {f'{k}hop': [] for k in range(1, max_hops + 1)}
            }
            results.append(output_entry)
            skipped_count += 1 # Count as skipped in terms of path finding
            continue

        # Build the graph from triples
        logging.debug(f"Building graph for entry {entry_id}...")
        G = build_graph(subgraph_triples)
        # Ensure source nodes exist in the graph built from triples
        valid_source_nodes = [s for s in source_nodes if s in G]
        if len(valid_source_nodes) != len(source_nodes):
            logging.warning(f"Entry {entry_id}: Some source nodes {[s for s in source_nodes if s not in G]} "
                            f"were not found in the graph constructed from triples.")
        if not valid_source_nodes:
            logging.warning(f"Entry {entry_id}: No valid source nodes found in the graph. Skipping path finding.")
            output_entry = {
                'id': entry_id,
                'answers': answers_info,
                'question': question,
                'gold_relations': golden_relations_data.get(entry_id, []),
                'paths': {f'{k}hop': [] for k in range(1, max_hops + 1)}
            }
            results.append(output_entry)
            skipped_count += 1
            continue


        # Find all paths between all valid source-target pairs
        all_paths = []
        logging.debug(f"Finding paths for entry {entry_id} from sources {valid_source_nodes} to targets {target_nodes}...")
        for source_node in valid_source_nodes:
            for target_node in target_nodes:
                # Check if target node is in the graph (it might not be if triples don't connect to it)
                if target_node not in G:
                    logging.warning(f"Entry {entry_id}: Target node {target_node} not found in graph. Skipping paths to it.")
                    continue

                # Use the BFS path finder
                paths = find_paths_bfs(G, source_node, target_node, max_hops)
                all_paths.extend(paths)
        logging.info(f"Entry {entry_id}: Found {len(all_paths)} raw paths before filtering.")


        # Get golden relations for filtering
        gold_rels_list = golden_relations_data.get(entry_id, [])
        gold_rels_set = set(gold_rels_list)
        if not gold_rels_set:
            logging.warning(f"Entry {entry_id}: No golden relations found.")
            # If no golden relations, no paths will be kept based on the filtering rule.

        # Filter paths: Keep only paths containing at least one golden relation
        filtered_paths_by_hop = defaultdict(list)
        if gold_rels_set: # Proceed with filtering only if there are golden relations
            logging.debug(f"Filtering {len(all_paths)} paths using {len(gold_rels_set)} golden relations: {gold_rels_set}")
            for path in all_paths:
                # Extract relation IDs from the path (at odd indices: 1, 3, 5, ...)
                path_relations = set(path[i] for i in range(1, len(path), 2))

                # Check for intersection between path relations and golden relations
                if not path_relations.isdisjoint(gold_rels_set):
                    num_hops = len(path) // 2
                    if 1 <= num_hops <= max_hops:
                        filtered_paths_by_hop[f'{num_hops}hop'].append(path)
            logging.info(f"Entry {entry_id}: Kept {sum(len(v) for v in filtered_paths_by_hop.values())} paths after filtering.")
        else:
            logging.info(f"Entry {entry_id}: No golden relations provided, so 0 paths kept after filtering.")


        # Format output for the current entry
        output_entry = {
            'id': entry_id,
            'answers': answers_info, # Report original answers list
            'question': question,
            'gold_relations': gold_rels_list, # Store the list of gold relations used
            'paths': {f'{k}hop': filtered_paths_by_hop.get(f'{k}hop', []) for k in range(1, max_hops + 1)}
        }
        results.append(output_entry)
        processed_count += 1

    # Save the final results to the output file
    logging.info(f"Saving {len(results)} processed entries to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4) # Use indent=4 for readability
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")
        raise

    logging.info(f"Processing complete. Processed: {processed_count}, Skipped/No Paths: {skipped_count} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and filter multi-hop paths in subgraphs.")
    parser.add_argument("--subgraph_file", required=True, help="Path to the input subgraph JSON file.")
    parser.add_argument("--kb_map_file", required=True, help="Path to the KB ID to integer ID mapping JSON file.")
    parser.add_argument("--golden_rels_file", required=True, help="Path to the golden relations JSON file.")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file.")
    parser.add_argument("--max_hops", type=int, default=4, help="Maximum number of hops for paths (default: 4).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    process_subgraphs(
        subgraph_file=args.subgraph_file,
        kb_id_map_file=args.kb_map_file,
        golden_relations_file=args.golden_rels_file,
        output_file=args.output_file,
        max_hops=args.max_hops
    )