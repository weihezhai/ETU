# graph_to_path/process_subgraphs.py
import json
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
import os
import spacy # Added spacy import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for Spacy model (load once)
NLP_MODEL = None

def load_spacy_model(model_name="en_core_web_lg"):
    """Loads the Spacy model."""
    global NLP_MODEL
    if NLP_MODEL is None:
        try:
            NLP_MODEL = spacy.load(model_name)
            logging.info(f"Spacy model '{model_name}' loaded successfully.")
        except OSError:
            logging.error(f"Spacy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
            raise
    return NLP_MODEL

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

def load_data_and_mappings(subgraph_file, kb_id_map_file, entity_labels_file, golden_relations_file):
    """Loads all necessary data files and creates required mappings."""
    subgraph_data = load_json(subgraph_file, is_json_lines=True)
    kb_id_to_int_id_map = load_json(kb_id_map_file, is_json_lines=False)
    kb_id_to_label_map = load_json(entity_labels_file, is_json_lines=False)
    golden_relations_data = load_json(golden_relations_file, is_json_lines=False)

    # Create map: Label (lowercase) -> integer ID
    label_to_int_id_map = defaultdict(list)
    logging.info("Creating label -> integer ID map...")
    # First map labels to KB IDs (handling potential multiple KB IDs per label)
    label_to_kb_id_map = defaultdict(list)
    for kb_id, label in kb_id_to_label_map.items():
        if label:
            label_to_kb_id_map[label.lower()].append(kb_id)

    # Now map labels to integer IDs using the intermediate map
    for label_lower, kb_ids in tqdm(label_to_kb_id_map.items(), desc="Building label->int_id map"):
        for kb_id in kb_ids:
            if kb_id in kb_id_to_int_id_map:
                int_id = kb_id_to_int_id_map[kb_id]
                if int_id not in label_to_int_id_map[label_lower]:
                    label_to_int_id_map[label_lower].append(int_id)
            # else: logging.warning(f"KB ID '{kb_id}' (from label '{label_lower}') not found in kb_id_map.") # Optional warning
    logging.info(f"Built map for {len(label_to_int_id_map)} unique labels to integer IDs.")

    return subgraph_data, kb_id_to_int_id_map, label_to_int_id_map, golden_relations_data


def extract_ner_entities(question, nlp, label_to_int_id_map, entry_id):
    """Performs NER on the question and returns a set of integer IDs found."""
    ner_source_nodes = set()
    num_added = 0
    if question and nlp:
        logging.debug(f"Running NER on question for entry {entry_id}: '{question[:100]}...'")
        doc = nlp(question)
        recognized_entity_labels = set(ent.text.lower() for ent in doc.ents)

        if recognized_entity_labels:
            logging.debug(f"  NER recognized labels: {recognized_entity_labels}")
            for label_lower in recognized_entity_labels:
                if label_lower in label_to_int_id_map:
                    potential_int_ids = label_to_int_id_map[label_lower]
                    original_size = len(ner_source_nodes)
                    ner_source_nodes.update(potential_int_ids)
                    added_count = len(ner_source_nodes) - original_size
                    if added_count > 0:
                         logging.debug(f"  Adding {added_count} source node(s) {potential_int_ids} (from NER label '{label_lower}') for entry {entry_id}")
                         num_added += added_count
                # else: logging.debug(f"  NER label '{label_lower}' not found in label_to_int_id map.")
    return ner_source_nodes, num_added


def map_answers_to_target_ids(answers_info, kb_id_to_int_id_map, entry_id):
    """Maps answer KB IDs to integer IDs."""
    target_nodes = set()
    valid_answers_info = []
    for ans in answers_info:
        kb_id = ans.get('kb_id')
        if kb_id and kb_id in kb_id_to_int_id_map:
            target_id = kb_id_to_int_id_map[kb_id]
            target_nodes.add(target_id)
            valid_answers_info.append(ans) # Keep original info for mapped answers
        else:
            logging.warning(f"KB ID '{kb_id}' for answer in entry {entry_id} not found in kb_id_map.")
    return list(target_nodes), valid_answers_info


def find_and_filter_paths(G, valid_source_nodes, target_nodes, gold_rels_set, max_hops, entry_id):
    """Finds paths using BFS and filters them based on golden relations."""
    all_paths = []
    logging.debug(f"Finding paths for entry {entry_id} from sources {valid_source_nodes} to targets {target_nodes}...")
    for source_node in valid_source_nodes:
        for target_node in target_nodes:
            # Target node existence check (already done for sources earlier)
            if target_node not in G:
                logging.warning(f"Entry {entry_id}: Target node {target_node} not found in graph. Skipping paths to it.")
                continue
            paths = find_paths_bfs(G, source_node, target_node, max_hops)
            all_paths.extend(paths)
    logging.info(f"Entry {entry_id}: Found {len(all_paths)} raw paths before filtering.")

    # Filter paths
    filtered_paths_by_hop = defaultdict(list)
    if not gold_rels_set:
        logging.warning(f"Entry {entry_id}: No golden relations found. No paths will be kept.")
    elif not all_paths:
        logging.info(f"Entry {entry_id}: No raw paths found between sources and targets.")
    else:
        logging.debug(f"Filtering {len(all_paths)} paths using {len(gold_rels_set)} golden relations: {gold_rels_set}")
        for path in all_paths:
            path_relations = set(path[i] for i in range(1, len(path), 2))
            if not path_relations.isdisjoint(gold_rels_set):
                num_hops = len(path) // 2
                if 1 <= num_hops <= max_hops:
                    filtered_paths_by_hop[f'{num_hops}hop'].append(path)
        logging.info(f"Entry {entry_id}: Kept {sum(len(v) for v in filtered_paths_by_hop.values())} paths after filtering.")

    # Ensure all hop keys exist in the output
    final_filtered_paths = {f'{k}hop': filtered_paths_by_hop.get(f'{k}hop', []) for k in range(1, max_hops + 1)}
    return final_filtered_paths


def process_single_entry(entry, nlp, label_to_int_id_map, kb_id_to_int_id_map, golden_relations_data, max_hops):
    """Processes a single entry from the subgraph data."""
    entry_id = entry.get('id')
    question = entry.get('question', '')
    original_source_nodes = entry.get('entities', [])
    answers_info = entry.get('answers', [])
    subgraph_triples = entry.get('subgraph', {}).get('tuples', [])

    if not entry_id or not answers_info or not subgraph_triples:
        logging.warning(f"Skipping entry due to missing critical data: id={entry_id}, "
                        f"has_answers={bool(answers_info)}, has_triples={bool(subgraph_triples)}")
        return None, 0 # Return None for skipped entry, 0 NER added

    logging.info(f"Processing entry ID: {entry_id}")

    # 1. NER Processing
    ner_source_nodes, ner_added_count = extract_ner_entities(question, nlp, label_to_int_id_map, entry_id)
    combined_source_nodes = set(original_source_nodes).union(ner_source_nodes)
    final_source_nodes = list(combined_source_nodes) # This is the list we want to save
    logging.info(f"Entry {entry_id}: Original sources: {len(original_source_nodes)}, NER added: {ner_added_count}, Total unique sources: {len(final_source_nodes)}")

    # Define the base structure for output, including the final source list
    base_output_entry = {
        'id': entry_id,
        'answers': answers_info,
        'question': question,
        'identified_source_entities': final_source_nodes, # Added field
        'gold_relations': golden_relations_data.get(entry_id, []),
        'paths': {f'{k}hop': [] for k in range(1, max_hops + 1)} # Default empty paths
    }

    if not final_source_nodes:
        logging.warning(f"Skipping entry {entry_id} as no source entities were found (original or via NER).")
        # Return the base structure indicating skipped status
        return base_output_entry, ner_added_count # Return skipped entry, but count NER additions

    # 2. Map Answers to Target IDs
    target_nodes, _ = map_answers_to_target_ids(answers_info, kb_id_to_int_id_map, entry_id)
    if not target_nodes:
        logging.warning(f"No valid target node IDs found for entry {entry_id} after mapping.")
        # Return the base structure
        return base_output_entry, ner_added_count

    # 3. Build Graph and Validate Nodes
    logging.debug(f"Building graph for entry {entry_id}...")
    G = build_graph(subgraph_triples)
    valid_source_nodes = [s for s in final_source_nodes if s in G]
    if len(valid_source_nodes) != len(final_source_nodes):
        invalid_sources = [s for s in final_source_nodes if s not in G]
        logging.warning(f"Entry {entry_id}: Some source nodes {invalid_sources} "
                        f"(original or NER-derived) were not found in the graph.")
    if not valid_source_nodes:
        logging.warning(f"Entry {entry_id}: No valid source nodes found in the graph. Skipping path finding.")
        # Return the base structure
        return base_output_entry, ner_added_count

    # 4. Find and Filter Paths
    gold_rels_list = golden_relations_data.get(entry_id, [])
    gold_rels_set = set(gold_rels_list)
    filtered_paths = find_and_filter_paths(G, valid_source_nodes, target_nodes, gold_rels_set, max_hops, entry_id)

    # 5. Format Final Output
    output_entry = {
        'id': entry_id,
        'answers': answers_info, # Report original answers list
        'question': question,
        'identified_source_entities': final_source_nodes, # Added final source list here too
        'gold_relations': gold_rels_list,
        'paths': filtered_paths # Add the found/filtered paths
    }
    return output_entry, ner_added_count

def save_results(results, output_file):
    """Saves the processed results to a JSON file."""
    logging.info(f"Saving {len(results)} processed entries to {output_file}...")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir: # Ensure output_dir is not empty (e.g., if output_file is just a filename)
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")
        raise


def process_subgraphs(subgraph_file, kb_id_map_file, entity_labels_file, golden_relations_file, output_file, max_hops=4):
    """
    Main function to process subgraph data: loads data, processes each entry, and saves results.
    """
    logging.info("Starting subgraph processing...")
    nlp = load_spacy_model() # Load Spacy model

    # 1. Load Data and Mappings
    try:
        subgraph_data, kb_id_to_int_id_map, label_to_int_id_map, golden_relations_data = load_data_and_mappings(
            subgraph_file, kb_id_map_file, entity_labels_file, golden_relations_file
        )
    except FileNotFoundError:
        logging.error("Exiting due to missing input file.")
        return # Stop processing if essential files are missing
    except Exception as e:
        logging.error(f"Exiting due to error during data loading: {e}")
        return

    # 2. Process Entries
    results = []
    processed_count = 0
    skipped_count = 0
    total_ner_added_sources = 0

    for entry in tqdm(subgraph_data, desc="Processing entries"):
        output_entry, ner_added_count = process_single_entry(
            entry, nlp, label_to_int_id_map, kb_id_to_int_id_map, golden_relations_data, max_hops
        )
        total_ner_added_sources += ner_added_count
        if output_entry is not None:
            results.append(output_entry)
            # Determine if it was processed (found paths) or skipped (no sources/targets/paths)
            # A simple check: if all path lists are empty, consider it skipped in terms of finding useful paths
            if all(not paths for paths in output_entry['paths'].values()):
                 # Check if the reason was missing source/target nodes originally, or just no paths found
                 # Log message inside process_single_entry already indicates the reason
                 skipped_count += 1
            else:
                 processed_count += 1
        else:
             # This case happens if critical data (id, answers, triples) was missing initially
             skipped_count += 1


    # 3. Save Results
    save_results(results, output_file)

    logging.info(f"Processing complete. Entries processed (paths found/filtered): {processed_count}, Entries skipped/empty paths: {skipped_count}")
    logging.info(f"Total source nodes added via NER across all entries: {total_ner_added_sources}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and filter multi-hop paths in subgraphs, using NER for additional source entities.")
    parser.add_argument("--subgraph_file", required=True, help="Path to the input subgraph JSON Lines file.")
    parser.add_argument("--kb_map_file", required=True, help="Path to the KB ID ('Qxxx') to integer ID mapping JSON file.")
    parser.add_argument("--entity_labels_file", required=True, help="Path to the KB ID ('Qxxx') to entity label mapping JSON file.") # Added
    parser.add_argument("--golden_rels_file", required=True, help="Path to the golden relations JSON file (map: entry_id -> [rel_id1, rel_id2]).")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file.")
    parser.add_argument("--max_hops", type=int, default=4, help="Maximum number of hops for paths (default: 4).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_lg", help="Name of the Spacy model to use (default: en_core_web_lg).") # Added Spacy model choice

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load Spacy model using the provided argument *before* calling process_subgraphs
    try:
        load_spacy_model(args.spacy_model)
    except (OSError, ImportError):
         logging.error(f"Could not load Spacy model '{args.spacy_model}'. Please ensure it's installed.")
         # Decide whether to exit or continue without NER
         exit(1) # Exit if Spacy/model loading fails

    process_subgraphs(
        subgraph_file=args.subgraph_file,
        kb_id_map_file=args.kb_map_file,
        entity_labels_file=args.entity_labels_file,
        golden_relations_file=args.golden_rels_file,
        output_file=args.output_file,
        max_hops=args.max_hops
    ) # Note: nlp model is now loaded via load_spacy_model and accessed globally