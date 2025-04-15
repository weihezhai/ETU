import json
import sys
import os

# --- Configuration ---
topk_filename = "/Users/rickzhai/Documents/GitHub/ETU/ETU/similarity/topsim/topk_relation_filtered_paths/extracted_filtered_paths/top15.json" # Input file with filtered paths
target_filename = "/Users/rickzhai/Documents/GitHub/ETU/ETU/similarity/topsim/generation_results_cleaned/15answers/top12_sim_filtered_paths_llm_results_cleaned.json" # Input file to be filtered
output_filename = "/Users/rickzhai/Documents/GitHub/ETU/ETU/similarity/topsim/generation_results_cleaned/15answers/top12_sim_filtered_paths_llm_results_cleaned_refined.json" # Output file
# --- End Configuration ---

# <<< Added string_overlap function >>>
def string_overlap(str1, str2):
    """Check if either string contains the other (case-insensitive)."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        # Handle cases where input might not be strings as expected
        print(f"Warning: string_overlap received non-string input: type({str1})='{type(str1).__name__}', type({str2})='{type(str2).__name__}'. Treating as no overlap.", file=sys.stderr)
        return False
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    # Avoid matching empty strings unintentionally if strip results in empty
    if not s1 or not s2:
        return False
    return s1 in s2 or s2 in s1

def extract_last_entity(path):
    """
    Extracts the last element from a path.
    Assumes path is a list or tuple.
    Modify if your path format is different (e.g., string with separators).
    """
    if isinstance(path, (list, tuple)) and path:
        return path[-1]
    # Add handling for other formats if needed, e.g.:
    # if isinstance(path, str):
    #     parts = path.split(' -> ') # Or your separator
    #     return parts[-1] if parts else None
    print(f"Warning: Could not determine last entity from path: {path}. Format unexpected.", file=sys.stderr)
    return None

def load_topk_data(filename):
    """Loads the topK JSON file and extracts last entities per ID."""
    last_entities_by_id = {}
    try:
        with open(filename, 'r') as f:
            topk_data = json.load(f)

        for entry in topk_data:
            entry_id = entry.get('id')
            paths = entry.get('filtered_path_by_relation_similarity')

            if entry_id is None or paths is None:
                print(f"Warning: Skipping entry in {filename} due to missing 'id' or 'filtered_path_by_relation_similarity': {entry}", file=sys.stderr)
                continue

            # Store last entities as strings in a set
            last_entities = set()
            for path in paths:
                last_entity = extract_last_entity(path)
                if last_entity is not None:
                    last_entities.add(str(last_entity)) # Ensure stored as string

            if entry_id in last_entities_by_id:
                 print(f"Warning: Duplicate ID '{entry_id}' found in {filename}. Merging last entities.", file=sys.stderr)
                 last_entities_by_id[entry_id].update(last_entities)
            else:
                last_entities_by_id[entry_id] = last_entities

        print(f"Successfully loaded and processed {filename}. Found data for {len(last_entities_by_id)} unique IDs.")
        return last_entities_by_id

    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{filename}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading {filename}: {e}", file=sys.stderr)
        sys.exit(1)


def filter_target_data(target_filename, output_filename, last_entities_by_id):
    """Filters the target JSON file based on the extracted last entities using string overlap."""
    filtered_target_data = []
    processed_count = 0
    filtered_count = 0

    try:
        with open(target_filename, 'r') as f:
            target_data = json.load(f) # Assuming target.json is a JSON array

        print(f"Filtering {target_filename} using string overlap...")
        for entry in target_data:
            entry_id = entry.get('id')
            processed_results = entry.get('processed_results')

            if entry_id is None or processed_results is None:
                print(f"Warning: Skipping entry in {target_filename} due to missing 'id' or 'processed_results': {entry}", file=sys.stderr)
                entry['filtered_processed_results'] = []
                filtered_target_data.append(entry)
                continue

            if not isinstance(processed_results, list):
                 print(f"Warning: 'processed_results' for ID '{entry_id}' in {target_filename} is not a list. Treating as empty.", file=sys.stderr)
                 processed_results = []

            allowed_last_entities = last_entities_by_id.get(entry_id)

            new_filtered_results = []
            if allowed_last_entities: # Only filter if we have data for this ID from topk
                processed_count += 1
                for item in processed_results:
                     item_str = str(item) # Ensure item is string for comparison
                     # <<< Modified Filter Logic >>>
                     # Check if item_str overlaps with *any* of the allowed last entities
                     match_found = False
                     for allowed_entity in allowed_last_entities:
                         # allowed_entity is already guaranteed to be a string from load_topk_data
                         if string_overlap(item_str, allowed_entity):
                             match_found = True
                             break # Found an overlap, no need to check further for this item
                     # <<< End Modified Filter Logic >>>

                     if match_found:
                         new_filtered_results.append(item) # Keep original item

                if new_filtered_results:
                    filtered_count +=1
            else:
                # ID not found in topk data, or no valid paths for that ID
                pass

            entry['filtered_processed_results'] = new_filtered_results
            filtered_target_data.append(entry)

        try:
            with open(output_filename, 'w') as outfile:
                json.dump(filtered_target_data, outfile, indent=4)
            print(f"Successfully filtered {target_filename}.")
            print(f"Processed entries with matching IDs: {processed_count}")
            print(f"Entries with at least one item kept after filtering: {filtered_count}")
            print(f"Output saved to '{output_filename}'")
        except IOError as e:
            print(f"Error writing to output file '{output_filename}': {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred writing the output file: {e}", file=sys.stderr)


    except FileNotFoundError:
        print(f"Error: Target file '{target_filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{target_filename}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred processing {target_filename}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # 1. Load and process the topK file
    source_entities = load_topk_data(topk_filename)

    # 2. Filter the target file using the processed data
    if source_entities is not None:
        filter_target_data(target_filename, output_filename, source_entities)
