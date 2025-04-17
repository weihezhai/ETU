import json
import statistics
import argparse
from pathlib import Path

def analyze_entities(json_file_path: Path):
    """
    Loads a JSON dataset file, counts the number of entities in the subgraph
    for each entry, and prints statistics about these counts.

    Assumes the JSON file contains a list of objects, or one object per line.

    Args:
        json_file_path (Path): The path to the JSON file.
    """
    entity_counts = []
    total_entries = 0
    entries_with_entities = 0

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # Try loading as a single JSON array first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    total_entries = len(data)
                    for entry in data:
                        count = _extract_entity_count(entry)
                        if count is not None:
                            entity_counts.append(count)
                            entries_with_entities += 1
                        elif isinstance(entry, dict):
                            print(f"Warning: Skipping entry due to missing/invalid 'subgraph' or 'entities': ID {entry.get('id', 'N/A')}")
                elif isinstance(data, dict): # Handle case of single object file
                        total_entries = 1
                        count = _extract_entity_count(data)
                        if count is not None:
                        entity_counts.append(count)
                        entries_with_entities += 1
                        else:
                            print(f"Warning: Single entry file missing/invalid 'subgraph' or 'entities': ID {data.get('id', 'N/A')}")
                else:
                        print(f"Error: Unexpected JSON structure in {json_file_path}. Expected a list or a single object.")
                        return

            except json.JSONDecodeError:
                # If loading as a single array fails, try JSON Lines format (one JSON object per line)
                print("Could not decode as single JSON object/array, trying JSON Lines format...")
                f.seek(0) # Reset file pointer
                entity_counts = [] # Reset counts
                entries_with_entities = 0
                total_entries = 0
                line_num = 0
                for line in f:
                    line_num += 1
                    total_entries += 1
                    try:
                        if line.strip(): # Skip empty lines
                            entry = json.loads(line)
                            count = _extract_entity_count(entry)
                            if count is not None:
                                entity_counts.append(count)
                                entries_with_entities += 1
                            elif isinstance(entry, dict):
                                    print(f"Warning: Skipping line {line_num} due to missing/invalid 'subgraph' or 'entities': ID {entry.get('id', 'N/A')}")
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON on line {line_num} in {json_file_path}")
                        continue # Skip malformed lines

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("\n--- Entity Count Statistics ---")
    print(f"Processed file: {json_file_path.name}")
    print(f"Total entries found: {total_entries}")
    print(f"Entries with valid 'subgraph.entities' list: {entries_with_entities}")


    if not entity_counts:
        print("\nNo valid entries with entities found to calculate statistics.")
        return

    min_entities = min(entity_counts)
    max_entities = max(entity_counts)
    avg_entities = statistics.mean(entity_counts)
    median_entities = statistics.median(entity_counts)
    # total_entities_sum = sum(entity_counts) # Uncomment if you need the sum of all entities across all valid entries

    print(f"\nStatistics based on {entries_with_entities} entries:")
    print(f"  Minimum entities per entry: {min_entities}")
    print(f"  Maximum entities per entry: {max_entities}")
    print(f"  Average entities per entry: {avg_entities:.2f}")
    print(f"  Median entities per entry: {median_entities}")
    # print(f"  Total entities across all valid entries: {total_entities_sum}") # Uncomment if needed


def _extract_entity_count(entry: dict) -> int | None:
    """Extracts the entity count from a single JSON object."""
    if isinstance(entry, dict) and \
        "subgraph" in entry and \
        isinstance(entry["subgraph"], dict) and \
        "entities" in entry["subgraph"] and \
        isinstance(entry["subgraph"]["entities"], list):
        return len(entry["subgraph"]["entities"])
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze entity counts in a JSON dataset file.")
    parser.add_argument("json_file", help="Path to the input JSON file (can be JSON array or JSON Lines format).")
    args = parser.parse_args()

    file_path = Path(args.json_file)
    analyze_entities(file_path)