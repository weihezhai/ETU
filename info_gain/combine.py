import json
import argparse
from tqdm import tqdm

def combine_json_files(file1_path, file2_path, output_file_path):
    """
    Combines two JSON files (lists of objects, each with an 'id' field).
    Entries from file1 are prioritized. Entries from file2 are added if their
    ID is not present in file1.

    Args:
        file1_path (str): Path to the first JSON file (higher priority).
        file2_path (str): Path to the second JSON file.
        output_file_path (str): Path to save the combined JSON file.
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
    except FileNotFoundError:
        print(f"Error: File not found - {file1_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file1_path}")
        return

    try:
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except FileNotFoundError:
        print(f"Error: File not found - {file2_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file2_path}")
        return

    if not isinstance(data1, list) or not isinstance(data2, list):
        print("Error: Both input files must contain a JSON list.")
        return

    combined_data = []
    ids_in_file1 = set()

    # Add all items from file1 and store their IDs
    print(f"Processing {file1_path}...")
    for item1 in tqdm(data1, desc=f"Reading from {file1_path.split('/')[-1]}"):
        if isinstance(item1, dict) and 'id' in item1:
            combined_data.append(item1)
            ids_in_file1.add(item1['id'])
        else:
            print(f"Warning: Skipping item in {file1_path} due to missing 'id' or incorrect format: {str(item1)[:100]}")


    # Add items from file2 only if their ID is not in file1
    print(f"\nProcessing {file2_path}...")
    added_from_file2_count = 0
    skipped_from_file2_count = 0
    for item2 in tqdm(data2, desc=f"Reading from {file2_path.split('/')[-1]}"):
        if isinstance(item2, dict) and 'id' in item2:
            if item2['id'] not in ids_in_file1:
                combined_data.append(item2)
                added_from_file2_count += 1
            else:
                skipped_from_file2_count +=1
        else:
            print(f"Warning: Skipping item in {file2_path} due to missing 'id' or incorrect format: {str(item2)[:100]}")


    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, indent=2)

    print(f"\nCombination complete. Output saved to {output_file_path}")
    print(f"Total items from {file1_path.split('/')[-1]}: {len(ids_in_file1)}")
    print(f"Items added from {file2_path.split('/')[-1]}: {added_from_file2_count}")
    print(f"Items skipped from {file2_path.split('/')[-1]} (ID already present): {skipped_from_file2_count}")
    print(f"Total items in combined file: {len(combined_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine two JSON files based on item IDs.')
    parser.add_argument('--file1', required=True, help='Path to the primary JSON file (items from here are prioritized).')
    parser.add_argument('--file2', required=True, help='Path to the secondary JSON file (items are added if their ID is not in file1).')
    parser.add_argument('--output', required=True, help='Path to the output combined JSON file.')

    args = parser.parse_args()

    combine_json_files(args.file1, args.file2, args.output)