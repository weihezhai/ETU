import json
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract 'id' and 'filtered_path_by_relation_similarity' from a topK JSONL file.")
    parser.add_argument('-k', '--k_value', type=int, required=True,
                        help='The value of k for the input file (e.g., 10 for top10_sim_filtered_paths.jsonl)')
    parser.add_argument('-i', '--input_dir', type=str, default='.',
                        help='Directory containing the input JSONL file (default: current directory)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output JSON file')
    args = parser.parse_args()

    k_value = args.k_value
    input_dir = args.input_dir
    output_dir = args.output_dir

    input_filename = os.path.join(input_dir, f"top{k_value}_sim_filtered_paths.jsonl")
    output_filename = os.path.join(output_dir, f"top{k_value}.json")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    extracted_data = []

    try:
        with open(input_filename, 'r') as infile:
            for line_number, line in enumerate(infile, 1):
                try:
                    # Remove leading/trailing whitespace and parse JSON
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    data = json.loads(line)

                    # Extract required fields
                    entry_id = data.get('id')
                    filtered_path = data.get('filtered_path_by_relation_similarity')

                    # Basic validation: ensure both fields exist
                    if entry_id is None or filtered_path is None:
                        print(f"Warning (File: {input_filename}, Line: {line_number}): Missing 'id' or 'filtered_path_by_relation_similarity'. Skipping.", file=sys.stderr)
                        continue

                    extracted_data.append({
                        'id': entry_id,
                        'filtered_path_by_relation_similarity': filtered_path
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON (File: {input_filename}, Line: {line_number}): {e}. Skipping line.", file=sys.stderr)
                except Exception as e:
                    print(f"An unexpected error occurred processing line {line_number} in {input_filename}: {e}. Skipping line.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
        # Don't exit immediately, maybe other files will process
        return # Skip processing for this k value
    except Exception as e:
        print(f"An unexpected error occurred reading the input file {input_filename}: {e}", file=sys.stderr)
        return # Skip processing for this k value


    # Write the extracted data to the output JSON file
    if not extracted_data:
        print(f"No data extracted from '{input_filename}'. Output file '{output_filename}' will not be created.", file=sys.stderr)
        return

    try:
        with open(output_filename, 'w') as outfile:
            json.dump(extracted_data, outfile, indent=4) # Use indent for readability
        print(f"Successfully extracted data from '{input_filename}' to '{output_filename}'")
    except IOError as e:
        print(f"Error writing to output file '{output_filename}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred writing the output file '{output_filename}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()