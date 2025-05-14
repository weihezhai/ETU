import json
import argparse

def reformat_paths(input_file, output_file):
    """
    Reformat web_filtered_paths.jsonl to match the path_ppl_scores.jsonl format.
    
    Args:
        input_file: Path to the input JSONL file (web_filtered_paths.jsonl)
        output_file: Path to save the reformatted output
    """
    reformatted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            # Create new reformatted entry
            new_entry = {
                "id": entry["id"],
                "question": entry["question"],
                "ground_truth": entry["ground_truth"],
                "prediction": entry["prediction"],
                "paths": []
            }
            
            # Process filtered paths (or regular paths if no filtered ones)
            paths_to_use = entry.get("filtered_path_by_relation", entry["paths"])
            
            # Convert paths to the new format
            for path in paths_to_use:
                path_str = " -> ".join(path)
                new_entry["paths"].append({
                    "path": path,
                    "path_str": path_str
                })
            
            reformatted_data.append(new_entry)
    
    # Write the reformatted data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in reformatted_data:
            f.write(json.dumps(entry, indent=4) + "\n")
    
    print(f"Reformatted data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reformat web_filtered_paths.jsonl to match path_ppl_scores.jsonl format')
    parser.add_argument('--input', required=True, help='Path to input JSONL file (web_filtered_paths.jsonl)')
    parser.add_argument('--output', required=True, help='Path to output JSONL file')
    
    args = parser.parse_args()
    
    reformat_paths(args.input, args.output)