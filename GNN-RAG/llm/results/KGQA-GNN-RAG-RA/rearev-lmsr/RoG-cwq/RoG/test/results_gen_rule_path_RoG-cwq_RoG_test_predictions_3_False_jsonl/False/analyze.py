import json

def analyze_predictions(file_path):
    """
    Analyzes a JSONL file to count entries where at least one ground_truth item
    is found in the input string.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        tuple: (matching_entries_count, total_entries_count, malformed_lines_count)
    """
    matching_entries_count = 0
    total_entries_count = 0
    malformed_json_lines = 0
    problematic_data_lines = 0

    print(f"Analyzing file: {file_path}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_entries_count += 1
            line_num = i + 1
            try:
                data = json.loads(line)
                
                ground_truth_list = data.get("ground_truth")
                input_str = data.get("input")

                if not isinstance(ground_truth_list, list):
                    print(f"Warning: Line {line_num} - 'ground_truth' is not a list or is missing. Skipping entry.")
                    problematic_data_lines += 1
                    continue
                
                if not isinstance(input_str, str):
                    print(f"Warning: Line {line_num} - 'input' is not a string or is missing. Skipping entry.")
                    problematic_data_lines += 1
                    continue

                entry_matches = False
                for gt_item in ground_truth_list:
                    if not isinstance(gt_item, str):
                        print(f"Warning: Line {line_num} - Found non-string item in 'ground_truth': '{gt_item}'. Skipping this ground truth item.")
                        continue
                    if gt_item in input_str:
                        entry_matches = True
                        break 
                
                if entry_matches:
                    matching_entries_count += 1

            except json.JSONDecodeError:
                print(f"Error: Line {line_num} - Failed to decode JSON. Skipping line.")
                malformed_json_lines += 1
            except Exception as e:
                print(f"Error: Line {line_num} - An unexpected error occurred: {e}. Skipping line.")
                problematic_data_lines += 1 # Count as problematic if not JSON decode error

    print(f"\n--- Analysis Complete ---")
    print(f"Total entries processed: {total_entries_count}")
    
    if malformed_json_lines > 0:
        print(f"Lines with JSON decoding errors: {malformed_json_lines}")
    if problematic_data_lines > 0:
        print(f"Lines with missing/malformed 'ground_truth' or 'input' fields (or other errors): {problematic_data_lines}")

    validly_parsed_entries = total_entries_count - malformed_json_lines - problematic_data_lines
    print(f"Valid entries considered for matching: {validly_parsed_entries}")
    print(f"Number of entries where at least one ground truth element was found in the input: {matching_entries_count}")
    
    if validly_parsed_entries > 0:
        upper_bound_percentage = (matching_entries_count / validly_parsed_entries) * 100
        print(f"Upper bound (percentage of valid entries matching): {upper_bound_percentage:.2f}%")
    elif total_entries_count > 0:
        print("No valid entries were found to calculate a percentage.")
    else:
        print("The file was empty or no lines were processed.")

    return matching_entries_count, total_entries_count, malformed_json_lines + problematic_data_lines

if __name__ == "__main__":
    # Path to your predictions.jsonl file
    # This path is taken from the information you provided.
    predictions_file_path = "/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"
    
    analyze_predictions(predictions_file_path)