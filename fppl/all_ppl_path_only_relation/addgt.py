import os
import json
import glob
from collections import OrderedDict

def append_ground_truth_to_jsonl_files(reference_file, target_folder, output_folder):
    """
    Append ground_truth attribute from a reference file to all JSONL files in a target folder
    and write the updated files to an output folder. Places ground_truth before prediction.
    
    Args:
        reference_file (str): Path to the reference JSONL file containing ground_truth data
        target_folder (str): Path to the folder containing JSONL files to update
        output_folder (str): Path to the folder where updated files will be written
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a dictionary mapping ID to ground_truth from the reference file
    id_to_ground_truth = {}
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'id' in data and 'ground_truth' in data:
                        id_to_ground_truth[data['id']] = data['ground_truth']
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in reference file: {line[:50]}...")
    
    # Get all JSONL files in the target folder
    jsonl_files = glob.glob(os.path.join(target_folder, '*.jsonl'))
    
    # Process each file
    for file_path in jsonl_files:
        print(f"Processing {file_path}...")
        
        # Get the filename without the path
        filename = os.path.basename(file_path)
        # Create the output file path
        output_file_path = os.path.join(output_folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Create a new ordered dictionary to control field order
                        ordered_data = OrderedDict()
                        
                        # Add id and question first if they exist
                        if 'id' in data:
                            ordered_data['id'] = data['id']
                        if 'question' in data:
                            ordered_data['question'] = data['question']
                        
                        # Add ground_truth before prediction if ID exists in our mapping
                        if 'id' in data and data['id'] in id_to_ground_truth:
                            ordered_data['ground_truth'] = id_to_ground_truth[data['id']]
                        
                        # Add remaining fields
                        for key, value in data.items():
                            if key not in ordered_data:
                                ordered_data[key] = value
                        
                        # Write the updated record
                        outfile.write(json.dumps(ordered_data, ensure_ascii=False) + '\n')
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {file_path}: {line[:50]}...")
                        # Write the original line if we can't parse it
                        outfile.write(line)
        
        print(f"Created updated file: {output_file_path}")

if __name__ == "__main__":
    # Example usage
    reference_file = "/data/home/mpx602/projects/ETU/ETU/GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-lmsr/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl"  # Replace with actual path
    target_folder = "/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path"          # Replace with actual path
    output_folder = "/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt"  # Replace with actual path
    
    append_ground_truth_to_jsonl_files(reference_file, target_folder, output_folder)
    print("Finished updating all files")