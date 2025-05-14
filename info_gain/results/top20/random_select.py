# import json
# import random
# from collections import defaultdict

# def create_random_baseline(eval_file_path, unprocessed_file_path, output_file_path):
#     """
#     Generates a baseline by randomly selecting paths from unprocessed data,
#     matching the number of processed results in the evaluation data.

#     Args:
#         eval_file_path (str): Path to the processed evaluation JSON file
#                               (e.g., 'processed_combined_evaluation_top20.json').
#         unprocessed_file_path (str): Path to the unprocessed JSON file with retrieved paths
#                                      (e.g., 'averaged_results_retrieved_prob.json').
#         output_file_path (str): Path to save the generated random baseline JSON.
#     """
#     try:
#         with open(eval_file_path, 'r') as f:
#             eval_data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Evaluation file not found at {eval_file_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {eval_file_path}")
#         return

#     try:
#         with open(unprocessed_file_path, 'r') as f:
#             unprocessed_data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: Unprocessed data file not found at {unprocessed_file_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {unprocessed_file_path}")
#         return

#     # Pre-process unprocessed_data for easier lookup
#     unprocessed_paths_by_id = defaultdict(list)
#     for item in unprocessed_data:
#         item_id = item.get("id")
#         paths = item.get("paths", [])
#         if item_id and paths:
#             unprocessed_paths_by_id[item_id].extend(paths)

#     output_results = []

#     for eval_entry in eval_data:
#         entry_id = eval_entry.get("id")
#         if not entry_id:
#             # Skip entries without an ID or handle as needed
#             new_entry = eval_entry.copy()
#             new_entry["randomly_selected_results"] = []
#             output_results.append(new_entry)
#             continue

#         # Get the number of results to select
#         # Assuming the key is "processed_results" as per the problem description
#         # and its value is a list.
#         num_to_select = len(eval_entry.get("processed_results", []))

#         all_available_paths = unprocessed_paths_by_id.get(entry_id, [])
        
#         randomly_selected_nodes = []
#         if all_available_paths:
#             num_paths_to_sample = min(num_to_select, len(all_available_paths))
            
#             if num_paths_to_sample > 0:
#                 selected_paths = random.sample(all_available_paths, num_paths_to_sample)
#                 for path in selected_paths:
#                     if path: # Ensure path is not empty
#                         randomly_selected_nodes.append(path[-1]) # Get the last node

#         # Create the new entry, copying all original fields
#         new_entry = eval_entry.copy()
#         # Remove the old "processed_results" key if it exists
#         if "processed_results" in new_entry:
#             del new_entry["processed_results"]
#         new_entry["randomly_selected_results"] = randomly_selected_nodes
#         output_results.append(new_entry)

#     try:
#         with open(output_file_path, 'w') as f:
#             json.dump(output_results, f, indent=4)
#         print(f"Successfully created random baseline at {output_file_path}")
#     except IOError:
#         print(f"Error: Could not write output to {output_file_path}")


# if __name__ == '__main__':
#     # Define file paths
#     eval_file = '/data/home/mpx602/projects/ETU/ETU/info_gain/results/processed_combined_evaluation_top20.json'
#     # Assuming 'averaged_results_retrieved_prob.json' is in the same 'results' directory
#     unprocessed_file = '/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/averaged_results_retrieved_prob.json' 
#     output_file = '/data/home/mpx602/projects/ETU/ETU/info_gain/results/top20/random_baseline_evaluation.json'

#     # Set a seed for reproducibility if desired
#     # random.seed(42) 

#     create_random_baseline(eval_file, unprocessed_file, output_file)

#     # Example of how to call from another script if needed:
#     # from info_gain.create_random_baseline import create_random_baseline
#     # create_random_baseline(
#     #     'path/to/processed_combined_evaluation_top20.json',
#     #     'path/to/averaged_results_retrieved_prob.json',
#     #     'path/to/output_random_baseline.json'
#     # )
import json
import random

# Load the processed evaluation data
with open('/data/home/mpx602/projects/ETU/ETU/info_gain/results/processed_combined_evaluation_top20.json', 'r') as f:
    processed_data = json.load(f)

# Load the path evaluation data
with open('/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/averaged_results_retrieved_prob.json', 'r') as f:
    path_data = json.load(f)

# Group path data by ID
path_data_by_id = {}
for entry in path_data:
    id_val = entry['id']
    if id_val not in path_data_by_id:
        path_data_by_id[id_val] = []
    path_data_by_id[id_val].append(entry)

# Create the output structure
random_results = []

# Process each entry in the processed data
for entry in processed_data:
    id_val = entry['id']
    num_results = len(entry['processed_results'])
    
    # Find paths with matching ID
    if id_val in path_data_by_id:
        matching_paths = path_data_by_id[id_val]
        
        # If we have enough paths, randomly select the required number
        if len(matching_paths) >= num_results:
            selected_paths = random.sample(matching_paths, num_results)
            # Extract the last node from each path
            randomly_selected_results = [path['path'][-1] for path in selected_paths]
        else:
            # If we don't have enough paths, use all available and repeat some if necessary
            randomly_selected_results = [path['path'][-1] for path in matching_paths]
            # If we still need more, randomly sample from the available ones
            if len(randomly_selected_results) < num_results:
                additional_needed = num_results - len(randomly_selected_results)
                randomly_selected_results.extend(random.choices(randomly_selected_results, k=additional_needed))
    else:
        # If no matching paths found, use an empty list
        randomly_selected_results = []
    
    # Add to output
    random_results.append({
        "id": id_val,
        "processed_results": randomly_selected_results
    })

# Write the output
with open('/data/home/mpx602/projects/ETU/ETU/info_gain/results/top20/random_baseline_evaluation.json', 'w') as f:
    json.dump(random_results, f, indent=2)