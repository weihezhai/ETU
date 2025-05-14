# import json

# def merge_predictions(results1, results2):
#     merged_results = {}
    
#     # Process each result set
#     for results in [results1, results2]:
#         for entry in results:
#             id = entry["id"]
#             if id not in merged_results:
#                 merged_results[id] = {"predictions": {}, "ground_truth": entry["ground_truth"]}
            
#             # Assign weights based on rank (reversed, so first gets highest weight)
#             for i, pred in enumerate(entry["predictions"]):
#                 weight = len(entry["predictions"]) - i
#                 if pred in merged_results[id]["predictions"]:
#                     merged_results[id]["predictions"][pred] += weight
#                 else:
#                     merged_results[id]["predictions"][pred] = weight
    
#     # Rerank predictions based on weights
#     final_results = []
#     for id, data in merged_results.items():
#         sorted_preds = sorted(data["predictions"].items(), key=lambda x: x[1], reverse=True)
#         reranked_preds = [pred for pred, _ in sorted_preds]
        
#         # Calculate metrics with ground truth
#         hit = any(pred in data["ground_truth"] for pred in reranked_preds)
#         h_1 = reranked_preds[0] in data["ground_truth"] if reranked_preds else False
        
#         # Calculate precision, recall, F1
#         tp = sum(1 for pred in reranked_preds if pred in data["ground_truth"])
#         precision = tp / len(reranked_preds) if reranked_preds else 0
#         recall = tp / len(data["ground_truth"]) if data["ground_truth"] else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         final_results.append({
#             "id": id,
#             "hit": hit,
#             "h_1": h_1,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "predictions": reranked_preds,
#             "ground_truth": data["ground_truth"]
#         })
    
#     return final_results

# with open("/data/home/mpx602/projects/ETU/ETU/info_gain/results/top20/avg_prob_metrics_top25.json") as f:
#     data1 = json.load(f)

# with open("/data/home/mpx602/projects/ETU/ETU/similarity/topsim/evaluation_metrics/15/top15_sim_filtered_paths_llm_results_cleaned_metrics.json") as f:
#     data2 = json.load(f)

# merged_results = merge_predictions(data1["results"], data2["results"])

# output = {"results": merged_results}
# with open("merged_results.json", "w") as f:
#     json.dump(output, f, indent=2)
# import json
# import argparse

# def merge_predictions(results1_data, results2_data):
#     merged_results = {}
    
#     # Process each result set
#     for results_list in [results1_data, results2_data]:
#         for entry in results_list:
#             entry_id = entry["id"]
#             if entry_id not in merged_results:
#                 merged_results[entry_id] = {"predictions": {}, "ground_truth": entry["ground_truth"]}
            
#             # Assign weights based on rank (reversed, so first gets highest weight)
#             num_predictions = len(entry["predictions"])
#             for i, pred in enumerate(entry["predictions"]):
#                 weight = num_predictions - i
#                 if pred in merged_results[entry_id]["predictions"]:
#                     merged_results[entry_id]["predictions"][pred] += weight
#                 else:
#                     merged_results[entry_id]["predictions"][pred] = weight
    
#     # Rerank predictions based on weights and prepare final output format
#     final_output_results = []
#     for entry_id, data in merged_results.items():
#         # Sort predictions by weight in descending order
#         sorted_preds_with_weights = sorted(data["predictions"].items(), key=lambda x: x[1], reverse=True)
#         # Extract just the prediction strings in the new order
#         reranked_preds = [pred for pred, _ in sorted_preds_with_weights]
        
#         # Keep the structure similar to the input, just update predictions
#         # Metrics like hit, h_1, precision, recall, f1 will be recalculated by the evaluation script
#         final_output_results.append({
#             "id": entry_id,
#             "predictions": reranked_preds,
#             "ground_truth": data["ground_truth"] 
#             # We will let the evaluate_results.py script calculate these metrics
#             # "hit": False, 
#             # "h_1": False,
#             # "precision": 0.0,
#             # "recall": 0.0,
#             # "f1": 0.0 
#         })
            
#     return final_output_results

# def main():
#     parser = argparse.ArgumentParser(description="Merge and rerank predictions from two JSON result files.")
#     parser.add_argument("--file1", required=True, help="Path to the first JSON result file.")
#     parser.add_argument("--file2", required=True, help="Path to the second JSON result file.")
#     parser.add_argument("--output", required=True, help="Path to save the merged JSON result file.")
    
#     args = parser.parse_args()
    
#     with open(args.file1, 'r') as f:
#         data1 = json.load(f)
        
#     with open(args.file2, 'r') as f:
#         data2 = json.load(f)
        
#     # Assuming the relevant data is under a "results" key, like in the examples
#     results1_list = data1.get("results", [])
#     results2_list = data2.get("results", [])
    
#     merged_data = merge_predictions(results1_list, results2_list)
    
#     # The output should also be in the format {"results": [...]}
#     output_json = {"results": merged_data}
    
#     with open(args.output, 'w') as f:
#         json.dump(output_json, f, indent=2)
        
#     print(f"Merged results saved to {args.output}")

# if __name__ == "__main__":
#     main()
import json
import argparse
import os

def merge_predictions(results1, results2):
    merged_results = {}
    
    # Process each result set
    for results in [results1, results2]:
        for entry in results:
            id = entry["id"]
            if id not in merged_results:
                merged_results[id] = {"predictions": {}}
            
            # Assign weights based on rank (reversed, so first gets highest weight)
            for i, pred in enumerate(entry["predictions"]):
                weight = len(entry["predictions"]) - i
                if pred in merged_results[id]["predictions"]:
                    merged_results[id]["predictions"][pred] += weight
                else:
                    merged_results[id]["predictions"][pred] = weight
    
    # Rerank predictions based on weights
    final_results = []
    for id, data in merged_results.items():
        sorted_preds = sorted(data["predictions"].items(), key=lambda x: x[1], reverse=True)
        reranked_preds = [pred for pred, _ in sorted_preds]
        
        final_results.append({
            "id": id,
            "processed_results": reranked_preds
        })
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Merge two result files based on weighted predictions')
    parser.add_argument('--file1', required=True, help='Path to first result file')
    parser.add_argument('--file2', required=True, help='Path to second result file')
    parser.add_argument('--output', required=True, help='Path to output merged file')
    
    args = parser.parse_args()
    
    with open(args.file1) as f:
        data1 = json.load(f)
        
    with open(args.file2) as f:
        data2 = json.load(f)
    
    merged_results = merge_predictions(data1["results"], data2["results"])
    
    # Calculate proper directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write merged results
    with open(args.output, 'w') as f:
        json.dump(merged_results, f, indent=2)

if __name__ == "__main__":
    main()