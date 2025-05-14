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
                merged_results[id] = {"processed_results": {}}
            
            # Assign weights based on rank (reversed, so first gets highest weight)
            for i, pred in enumerate(entry["processed_results"]):
                weight = len(entry["processed_results"]) - i
                if pred in merged_results[id]["processed_results"]:
                    merged_results[id]["processed_results"][pred] += weight
                else:
                    merged_results[id]["processed_results"][pred] = weight
    
    # Rerank predictions based on weights
    final_results = []
    for id, data in merged_results.items():
        sorted_preds = sorted(data["processed_results"].items(), key=lambda x: x[1], reverse=True)
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
    
    merged_results = merge_predictions(data1, data2)
    
    # Calculate proper directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write merged results
    with open(args.output, 'w') as f:
        json.dump(merged_results, f, indent=2)

if __name__ == "__main__":
    main()