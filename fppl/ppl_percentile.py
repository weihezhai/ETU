import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def filter_by_ppl_percentile(data: List[Dict[str, Any]], percentile: float, higher_is_better: bool = False) -> List[Dict[str, Any]]:
    """
    Filter paths based on PPL score percentile.
    
    Args:
        data: List of data items
        percentile: The percentile threshold (e.g., 15 for top 15%)
        higher_is_better: If True, higher PPL scores are better; if False, lower PPL scores are better
        
    Returns:
        Filtered data with paths meeting the percentile criterion
    """
    # Collect all PPL scores from all paths
    all_ppl_scores = []
    for item in data:
        for path in item.get('paths', []):
            if 'ppl_score' in path:
                all_ppl_scores.append(path['ppl_score'])
    
    if not all_ppl_scores:
        return data
    
    # Calculate the percentile threshold value
    if higher_is_better:
        # For higher_is_better=True, we want scores >= the (100-percentile)th percentile
        threshold = np.percentile(all_ppl_scores, 100 - percentile)
        keep_condition = lambda score: score >= threshold
    else:
        # For higher_is_better=False, we want scores <= the percentile-th percentile
        threshold = np.percentile(all_ppl_scores, percentile)
        keep_condition = lambda score: score <= threshold
    
    # Filter paths for each item
    filtered_data = []
    for item in data:
        filtered_item = item.copy()
        filtered_paths = [
            path for path in item.get('paths', [])
            if 'ppl_score' in path and keep_condition(path['ppl_score'])
        ]
        
        # If there are no paths left after filtering, keep the best path instead of skipping
        if not filtered_paths and item.get('paths'):
            paths_with_ppl = [p for p in item.get('paths', []) if 'ppl_score' in p]
            if paths_with_ppl:
                # Sort by PPL score (ascending or descending based on higher_is_better)
                # If higher_is_better, we want the highest score (reverse=True)
                # If not higher_is_better, we want the lowest score (reverse=False)
                sorted_paths = sorted(paths_with_ppl, key=lambda p: p['ppl_score'], reverse=higher_is_better)
                # Keep only the best path
                filtered_paths = [sorted_paths[0]]
        
        # Now we only skip if there are no valid paths at all
        if not filtered_paths:
            continue
        
        filtered_item['paths'] = filtered_paths
        filtered_data.append(filtered_item)
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description="Filter paths by PPL score percentile")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument("--percentile", type=float, default=15.0, 
                       help="Percentile threshold (e.g., 15 for top 15%)")
    parser.add_argument("--higher-is-better", action="store_true", 
                       help="If set, higher PPL scores are considered better")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the data
    data = load_jsonl(args.input_file)
    filtered_data = filter_by_ppl_percentile(data, args.percentile, args.higher_is_better)
    
    # Save the filtered data
    save_jsonl(filtered_data, args.output_file)
    
    print(f"Processed {len(data)} items, kept {len(filtered_data)} after filtering")
    
    # Calculate stats on kept paths
    total_input_paths = sum(len(item.get('paths', [])) for item in data)
    total_kept_paths = sum(len(item.get('paths', [])) for item in filtered_data)
    
    print(f"Input paths: {total_input_paths}")
    print(f"Kept paths: {total_kept_paths} ({total_kept_paths/total_input_paths*100:.2f}%)")


if __name__ == "__main__":
    main()