"""
Analyzes filtered paths to provide statistical insights and evaluate filtering effectiveness.

This script:
1. Counts questions with filtered paths vs original paths
2. Calculates average path counts before and after filtering
3. Analyzes path length distributions
4. Identifies common relations in filtered paths
5. Generates a summary report
"""

import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def analyze_filtered_paths(filtered_file, output_dir=None):
    """
    Analyze filtered paths and generate statistics.
    
    Args:
        filtered_file: Path to JSONL file with filtered paths
        output_dir: Directory to save analysis results (optional)
    """
    # Load filtered data
    data = []
    with open(filtered_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Initialize counters and statistics
    total_questions = len(data)
    questions_with_different_paths = 0
    original_path_counts = []
    filtered_path_counts = []
    original_path_lengths = []
    filtered_path_lengths = []
    relation_counter = Counter()
    
    # Process each question
    for item in tqdm(data, desc="Analyzing paths"):
        original_paths = item['paths']
        filtered_paths = item['filtered_path_by_relation']
        
        # Count paths
        original_count = len(original_paths)
        filtered_count = len(filtered_paths)
        original_path_counts.append(original_count)
        filtered_path_counts.append(filtered_count)
        
        # Check if filtering changed anything
        if original_paths != filtered_paths:
            questions_with_different_paths += 1
        
        # Path lengths
        for path in original_paths:
            original_path_lengths.append(len(path))
        
        for path in filtered_paths:
            filtered_path_lengths.append(len(path))
            # Count relations in filtered paths
            for i in range(1, len(path) - 1):
                if isinstance(path[i], str):
                    relation_counter[path[i]] += 1
    
    # Calculate statistics
    avg_original_paths = np.mean(original_path_counts)
    avg_filtered_paths = np.mean(filtered_path_counts)
    avg_original_length = np.mean(original_path_lengths)
    avg_filtered_length = np.mean(filtered_path_lengths)
    
    # Prepare results
    results = {
        "total_questions": total_questions,
        "questions_with_different_paths": questions_with_different_paths,
        "percentage_questions_changed": (questions_with_different_paths / total_questions) * 100,
        "avg_original_paths_per_question": avg_original_paths,
        "avg_filtered_paths_per_question": avg_filtered_paths,
        "path_reduction_percentage": ((avg_original_paths - avg_filtered_paths) / avg_original_paths) * 100 if avg_original_paths > 0 else 0,
        "avg_original_path_length": avg_original_length,
        "avg_filtered_path_length": avg_filtered_length,
        "most_common_relations": relation_counter.most_common(20)
    }
    
    # Print results
    print("\n===== Filtered Paths Analysis =====")
    print(f"Total questions analyzed: {results['total_questions']}")
    print(f"Questions with changed paths: {results['questions_with_different_paths']} ({results['percentage_questions_changed']:.2f}%)")
    print(f"Average paths per question: {results['avg_original_paths_per_question']:.2f} → {results['avg_filtered_paths_per_question']:.2f} ({results['path_reduction_percentage']:.2f}% reduction)")
    print(f"Average path length: {results['avg_original_path_length']:.2f} → {results['avg_filtered_path_length']:.2f}")
    
    print("\nTop 10 most common relations in filtered paths:")
    for rel, count in results['most_common_relations'][:10]:
        print(f"  {rel}: {count}")
    
    # Generate visualizations if output directory provided
    if output_dir:
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Path count distribution
        plt.figure(figsize=(10, 6))
        plt.hist([original_path_counts, filtered_path_counts], bins=20, 
                 label=['Original', 'Filtered'], alpha=0.7)
        plt.xlabel('Number of Paths per Question')
        plt.ylabel('Count')
        plt.title('Distribution of Path Counts')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'path_count_distribution.png'))
        
        # Path length distribution
        plt.figure(figsize=(10, 6))
        plt.hist([original_path_lengths, filtered_path_lengths], bins=20, 
                 label=['Original', 'Filtered'], alpha=0.7)
        plt.xlabel('Path Length')
        plt.ylabel('Count')
        plt.title('Distribution of Path Lengths')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'path_length_distribution.png'))
        
        # Save results as JSON
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis results and visualizations saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze filtered paths')
    parser.add_argument('--input', required=True, help='Path to filtered JSONL file')
    parser.add_argument('--output-dir', help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_filtered_paths(args.input, args.output_dir)