import json
import argparse
from tqdm import tqdm

def analyze_path_filtering(original_file, filtered_file):
    """
    Compare original and filtered JSONL files to analyze path filtering statistics.
    
    Args:
        original_file: Path to original JSONL file
        filtered_file: Path to filtered JSONL file with the filtered_path_by_relation_similarity field
    
    Returns:
        Dictionary with statistics
    """
    # Load the original data
    original_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_data.append(json.loads(line.strip()))
    
    # Load the filtered data
    filtered_data = []
    with open(filtered_file, 'r', encoding='utf-8') as f:
        for line in f:
            filtered_data.append(json.loads(line.strip()))
    
    # Ensure data is the same length and order
    assert len(original_data) == len(filtered_data), "Files have different numbers of questions"
    
    # Initialize variables for statistics
    total_original_paths = 0
    total_filtered_paths = 0
    questions_no_paths = 0
    questions_all_paths = 0
    
    # Compare each question
    for orig, filt in tqdm(zip(original_data, filtered_data), total=len(original_data), desc="Analyzing questions"):
        # Get original paths
        orig_paths = orig['paths']
        # Get filtered paths
        filt_paths = filt['filtered_path_by_relation_similarity']
        
        # Update total counts
        total_original_paths += len(orig_paths)
        total_filtered_paths += len(filt_paths)
        
        # Check if no paths were selected
        if len(filt_paths) == 0:
            questions_no_paths += 1
        
        # Check if all paths remained
        if len(filt_paths) == len(orig_paths):
            # Need to compare sets of paths as lists might have different orders
            orig_paths_set = {tuple(map(tuple, path)) if isinstance(path[0], list) else tuple(path) for path in orig_paths}
            filt_paths_set = {tuple(map(tuple, path)) if isinstance(path[0], list) else tuple(path) for path in filt_paths}
            
            if orig_paths_set == filt_paths_set:
                questions_all_paths += 1
    
    # Calculate averages
    avg_original_paths = total_original_paths / len(original_data) if original_data else 0
    avg_filtered_paths = total_filtered_paths / len(filtered_data) if filtered_data else 0
    
    # Return the statistics
    return {
        "avg_original_paths": avg_original_paths,
        "avg_filtered_paths": avg_filtered_paths,
        "questions_no_paths": questions_no_paths,
        "questions_all_paths": questions_all_paths,
        "total_questions": len(original_data),
        "total_original_paths": total_original_paths,
        "total_filtered_paths": total_filtered_paths
    }

def save_analysis_results(stats, output_file):
    """
    Save analysis results to a file.
    
    Args:
        stats: Dictionary with statistics
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PATH FILTERING ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total questions: {stats['total_questions']}\n")
        f.write(f"Total paths (original): {stats['total_original_paths']}\n")
        f.write(f"Total paths (filtered): {stats['total_filtered_paths']}\n")
        f.write(f"Average paths per question (original): {stats['avg_original_paths']:.2f}\n")
        f.write(f"Average paths per question (filtered): {stats['avg_filtered_paths']:.2f}\n")
        if stats['avg_original_paths'] > 0:
            reduction = (1 - stats['avg_filtered_paths']/stats['avg_original_paths']) * 100
            f.write(f"Reduction percentage: {reduction:.2f}%\n")
        f.write(f"Questions with NO paths selected: {stats['questions_no_paths']} ({stats['questions_no_paths']/stats['total_questions'] * 100:.2f}%)\n")
        f.write(f"Questions with ALL paths retained: {stats['questions_all_paths']} ({stats['questions_all_paths']/stats['total_questions'] * 100:.2f}%)\n")
        f.write(f"Questions with SOME paths filtered: {stats['total_questions'] - stats['questions_no_paths'] - stats['questions_all_paths']} ({(stats['total_questions'] - stats['questions_no_paths'] - stats['questions_all_paths'])/stats['total_questions'] * 100:.2f}%)\n")
    
    print(f"Analysis results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare original and filtered path files')
    parser.add_argument('--original', required=True, help='Path to original JSONL file')
    parser.add_argument('--filtered', required=True, help='Path to filtered JSONL file')
    parser.add_argument('--output', required=False, help='Path to save analysis results', default='analysis_results.txt')
    
    args = parser.parse_args()
    
    stats = analyze_path_filtering(args.original, args.filtered)
    
    # Print to console
    print("\nPATH FILTERING ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Total paths (original): {stats['total_original_paths']}")
    print(f"Total paths (filtered): {stats['total_filtered_paths']}")
    print(f"Average paths per question (original): {stats['avg_original_paths']:.2f}")
    print(f"Average paths per question (filtered): {stats['avg_filtered_paths']:.2f}")
    if stats['avg_original_paths'] > 0:
        reduction = (1 - stats['avg_filtered_paths']/stats['avg_original_paths']) * 100
        print(f"Reduction percentage: {reduction:.2f}%")
    print(f"Questions with NO paths selected: {stats['questions_no_paths']} ({stats['questions_no_paths']/stats['total_questions'] * 100:.2f}%)")
    print(f"Questions with ALL paths retained: {stats['questions_all_paths']} ({stats['questions_all_paths']/stats['total_questions'] * 100:.2f}%)")
    print(f"Questions with SOME paths filtered: {stats['total_questions'] - stats['questions_no_paths'] - stats['questions_all_paths']} ({(stats['total_questions'] - stats['questions_no_paths'] - stats['questions_all_paths'])/stats['total_questions'] * 100:.2f}%)")
    
    # Save to file
    save_analysis_results(stats, args.output) 