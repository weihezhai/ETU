import json
import argparse
import os
from collections import defaultdict

def string_overlap(str1, str2):
    """Check if either string contains the other."""
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    return str1 in str2 or str2 in str1

def evaluate_results(cleaned_results_file, ground_truth_file):
    # Load cleaned results
    with open(cleaned_results_file, 'r') as f:
        cleaned_data = json.load(f)
    
    # Load ground truth from JSONL file
    ground_truth = []
    with open(ground_truth_file, 'r') as f:
        for line in f:
            try:
                gt_entry = json.loads(line.strip())
                ground_truth.append(gt_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing ground truth line: {e}")
                continue
    
    # Metrics counters
    total_questions = len(cleaned_data)
    hit_count = 0
    h_at_1_count = 0
    
    # Results for detailed analysis
    results = []
    
    for entry in cleaned_data:
        question_id = entry.get("id", None)
        
        # Find matching ground truth
        gt_answers = None
        for gt_entry in ground_truth:
            if gt_entry.get("id") == question_id:
                gt_answers = gt_entry.get("ground_truth", [])
                break
        
        if not gt_answers:
            print(f"Warning: No ground truth found for question ID {question_id}")
            continue
        
        processed_results = entry.get("processed_results", [])
        
        # Calculate Hit (any match)
        hit = False
        for pred in processed_results:
            for gt in gt_answers:
                if string_overlap(pred, gt):
                    hit = True
                    break
            if hit:
                break
        
        # Calculate H@1 (first prediction matches)
        h_at_1 = False
        if processed_results:
            first_pred = processed_results[0]
            for gt in gt_answers:
                if string_overlap(first_pred, gt):
                    h_at_1 = True
                    break
        
        if hit:
            hit_count += 1
        if h_at_1:
            h_at_1_count += 1
        
        # Store individual result
        results.append({
            "id": question_id,
            "hit": hit,
            "h_1": h_at_1,
            "predictions": processed_results,
            "ground_truth": gt_answers
        })
    
    # Calculate metrics
    hit_rate = hit_count / total_questions if total_questions > 0 else 0
    h_at_1_rate = h_at_1_count / total_questions if total_questions > 0 else 0
    
    metrics = {
        "total_questions": total_questions,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "h_at_1_count": h_at_1_count,
        "h_at_1_rate": h_at_1_rate
    }
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate cleaned results against ground truth')
    parser.add_argument('--cleaned', '-c', required=True, help='Cleaned results JSON file path')
    parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth JSONL file path')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    metrics, results = evaluate_results(args.cleaned, args.ground_truth)
    
    # Print metrics
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Hit@K: {metrics['hit_count']} ({metrics['hit_rate']:.4f})")
    print(f"Hit@1: {metrics['h_at_1_count']} ({metrics['h_at_1_rate']:.4f})")
    
    # Save detailed results if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main() 