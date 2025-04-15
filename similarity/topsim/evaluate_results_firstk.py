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
    
    # Metrics for macro precision, recall, F1
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    valid_questions = 0  # Count of questions with ground truth
    
    # Counters for micro F1 calculation
    micro_tp_total = 0
    micro_predicted_total = 0
    micro_actual_total = 0
    
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
        
        valid_questions += 1
        processed_results = entry.get("processed_results", [])
        # Limit to the first 5 predictions if more exist
        if len(processed_results) > 5: # Check specifically for the file evaluate_results_firstk.py
            processed_results = processed_results[:3]
        
        # Calculate Hit (any match among the considered predictions)
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
        
        # Calculate Precision, Recall, F1 for this question (for macro average)
        true_positives = 0
        # Use a set to track matched ground truths for TP calculation within this question
        matched_gt_indices = set()
        for pred in processed_results:
            for i, gt in enumerate(gt_answers):
                if i not in matched_gt_indices and string_overlap(pred, gt):
                    true_positives += 1
                    matched_gt_indices.add(i) # Mark this GT as matched by this prediction
                    break # A prediction contributes at most one TP
        
        precision = true_positives / len(processed_results) if processed_results else 0
        recall = true_positives / len(gt_answers) if gt_answers else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Update micro F1 counters
        micro_tp_total += true_positives
        micro_predicted_total += len(processed_results)
        micro_actual_total += len(gt_answers)
        
        if hit:
            hit_count += 1
        if h_at_1:
            h_at_1_count += 1
        
        # Store individual result
        results.append({
            "id": question_id,
            "hit": hit,
            "h_1": h_at_1,
            "precision": precision, # Macro calculation component
            "recall": recall,       # Macro calculation component
            "f1": f1,               # Macro calculation component
            "predictions": processed_results,
            "ground_truth": gt_answers
        })
    
    # Calculate macro-average metrics
    hit_rate = hit_count / valid_questions if valid_questions > 0 else 0
    h_at_1_rate = h_at_1_count / valid_questions if valid_questions > 0 else 0
    macro_precision = total_precision / valid_questions if valid_questions > 0 else 0
    macro_recall = total_recall / valid_questions if valid_questions > 0 else 0
    macro_f1 = total_f1 / valid_questions if valid_questions > 0 else 0
    
    # Calculate micro-average metrics
    micro_precision = micro_tp_total / micro_predicted_total if micro_predicted_total > 0 else 0
    micro_recall = micro_tp_total / micro_actual_total if micro_actual_total > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    metrics = {
        "total_questions": total_questions,
        "valid_questions": valid_questions,
        "hit_count": hit_count,
        "hit_rate": hit_rate, # Effectively Hit@1 rate due to [:1] slicing
        "h_at_1_count": h_at_1_count,
        "h_at_1_rate": h_at_1_rate,
        "macro_precision": macro_precision, # Averaged per-question precision
        "macro_recall": macro_recall,       # Averaged per-question recall
        "macro_f1": macro_f1,               # Averaged per-question F1
        "micro_precision": micro_precision, # Precision calculated from total TP, FP
        "micro_recall": micro_recall,       # Recall calculated from total TP, FN
        "micro_f1": micro_f1                # F1 calculated from micro P/R
    }
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate cleaned results against ground truth, considering only the first prediction.')
    parser.add_argument('--cleaned', '-c', required=True, help='Cleaned results JSON file path')
    parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth JSONL file path')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    metrics, results = evaluate_results(args.cleaned, args.ground_truth)
    
    # Print metrics
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Valid questions (with ground truth): {metrics['valid_questions']}")
    # Since we only consider the first prediction, Hit@K is the same as Hit@1
    print(f"Hit@1 Count: {metrics['h_at_1_count']} ({metrics['h_at_1_rate']:.4f})")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro Recall: {metrics['micro_recall']:.4f}")
    print(f"Micro F1 Score: {metrics['micro_f1']:.4f}")
    
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