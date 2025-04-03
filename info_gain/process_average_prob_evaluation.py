import json
import os
from collections import defaultdict

def process_data(input_file, output_file, topk=None):
    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group data by question ID
    questions = defaultdict(list)
    for entry in data:
        question_id = entry.get("id")
        path = entry.get("path", [])
        avg_prob = entry.get("average_retrieved_prob", 0)
        
        # Extract the last entity from the path as the answer
        if path and len(path) > 0:
            answer = path[-1]
            questions[question_id].append({
                "answer": answer,
                "avg_prob": avg_prob,
                "path": path
            })
    
    # Create formatted output with processed_results field expected by the evaluation script
    formatted_data = []
    for question_id, answers in questions.items():
        # Sort answers by average probability (higher first)
        sorted_answers = sorted(answers, key=lambda x: x["avg_prob"], reverse=True)
        
        # Limit to top-k answers if specified
        if topk is not None:
            sorted_answers = sorted_answers[:topk]
        
        # Remove duplicate answers (keeping only the first occurrence which has highest probability)
        unique_answers = []
        seen_answers = set()
        for answer_obj in sorted_answers:
            answer = answer_obj["answer"]
            if answer not in seen_answers:
                unique_answers.append(answer_obj)
                seen_answers.add(answer)
        
        # Extract just the answer strings
        processed_results = [answer["answer"] for answer in unique_answers]
        
        formatted_data.append({
            "id": question_id,
            "processed_results": processed_results
        })
    
    # Write the formatted data to output file
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Processed {len(formatted_data)} questions with answers")
    if topk is not None:
        print(f"Limited to top-{topk} answers per question")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process averaged results for evaluation')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path (averaged_results.json)')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path for processed results')
    parser.add_argument('--topk', '-k', type=int, help='Limit to top-k answers with highest average probability')
    
    args = parser.parse_args()
    
    process_data(args.input, args.output, args.topk)