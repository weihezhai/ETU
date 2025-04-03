import json
import re
import argparse

def clean_generated_results(input_file, output_file, top_k=5):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        if "generated_result" in entry:
            # Step 1: Split by newline and comma
            result_text = entry["generated_result"]
            # Split by newline first
            items = result_text.split('\n')
            # Further split by comma and flatten the list
            split_items = []
            for item in items:
                split_items.extend([s.strip() for s in item.split(',')])
            
            # Step 2: Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in split_items:
                if item and item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            
            # Step 3: Remove specific phrases
            phrases_to_remove = [
                'None', 'Neither', 'Both', 'All', 'I don\'t know', 
                'One of them', 'Neither of them'
            ]
            # Use regex to match any "*of the above" pattern
            of_the_above_pattern = re.compile(r'.*\s+of\s+the\s+above', re.IGNORECASE)
            
            filtered_items = [item for item in unique_items 
                             if not any(phrase.lower() in item.lower() for phrase in phrases_to_remove)
                             and not of_the_above_pattern.match(item)]
            
            # Step 4: Keep top k answers
            top_results = filtered_items[:top_k]
            
            # Store both the processed list and a cleaned string
            entry["processed_results"] = top_results
            entry["cleaned_result_string"] = ", ".join(top_results)
    
    # Write the updated data to output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    return data

def main():
    parser = argparse.ArgumentParser(description='Clean generated results in JSON file')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of top answers to keep (default: 5)')
    
    args = parser.parse_args()
    
    cleaned_data = clean_generated_results(args.input, args.output, args.top_k)
    print(f"Processed {len(cleaned_data)} entries. Results saved to {args.output}")

if __name__ == "__main__":
    main()