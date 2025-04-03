import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def average_results(file_paths):
    # Dictionary to store aggregated results
    aggregated = defaultdict(lambda: {
        'relative_improvement': [],
        'count': 0
    })
    
    # Process each file
    for file_path in file_paths:
        results = load_json(file_path)
        for result in results:
            id = result['id']
            for path_eval in result['path_evaluations']:
                path_str = str(path_eval['path'])  # Convert path to string for use as key
                key = (id, path_str)
                aggregated[key]['relative_improvement'].append(path_eval['relative_improvement'])
                aggregated[key]['count'] += 1
    
    # Calculate averages and prepare output
    output = []
    for (id, path_str), data in aggregated.items():
        if data['count'] == len(file_paths):  # Only include paths present in all files
            avg_prob = sum(data['relative_improvement']) / len(data['relative_improvement'])
            output.append({
                'id': id,
                'path': eval(path_str),  # Convert back to list
                'average_retrieved_prob': avg_prob
            })
    
    return output

def main():
    # List of your three JSON files
    file_paths = [
        '/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/no_middle_entity_path_eval_results_1.json',
        '/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/no_middle_entity_path_eval_results_2.json',
        '/data/home/mpx602/projects/ETU/ETU/info_gain/results/path_evaluation(src)/no_middle_entity_path_eval_results.json'
    ]
    
    # Calculate averages
    averaged_results = average_results(file_paths)
    
    # Save results to a new file
    output_file = 'averaged_results_rel_imp.json'
    with open(output_file, 'w') as f:
        json.dump(averaged_results, f, indent=2)
    
    print(f"Results averaged and saved to {output_file}")

if __name__ == "__main__":
    main() 