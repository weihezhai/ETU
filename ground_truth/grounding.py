import json
import re
import os
import anthropic
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm
import time


def input_to_path_list(input_text):
    """
    Converts a string input to a list of paths and a list of nodes.
    
    Args:
        input_text (str): The input string containing paths and nodes.
    
    Returns:
        List of valid paths and relevant lists of nodes.
    """
    # Find the reasoning paths section between "Reasoning Paths:" and "Question:"
    pattern = r'Reasoning Paths:\n(.*?)\n\nQuestion:'
    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        return [], []
    
    paths_text = match.group(1).strip()
    if not paths_text:
        return [], []
    
    paths = paths_text.split('\n')
    nodes = []
    # Split each path into nodes and clean up whitespace
    for path in paths:
        if '->' in path:
            # Split path into nodes and clean up whitespace
            elements = [elem.strip() for elem in path.split('->')]
            # Only include paths that end with one of the prediction values
            nodes.append(elements)
    return paths, nodes

def load_predictions(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    return predictions

def extract_ranked_paths_from_response(response: str) -> List[List[str]]:
    """Extract ranked paths from Claude's response between <paths></paths> tags."""
    pattern = r'<paths>(.*?)</paths>'
    match = re.search(pattern, response, re.DOTALL)
    
    if not match:
        # Try to find any list-like structure in the response
        print("No <paths> tags found, trying to extract list structure directly")
        list_pattern = r'\[\s*\[.*?\]\s*\]'
        list_match = re.search(list_pattern, response, re.DOTALL)
        if list_match:
            paths_text = list_match.group(0)
        else:
            print(f"Could not find list structure in response. Response snippet: {response[:200]}...")
            return []
    else:
        paths_text = match.group(1).strip()
    
    # Clean up the text to make it valid Python syntax
    paths_text = paths_text.replace('```python', '').replace('```', '').strip()
    
    try:
        # Try to parse as JSON first (safer than eval)
        import json
        ranked_paths = json.loads(paths_text)
        return ranked_paths
    except json.JSONDecodeError:
        try:
            # Fall back to eval with safety checks
            import ast
            ranked_paths = ast.literal_eval(paths_text)
            return ranked_paths
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing ranked paths: {e}")
            print(f"Raw response content: {paths_text[:200]}...")  # Print first 200 chars for debugging
            
            # Last resort: try to manually parse the response
            # Look for path entries that might be in a different format
            lines = paths_text.split('\n')
            ranked_paths = []
            current_path = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    # This might be a complete path
                    try:
                        path = ast.literal_eval(line)
                        if isinstance(path, list):
                            ranked_paths.append(path)
                    except:
                        pass
            
            if ranked_paths:
                return ranked_paths
            
            return []

def get_ranked_paths(client: anthropic.Client, question: str, paths: List[List[str]], model: str, max_retries: int = 3) -> List[List[str]]:
    """
    Query Claude to rank paths based on their explanability for a given question.
    
    Args:
        client: Anthropic Claude client
        question: The question to rank paths for
        paths: List of reasoning paths (as lists of nodes)
        model: Claude model to use
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of ranked paths, organized as a list of lists (up to 5 paths)
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Format paths for prompt while preserving original format
            paths_text = ""
            for i, path in enumerate(paths):
                # Use a string representation that shows the full path with exact node names
                path_str = " -> ".join(path)
                paths_text += f"{i+1}. {path_str}\n"
            
            prompt = f"""Please rank the following reasoning paths based on how well they explain or answer the given question. Return the original paths in your ranked order.

Question: {question}

Reasoning Paths:
{paths_text}

Analyze each path and rank them from most to least helpful for answering the question. Return the paths in their EXACT ORIGINAL FORMAT with the same node and relation names, just in your ranked order.

Output your ranked paths as a Python list of lists between <paths></paths> tags, like this:
<paths>
[
    ["EXACTLY", "SAME", "NODE", "AND", "RELATION", "NAMES"],
    ["ANOTHER", "PATH", "WITH", "EXACT", "NAMES"],
    ...more paths...
]
</paths>

IMPORTANT: PRESERVE THE EXACT ORIGINAL SPELLING AND FORMAT of each node and relation name in the paths. Do not modify, rewrite, or summarize them.
"""
            
            # Make API call
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract ranked paths from response
            ranked_paths = extract_ranked_paths_from_response(response.content[0].text)
            
            if ranked_paths:
                # Return only top 5 paths if there are more than 5
                if len(ranked_paths) > 5:
                    return ranked_paths[:5]
                return ranked_paths
            else:
                print(f"Failed to extract ranked paths (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
        
        except Exception as e:
            print(f"Error during API call (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(5)  # Longer wait for API errors
    
    print("All retry attempts failed")
    return []

def process_predictions(predictions_path: str, api_key: str, output_path: str, limit: int = None, model: str = "claude-3-7-sonnet-20250219"):
    """
    Process predictions and generate ground truth.
    
    Args:
        predictions_path: Path to the predictions JSONL file
        api_key: Anthropic API key
        output_path: Path to save the results
        limit: Optional limit on the number of predictions to process
        model: Claude model to use
    """
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load predictions
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    # Apply limit if specified
    if limit is not None:
        predictions = predictions[:limit]
    
    # Check if output file exists and load existing results
    results = {}
    completed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                completed_ids = set(results.keys())
                print(f"Loaded {len(completed_ids)} existing results from {output_path}")
        except json.JSONDecodeError:
            print(f"Error loading existing results from {output_path}, starting fresh")
    
    # Create log file for completed IDs
    log_path = f"{os.path.splitext(output_path)[0]}_completed_ids.log"
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Process each prediction
        api_call_counter = 0
        
        for pred in tqdm(predictions, desc="Processing predictions"):
            question_id = pred.get("id", "unknown")
            
            # Skip if already processed
            if question_id in completed_ids:
                continue
                
            question = pred.get("question", "")
            
            if not question:
                print(f"Skipping {question_id}: No question found")
                continue
            
            # Extract paths directly from the 'paths' field without any conversion
            paths = pred.get("paths", [])
            
            if not paths:
                print(f"Skipping {question_id}: No paths found")
                continue
            
            try:
                # Get ranked paths from Claude using the paths directly from the JSONL
                ranked_paths = get_ranked_paths(client, question, paths, model)
                api_call_counter += 1
                
                if ranked_paths:
                    results[question_id] = {
                        "question": question,
                        "ranked_paths": ranked_paths,
                        "predictions": pred.get("prediction", [])
                    }
                    completed_ids.add(question_id)
                    
                    # Log completed ID
                    log_file.write(f"Completed: {question_id}\n")
                    log_file.flush()  # Ensure it's written immediately
                else:
                    print(f"Failed to get ranked paths for {question_id}")
                
                # Save results every 10 API calls
                if api_call_counter % 10 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Checkpoint saved: {api_call_counter} items processed, {len(completed_ids)} total completed")
                    
            except Exception as e:
                print(f"Error processing {question_id}: {e}")
                # Save current results before giving up
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved progress after error. Completed {len(completed_ids)} items.")
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Completed! Results saved to {output_path}")
    print(f"Processed {len(completed_ids)} questions.")

def main():
    parser = argparse.ArgumentParser(description="Generate ground truth for reasoning paths using Claude API")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl file")
    parser.add_argument("--output", required=True, help="Path to save the results")
    parser.add_argument("--api_key", required=True, help="Anthropic API key")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of predictions to process")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use (default: claude-3-7-sonnet-20250219)")
    
    args = parser.parse_args()
    
    process_predictions(args.predictions, args.api_key, args.output, args.limit, args.model)

if __name__ == "__main__":
    main()