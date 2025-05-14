import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
import os

class PathEvaluator:
    def __init__(self, model_name, device="cuda"):
        """Initialize the evaluator with a model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        
    def _format_path(self, path):
        """Format a path for inclusion in prompts."""
        return " -> ".join(path)
    
    def compute_answer_probability(self, prompt, answer):
        """
        Compute the probability of an answer given a prompt.
        Handles multi-token answers.
        """
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # If answer is single token
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        
        if len(answer_tokens) == 1:
            # Get logits for the last token
            with torch.no_grad():
                outputs = self.model(prompt_tokens)
                logits = outputs.logits[0, -1, :]
                
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Return probability of the answer token
            return probs[answer_tokens[0]].item()
        else:
            # For multi-token answers, compute joint probability
            log_probs = []
            current_prompt = prompt
            
            for token_id in answer_tokens:
                # Encode the current prompt
                current_tokens = self.tokenizer.encode(current_prompt, return_tensors="pt").to(self.device)
                
                # Get logits for the last token
                with torch.no_grad():
                    outputs = self.model(current_tokens)
                    logits = outputs.logits[0, -1, :]
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Add log probability to list
                log_probs.append(torch.log(probs[token_id]).item())
                
                # Update the prompt with this token
                current_prompt += self.tokenizer.decode([token_id])
            
            # Return normalized geometric mean (equivalent to exp of mean log prob)
            return np.exp(np.mean(log_probs))
    
    def evaluate_paths(self, data_file, output_file, num_samples=None, system_prompts=None):
        """
        Evaluate the impact of each path on answer probability.
        
        Args:
            data_file: Path to the JSONL file containing questions, answers, and paths
            output_file: Path to save the results
            num_samples: Number of samples to evaluate (None for all)
            system_prompts: List of system prompts to use and average results across
            
        Returns:
            List of dictionaries with evaluation results
        """
        results = []
        
        if system_prompts is None or len(system_prompts) == 0:
            system_prompts = ["You are a helpful assistant that answers questions accurately based on the given information."]
        
        # Load data
        with open(data_file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Limit number of samples if specified
        if num_samples:
            data = data[:num_samples]
        
        for item in tqdm(data):
            question = item["question"]
            paths = item["paths"]
            
            item_results = {
                "id": item["id"],
                "question": question,
                "path_evaluations": []
            }
            
            # For each path, use the last entity as the answer
            for path in paths:
                if not path or len(path) == 0:
                    continue
                
                answer = path[-1]  # Last entity in the path
                formatted_path = self._format_path(path)
                
                # Store probabilities across all system prompts
                baseline_probs = []
                retrieved_probs = []
                
                # Evaluate with each system prompt
                for system_prompt in system_prompts:
                    # Compute baseline probability (without retrieval)
                    baseline_prompt = f"{system_prompt}\nQuestion: {question} Answer:"
                    baseline_prob = self.compute_answer_probability(baseline_prompt, answer)
                    baseline_probs.append(baseline_prob)
                    
                    # Compute probability with retrieved path
                    augmented_prompt = f"{system_prompt}\nSupport Path: {formatted_path}\nQuestion: {question} Answer:"
                    retrieved_prob = self.compute_answer_probability(augmented_prompt, answer)
                    retrieved_probs.append(retrieved_prob)
                
                # Average probabilities across all system prompts
                avg_baseline_prob = np.mean(baseline_probs)
                avg_retrieved_prob = np.mean(retrieved_probs)
                
                # Calculate improvements
                absolute_improvement = avg_retrieved_prob - avg_baseline_prob
                relative_improvement = (avg_retrieved_prob - avg_baseline_prob) / avg_baseline_prob if avg_baseline_prob > 0 else float('inf')
                
                # Add individual prompt results for analysis
                prompt_results = []
                for i, system_prompt in enumerate(system_prompts):
                    prompt_results.append({
                        "system_prompt": system_prompt,
                        "baseline_prob": baseline_probs[i],
                        "retrieved_prob": retrieved_probs[i]
                    })
                
                item_results["path_evaluations"].append({
                    "path": path,
                    "answer": answer,
                    "baseline_prob": avg_baseline_prob,
                    "retrieved_prob": avg_retrieved_prob,
                    "absolute_improvement": absolute_improvement,
                    "relative_improvement": relative_improvement,
                    "prompt_results": prompt_results
                })
            
            results.append(item_results)
        
        # Save results to the output file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Print summary statistics
        print("Average absolute improvement across all paths:")
        all_abs_improvements = [path_eval["absolute_improvement"] 
                               for result in results 
                               for path_eval in result["path_evaluations"]]
        print(np.mean(all_abs_improvements))
        
        print("Average relative improvement across all paths:")
        all_rel_improvements = [path_eval["relative_improvement"] 
                               for result in results 
                               for path_eval in result["path_evaluations"]
                               if not np.isinf(path_eval["relative_improvement"])]
        print(np.mean(all_rel_improvements))
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the impact of paths on answer probability")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--system-prompts", type=str, nargs="+", default=None, 
                        help="List of system prompts to use and average results across")
    parser.add_argument("--system-prompts-file", type=str, default=None,
                        help="Path to a file containing system prompts, one per line")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle system prompts from command line or file
    system_prompts = args.system_prompts
    if args.system_prompts_file:
        with open(args.system_prompts_file, 'r') as f:
            system_prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Initializing evaluator with model: {args.model}")
    evaluator = PathEvaluator(args.model, args.device)
    
    print(f"Evaluating paths in: {args.input}")
    print(f"Results will be saved to: {args.output}")
    print(f"Using {len(system_prompts) if system_prompts else 1} system prompts for evaluation")
    
    evaluator.evaluate_paths(
        args.input,
        args.output,
        num_samples=args.max_samples,
        system_prompts=system_prompts
    )
    
    print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main() 