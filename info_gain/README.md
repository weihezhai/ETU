# Path Evaluation Tool

This tool measures how retrieved paths influence a language model's confidence in producing the correct answer to a question. It implements the method described in the project documentation for evaluating path utility.

## Core Concept

The method directly measures how retrieved texts influence the model's confidence in producing the correct final answer or prediction by:

1. Computing the probability of an answer given just the question (baseline)
2. Computing the probability of the same answer given the question plus a retrieved path
3. Measuring both absolute and relative improvement in probability

## Installation

```bash
pip install torch transformers numpy tqdm
```

## Usage

```python
from path_evaluation import PathEvaluator

# Initialize the evaluator with a model
evaluator = PathEvaluator("facebook/opt-1.3b")  # Choose appropriate model

# Evaluate paths in a JSONL file
results = evaluator.evaluate_paths(
    "fppl/all_paths.jsonl",
    num_samples=10,  # Optional, set to None to process all samples
    system_prompt="You are a helpful assistant that answers questions accurately based on the given information."
)

# Analyze results
# results contains details about how each path influenced answer probability
```

## Data Format

The tool expects JSONL files with the following structure for each line:

```json
{
  "id": "unique_id",
  "question": "Question text?",
  "prediction": "Answer1\nAnswer2\nAnswer3",
  "paths": [
    ["Entity1", "relation", "Entity2", "relation", "Entity3"],
    ["Entity1", "relation", "Entity4", "relation", "Entity5"]
  ]
}
```

Where:
- Multiple predictions are separated by newlines
- Each path is a list of entities and relations

## Method Details

### Step 1: Calculate Baseline Probability
- Compute P(answer|question) without any retrieved paths

### Step 2: Calculate Path-Augmented Probability
- For each path, compute P(answer|path+question)

### Step 3: Measure Improvement
- Absolute improvement = P(answer|path+question) - P(answer|question)
- Relative improvement = (P(answer|path+question) - P(answer|question)) / P(answer|question)

### Multi-token Handling
- For multi-token answers, the tool computes the joint probability of all tokens

## Output Format

The tool produces a structured output with the following information:

```json
[
  {
    "id": "unique_id",
    "question": "Question text?",
    "prediction": "Answer1",
    "path_evaluations": [
      {
        "path": ["Entity1", "relation", "Entity2", "relation", "Entity3"],
        "baseline_prob": 0.05,
        "retrieved_prob": 0.75,
        "absolute_improvement": 0.7,
        "relative_improvement": 14.0
      },
      ...
    ]
  },
  ...
]
```

The paths with the highest improvement values are considered the most valuable for answering the question. 
