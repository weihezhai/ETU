# Path-based LLM Reasoning Evaluation

This directory contains scripts for evaluating the ability of LLMs to perform reasoning based on knowledge paths.

## Overview

The workflow consists of:

1. **Prompt Generation**: Create prompts with reasoning paths for each question
2. **LLM Evaluation**: Test the LLM's ability to answer questions using the generated prompts
3. **Result Evaluation**: Compare the LLM's responses with ground truth or predictions

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Scripts

### Prompt Generation

- **generate_prompts.py**: Processes JSONL files to generate prompts for each question
- **process_all_files.sh**: Batch processes all the JSONL files
- **test_prompt_generation.py**: Tests prompt generation with a single example

### LLM Evaluation

- **test_prompts_with_llm.py**: Tests the LLM's ability to answer questions using the generated prompts
- **run_llm_evaluation.sh**: Batch processes all the files
- **test_single_prompt.py**: Tests a single prompt for debugging
- **run_test_single_prompt.sh**: Examples of testing a single prompt

### Result Evaluation

- **evaluate_results.py**: Evaluates the quality of LLM-generated answers
- **run_evaluation.sh**: Batch evaluates all result files

## Usage

### 1. Generate Prompts

```bash
# First edit process_all_files.sh to match your environment
bash process_all_files.sh
```

This creates a "topk_paths_prompts" directory with processed JSONL files containing prompts.

### 2. Test with LLM

Before running the LLM evaluation, make sure to:
- Update the model path in `run_llm_evaluation.sh` to point to your local Llama 3.1 8B model
- Adjust batch size and other parameters as needed

```bash
bash run_llm_evaluation.sh
```

This creates "evaluation_results" directory with JSON files containing the LLM outputs.

### 3. Test a Single Prompt

For debugging or testing specific examples:

```bash
# Update the model path in run_test_single_prompt.sh
bash run_test_single_prompt.sh
```

### 4. Evaluate Results

```bash
bash run_evaluation.sh
```

This creates "evaluation_metrics" directory with JSON files containing evaluation metrics.

## Prompt Format

The prompt format follows this structure:

```
Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list. The answer list is wrapped with [ANS][/ANS], each entry in the answer list can contain nothing but the answer text itself.

Reasoning Paths:
[Entity1] -> [Relation1] -> [Entity2]
[Entity1] -> [Relation2] -> [Entity3]
...

Question:
[Question Text]
```

## Output Format

The LLM evaluation outputs a JSON file with:
- `id`: The question ID
- `question`: The original question
- `generated_result`: The LLM's response

The evaluation metrics include:
- Exact match percentage
- Partial match percentage
- Semantic similarity scores
- Example answers for each category 