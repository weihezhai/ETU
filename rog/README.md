# ROG Folder Documentation

This document provides a comprehensive overview of the files and directories in the `rog` folder, which appears to implement a system for testing and evaluating Reasoning over Graph (ROG) prompts with various language models.

## Directory Structure

```
rog/
├── evaluation_results/
│   ├── predictions.jsonl_llm_results_rog.json
│   ├── eval_result_top15_sim.json
│   ├── evaluate_results.py
│   ├── top15_sim_filtered_paths_llm_results_04.json
│   └── top15_sim_filtered_paths_llm_results.json
├── qm/
│   ├── test_prompts_with_llm.py
│   ├── pp_sub_qm.sh
│   ├── run_llm_evaluation.sh
│   ├── test_single_prompt.py
│   ├── logs/
│   └── run_test_single_prompt.sh
├── topk_paths_prompts_rog/
│   ├── top1_sim_filtered_paths_with_prompts.jsonl
│   ├── top2_sim_filtered_paths_with_prompts.jsonl
│   ├── ...
│   └── top15_sim_filtered_paths_with_prompts.jsonl
├── run_test_single_prompt.sh
├── process_all_files.sh
├── generate_prompts.py
├── test_prompts_with_llm.py
├── test_single_prompt.py
└── run_llm_evaluation.sh
```

## Main Files

### Python Scripts

#### generate_prompts.py
- **Purpose**: Generates prompts for LLM evaluation based on reasoning paths
- **Main Functions**:
  - `format_path()`: Formats reasoning paths by connecting elements with `->`
  - `generate_prompt()`: Creates formatted prompts using reasoning paths and questions
  - `process_file()`: Processes JSONL files to add prompts to each entry
- **Usage**: Takes input JSONL files with questions and reasoning paths, adds formatted prompts, and outputs the enhanced JSONL

#### test_prompts_with_llm.py
- **Purpose**: Tests prompts with language models (specifically targeting Llama models)
- **Main Functions**:
  - `test_prompts_with_llm()`: Loads a language model, processes prompts, and generates responses
- **Features**:
  - Supports batch processing for efficiency
  - Configurable parameters (temperature, top_p, etc.)
  - Periodic saving of results to prevent data loss
- **Output**: JSON file containing generated results for each prompt

#### test_single_prompt.py
- **Purpose**: Tests a single prompt with a language model
- **Features**:
  - Similar to test_prompts_with_llm.py but focused on testing one prompt at a time
  - Useful for debugging and testing specific prompts

### Shell Scripts

#### run_llm_evaluation.sh
- **Purpose**: SLURM job script for running LLM evaluations on a compute cluster
- **Features**:
  - Configures GPU resources (H100 GPU)
  - Sets up environment (Anaconda, CUDA, cuDNN)
  - Processes multiple files in parallel using SLURM array jobs
  - Each job evaluates a different "top-k" filtered paths file

#### process_all_files.sh
- **Purpose**: Helper script to process all files in a given directory
- **Features**:
  - Automates the processing of multiple files
  - Likely used in conjunction with generate_prompts.py to prepare data for evaluation

#### run_test_single_prompt.sh
- **Purpose**: Script to test a single prompt with an LLM
- **Features**:
  - Simplified interface for testing individual prompts
  - Useful for quick tests and debugging

## Subdirectories

### evaluation_results/
- **Purpose**: Contains evaluation results and evaluation scripts
- **Key Files**:
  - `evaluate_results.py`: Script for comparing model outputs against ground truth
  - `*.json`: Result files from LLM evaluations
- **Features of evaluate_results.py**:
  - Calculates metrics like Hit@K and Hit@1
  - Handles fallback prediction logic
  - Outputs detailed evaluation metrics

### qm/
- **Purpose**: Contains scripts for "queen mary HPC" interactions, possibly a variant of the main ROG implementation
- **Key Files**:
  - Alternative versions of the main scripts adapted for the QM approach
  - Scripts for submitting evaluation jobs to a cluster

### topk_paths_prompts_rog/
- **Purpose**: Contains data files with top-k filtered reasoning paths and associated prompts
- **Content**:
  - 15 JSONL files representing different "top-k" settings (top1 through top15)
  - Each file contains filtered reasoning paths and generated prompts for questions
  - These files serve as the primary input for the LLM evaluation

## Workflow

The typical workflow appears to be:

1. Generate prompts from reasoning paths using `generate_prompts.py`
2. Store the resulting prompts in the `topk_paths_prompts_rog/` directory
3. Run evaluations on these prompts using `run_llm_evaluation.sh`, which utilizes `test_prompts_with_llm.py`
4. Store evaluation results in the `evaluation_results/` directory
5. Analyze results using `evaluation_results/evaluate_results.py`

## Usage Examples

### Generating Prompts
```bash
python generate_prompts.py --input input_data.jsonl --output topk_paths_prompts_rog/output_with_prompts.jsonl
```

### Running Evaluation
```bash
sbatch run_llm_evaluation.sh
```

### Evaluating Results
```bash
python evaluation_results/evaluate_results.py --cleaned results.json --ground-truth ground_truth.jsonl --fallback fallback.json --output detailed_results.json
```

### Testing a Single Prompt
```bash
./run_test_single_prompt.sh "Your prompt here"
```