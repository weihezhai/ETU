# FPPL (Filtered Path Perplexity for LLMs) Documentation

This document provides a comprehensive explanation of the files and directories in the `fppl` folder, which implements a system for extracting, processing, and evaluating reasoning paths using perplexity scoring with language models.

## Core Files

### Python Scripts

#### `extract_path.py`
- **Purpose**: Extracts reasoning paths from KGQA (Knowledge Graph Question Answering) results.
- **Key Functions**:
  - `extract_paths()`: Extracts reasoning paths from LLM-generated text
  - `is_valid_prediction()`: Validates if model predictions are subset of ground truth answers
  - `process_jsonl()`: Processes JSONL file to create dataset with reasoning paths
- **Output**: Creates a JSONL file containing questions, predictions, and extracted reasoning paths

#### `remove_middle_entity.py`
- **Purpose**: Transforms reasoning paths by removing intermediate entities and duplicate relations.
- **Key Functions**:
  - `transform_paths()`: Processes paths to keep only head entities, relations, and tail entities
- **Output**: Creates a JSONL file with transformed paths containing only head, relations, and tail

#### `calculate_path_ppl.py`
- **Purpose**: Calculates perplexity scores for path+question combinations using a language model.
- **Key Functions**:
  - `load_model_and_tokenizer()`: Loads LLM and tokenizer from local or HuggingFace
  - `compute_perplexity()`: Calculates perplexity scores for text using the LLM
  - `format_path()`: Formats a path list into a string with arrow notation
  - `calculate_path_ppl_scores()`: Main function to calculate perplexity for all paths
- **Output**: Creates a JSONL file with paths ranked by perplexity scores

#### `ppl_percentile.py`
- **Purpose**: Filters paths based on their perplexity score percentiles.
- **Key Functions**:
  - `load_jsonl()`: Loads data from JSONL files with fallbacks for malformed files
  - `save_jsonl()`: Saves data to JSONL format
  - `filter_by_ppl_percentile()`: Filters paths based on percentile thresholds
- **Output**: Creates a JSONL file containing only paths meeting the percentile criterion

#### `generate_prompts.py`
- **Purpose**: Generates prompts for LLM evaluation based on filtered paths.
- **Key Functions**:
  - `format_path()`: Formats a path by connecting elements with arrows
  - `generate_prompt()`: Creates a structured prompt with paths and question
  - `process_file()`: Processes JSONL files to add prompts to each entry
- **Output**: Creates a JSONL file with the original data plus generated prompts

### Shell Scripts

#### `run_calculate_path_ppl.sh`
- **Purpose**: Wrapper script to run the path perplexity calculation with command-line options.
- **Parameters**:
  - Input file path
  - Output file path
  - Model directory
  - Prompt format (e.g., path_then_question, integrated, etc.)

#### `calculate_path_ppl_sub.sh`
- **Purpose**: Slurm job submission script for running perplexity calculations on HPC clusters.
- **Features**:
  - Runs different prompt format variations as separate array jobs
  - Configures GPU resources and memory requirements
  - Sets up environment and dependencies

#### `ppl_percentile.sh`
- **Purpose**: Batch processes multiple JSONL files to filter by perplexity percentile.
- **Features**:
  - Processes all JSONL files in an input directory
  - Preserves directory structure in output location
  - Configurable percentile threshold

#### `generate_prompts.sh`
- **Purpose**: Batch processes filtered path files to generate prompts for all files.
- **Features**:
  - Processes all JSONL files in the input directory
  - Creates corresponding output files with prompts

## Data Files

#### `all_paths.jsonl`
- Contains extracted reasoning paths from KGQA results
- Input for path transformation and perplexity calculations

#### `no_middle_entity.jsonl`
- Contains transformed paths with intermediate entities removed
- Used as input for perplexity calculations

## Directories

#### `res/`
- Contains result files from perplexity calculations
- Raw output from various prompt formats

#### `res_filtered_percent/`
- Contains filtered results based on percentile thresholds
- Organized by percentile value (e.g., top 15%, top 75%)

#### `top_percentile_ppl_prompts/`
- Contains final prompt files ready for LLM evaluation
- Used as input to the LLM evaluation process

#### `logs/`
- Contains log files from batch processing and job submissions
- Useful for debugging and monitoring job status

#### `qm/` (Quality Measurement)
- Contains scripts for evaluating the quality of generated paths using LLMs
- Key files:
  - `test_prompts_with_llm.py`: Tests prompts with an LLM and saves results
  - `test_single_prompt.py`: Tests a single prompt for debugging
  - `run_llm_evaluation.sh`: Batch script for LLM evaluations
  - `pp_sub_qm.sh`: Submission script for quality measurement jobs

## Directory Structure Overview

```
fppl/
├── extract_path.py                   # Extract reasoning paths from KGQA results
├── all_paths.jsonl                   # Extracted paths
├── remove_middle_entity.py           # Remove intermediate entities from paths
├── no_middle_entity.jsonl            # Paths with intermediate entities removed
├── calculate_path_ppl.py             # Calculate perplexity scores
├── run_calculate_path_ppl.sh         # Script to run perplexity calculations
├── calculate_path_ppl_sub.sh         # Slurm submission script
├── ppl_percentile.py                 # Filter paths by percentile
├── ppl_percentile.sh                 # Batch process percentile filtering
├── generate_prompts.py               # Generate LLM prompts from filtered paths
├── generate_prompts.sh               # Batch process prompt generation
├── res/                              # Raw results directory
├── res_filtered_percent/             # Filtered results by percentile
├── top_percentile_ppl_prompts/       # Final prompts for LLM evaluation
├── logs/                             # Log files
└── qm/                               # Quality measurement scripts
    ├── test_prompts_with_llm.py      # Test prompts with LLM
    ├── test_single_prompt.py         # Test single prompt
    ├── run_llm_evaluation.sh         # Run LLM evaluation
    └── logs/                         # QM log files
```

## Workflow Summary

1. **Path Extraction**: Extract reasoning paths from KGQA results using `extract_path.py`
2. **Path Transformation**: Remove intermediate entities using `remove_middle_entity.py`
3. **Perplexity Calculation**: Calculate perplexity scores using `calculate_path_ppl.py`
4. **Percentile Filtering**: Filter paths by percentile using `ppl_percentile.py`
5. **Prompt Generation**: Generate prompts for filtered paths using `generate_prompts.py`
6. **LLM Evaluation**: Evaluate prompts using `test_prompts_with_llm.py` in the `qm/` directory

This workflow allows for automated extraction, filtering, and evaluation of reasoning paths for knowledge-based question answering tasks.