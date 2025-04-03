 # Similarity Module Documentation

This documentation provides a comprehensive overview of the `similarity` folder, which contains code for calculating and analyzing similarity between questions and paths in the ETU project.

## Main Files

### Python Scripts

#### `sim_question_path.py`
This script calculates similarity scores between questions and path strings using embedding models.
- **Main functionality**: Computes cosine similarity between question embeddings and path embeddings
- **Model used**: Alibaba-NLP/gte-Qwen2-7B-instruct
- **Key components**:
  - `load_model_and_tokenizer`: Loads embedding model and tokenizer either locally or from HuggingFace
  - `get_embedding`: Generates embeddings for input text using the model
  - `compute_similarity`: Calculates cosine similarity between question and prompt embeddings
  - `calculate_similarity_scores`: Processes input files to compute similarity for all question-path pairs

#### `sim_diff_hit_unhit.py`
Analyzes the relationship between similarity scores and whether paths are hits (end node matches ground truth) or not.
- **Main functionality**: Statistical analysis and visualization of similarity scores
- **Key components**:
  - `extract_features`: Extracts features from data including similarity scores, hit status, and normalizations
  - `statistical_analysis`: Performs various statistical tests to analyze the relationship between similarity and hit status
  - `plot_results`: Generates visualizations (histograms, ROC curves, PR curves, etc.)

### Shell Scripts

#### `run_similarity.sh`
Simple wrapper script to run the similarity calculation with command-line parameters.
- **Usage**: `./run_similarity.sh -i <input_file> -o <output_file> -m <model_dir>`
- **Parameters**:
  - `-i`: Input JSONL file
  - `-o`: Output JSONL file (with similarity scores)
  - `-m`: Model directory for caching the embedding model

#### `sub_similarity.sh`
SLURM job submission script for running similarity calculations on a GPU cluster.
- **Purpose**: Submits array jobs to process multiple input files in parallel
- **Configuration**:
  - Requests 1 GPU, 82GB memory, 12 CPUs
  - Sets up environment with Anaconda and CUDA
  - Processes a predefined list of input files

## Subdirectories

### `topsim/`
Contains code and resources for filtering and evaluating top-K paths based on similarity scores.

#### Key files:
- `filter_paths_by_relation.py`: Filters paths based on relation criteria
- `generate_prompts.py`: Generates prompts from filtered paths
- `evaluate_results.py`: Evaluates filtered path results
- `analyze_path_filtering.py`: Analyzes the effectiveness of path filtering

#### Subdirectories:
- `topk_paths/`: Stores filtered top-K paths
- `topk_paths_prompts/`: Stores prompts generated from top-K paths
- `evaluation_metrics/`: Contains evaluation metrics for filtered paths
- `evaluation_results/`: Stores raw evaluation results
- `evaluation_results_cleaned/`: Stores cleaned evaluation results

### `llm/`
Contains scripts for evaluating path quality using language models.

#### Key files:
- `test_prompts_with_llm.py`: Tests prompts using language models
- `test_single_prompt.py`: Tests individual prompts
- `run_llm_evaluation.sh`: Shell script for running LLM evaluations
- `run_test_single_prompt.sh`: Shell script for testing single prompts

### `sim_res/`
Storage directory for similarity calculation results.

### `logs/`
Contains log files from similarity calculation and analysis runs.

## Workflow

1. **Calculate Similarity Scores**:
   - Input: JSONL file with questions and paths
   - Process: Calculate similarity between questions and paths using `sim_question_path.py`
   - Output: JSONL file with similarity scores added to each path

2. **Analyze Similarity Results**:
   - Input: JSONL file with similarity scores
   - Process: Analyze and visualize results using `sim_diff_hit_unhit.py`
   - Output: Statistical analysis and visualization plots

3. **Filter Top-K Paths**:
   - Use the `topsim` module to filter paths based on similarity scores and other criteria
   - Generate prompts from filtered paths
   - Evaluate the quality of filtered paths

4. **LLM Evaluation**:
   - Use the `llm` module to evaluate path quality using language models
   - Test different prompt formats
   - Analyze LLM performance on different path types

## Usage Examples

### Calculate Similarity Scores
```bash
./run_similarity.sh -i "path_data.jsonl" -o "path_with_similarity.jsonl" -m "models/gte_qwen2"
```

### Analyze Similarity Results
```bash
python sim_diff_hit_unhit.py --input_file "path_with_similarity.jsonl" --output_dir "analysis_results"
```

### Submit GPU Job for Similarity Calculation
```bash
sbatch sub_similarity.sh
```