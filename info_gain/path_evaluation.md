# Path Evaluation Script

This script (`path_evaluation.py`) is designed to evaluate the impact of providing knowledge paths on a Language Model's (LM) ability to predict a target answer. The core idea is to measure how much a "support path" (a sequence of entities representing a reasoning chain) helps the model in generating or predicting the final entity of that path, when framed as an answer to a question.

## How it Works

The script performs the following steps:

1.  **Loads Data**: It reads input from a JSONL file. Each line in this file is expected to be a JSON object containing a "question" and a list of "paths".
2.  **Iterates Through Data**: For each item (question and its associated paths):
    *   It takes each path and considers the last entity in the path as the target "answer".
    *   It formats the path into a string representation (e.g., "EntityA -> EntityB -> EntityC").
3.  **Computes Probabilities**: For each path and its derived answer, it calculates two probabilities using a specified Hugging Face transformer model:
    *   **Baseline Probability**: The probability of the model predicting the `answer` given a `system_prompt` and the `question`.
        *   Prompt format: `"{system_prompt}\nQuestion: {question} Answer:"`
    *   **Retrieved Probability**: The probability of the model predicting the `answer` given the `system_prompt`, the `question`, and the `formatted_path` as supporting context.
        *   Prompt format: `"{system_prompt}\nSupport Path: {formatted_path}\nQuestion: {question} Answer:"`
    *   The script can use multiple system prompts and will average the probabilities across them.

    #### 3.1 Probability Calculation Details

    Let \( \text{prompt} \) be the input sequence to the model (either a baseline or augmented prompt). Let the target `answer` be \( A \).

    *   **Single-Token Answer**:
        If the `answer` \( A \) consists of a single token \( t_A \), its probability is calculated as:
        \[ P(A | \text{prompt}) = \text{softmax}(\text{logits}(\text{prompt}))[t_A] \]
        where \( \text{logits}(\text{prompt}) \) are the output logits from the language model for the next token prediction after processing the \( \text{prompt} \), and \( [t_A] \) denotes selecting the probability corresponding to the token \( t_A \).

    *   **Multi-Token Answer**:
        If the `answer` \( A \) consists of a sequence of tokens \( (t_1, t_2, \ldots, t_k) \), the script computes the probability by modeling it as a sequence of conditional probabilities. For each token \( t_i \) in the answer sequence:
        1.  The current prompt is \( \text{prompt}_{\text{current}} = \text{prompt} + t_1 + \ldots + t_{i-1} \).
        2.  The probability of the current token \( t_i \) is \( P(t_i | \text{prompt}_{\text{current}}) \).
        3.  The log-probability \( \log P(t_i | \text{prompt}_{\text{current}}) \) is computed.

        The script then calculates the mean of these log-probabilities:
        \[ \text{MeanLogProb}(A | \text{prompt}) = \frac{1}{k} \sum_{i=1}^{k} \log P(t_i | \text{prompt} + t_1 + \ldots + t_{i-1}) \]
        The final reported "probability" for the multi-token answer is the exponentiated mean of the log-probabilities, which is equivalent to the geometric mean of the individual token probabilities:
        \[ P(A | \text{prompt}) = \exp(\text{MeanLogProb}(A | \text{prompt})) = \left( \prod_{i=1}^{k} P(t_i | \text{prompt} + t_1 + \ldots + t_{i-1}) \right)^{1/k} \]
        This normalization helps in comparing probabilities of answers with different lengths.

4.  **Calculates Improvement**:
    *   **Absolute Improvement**: `avg_retrieved_prob - avg_baseline_prob`
    *   **Relative Improvement**: `(avg_retrieved_prob - avg_baseline_prob) / avg_baseline_prob` (handles cases where `avg_baseline_prob` is zero).
5.  **Stores Results**: The evaluation details for each path, including baseline probability, retrieved probability, and improvements, are stored.
6.  **Outputs Results**: All results are saved to a specified JSON output file. The script also prints summary statistics (average absolute and relative improvements) to the console.

## Dependencies

*   `torch`
*   `transformers`
*   `numpy`
*   `tqdm`

You can install these dependencies using pip:
```bash
pip install torch transformers numpy tqdm
```

## Usage

The script is run from the command line.

```bash
python info_gain/path_evaluation.py --model <MODEL_NAME_OR_PATH> \
                                  --input <PATH_TO_INPUT_JSONL> \
                                  --output <PATH_TO_OUTPUT_JSON> \
                                  [--device <cuda_or_cpu>] \
                                  [--max-samples <NUM_SAMPLES>] \
                                  [--system-prompts "Prompt 1" "Prompt 2"] \
                                  [--system-prompts-file <PATH_TO_PROMPTS_FILE>]
```

### Arguments:

*   `--model` (required): Name or path to a Hugging Face causal language model (e.g., `meta-llama/Llama-2-7b-chat-hf`).
*   `--input` (required): Path to the input JSONL file.
*   `--output` (required): Path where the output JSON results will be saved.
*   `--device` (optional, default: `cuda`): Device to run the model on (`cuda` or `cpu`).
*   `--max-samples` (optional, default: `None`): Maximum number of data samples (lines from the input file) to process. If not specified, all samples are processed.
*   `--system-prompts` (optional, default: A generic helpful assistant prompt): A list of system prompts to use. The evaluation will be run for each system prompt, and the results will be averaged.
*   `--system-prompts-file` (optional, default: `None`): Path to a text file containing system prompts, one prompt per line. These will be used if `--system-prompts` is not provided or to augment them.

### Example:

```bash
python info_gain/path_evaluation.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --input "data/input_data.jsonl" \
    --output "results/evaluation_results.json" \
    --max-samples 100 \
    --system-prompts-file "prompts/system_prompts.txt"
```

## Input Data Format

The input file (`--input`) must be a JSONL file, where each line is a JSON object with the following structure:

```json
{
  "id": "unique_identifier_for_the_item",
  "question": "The question related to the paths.",
  "paths": [
    ["Entity1_Path1", "Entity2_Path1", "Answer1"],
    ["EntityA_Path2", "EntityB_Path2", "EntityC_Path2", "Answer2"],
    [] // Empty paths are skipped
  ]
}
```
*   `id`: A unique identifier for the data point.
*   `question`: The question string. The script doesn't directly use the question to find the answer but uses it as part of the prompt to the LM.
*   `paths`: A list of paths. Each path is a list of strings, where each string is an entity. The last entity in each path is treated as the `answer` for evaluation purposes.

## Output Data Format

The output file (`--output`) will be a JSON file containing a list of results, one for each input item. Each item in the list will have the following structure:

```json
[
  {
    "id": "unique_identifier_for_the_item",
    "question": "The question related to the paths.",
    "path_evaluations": [
      {
        "path": ["Entity1_Path1", "Entity2_Path1", "Answer1"],
        "answer": "Answer1",
        "baseline_prob": 0.123, // Average baseline probability across system prompts
        "retrieved_prob": 0.456, // Average retrieved probability across system prompts
        "absolute_improvement": 0.333,
        "relative_improvement": 2.707, // (0.456 - 0.123) / 0.123
        "prompt_results": [ // Results for each individual system prompt
          {
            "system_prompt": "System prompt 1 text...",
            "baseline_prob": 0.120,
            "retrieved_prob": 0.450
          },
          {
            "system_prompt": "System prompt 2 text...",
            "baseline_prob": 0.126,
            "retrieved_prob": 0.462
          }
        ]
      },
      // ... more path evaluations for this question
    ]
  },
  // ... more items
]
```
*   `id`, `question`: Copied from the input.
*   `path_evaluations`: A list of evaluation results for each path associated with the question.
    *   `path`: The original path (list of strings).
    *   `answer`: The target answer (last entity of the path).
    *   `baseline_prob`: The model's probability of generating the answer given only the question (and system prompt), averaged over all system prompts.
    *   `retrieved_prob`: The model's probability of generating the answer given the question and the support path (and system prompt), averaged over all system prompts.
    *   `absolute_improvement`: `retrieved_prob - baseline_prob`.
    *   `relative_improvement`: Improvement relative to the baseline. Can be `Infinity` if `baseline_prob` is 0.
    *   `prompt_results`: A list detailing the baseline and retrieved probabilities for each system prompt used.

This README should provide a good overview of your script.