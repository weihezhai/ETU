# Similarity Score Analysis for Question-Path Pairs

## Overview

This tool statistically analyzes the relationship between similarity scores of reasoning paths and their correctness. It helps determine whether higher similarity scores between a question and a reasoning path correlate with the path ending at a correct answer (ground truth).

The script handles various data scenarios, including:
- Questions with multiple reasoning paths
- Questions with only one reasoning path
- Questions where no paths lead to correct answers
- Questions where some paths lead to correct answers

## Installation

### Prerequisites

This script requires the following Python packages:
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

## Usage

Run the script from the command line:

```bash
python sim_diff_hit_unhit.py --input path/to/your/data.jsonl --output results_directory
```

### Command-line Arguments

- `--input`: Path to your JSONL file containing questions and paths (required)
- `--output`: Directory where results will be saved (default: "./analysis_results")

## Input Data Format

The script expects a JSONL file where each line contains a JSON object with the following structure:

```json
{
    "id": "unique_question_id",
    "question": "The question text",
    "ground_truth": ["Correct answer 1", "Correct answer 2"],
    "prediction": "Model prediction",
    "prompt_format": "format_type",
    "paths": [
        {
            "path": ["Node1", "relation.type", "next_node", "Destination"],
            "path_str": "Node1 -> relation.type -> next_node -> Destination",
            "ppl_score": 40.90657615599676,
            "prompt_format": "format_type",
            "prompt": "The reasoning path...",
            "similarity_score": 0.3508232831954956,
            "shortened_path": "Node1 -> relation.type -> next_node"
        },
        // More paths...
    ]
}
```

## Analysis Performed

The script performs several statistical analyses to determine the relationship between similarity scores and correct answers:

### Basic Statistics
- Mean and standard deviation of similarity scores for hit and no-hit paths
- Hit rate (proportion of paths leading to correct answers)
- Path count distribution

### Statistical Tests
- **T-test**: Determines if the difference in similarity scores between hit and no-hit paths is statistically significant 
- **Mann-Whitney U test**: Non-parametric alternative to t-test
- **Point-biserial correlation**: Measures correlation between similarity scores and correctness
- **Logistic regression**: Models relationship between similarity scores and probability of correctness

### Question-Level Analysis
- Accuracy when selecting the path with highest similarity score
- Analysis of normalized similarity scores within each question
- Rank comparison of hit vs. non-hit paths
- Comparison between questions with all unhit paths and questions with hits

## Visualizations

The script generates numerous visualizations to help understand the data:

### Overall Distributions
- Distribution of similarity scores for hit vs. no-hit paths
- ROC curve measuring classification performance
- Precision-Recall curve
- Boxplot of similarity scores by hit/no-hit status

### Question-Level Visualizations
- Distribution of normalized similarity scores
- Path count distribution (how many questions have 1, 2, 3... paths)
- Rank boxplot for multi-path questions
- Comparison of similarity scores between questions with hits and questions without hits

### Special Case Analysis
- Similarity score distribution for questions with all unhit paths
- Heatmap showing hit rate of highest similarity paths split by question type

## Interpretation of Results

The script provides detailed interpretation of the analysis results:

### Statistical Significance
- Whether the difference in similarity scores between hits and non-hits is statistically significant
- Strength and direction of correlation between similarity scores and correctness
- Odds ratio interpretation from logistic regression

### Question-Level Interpretation
- How often the highest similarity path is a correct path
- For questions with correct answers, how often the highest similarity path is that correct answer
- Cross-question generalization analysis (do similarity scores work consistently across questions)

## Example Output

The script generates:

1. Terminal output with detailed statistics and interpretations
2. Visualization files saved to the output directory
3. A CSV file with processed data for further analysis
4. A JSON file with statistical results

### Terminal Output

```
===== Statistical Analysis Results =====
Number of paths analyzed: 58514
Number of questions: 3317
Questions with only 1 path: 428 (12.90%)
Questions with all unhit paths: 1016 (30.63%)
Hit rate: 0.1995

Mean similarity score for hits: 0.3818 ± 0.1168
Mean similarity score for non-hits: 0.3634 ± 0.1200

T-test: t=15.1905, p-value=0.00000000
Mann-Whitney U test: U=299381406.5000, p-value=0.00000000

Point-biserial correlation: r=0.0616, p-value=0.00000000
Normalized score correlation: r=-0.0091, p-value=0.02713268

Logistic Regression:
Coefficient: 1.2778
Intercept: -1.8656
Odds Ratio: 3.5889

===== Question-Level Analysis =====
Questions with at least one hit: 2301 out of 3317 (69.37%)
Accuracy when taking highest similarity path: 0.3552
Accuracy when taking highest similarity path (multi-path questions only): 0.3357
For questions with hits, highest similarity path accuracy: 0.5260
For questions with hits (multi-path only), highest similarity path accuracy: 0.4949
Average rank of hit paths within their questions: 7.43

Analysis results and visualizations saved to ./sim_res/analysis_results

===== Interpretation =====
The difference in similarity scores between hits and non-hits is statistically significant.
There is a weak positive correlation between similarity scores and correctness.
For each unit increase in similarity score, the odds of being correct increase by a factor of 3.59.

===== Question-Level Interpretation =====
For questions that have at least one correct path, the highest similarity path is that correct path 52.60% of the time.

Cross-question similarity analysis:
Maximum average similarity score for questions with NO hits: 0.8317
Minimum average similarity score for questions with hits: 0.0805
WARNING: Some questions with no hits have higher average similarity scores than questions with hits.
This suggests similarity scores may not generalize well across different questions.
Within-question comparisons are likely more reliable than across-question comparisons.
```

## Troubleshooting

### Common Issues

1. **Not enough data**: If you have too few examples or only examples of one class (all hits or all misses), some statistical tests will be skipped.

2. **Errors in visualization**: If certain visualizations fail, check that your data contains sufficient examples of both hit and no-hit paths.

3. **NaN values**: The script handles missing values, but if you see many NaN results, check your input data format.

## Contributing

Feel free to submit issues or pull requests for improvements to this analysis tool. 