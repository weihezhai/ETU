#!/bin/bash

# Default values
INPUT_FILE="/data/home/mpx602/projects/ETU/ETU/similarity/sim_res/path_ppl_scores_explicit_reasoning_with_sim.jsonl"
GROUND_TRUTH="/data/home/mpx602/projects/ETU/ETU/fppl/all_ppl_path_with_gt_only_relation/path_ppl_scores_explicit_reasoning.jsonl"
OUTPUT_DIR="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/last_entity_ans_path_ordered_by_gt_rel_sim/gt_relation"
TOP_K=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
GT_RELATIONS_FILE="/data/home/mpx602/projects/ETU/ETU/similarity/topsim/last_entity_ans_path_ordered_by_filtered_rel_sim/gt_relation_test.json"

# Display help
function show_help {
    echo "Usage: ./evaluate_gt_relations.sh [options]"
    echo ""
    echo "Evaluate answers based on similarity scores of paths filtered by ground truth relations"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE       Path to JSONL file with paths and similarity scores"
    echo "                         Default: $INPUT_FILE"
    echo ""
    echo "  -g, --ground-truth FILE Ground truth JSONL file path"
    echo "                         Default: $GROUND_TRUTH"
    echo ""
    echo "  -o, --output-dir DIR   Directory to save results"
    echo "                         Default: $OUTPUT_DIR"
    echo ""
    echo "  -k, --top-k NUM        Number of top paths to consider (by similarity score)"
    echo "                         Default: $TOP_K"
    echo ""
    echo "  -r, --gt-relations-file FILE Path to ground truth relations JSON file for filtering"
    echo "                         Default: $GT_RELATIONS_FILE"
    echo ""
    echo "  -h, --help             Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  ./evaluate_gt_relations.sh --top-k 3"
    echo "  ./evaluate_gt_relations.sh -i custom_paths.jsonl -o custom_results -k 10"
    echo "  ./evaluate_gt_relations.sh -r custom_gt_relations.json -k 5"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -g|--ground-truth)
            GROUND_TRUTH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -r|--gt-relations-file)
            GT_RELATIONS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: Ground truth file '$GROUND_TRUTH' not found"
    exit 1
fi

if [ ! -z "$GT_RELATIONS_FILE" ] && [ ! -f "$GT_RELATIONS_FILE" ]; then
    echo "Error: Ground truth relations file '$GT_RELATIONS_FILE' not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Prepare relation arguments if provided
GT_RELATION_ARGS=""
if [ ! -z "$GT_RELATIONS_FILE" ]; then
    GT_RELATION_ARGS="--gt-relations-file $GT_RELATIONS_FILE"
    echo "Using filtering with ground truth relations from $GT_RELATIONS_FILE"
fi

# Run multiple top-k values if requested
if [[ "$TOP_K" == *","* ]]; then
    echo "Running evaluation for multiple top-k values: $TOP_K"
    IFS=',' read -ra K_VALUES <<< "$TOP_K"
    for k in "${K_VALUES[@]}"; do
        echo "Evaluating with top-k = $k..."
        python3 $(dirname "$0")/evaluate_by_last_entity_ans_path_filter_by_gt_rel.py \
            --input "$INPUT_FILE" \
            --ground-truth "$GROUND_TRUTH" \
            --output-dir "$OUTPUT_DIR" \
            --top-k "$k" \
            $GT_RELATION_ARGS
    done
else
    # Run for a single top-k value
    echo "Evaluating with top-k = $TOP_K..."
    python3 $(dirname "$0")/evaluate_by_last_entity_ans_path_filter_by_gt_rel.py \
        --input "$INPUT_FILE" \
        --ground-truth "$GROUND_TRUTH" \
        --output-dir "$OUTPUT_DIR" \
        --top-k "$TOP_K" \
        $GT_RELATION_ARGS
fi

echo "Evaluation complete. Results saved in $OUTPUT_DIR"