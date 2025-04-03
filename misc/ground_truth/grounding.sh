#!/bin/bash

# Default values
PREDICTIONS_FILE="/data/home/mpx602/projects/ETU/ETU/clean_trajectories.jsonl"
OUTPUT_FILE="grounded_paths.json"
SAMPLE_SIZE=100
API_KEY="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"

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --predictions FILE   Path to the predictions JSONL file (required)"
    echo "  -o, --output FILE        Output file path (default: ranked_paths.json)"
    echo "  -s, --sample SIZE        Number of samples to process (default: all)"
    echo "  -k, --api-key KEY        Anthropic API key (if not set as environment variable)"
    echo "  -h, --help               Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--predictions)
            PREDICTIONS_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
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

# Check if required parameters are provided
if [ -z "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file is required"
    show_help
    exit 1
fi

# Check if file exists
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file '$PREDICTIONS_FILE' does not exist"
    exit 1
fi

# Set API key if provided
if [ ! -z "$API_KEY" ]; then
    export ANTHROPIC_API_KEY="$API_KEY"
fi

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY environment variable is not set"
    echo "Please set it or provide it with --api-key"
    exit 1
fi
# Set default model if not provided
if [ -z "$MODEL" ]; then
    MODEL="claude-3-7-sonnet-20250219"
fi

echo "Processing paths with the following settings:"
echo "Predictions file: $PREDICTIONS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Sample size: $SAMPLE_SIZE"
echo "Model: $MODEL"

# Run the Python script
python grounding.py \
    --predictions "$PREDICTIONS_FILE" \
    --output "$OUTPUT_FILE" \
    --api_key "$ANTHROPIC_API_KEY" \
    --limit "$SAMPLE_SIZE" \
    --model "$MODEL"    

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "Completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Error: Script execution failed"
fi