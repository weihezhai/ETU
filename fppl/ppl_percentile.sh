#!/bin/bash

# Script to run ppl_percentile.py on all JSONL files in a directory

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
PERCENTILE=15

# Function to display usage
show_usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR [-p PERCENTILE]"
    echo "  -i INPUT_DIR   : Directory containing JSONL files to process"
    echo "  -o OUTPUT_DIR  : Directory where filtered files will be saved"
    echo "  -p PERCENTILE  : Percentile to filter (default: 15)"
    echo "  -h             : Display this help message"
}

# Parse command line arguments
while getopts "i:o:p:h" opt; do
    case ${opt} in
        i )
            INPUT_DIR=$OPTARG
            ;;
        o )
            OUTPUT_DIR=$OPTARG
            ;;
        p )
            PERCENTILE=$OPTARG
            ;;
        h )
            show_usage
            exit 0
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            show_usage
            exit 1
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            show_usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Input and output directories are required."
    show_usage
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all JSONL files in the input directory and process them
echo "Processing JSONL files with percentile: $PERCENTILE..."
echo "---------------------------------------------------"

# Counter for processed files
total_files=0
processed_files=0

# Process all JSONL files recursively
find "$INPUT_DIR" -type f -name "*.jsonl" | while read -r file; do
    # Get relative path from INPUT_DIR
    rel_path="${file#$INPUT_DIR}"
    # Remove leading slash if present
    rel_path="${rel_path#/}"
    
    # Create output directory structure
    output_file="$OUTPUT_DIR/$rel_path"
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"
    
    echo "Processing: $file"
    echo "Output to: $output_file"
    
    # Run the Python script
    python fppl/ppl_percentile.py "$file" "$output_file" --percentile "$PERCENTILE"
    
    echo "---------------------------------------------------"
    
    ((total_files++))
    if [ $? -eq 0 ]; then
        ((processed_files++))
    fi
done

echo "Completed processing $processed_files out of $total_files files."
echo "Filtered files are saved in $OUTPUT_DIR"