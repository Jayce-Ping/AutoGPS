#!/bin/bash

# 显示使用方法
usage() {
    echo "Usage: $0 [-d|--details] <results_dir>"
    echo "Options:"
    echo "  -d, --details    Show detailed evaluation results"
    echo "Example: bash eval_results.sh -d symbolic_reasoner/our_results"
    exit 1
}

# Default no details
SHOW_DETAILS=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--details) SHOW_DETAILS="--show_details"; shift ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) RESULTS_DIR="$1"; shift ;;
    esac
done

# Check if results directory is provided
if [ -z "$RESULTS_DIR" ]; then
    usage
fi

# Check if the provided directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: No such directory: $RESULTS_DIR"
    exit 1
fi

# Evaluation
for dir in "$RESULTS_DIR"/*/; do
    if [ -d "$dir" ]; then
        echo "------------- Evaluating $dir -------------"
        python symbolic_reasoner/evaluate.py --proof_dir "$dir" $SHOW_DETAILS
        echo ""
    fi
done