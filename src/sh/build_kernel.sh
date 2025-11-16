#!/bin/bash

# Usage: ./build_kernel.sh [KERNEL_SRC] [OUTPUT_AOCX] [BOARD]

# Default values
KERNEL_SRC=${1:-"src/opencl/kernels/fc_fp32.cl"}
OUTPUT_AOCX=${2:-"output/fc_fp32.aocx"}
BOARD=${3:-"$BOARD"}  # Can also export BOARD in environment

# Compile kernel
echo "Compiling $KERNEL_SRC to $OUTPUT_AOCX for board $BOARD..."
aoc -v -g -report -board=$BOARD "$KERNEL_SRC" -o "$OUTPUT_AOCX"

echo "Done."