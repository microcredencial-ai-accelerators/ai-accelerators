#!/bin/bash

# Usage: ./build_fc.sh [SOURCE_CPP] [OUTPUT_APP]

# Default source and output (change if needed)
SRC=${1:-"opencl/host/main_fc_fp32.cpp"}
OUT=${2:-"opencl/apps/fc_fp32_host"}

# Compile
echo "Compiling $SRC to $OUT..."
g++ -O2 -std=c++11 "$SRC" -o "$OUT" \
    $(aocl compile-config) \
    $(aocl link-config)

echo "Done."