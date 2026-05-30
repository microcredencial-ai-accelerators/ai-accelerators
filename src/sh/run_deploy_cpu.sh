#!/bin/bash

# Usage: ./run_deploy_cpu.sh [APP] [WEIGHTS_DIR]

APP=${1:-"opencl/apps/fc_fp32_cpu"}
WEIGHTS_DIR=${2:-"opencl/weights/fc_fp32"}

IMAGES="opencl/data/test_images_u8.bin"
LABELS="opencl/data/test_labels.bin"

# Check if files exist
if [ ! -f "$APP" ]; then
    echo "Error: application '$APP' not found!"
    exit 1
fi

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Error: weights directory '$WEIGHTS_DIR' not found!"
    exit 1
fi

# Display options
echo "Running $APP"
echo "Weights directory: $WEIGHTS_DIR"

# Run App
$APP $IMAGES $LABELS "$WEIGHTS_DIR"