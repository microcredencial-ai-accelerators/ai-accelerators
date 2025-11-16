#!/bin/bash

# Usage: ./run_deploy.sh [APP] [KERNEL] [WEIGHTS_DIR]

APP=${1:-"opencl/apps/fc_fp32_host"}
KERNEL=${2:-"opencl/kernels/fc_fp32.aocx"}
WEIGHTS_DIR=${3:-"opencl/weights/fc_fp32"}

IMAGES="opencl/data/test_images_u8.bin"
LABELS="opencl/data/test_labels.bin"

# Check if files exist
if [ ! -f "$APP" ]; then
    echo "Error: application '$APP' not found!"
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "Error: kernel '$KERNEL' not found!"
    exit 1
fi

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Error: weights directory '$WEIGHTS_DIR' not found!"
    exit 1
fi

# Display options
echo "Running $APP"
echo "Kernel: $KERNEL"
echo "Weights directory: $WEIGHTS_DIR"

# Run App
$APP $KERNEL $IMAGES $LABELS "$WEIGHTS_DIR"