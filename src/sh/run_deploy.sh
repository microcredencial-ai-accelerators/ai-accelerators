#!/bin/bash

# Usage: ./run_fc.sh [APP] [KERNEL] [WEIGHTS]

# Default paths (change if needed)
APP=${1:-"opencl/apps/fc_fp32_host"}
KERNEL=${2:-"opencl/kernels/fc_fp32.aocx"}
WEIGHTS=${3:-"opencl/weights/fc_fp32"}

# Fixed data files
IMAGES="opencl/data/test_images_u8.bin"
LABELS="opencl/data/test_labels.bin"

# Run the application
echo "Running $APP with kernel $KERNEL and weights $WEIGHTS..."
$APP $KERNEL $IMAGES $LABELS $WEIGHTS
  opencl/weights/fc_fp32
