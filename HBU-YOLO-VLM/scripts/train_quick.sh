#!/bin/bash
# Quick Start Training Script (Single GPU)
# For testing and debugging

set -e

# Configuration
CONFIG="configs/hbu_yolo_vlm_base.yaml"
OUTPUT_DIR="checkpoints/hbu_yolo_vlm_quick"
NUM_GPUS=1

echo "=============================================="
echo "HBU-YOLO-VLM Quick Training (Single GPU)"
echo "=============================================="
echo ""
echo "Configuration: $CONFIG"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Training on single GPU
python training/train.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    training.batch_size=4 \
    training.num_epochs=10 \
    data.num_workers=4 \
    "${@}"

echo ""
echo "Quick training completed!"
