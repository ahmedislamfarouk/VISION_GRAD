#!/bin/bash
# Training Script for 8×A6000 GPUs
# HBU-YOLO-VLM Full Model Training

set -e

# Configuration
CONFIG="configs/hbu_yolo_vlm_full.yaml"
OUTPUT_DIR="checkpoints/hbu_yolo_vlm_full"
NUM_GPUS=8

echo "=============================================="
echo "HBU-YOLO-VLM Training on 8×A6000 GPUs"
echo "=============================================="
echo ""
echo "Configuration: $CONFIG"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if GPUs are available
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Starting training..."
echo ""

# Training with torchrun (recommended for multi-GPU)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port="29500" \
    training/train.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    "${@}"

echo ""
echo "Training completed!"
