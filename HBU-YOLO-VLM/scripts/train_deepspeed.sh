#!/bin/bash
# Training Script with DeepSpeed for 8×A6000 GPUs
# HBU-YOLO-VLM Full Model Training with ZeRO Optimization

set -e

# Configuration
CONFIG="configs/hbu_yolo_vlm_full.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_config.json"
OUTPUT_DIR="checkpoints/hbu_yolo_vlm_deepspeed"
NUM_GPUS=8

echo "=============================================="
echo "HBU-YOLO-VLM DeepSpeed Training on 8×A6000"
echo "=============================================="
echo ""
echo "Configuration: $CONFIG"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if GPUs are available
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Starting DeepSpeed training..."
echo ""

# Training with DeepSpeed
deepspeed \
    --num_gpus=$NUM_GPUS \
    training/train_ds.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    "${@}"

echo ""
echo "Training completed!"
