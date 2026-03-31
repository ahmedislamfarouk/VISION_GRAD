# HBU-YOLO-VLM

**Hierarchical Bidirectional Uncertainty-Aware Deep Fusion of Detection and Vision-Language Models for UAV-Based Disaster Assessment**

## 🎯 Overview

HBU-YOLO-VLM is a novel deep fusion architecture that fundamentally reimagines how object detection and vision-language models interact for UAV-based disaster assessment.

### Key Contributions

1. **Hierarchical Feature Injection**: Maps YOLO's multi-scale FPN outputs to semantically-aligned VLM transformer layers
2. **Uncertainty-Aware Adaptive Attention**: Dynamically allocates VLM resources based on detection confidence
3. **Bidirectional Mutual Refinement**: Enables iterative co-improvement between detector and VLM

## ️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HBU-YOLO-VLM Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image → YOLO Backbone → FPN Features (P2-P6)             │
│                              ↓                                    │
│              ┌───────────────────────────────────┐               │
│              │   Hierarchical FPN-VLM Fusion     │               │
│              │  P2 → Layers 1-6   (tiny details) │               │
│              │  P3 → Layers 7-12  (small objects)│               │
│              │  P4 → Layers 13-20 (medium)       │               │
│              │  P5 → Layers 21-26 (large)        │               │
│              │  P6 → Layers 27-32 (context)      │               │
│              └───────────────────────────────────┘               │
│                              ↓                                    │
│              ┌───────────────────────────────────┐               │
│              │  Uncertainty-Aware Attention      │               │
│              │  High conf → minimal VLM compute  │               │
│              │  Low conf  → heavy VLM attention  │               │
│              └───────────────────────────────────┘               │
│                              ↓                                    │
│              ┌───────────────────────────────────┐               │
│              │  Bidirectional Mutual Refinement  │               │
│              │  YOLO → detections → VLM          │               │
│              │  VLM → feedback → YOLO (refined)  │               │
│              └───────────────────────────────────┘               │
│                              ↓                                    │
│  Output: Detections + Captions + Uncertainty Estimates           │
│                                                                  │
─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
HBU-YOLO-VLM/
├── configs/              # Configuration files
├── data/                 # Processed data
── datasets/             # Dataset loaders (xBD, RescueNet, FloodNet)
├── models/
│   ├── yolo/             # YOLO detection modules
│   ├── vlm/              # VLM modules (LLaVA-based)
│   ├── fusion/           # Hierarchical fusion modules
│   └── uncertainty/      # Uncertainty estimation
├── training/             # Training scripts
├── evaluation/           # Evaluation scripts
├── utils/                # Utility functions
├── simulation/           # Gazebo/AirSim integration
├── scripts/              # Utility scripts
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── results/              # Evaluation results
└── papers/               # Paper drafts
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /home/skyvision/VISION_GRAD/HBU-YOLO-VLM

# Create conda environment
conda create -n hbu-yolo-vlm python=3.10 -y
conda activate hbu-yolo-vlm

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install LLaVA
pip install git+https://github.com/haotian-liu/LLaVA.git
```

### Training

```bash
# Single GPU training
python training/train.py --config configs/hbu_yolo_vlm_base.yaml

# Multi-GPU training (8×A6000)
torchrun --nproc_per_node=8 training/train.py \
    --config configs/hbu_yolo_vlm_full.yaml \
    --output_dir checkpoints/hbu_yolo_vlm_full

# Distributed training with DeepSpeed
deepspeed --num_gpus=8 training/train_ds.py \
    --config configs/hbu_yolo_vlm_deepspeed.yaml
```

### Inference

```bash
# Single image inference
python evaluation/inference.py \
    --image data/sample_disaster.jpg \
    --checkpoint checkpoints/hbu_yolo_vlm_best.pth \
    --output results/predictions

# Batch inference
python evaluation/inference.py \
    --data_dir data/test_images \
    --checkpoint checkpoints/hbu_yolo_vlm_best.pth \
    --batch_size 32
```

### Evaluation

```bash
# Evaluate on xBD test set
python evaluation/evaluate.py \
    --dataset xbd \
    --data_dir datasets/xBD/test \
    --checkpoint checkpoints/hbu_yolo_vlm_best.pth

# Evaluate on RescueNet
python evaluation/evaluate.py \
    --dataset rescuenet \
    --data_dir datasets/RescueNet/test \
    --checkpoint checkpoints/hbu_yolo_vlm_best.pth

# Evaluate on FloodNet
python evaluation/evaluate.py \
    --dataset floodnet \
    --data_dir datasets/FloodNet/test \
    --checkpoint checkpoints/hbu_yolo_vlm_best.pth
```

## 📊 Datasets

### Supported Datasets

1. **xBD** - Building damage assessment
2. **RescueNet** - Disaster scene understanding
3. **FloodNet** - Flood damage assessment

### Dataset Preparation

```bash
# Download and prepare xBD
python datasets/prepare_xbd.py --data_dir datasets/xBD

# Download and prepare RescueNet
python datasets/prepare_rescuenet.py --data_dir datasets/RescueNet

# Download and prepare FloodNet
python datasets/prepare_floodnet.py --data_dir datasets/FloodNet
```

## 🎓 Model Zoo

| Model | Backbone | VLM | mAP | BLEU-4 | ROUGE-L | Checkpoint |
|-------|----------|-----|-----|--------|---------|------------|
| HBU-YOLO-VLM-T | YOLOv8-Tiny | LLaVA-1.5-7B | - | - | - | [Coming Soon] |
| HBU-YOLO-VLM-S | YOLOv8-Small | LLaVA-1.5-7B | - | - | - | [Coming Soon] |
| HBU-YOLO-VLM-M | YOLOv8-Medium | LLaVA-1.6-Mistral | - | - | - | [Coming Soon] |
| HBU-YOLO-VLM-L | YOLOv8-Large | LLaVA-1.6-Mistral | - | - | - | [Coming Soon] |

## 🔧 Configuration

See `configs/` directory for all configuration options:

- `hbu_yolo_vlm_base.yaml` - Base configuration
- `hbu_yolo_vlm_full.yaml` - Full model with all components
- `hbu_yolo_vlm_deepspeed.yaml` - DeepSpeed optimization for 8-GPU training

## 📈 Training on 8×A6000

Our implementation is optimized for multi-GPU training:

```bash
# With 8×A6000 (48GB each), you can train:
# - LLaVA-1.5-7B + YOLOv8-Large with batch_size=64
# - LLaVA-1.6-Mistral-7B + YOLOv8-Medium with batch_size=48

# Recommended: DeepSpeed ZeRO-2
deepspeed --num_gpus=8 training/train_ds.py \
    --config configs/hbu_yolo_vlm_deepspeed.yaml \
    --deepspeed
```

## 🧪 Simulation Integration

Generate synthetic disaster data using Gazebo or AirSim:

```bash
# Gazebo data generation
python simulation/gazebo/generate_data.py \
    --world disaster_scene \
    --num_images 10000 \
    --output data/synthetic_gazebo

# AirSim data generation
python simulation/airsim/generate_data.py \
    --environment disaster \
    --num_images 10000 \
    --output data/synthetic_airsim
```

## 📄 Citation

```bibtex
@article{hbu2026yolovlm,
  title={HBU-YOLO-VLM: Hierarchical Bidirectional Uncertainty-Aware Deep Fusion of Detection and Vision-Language Models for UAV-Based Disaster Assessment},
  author={Your Team},
  journal={arXiv preprint},
  year={2026}
}
```

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) for the VLM base
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection
- [xBD Dataset](https://xview2.org/dataset) for disaster imagery
- [RescueNet](https://github.com/remi-md/rescuenet) for disaster scene understanding
- [FloodNet](https://github.com/Ankush191/FloodNet) for flood damage assessment
