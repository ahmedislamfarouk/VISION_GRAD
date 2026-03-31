# HBU-YOLO-VLM: Complete Directory Documentation

**Hierarchical Bidirectional Uncertainty-Aware Deep Fusion of Detection and Vision-Language Models for UAV-Based Disaster Assessment**

---

## 📁 Root Directory Structure

```
HBU-YOLO-VLM/
├── __init__.py              # Package initialization
├── README.md                # Project overview and quick start
├── DOCUMENTATION.md         # This file - detailed documentation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── configs/                # Configuration files
├── data/                   # Temporary data storage
├── datasets/               # Dataset loaders and utilities
├── evaluation/             # Inference and evaluation scripts
├── models/                 # Core model architecture
├── scripts/                # Training shell scripts
├── training/               # Training pipeline
├── utils/                  # Utility functions
├── checkpoints/            # Model checkpoints (git-ignored)
├── logs/                   # Training logs (git-ignored)
└── results/                # Evaluation results (git-ignored)
```

---

## 📄 Root Level Files

### `__init__.py`
Package initialization file that makes `HBU-YOLO-VLM` importable as a Python package.

### `README.md`
Comprehensive project documentation including:
- Architecture overview
- Installation instructions
- Quick start guide
- Training and inference commands
- Model zoo table
- Citation information

### `requirements.txt`
Python package dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
ultralytics>=8.0.0
albumentations>=1.3.0
deepspeed>=0.12.0
wandb>=0.16.0
...
```

### `.gitignore`
Git ignore rules for:
- Python cache files (`__pycache__/`, `*.pyc`)
- Checkpoints and logs
- Dataset files (too large)
- IDE configurations
- OS files (`.DS_Store`)

---

## 📁 1. `/models/` - Core Model Architecture

**Purpose:** Contains all neural network module implementations for HBU-YOLO-VLM.

### Directory Structure:
```
models/
├── __init__.py                    # Package exports
├── hbu_yolo_vlm.py               # Main integrated model
├── yolo/                         # YOLO detection components
│   ├── __init__.py
│   └── yolo_backbone.py
├── vlm/                          # Vision-language model components
│   ├── __init__.py
│   └── vlm_module.py
├── fusion/                       # Fusion and refinement modules
│   ├── __init__.py
│   ├── hierarchical_fusion.py
│   └── bidirectional_refinement.py
└── uncertainty/                  # Uncertainty-aware modules
    ├── __init__.py
    └── uncertainty_attention.py
```

---

### `models/hbu_yolo_vlm.py` - Main Integrated Model

**Class:** `HBUYOLOVLM(nn.Module)`

**Purpose:** Integrates all components into a unified architecture.

**Key Components:**
1. **YOLO Backbone** - Multi-scale feature extraction
2. **FPN** - Feature enhancement
3. **Hierarchical Fusion** - FPN→VLM injection
4. **Uncertainty Attention** - Adaptive computation
5. **Bidirectional Refinement** - Iterative improvement

**Main Methods:**
```python
def forward(images, texts, boxes, labels, return_loss=True)
    # Complete forward pass through all modules

def _compute_loss(detections, vlm_states, uncertainty, ground_truth)
    # Multi-task loss: detection + language + uncertainty

def generate(images, prompts, max_length=100, temperature=1.0)
    # Inference with caption generation

def get_architecture_info()
    # Return model statistics and configuration
```

**Forward Pass Flow:**
```
Input Image → YOLO Backbone → FPN → Hierarchical Fusion → 
Uncertainty Estimation → Adaptive Attention → 
Bidirectional Refinement → Outputs
```

---

### `models/yolo/yolo_backbone.py` - YOLO Detection Components

**Classes:**

#### `YOLOBackbone(nn.Module)`
- Loads pre-trained YOLOv8 (n/s/m/l/x variants)
- Extracts features at 5 scales (P2-P6)
- Output channels: P2=256, P3-P6=512

**Feature Scales:**
| Scale | Resolution | Use Case |
|-------|------------|----------|
| P2 | 1/4 | Tiny details (cracks, sparks) |
| P3 | 1/8 | Small objects (person, debris) |
| P4 | 1/16 | Medium objects (cars, buildings) |
| P5 | 1/32 | Large structures |
| P6 | 1/64 | Global context |

#### `FeaturePyramidNetwork(nn.Module)`
- Top-down pathway with lateral connections
- Enhances multi-scale features
- Outputs 256-channel features for fusion

#### `DetectionHead(nn.Module)`
Three output branches:
1. **Classification** - Object classes (80 classes)
2. **Regression** - Bounding box coordinates (4 values)
3. **Uncertainty** - Aleatoric uncertainty estimation

---

### `models/vlm/vlm_module.py` - Vision-Language Model Components

**Classes:**

#### `VLMEncoder(nn.Module)`
- Vision backbone: CLIP ViT-L/14
- Extracts visual features: `(B, 576, 1024)`
- Can be frozen or partially fine-tuned

#### `VLMDecoder(nn.Module)`
- Language model: Llama-7B or Mistral-7B
- Generates text captions
- Supports LoRA fine-tuning

#### `LLaVAModule(nn.Module)`
Main VLM integration:
- Connects vision encoder to language decoder
- Projection types: `linear`, `mlp`, `resampler`
- Tokenizer integration

#### `PerceiverResampler(nn.Module)`
- Compresses visual tokens: 576 → 64
- Uses cross-attention with learnable latents
- Reduces memory for long sequences

---

### `models/fusion/hierarchical_fusion.py` - Contribution #1

**Hierarchical Feature Injection**

#### `HierarchicalFPNVLMFusion(nn.Module)`

**Layer Mapping:**
| FPN Scale | VLM Layers | Semantic Level | Examples |
|-----------|------------|----------------|----------|
| P2 | 1-6 | Tiny details | "crack", "spark" |
| P3 | 7-12 | Small objects | "person", "debris" |
| P4 | 13-20 | Medium | "car", "building section" |
| P5 | 21-26 | Large | "collapsed structure" |
| P6 | 27-32 | Context | "disaster zone" |

**How it Works:**
1. Project FPN features to VLM hidden size (4096)
2. Inject at assigned transformer layers
3. Use cross-attention for feature merging

#### `CrossAttentionFusion(nn.Module)`
- VLM queries attend to FPN features
- Multi-head cross-attention
- Gated residual connection + FFN

#### `AdaptiveLayerNormFusion(nn.Module)`
- FPN features modulate LayerNorm parameters
- `gamma` and `beta` learned from FPN

#### `GatedFusion(nn.Module)`
- Learnable gate controls injection strength
- Formula: `gate * fpn_features + (1-gate) * vlm_features`

---

### `models/fusion/bidirectional_refinement.py` - Contribution #2

**Bidirectional Mutual Refinement**

#### `BidirectionalMutualRefinement(nn.Module)`

**Iterative Refinement Loop (3 iterations):**
```
Iteration 1:
  YOLO → initial detections
  VLM → semantic analysis  
  Feedback → refine YOLO

Iteration 2:
  Refined YOLO → better detections
  VLM → improved understanding
  Feedback → more refinement

Iteration 3:
  Final refined outputs
```

#### `DetectionRefinementModule(nn.Module)`
- Refines boxes using VLM feedback
- Cross-attention from detections to VLM
- Box delta prediction

#### `VLMSemanticRefiner(nn.Module)`
- Refines VLM states using detection features
- Self-attention + cross-attention
- Iteration embedding for tracking

#### `ConfidenceRefinementModule(nn.Module)`
- Predicts confidence score adjustments
- Combines YOLO + VLM features

#### `IterativeRefinementTracker`
- Monitors convergence
- Early stopping if confidence stabilizes

---

### `models/uncertainty/uncertainty_attention.py` - Contribution #3

**Uncertainty-Aware Adaptive Attention**

#### `UncertaintyAwareAdaptiveAttention(nn.Module)`

**Dynamic Routing Based on Uncertainty:**
```
Uncertainty < 0.3  → Fast Path (2 layers)
0.3 ≤ Unc ≤ 0.7    → Medium Path (8 layers)
Uncertainty > 0.7  → Full Path (32 layers)
```

#### `UncertaintyAttentionModulator(nn.Module)`
- Scales attention heads based on uncertainty
- High uncertainty → stronger attention
- Low uncertainty → minimal attention

#### `UncertaintyEstimationHead(nn.Module)`
Two uncertainty types:
- **Aleatoric** - Data uncertainty (irreducible)
- **Epistemic** - Model uncertainty (reducible)
- Combined: `uncertainty = aleatoric + epistemic`

#### Computation Paths:
- **`FastPath`** - 2 layers, high confidence (efficient)
- **`MediumPath`** - 8 layers, medium confidence
- **`FullPath`** - 32 layers, low confidence (accurate)

---

## 📁 2. `/training/` - Training Pipeline

**Purpose:** Training scripts and utilities for multi-GPU training.

### Directory Structure:
```
training/
├── __init__.py           # Package exports
├── train.py              # Main training script (DDP)
├── train_ds.py           # DeepSpeed training script
└── trainer.py            # Trainer class
```

---

### `training/train.py` - Main Training Script

**Class:** `TrainingRunner`

**Features:**
- DDP support for 8-GPU training
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- LoRA fine-tuning support

**Key Methods:**
```python
def _build_optimizer()
    # AdamW with different LR for backbone vs other params

def _build_scheduler()
    # Cosine annealing with warmup

def _load_checkpoint(path)
    # Resume training from checkpoint

def _load_pretrained(path)
    # Load pretrained model

def train()
    # Main training loop
```

**Usage:**
```bash
# Single GPU
python training/train.py --config configs/hbu_yolo_vlm_base.yaml

# Multi-GPU (8×A6000)
torchrun --nproc_per_node=8 training/train.py \
    --config configs/hbu_yolo_vlm_full.yaml
```

---

### `training/trainer.py` - Trainer Class

**Class:** `Trainer`

**Handles:**
- Single epoch training loop
- Validation
- Loss computation
- Metrics calculation
- Checkpoint saving/loading

**Key Methods:**
```python
def train_one_epoch(train_loader, epoch)
    # Training loop with progress bar
    # Returns: metrics dictionary

@torch.no_grad()
def validate(val_loader, epoch)
    # Validation with mAP computation
    # Returns: validation metrics

def save_checkpoint(epoch, metrics, is_best=False)
    # Save model states
    # Keeps last N checkpoints

def load_checkpoint(checkpoint_path)
    # Load from checkpoint
```

**Loss Components:**
- `box_loss` - L1 + IoU loss for boxes
- `cls_loss` - Focal loss for classification
- `lm_loss` - Language modeling loss
- `unc_loss` - Uncertainty calibration loss
- `total_loss` - Weighted sum

---

### `training/train_ds.py` - DeepSpeed Training

**Purpose:** DeepSpeed ZeRO-2 optimization for 8×A6000.

**Features:**
- ZeRO-2 stage (optimizer + gradient sharding)
- BF16 mixed precision
- Activation checkpointing
- Gradient accumulation

**Usage:**
```bash
deepspeed --num_gpus=8 training/train_ds.py \
    --config configs/hbu_yolo_vlm_full.yaml \
    --deepspeed_config configs/deepspeed_config.json
```

---

## 📁 3. `/datasets/` - Data Loading & Augmentation

**Purpose:** Dataset loaders, augmentations, and preparation scripts.

### Directory Structure:
```
datasets/
├── __init__.py              # Package exports
├── disaster_dataset.py      # Dataset classes
├── augmentations.py         # Image augmentations
├── text_templates.py        # Caption templates
└── prepare_data.py          # Dataset preparation
```

---

### `datasets/disaster_dataset.py` - Dataset Loaders

**Base Class:** `DisasterDataset(Dataset)`

**Features:**
- Loads images and annotations
- Applies augmentations
- Generates captions from templates
- Multi-class support

**Disaster Classes (10):**
```python
['building', 'vehicle', 'person', 'debris', 'vegetation',
 'flood', 'fire', 'collapsed_structure', 'road', 'other']
```

#### `XBDDataset(DisasterDataset)`
- **Purpose:** Building damage assessment
- **Source:** xView2 xBD dataset
- **Annotations:** Polygon → bounding box conversion
- **Labels:** Damage levels 0-4 (none to destroy)

#### `RescueNetDataset(DisasterDataset)`
- **Purpose:** Disaster scene understanding
- **Source:** RescueNet dataset
- **Format:** YOLO-style annotations
- **Classes:** Multiple disaster types

#### `FloodNetDataset(DisasterDataset)`
- **Purpose:** Flood damage assessment
- **Source:** FloodNet dataset
- **Annotations:** JSON format
- **Classes:** Flood-specific categories

#### `CombinedDataset(DisasterDataset)`
- **Purpose:** Merge all three datasets
- **Features:** Cumulative indexing, balanced sampling
- **Usage:** Multi-source training

**DataLoader Functions:**
```python
def build_dataset(config, data_dir, split, augmentations)
    # Factory function to create dataset

def build_dataloader(dataset, batch_size, num_workers, distributed)
    # Create DataLoader with DistributedSampler

def collate_fn(batch)
    # Custom collate for variable-size annotations
```

---

### `datasets/augmentations.py` - Image Augmentations

**Library:** albumentations

**Augmentation Pipeline:**
```python
def build_augmentations(config):
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        A.RandomCrop(480, 480),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
```

**Configurable Parameters:**
- `hflip` - Horizontal flip probability
- `vflip` - Vertical flip probability
- `rotation` - Max rotation angle
- `scale` - Scale limit
- `color_jitter` - Color jitter strength
- `random_crop` - Enable random cropping

---

### `datasets/text_templates.py` - Caption Templates

**Class:** `DisasterTemplates`

**Template Categories:**
1. **General** - Basic object descriptions
2. **Building Damage** - Damage level assessments
3. **Flood** - Flood-specific descriptions
4. **Earthquake** - Seismic impact descriptions
5. **Fire** - Fire damage descriptions

**Example Templates:**
```python
{
    'general': [
        "This disaster scene shows {objects}.",
        "The aerial view reveals {objects} in the affected area."
    ],
    'building_damage': [
        "Building damage level: {damage_level}. {objects} visible."
    ]
}
```

**Usage:**
```python
templates = get_disaster_templates()
caption = templates.format_counts({'building': 5, 'vehicle': 3})
# Output: "This disaster scene shows 5 buildings and 3 vehicles."
```

---

### `datasets/prepare_data.py` - Dataset Preparation

**Functions:**

#### `prepare_xbd_dataset(output_dir)`
- Downloads xBD dataset instructions
- Expected structure:
  ```
  datasets/xBD/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
  ```

#### `prepare_rescuenet_dataset(output_dir)`
- RescueNet setup instructions
- GitHub: https://github.com/remi-md/rescuenet

#### `prepare_floodnet_dataset(output_dir)`
- FloodNet setup instructions
- GitHub: https://github.com/Ankush191/FloodNet

#### `prepare_combined_dataset(...)`
- Merges all datasets into unified structure
- Prefixes filenames with dataset name

**Usage:**
```bash
python datasets/prepare_data.py --dataset all --output-dir datasets/
```

---

## 📁 4. `/evaluation/` - Inference & Evaluation

**Purpose:** Inference scripts and evaluation metrics.

### Directory Structure:
```
evaluation/
├── __init__.py           # Package exports
├── inference.py          # Inference script
└── evaluate.py           # Evaluation script
```

---

### `evaluation/inference.py` - Inference Script

**Class:** `HBUYOLOVLMPredictor`

**Features:**
- Single image inference
- Batch inference
- Visualization
- Confidence thresholding
- NMS (Non-Maximum Suppression)

**Usage:**
```python
predictor = HBUYOLOVLMPredictor(
    checkpoint_path="checkpoints/best.pth",
    config_path="configs/hbu_yolo_vlm_base.yaml",
    confidence_threshold=0.5
)

# Single image
predictions = predictor.predict(image, prompt="Describe this scene")

# Batch
predictions = predictor.predict_batch(images, prompts)

# Visualize
vis_image = predictor.visualize(image, predictions)
```

**Output Format:**
```python
{
    'boxes': np.array([[x1, y1, x2, y2], ...]),
    'scores': np.array([0.95, 0.87, ...]),
    'labels': np.array([0, 2, 1, ...]),
    'caption': "This disaster scene shows 5 buildings...",
    'uncertainty': {...}
}
```

**Command Line:**
```bash
python evaluation/inference.py \
    --image data/sample.jpg \
    --checkpoint checkpoints/best.pth \
    --config configs/hbu_yolo_vlm_base.yaml \
    --visualize \
    --output-dir results/predictions
```

---

### `evaluation/evaluate.py` - Evaluation Script

**Class:** `Evaluator`

**Metrics Computed:**

**Detection Metrics:**
- `mAP_50` - mAP at IoU=0.50
- `mAP_75` - mAP at IoU=0.75
- `mAP` - Mean mAP (average of 50 and 75)
- `precision` - Detection precision
- `recall` - Detection recall
- `f1` - F1 score

**Language Metrics:**
- `BLEU_4` - BLEU score (4-gram)
- `ROUGE_L` - ROUGE-L score
- `CIDEr` - CIDEr score

**Loss Metrics:**
- `loss` - Total loss
- `box_loss` - Box regression loss
- `cls_loss` - Classification loss
- `lm_loss` - Language modeling loss

**Usage:**
```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --config configs/hbu_yolo_vlm_base.yaml \
    --data-dir datasets/xBD/test \
    --split test \
    --output-dir results/evaluation
```

---

## 📁 5. `/configs/` - Configuration Files

**Purpose:** YAML and JSON configuration files for training.

### Files:

#### `configs/hbu_yolo_vlm_base.yaml`
Base configuration (~180 lines):

```yaml
# Model Architecture
model:
  num_classes: 5
  num_anchors: 3
  input_size: [512, 512]

# YOLO Backbone
yolo:
  model_size: "yolov8m.pt"
  pretrained: true
  freeze_backbone: false

# VLM Configuration
vlm:
  vision_model: "openai/clip-vit-large-patch14"
  llm_model: "lmsys/vicuna-7b-v1.5"
  tokenizer: "lmsys/vicuna-7b-v1.5"
  projection_type: "mlp"
  lora_enabled: true
  lora_rank: 64

# Fusion Configuration
fusion:
  vlm_hidden_size: 4096
  vlm_num_layers: 32
  num_heads: 8
  fusion_type: "cross_attention"

# Uncertainty Configuration
uncertainty:
  threshold_low: 0.3
  threshold_high: 0.7

# Refinement Configuration
refinement:
  num_iterations: 3
  early_stopping: true

# Loss Weights
loss:
  box_weight: 5.0
  cls_weight: 1.0
  lm_weight: 1.0
  unc_weight: 0.5

# Training Configuration
training:
  batch_size: 8
  num_epochs: 100
  warmup_epochs: 5
  base_lr: 0.0001
  optimizer: "adamw"
  lr_scheduler: "cosine"
  mixed_precision: true

# Validation
validation:
  val_interval: 1
  metric: "map"

# Checkpoint
checkpoint:
  save_interval: 5
  keep_last_n: 3

# Logging
logging:
  project_name: "HBU-YOLO-VLM"
  tensorboard: true
  wandb: true
```

---

#### `configs/hbu_yolo_vlm_full.yaml`
Full model configuration (8×A6000 optimized):

**Key Differences from Base:**
```yaml
model:
  num_classes: 10
  input_size: [640, 640]

yolo:
  model_size: "yolov8l.pt"  # Large backbone

vlm:
  llm_model: "mistralai/Mistral-7B-Instruct-v0.1"
  projection_type: "resampler"
  lora_rank: 128

fusion:
  num_heads: 16

training:
  batch_size: 8  # Per GPU → Total = 64
  num_epochs: 150
  base_lr: 0.0002  # Higher LR for larger batch
  amp_dtype: "bfloat16"  # Better for A6000

distributed:
  deepspeed:
    enabled: true
    stage: 2  # ZeRO-2
```

---

#### `configs/deepspeed_config.json`
DeepSpeed ZeRO-2 configuration:

```json
{
  "bf16": {"enabled": true},
  
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 8,
  
  "gradient_clipping": 1.0,
  
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

---

## 📁 6. `/scripts/` - Training Shell Scripts

**Purpose:** Convenience scripts for training.

### Files:

#### `scripts/train_quick.sh`
Single GPU quick test:
```bash
#!/bin/bash
python training/train.py \
    --config configs/hbu_yolo_vlm_base.yaml \
    --output_dir checkpoints/hbu_yolo_vlm_quick \
    training.batch_size=4 \
    training.num_epochs=10
```

**Use Case:** Debugging and testing

---

#### `scripts/train_8gpu.sh`
8-GPU DDP training:
```bash
#!/bin/bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    training/train.py \
    --config configs/hbu_yolo_vlm_full.yaml \
    --output_dir checkpoints/hbu_yolo_vlm_full
```

**Use Case:** Full model training on 8×A6000

---

#### `scripts/train_deepspeed.sh`
DeepSpeed ZeRO-2 training:
```bash
#!/bin/bash
deepspeed \
    --num_gpus=8 \
    training/train_ds.py \
    --config configs/hbu_yolo_vlm_full.yaml \
    --deepspeed_config configs/deepspeed_config.json
```

**Use Case:** Memory-efficient training (recommended)

---

## 📁 7. `/utils/` - Utility Functions

**Purpose:** Helper functions and utilities.

### Directory Structure:
```
utils/
├── __init__.py           # Package exports
├── distributed.py        # Distributed training utilities
├── metrics.py            # Metrics computation
└── logger.py             # Logging utilities
```

---

### `utils/distributed.py` - Distributed Utilities

**Functions:**
```python
def init_distributed() -> bool
    # Initialize DDP with NCCL backend

def cleanup_distributed()
    # Destroy process group

def get_rank() -> int
    # Get current process rank

def get_world_size() -> int
    # Get number of processes

def reduce_tensor(tensor, avg=True) -> torch.Tensor
    # All-reduce tensor across GPUs

def synchronize()
    # Barrier synchronization

def set_seed(seed)
    # Set random seed for reproducibility

def to_cuda(batch, device) -> Dict
    # Move batch to CUDA
```

---

### `utils/metrics.py` - Metrics Computation

**Detection Metrics:**
```python
def compute_iou(boxes1, boxes2) -> torch.Tensor
    # Compute IoU matrix

def compute_ap(scores, labels, gt_labels) -> float
    # Average Precision

def compute_map(detections, ground_truth, iou_thresholds) -> Dict
    # mean Average Precision at multiple IoU thresholds

def compute_detection_metrics(detections, ground_truth) -> Dict
    # Precision, Recall, F1 score
```

**Language Metrics:**
```python
def compute_language_metrics(predictions, ground_truth) -> Dict
    # BLEU, ROUGE, CIDEr using COCO evaluation
```

---

### `utils/logger.py` - Logging Utilities

**Functions:**

#### `setup_logger(name, output_dir, rank)`
- Console logging
- File logging (main process only)
- Formatted with timestamp

#### `WandBLogger`
```python
class WandBLogger:
    def __init__(config)
        # Initialize wandb run
    
    def log(data, step)
        # Log dictionary of metrics
    
    def log_train_step(step, metrics)
        # Log training metrics
    
    def log_validation(epoch, metrics)
        # Log validation metrics
    
    def log_image(image, detections, index)
        # Log image with bounding boxes
    
    def finish()
        # Close wandb run
```

#### `TensorBoardLogger`
```python
class TensorBoardLogger:
    def __init__(log_dir)
        # Create SummaryWriter
    
    def log_scalar(name, value, step)
        # Log single scalar
    
    def log_scalars(scalars, step)
        # Log multiple scalars
    
    def log_image(name, image, step)
        # Log image
    
    def close()
        # Close writer
```

---

## 📁 8. Empty Directories (with `.gitkeep`)

### `/checkpoints/`
**Purpose:** Store model checkpoints

**Files Created During Training:**
- `checkpoint_last.pth` - Latest checkpoint
- `checkpoint_best.pth` - Best mAP checkpoint
- `checkpoint_epoch_X.pth` - Per-epoch checkpoints

**Checkpoint Contents:**
```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'scheduler_state_dict': state_dict,
    'scaler_state_dict': state_dict,
    'metrics': dict,
    'best_metric': float,
    'config': OmegaConf
}
```

---

### `/logs/`
**Purpose:** Training logs

**Contents:**
- `log_YYYYMMDD_HHMMSS.txt` - Training log file
- `tensorboard/` - TensorBoard events
- `wandb/` - WandB run files

---

### `/results/`
**Purpose:** Evaluation results

**Contents:**
- `predictions/` - Inference results (JSON)
- `evaluation/` - Evaluation metrics (JSON)
- `visualizations/` - Annotated images

---

### `/data/`
**Purpose:** Temporary data storage

**Contents:**
- Preprocessed datasets
- Cache files
- Temporary downloads

---

### `/datasets/`
**Purpose:** Raw dataset files (git-ignored)

**Expected Structure:**
```
datasets/
├── xBD/
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   └── val/labels/
├── RescueNet/
│   ├── train/images/
│   └── ...
├── FloodNet/
│   ├── train/images/
│   └── ...
└── combined/
    ├── train/images/
    └── ...
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image                               │
│                  (B, 3, 512, 512)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  YOLO Backbone                                               │
│  └─→ Extracts P2, P3, P4, P5, P6 features                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Feature Pyramid Network                                     │
│  └─→ Enhances multi-scale features                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Hierarchical FPN-VLM Fusion (Contribution #1)              │
│  P2 → VLM Layers 1-6   (tiny details)                       │
│  P3 → VLM Layers 7-12  (small objects)                      │
│  P4 → VLM Layers 13-20 (medium)                             │
│  P5 → VLM Layers 21-26 (large)                              │
│  P6 → VLM Layers 27-32 (context)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Uncertainty Estimation                                      │
│  └─→ Aleatoric + Epistemic uncertainty                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Uncertainty-Aware Adaptive Attention (Contribution #2)     │
│  Low unc  → Fast Path (2 layers)   - efficient              │
│  Med unc  → Medium Path (8 layers)                          │
│  High unc → Full Path (32 layers)  - accurate               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Bidirectional Mutual Refinement (Contribution #3)          │
│  Iteration 1: YOLO → VLM → YOLO (refined)                   │
│  Iteration 2: Refined → VLM → Refined                       │
│  Iteration 3: Final refinement                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Outputs                                                     │
│  - Detections (boxes, scores, labels)                       │
│  - Captions (generated text)                                │
│  - Uncertainty estimates                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Installation
```bash
cd /home/skyvision/VISION_GRAD/HBU-YOLO-VLM

# Create conda environment
conda create -n hbu-yolo-vlm python=3.10 -y
conda activate hbu-yolo-vlm

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Quick test (single GPU)
bash scripts/train_quick.sh

# Full training (8×A6000)
bash scripts/train_8gpu.sh

# DeepSpeed training (recommended)
bash scripts/train_deepspeed.sh
```

### Inference
```bash
python evaluation/inference.py \
    --image data/sample.jpg \
    --checkpoint checkpoints/best.pth \
    --config configs/hbu_yolo_vlm_base.yaml \
    --visualize
```

### Evaluation
```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --config configs/hbu_yolo_vlm_base.yaml \
    --data-dir datasets/xBD/test
```

---

## 📝 File Summary Table

| Directory | Files | Purpose |
|-----------|-------|---------|
| `/models/` | 10 | Neural network architectures |
| `/training/` | 4 | Training pipeline |
| `/datasets/` | 5 | Data loading and augmentation |
| `/evaluation/` | 3 | Inference and evaluation |
| `/configs/` | 3 | Configuration files |
| `/scripts/` | 3 | Training shell scripts |
| `/utils/` | 4 | Utility functions |
| **Total** | **32** | **~5,000+ lines of code** |

---

## 📚 Additional Resources

- **Paper Draft:** See `/papers/` directory (to be added)
- **Checkpoints:** See `/checkpoints/` directory
- **Logs:** See `/logs/` directory
- **Results:** See `/results/` directory

---

**For questions or issues, please refer to the main README.md or contact the development team.**
