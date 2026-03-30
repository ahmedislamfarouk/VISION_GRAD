# VISION_GRAD 🚁👁️

**Vision-Enhanced Multi-Agent UAV Systems for Federated Learning**

This repository explores computer vision integration with the HFL-UAV-Swarm project, focusing on visual perception, optimization, and intelligence for drone swarm coordination.

---

## 🎯 Research Directions

### 1. **Visual Task Offloading Optimization**
Currently, tasks are defined by CPU cycles, data size, and deadlines. Vision can transform this:

| Current | Vision-Enhanced |
|---------|-----------------|
| Abstract task metrics | Actual image/video data from onboard cameras |
| Random task generation | Scene-based task complexity estimation |
| Fixed data sizes | Adaptive compression based on visual content |

**Research Questions:**
- How does image complexity affect optimal offloading decisions?
- Can we predict computation requirements from visual features?
- Semantic compression: offload features vs. raw pixels?

---

### 2. **Vision-Based Swarm Coordination**
Replace or augment state observations with visual inputs:

```
Current Observation Space:
├── Self-state (position, battery, CPU)
├── Task queue features
└── Neighbor relative positions

Vision-Enhanced Observation Space:
├── Onboard camera feed (semantic segmentation)
├── Visual detection of other drones
├── Ground device localization from aerial view
├── Obstacle/no-fly zone detection
└── Weather/visibility estimation
```

**Potential Models:**
- **ViT-MARL**: Vision Transformer backbone for multi-agent coordination
- **Visual Attention Networks**: Attend to relevant scene regions for task routing
- **3D Scene Understanding**: Depth estimation for collision avoidance

---

### 3. **Federated Visual Learning**
Train vision models across the drone swarm without centralizing data:

**Privacy-Preserving Scenarios:**
- Surveillance drones: can't share raw footage
- Disaster response: bandwidth-limited communication
- Agricultural monitoring: proprietary farm data

**Architectures to Explore:**
- Split Learning: early CNN layers on drones, later layers aggregated
- FedAvg with CNN backbones
- Attention-based aggregation (from Attn-FedHAdam) for visual features

---

### 4. **Visual Anomaly Detection for Task Generation**
Replace synthetic task generation with vision-based triggers:

| Zone Type | Visual Trigger | Task Characteristics |
|-----------|---------------|---------------------|
| Emergency (Zone 0) | Fire/accident detection | Latency-critical, high priority |
| Industrial (Zone 1) | Equipment monitoring | Compute-heavy analysis |
| Residential (Zone 2) | Routine surveillance | Best-effort, low priority |

**Implementation Path:**
1. Object detection (YOLOv8/RT-DETR) on drone
2. Scene classification → QoS template selection
3. Visual complexity → CPU cycle estimation

---

### 5. **Communication-Aware Visual Compression**
The current U2U channel model can be extended for visual data:

```python
# Current: fixed data_size in MB
task = [cpu_cycles, data_size, deadline, priority]

# Vision-enhanced: adaptive compression
visual_task = {
    'raw_image': (H, W, 3),
    'compressed': encode(image, target_quality),
    'features': backbone.extract(image),  # Much smaller!
    'semantic_map': segment(image),
}
```

**Research Directions:**
- Neural compression (learned codecs) for drone-to-drone transmission
- Feature-level offloading vs. pixel-level
- Quality-latency tradeoffs under deadline constraints

---

### 6. **Multi-View Collaborative Perception**
Multiple drones observing the same scene from different angles:

```
Drone A ──────┐
              │   ┌─────────────────┐
Drone B ──────┼──►│ 3D Reconstruction│──► Better Scene Understanding
              │   │   or BEV Fusion  │
Drone C ──────┘   └─────────────────┘
```

**Applications:**
- Collaborative SLAM for mapping
- Multi-view object detection (occluded objects)
- Consensus-based scene understanding

---

### 7. **Visual Reward Shaping**
Enhance the reward function with visual feedback:

```python
# Current reward components
reward = (
    task_completion_reward +
    energy_efficiency_reward +
    fairness_reward +
    deadline_penalty
)

# Vision-enhanced rewards
visual_reward = (
    coverage_quality_reward +      # How well is the area visually covered?
    detection_accuracy_reward +    # Did we detect all targets?
    visual_diversity_reward +      # Are drones looking at different things?
    occlusion_penalty              # Drones blocking each other's view
)
```

---

## 🛠️ Suggested Starting Points

### Quick Wins (1-2 weeks)
1. **Image Complexity Predictor**: Train a small CNN to predict task CPU cycles from images
2. **Visual State Encoder**: Add a frozen ResNet to encode "what the drone sees" into obs space
3. **Synthetic Visual Tasks**: Generate fake "images" with complexity labels

### Medium Projects (1-2 months)
4. **FedAvg with Visual Backbones**: Extend the existing FL infrastructure for CNN models
5. **Visual Attention for U2U Routing**: Which drone should I offload this image to?
6. **AirSim/Isaac Sim Integration**: Real visual rendering for the UAV environment

### Research Contributions (3+ months)
7. **ViT-HFL**: Hierarchical Federated Learning with Vision Transformers
8. **Communication-Constrained Visual FL**: Optimize what to send given bandwidth limits
9. **Visual Non-IID**: Different drones see different scenes → natural data heterogeneity

---

## 📚 Related Work to Review

### Vision + MARL
- "Learning to Communicate with Deep Multi-Agent RL" (Foerster et al.)
- "Multi-Agent Reinforcement Learning with Visual Observations" (OpenAI)

### Federated Learning + Vision
- "FedVision: Federated Video Analytics" 
- "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)

### UAV Vision Systems
- "A Survey on Deep Learning for UAV-Based Object Detection"
- "Collaborative Perception for Autonomous Driving"

### Neural Compression
- "Neural Compression for Video" (Google)
- "Learned Image Compression" (Ballé et al.)

---

## 🔗 Integration with HFL-UAV-Swarm

This project is designed to extend:
- **Repository**: `YoussefKamelKamel1/marl-drones`
- **Environment**: `BatchedUAVSwarmEnvGPU` 
- **Algorithms**: MAPPO, MADDPG, QMIX, Attn-FedHAdam, HiGAT-HAPPO, etc.

The goal is to add a **vision module** that can:
1. Generate visual observations for agents
2. Produce vision-based tasks
3. Enable visual feature communication between drones
4. Support federated visual model training

---

## 📁 Planned Structure

```
VISION_GRAD/
├── README.md
├── src/
│   ├── visual_encoder/       # CNN/ViT backbones
│   ├── task_generator/       # Vision-based task creation
│   ├── compression/          # Neural compression modules
│   ├── federated_vision/     # FL for visual models
│   └── perception/           # Detection, segmentation, depth
├── configs/
├── experiments/
└── docs/
```

---

## 🚀 Getting Started

```bash
# Clone alongside the main project
cd /home/skyvision
git clone https://github.com/ahmedislamfarouk/VISION_GRAD.git

# Link to main project (optional)
ln -s ../HFL-UAV-Swarm-Comparison/src/environments VISION_GRAD/src/environments
```

---

**Author**: Ahmed Islam Farouk (@ahmedislamfarouk)  
**Related Project**: [HFL-UAV-Swarm-Comparison](https://github.com/YoussefKamelKamel1/marl-drones)

---

## 🔬 YOLO Optimization & Research Directions

### 8. **YOLO for Edge UAV Deployment**

Running YOLO on resource-constrained drones requires aggressive optimization:

#### Model Compression Techniques

| Technique | Description | Speedup | Accuracy Drop |
|-----------|-------------|---------|---------------|
| **Pruning** | Remove unimportant weights/channels | 2-4x | 1-3% |
| **Quantization** | FP32 → INT8/INT4 | 2-4x | 0.5-2% |
| **Knowledge Distillation** | Large teacher → small student | N/A | Minimal |
| **Neural Architecture Search** | Auto-find optimal architecture | Variable | Can improve! |

```python
# Example: YOLOv8 Quantization for Drone
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Nano version for drones
model.export(format='tflite', int8=True)  # INT8 for edge TPU
model.export(format='onnx', half=True)    # FP16 for Jetson
```

#### YOLO Variants Comparison for UAVs

| Model | Params | FLOPs | mAP | FPS (Jetson) | Best For |
|-------|--------|-------|-----|--------------|----------|
| YOLOv8n | 3.2M | 8.7G | 37.3 | 45+ | Real-time on weak hardware |
| YOLOv8s | 11.2M | 28.6G | 44.9 | 25-35 | Balanced performance |
| YOLOv8m | 25.9M | 78.9G | 50.2 | 15-20 | High accuracy needed |
| YOLO-NAS | 12.9M | 36.5G | 47.5 | 30+ | AutoML optimized |
| RT-DETR | 32M | 103G | 53.1 | 10-15 | Transformer-based |

---

### 9. **Federated YOLO Training Across Drone Swarm**

Train a shared YOLO model without centralizing sensitive visual data:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Drone 1   │     │   Drone 2   │     │   Drone 3   │
│ Local Data  │     │ Local Data  │     │ Local Data  │
│ (Zone: City)│     │(Zone: Forest)│    │(Zone: Water)│
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
   ┌───────┐           ┌───────┐           ┌───────┐
   │ Local │           │ Local │           │ Local │
   │ YOLO  │           │ YOLO  │           │ YOLO  │
   │Training│          │Training│          │Training│
   └───┬───┘           └───┬───┘           └───┬───┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   FedAvg /  │
                    │ Attn-FedHAdam│
                    │  Aggregation │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Global YOLO │
                    │    Model    │
                    └─────────────┘
```

**Research Challenges:**
- **Non-IID Visual Data**: Each drone sees different scenes
- **Label Heterogeneity**: Different annotation quality per drone
- **Bandwidth Constraints**: YOLO gradients are large (~25MB for YOLOv8s)
- **Asynchronous Training**: Drones have different computation speeds

**Solutions to Explore:**
```python
# Gradient compression for YOLO FL
def compress_yolo_gradients(gradients, compression_ratio=0.1):
    """Top-K sparsification for bandwidth reduction"""
    flat = torch.cat([g.flatten() for g in gradients])
    k = int(len(flat) * compression_ratio)
    topk_vals, topk_idx = torch.topk(flat.abs(), k)
    return topk_vals, topk_idx  # Send only 10% of gradients!

# Layer-wise aggregation (backbone vs head)
aggregation_weights = {
    'backbone': 0.3,  # Share general features
    'neck': 0.5,      # Medium sharing
    'head': 1.0,      # Full local adaptation
}
```

---

### 10. **YOLO-Based Task Generation for UAV Swarm**

Replace synthetic tasks with detection-triggered computational tasks:

```python
class VisualTaskGenerator:
    def __init__(self, yolo_model='yolov8n.pt'):
        self.detector = YOLO(yolo_model)
        
        # Detection → Task mapping
        self.task_templates = {
            'person': {'cpu_cycles': 100, 'priority': 2, 'deadline_factor': 0.5},
            'car': {'cpu_cycles': 150, 'priority': 1, 'deadline_factor': 0.8},
            'fire': {'cpu_cycles': 50, 'priority': 3, 'deadline_factor': 0.2},  # URGENT!
            'animal': {'cpu_cycles': 200, 'priority': 1, 'deadline_factor': 1.0},
        }
    
    def generate_tasks(self, frame):
        detections = self.detector(frame)
        tasks = []
        for det in detections:
            cls_name = det.names[det.cls]
            template = self.task_templates.get(cls_name, default_template)
            
            # Scale CPU by detection confidence & bbox size
            complexity = det.conf * (det.box.area / frame_area)
            task = {
                'cpu_cycles': template['cpu_cycles'] * complexity,
                'data_size': estimate_crop_size(det.box),
                'deadline': template['deadline_factor'] * MAX_DEADLINE,
                'priority': template['priority'],
                'visual_crop': frame[det.box],  # Actual image data!
            }
            tasks.append(task)
        return tasks
```

---

### 11. **YOLO Inference Optimization Research**

#### A. **Dynamic Resolution Scaling**
Adjust input resolution based on flight speed and altitude:

```python
def adaptive_resolution(altitude, speed, base_res=640):
    """Higher altitude = smaller objects = need higher res"""
    altitude_factor = min(altitude / 50, 2.0)  # Cap at 2x
    speed_factor = max(1.0 - speed / 20, 0.5)   # Faster = lower res
    
    optimal_res = int(base_res * altitude_factor * speed_factor)
    return round(optimal_res / 32) * 32  # Multiple of 32 for YOLO
```

#### B. **Temporal Redundancy Exploitation**
Skip inference on similar frames:

```python
class TemporalYOLO:
    def __init__(self, model, skip_threshold=0.95):
        self.model = model
        self.skip_threshold = skip_threshold
        self.last_frame = None
        self.last_detections = None
    
    def detect(self, frame):
        if self.last_frame is not None:
            similarity = ssim(frame, self.last_frame)
            if similarity > self.skip_threshold:
                return self.last_detections  # Skip inference!
        
        detections = self.model(frame)
        self.last_frame = frame
        self.last_detections = detections
        return detections
```

#### C. **ROI-Based Inference**
Only run YOLO on regions of interest:

```python
def roi_inference(frame, motion_mask, model):
    """Run YOLO only where motion is detected"""
    contours = find_contours(motion_mask)
    detections = []
    
    for roi in contours:
        x, y, w, h = bounding_rect(roi)
        crop = frame[y:y+h, x:x+w]
        dets = model(crop)
        # Remap to full frame coordinates
        dets = offset_detections(dets, x, y)
        detections.extend(dets)
    
    return detections
```

---

### 12. **Specialized YOLO for Aerial/Drone Imagery**

Standard YOLO struggles with:
- **Small objects** (people from 100m altitude)
- **Rotation variance** (drone can be at any angle)
- **Altitude changes** (scale variance)

#### Research Directions:

**A. YOLO + SAHI (Slicing Aided Hyper Inference)**
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt',
)

# Slice image into overlapping tiles for small object detection
result = get_sliced_prediction(
    image,
    model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

**B. Rotation-Invariant Detection**
```python
# Train with rotation augmentation
augmentations = A.Compose([
    A.RandomRotate90(p=1.0),
    A.Rotate(limit=180, p=0.5),  # Full rotation
    A.Perspective(p=0.3),        # Simulate drone tilt
])

# Or use oriented bounding boxes (OBB)
# YOLOv8-OBB supports rotated boxes natively!
model = YOLO('yolov8n-obb.pt')
```

**C. Multi-Scale Feature Pyramid for Altitude Variance**
```python
# Custom YOLO head for extreme scale variance
class AltitudeAwareFPN(nn.Module):
    """Feature Pyramid optimized for drone altitude changes"""
    def __init__(self):
        self.scales = [8, 16, 32, 64, 128]  # More scales than default
        # Extra P6, P7 levels for very small objects
```

---

### 13. **YOLO + Swarm Coordination**

#### Collaborative Detection
Multiple drones share detections to improve accuracy:

```python
class SwarmDetector:
    def __init__(self, drone_id, comm_module):
        self.drone_id = drone_id
        self.comm = comm_module
        self.local_model = YOLO('yolov8n.pt')
    
    def collaborative_detect(self, frame, position):
        # Local detection
        local_dets = self.local_model(frame)
        
        # Broadcast to nearby drones
        self.comm.broadcast({
            'drone_id': self.drone_id,
            'position': position,
            'detections': local_dets,
            'timestamp': time.time(),
        })
        
        # Receive from neighbors
        neighbor_dets = self.comm.receive_all()
        
        # Fuse detections (NMS across drones)
        fused = multi_drone_nms(local_dets, neighbor_dets, positions)
        return fused
```

#### Detection-Based Task Routing
Route visual tasks to the best drone for processing:

```python
def route_detection_task(detection, swarm_state):
    """
    Given a detection that needs further processing,
    find the optimal drone to handle it.
    """
    candidates = []
    for drone in swarm_state:
        score = (
            0.3 * drone.battery_remaining +
            0.3 * drone.cpu_available +
            0.2 * (1 / distance(drone.pos, detection.pos)) +
            0.2 * drone.yolo_accuracy  # Some drones have better models!
        )
        candidates.append((drone.id, score))
    
    return max(candidates, key=lambda x: x[1])[0]
```

---

### 14. **Benchmark Datasets for Drone YOLO**

| Dataset | Images | Classes | Description |
|---------|--------|---------|-------------|
| **VisDrone** | 10K | 10 | Drone-captured urban scenes |
| **UAVDT** | 80K | 3 | Vehicle detection from UAV |
| **AU-AIR** | 32K | 8 | Multi-class aerial detection |
| **DOTA** | 2.8K | 15 | Oriented object detection |
| **SeaDronesSee** | 54K | 5 | Maritime search & rescue |

**Training Recipe:**
```bash
# Fine-tune YOLOv8 on VisDrone
yolo detect train data=VisDrone.yaml model=yolov8n.pt epochs=100 imgsz=1280

# Export optimized for Jetson
yolo export model=best.pt format=engine half=True device=0
```

---

## 🎯 Recommended YOLO Research Projects

### Beginner (1-2 weeks)
1. **Benchmark YOLOv8 variants** on VisDrone dataset
2. **Implement temporal skip** for video inference
3. **Quantize YOLOv8** and measure accuracy/speed tradeoff

### Intermediate (1-2 months)
4. **FedAvg for YOLO** across simulated drone swarm
5. **SAHI integration** for small object detection
6. **Detection-triggered task generation** for UAV env

### Advanced (3+ months)
7. **Gradient compression** for bandwidth-efficient YOLO FL
8. **Multi-drone NMS** for collaborative detection
9. **Custom YOLO architecture** optimized for aerial imagery
10. **VisDrone leaderboard submission** with FL-trained model

---

