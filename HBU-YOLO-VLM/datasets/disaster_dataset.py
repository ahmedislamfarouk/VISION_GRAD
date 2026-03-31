"""
Dataset Loaders for HBU-YOLO-VLM

Supports:
- xBD Dataset (Building damage assessment)
- RescueNet (Disaster scene understanding)
- FloodNet (Flood damage assessment)
- Combined dataset
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from datasets.augmentations import build_augmentations
from datasets.text_templates import get_disaster_templates


class DisasterDataset(Dataset):
    """Base dataset for disaster imagery"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augmentations: Optional[Dict] = None,
        image_size: Tuple[int, int] = (512, 512),
        max_detections: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_detections = max_detections
        
        # Augmentations
        self.augmentations = build_augmentations(augmentations) if augmentations else None
        
        # Class names (disaster-specific)
        self.class_names = [
            'building', 'vehicle', 'person', 'debris', 'vegetation',
            'flood', 'fire', 'collapsed_structure', 'road', 'other'
        ]
        self.num_classes = len(self.class_names)
        
        # Load annotations
        self.images = []
        self.annotations = []
        self._load_annotations()
    
    def _load_annotations(self):
        """Load annotations from dataset"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load annotations
        ann = self.annotations[idx]
        boxes = ann.get('boxes', np.zeros((0, 4), dtype=np.float32))
        labels = ann.get('labels', np.zeros(0, dtype=np.int64))
        captions = ann.get('captions', [''])
        
        # Apply augmentations
        if self.augmentations:
            image, boxes = self.augmentations(image, boxes)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        # Prepare text data
        text_data = self._prepare_text(captions, boxes, labels)
        
        return {
            'images': image_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'texts': text_data,
            'image_path': str(img_path),
            'original_size': (image.shape[0], image.shape[1])
        }
    
    def _prepare_text(
        self,
        captions: List[str],
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare text data for VLM"""
        
        # Generate template-based captions if none provided
        if not captions or captions[0] == '':
            captions = self._generate_captions(boxes, labels)
        
        # Tokenize
        text_data = {
            'raw_texts': captions,
            'input_ids': None,
            'attention_mask': None,
            'labels': None
        }
        
        return text_data
    
    def _generate_captions(
        self,
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> List[str]:
        """Generate captions from detections using templates"""
        if len(boxes) == 0:
            return ["No objects detected in this disaster scene."]
        
        # Count objects by class
        class_counts = {}
        for label in labels:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Generate caption using template
        template = get_disaster_templates()
        caption = template.format_counts(class_counts)
        
        return [caption]


class XBDDataset(DisasterDataset):
    """xBD Dataset for building damage assessment"""
    
    def _load_annotations(self):
        """Load xBD annotations"""
        split_dir = self.data_dir / self.split
        
        # Load image paths
        images_dir = split_dir / "images"
        self.images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        
        # Load annotations
        labels_dir = split_dir / "labels"
        
        for img_path in self.images:
            ann_path = labels_dir / f"{img_path.stem}.json"
            
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                # Parse annotations
                boxes = []
                labels = []
                
                for building in ann_data.get('buildings', []):
                    # Get bounding box
                    poly = building.get('polygon', [])
                    if poly:
                        # Convert polygon to bbox
                        poly = np.array(poly).reshape(-1, 2)
                        x_min, y_min = poly.min(axis=0)
                        x_max, y_max = poly.max(axis=0)
                        boxes.append([x_min, y_min, x_max, y_max])
                        
                        # Get damage level as label
                        damage_level = building.get('damage_level', 0)
                        labels.append(damage_level)
                
                if boxes:
                    self.annotations.append({
                        'boxes': np.array(boxes, dtype=np.float32),
                        'labels': np.array(labels, dtype=np.int64),
                        'captions': [ann_data.get('caption', '')]
                    })
                else:
                    self.annotations.append({
                        'boxes': np.zeros((0, 4), dtype=np.float32),
                        'labels': np.zeros(0, dtype=np.int64),
                        'captions': ['']
                    })
            else:
                self.annotations.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros(0, dtype=np.int64),
                    'captions': ['']
                })


class RescueNetDataset(DisasterDataset):
    """RescueNet Dataset for disaster scene understanding"""
    
    def _load_annotations(self):
        """Load RescueNet annotations"""
        split_dir = self.data_dir / self.split
        
        # Load image paths
        images_dir = split_dir / "images"
        self.images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        
        # Load annotations
        labels_dir = split_dir / "annotations"
        
        for img_path in self.images:
            ann_path = labels_dir / f"{img_path.stem}.txt"
            
            boxes = []
            labels = []
            
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Format: class x_center y_center width height
                            cls = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to x_min, y_min, x_max, y_max
                            x_min = x_center - width / 2
                            y_min = y_center - height / 2
                            x_max = x_center + width / 2
                            y_max = y_center + height / 2
                            
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(cls)
            
            if boxes:
                self.annotations.append({
                    'boxes': np.array(boxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64),
                    'captions': ['']
                })
            else:
                self.annotations.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros(0, dtype=np.int64),
                    'captions': ['']
                })


class FloodNetDataset(DisasterDataset):
    """FloodNet Dataset for flood damage assessment"""
    
    def _load_annotations(self):
        """Load FloodNet annotations"""
        split_dir = self.data_dir / self.split
        
        # Load image paths
        images_dir = split_dir / "images"
        self.images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        
        # Load annotations
        labels_dir = split_dir / "labels"
        
        for img_path in self.images:
            ann_path = labels_dir / f"{img_path.stem}.json"
            
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                
                boxes = []
                labels = []
                
                for obj in ann_data.get('objects', []):
                    bbox = obj.get('bbox', [])
                    if bbox:
                        boxes.append(bbox)
                        labels.append(obj.get('category', 0))
                
                if boxes:
                    self.annotations.append({
                        'boxes': np.array(boxes, dtype=np.float32),
                        'labels': np.array(labels, dtype=np.int64),
                        'captions': [ann_data.get('caption', '')]
                    })
                else:
                    self.annotations.append({
                        'boxes': np.zeros((0, 4), dtype=np.float32),
                        'labels': np.zeros(0, dtype=np.int64),
                        'captions': ['']
                    })
            else:
                self.annotations.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros(0, dtype=np.int64),
                    'captions': ['']
                })


class CombinedDataset(DisasterDataset):
    """Combined dataset from multiple sources"""
    
    def __init__(
        self,
        xbd_dir: str,
        rescuenet_dir: str,
        floodnet_dir: str,
        **kwargs
    ):
        self.datasets = []
        
        # Load individual datasets
        if os.path.exists(xbd_dir):
            self.datasets.append(XBDDataset(xbd_dir, **kwargs))
        
        if os.path.exists(rescuenet_dir):
            self.datasets.append(RescueNetDataset(rescuenet_dir, **kwargs))
        
        if os.path.exists(floodnet_dir):
            self.datasets.append(FloodNetDataset(floodnet_dir, **kwargs))
        
        # Compute cumulative lengths
        self.cumulative_lengths = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)
        
        self.data_dir = Path(kwargs.get('data_dir', 'datasets/combined'))
        self.split = kwargs.get('split', 'train')
        self.augmentations = None
        self.image_size = kwargs.get('image_size', (512, 512))
        self.max_detections = kwargs.get('max_detections', 100)
        self.class_names = self.datasets[0].class_names if self.datasets else []
        self.num_classes = self.datasets[0].num_classes if self.datasets else 0
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Find which dataset
        dataset_idx = 0
        for i, length in enumerate(self.cumulative_lengths):
            if idx < length:
                dataset_idx = i
                break
        
        # Adjust index
        if dataset_idx > 0:
            idx -= self.cumulative_lengths[dataset_idx - 1]
        
        return self.datasets[dataset_idx][idx]


def build_dataset(
    config: Any,
    data_dir: str,
    split: str = "train",
    augmentations: Optional[Dict] = None
) -> DisasterDataset:
    """Build dataset from config"""
    
    dataset_name = config.name.lower()
    
    if dataset_name == "xbd":
        dataset = XBDDataset(
            data_dir=data_dir,
            split=split,
            augmentations=augmentations,
            image_size=tuple(config.get('input_size', [512, 512]))
        )
    elif dataset_name == "rescuenet":
        dataset = RescueNetDataset(
            data_dir=data_dir,
            split=split,
            augmentations=augmentations,
            image_size=tuple(config.get('input_size', [512, 512]))
        )
    elif dataset_name == "floodnet":
        dataset = FloodNetDataset(
            data_dir=data_dir,
            split=split,
            augmentations=augmentations,
            image_size=tuple(config.get('input_size', [512, 512]))
        )
    elif dataset_name == "combined":
        dataset = CombinedDataset(
            xbd_dir=os.path.join(data_dir, "xbd"),
            rescuenet_dir=os.path.join(data_dir, "rescuenet"),
            floodnet_dir=os.path.join(data_dir, "floodnet"),
            split=split,
            augmentations=augmentations,
            image_size=tuple(config.get('input_size', [512, 512]))
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function"""
    
    # Stack images
    images = torch.stack([item['images'] for item in batch])
    
    # Collect other fields
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    texts = [item['texts'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'texts': texts,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }


def build_dataloader(
    dataset: DisasterDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """Build dataloader"""
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
        shuffle = None
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    return dataloader
