"""
Data Augmentations for HBU-YOLO-VLM
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, Any


def build_augmentations(config: Optional[Dict[str, Any]] = None):
    """Build augmentation pipeline"""
    
    if config is None or not config.get('enabled', True):
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    # Build augmentation pipeline
    transforms = []
    
    # Resize
    transforms.append(A.Resize(512, 512))
    
    # Geometric augmentations
    if config.get('hflip', 0.0) > 0:
        transforms.append(A.HorizontalFlip(p=config['hflip']))
    
    if config.get('vflip', 0.0) > 0:
        transforms.append(A.VerticalFlip(p=config['vflip']))
    
    if config.get('rotation', 0) > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=0,
            rotate_limit=config['rotation'],
            p=0.5
        ))
    
    if config.get('scale', 0) > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=config['scale'],
            rotate_limit=0,
            p=0.5
        ))
    
    # Color augmentations
    if config.get('color_jitter', 0) > 0:
        transforms.append(A.ColorJitter(
            brightness=config['color_jitter'],
            contrast=config['color_jitter'],
            saturation=config['color_jitter'],
            hue=0,
            p=0.5
        ))
    
    # Random crop
    if config.get('random_crop', False):
        transforms.append(A.RandomCrop(480, 480))
    
    # Normalization
    if config.get('normalize', True):
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.1
        )
    )
