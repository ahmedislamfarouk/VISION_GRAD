"""
YOLO Backbone and Feature Pyramid Network for HBU-YOLO-VLM
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from ultralytics import YOLO


class YOLOBackbone(nn.Module):
    """
    YOLOv8 backbone with multi-scale feature extraction
    
    Extracts features at multiple scales (P2-P6) for hierarchical fusion
    """
    
    def __init__(self, model_size: str = 'yolov8m.pt', pretrained: bool = True):
        super().__init__()
        self.model_size = model_size
        
        # Load YOLO model
        if pretrained:
            self.yolo = YOLO(model_size)
        else:
            self.yolo = YOLO(model_size, pretrained=False)
        
        # Extract backbone components
        self.backbone = self.yolo.model[:10]  # First 10 layers
        
        # Feature channels for different scales
        self.out_channels = {
            'P2': 256,
            'P3': 512,
            'P4': 512,
            'P5': 512,
            'P6': 512
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from YOLO backbone
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Dictionary of features at different scales
        """
        features = {}
        
        # Pass through backbone layers
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            # Extract features at different scales
            if i == 3:   # P2 - 1/4 scale
                features['P2'] = x
            elif i == 5: # P3 - 1/8 scale
                features['P3'] = x
            elif i == 7: # P4 - 1/16 scale
                features['P4'] = x
            elif i == 9: # P5 - 1/32 scale
                features['P5'] = x
        
        # P6 - Global context (1/64 scale)
        features['P6'] = torch.mean(x, dim=[2, 3], keepdim=True).expand(-1, -1, x.shape[2]//2, x.shape[3]//2)
        
        return features
    
    def get_out_channels(self) -> Dict[str, int]:
        return self.out_channels


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """
    
    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleDict({
            scale: nn.Conv2d(channels, out_channels, 1)
            for scale, channels in in_channels.items()
        })
        
        # Output convolutions
        self.output_convs = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            for scale in ['P2', 'P3', 'P4', 'P5', 'P6']
        })
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply FPN to merge multi-scale features
        
        Args:
            features: Dictionary of features at different scales
            
        Returns:
            Enhanced multi-scale features
        """
        # Top-down pathway
        top_down = {}
        
        # Start from highest level (P6)
        top_down['P6'] = self.lateral_convs['P6'](features['P6'])
        
        # Bottom-up refinement
        for scale in ['P5', 'P4', 'P3', 'P2']:
            lateral = self.lateral_convs[scale](features[scale])
            
            if scale in top_down:
                # Upsample and add
                top_down_feature = nn.functional.interpolate(
                    top_down[scale],
                    size=lateral.shape[2:],
                    mode='nearest'
                )
                lateral = lateral + top_down_feature
            
            top_down[scale] = lateral
        
        # Apply output convolutions
        enhanced_features = {}
        for scale in ['P2', 'P3', 'P4', 'P5', 'P6']:
            enhanced_features[scale] = self.output_convs[scale](top_down[scale])
        
        return enhanced_features


class DetectionHead(nn.Module):
    """
    YOLO detection head with uncertainty estimation
    """
    
    def __init__(self, in_channels: int, num_classes: int = 80, num_anchors: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        )
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * num_anchors, 1)
        )
        
        # Uncertainty branch (aleatoric)
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict classes, boxes, and uncertainty
        
        Args:
            x: Input features
            
        Returns:
            cls_pred: Class predictions (B, num_anchors*num_classes, H, W)
            reg_pred: Box predictions (B, num_anchors*4, H, W)
            uncertainty: Uncertainty estimates (B, num_anchors, H, W)
        """
        cls_pred = self.cls_conv(x)
        reg_pred = self.reg_conv(x)
        uncertainty = self.uncertainty_conv(x)
        
        return cls_pred, reg_pred, uncertainty
