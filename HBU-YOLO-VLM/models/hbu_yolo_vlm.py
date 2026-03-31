"""
HBU-YOLO-VLM: Main Integrated Model

Hierarchical Bidirectional Uncertainty-Aware Deep Fusion
of Detection and Vision-Language Models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import DictConfig

from models.yolo.yolo_backbone import YOLOBackbone, FeaturePyramidNetwork, DetectionHead
from models.vlm.vlm_module import LLaVAModule
from models.fusion.hierarchical_fusion import HierarchicalFPNVLMFusion
from models.fusion.bidirectional_refinement import BidirectionalMutualRefinement
from models.uncertainty.uncertainty_attention import (
    UncertaintyAwareAdaptiveAttention,
    UncertaintyEstimationHead
)


class HBUYOLOVLM(nn.Module):
    """
    HBU-YOLO-VLM: Complete integrated model
    
    Architecture:
    1. YOLO backbone extracts multi-scale features
    2. FPN enhances and merges features
    3. Hierarchical fusion injects FPN features into VLM layers
    4. Uncertainty-aware attention adapts computation
    5. Bidirectional refinement iteratively improves both models
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        
        # === YOLO Components ===
        self.yolo_backbone = YOLOBackbone(
            model_size=config.yolo.model_size,
            pretrained=config.yolo.pretrained
        )
        
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.yolo_backbone.get_out_channels(),
            out_channels=config.fusion.vlm_hidden_size
        )
        
        self.detection_head = DetectionHead(
            in_channels=config.fusion.vlm_hidden_size,
            num_classes=config.model.num_classes,
            num_anchors=config.model.num_anchors
        )
        
        # === VLM Components ===
        self.vlm = LLaVAModule(
            vision_model=config.vlm.vision_model,
            llm_model=config.vlm.llm_model,
            projection_type=config.vlm.projection_type
        )
        
        # Load tokenizer
        self.vlm.load_tokenizer(config.vlm.tokenizer)
        
        # === Fusion Components ===
        self.hierarchical_fusion = HierarchicalFPNVLMFusion(
            fpn_channels=self.yolo_backbone.get_out_channels(),
            vlm_hidden_size=config.fusion.vlm_hidden_size,
            vlm_num_layers=config.fusion.vlm_num_layers,
            num_heads=config.fusion.num_heads,
            fusion_type=config.fusion.fusion_type
        )
        
        # === Uncertainty Components ===
        self.uncertainty_attention = UncertaintyAwareAdaptiveAttention(
            vlm_hidden_size=config.fusion.vlm_hidden_size,
            num_vlm_layers=config.fusion.vlm_num_layers,
            uncertainty_threshold_low=config.uncertainty.threshold_low,
            uncertainty_threshold_high=config.uncertainty.threshold_high
        )
        
        self.uncertainty_head = UncertaintyEstimationHead(
            in_channels=config.fusion.vlm_hidden_size,
            num_anchors=config.model.num_anchors
        )
        
        # === Bidirectional Refinement ===
        self.bidirectional_refinement = BidirectionalMutualRefinement(
            yolo_hidden_size=config.fusion.vlm_hidden_size,
            vlm_hidden_size=config.fusion.vlm_hidden_size,
            num_refinement_iterations=config.refinement.num_iterations
        )
        
        # === Output Heads ===
        # Caption head for disaster scene description
        self.caption_head = nn.Sequential(
            nn.Linear(config.fusion.vlm_hidden_size, config.fusion.vlm_hidden_size),
            nn.GELU(),
            nn.Linear(config.fusion.vlm_hidden_size, self.vlm.llm_decoder.vocab_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[Dict[str, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            images: Input images (B, 3, H, W)
            texts: Text data for VLM (optional, for training)
            boxes: Ground truth boxes (optional, for training)
            labels: Ground truth labels (optional, for training)
            return_loss: Whether to compute loss
            
        Returns:
            Dictionary with predictions and losses
        """
        
        output = {}
        
        # === Step 1: YOLO Backbone ===
        yolo_features = self.yolo_backbone(images)
        
        # === Step 2: FPN ===
        fpn_features = self.fpn(yolo_features)
        
        # === Step 3: Initial Detection ===
        detection_features = fpn_features['P4']  # Use P4 for detection
        cls_pred, reg_pred, uncertainty_pred = self.detection_head(detection_features)
        
        # Extract initial detections
        initial_detections = self._decode_detections(cls_pred, reg_pred, uncertainty_pred)
        
        # === Step 4: VLM Encoding ===
        vlm_visual_embeddings = self.vlm.encode_images(images)
        
        # Get VLM hidden states (simplified - would normally go through LLM layers)
        vlm_hidden_states = [vlm_visual_embeddings] * self.config.fusion.vlm_num_layers
        
        # === Step 5: Hierarchical Fusion ===
        fused_vlm_states = self.hierarchical_fusion(
            fpn_features=fpn_features,
            vlm_hidden_states=vlm_hidden_states
        )
        
        # === Step 6: Uncertainty Estimation ===
        aleatoric, epistemic = self.uncertainty_head(detection_features)
        combined_uncertainty = aleatoric + epistemic
        
        # === Step 7: Uncertainty-Aware Adaptive Attention ===
        adapted_vlm_states, routing_info = self.uncertainty_attention(
            vlm_hidden_states=fused_vlm_states,
            uncertainty_map=combined_uncertainty
        )
        
        # === Step 8: Bidirectional Mutual Refinement ===
        refined_detections, refined_vlm_states, refinement_info = \
            self.bidirectional_refinement(
                yolo_features=fpn_features,
                vlm_hidden_states=adapted_vlm_states,
                initial_detections=initial_detections
            )
        
        # === Step 9: Generate Outputs ===
        # Final detections
        output['detections'] = refined_detections
        
        # Uncertainty estimates
        output['uncertainty'] = {
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'combined': combined_uncertainty
        }
        
        # Routing information
        output['routing'] = routing_info
        
        # Refinement metrics
        output['refinement'] = refinement_info
        
        # === Step 10: Loss Computation (if training) ===
        if return_loss and texts is not None:
            losses = self._compute_loss(
                detections=refined_detections,
                vlm_states=refined_vlm_states,
                uncertainty=output['uncertainty'],
                ground_truth_boxes=boxes,
                ground_truth_labels=labels,
                ground_truth_texts=texts
            )
            output['losses'] = losses
        
        return output
    
    def _decode_detections(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        uncertainty_pred: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decode detection predictions
        
        Args:
            cls_pred: Class predictions
            reg_pred: Box regression predictions
            uncertainty_pred: Uncertainty predictions
            
        Returns:
            Dictionary with boxes, scores, and features
        """
        B, _, H, W = cls_pred.shape
        
        # Reshape predictions
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, H * W * self.config.model.num_anchors, -1)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, H * W * self.config.model.num_anchors, 4)
        
        # Convert to boxes (simplified - would need anchor decoding)
        boxes = reg_pred.sigmoid() * 2 - 0.5  # Normalize to [-0.5, 1.5]
        
        # Scores
        scores = cls_pred.sigmoid()
        
        # Features (use class prediction features)
        features = cls_pred
        
        return {
            'boxes': boxes,
            'scores': scores,
            'features': features
        }
    
    def _compute_loss(
        self,
        detections: Dict[str, torch.Tensor],
        vlm_states: List[torch.Tensor],
        uncertainty: Dict[str, torch.Tensor],
        ground_truth_boxes: Optional[torch.Tensor] = None,
        ground_truth_labels: Optional[torch.Tensor] = None,
        ground_truth_texts: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            detections: Model predictions
            vlm_states: VLM hidden states
            uncertainty: Uncertainty estimates
            ground_truth_boxes: GT boxes
            ground_truth_labels: GT labels
            ground_truth_texts: GT texts
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # === Detection Loss ===
        if ground_truth_boxes is not None and ground_truth_labels is not None:
            # Box loss (L1 + IoU)
            pred_boxes = detections['boxes']
            gt_boxes = ground_truth_boxes
            
            box_loss = nn.functional.l1_loss(pred_boxes, gt_boxes, reduction='mean')
            losses['box_loss'] = box_loss
            
            # Classification loss (focal loss)
            pred_scores = detections['scores']
            gt_labels = ground_truth_labels
            
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                pred_scores, gt_labels, reduction='mean'
            )
            losses['cls_loss'] = cls_loss
        
        # === VLM Loss ===
        if ground_truth_texts is not None:
            # Language modeling loss
            input_ids = ground_truth_texts.get('input_ids')
            attention_mask = ground_truth_texts.get('attention_mask')
            labels = ground_truth_texts.get('labels')
            
            # Get last VLM state
            last_vlm_state = vlm_states[-1]
            
            # Project to vocab
            logits = self.caption_head(last_vlm_state)
            
            # Compute loss
            if labels is not None:
                lm_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100
                )
                losses['lm_loss'] = lm_loss
        
        # === Uncertainty Loss ===
        # Encourage accurate uncertainty estimation
        if ground_truth_boxes is not None:
            pred_boxes = detections['boxes']
            gt_boxes = ground_truth_boxes
            
            # Prediction error
            pred_error = (pred_boxes - gt_boxes).abs().mean(dim=-1)
            
            # Uncertainty should correlate with error
            unc_pred = uncertainty['combined'].mean(dim=[1, 2, 3])
            
            # Uncertainty calibration loss
            unc_loss = nn.functional.mse_loss(unc_pred, pred_error.detach())
            losses['unc_loss'] = unc_loss
        
        # === Total Loss ===
        total_loss = (
            losses.get('box_loss', 0) * self.config.loss.box_weight +
            losses.get('cls_loss', 0) * self.config.loss.cls_weight +
            losses.get('lm_loss', 0) * self.config.loss.lm_weight +
            losses.get('unc_loss', 0) * self.config.loss.unc_weight
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def generate(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate predictions and captions
        
        Args:
            images: Input images
            prompts: Optional text prompts
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with detections and generated text
        """
        # Forward pass
        output = self(images, return_loss=False)
        
        # Generate captions
        if prompts is None:
            prompts = ["Describe this disaster scene:"]
        
        # Tokenize prompts
        if self.vlm.tokenizer is not None:
            tokenized = self.vlm.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(images.device)
            
            # Generate
            generated_ids = self.vlm.llm.generate(
                **tokenized,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 1.0
            )
            
            # Decode
            generated_text = self.vlm.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            output['generated_text'] = generated_text
        
        return output
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            'yolo_model': self.config.yolo.model_size,
            'vlm_vision': self.config.vlm.vision_model,
            'vlm_llm': self.config.vlm.llm_model,
            'fusion_type': self.config.fusion.fusion_type,
            'vlm_hidden_size': self.config.fusion.vlm_hidden_size,
            'vlm_num_layers': self.config.fusion.vlm_num_layers,
            'num_classes': self.config.model.num_classes,
            'refinement_iterations': self.config.refinement.num_iterations
        }


def build_model(config: DictConfig) -> HBUYOLOVLM:
    """
    Build HBU-YOLO-VLM model from config
    
    Args:
        config: Configuration
        
    Returns:
        Model
    """
    model = HBUYOLOVLM(config)
    return model
