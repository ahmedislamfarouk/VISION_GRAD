"""
Bidirectional Mutual Refinement Module

Enables iterative co-improvement between detector and VLM:
YOLO → detections → VLM → feedback → YOLO (refined)
Both models improve through iteration!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange


class BidirectionalMutualRefinement(nn.Module):
    """
    Bidirectional Mutual Refinement
    
    Iterative refinement loop:
    1. YOLO produces initial detections
    2. VLM analyzes detections and generates semantic feedback
    3. Feedback refines YOLO detections
    4. Refined detections improve VLM understanding
    5. Repeat for N iterations
    """
    
    def __init__(
        self,
        yolo_hidden_size: int = 512,
        vlm_hidden_size: int = 4096,
        num_refinement_iterations: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.yolo_hidden_size = yolo_hidden_size
        self.vlm_hidden_size = vlm_hidden_size
        self.num_refinement_iterations = num_refinement_iterations
        
        # YOLO → VLM projection
        self.yolo_to_vlm = nn.Sequential(
            nn.Linear(yolo_hidden_size, vlm_hidden_size),
            nn.LayerNorm(vlm_hidden_size),
            nn.GELU()
        )
        
        # VLM → YOLO feedback projection
        self.vlm_to_yolo = nn.Sequential(
            nn.Linear(vlm_hidden_size, yolo_hidden_size * 2),
            nn.GELU(),
            nn.Linear(yolo_hidden_size * 2, yolo_hidden_size),
            nn.LayerNorm(yolo_hidden_size)
        )
        
        # Detection refinement module
        self.detection_refiner = DetectionRefinementModule(
            hidden_size=yolo_hidden_size,
            num_heads=num_heads
        )
        
        # VLM semantic refiner
        self.vlm_semantic_refiner = VLMSemanticRefiner(
            hidden_size=vlm_hidden_size,
            num_heads=num_heads
        )
        
        # Confidence refinement
        self.confidence_refiner = ConfidenceRefinementModule(
            input_size=vlm_hidden_size + yolo_hidden_size
        )
        
        # Iteration tracking
        self.iteration_embed = nn.Embedding(num_refinement_iterations, vlm_hidden_size)
    
    def forward(
        self,
        yolo_features: Dict[str, torch.Tensor],
        vlm_hidden_states: List[torch.Tensor],
        initial_detections: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Bidirectional mutual refinement
        
        Args:
            yolo_features: YOLO FPN features at different scales
            vlm_hidden_states: VLM hidden states for each layer
            initial_detections: Initial detections from YOLO
                - boxes: (B, num_detections, 4)
                - scores: (B, num_detections, num_classes)
                - features: (B, num_detections, yolo_hidden_size)
            attention_mask: Optional attention mask
            num_iterations: Number of refinement iterations (default: self.num_refinement_iterations)
            
        Returns:
            refined_detections: Refined detections
            refined_vlm_states: Refined VLM hidden states
            refinement_info: Dictionary with refinement metrics
        """
        
        if num_iterations is None:
            num_iterations = self.num_refinement_iterations
        
        # Initialize
        detections = initial_detections
        vlm_states = vlm_hidden_states.copy()
        
        refinement_history = {
            'detection_scores': [],
            'vlm_features_norm': [],
            'confidence_changes': []
        }
        
        prev_confidence = detections['scores'].max(dim=-1, keepdim=True)[0]
        
        # Refinement iterations
        for iter_idx in range(num_iterations):
            iter_embedding = self.iteration_embed(torch.tensor(iter_idx, device=detections['boxes'].device))
            
            # === Step 1: YOLO → VLM ===
            # Project YOLO features to VLM space
            yolo_features_flat = {}
            for scale, features in yolo_features.items():
                B, C, H, W = features.shape
                yolo_features_flat[scale] = rearrange(features, 'b c h w -> b (h w) c')
            
            # Inject detection features into VLM
            detection_features = detections['features']
            detection_embedded = self.yolo_to_vlm(detection_features)
            
            # === Step 2: VLM Semantic Analysis ===
            # VLM processes detection features and generates semantic feedback
            vlm_states = self.vlm_semantic_refiner(
                vlm_hidden_states=vlm_states,
                detection_features=detection_embedded,
                iteration_embedding=iter_embedding,
                attention_mask=attention_mask
            )
            
            # === Step 3: VLM → YOLO Feedback ===
            # Extract feedback from VLM
            vlm_feedback = self._extract_vlm_feedback(vlm_states)
            yolo_feedback = self.vlm_to_yolo(vlm_feedback)
            
            # === Step 4: Refine Detections ===
            refined_detections = self.detection_refiner(
                detections=detections,
                feedback=yolo_feedback,
                yolo_features=yolo_features
            )
            
            # === Step 5: Refine Confidence ===
            combined_features = torch.cat([
                refined_detections['features'],
                vlm_feedback.mean(dim=1, keepdim=True).expand(-1, refined_detections['features'].shape[1], -1)
            ], dim=-1)
            
            confidence_delta = self.confidence_refiner(combined_features)
            refined_detections['scores'] = refined_detections['scores'] + confidence_delta
            
            # Update detections
            detections = refined_detections
            
            # Record history
            refinement_history['detection_scores'].append(
                detections['scores'].max().item()
            )
            refinement_history['vlm_features_norm'].append(
                torch.stack(vlm_states).norm().item()
            )
            refinement_history['confidence_changes'].append(
                (detections['scores'].max() - prev_confidence.max()).item()
            )
            prev_confidence = detections['scores'].max(dim=-1, keepdim=True)[0]
        
        refinement_history['final_detections'] = detections
        
        return detections, vlm_states, refinement_history
    
    def _extract_vlm_feedback(self, vlm_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract feedback from VLM hidden states
        
        Args:
            vlm_states: List of VLM hidden states
            
        Returns:
            Feedback features (B, seq_len, vlm_hidden_size)
        """
        # Aggregate information from all layers
        # Use last layer + attention-weighted sum of intermediate layers
        last_layer = vlm_states[-1]
        
        # Compute layer-wise attention weights
        layer_weights = []
        for i, state in enumerate(vlm_states):
            weight = torch.ones_like(state[:, 0:1, :]) * (i + 1) / len(vlm_states)
            layer_weights.append(weight)
        
        # Weighted sum
        weighted_sum = sum(w * s for w, s in zip(layer_weights, vlm_states))
        
        return weighted_sum


class DetectionRefinementModule(nn.Module):
    """
    Refines object detections based on VLM feedback
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Box refinement
        self.box_refiner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4)  # Box deltas
        )
        
        # Feature refinement
        self.feature_refiner = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        detections: Dict[str, torch.Tensor],
        feedback: torch.Tensor,
        yolo_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Refine detections
        
        Args:
            detections: Current detections
            feedback: VLM feedback features
            yolo_features: YOLO FPN features
            
        Returns:
            Refined detections
        """
        features = detections['features']
        boxes = detections['boxes']
        
        # Refine features with cross-attention to feedback
        features_norm = self.norm1(features)
        feedback_attended, _ = self.feature_refiner(
            features_norm,
            feedback,
            feedback,
            need_weights=False
        )
        features = features + feedback_attended
        
        # Refine boxes
        box_deltas = self.box_refiner(features)
        refined_boxes = boxes + box_deltas * 0.1  # Scale deltas
        
        return {
            'boxes': refined_boxes,
            'scores': detections['scores'],
            'features': features
        }


class VLMSemanticRefiner(nn.Module):
    """
    Refines VLM representations based on detection features
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Cross-attention: VLM attends to detection features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        vlm_hidden_states: List[torch.Tensor],
        detection_features: torch.Tensor,
        iteration_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Refine VLM states
        
        Args:
            vlm_hidden_states: VLM hidden states
            detection_features: Detection features from YOLO
            iteration_embedding: Iteration number embedding
            attention_mask: Attention mask
            
        Returns:
            Refined VLM hidden states
        """
        refined_states = []
        
        for hidden_state in vlm_hidden_states:
            # Add iteration embedding
            hidden_state = hidden_state + iteration_embedding.unsqueeze(1)
            
            # Self-attention
            hidden_norm = self.norm1(hidden_state)
            self_attended, _ = self.self_attn(
                hidden_norm,
                hidden_norm,
                hidden_norm,
                key_padding_mask=attention_mask,
                need_weights=False
            )
            hidden_state = hidden_state + self_attended
            
            # Cross-attention to detection features
            hidden_norm = self.norm2(hidden_state)
            cross_attended, _ = self.cross_attn(
                hidden_norm,
                detection_features,
                detection_features,
                need_weights=False
            )
            hidden_state = hidden_state + cross_attended
            
            # FFN
            hidden_norm = self.norm3(hidden_state)
            hidden_state = hidden_state + self.ffn(hidden_norm)
            
            refined_states.append(hidden_state)
        
        return refined_states


class ConfidenceRefinementModule(nn.Module):
    """
    Refines detection confidence scores based on VLM analysis
    """
    
    def __init__(self, input_size: int):
        super().__init__()
        
        self.confidence_network = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.LayerNorm(input_size // 2),
            nn.Linear(input_size // 2, input_size // 4),
            nn.GELU(),
            nn.Linear(input_size // 4, 1)
        )
    
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence refinement
        
        Args:
            combined_features: Combined VLM + YOLO features
            
        Returns:
            Confidence delta
        """
        return self.confidence_network(combined_features)


class IterativeRefinementTracker:
    """
    Tracks refinement progress across iterations
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.history = []
    
    def record_iteration(
        self,
        iteration: int,
        detections: Dict[str, torch.Tensor],
        vlm_states: List[torch.Tensor],
        loss: Optional[float] = None
    ):
        """Record refinement iteration metrics"""
        metrics = {
            'iteration': iteration,
            'num_detections': detections['boxes'].shape[1],
            'avg_confidence': detections['scores'].sigmoid().mean().item(),
            'vlm_feature_norm': torch.stack(vlm_states).norm().item(),
            'loss': loss
        }
        self.history.append(metrics)
    
    def get_summary(self) -> Dict:
        """Get summary of refinement process"""
        if not self.history:
            return {}
        
        summary = {
            'num_iterations': len(self.history),
            'confidence_trajectory': [h['avg_confidence'] for h in self.history],
            'vlm_norm_trajectory': [h['vlm_feature_norm'] for h in self.history]
        }
        
        # Check for convergence
        if len(self.history) >= 2:
            confidence_change = abs(
                self.history[-1]['avg_confidence'] - self.history[-2]['avg_confidence']
            )
            summary['converged'] = confidence_change < 0.01
        
        return summary
    
    def should_stop_early(self, threshold: float = 0.005) -> bool:
        """Check if refinement has converged"""
        if len(self.history) < 2:
            return False
        
        confidence_change = abs(
            self.history[-1]['avg_confidence'] - self.history[-2]['avg_confidence']
        )
        
        return confidence_change < threshold
