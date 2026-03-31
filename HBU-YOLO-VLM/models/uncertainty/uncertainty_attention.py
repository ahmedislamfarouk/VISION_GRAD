"""
Uncertainty-Aware Adaptive Attention Module

Dynamically allocates VLM resources based on detection confidence:
- High confidence → minimal VLM compute (efficient!)
- Low confidence → heavy VLM attention (accurate!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange


class UncertaintyAwareAdaptiveAttention(nn.Module):
    """
    Uncertainty-Aware Adaptive Attention
    
    Dynamically routes samples through different computation paths
    based on detection uncertainty:
    
    - Low uncertainty (high confidence): Skip VLM computation, use fast path
    - Medium uncertainty: Partial VLM layers
    - High uncertainty (low confidence): Full VLM computation
    """
    
    def __init__(
        self,
        vlm_hidden_size: int = 4096,
        num_vlm_layers: int = 32,
        uncertainty_threshold_low: float = 0.3,
        uncertainty_threshold_high: float = 0.7,
        num_routing_paths: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vlm_hidden_size = vlm_hidden_size
        self.num_vlm_layers = num_vlm_layers
        self.uncertainty_threshold_low = uncertainty_threshold_low
        self.uncertainty_threshold_high = uncertainty_threshold_high
        self.num_routing_paths = num_routing_paths
        
        # Uncertainty encoder
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(1, vlm_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(vlm_hidden_size // 4, vlm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(vlm_hidden_size // 2, vlm_hidden_size)
        )
        
        # Routing network (predicts computation path)
        self.routing_network = nn.Sequential(
            nn.Linear(vlm_hidden_size, vlm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vlm_hidden_size // 2, num_routing_paths),
            nn.Softmax(dim=-1)
        )
        
        # Computation paths with different depths
        self.computation_paths = nn.ModuleDict({
            'fast': FastPath(vlm_hidden_size),
            'medium': MediumPath(vlm_hidden_size, num_layers=8),
            'full': FullPath(vlm_hidden_size, num_layers=num_vlm_layers)
        })
        
        # Uncertainty-guided attention modulation
        self.attention_modulator = UncertaintyAttentionModulator(
            hidden_size=vlm_hidden_size
        )
    
    def forward(
        self,
        vlm_hidden_states: List[torch.Tensor],
        uncertainty_map: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply uncertainty-aware adaptive attention
        
        Args:
            vlm_hidden_states: VLM hidden states for each layer
            uncertainty_map: Uncertainty estimates (B, num_anchors, H, W)
            attention_mask: Optional attention mask
            
        Returns:
            enhanced_states: Enhanced VLM hidden states
            routing_info: Dictionary with routing decisions and metrics
        """
        B = vlm_hidden_states[0].shape[0]
        
        # Aggregate uncertainty to global measure per sample
        uncertainty_global = self._aggregate_uncertainty(uncertainty_map)
        
        # Encode uncertainty
        uncertainty_embedding = self.uncertainty_encoder(uncertainty_global.unsqueeze(-1))
        
        # Compute routing probabilities
        routing_probs = self.routing_network(uncertainty_embedding)
        
        # Route samples to different computation paths
        routing_info = {
            'routing_probs': routing_probs,
            'routing_decisions': routing_probs.argmax(dim=-1)
        }
        
        # Apply adaptive computation
        enhanced_states = []
        
        for layer_idx, hidden_state in enumerate(vlm_hidden_states):
            # Modulate attention based on uncertainty
            modulated_state = self.attention_modulator(
                hidden_state,
                uncertainty_embedding,
                attention_mask
            )
            
            enhanced_states.append(modulated_state)
        
        # Apply computation path based on routing
        path_outputs = {}
        for path_name, path_module in self.computation_paths.items():
            path_outputs[path_name] = path_module(enhanced_states)
        
        # Weighted combination based on routing probabilities
        final_states = self._combine_path_outputs(
            path_outputs,
            routing_probs
        )
        
        routing_info['path_weights'] = routing_probs
        routing_info['final_states_norm'] = torch.stack(final_states).norm()
        
        return final_states, routing_info
    
    def _aggregate_uncertainty(self, uncertainty_map: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spatial uncertainty map to global uncertainty per sample
        
        Args:
            uncertainty_map: (B, num_anchors, H, W)
            
        Returns:
            Global uncertainty (B,)
        """
        # Global average pooling
        global_uncertainty = uncertainty_map.mean(dim=[1, 2, 3])
        
        # Normalize to [0, 1]
        global_uncertainty = (global_uncertainty - global_uncertainty.min()) / \
                            (global_uncertainty.max() - global_uncertainty.min() + 1e-8)
        
        return global_uncertainty
    
    def _combine_path_outputs(
        self,
        path_outputs: Dict[str, List[torch.Tensor]],
        routing_probs: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Combine outputs from different computation paths
        
        Args:
            path_outputs: Dictionary of path outputs
            routing_probs: Routing probabilities (B, num_paths)
            
        Returns:
            Combined hidden states
        """
        # Get outputs from each path
        fast_outputs = path_outputs['fast']
        medium_outputs = path_outputs['medium']
        full_outputs = path_outputs['full']
        
        # Weight by routing probabilities
        B = routing_probs.shape[0]
        combined_states = []
        
        for layer_idx in range(len(fast_outputs)):
            weighted_sum = (
                routing_probs[:, 0:1].view(B, 1, 1) * fast_outputs[layer_idx] +
                routing_probs[:, 1:2].view(B, 1, 1) * medium_outputs[layer_idx] +
                routing_probs[:, 2:3].view(B, 1, 1) * full_outputs[layer_idx]
            )
            combined_states.append(weighted_sum)
        
        return combined_states


class UncertaintyAttentionModulator(nn.Module):
    """
    Modulates attention strength based on uncertainty
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Uncertainty-guided scaling
        self.uncertainty_scale = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_heads),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        uncertainty_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply uncertainty-modulated self-attention
        
        Args:
            hidden_state: (B, seq_len, hidden_size)
            uncertainty_embedding: (B, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Attended hidden state
        """
        B, seq_len, _ = hidden_state.shape
        
        # Compute Q, K, V
        Q = self.q_proj(hidden_state).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_state).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_state).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        # Apply uncertainty-guided scaling
        scale_factors = self.uncertainty_scale(uncertainty_embedding)  # (B, num_heads)
        scale_factors = scale_factors.view(B, self.num_heads, 1, 1)
        attn_scores = attn_scores * scale_factors
        
        # Softmax and weighted sum
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection with normalization
        output = self.norm(hidden_state + attn_output)
        
        return output


class FastPath(nn.Module):
    """
    Fast computation path: Minimal processing (2 layers)
    Used for high-confidence detections
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(2)
        ])
    
    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply fast path processing"""
        outputs = []
        for hidden_state in hidden_states:
            for layer in self.layers:
                hidden_state = layer(hidden_state)
            outputs.append(hidden_state)
        return outputs


class MediumPath(nn.Module):
    """
    Medium computation path: Moderate processing (8 layers)
    Used for medium-confidence detections
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 8):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply medium path processing"""
        outputs = []
        for hidden_state in hidden_states:
            for layer in self.layers:
                hidden_state = layer(hidden_state)
            outputs.append(hidden_state)
        return outputs


class FullPath(nn.Module):
    """
    Full computation path: Complete VLM processing (32 layers)
    Used for low-confidence (high uncertainty) detections
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 32):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply full path processing"""
        outputs = []
        for hidden_state in hidden_states:
            for layer in self.layers:
                hidden_state = layer(hidden_state)
            outputs.append(hidden_state)
        return outputs


class UncertaintyEstimationHead(nn.Module):
    """
    Predicts aleatoric and epistemic uncertainty
    """
    
    def __init__(self, in_channels: int, num_anchors: int = 3):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # Aleatoric uncertainty (data uncertainty)
        self.aleatoric_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Epistemic uncertainty (model uncertainty)
        self.epistemic_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors, 1),
            nn.Sigmoid()  # Range [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict uncertainty
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            aleatoric: Aleatoric uncertainty (B, num_anchors, H, W)
            epistemic: Epistemic uncertainty (B, num_anchors, H, W)
        """
        aleatoric = self.aleatoric_head(features)
        epistemic = self.epistemic_head(features)
        
        return aleatoric, epistemic


class AdaptiveComputationGate(nn.Module):
    """
    Learns to gate computation based on uncertainty and input features
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_computation_levels: int = 3,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.num_computation_levels = num_computation_levels
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_computation_levels)
        )
        
        # Uncertainty embedding
        self.uncertainty_proj = nn.Linear(1, hidden_size // 4)
    
    def forward(
        self,
        features: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating weights
        
        Args:
            features: Input features (B, hidden_size)
            uncertainty: Uncertainty estimates (B, 1)
            
        Returns:
            gate_weights: Soft gating weights (B, num_levels)
            gate_decision: Hard gate decision (B,)
        """
        # Project uncertainty
        unc_embedded = self.uncertainty_proj(uncertainty)
        
        # Concatenate features and uncertainty
        combined = torch.cat([features, unc_embedded], dim=-1)
        
        # Compute gate logits
        gate_logits = self.gate_network(combined)
        
        # Apply temperature-scaled softmax
        gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        
        # Hard decision
        gate_decision = gate_weights.argmax(dim=-1)
        
        return gate_weights, gate_decision
