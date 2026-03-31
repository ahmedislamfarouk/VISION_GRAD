"""
Hierarchical FPN-VLM Fusion Module

Maps YOLO's multi-scale FPN outputs to semantically-aligned VLM transformer layers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from einops import rearrange


class HierarchicalFPNVLMFusion(nn.Module):
    """
    Hierarchical Feature Injection: Maps YOLO's multi-scale FPN outputs
    to semantically-aligned VLM transformer layers
    
    P2 → Layers 1-6   (tiny details: "crack", "spark")
    P3 → Layers 7-12  (small objects: "person", "debris")
    P4 → Layers 13-20 (medium: "car", "building section")
    P5 → Layers 21-26 (large: "collapsed structure")
    P6 → Layers 27-32 (context: "disaster zone")
    """
    
    def __init__(
        self,
        fpn_channels: Dict[str, int],
        vlm_hidden_size: int = 4096,
        vlm_num_layers: int = 32,
        num_heads: int = 8,
        fusion_type: str = 'cross_attention'
    ):
        super().__init__()
        
        self.fpn_channels = fpn_channels
        self.vlm_hidden_size = vlm_hidden_size
        self.vlm_num_layers = vlm_num_layers
        
        # Layer assignments for each FPN scale
        self.layer_assignments = {
            'P2': (0, 6),      # Layers 1-6
            'P3': (6, 12),     # Layers 7-12
            'P4': (12, 20),    # Layers 13-20
            'P5': (20, 26),    # Layers 21-26
            'P6': (26, 32)     # Layers 27-32
        }
        
        # Projection layers for each scale
        self.fpn_projections = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv2d(channels, vlm_hidden_size, 1),
                nn.LayerNorm(vlm_hidden_size),
                nn.GELU()
            )
            for scale, channels in fpn_channels.items()
        })
        
        # Fusion modules for each layer range
        self.fusion_modules = nn.ModuleDict()
        
        for scale, (start_layer, end_layer) in self.layer_assignments.items():
            for layer_idx in range(start_layer, end_layer):
                module_key = f"layer_{layer_idx}"
                
                if fusion_type == 'cross_attention':
                    self.fusion_modules[module_key] = CrossAttentionFusion(
                        hidden_size=vlm_hidden_size,
                        num_heads=num_heads
                    )
                elif fusion_type == 'adaptive_ln':
                    self.fusion_modules[module_key] = AdaptiveLayerNormFusion(
                        hidden_size=vlm_hidden_size
                    )
                elif fusion_type == 'gating':
                    self.fusion_modules[module_key] = GatedFusion(
                        hidden_size=vlm_hidden_size
                    )
                else:
                    raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def get_layer_range(self, scale: str) -> Tuple[int, int]:
        """Get layer range for a given FPN scale"""
        return self.layer_assignments[scale]
    
    def forward(
        self,
        fpn_features: Dict[str, torch.Tensor],
        vlm_hidden_states: List[torch.Tensor],
        vlm_attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Inject FPN features into VLM transformer layers
        
        Args:
            fpn_features: Dictionary of FPN features {scale: (B, C, H, W)}
            vlm_hidden_states: List of VLM hidden states for each layer
            vlm_attention_mask: Optional attention mask
            
        Returns:
            Enhanced VLM hidden states
        """
        # Project FPN features to VLM space
        projected_features = {}
        for scale in fpn_features.keys():
            # Conv2d -> flatten spatial dims
            B, C, H, W = fpn_features[scale].shape
            projected = self.fpn_projections[scale](fpn_features[scale])
            
            # Reshape to (B, H*W, C)
            projected = rearrange(projected, 'b c h w -> b (h w) c')
            projected_features[scale] = projected
        
        # Inject features at assigned layers
        enhanced_states = vlm_hidden_states.copy()
        
        for scale, (start_layer, end_layer) in self.layer_assignments.items():
            fpn_feat = projected_features[scale]
            
            for layer_idx in range(start_layer, end_layer):
                if layer_idx < len(enhanced_states):
                    module_key = f"layer_{layer_idx}"
                    
                    if module_key in self.fusion_modules:
                        enhanced_states[layer_idx] = self.fusion_modules[module_key](
                            vlm_hidden=enhanced_states[layer_idx],
                            fpn_features=fpn_feat,
                            attention_mask=vlm_attention_mask
                        )
        
        return enhanced_states


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion module
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Cross-attention: VLM attends to FPN features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        vlm_hidden: torch.Tensor,
        fpn_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse VLM hidden states with FPN features via cross-attention
        
        Args:
            vlm_hidden: VLM hidden states (B, seq_len, hidden_size)
            fpn_features: FPN features (B, num_spatial, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Fused hidden states
        """
        # Normalize
        vlm_norm = self.norm1(vlm_hidden)
        
        # Cross-attention
        attn_output, _ = self.cross_attn(
            vlm_norm,
            fpn_features,
            fpn_features,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        
        # Gated residual connection
        vlm_hidden = vlm_hidden + self.gate * attn_output
        
        # FFN
        vlm_norm2 = self.norm2(vlm_hidden)
        vlm_hidden = vlm_hidden + self.ffn(vlm_norm2)
        
        return vlm_hidden


class AdaptiveLayerNormFusion(nn.Module):
    """
    Adaptive LayerNorm fusion: FPN features modulate LayerNorm parameters
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Base LayerNorm
        self.base_norm = nn.LayerNorm(hidden_size)
        
        # FPN-driven modulation
        self.modulate_gamma = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.modulate_beta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        vlm_hidden: torch.Tensor,
        fpn_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive LayerNorm modulated by FPN features
        
        Args:
            vlm_hidden: VLM hidden states
            fpn_features: FPN features for modulation
            attention_mask: Unused
            
        Returns:
            Adaptively normalized hidden states
        """
        # Base normalization
        vlm_norm = self.base_norm(vlm_hidden)
        
        # Aggregate FPN features for modulation (mean pooling)
        fpn_global = fpn_features.mean(dim=1)  # (B, hidden_size)
        
        # Compute modulation parameters
        gamma = self.modulate_gamma(fpn_global).unsqueeze(1)
        beta = self.modulate_beta(fpn_global).unsqueeze(1)
        
        # Apply adaptive normalization
        vlm_adapted = gamma * vlm_norm + beta
        
        return vlm_adapted


class GatedFusion(nn.Module):
    """
    Gated fusion: Learnable gate controls FPN feature injection
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Projection for FPN features
        self.fpn_proj = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        vlm_hidden: torch.Tensor,
        fpn_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply gated fusion
        
        Args:
            vlm_hidden: VLM hidden states
            fpn_features: FPN features
            attention_mask: Unused
            
        Returns:
            Fused hidden states
        """
        # Project FPN features
        fpn_proj = self.fpn_proj(fpn_features)
        
        # Compute gate (for each position)
        gate_input = torch.cat([vlm_hidden, fpn_proj], dim=-1)
        gate = self.gate(gate_input)
        
        # Gated fusion
        fused = gate * fpn_proj + (1 - gate) * vlm_hidden
        
        # Output projection with residual
        output = self.output_proj(self.norm(fused))
        output = output + vlm_hidden
        
        return output


class FeatureAlignmentModule(nn.Module):
    """
    Align FPN features with VLM token space
    """
    
    def __init__(
        self,
        fpn_channels: int,
        vlm_hidden_size: int,
        num_queries: int = 64,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, vlm_hidden_size))
        
        # Cross-attention to extract relevant features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vlm_hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FPN projection
        self.fpn_projection = nn.Sequential(
            nn.Conv2d(fpn_channels, vlm_hidden_size, 1),
            nn.GELU()
        )
        
        self.norm = nn.LayerNorm(vlm_hidden_size)
    
    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        """
        Extract aligned features from FPN
        
        Args:
            fpn_features: FPN features (B, C, H, W)
            
        Returns:
            Aligned features (B, num_queries, hidden_size)
        """
        B = fpn_features.shape[0]
        
        # Project and flatten FPN features
        fpn_proj = self.fpn_projection(fpn_features)
        fpn_flat = rearrange(fpn_proj, 'b c h w -> b (h w) c')
        
        # Expand queries
        queries = self.queries.expand(B, -1, -1)
        
        # Cross-attention: queries attend to FPN features
        aligned_features, _ = self.cross_attn(
            queries,
            fpn_flat,
            fpn_flat,
            need_weights=False
        )
        
        # Normalize
        aligned_features = self.norm(aligned_features)
        
        return aligned_features
