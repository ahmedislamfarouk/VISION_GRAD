"""
VLM Module for HBU-YOLO-VLM
Based on LLaVA architecture
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import LlamaForCausalLM, CLIPVisionModel, LlamaTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration


class VLMEncoder(nn.Module):
    """
    Vision encoder for VLM (CLIP-based)
    """
    
    def __init__(self, vision_model_name: str = 'openai/clip-vit-large-patch14'):
        super().__init__()
        
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.hidden_size = self.vision_encoder.config.hidden_size
        
        # Freeze vision encoder by default
        self.freeze_vision_encoder()
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_vision_encoder(self, last_n_layers: int = 0):
        """Unfreeze last N layers of vision encoder"""
        self.freeze_vision_encoder()
        if last_n_layers > 0:
            layers = self.vision_encoder.encoder.layers
            for layer in layers[-last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Visual features (B, num_patches, hidden_size)
        """
        outputs = self.vision_encoder(images, output_hidden_states=True)
        
        # Use last hidden state
        visual_features = outputs.hidden_states[-1]
        
        return visual_features


class VLMDecoder(nn.Module):
    """
    Language decoder for VLM (Llama-based)
    """
    
    def __init__(self, llm_model_name: str = 'lmsys/vicuna-7b-v1.5'):
        super().__init__()
        
        self.llm = LlamaForCausalLM.from_pretrained(llm_model_name)
        self.hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate text or compute loss
        
        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            inputs_embeds: Embedded inputs (B, seq_len, hidden_size)
            labels: Labels for loss computation (B, seq_len)
            
        Returns:
            logits: Output logits (B, seq_len, vocab_size)
            loss: Loss if labels provided
        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True
        )
        
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        
        return logits, loss


class LLaVAModule(nn.Module):
    """
    LLaVA-style VLM module with vision-language projection
    """
    
    def __init__(
        self,
        vision_model: str = 'openai/clip-vit-large-patch14',
        llm_model: str = 'lmsys/vicuna-7b-v1.5',
        projection_type: str = 'linear'
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VLMEncoder(vision_model)
        
        # Language decoder
        self.llm_decoder = VLMDecoder(llm_model)
        
        # Vision-language projection
        vision_hidden_size = self.vision_encoder.hidden_size
        llm_hidden_size = self.llm_decoder.hidden_size
        
        if projection_type == 'linear':
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projection_type == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        elif projection_type == 'resampler':
            # Perceiver resampler for efficient token compression
            self.projector = PerceiverResampler(
                input_dim=vision_hidden_size,
                output_dim=llm_hidden_size,
                num_latents=64
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        # Tokenizer (will be loaded with LLM)
        self.tokenizer = None
    
    def load_tokenizer(self, tokenizer_name: str = 'lmsys/vicuna-7b-v1.5'):
        """Load tokenizer"""
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'right'
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual embeddings
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Visual embeddings (B, num_tokens, llm_hidden_size)
        """
        visual_features = self.vision_encoder(images)
        visual_embeddings = self.projector(visual_features)
        
        return visual_embeddings
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            images: Input images (B, 3, H, W)
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            labels: Labels for loss computation (B, seq_len)
            visual_embeddings: Pre-computed visual embeddings
            
        Returns:
            logits: Output logits
            loss: Loss if labels provided
        """
        # Encode images if provided
        if images is not None:
            visual_embeddings = self.encode_images(images)
        
        # Embed input tokens
        if input_ids is not None:
            inputs_embeds = self.llm_decoder.llm.model.embed_tokens(input_ids)
            
            # Merge visual embeddings with text embeddings
            if visual_embeddings is not None:
                # Assume visual embeddings come before text
                num_visual_tokens = visual_embeddings.shape[1]
                
                # Create combined embeddings
                B = input_ids.shape[0]
                text_len = input_ids.shape[1]
                
                # Placeholder for merging logic (simplified)
                inputs_embeds = torch.cat([visual_embeddings, inputs_embeds[:, -text_len:]], dim=1)
        
        # Pass through LLM
        logits, loss = self.llm_decoder(
            inputs_embeds=inputs_embeds if inputs_embeds is not None else None,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return logits, loss


class PerceiverResampler(nn.Module):
    """
    Perceiver resampler for efficient token compression
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_latents: int = 64):
        super().__init__()
        
        self.num_latents = num_latents
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, input_dim))
        
        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resample input features to fixed number of latents
        
        Args:
            x: Input features (B, num_tokens, input_dim)
            
        Returns:
            Resampled features (B, num_latents, output_dim)
        """
        B = x.shape[0]
        
        # Expand latents to batch size
        latents = self.latents.expand(B, -1, -1)
        
        # Cross-attention
        latents_norm = self.norm1(latents)
        x_norm = self.norm2(x)
        
        attn_output, _ = self.cross_attn(
            latents_norm,
            x_norm,
            x_norm,
            need_weights=False
        )
        
        # Residual connection
        latents = latents + attn_output
        
        # Output projection
        output = self.output_proj(latents)
        
        return output
