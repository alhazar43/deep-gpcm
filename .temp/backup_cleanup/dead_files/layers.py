"""Additional neural network layers for DKVMN-GPCM models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionRefinementModule(nn.Module):
    """Multi-head attention with iterative refinement for embedding enhancement."""
    
    def __init__(self, embed_dim: int, n_heads: int = 4, n_cycles: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.dropout_rate = dropout_rate
        
        # Multi-head attention for each cycle
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(n_cycles)
        ])
        
        # Feature fusion for each cycle
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(n_cycles)
        ])
        
        # Cycle normalization
        self.cycle_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_cycles)
        ])
        
        # Refinement gates (to control update magnitude)
        self.refinement_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            ) for _ in range(n_cycles)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for numerical stability."""
        # Refinement components
        for fusion in self.fusion_layers:
            for module in fusion:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        
        for gate in self.refinement_gates:
            for module in gate:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply iterative refinement using multi-head attention.
        
        Args:
            embeddings: Input embeddings, shape (batch_size, seq_len, embed_dim)
            
        Returns:
            refined_embeddings: Enhanced embeddings, shape (batch_size, seq_len, embed_dim)
        """
        # Start with initial embeddings
        current_embed = embeddings
        
        for cycle in range(self.n_cycles):
            # Multi-head self-attention
            attn_output, _ = self.attention_layers[cycle](
                current_embed, current_embed, current_embed
            )
            
            # Feature fusion
            fused_input = torch.cat([current_embed, attn_output], dim=-1)
            fused_output = self.fusion_layers[cycle](fused_input)
            
            # Apply refinement gate
            gate = self.refinement_gates[cycle](current_embed)
            refined_output = gate * fused_output + (1 - gate) * current_embed
            
            # Update current embedding
            current_embed = refined_output
            
            # Cycle normalization
            current_embed = self.cycle_norms[cycle](current_embed)
        
        return current_embed


class EmbeddingProjection(nn.Module):
    """Project embeddings from one dimension to another."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings."""
        return self.projection(x)