"""
Integration module for ordinal-aware attention in GPCM models.
Provides backward-compatible integration with existing AttentionGPCM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .ordinal_attention import BaseOrdinalAttention, AttentionRegistry, OrdinalAttentionPipeline
from .attention_layers import AttentionRefinementModule


class OrdinalAwareAttentionRefinement(nn.Module):
    """Drop-in replacement for AttentionRefinementModule with ordinal awareness."""
    
    def __init__(self, embed_dim: int, n_heads: int = 4, n_cycles: int = 2,
                 dropout_rate: float = 0.1, n_cats: int = 4,
                 attention_types: Optional[List[str]] = None,
                 use_legacy: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.dropout_rate = dropout_rate
        self.n_cats = n_cats
        self.use_legacy = use_legacy
        
        if use_legacy:
            # Fall back to original implementation
            self.legacy_module = AttentionRefinementModule(
                embed_dim, n_heads, n_cycles, dropout_rate
            )
        else:
            # Build ordinal-aware attention pipeline
            if attention_types is None:
                attention_types = ["ordinal_aware"]
            
            mechanisms = []
            for attn_type in attention_types:
                attn_class = AttentionRegistry.get(attn_type)
                mechanism = attn_class(
                    embed_dim=embed_dim,
                    n_cats=n_cats,
                    n_heads=n_heads,
                    dropout=dropout_rate
                )
                mechanisms.append(mechanism)
            
            self.attention_pipeline = OrdinalAttentionPipeline(mechanisms)
            
            # Gating mechanism for iterative refinement
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            
            # Layer norm
            self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, embeddings: torch.Tensor, 
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply ordinal-aware attention refinement.
        
        Args:
            embeddings: [batch, seq, embed_dim]
            responses: [batch, seq] response values (optional)
            
        Returns:
            refined: [batch, seq, embed_dim]
        """
        if self.use_legacy:
            return self.legacy_module(embeddings)
        
        # Iterative refinement with ordinal awareness
        refined = embeddings
        
        for cycle in range(self.n_cycles):
            # Apply ordinal-aware attention
            attended = self.attention_pipeline(
                query=refined,
                key=refined,
                value=refined,
                mask=None,  # No mask for self-attention within sequence
                responses=responses
            )
            
            # Gated update
            gate_input = torch.cat([refined, attended], dim=-1)
            gate_values = self.gate(gate_input)
            
            # Update with residual connection
            refined = gate_values * attended + (1 - gate_values) * refined
            refined = self.layer_norm(refined)
        
        return refined


class EnhancedAttentionGPCM(nn.Module):
    """Enhanced AttentionGPCM with ordinal-aware attention.
    
    This is a wrapper that can be used to upgrade existing AttentionGPCM
    models with ordinal awareness while maintaining backward compatibility.
    """
    
    def __init__(self, base_model: nn.Module, n_cats: int = 4,
                 attention_types: Optional[List[str]] = None):
        super().__init__()
        
        self.base_model = base_model
        self.n_cats = n_cats
        
        # Replace the attention refinement module
        if hasattr(base_model, 'attention_refinement'):
            embed_dim = base_model.embed_dim
            n_heads = base_model.n_heads
            n_cycles = base_model.n_cycles
            dropout_rate = base_model.dropout_rate
            
            # Create new ordinal-aware module
            self.base_model.attention_refinement = OrdinalAwareAttentionRefinement(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_cycles=n_cycles,
                dropout_rate=dropout_rate,
                n_cats=n_cats,
                attention_types=attention_types,
                use_legacy=False
            )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor):
        """Forward pass with response information passed to attention."""
        # Store responses for attention module to access
        self._current_responses = responses
        
        # Call base model forward
        return self.base_model(questions, responses)
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor) -> torch.Tensor:
        """Override to pass responses to attention refinement."""
        if hasattr(self, '_current_responses') and hasattr(self.base_model.attention_refinement, 'forward'):
            # Pass responses to ordinal-aware attention
            return self.base_model.attention_refinement(gpcm_embeds, self._current_responses)
        else:
            # Fall back to base implementation
            return self.base_model.process_embeddings(gpcm_embeds, q_embeds)


def create_ordinal_attention(attention_type: str, embed_dim: int, n_cats: int,
                           n_heads: int = 8, dropout: float = 0.1,
                           **kwargs) -> BaseOrdinalAttention:
    """Factory function to create ordinal attention mechanisms.
    
    Args:
        attention_type: Type of attention ("ordinal_aware", "response_conditioned", etc.)
        embed_dim: Embedding dimension
        n_cats: Number of response categories
        n_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        Attention mechanism instance
    """
    attn_class = AttentionRegistry.get(attention_type)
    return attn_class(
        embed_dim=embed_dim,
        n_cats=n_cats,
        n_heads=n_heads,
        dropout=dropout,
        **kwargs
    )