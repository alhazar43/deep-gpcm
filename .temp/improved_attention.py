"""Improved Enhanced AttentionGPCM with modular architectural improvements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .attention_enhanced import EnhancedAttentionGPCM


class RefinementGate(nn.Module):
    """Simple gating mechanism for controlled feature updates."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Apply gated update."""
        gate_value = self.gate(x)
        return x + gate_value * update


class MemoryFusion(nn.Module):
    """Fuses attention and memory features."""
    
    def __init__(self, embed_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, attn_features: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        """Fuse attention and memory features."""
        combined = torch.cat([attn_features, memory_features], dim=-1)
        return self.fusion(combined)


class ImprovedEnhancedAttentionGPCM(EnhancedAttentionGPCM):
    """Improved version with key architectural enhancements from old AKVMN."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 2.0,
                 dropout_rate: float = 0.1):
        
        # Initialize parent
        super().__init__(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=embed_dim,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            n_heads=n_heads,
            n_cycles=n_cycles,
            embedding_strategy=embedding_strategy,
            ability_scale=ability_scale,
            dropout_rate=dropout_rate
        )
        
        # Override model name
        self.model_name = "improved_akvmn_gpcm"
        
        # Replace summary network with deeper architecture (key difference from old AKVMN)
        summary_input_dim = value_dim + key_dim
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.ReLU(),  # ReLU instead of Tanh
            nn.Dropout(dropout_rate),
            nn.Linear(final_fc_dim, final_fc_dim),  # Additional layer
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize new summary network
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Add modular components for each cycle
        self.memory_fusions = nn.ModuleList([
            MemoryFusion(embed_dim, dropout_rate) for _ in range(n_cycles)
        ])
        
        self.refinement_gates = nn.ModuleList([
            RefinementGate(embed_dim) for _ in range(n_cycles)
        ])
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor) -> torch.Tensor:
        """Enhanced processing with memory fusion and refinement gates."""
        refined_embeds = gpcm_embeds
        
        for cycle in range(self.n_cycles):
            # Apply attention refinement (from parent)
            attn_output = self.attention_refinement(refined_embeds)
            
            # Create memory context from question embeddings
            # This simulates memory interaction without modifying core DKVMN
            memory_context = q_embeds.unsqueeze(2).expand(-1, -1, self.embed_dim, -1).mean(dim=-1)
            
            # Fuse attention and memory
            fused_features = self.memory_fusions[cycle](attn_output, memory_context)
            
            # Apply gated update
            refined_embeds = self.refinement_gates[cycle](refined_embeds, fused_features)
        
        return refined_embeds
    
    def get_model_info(self):
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'name': self.model_name,
            'has_memory_fusion': True,
            'has_refinement_gates': True,
            'summary_network_depth': 2,
            'activation': 'ReLU'
        })
        return info