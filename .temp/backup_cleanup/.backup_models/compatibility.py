"""Compatibility wrappers for legacy interfaces."""

import torch
import torch.nn as nn
from typing import Tuple

from .dkvmn_gpcm import DKVMNGPCM as NewDKVMNGPCM
from .attention_dkvmn_gpcm import AttentionDKVMNGPCM as NewAttentionDKVMNGPCM


class BaselineGPCM(nn.Module):
    """Compatibility wrapper for baseline GPCM model."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, 
                 memory_size: int = 50, key_dim: int = 50, 
                 value_dim: int = 200, final_fc_dim: int = 50,
                 embedding_strategy: str = "linear_decay",
                 prediction_method: str = "cumulative"):
        super().__init__()
        
        self.model_name = "baseline"
        self.n_questions = n_questions
        self.n_cats = n_cats
        
        # Use new modular implementation
        self.gpcm_model = NewDKVMNGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            embedding_strategy=embedding_strategy
        )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through baseline model."""
        return self.gpcm_model(questions, responses)
    
    def get_model_info(self):
        """Get model information."""
        return {
            "name": self.model_name,
            "type": "baseline",
            "parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "DKVMN-GPCM",
            "features": ["Dynamic Memory", "Optimal Embedding", "Polytomous Support"]
        }


class AKVMNGPCM(nn.Module):
    """Compatibility wrapper for attention DKVMN GPCM model."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay"):
        super().__init__()
        
        self.model_name = "akvmn_gpcm"
        self.n_questions = n_questions
        self.n_cats = n_cats
        
        # Use new modular implementation
        self.attention_model = NewAttentionDKVMNGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=embed_dim,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            n_heads=n_heads,
            n_cycles=n_cycles,
            embedding_strategy=embedding_strategy
        )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through attention model."""
        return self.attention_model(questions, responses)
    
    def get_model_info(self):
        """Get model information."""
        return {
            "name": self.model_name,
            "type": "attention_dkvmn_gpcm", 
            "parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "Attention-DKVMN-GPCM",
            "features": ["Dynamic Memory", "Multi-Head Attention", "Iterative Refinement", "Polytomous Support"]
        }


def create_baseline_gpcm(**kwargs):
    """Factory function to create baseline GPCM model."""
    return BaselineGPCM(**kwargs)