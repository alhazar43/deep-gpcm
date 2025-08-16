"""
Simplified Ordinal Attention GPCM Model

This model fixes ONLY the dimensional projection issue in attn_gpcm_linear
while keeping everything else as close as possible to the working implementation.

Key Fix: Direct embedding to target dimension without projection bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .attention_gpcm import EnhancedAttentionGPCM


class SimpleOrdinalLinearDecayEmbedding(nn.Module):
    """
    Simplified direct linear decay embedding to target dimension.
    
    This eliminates the problematic projection from 800→64 that causes
    gradient explosion in attn_gpcm_linear.
    """
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int = 64):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        
        # Direct linear transformation per question (no expansion to K*Q)
        self.question_embed = nn.Linear(n_questions, embed_dim)
        
        # Response weighting layer
        self.response_weights = nn.Linear(n_cats, 1)
        
        # Initialize conservatively
        nn.init.xavier_uniform_(self.question_embed.weight, gain=0.5)
        nn.init.zeros_(self.question_embed.bias)
        nn.init.xavier_uniform_(self.response_weights.weight, gain=0.5)
        nn.init.zeros_(self.response_weights.bias)
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """
        Direct embedding with linear decay weights.
        
        Key difference: No expansion to K*Q dimensions, direct embedding.
        """
        batch_size, seq_len = r_data.shape
        device = r_data.device
        
        # Process each timestep
        embeddings = []
        for t in range(seq_len):
            q_t = q_data[:, t, :]  # (batch_size, n_questions)
            r_t = r_data[:, t]     # (batch_size,)
            
            # Create linear decay weights (same as original)
            k_indices = torch.arange(n_cats, device=device).float()
            r_expanded = r_t.unsqueeze(-1).float()  # (batch_size, 1)
            distance = torch.abs(k_indices.unsqueeze(0) - r_expanded) / (n_cats - 1)
            weights = torch.clamp(1.0 - distance, min=0.0)  # (batch_size, n_cats)
            
            # Get response weight for this response
            response_weight = torch.sigmoid(self.response_weights(weights))  # (batch_size, 1)
            
            # Apply to question vector and embed directly
            weighted_q = q_t * response_weight  # Broadcasting: (batch_size, n_questions) * (batch_size, 1)
            embed_t = self.question_embed(weighted_q)  # (batch_size, embed_dim)
            
            embeddings.append(embed_t)
        
        # Stack timesteps
        result = torch.stack(embeddings, dim=1)  # (batch_size, seq_len, embed_dim)
        return result


class SimpleOrdinalAttentionGPCM(EnhancedAttentionGPCM):
    """
    Simplified ordinal attention model that fixes only the embedding projection issue.
    
    Based on EnhancedAttentionGPCM but replaces the problematic embedding strategy
    with a direct embedding approach that eliminates the 800→64 bottleneck.
    """
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 ability_scale: float = 1.0, dropout_rate: float = 0.1, **kwargs):
        
        # Initialize parent but override the embedding
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
            ability_scale=ability_scale,
            dropout_rate=dropout_rate,
            use_learnable_embedding=False  # Use linear embedding strategy
        )
        
        # Override model name
        self.model_name = "simple_ordinal_attention_gpcm"
        
        # Replace the problematic embedding with direct embedding
        self.simple_ordinal_embedding = SimpleOrdinalLinearDecayEmbedding(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=embed_dim
        )
        
        # Update embedding projection to be pass-through since we're already at target dim
        from ..components.attention_layers import EmbeddingProjection
        self.embedding_projection = EmbeddingProjection(
            input_dim=embed_dim,  # Already at target dimension
            output_dim=embed_dim,
            dropout_rate=dropout_rate
        )
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings using simplified direct approach."""
        # Get question one-hot vectors
        q_one_hot = F.one_hot(questions, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Use simplified ordinal embedding (direct to target dimension)
        gpcm_embeds = self.simple_ordinal_embedding.embed(
            q_one_hot, responses, self.n_questions, self.n_cats
        )
        
        # Pass through projection (should be minimal since we're already at target dim)
        projected_embeds = self.embedding_projection(gpcm_embeds)
        
        return projected_embeds
    
    def get_model_info(self):
        """Get model information."""
        info = super().get_model_info()
        info['model_name'] = self.model_name
        info['architecture'] = 'Simplified Ordinal Attention GPCM'
        info['fixes'] = 'Direct embedding eliminates 800→64 projection bottleneck'
        return info


# Factory helper function
def create_simple_ordinal_attention_gpcm(n_questions: int, n_cats: int = 4, **kwargs) -> SimpleOrdinalAttentionGPCM:
    """Create SimpleOrdinalAttentionGPCM model with safe defaults."""
    return SimpleOrdinalAttentionGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        **kwargs
    )