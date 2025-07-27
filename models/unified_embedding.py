"""
Unified Embedding Module for Deep Integration
Compact unified embedding space for both memory and attention operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedEmbedding(nn.Module):
    """
    Unified embedding space for both memory and attention operations.
    
    Parameter count: 14,500 parameters
    Preserves linear decay for GPCM compatibility.
    """
    
    def __init__(self, n_questions: int, embed_dim: int = 64, n_cats: int = 4):
        super().__init__()
        
        self.n_questions = n_questions
        self.embed_dim = embed_dim
        self.n_cats = n_cats
        
        # Question embeddings for attention and memory
        self.question_embed = nn.Embedding(n_questions + 1, embed_dim, padding_idx=0)
        
        # Linear decay weights for GPCM compatibility
        self.decay_weights = nn.Parameter(torch.ones(n_cats))
        
        # Response projection to maintain embedding dimension
        self.response_projection = nn.Linear(n_cats, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for optimal performance."""
        nn.init.kaiming_normal_(self.question_embed.weight)
        nn.init.ones_(self.decay_weights)
        nn.init.kaiming_normal_(self.response_projection.weight)
        nn.init.zeros_(self.response_projection.bias)
    
    def forward(self, q_data, r_data):
        """
        Forward pass creating unified embeddings.
        
        Args:
            q_data: Question sequence [batch_size, seq_len]
            r_data: Response sequence [batch_size, seq_len]
            
        Returns:
            unified_embed: Unified embeddings [batch_size, seq_len, embed_dim]
        """
        # Question embeddings
        q_embed = self.question_embed(q_data)  # [batch_size, seq_len, embed_dim]
        
        # Response embeddings with preserved linear decay
        r_onehot = F.one_hot(r_data, num_classes=self.n_cats).float()
        decay_weights = F.softmax(self.decay_weights, dim=0)
        r_embed = self.response_projection(r_onehot * decay_weights)
        
        # Combine question and response embeddings
        unified_embed = q_embed + r_embed
        
        return unified_embed