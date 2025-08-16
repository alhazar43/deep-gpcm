"""
Adaptive Ordinal Embeddings with Attention-Modulated Weight Suppression

Research-driven solutions for reducing adjacent category interference in ordinal embeddings
while maintaining mathematical rigor for educational assessment applications.

Key Features:
- Learnable temperature sharpening (based on 2024 research)
- Attention-modulated suppression mechanisms
- Confidence-aware adaptive behavior
- Differential gating for adjacent vs distant categories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .embeddings import EmbeddingStrategy


class AttentionModulatedOrdinalEmbedding(nn.Module):
    """
    Solution 1: Attention-Modulated Ordinal Embedding with Learnable Temperature
    
    Combines learnable temperature sharpening with attention-based suppression
    to reduce adjacent category interference while preserving ordinal structure.
    """
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int = 64,
                 attention_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        
        # Learnable temperature for sharpening (inspired by 2024 research)
        self.temperature = nn.Parameter(torch.ones(n_heads) * 2.0)
        
        # Attention mechanism for context-aware suppression
        self.attention_module = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Context projection
        self.context_proj = nn.Linear(embed_dim, attention_dim)
        
        # Suppression score generator
        self.suppression_proj = nn.Linear(attention_dim, n_cats)
        
        # Direct embedding matrix
        self.direct_embed = nn.Linear(n_cats * n_questions, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        nn.init.kaiming_normal_(self.direct_embed.weight)
        nn.init.zeros_(self.direct_embed.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.xavier_uniform_(self.suppression_proj.weight)
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def _compute_base_weights(self, r_data: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute base triangular weights."""
        batch_size, seq_len = r_data.shape
        
        k_indices = torch.arange(self.n_cats, device=device).float()
        r_expanded = r_data.unsqueeze(-1).float()
        k_expanded = k_indices.unsqueeze(0).unsqueeze(0)
        
        distance = torch.abs(k_expanded - r_expanded) / (self.n_cats - 1)
        base_weights = torch.clamp(1.0 - distance, min=0.0)
        
        return base_weights
    
    def _apply_attention_suppression(self, base_weights: torch.Tensor, 
                                   context_embedding: torch.Tensor) -> torch.Tensor:
        """Apply attention-based suppression to reduce adjacent interference."""
        batch_size, seq_len, _ = context_embedding.shape
        
        # Project context for attention
        context_proj = self.context_proj(context_embedding)
        
        # Self-attention to identify suppression patterns
        attn_output, _ = self.attention_module(
            context_proj, context_proj, context_proj
        )
        
        # Generate suppression scores for each category
        suppression_scores = torch.sigmoid(self.suppression_proj(attn_output))
        
        return suppression_scores
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int, 
              context_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed with attention-modulated weight suppression.
        
        Args:
            q_data: Question one-hot vectors (batch_size, seq_len, n_questions)
            r_data: Response categories (batch_size, seq_len)
            n_questions: Number of questions
            n_cats: Number of categories
            context_embedding: Context for attention (batch_size, seq_len, embed_dim)
        """
        device = r_data.device
        batch_size, seq_len = r_data.shape
        
        # Compute base triangular weights
        base_weights = self._compute_base_weights(r_data, device)
        
        # Apply learnable temperature sharpening per head
        sharpened_weights_list = []
        for h in range(len(self.temperature)):
            temp_weights = F.softmax(base_weights / self.temperature[h], dim=-1)
            sharpened_weights_list.append(temp_weights.unsqueeze(-1))
        
        # Average across heads (or use attention to combine)
        sharpened_weights = torch.cat(sharpened_weights_list, dim=-1).mean(dim=-1)
        
        # Apply attention-based suppression if context available
        if context_embedding is not None:
            suppression_scores = self._apply_attention_suppression(
                base_weights, context_embedding
            )
            final_weights = sharpened_weights * (1.0 - 0.5 * suppression_scores)
        else:
            final_weights = sharpened_weights
        
        # Apply weights to question vectors
        weighted_q = final_weights.unsqueeze(-1) * q_data.unsqueeze(2)
        flattened = weighted_q.view(batch_size, seq_len, n_cats * n_questions)
        
        # Direct embedding to target dimension
        embedded = self.direct_embed(flattened)
        
        return embedded


class ConfidenceAwareOrdinalEmbedding(nn.Module):
    """
    Solution 2: Confidence-Aware Adaptive Suppression
    
    Adapts weight sharpening based on confidence estimation from response history
    and question difficulty, providing personalized suppression.
    """
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int = 64,
                 confidence_dim: int = 32):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(embed_dim + n_questions, confidence_dim),
            nn.ReLU(),
            nn.Linear(confidence_dim, 1),
            nn.Sigmoid()
        )
        
        # Sharpness factor (learnable)
        self.sharpness_factor = nn.Parameter(torch.tensor(3.0))
        
        # Direct embedding matrix
        self.direct_embed = nn.Linear(n_cats * n_questions, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for layer in self.confidence_estimator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.kaiming_normal_(self.direct_embed.weight)
        nn.init.zeros_(self.direct_embed.bias)
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def _estimate_confidence(self, response_history: torch.Tensor, 
                           question_features: torch.Tensor) -> torch.Tensor:
        """Estimate confidence from response history and question features."""
        # Combine response history and question features
        combined_features = torch.cat([response_history, question_features], dim=-1)
        confidence = self.confidence_estimator(combined_features)
        return confidence
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int,
              response_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed with confidence-aware adaptive suppression.
        """
        device = r_data.device
        batch_size, seq_len = r_data.shape
        
        # Compute base triangular weights
        k_indices = torch.arange(n_cats, device=device).float()
        r_expanded = r_data.unsqueeze(-1).float()
        k_expanded = k_indices.unsqueeze(0).unsqueeze(0)
        
        distance = torch.abs(k_expanded - r_expanded) / (n_cats - 1)
        base_weights = torch.clamp(1.0 - distance, min=0.0)
        
        # Estimate confidence if history available
        if response_history is not None:
            confidence = self._estimate_confidence(response_history, q_data)
            alpha = 1.0 + confidence * self.sharpness_factor
        else:
            alpha = torch.ones_like(base_weights[:, :, 0:1]) * 2.0
        
        # Apply adaptive sharpening
        adaptive_weights = base_weights ** alpha.unsqueeze(-1)
        
        # Normalize to maintain probability distribution
        normalized_weights = adaptive_weights / (
            adaptive_weights.sum(dim=-1, keepdim=True) + 1e-8
        )
        
        # Apply weights to question vectors
        weighted_q = normalized_weights.unsqueeze(-1) * q_data.unsqueeze(2)
        flattened = weighted_q.view(batch_size, seq_len, n_cats * n_questions)
        
        # Direct embedding
        embedded = self.direct_embed(flattened)
        
        return embedded


class GatedOrdinalEmbedding(nn.Module):
    """
    Solution 3: Gated Ordinal Embedding with Differential Suppression
    
    Uses learnable gating to apply differential suppression to adjacent vs 
    distant categories, inspired by M-DGSA 2024 research.
    """
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int = 64,
                 gate_dim: int = 32):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        
        # Gate network for differential suppression
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 2),  # Adjacent and distant suppression
            nn.Sigmoid()
        )
        
        # Direct embedding matrix
        self.direct_embed = nn.Linear(n_cats * n_questions, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.kaiming_normal_(self.direct_embed.weight)
        nn.init.zeros_(self.direct_embed.bias)
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int,
              context_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed with differential gated suppression.
        """
        device = r_data.device
        batch_size, seq_len = r_data.shape
        
        # Compute base triangular weights
        k_indices = torch.arange(n_cats, device=device).float()
        r_expanded = r_data.unsqueeze(-1).float()
        k_expanded = k_indices.unsqueeze(0).unsqueeze(0)
        
        distance = torch.abs(k_expanded - r_expanded) / (n_cats - 1)
        base_weights = torch.clamp(1.0 - distance, min=0.0)
        
        # Apply differential gating if context available
        if context_embedding is not None:
            # Generate suppression gates
            gate_scores = self.gate_network(context_embedding)
            adjacent_suppression = gate_scores[:, :, 0:1]
            distant_suppression = gate_scores[:, :, 1:2]
            
            # Create adjacency masks
            adjacency_mask = (torch.abs(k_expanded - r_expanded) <= 1.0).float()
            distant_mask = 1.0 - adjacency_mask
            
            # Apply differential suppression
            suppressed_weights = base_weights * (
                adjacency_mask * (1.0 - adjacent_suppression) + 
                distant_mask * (1.0 - distant_suppression)
            )
            
            # Normalize
            final_weights = suppressed_weights / (
                suppressed_weights.sum(dim=-1, keepdim=True) + 1e-8
            )
        else:
            final_weights = base_weights
        
        # Apply weights to question vectors
        weighted_q = final_weights.unsqueeze(-1) * q_data.unsqueeze(2)
        flattened = weighted_q.view(batch_size, seq_len, n_cats * n_questions)
        
        # Direct embedding
        embedded = self.direct_embed(flattened)
        
        return embedded


# Factory functions for easy integration
def create_attention_modulated_embedding(n_questions: int, n_cats: int = 4, 
                                       **kwargs) -> AttentionModulatedOrdinalEmbedding:
    """Create attention-modulated ordinal embedding."""
    return AttentionModulatedOrdinalEmbedding(n_questions, n_cats, **kwargs)


def create_confidence_aware_embedding(n_questions: int, n_cats: int = 4, 
                                    **kwargs) -> ConfidenceAwareOrdinalEmbedding:
    """Create confidence-aware ordinal embedding."""
    return ConfidenceAwareOrdinalEmbedding(n_questions, n_cats, **kwargs)


def create_gated_embedding(n_questions: int, n_cats: int = 4, 
                          **kwargs) -> GatedOrdinalEmbedding:
    """Create gated ordinal embedding."""
    return GatedOrdinalEmbedding(n_questions, n_cats, **kwargs)