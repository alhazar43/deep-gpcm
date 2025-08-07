"""
Ordinal-aware attention mechanisms for GPCM models.
Implements modular attention components that understand ordinal relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Type
from abc import ABC, abstractmethod
import math


class BaseOrdinalAttention(nn.Module, ABC):
    """Base class for all ordinal-aware attention mechanisms."""
    
    def __init__(self, embed_dim: int, n_cats: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cats = n_cats
        self.dropout = dropout
        
    @abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention mechanism.
        
        Args:
            query: [batch, seq, embed_dim]
            key: [batch, seq, embed_dim]
            value: [batch, seq, embed_dim]
            mask: [batch, seq, seq] or None
            
        Returns:
            attended: [batch, seq, embed_dim]
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """Get configuration for this attention mechanism."""
        pass


class AttentionRegistry:
    """Registry for attention mechanisms - plugin pattern."""
    _registry: Dict[str, Type[BaseOrdinalAttention]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register attention mechanisms."""
        def decorator(attention_class: Type[BaseOrdinalAttention]):
            cls._registry[name] = attention_class
            return attention_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseOrdinalAttention]:
        """Get attention class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown attention mechanism: {name}")
        return cls._registry[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered attention mechanisms."""
        return list(cls._registry.keys())


@AttentionRegistry.register("ordinal_aware")
class OrdinalAwareSelfAttention(BaseOrdinalAttention):
    """Self-attention with ordinal distance penalty."""
    
    def __init__(self, embed_dim: int, n_cats: int, n_heads: int = 8,
                 dropout: float = 0.1, distance_penalty: float = 0.1):
        super().__init__(embed_dim, n_cats, dropout)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.distance_penalty = distance_penalty
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Ordinal distance embedding
        self.ordinal_embed = nn.Embedding(n_cats, n_heads)
        nn.init.normal_(self.ordinal_embed.weight, std=0.02)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def compute_ordinal_distances(self, responses: torch.Tensor) -> torch.Tensor:
        """Compute pairwise ordinal distances between responses.
        
        Args:
            responses: [batch, seq] response values (0 to n_cats-1)
            
        Returns:
            distances: [batch, seq, seq] normalized distances
        """
        # Expand for pairwise comparison
        r1 = responses.unsqueeze(2)  # [batch, seq, 1]
        r2 = responses.unsqueeze(1)  # [batch, 1, seq]
        
        # Compute absolute ordinal distance
        distances = torch.abs(r1 - r2).float()  # [batch, seq, seq]
        
        # Normalize by max possible distance
        distances = distances / (self.n_cats - 1)
        
        return distances
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply ordinal-aware self-attention.
        
        Args:
            query: [batch, seq, embed_dim]
            key: [batch, seq, embed_dim]
            value: [batch, seq, embed_dim]
            mask: [batch, seq, seq] or None
            responses: [batch, seq] response values for ordinal distance
            
        Returns:
            attended: [batch, seq, embed_dim]
        """
        batch_size, seq_len = query.shape[:2]
        
        # Project Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention: [batch, n_heads, seq, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply ordinal distance penalty if responses provided
        if responses is not None:
            distances = self.compute_ordinal_distances(responses)  # [batch, seq, seq]
            
            # Get ordinal embeddings for each head
            ordinal_weights = self.ordinal_embed(torch.zeros(1, dtype=torch.long, device=query.device))
            ordinal_weights = F.softplus(ordinal_weights).squeeze(0)  # [n_heads]
            
            # Apply distance penalty per head
            distances = distances.unsqueeze(1)  # [batch, 1, seq, seq]
            ordinal_weights = ordinal_weights.view(1, self.n_heads, 1, 1)
            
            # Subtract penalty from scores
            scores = scores - self.distance_penalty * ordinal_weights * distances
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Transpose back and concat heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output
    
    def get_config(self) -> Dict:
        return {
            "type": "ordinal_aware",
            "embed_dim": self.embed_dim,
            "n_cats": self.n_cats,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "distance_penalty": self.distance_penalty
        }


@AttentionRegistry.register("response_conditioned")
class ResponseConditionedAttention(BaseOrdinalAttention):
    """Attention conditioned on response patterns."""
    
    def __init__(self, embed_dim: int, n_cats: int, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, n_cats, dropout)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Response-specific key/value modulation
        self.response_key_modulation = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_cats)
        ])
        self.response_value_modulation = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_cats)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply response-conditioned attention.
        
        Args:
            query: [batch, seq, embed_dim]
            key: [batch, seq, embed_dim]
            value: [batch, seq, embed_dim]
            mask: [batch, seq, seq] or None
            responses: [batch, seq] response values for conditioning
            
        Returns:
            attended: [batch, seq, embed_dim]
        """
        batch_size, seq_len = query.shape[:2]
        
        # Base projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Apply response-specific modulation if responses provided
        if responses is not None:
            # Initialize modulated K and V
            K_modulated = torch.zeros_like(K)
            V_modulated = torch.zeros_like(V)
            
            # Apply response-specific transformations
            for r in range(self.n_cats):
                mask_r = (responses == r)  # [batch, seq]
                if mask_r.any():
                    # Get indices where response == r
                    K_r = self.response_key_modulation[r](key)
                    V_r = self.response_value_modulation[r](value)
                    
                    # Apply modulation where response matches
                    K_modulated[mask_r] = K_r[mask_r]
                    V_modulated[mask_r] = V_r[mask_r]
            
            # Combine with base projections
            K = K + K_modulated
            V = V + V_modulated
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output
    
    def get_config(self) -> Dict:
        return {
            "type": "response_conditioned",
            "embed_dim": self.embed_dim,
            "n_cats": self.n_cats,
            "n_heads": self.n_heads,
            "dropout": self.dropout
        }


class OrdinalAttentionPipeline(nn.Module):
    """Pipeline for composing multiple attention mechanisms."""
    
    def __init__(self, mechanisms: List[BaseOrdinalAttention]):
        super().__init__()
        self.mechanisms = nn.ModuleList(mechanisms)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention mechanisms in sequence."""
        output = query
        
        for mechanism in self.mechanisms:
            if hasattr(mechanism, 'forward'):
                # Check if mechanism accepts responses parameter
                import inspect
                sig = inspect.signature(mechanism.forward)
                if 'responses' in sig.parameters:
                    output = mechanism(output, key, value, mask, responses)
                else:
                    output = mechanism(output, key, value, mask)
        
        return output


@AttentionRegistry.register("ordinal_pattern")
class OrdinalPatternAttention(BaseOrdinalAttention):
    """Attention mechanism that learns ordinal response patterns."""
    
    def __init__(self, embed_dim: int, n_cats: int, n_heads: int = 8,
                 dropout: float = 0.1, pattern_size: int = 3):
        super().__init__(embed_dim, n_cats, dropout)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.pattern_size = pattern_size
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Pattern learning components
        self.pattern_embed = nn.Embedding(n_cats ** pattern_size, embed_dim)
        self.pattern_mixer = nn.Linear(embed_dim * 2, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def extract_patterns(self, responses: torch.Tensor) -> torch.Tensor:
        """Extract local ordinal patterns from response sequence.
        
        Args:
            responses: [batch, seq] response values
            
        Returns:
            patterns: [batch, seq] pattern indices
        """
        batch_size, seq_len = responses.shape
        device = responses.device
        
        # Pad for pattern extraction
        padded = F.pad(responses, (self.pattern_size - 1, 0), value=0)
        
        # Extract sliding window patterns
        patterns = []
        for i in range(seq_len):
            # Get pattern window
            window = padded[:, i:i+self.pattern_size]  # [batch, pattern_size]
            
            # Convert to pattern index (base n_cats number)
            pattern_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            for j in range(self.pattern_size):
                pattern_idx = pattern_idx * self.n_cats + window[:, j]
            
            patterns.append(pattern_idx)
        
        patterns = torch.stack(patterns, dim=1)  # [batch, seq]
        return patterns
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply pattern-aware attention."""
        batch_size, seq_len = query.shape[:2]
        
        # Project Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Mix with pattern information if responses provided
        if responses is not None:
            patterns = self.extract_patterns(responses)
            pattern_embeds = self.pattern_embed(patterns)  # [batch, seq, embed_dim]
            
            # Combine attended values with pattern embeddings
            combined = torch.cat([attended, pattern_embeds], dim=-1)
            attended = self.pattern_mixer(combined)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output
    
    def get_config(self) -> Dict:
        return {
            "type": "ordinal_pattern",
            "embed_dim": self.embed_dim,
            "n_cats": self.n_cats,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "pattern_size": self.pattern_size
        }


@AttentionRegistry.register("qwk_aligned")
class QWKAlignedAttention(BaseOrdinalAttention):
    """Attention mechanism optimized for Quadratic Weighted Kappa."""
    
    def __init__(self, embed_dim: int, n_cats: int, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, n_cats, dropout)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # QWK-specific components
        self.qwk_weight_matrix = self._create_qwk_weights()
        self.qwk_modulation = nn.Linear(1, n_heads)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def _create_qwk_weights(self) -> torch.Tensor:
        """Create QWK weight matrix for ordinal distances."""
        weights = torch.zeros(self.n_cats, self.n_cats)
        for i in range(self.n_cats):
            for j in range(self.n_cats):
                weights[i, j] = 1.0 - ((i - j) ** 2) / ((self.n_cats - 1) ** 2)
        return weights
    
    def compute_qwk_attention_bias(self, responses: torch.Tensor) -> torch.Tensor:
        """Compute attention bias based on QWK weights.
        
        Args:
            responses: [batch, seq] response values
            
        Returns:
            bias: [batch, n_heads, seq, seq] attention bias
        """
        batch_size, seq_len = responses.shape
        device = responses.device
        
        # Get QWK weights matrix
        qwk_weights = self.qwk_weight_matrix.to(device)
        
        # Compute pairwise QWK weights
        r1 = responses.unsqueeze(2)  # [batch, seq, 1]
        r2 = responses.unsqueeze(1)  # [batch, 1, seq]
        
        # Index into QWK weight matrix
        bias = qwk_weights[r1, r2]  # [batch, seq, seq]
        
        # Modulate across heads
        bias = bias.unsqueeze(-1)  # [batch, seq, seq, 1]
        head_weights = self.qwk_modulation(bias)  # [batch, seq, seq, n_heads]
        head_weights = head_weights.permute(0, 3, 1, 2)  # [batch, n_heads, seq, seq]
        
        return head_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply QWK-aligned attention."""
        batch_size, seq_len = query.shape[:2]
        
        # Project Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply QWK-based bias if responses provided
        if responses is not None:
            qwk_bias = self.compute_qwk_attention_bias(responses)
            scores = scores + qwk_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output
    
    def get_config(self) -> Dict:
        return {
            "type": "qwk_aligned",
            "embed_dim": self.embed_dim,
            "n_cats": self.n_cats,
            "n_heads": self.n_heads,
            "dropout": self.dropout
        }


@AttentionRegistry.register("hierarchical_ordinal")
class HierarchicalOrdinalAttention(BaseOrdinalAttention):
    """Hierarchical attention for ordinal categories."""
    
    def __init__(self, embed_dim: int, n_cats: int, n_heads: int = 8,
                 dropout: float = 0.1, n_levels: int = 2):
        super().__init__(embed_dim, n_cats, dropout)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_levels = n_levels
        
        # Create hierarchical groupings
        self.hierarchy = self._create_hierarchy()
        
        # Attention components for each level
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_levels)
        ])
        
        # Level aggregation
        self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        self.level_mixer = nn.Linear(embed_dim * n_levels, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _create_hierarchy(self) -> List[List[int]]:
        """Create hierarchical groupings of ordinal categories."""
        hierarchy = []
        
        # Level 1: Binary split (low vs high)
        mid = self.n_cats // 2
        hierarchy.append([
            list(range(mid)),  # Low responses
            list(range(mid, self.n_cats))  # High responses
        ])
        
        # Level 2: Individual categories
        hierarchy.append([[i] for i in range(self.n_cats)])
        
        return hierarchy
    
    def aggregate_by_hierarchy(self, embeddings: torch.Tensor, responses: torch.Tensor,
                              level: int) -> torch.Tensor:
        """Aggregate embeddings according to hierarchical level.
        
        Args:
            embeddings: [batch, seq, embed_dim]
            responses: [batch, seq]
            level: Hierarchy level
            
        Returns:
            aggregated: [batch, seq, embed_dim]
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Get groupings for this level
        groups = self.hierarchy[level]
        
        # Create aggregated embeddings
        aggregated = torch.zeros_like(embeddings)
        
        for group_idx, group_categories in enumerate(groups):
            # Create mask for this group
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            for cat in group_categories:
                mask = mask | (responses == cat)
            
            # Aggregate embeddings in this group
            if mask.any():
                group_embeds = embeddings[mask].mean(dim=0, keepdim=True)
                aggregated[mask] = group_embeds.expand(mask.sum(), -1)
        
        return aggregated
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply hierarchical ordinal attention."""
        # Process at each hierarchical level
        level_outputs = []
        
        for level in range(self.n_levels):
            # Apply attention at this level
            if level > 0 and responses is not None:
                # Aggregate keys and values by hierarchy
                key_agg = self.aggregate_by_hierarchy(key, responses, level-1)
                value_agg = self.aggregate_by_hierarchy(value, responses, level-1)
            else:
                key_agg = key
                value_agg = value
            
            # Apply attention
            attended, _ = self.level_attentions[level](
                query, key_agg, value_agg,
                attn_mask=mask
            )
            
            level_outputs.append(attended)
        
        # Combine level outputs
        if len(level_outputs) > 1:
            # Weighted combination
            weights = F.softmax(self.level_weights, dim=0)
            combined = torch.stack(level_outputs, dim=-1)  # [batch, seq, embed_dim, n_levels]
            combined = combined * weights.view(1, 1, 1, -1)
            combined = combined.sum(dim=-1)  # [batch, seq, embed_dim]
            
            # Alternative: concatenate and mix
            concat = torch.cat(level_outputs, dim=-1)  # [batch, seq, embed_dim * n_levels]
            output = self.level_mixer(concat)
        else:
            output = level_outputs[0]
        
        return output
    
    def get_config(self) -> Dict:
        return {
            "type": "hierarchical_ordinal",
            "embed_dim": self.embed_dim,
            "n_cats": self.n_cats,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "n_levels": self.n_levels
        }