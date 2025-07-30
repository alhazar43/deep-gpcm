"""Neural network layers for DKVMN-GPCM models including IRT parameter extraction."""

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


class IRTParameterExtractor(nn.Module):
    """Extract IRT parameters (theta, alpha, beta) from neural features."""
    
    def __init__(self, input_dim: int, n_cats: int, ability_scale: float = 1.0, 
                 use_discrimination: bool = True, dropout_rate: float = 0.0, question_dim: int = None):
        super().__init__()
        self.n_cats = n_cats
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.question_dim = question_dim or input_dim
        
        # Student ability network (theta)
        self.ability_network = nn.Linear(input_dim, 1)
        
        # Question difficulty thresholds (beta) - K-1 thresholds per question
        self.threshold_network = nn.Sequential(
            nn.Linear(self.question_dim, n_cats - 1),
            nn.Tanh()
        )
        
        # Discrimination parameter (alpha) - optional
        if use_discrimination:
            discrim_input_dim = input_dim + self.question_dim
            self.discrimination_network = nn.Sequential(
                nn.Linear(discrim_input_dim, 1),
                nn.Softplus()  # Positive constraint
            )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        nn.init.kaiming_normal_(self.ability_network.weight)
        nn.init.constant_(self.ability_network.bias, 0)
        
        for module in self.threshold_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        if self.use_discrimination:
            for module in self.discrimination_network:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor, 
                question_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract IRT parameters from features.
        
        Args:
            features: Neural features for ability, shape (batch_size, seq_len, input_dim)
            question_features: Question features for discrimination/thresholds, 
                             shape (batch_size, seq_len, question_dim)
            
        Returns:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            beta: Threshold parameters, shape (batch_size, seq_len, n_cats-1)
        """
        # Apply dropout
        features = self.dropout(features)
        
        # Extract ability (theta)
        theta = self.ability_network(features).squeeze(-1) * self.ability_scale
        
        # Extract thresholds (beta) - use question features if provided
        threshold_input = question_features if question_features is not None else features
        beta = self.threshold_network(threshold_input)
        
        # Extract discrimination (alpha) if enabled
        if self.use_discrimination:
            if question_features is not None:
                discrim_input = torch.cat([features, question_features], dim=-1)
            else:
                discrim_input = torch.cat([features, features], dim=-1)  # Use features twice if no question features
            alpha = self.discrimination_network(discrim_input).squeeze(-1)
        else:
            alpha = torch.ones_like(theta)
        
        return theta, alpha, beta


class GPCMProbabilityLayer(nn.Module):
    """Compute GPCM response probabilities."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, theta: torch.Tensor, alpha: torch.Tensor, 
                beta: torch.Tensor) -> torch.Tensor:
        """Calculate GPCM response probabilities.
        
        Args:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            beta: Threshold parameters, shape (batch_size, seq_len, K-1)
            
        Returns:
            probs: GPCM probabilities, shape (batch_size, seq_len, K)
        """
        batch_size, seq_len = theta.shape
        K = beta.shape[-1] + 1  # Number of categories
        
        # Compute cumulative logits
        cum_logits = torch.zeros(batch_size, seq_len, K, device=theta.device)
        cum_logits[:, :, 0] = 0  # First category baseline
        
        # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - beta_h)
        for k in range(1, K):
            cum_logits[:, :, k] = torch.sum(
                alpha.unsqueeze(-1) * (theta.unsqueeze(-1) - beta[:, :, :k]), 
                dim=-1
            )
        
        # Convert to probabilities via softmax
        probs = F.softmax(cum_logits, dim=-1)
        return probs