"""IRT parameter extraction layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class IRTParameterExtractor(nn.Module):
    """Extract IRT parameters (theta, alpha, beta) from neural features."""
    
    def __init__(self, input_dim: int, n_cats: int, ability_scale: float = 3.0, 
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