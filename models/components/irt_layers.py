import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional



class IRTParameterExtractor(nn.Module):
    """Extract IRT parameters (theta, alpha, beta) from neural features."""
    
    def __init__(self, input_dim: int, n_cats: int, ability_scale: float = 1.0, 
                 use_discrimination: bool = True, dropout_rate: float = 0.0, question_dim: int = None,
                 use_research_beta: bool = True, use_bounded_beta: bool = False, 
                 conservative_research: bool = False):
        super().__init__()
        self.n_cats = n_cats
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.use_research_beta = use_research_beta
        self.use_bounded_beta = use_bounded_beta
        self.conservative_research = conservative_research
        self.question_dim = question_dim or input_dim
        
        # Student ability network (theta)
        self.ability_network = nn.Linear(input_dim, 1)
        
        # Question difficulty thresholds (beta) - K-1 thresholds per question
        if self.use_research_beta:
            # RESEARCH-BASED SOLUTION: Monotonic Gap Parameterization
            # Unconstrained base threshold + positive gaps for monotonicity
            self.threshold_base = nn.Linear(self.question_dim, 1)
            if self.n_cats > 2:
                self.threshold_gaps = nn.Linear(self.question_dim, self.n_cats - 2)
        else:
            # STABLE IMPLEMENTATION: Tanh-constrained for attention models
            self.threshold_network = nn.Sequential(
                nn.Linear(self.question_dim, n_cats - 1),
                nn.Tanh()
            )
        
        # Discrimination parameter (alpha) - optional
        if use_discrimination:
            discrim_input_dim = input_dim + self.question_dim
            # Direct linear layer for lognormal(0, 0.3) mapping via exp(0.3 * x)
            self.discrimination_network = nn.Linear(discrim_input_dim, 1)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        nn.init.kaiming_normal_(self.ability_network.weight)
        nn.init.constant_(self.ability_network.bias, 0)
        
        if self.use_research_beta:
            # RESEARCH-BASED SOLUTION: Initialize monotonic gap parameterization
            if self.conservative_research:
                # Very conservative initialization for problematic architectures
                nn.init.normal_(self.threshold_base.weight, std=0.01)
                nn.init.constant_(self.threshold_base.bias, 0.0)
                if self.n_cats > 2:
                    nn.init.normal_(self.threshold_gaps.weight, std=0.01)
                    nn.init.constant_(self.threshold_gaps.bias, 0.1)
            else:
                # Standard research-based initialization
                nn.init.normal_(self.threshold_base.weight, std=0.05)
                nn.init.constant_(self.threshold_base.bias, 0.0)
                if self.n_cats > 2:
                    nn.init.normal_(self.threshold_gaps.weight, std=0.05)
                    nn.init.constant_(self.threshold_gaps.bias, 0.3)
        else:
            # STABLE IMPLEMENTATION: Initialize tanh-constrained network
            for module in self.threshold_network:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
        
        if self.use_discrimination:
            # Proper initialization for lognormal(0, 0.3) mapping
            # If x ~ Normal(0, std), then exp(0.3 * x) approximates lognormal(0, 0.3)
            nn.init.normal_(self.discrimination_network.weight, std=0.1)
            nn.init.constant_(self.discrimination_network.bias, 0)
    
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
        
        if self.use_research_beta:
            # RESEARCH-BASED SOLUTION: Monotonic Gap Parameterization
            # β₀ ∈ ℝ (unconstrained base threshold)
            beta_0 = self.threshold_base(threshold_input)  # Shape: (batch, seq, 1)
            
            # Apply bounds for numerical stability if requested
            if self.use_bounded_beta:
                beta_0 = torch.clamp(beta_0, min=-3.0, max=3.0)
            
            if self.n_cats == 2:
                # Binary case: only one threshold
                beta = beta_0
            else:
                # Multi-category: construct monotonic sequence β₀ < β₁ < β₂ < ...
                # Using positive gaps: βₖ = β₀ + Σᵢ₌₁ᵏ softplus(gapᵢ)
                gaps = F.softplus(self.threshold_gaps(threshold_input))  # Shape: (batch, seq, n_cats-2)
                
                # Apply bounds to gaps for stability
                if self.use_bounded_beta:
                    gaps = torch.clamp(gaps, min=0.1, max=2.0)
                
                # Construct monotonic thresholds (CUMULATIVE - this explodes for 5+ categories)
                betas = [beta_0]
                for i in range(gaps.shape[-1]):
                    next_beta = betas[-1] + gaps[:, :, i:i+1]
                    betas.append(next_beta)
                
                beta = torch.cat(betas, dim=-1)  # Shape: (batch, seq, n_cats-1)
        else:
            # STABLE IMPLEMENTATION: Tanh-constrained for attention models
            beta = self.threshold_network(threshold_input)
        
        # Extract discrimination (alpha) if enabled
        if self.use_discrimination:
            if question_features is not None:
                discrim_input = torch.cat([features, question_features], dim=-1)
            else:
                discrim_input = torch.cat([features, features], dim=-1)  # Use features twice if no question features
            
            # Mathematically correct lognormal(0, 0.3) mapping
            # If network output ~ Normal(0, std), then exp(0.3 * output) ~ lognormal(0, 0.3)
            raw_alpha = self.discrimination_network(discrim_input).squeeze(-1)
            alpha = torch.exp(0.3 * raw_alpha)
        else:
            alpha = torch.ones_like(theta)
        
        return theta, alpha, beta



class GPCMProbabilityLayer(nn.Module):
    """Compute GPCM response probabilities."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, theta: torch.Tensor, alpha: torch.Tensor, 
                beta: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Calculate GPCM response probabilities or logits.
        
        Args:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            beta: Threshold parameters, shape (batch_size, seq_len, K-1)
            return_logits: If True, return raw logits; if False, return probabilities
            
        Returns:
            probs/logits: GPCM probabilities or logits, shape (batch_size, seq_len, K)
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
        
        # Return logits for training (better numerical stability) or probabilities for inference
        if return_logits:
            return cum_logits
        else:
            return F.softmax(cum_logits, dim=-1)

