"""
Mathematical threshold coupling mechanisms for integrating GPCM β thresholds 
with CORAL ordinal thresholds.

This module implements several approaches for deep mathematical integration
of the K-1 threshold parameters from both GPCM and CORAL models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import math


class HierarchicalThresholdCoupler(nn.Module):
    """
    Hierarchical coupling where GPCM provides base structure and CORAL refines.
    
    Mathematical formulation:
    β'_{i,k} = β_{i,k} + λ_k(θ, α_i) × R_k(τ_k, β_{i,k})
    
    Where:
    - β'_{i,k}: Unified threshold for item i, step k
    - λ_k(θ, α_i): Adaptive weighting based on ability and discrimination
    - R_k: Refinement function based on CORAL thresholds
    """
    
    def __init__(self, n_thresholds: int, enable_adaptive_weighting: bool = True):
        super().__init__()
        self.n_thresholds = n_thresholds
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        # Learnable refinement parameters
        self.refinement_weights = nn.Parameter(torch.ones(n_thresholds))
        self.refinement_bias = nn.Parameter(torch.zeros(n_thresholds))
        
        # Adaptive weighting network
        if enable_adaptive_weighting:
            self.adaptive_weighting = nn.Sequential(
                nn.Linear(2, n_thresholds * 2),  # θ, α input
                nn.ReLU(),
                nn.Linear(n_thresholds * 2, n_thresholds),
                nn.Sigmoid()
            )
        
        # Refinement function
        self.refinement_network = nn.Sequential(
            nn.Linear(n_thresholds * 2, n_thresholds),  # β and τ concatenated
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        # Refinement weights close to identity
        nn.init.normal_(self.refinement_weights, 1.0, 0.1)
        nn.init.zeros_(self.refinement_bias)
        
        # Initialize adaptive weighting to balanced state
        if self.enable_adaptive_weighting:
            for module in self.adaptive_weighting:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize refinement network
        for module in self.refinement_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                gpcm_betas: torch.Tensor,
                coral_taus: torch.Tensor,
                theta: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply hierarchical coupling
        
        Args:
            gpcm_betas: GPCM thresholds, shape (batch_size, seq_len, n_thresholds)
            coral_taus: CORAL thresholds, shape (n_thresholds,)
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            
        Returns:
            unified_thresholds: Coupled thresholds
            coupling_info: Dictionary with coupling diagnostics
        """
        batch_size, seq_len = theta.shape
        
        # Expand CORAL thresholds to match GPCM dimensions
        coral_taus_expanded = coral_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Compute refinement
        combined_input = torch.cat([gpcm_betas, coral_taus_expanded], dim=-1)
        refinement = self.refinement_network(combined_input)
        
        # Apply learnable scaling
        scaled_refinement = self.refinement_weights * refinement + self.refinement_bias
        
        # Adaptive weighting based on student ability and discrimination
        if self.enable_adaptive_weighting:
            if alpha is None:
                alpha = torch.ones_like(theta)
            
            # Create context vector
            context = torch.stack([theta, alpha], dim=-1)  # (batch_size, seq_len, 2)
            adaptive_weights = self.adaptive_weighting(context)  # (batch_size, seq_len, n_thresholds)
        else:
            adaptive_weights = torch.ones_like(gpcm_betas) * 0.5
        
        # Apply hierarchical coupling
        unified_thresholds = gpcm_betas + adaptive_weights * scaled_refinement
        
        # Coupling diagnostics
        coupling_info = {
            'gpcm_contribution': gpcm_betas,
            'coral_contribution': scaled_refinement,
            'adaptive_weights': adaptive_weights,
            'refinement_raw': refinement,
            'coral_taus_expanded': coral_taus_expanded
        }
        
        return unified_thresholds, coupling_info


class AttentionThresholdCoupler(nn.Module):
    """
    Cross-attention mechanism between GPCM and CORAL thresholds.
    
    Allows GPCM and CORAL thresholds to attend to each other for
    mutual refinement and information exchange.
    """
    
    def __init__(self, n_thresholds: int, n_heads: int = 2):
        super().__init__()
        self.n_thresholds = n_thresholds
        self.n_heads = n_heads
        
        # Cross-attention layers
        self.gpcm_to_coral_attention = nn.MultiheadAttention(
            embed_dim=n_thresholds,
            num_heads=n_heads,
            batch_first=True
        )
        
        self.coral_to_gpcm_attention = nn.MultiheadAttention(
            embed_dim=n_thresholds,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(n_thresholds)
        self.norm2 = nn.LayerNorm(n_thresholds)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(n_thresholds, n_thresholds * 2),
            nn.ReLU(),
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Dropout(0.1)
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(n_thresholds, n_thresholds * 2),
            nn.ReLU(), 
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Dropout(0.1)
        )
        
        # Final integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Tanh()
        )
    
    def forward(self,
                gpcm_betas: torch.Tensor,
                coral_taus: torch.Tensor,
                theta: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply cross-attention coupling"""
        
        batch_size, seq_len = theta.shape
        
        # Expand CORAL thresholds
        coral_taus_expanded = coral_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Reshape for attention (combine batch and sequence dimensions)
        gpcm_flat = gpcm_betas.view(-1, 1, self.n_thresholds)
        coral_flat = coral_taus_expanded.view(-1, 1, self.n_thresholds)
        
        # GPCM attends to CORAL
        gpcm_attended, gpcm_attn_weights = self.gpcm_to_coral_attention(
            gpcm_flat, coral_flat, coral_flat
        )
        gpcm_attended = gpcm_attended.view(batch_size, seq_len, self.n_thresholds)
        gpcm_attended = self.norm1(gpcm_attended + gpcm_betas)
        gpcm_attended = gpcm_attended + self.ffn1(gpcm_attended)
        
        # CORAL attends to GPCM
        coral_attended, coral_attn_weights = self.coral_to_gpcm_attention(
            coral_flat, gpcm_flat, gpcm_flat
        )
        coral_attended = coral_attended.view(batch_size, seq_len, self.n_thresholds)
        coral_attended = self.norm2(coral_attended + coral_taus_expanded)
        coral_attended = coral_attended + self.ffn2(coral_attended)
        
        # Final integration
        integrated_input = torch.cat([gpcm_attended, coral_attended], dim=-1)
        unified_thresholds = self.integration_layer(integrated_input)
        
        coupling_info = {
            'gpcm_attended': gpcm_attended,
            'coral_attended': coral_attended,
            'gpcm_attention_weights': gpcm_attn_weights,
            'coral_attention_weights': coral_attn_weights
        }
        
        return unified_thresholds, coupling_info


class NeuralThresholdCoupler(nn.Module):
    """
    Deep neural coupling of GPCM and CORAL thresholds with context awareness.
    
    Uses a deep network to learn complex interactions between the two
    threshold systems while incorporating student and item context.
    """
    
    def __init__(self, n_thresholds: int, hidden_dim: int = None):
        super().__init__()
        self.n_thresholds = n_thresholds
        hidden_dim = hidden_dim or n_thresholds * 2
        
        # Context encoding
        self.context_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # θ, α
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Threshold interaction network
        self.interaction_network = nn.Sequential(
            nn.Linear(n_thresholds * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_thresholds),
            nn.Tanh()
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.context_encoder, self.interaction_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self,
                gpcm_betas: torch.Tensor,
                coral_taus: torch.Tensor,
                theta: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply neural coupling"""
        
        batch_size, seq_len = theta.shape
        
        if alpha is None:
            alpha = torch.ones_like(theta)
        
        # Encode context
        context = torch.stack([theta, alpha], dim=-1)
        context_encoded = self.context_encoder(context)
        
        # Expand CORAL thresholds
        coral_taus_expanded = coral_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Create interaction input
        interaction_input = torch.cat([
            gpcm_betas,
            coral_taus_expanded, 
            context_encoded
        ], dim=-1)
        
        # Apply deep interaction network
        interaction_output = self.interaction_network(interaction_input)
        
        # Residual connection with learnable weight
        unified_thresholds = (
            self.residual_weight * gpcm_betas + 
            (1 - self.residual_weight) * interaction_output
        )
        
        coupling_info = {
            'context_encoded': context_encoded,
            'interaction_output': interaction_output,
            'residual_weight': self.residual_weight,
            'gpcm_weight': self.residual_weight,
            'neural_weight': 1 - self.residual_weight
        }
        
        return unified_thresholds, coupling_info


class AdaptiveThresholdCoupler(nn.Module):
    """
    Adaptive coupling that dynamically selects between different coupling mechanisms
    based on data characteristics and context.
    """
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        self.n_thresholds = n_thresholds
        
        # Individual coupling mechanisms
        self.hierarchical_coupler = HierarchicalThresholdCoupler(n_thresholds)
        self.attention_coupler = AttentionThresholdCoupler(n_thresholds)
        self.neural_coupler = NeuralThresholdCoupler(n_thresholds)
        
        # Mechanism selection network
        self.mechanism_selector = nn.Sequential(
            nn.Linear(2 + n_thresholds * 2, 16),  # θ, α, β, τ
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 mechanisms
            nn.Softmax(dim=-1)
        )
    
    def forward(self,
                gpcm_betas: torch.Tensor,
                coral_taus: torch.Tensor,
                theta: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply adaptive coupling"""
        
        batch_size, seq_len = theta.shape
        
        if alpha is None:
            alpha = torch.ones_like(theta)
        
        # Expand CORAL thresholds for selection
        coral_taus_expanded = coral_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Create selection input
        selection_input = torch.cat([
            theta.unsqueeze(-1),
            alpha.unsqueeze(-1),
            gpcm_betas,
            coral_taus_expanded
        ], dim=-1)
        
        # Select mechanism weights
        mechanism_weights = self.mechanism_selector(selection_input)  # (batch, seq, 3)
        
        # Apply all mechanisms
        hier_output, hier_info = self.hierarchical_coupler(gpcm_betas, coral_taus, theta, alpha)
        attn_output, attn_info = self.attention_coupler(gpcm_betas, coral_taus, theta, alpha)
        neur_output, neur_info = self.neural_coupler(gpcm_betas, coral_taus, theta, alpha)
        
        # Weighted combination
        unified_thresholds = (
            mechanism_weights[:, :, 0:1] * hier_output +
            mechanism_weights[:, :, 1:2] * attn_output +
            mechanism_weights[:, :, 2:3] * neur_output
        )
        
        coupling_info = {
            'mechanism_weights': mechanism_weights,
            'hierarchical_output': hier_output,
            'attention_output': attn_output,
            'neural_output': neur_output,
            'hierarchical_info': hier_info,
            'attention_info': attn_info,
            'neural_info': neur_info
        }
        
        return unified_thresholds, coupling_info


def test_threshold_couplers():
    """Test all threshold coupling mechanisms"""
    
    print("Testing Threshold Coupling Mechanisms")
    print("=" * 50)
    
    # Test parameters
    batch_size, seq_len = 8, 12
    n_thresholds = 3  # For 4-category model
    
    # Sample data
    gpcm_betas = torch.randn(batch_size, seq_len, n_thresholds)
    coral_taus = torch.randn(n_thresholds)
    theta = torch.randn(batch_size, seq_len)
    alpha = torch.ones(batch_size, seq_len) * 1.5
    
    # Test each coupler
    couplers = {
        'Hierarchical': HierarchicalThresholdCoupler(n_thresholds),
        'Attention': AttentionThresholdCoupler(n_thresholds),
        'Neural': NeuralThresholdCoupler(n_thresholds),
        'Adaptive': AdaptiveThresholdCoupler(n_thresholds)
    }
    
    for name, coupler in couplers.items():
        print(f"\nTesting {name} Coupler:")
        
        # Forward pass
        unified_thresholds, coupling_info = coupler(gpcm_betas, coral_taus, theta, alpha)
        
        print(f"  Input GPCM shape: {gpcm_betas.shape}")
        print(f"  Input CORAL shape: {coral_taus.shape}")
        print(f"  Output shape: {unified_thresholds.shape}")
        print(f"  Info keys: {list(coupling_info.keys())[:3]}...")  # Show first 3 keys
        
        # Basic checks
        assert unified_thresholds.shape == gpcm_betas.shape, f"Shape mismatch for {name}"
        assert torch.isfinite(unified_thresholds).all(), f"Non-finite values in {name}"
        
        print(f"  ✓ Shape and finiteness checks passed")
    
    print("\n" + "=" * 50)
    print("All coupling mechanisms tested successfully!")


def analyze_coupling_behavior():
    """Analyze how different coupling mechanisms behave"""
    
    print("\nCoupling Behavior Analysis")
    print("=" * 50)
    
    # Create test scenarios
    batch_size, seq_len, n_thresholds = 4, 6, 3
    
    # Scenario 1: High ability student
    high_theta = torch.ones(batch_size, seq_len) * 2.0
    alpha = torch.ones(batch_size, seq_len) * 1.0
    
    # Scenario 2: Low ability student  
    low_theta = torch.ones(batch_size, seq_len) * -2.0
    
    # Sample thresholds
    gpcm_betas = torch.linspace(-1, 1, n_thresholds).unsqueeze(0).unsqueeze(0).expand(
        batch_size, seq_len, -1
    )
    coral_taus = torch.linspace(-0.5, 0.5, n_thresholds)
    
    coupler = AdaptiveThresholdCoupler(n_thresholds)
    
    # High ability case
    high_unified, high_info = coupler(gpcm_betas, coral_taus, high_theta, alpha)
    high_weights = high_info['mechanism_weights'].mean(dim=(0, 1))
    
    # Low ability case
    low_unified, low_info = coupler(gpcm_betas, coral_taus, low_theta, alpha)
    low_weights = low_info['mechanism_weights'].mean(dim=(0, 1))
    
    print(f"High ability mechanism weights: {high_weights}")
    print(f"Low ability mechanism weights: {low_weights}")
    
    print(f"High ability unified thresholds: {high_unified[0, 0, :]}")
    print(f"Low ability unified thresholds: {low_unified[0, 0, :]}")


if __name__ == "__main__":
    test_threshold_couplers()
    analyze_coupling_behavior()