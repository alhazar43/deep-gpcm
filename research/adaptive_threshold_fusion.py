"""
Novel Adaptive Threshold Fusion Architecture

This module implements a deep mathematical integration of GPCM β thresholds 
and CORAL ordinal thresholds through an adaptive fusion mechanism.

Research Innovation:
- Unified threshold parameterization with learnable interaction
- Cross-threshold attention mechanism
- Adaptive weighting based on student ability and question difficulty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math


class AdaptiveThresholdFusion(nn.Module):
    """
    Novel architecture that deeply integrates GPCM and CORAL thresholds
    through adaptive fusion mechanisms.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_cats: int,
                 fusion_mode: str = 'hierarchical',
                 enable_cross_attention: bool = True,
                 enable_adaptive_weighting: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_cats = n_cats
        self.fusion_mode = fusion_mode
        self.enable_cross_attention = enable_cross_attention
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        # GPCM threshold pathway
        self.gpcm_threshold_network = nn.Linear(input_dim, n_cats - 1)
        
        # CORAL threshold pathway  
        self.coral_threshold_network = nn.Linear(input_dim, n_cats - 1)
        
        # Cross-threshold interaction mechanisms
        if enable_cross_attention:
            self.threshold_cross_attention = CrossThresholdAttention(n_cats - 1)
        
        # Adaptive weighting network
        if enable_adaptive_weighting:
            self.adaptive_weighting = AdaptiveWeightingNetwork(input_dim, n_cats - 1)
        
        # Fusion mechanisms by mode
        if fusion_mode == 'hierarchical':
            self.fusion_layer = HierarchicalFusion(n_cats - 1)
        elif fusion_mode == 'multiplicative':
            self.fusion_layer = MultiplicativeFusion(n_cats - 1)
        elif fusion_mode == 'neural':
            self.fusion_layer = NeuralFusion(n_cats - 1)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with mathematical considerations"""
        # GPCM thresholds: initialize with increasing difficulty
        nn.init.normal_(self.gpcm_threshold_network.weight, 0, 0.1)
        gpcm_bias_init = torch.linspace(-1, 1, self.n_cats - 1)
        self.gpcm_threshold_network.bias.data = gpcm_bias_init
        
        # CORAL thresholds: initialize with equal spacing
        nn.init.normal_(self.coral_threshold_network.weight, 0, 0.1)
        coral_bias_init = torch.linspace(-0.5, 0.5, self.n_cats - 1)
        self.coral_threshold_network.bias.data = coral_bias_init
    
    def forward(self, 
                student_features: torch.Tensor,
                question_features: torch.Tensor,
                theta: torch.Tensor,
                alpha: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive threshold fusion
        
        Args:
            student_features: Student state features
            question_features: Question embeddings  
            theta: Student abilities
            alpha: Discrimination parameters
            
        Returns:
            unified_thresholds: Fused threshold parameters
            fusion_info: Dictionary with fusion diagnostics
        """
        # Extract base thresholds
        gpcm_betas = self.gpcm_threshold_network(question_features)
        coral_taus = self.coral_threshold_network(student_features)
        
        # Cross-threshold attention
        fusion_info = {}
        if self.enable_cross_attention:
            gpcm_attended, coral_attended, attention_weights = self.threshold_cross_attention(
                gpcm_betas, coral_taus
            )
            fusion_info['attention_weights'] = attention_weights
        else:
            gpcm_attended, coral_attended = gpcm_betas, coral_taus
        
        # Adaptive weighting based on ability and discrimination
        if self.enable_adaptive_weighting:
            context = torch.cat([theta.unsqueeze(-1), alpha.unsqueeze(-1)], dim=-1)
            adaptive_weights = self.adaptive_weighting(context)
            fusion_info['adaptive_weights'] = adaptive_weights
        else:
            adaptive_weights = torch.ones_like(gpcm_attended) * 0.5
        
        # Apply fusion mechanism
        unified_thresholds = self.fusion_layer(
            gpcm_attended, coral_attended, adaptive_weights
        )
        
        # Additional fusion diagnostics
        fusion_info.update({
            'gpcm_betas': gpcm_betas,
            'coral_taus': coral_taus,
            'unified_thresholds': unified_thresholds,
            'gpcm_contribution': adaptive_weights,
            'coral_contribution': 1 - adaptive_weights
        })
        
        return unified_thresholds, fusion_info


class CrossThresholdAttention(nn.Module):
    """Cross-attention between GPCM and CORAL thresholds"""
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        self.n_thresholds = n_thresholds
        
        # Query, Key, Value projections
        self.gpcm_to_q = nn.Linear(n_thresholds, n_thresholds)
        self.coral_to_k = nn.Linear(n_thresholds, n_thresholds)
        self.coral_to_v = nn.Linear(n_thresholds, n_thresholds)
        
        # Reverse direction
        self.coral_to_q = nn.Linear(n_thresholds, n_thresholds)
        self.gpcm_to_k = nn.Linear(n_thresholds, n_thresholds)
        self.gpcm_to_v = nn.Linear(n_thresholds, n_thresholds)
        
        self.scale = math.sqrt(n_thresholds)
    
    def forward(self, gpcm_thresholds: torch.Tensor, coral_thresholds: torch.Tensor):
        """Apply cross-attention between threshold types"""
        
        # GPCM attends to CORAL
        q_gpcm = self.gpcm_to_q(gpcm_thresholds)
        k_coral = self.coral_to_k(coral_thresholds)
        v_coral = self.coral_to_v(coral_thresholds)
        
        attn_weights_g2c = F.softmax(
            torch.matmul(q_gpcm, k_coral.transpose(-2, -1)) / self.scale, dim=-1
        )
        gpcm_attended = gpcm_thresholds + torch.matmul(attn_weights_g2c, v_coral)
        
        # CORAL attends to GPCM  
        q_coral = self.coral_to_q(coral_thresholds)
        k_gpcm = self.gpcm_to_k(gpcm_thresholds)
        v_gpcm = self.gpcm_to_v(gpcm_thresholds)
        
        attn_weights_c2g = F.softmax(
            torch.matmul(q_coral, k_gpcm.transpose(-2, -1)) / self.scale, dim=-1
        )
        coral_attended = coral_thresholds + torch.matmul(attn_weights_c2g, v_gpcm)
        
        return gpcm_attended, coral_attended, {
            'gpcm_to_coral': attn_weights_g2c,
            'coral_to_gpcm': attn_weights_c2g
        }


class AdaptiveWeightingNetwork(nn.Module):
    """Learn adaptive weights based on student ability and question difficulty"""
    
    def __init__(self, context_dim: int, n_thresholds: int):
        super().__init__()
        
        self.weighting_network = nn.Sequential(
            nn.Linear(context_dim, n_thresholds * 2),
            nn.ReLU(),
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Sigmoid()  # Output in [0,1] for GPCM vs CORAL weighting
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on context"""
        return self.weighting_network(context)


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion: GPCM provides base, CORAL refines"""
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        self.refinement_network = nn.Sequential(
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Tanh()
        )
    
    def forward(self, gpcm_thresholds: torch.Tensor, 
                coral_thresholds: torch.Tensor, 
                weights: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical fusion"""
        # Base structure from GPCM
        base_structure = gpcm_thresholds
        
        # Refinement from CORAL
        combined_input = torch.cat([gpcm_thresholds, coral_thresholds], dim=-1)
        refinement = self.refinement_network(combined_input)
        
        # Weight-based combination
        return base_structure + weights * refinement


class MultiplicativeFusion(nn.Module):
    """Multiplicative fusion with learnable interaction"""
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        self.interaction_scale = nn.Parameter(torch.ones(n_thresholds))
    
    def forward(self, gpcm_thresholds: torch.Tensor,
                coral_thresholds: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative fusion"""
        # Multiplicative interaction
        interaction = gpcm_thresholds * torch.exp(self.interaction_scale * coral_thresholds)
        
        # Weighted combination
        return weights * gpcm_thresholds + (1 - weights) * interaction


class NeuralFusion(nn.Module):
    """Deep neural fusion of threshold parameters"""
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        
        self.fusion_network = nn.Sequential(
            nn.Linear(n_thresholds * 3, n_thresholds * 2),  # gpcm + coral + weights
            nn.ReLU(),
            nn.Linear(n_thresholds * 2, n_thresholds),
            nn.Tanh()
        )
    
    def forward(self, gpcm_thresholds: torch.Tensor,
                coral_thresholds: torch.Tensor, 
                weights: torch.Tensor) -> torch.Tensor:
        """Apply neural fusion"""
        # Concatenate all inputs
        fusion_input = torch.cat([gpcm_thresholds, coral_thresholds, weights], dim=-1)
        
        # Deep fusion
        return self.fusion_network(fusion_input)


class UnifiedOrdinalModel(nn.Module):
    """Complete model with adaptive threshold fusion"""
    
    def __init__(self, 
                 input_dim: int,
                 n_cats: int,
                 fusion_mode: str = 'hierarchical'):
        super().__init__()
        
        # Core fusion mechanism
        self.threshold_fusion = AdaptiveThresholdFusion(
            input_dim=input_dim,
            n_cats=n_cats,
            fusion_mode=fusion_mode
        )
        
        # Ability and discrimination networks
        self.ability_network = nn.Linear(input_dim, 1)
        self.discrimination_network = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, student_features: torch.Tensor, 
                question_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Full forward pass with unified thresholds"""
        
        # Extract basic IRT parameters
        theta = self.ability_network(student_features).squeeze(-1)
        alpha = self.discrimination_network(student_features).squeeze(-1)
        
        # Apply adaptive threshold fusion
        unified_betas, fusion_info = self.threshold_fusion(
            student_features, question_features, theta, alpha
        )
        
        # Compute probabilities using unified thresholds
        probs = self._compute_unified_probabilities(theta, alpha, unified_betas)
        
        return probs, {
            'theta': theta,
            'alpha': alpha,
            'unified_betas': unified_betas,
            'fusion_info': fusion_info
        }
    
    def _compute_unified_probabilities(self, theta: torch.Tensor, 
                                     alpha: torch.Tensor,
                                     unified_betas: torch.Tensor) -> torch.Tensor:
        """Compute probabilities using unified threshold system"""
        # Use GPCM formulation with unified thresholds
        batch_size, seq_len = theta.shape
        K = unified_betas.shape[-1] + 1
        
        # Cumulative logits with unified thresholds
        cum_logits = torch.zeros(batch_size, seq_len, K, device=theta.device)
        cum_logits[:, :, 0] = 0
        
        for k in range(1, K):
            cum_logits[:, :, k] = torch.sum(
                alpha.unsqueeze(-1) * (theta.unsqueeze(-1) - unified_betas[:, :, :k]),
                dim=-1
            )
        
        return F.softmax(cum_logits, dim=-1)


# Example usage and testing
def test_adaptive_threshold_fusion():
    """Test the adaptive threshold fusion mechanism"""
    
    batch_size, seq_len = 32, 20
    input_dim = 64
    n_cats = 4
    
    # Create model
    model = UnifiedOrdinalModel(input_dim, n_cats, fusion_mode='hierarchical')
    
    # Sample data
    student_features = torch.randn(batch_size, seq_len, input_dim)
    question_features = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    probs, info = model(student_features, question_features)
    
    print(f"Probabilities shape: {probs.shape}")
    print(f"Unified betas shape: {info['unified_betas'].shape}")
    print(f"Fusion diagnostics available: {list(info['fusion_info'].keys())}")
    
    # Verify probability constraints
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))}")
    
    return model, probs, info


if __name__ == "__main__":
    test_adaptive_threshold_fusion()