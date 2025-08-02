"""
Threshold-Distance-Based Dynamic Blending for Deep-GPCM CORAL Enhancement.

This module implements adaptive blending between GPCM (item-specific β thresholds) 
and CORAL (global ordinal τ thresholds) predictions using semantic threshold alignment
and geometric analysis of threshold relationships.

Core Innovation: Uses threshold distances |τᵢ - βᵢ| and range divergences to 
dynamically adjust blend weights for each category, addressing middle category 
prediction imbalance in ordinal classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class NumericalStabilityMixin:
    """Mixin class providing numerical stability utilities for threshold computations."""
    
    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Numerically stable logarithm computation."""
        return torch.log(torch.clamp(x + eps, min=eps))
    
    @staticmethod
    def safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Numerically stable division computation."""
        return numerator / torch.clamp(torch.abs(denominator) + eps, min=eps)
    
    @staticmethod  
    def safe_sigmoid(x: torch.Tensor, clamp_range: float = 10.0) -> torch.Tensor:
        """Numerically stable sigmoid computation with input clamping."""
        return torch.sigmoid(torch.clamp(x, -clamp_range, clamp_range))
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str) -> bool:
        """Validate tensor for NaN/Inf values."""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            print(f"Warning: {name} contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'Inf' if has_inf else ''} values")
            return False
        return True


class ThresholdDistanceBlender(nn.Module, NumericalStabilityMixin):
    """Adaptive blending based on IRT threshold geometry analysis.
    
    This class implements threshold-distance-based dynamic blending between
    GPCM (item-specific β thresholds) and CORAL (global ordinal τ thresholds)
    predictions using semantic threshold alignment.
    
    Key Innovation: Uses semantic alignment where both τ and β represent the same 
    category boundaries (τ₀↔β₀: 0→1, τ₁↔β₁: 1→2, τ₂↔β₂: 2→3) to compute meaningful
    threshold distances and dynamically adjust blend weights per category.
    """
    
    def __init__(self, 
                 n_categories: int = 4,
                 range_sensitivity_init: float = 1.0,
                 distance_sensitivity_init: float = 1.0,
                 baseline_bias_init: float = 0.0,
                 weight_clamp_range: Tuple[float, float] = (0.05, 0.95),
                 numerical_eps: float = 1e-7):
        """Initialize threshold distance blender.
        
        Args:
            n_categories: Number of ordinal categories (default: 4)
            range_sensitivity_init: Initial range sensitivity parameter (default: 1.0)
            distance_sensitivity_init: Initial distance sensitivity parameter (default: 1.0)  
            baseline_bias_init: Initial baseline bias parameter (default: 0.0)
            weight_clamp_range: Min/max bounds for blend weights (default: [0.05, 0.95])
            numerical_eps: Epsilon for numerical stability (default: 1e-7)
        """
        super().__init__()
        self.n_categories = n_categories
        self.weight_clamp_range = weight_clamp_range
        self.eps = numerical_eps
        
        # Learnable sensitivity parameters (avoiding IRT α, β conflicts)
        self.range_sensitivity = nn.Parameter(torch.tensor(range_sensitivity_init))
        self.distance_sensitivity = nn.Parameter(torch.tensor(distance_sensitivity_init))
        self.baseline_bias = nn.Parameter(torch.tensor(baseline_bias_init))
        
        # Parameter constraint bounds (registered as buffers for device handling)
        self.register_buffer('range_bounds', torch.tensor([0.1, 2.0]))
        self.register_buffer('distance_bounds', torch.tensor([0.5, 3.0]))
        self.register_buffer('bias_bounds', torch.tensor([-1.0, 1.0]))
        
        # Analysis state (for debugging and monitoring)
        self._last_geometry_metrics = None
        self._last_blend_weights = None
    
    def _clamp_learnable_parameters(self):
        """Ensure learnable parameters stay within valid bounds during training."""
        with torch.no_grad():
            self.range_sensitivity.clamp_(self.range_bounds[0], self.range_bounds[1])
            self.distance_sensitivity.clamp_(self.distance_bounds[0], self.distance_bounds[1])
            self.baseline_bias.clamp_(self.bias_bounds[0], self.bias_bounds[1])
    
    def analyze_threshold_geometry(self, 
                                   item_betas: torch.Tensor,
                                   ordinal_taus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze geometric relationships between IRT threshold systems.
        
        Args:
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_categories-1)
            ordinal_taus: CORAL τ thresholds, shape (n_categories-1,)
            
        Returns:
            Dictionary containing comprehensive threshold geometry metrics
        """
        batch_size, seq_len, n_thresholds = item_betas.shape
        
        # Semantic threshold alignment: τᵢ vs βᵢ (same category boundaries)
        # Broadcast ordinal_taus to match item_betas dimensions
        ordinal_taus_expanded = ordinal_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )  # Shape: (batch_size, seq_len, n_categories-1)
        
        # Semantic distance computation: |τᵢ - βᵢ| for aligned boundaries
        beta_tau_distances = torch.abs(ordinal_taus_expanded - item_betas)
        
        # Distance statistics for geometric analysis
        min_distance = beta_tau_distances.min(dim=-1)[0]        # (batch_size, seq_len)
        max_distance = beta_tau_distances.max(dim=-1)[0]        # (batch_size, seq_len)
        avg_distance = beta_tau_distances.mean(dim=-1)          # (batch_size, seq_len)
        distance_spread = beta_tau_distances.std(dim=-1)        # (batch_size, seq_len)
        
        # Range analysis: How spread out are the thresholds?
        item_range = item_betas.max(dim=-1)[0] - item_betas.min(dim=-1)[0]     # (batch_size, seq_len)
        ordinal_range = ordinal_taus.max() - ordinal_taus.min()                # scalar
        
        # Range divergence: normalized difference in threshold spreads
        range_sum = item_range + ordinal_range + self.eps
        range_divergence = torch.abs(ordinal_range - item_range) / range_sum   # (batch_size, seq_len)
        
        # Threshold correlation: geometric alignment measure
        item_norm = torch.norm(item_betas, dim=-1) + self.eps                  # (batch_size, seq_len)
        ordinal_norm = torch.norm(ordinal_taus) + self.eps                     # scalar
        dot_product = torch.sum(item_betas * ordinal_taus_expanded, dim=-1)    # (batch_size, seq_len)
        threshold_correlation = dot_product / (item_norm * ordinal_norm)       # (batch_size, seq_len)
        
        return {
            'beta_tau_distances': beta_tau_distances,      # (B, S, T) - Full distance matrix
            'min_distance': min_distance,                  # (B, S) - Closest boundary alignment
            'max_distance': max_distance,                  # (B, S) - Farthest boundary alignment  
            'avg_distance': avg_distance,                  # (B, S) - Average boundary divergence
            'distance_spread': distance_spread,            # (B, S) - Boundary alignment variability
            'range_divergence': range_divergence,          # (B, S) - Threshold spread difference
            'threshold_correlation': threshold_correlation, # (B, S) - Geometric alignment
            'item_range': item_range,                      # (B, S) - GPCM threshold spread
            'ordinal_range': ordinal_range                 # scalar - CORAL threshold spread
        }
    
    def calculate_blend_weights(self, 
                                item_betas: torch.Tensor,
                                ordinal_taus: torch.Tensor,
                                student_abilities: torch.Tensor,
                                discrimination_alphas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate category-specific adaptive blend weights.
        
        Args:
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_categories-1)
            ordinal_taus: CORAL τ thresholds, shape (n_categories-1,)
            student_abilities: Student θ abilities, shape (batch_size, seq_len)  
            discrimination_alphas: Discrimination α parameters, shape (batch_size, seq_len)
            
        Returns:
            blend_weights: Category-specific weights, shape (batch_size, seq_len, n_categories)
        """
        # Input validation
        if not (self.validate_tensor(item_betas, "item_betas") and 
                self.validate_tensor(ordinal_taus, "ordinal_taus") and
                self.validate_tensor(student_abilities, "student_abilities")):
            # Fallback to uniform weights on invalid input
            batch_size, seq_len = student_abilities.shape
            return torch.full((batch_size, seq_len, self.n_categories), 0.5, 
                            device=student_abilities.device, dtype=student_abilities.dtype)
        
        # Apply parameter constraints to prevent training instability
        self._clamp_learnable_parameters()
        
        # Analyze threshold geometry relationships
        geometry_metrics = self.analyze_threshold_geometry(item_betas, ordinal_taus)
        
        batch_size, seq_len = student_abilities.shape
        
        # Extract key geometric features
        min_distance = geometry_metrics['min_distance']                # (B, S)
        range_divergence = geometry_metrics['range_divergence']        # (B, S)
        threshold_correlation = geometry_metrics['threshold_correlation'] # (B, S)
        distance_spread = geometry_metrics['distance_spread']          # (B, S)
        
        # Expand spatial features to threshold dimensions for per-boundary analysis
        min_distance_expanded = min_distance.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        range_divergence_expanded = range_divergence.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        threshold_correlation_expanded = threshold_correlation.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        distance_spread_expanded = distance_spread.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        
        # Core adaptive weight formula using threshold geometry with safe operations
        log_term = self.safe_log(1 + range_divergence_expanded, self.eps)      # Range sensitivity
        distance_term = self.safe_div(min_distance_expanded, 1 + min_distance_expanded, self.eps)  # Distance sensitivity
        
        # Compute threshold-level weights (for category boundaries)
        threshold_weights = self.safe_sigmoid(
            self.range_sensitivity * log_term + 
            self.distance_sensitivity * distance_term + 
            0.3 * threshold_correlation_expanded +    # Geometric alignment bonus
            0.1 * distance_spread_expanded +          # Variability penalty
            self.baseline_bias
        )  # Shape: (batch_size, seq_len, n_categories-1)
        
        # Map threshold weights to category weights
        # Category 0: Based on first threshold (0→1 boundary)
        # Categories 1 to n-2: Direct threshold mapping  
        # Category n-1: Based on last threshold (n-2→n-1 boundary)
        category_weights = torch.zeros(
            batch_size, seq_len, self.n_categories, 
            device=threshold_weights.device, dtype=threshold_weights.dtype
        )
        
        # Category 0 weight (influenced by 0→1 boundary alignment)
        category_weights[:, :, 0] = threshold_weights[:, :, 0]
        
        # Middle category weights (direct threshold-to-category mapping)
        if self.n_categories > 2:
            category_weights[:, :, 1:-1] = threshold_weights[:, :, :self.n_categories-2]
        
        # Final category weight (influenced by final boundary alignment)
        category_weights[:, :, -1] = threshold_weights[:, :, -1]
        
        # Clamp weights to prevent extreme blending behavior
        category_weights = torch.clamp(
            category_weights, 
            self.weight_clamp_range[0], 
            self.weight_clamp_range[1]
        )
        
        # Store for analysis and debugging
        self._last_geometry_metrics = geometry_metrics
        self._last_blend_weights = category_weights
        
        return category_weights
    
    def get_analysis_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed analysis information from last forward pass."""
        if self._last_geometry_metrics is None or self._last_blend_weights is None:
            return None
            
        return {
            'geometry_metrics': self._last_geometry_metrics,
            'blend_weights': self._last_blend_weights,
            'learnable_params': {
                'range_sensitivity': self.range_sensitivity.item(),
                'distance_sensitivity': self.distance_sensitivity.item(),
                'baseline_bias': self.baseline_bias.item()
            }
        }


def test_threshold_distance_blender():
    """Test the ThresholdDistanceBlender implementation."""
    print("Testing ThresholdDistanceBlender")
    print("=" * 50)
    
    # Test parameters
    batch_size, seq_len, n_categories = 4, 8, 4
    n_thresholds = n_categories - 1
    
    # Create sample data following semantic alignment
    item_betas = torch.randn(batch_size, seq_len, n_thresholds)  # GPCM β thresholds
    ordinal_taus = torch.randn(n_thresholds)  # CORAL τ thresholds
    student_abilities = torch.randn(batch_size, seq_len)  # Student θ abilities
    discrimination_alphas = torch.randn(batch_size, seq_len)  # Discrimination α
    
    # Create blender
    blender = ThresholdDistanceBlender(
        n_categories=n_categories,
        range_sensitivity_init=1.0,
        distance_sensitivity_init=1.5,
        baseline_bias_init=0.0
    )
    
    print(f"✓ Created ThresholdDistanceBlender with {n_categories} categories")
    print(f"✓ Learnable parameters: range_sensitivity={blender.range_sensitivity.item():.3f}, "
          f"distance_sensitivity={blender.distance_sensitivity.item():.3f}, "
          f"baseline_bias={blender.baseline_bias.item():.3f}")
    
    # Test threshold geometry analysis
    geometry_metrics = blender.analyze_threshold_geometry(item_betas, ordinal_taus)
    
    print(f"✓ Threshold geometry analysis completed")
    print(f"  - Beta-tau distances shape: {geometry_metrics['beta_tau_distances'].shape}")
    print(f"  - Min distance range: [{geometry_metrics['min_distance'].min():.3f}, {geometry_metrics['min_distance'].max():.3f}]")
    print(f"  - Range divergence range: [{geometry_metrics['range_divergence'].min():.3f}, {geometry_metrics['range_divergence'].max():.3f}]")
    print(f"  - Threshold correlation range: [{geometry_metrics['threshold_correlation'].min():.3f}, {geometry_metrics['threshold_correlation'].max():.3f}]")
    
    # Test blend weight calculation
    blend_weights = blender.calculate_blend_weights(
        item_betas=item_betas,
        ordinal_taus=ordinal_taus,
        student_abilities=student_abilities,
        discrimination_alphas=discrimination_alphas
    )
    
    # Validate results
    expected_shape = (batch_size, seq_len, n_categories)
    assert blend_weights.shape == expected_shape, f"Shape mismatch: {blend_weights.shape} vs {expected_shape}"
    assert torch.isfinite(blend_weights).all(), "Non-finite values in blend weights"
    assert (blend_weights >= blender.weight_clamp_range[0]).all(), "Weights below minimum clamp"
    assert (blend_weights <= blender.weight_clamp_range[1]).all(), "Weights above maximum clamp"
    
    print(f"✓ Blend weight calculation completed")
    print(f"  - Output shape: {blend_weights.shape}")
    print(f"  - Weight range: [{blend_weights.min():.3f}, {blend_weights.max():.3f}]")
    print(f"  - Weight clamping range: [{blender.weight_clamp_range[0]}, {blender.weight_clamp_range[1]}]")
    
    # Test analysis info
    analysis_info = blender.get_analysis_info()
    assert analysis_info is not None, "Analysis info should be available after forward pass"
    assert 'geometry_metrics' in analysis_info, "Missing geometry metrics in analysis"
    assert 'blend_weights' in analysis_info, "Missing blend weights in analysis"
    assert 'learnable_params' in analysis_info, "Missing learnable params in analysis"
    
    print(f"✓ Analysis info extraction completed")
    print(f"  - Geometry metrics keys: {list(analysis_info['geometry_metrics'].keys())}")
    print(f"  - Learnable params: {analysis_info['learnable_params']}")
    
    # Test parameter clamping
    with torch.no_grad():
        blender.range_sensitivity.fill_(5.0)  # Above upper bound
        blender.distance_sensitivity.fill_(-1.0)  # Below lower bound
        blender.baseline_bias.fill_(2.0)  # Above upper bound
    
    blender._clamp_learnable_parameters()
    
    assert blender.range_sensitivity.item() <= blender.range_bounds[1], "Range sensitivity not clamped"
    assert blender.distance_sensitivity.item() >= blender.distance_bounds[0], "Distance sensitivity not clamped"
    assert blender.baseline_bias.item() <= blender.bias_bounds[1], "Baseline bias not clamped"
    
    print(f"✓ Parameter clamping validated")
    print(f"  - Clamped range_sensitivity: {blender.range_sensitivity.item():.3f}")
    print(f"  - Clamped distance_sensitivity: {blender.distance_sensitivity.item():.3f}")
    print(f"  - Clamped baseline_bias: {blender.baseline_bias.item():.3f}")
    
    # Test numerical stability with edge cases
    edge_item_betas = torch.zeros_like(item_betas)  # All zeros
    edge_ordinal_taus = torch.zeros_like(ordinal_taus)  # All zeros
    
    edge_weights = blender.calculate_blend_weights(
        item_betas=edge_item_betas,
        ordinal_taus=edge_ordinal_taus,
        student_abilities=student_abilities,
        discrimination_alphas=discrimination_alphas
    )
    
    assert torch.isfinite(edge_weights).all(), "Non-finite values with edge case inputs"
    print(f"✓ Numerical stability validated with edge cases")
    
    print("✅ All ThresholdDistanceBlender tests passed!")
    return True


if __name__ == "__main__":
    test_threshold_distance_blender()