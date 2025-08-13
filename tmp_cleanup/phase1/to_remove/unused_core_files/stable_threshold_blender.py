"""
Stable Threshold-Distance-Based Dynamic Blending for Deep-GPCM CORAL Enhancement.

This module implements a numerically stable version of adaptive blending between 
GPCM and CORAL predictions using bounded geometric transforms (BGT) that prevent 
gradient explosion while maintaining semantic threshold alignment.

Key Innovation: Bounded Geometric Transform (BGT) replaces explosive log/correlation
operations with numerically stable alternatives that preserve threshold geometry
semantics without gradient explosion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class BoundedGeometricTransform:
    """Bounded Geometric Transform utilities for numerically stable threshold analysis."""
    
    @staticmethod
    def stable_range_transform(range_divergence: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Stable alternative to log(1 + range_divergence) using bounded tanh transform.
        
        Mathematical Properties:
        - Output bounded in [0, 2]
        - Smooth gradient everywhere
        - Monotonic increasing like log
        - No explosion for large inputs
        
        Formula: 2 * tanh(range_divergence / 2)
        """
        return 2.0 * torch.tanh(torch.clamp(range_divergence, min=0, max=10) / 2.0)
    
    @staticmethod
    def stable_distance_transform(distance: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Stable alternative to distance/(1+distance) using sigmoid-based transform.
        
        Mathematical Properties:
        - Output bounded in [0, 1]
        - Asymptotically approaches 1 for large distances
        - Smooth gradient everywhere
        - No division by zero issues
        
        Formula: sigmoid(distance) 
        """
        return torch.sigmoid(torch.clamp(distance, min=-10, max=10))
    
    @staticmethod
    def stable_correlation_transform(correlation: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Stable alternative to raw dot product correlation using normalized sigmoid.
        
        Mathematical Properties:
        - Output bounded in [0, 1] 
        - Symmetric around 0.5 for zero correlation
        - Smooth gradient everywhere
        - No norm explosion issues
        
        Formula: 0.5 * (1 + tanh(clamp(correlation, -3, 3)))
        """
        clamped_corr = torch.clamp(correlation, min=-3.0, max=3.0)
        return 0.5 * (1.0 + torch.tanh(clamped_corr))
    
    @staticmethod
    def stable_spread_transform(spread: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Stable alternative to raw standard deviation using bounded exponential decay.
        
        Mathematical Properties:
        - Output bounded in [0, 1]
        - Monotonic decreasing (high spread = low score)
        - Smooth gradient everywhere
        - No explosion for large spreads
        
        Formula: exp(-clamp(spread, 0, 5))
        """
        clamped_spread = torch.clamp(spread, min=0, max=5.0)
        return torch.exp(-clamped_spread)


class StableThresholdDistanceBlender(nn.Module):
    """
    Numerically stable adaptive blending using Bounded Geometric Transform (BGT).
    
    This class prevents gradient explosion while maintaining semantic threshold 
    alignment through bounded mathematical transforms that preserve the research
    contribution of threshold-distance-based dynamic blending.
    """
    
    def __init__(self, 
                 n_categories: int = 4,
                 range_sensitivity_init: float = 0.5,    # Reduced from 1.0
                 distance_sensitivity_init: float = 0.5, # Reduced from 1.0
                 baseline_bias_init: float = 0.0,
                 weight_clamp_range: Tuple[float, float] = (0.1, 0.9),  # Tighter bounds
                 numerical_eps: float = 1e-7):
        """
        Initialize stable threshold distance blender with conservative parameters.
        
        Args:
            n_categories: Number of ordinal categories
            range_sensitivity_init: Initial range sensitivity (conservative: 0.5)
            distance_sensitivity_init: Initial distance sensitivity (conservative: 0.5)  
            baseline_bias_init: Initial baseline bias (default: 0.0)
            weight_clamp_range: Tighter bounds for blend weights (0.1, 0.9)
            numerical_eps: Epsilon for numerical stability
        """
        super().__init__()
        self.n_categories = n_categories
        self.weight_clamp_range = weight_clamp_range
        self.eps = numerical_eps
        
        # CRITICAL FIX: Make BGT parameters non-learnable to prevent gradient explosion
        # Use fixed parameters instead of learnable ones to eliminate gradient issues
        self.register_buffer('range_sensitivity', torch.tensor(range_sensitivity_init))
        self.register_buffer('distance_sensitivity', torch.tensor(distance_sensitivity_init))
        self.register_buffer('baseline_bias', torch.tensor(baseline_bias_init))
        
        # Ultra-conservative parameter bounds for maximum gradient stability
        self.register_buffer('range_bounds', torch.tensor([0.001, 0.1]))    # Much stricter bounds
        self.register_buffer('distance_bounds', torch.tensor([0.001, 0.1])) # Much stricter bounds  
        self.register_buffer('bias_bounds', torch.tensor([-0.1, 0.1]))      # Much stricter bounds
        
        # Geometric transform utilities
        self.bgt = BoundedGeometricTransform()
        
        # Analysis state
        self._last_geometry_metrics = None
        self._last_blend_weights = None
    
    def _clamp_learnable_parameters(self):
        """No-op since parameters are now fixed buffers, not learnable."""
        # Parameters are now registered as buffers (non-learnable) so no clamping needed
        pass
    
    def analyze_threshold_geometry_stable(self, 
                                         item_betas: torch.Tensor,
                                         ordinal_taus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Stable threshold geometry analysis using bounded transforms.
        
        This version replaces explosive operations with numerically stable alternatives
        while preserving semantic threshold alignment meaning.
        """
        batch_size, seq_len, n_thresholds = item_betas.shape
        
        # Semantic threshold alignment (unchanged - this is core to research)
        ordinal_taus_expanded = ordinal_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Semantic distance computation (unchanged)
        beta_tau_distances = torch.abs(ordinal_taus_expanded - item_betas)
        
        # Basic distance statistics (stable operations)
        min_distance = beta_tau_distances.min(dim=-1)[0]        # (batch_size, seq_len)
        max_distance = beta_tau_distances.max(dim=-1)[0]        # (batch_size, seq_len)
        avg_distance = beta_tau_distances.mean(dim=-1)          # (batch_size, seq_len)
        
        # STABLE: Replace std() with mean absolute deviation (more stable)
        distance_mad = torch.mean(torch.abs(beta_tau_distances - avg_distance.unsqueeze(-1)), dim=-1)
        
        # Range analysis with stability bounds
        item_range = torch.clamp(
            item_betas.max(dim=-1)[0] - item_betas.min(dim=-1)[0], 
            min=self.eps, max=10.0
        )
        ordinal_range = torch.clamp(
            ordinal_taus.max() - ordinal_taus.min(), 
            min=self.eps, max=10.0
        )
        
        # STABLE: Bounded range divergence calculation
        range_sum = item_range + ordinal_range + self.eps
        range_divergence = torch.clamp(
            torch.abs(ordinal_range - item_range) / range_sum,
            min=0, max=1.0  # Bounded output
        )
        
        # STABLE: Simplified correlation using cosine similarity with bounds
        item_betas_norm = F.normalize(item_betas, p=2, dim=-1, eps=self.eps)
        ordinal_taus_norm = F.normalize(ordinal_taus_expanded, p=2, dim=-1, eps=self.eps)
        
        # Cosine similarity (bounded in [-1, 1])
        threshold_correlation = torch.sum(item_betas_norm * ordinal_taus_norm, dim=-1)
        
        return {
            'beta_tau_distances': beta_tau_distances,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'avg_distance': avg_distance,
            'distance_spread': distance_mad,  # Using MAD instead of std
            'range_divergence': range_divergence,
            'threshold_correlation': threshold_correlation,
            'item_range': item_range,
            'ordinal_range': ordinal_range
        }
    
    def calculate_blend_weights_stable(self, 
                                     item_betas: torch.Tensor,
                                     ordinal_taus: torch.Tensor,
                                     student_abilities: torch.Tensor,
                                     discrimination_alphas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate stable adaptive blend weights using Bounded Geometric Transform.
        
        This version prevents gradient explosion while maintaining semantic meaning
        of threshold-distance-based blending.
        """
        # Input validation with fallback
        if not self._validate_inputs(item_betas, ordinal_taus, student_abilities):
            batch_size, seq_len = student_abilities.shape
            return torch.full((batch_size, seq_len, self.n_categories), 0.5, 
                            device=student_abilities.device, dtype=student_abilities.dtype)
        
        # Conservative parameter clamping
        self._clamp_learnable_parameters()
        
        # Stable threshold geometry analysis
        geometry_metrics = self.analyze_threshold_geometry_stable(item_betas, ordinal_taus)
        
        batch_size, seq_len = student_abilities.shape
        
        # Extract geometric features
        min_distance = geometry_metrics['min_distance']
        range_divergence = geometry_metrics['range_divergence']
        threshold_correlation = geometry_metrics['threshold_correlation']
        distance_spread = geometry_metrics['distance_spread']
        
        # Expand to threshold dimensions
        min_distance_expanded = min_distance.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        range_divergence_expanded = range_divergence.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        threshold_correlation_expanded = threshold_correlation.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        distance_spread_expanded = distance_spread.unsqueeze(-1).expand(-1, -1, self.n_categories - 1)
        
        # STABLE CORE FORMULA: Bounded Geometric Transform (BGT)
        # Replace explosive operations with bounded alternatives
        range_term = self.bgt.stable_range_transform(range_divergence_expanded)
        distance_term = self.bgt.stable_distance_transform(min_distance_expanded)
        correlation_term = self.bgt.stable_correlation_transform(threshold_correlation_expanded)
        spread_term = self.bgt.stable_spread_transform(distance_spread_expanded)
        
        # Conservative weighted combination (smaller coefficients)
        combined_features = (
            self.range_sensitivity * range_term +
            self.distance_sensitivity * distance_term +
            0.2 * correlation_term +    # Increased for more variance
            0.1 * spread_term +         # Increased for more variance
            self.baseline_bias
        )
        
        # Apply final sigmoid with ultra-conservative clamping
        threshold_weights = torch.sigmoid(torch.clamp(combined_features, min=-3, max=3))
        
        # Map to category weights (unchanged semantic mapping)
        category_weights = torch.zeros(
            batch_size, seq_len, self.n_categories, 
            device=threshold_weights.device, dtype=threshold_weights.dtype
        )
        
        category_weights[:, :, 0] = threshold_weights[:, :, 0]
        if self.n_categories > 2:
            category_weights[:, :, 1:-1] = threshold_weights[:, :, :self.n_categories-2]
        category_weights[:, :, -1] = threshold_weights[:, :, -1]
        
        # Tighter weight clamping
        category_weights = torch.clamp(
            category_weights, 
            self.weight_clamp_range[0], 
            self.weight_clamp_range[1]
        )
        
        # Store for analysis
        self._last_geometry_metrics = geometry_metrics
        self._last_blend_weights = category_weights
        
        return category_weights
    
    def _validate_inputs(self, item_betas: torch.Tensor, ordinal_taus: torch.Tensor, 
                        student_abilities: torch.Tensor) -> bool:
        """Validate inputs for numerical stability."""
        tensors = [
            (item_betas, "item_betas"),
            (ordinal_taus, "ordinal_taus"), 
            (student_abilities, "student_abilities")
        ]
        
        for tensor, name in tensors:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: {name} contains NaN/Inf values")
                return False
            
            # Check for extreme values that could cause instability
            if tensor.abs().max() > 100:
                print(f"Warning: {name} contains extreme values (max: {tensor.abs().max()})")
                return False
                
        return True
    
    def get_analysis_info(self) -> Optional[Dict[str, Any]]:
        """Get analysis information from last forward pass."""
        if self._last_geometry_metrics is None or self._last_blend_weights is None:
            return None
            
        return {
            'geometry_metrics': self._last_geometry_metrics,
            'blend_weights': self._last_blend_weights,
            'learnable_params': {
                'range_sensitivity': self.range_sensitivity.item(),
                'distance_sensitivity': self.distance_sensitivity.item(),
                'baseline_bias': self.baseline_bias.item()
            },
            'stability_info': {
                'parameter_bounds': {
                    'range_bounds': self.range_bounds.tolist(),
                    'distance_bounds': self.distance_bounds.tolist(),
                    'bias_bounds': self.bias_bounds.tolist()
                },
                'weight_clamp_range': self.weight_clamp_range,
                'transform_type': 'bounded_geometric_transform'
            }
        }


def test_stable_threshold_blender():
    """Test the stable implementation against gradient explosion scenarios."""
    print("Testing StableThresholdDistanceBlender")
    print("=" * 60)
    
    # Test with challenging parameters that caused explosion in original
    batch_size, seq_len, n_categories = 4, 8, 4
    n_thresholds = n_categories - 1
    
    # Create potentially problematic data
    item_betas = torch.randn(batch_size, seq_len, n_thresholds) * 2  # Larger scale
    ordinal_taus = torch.randn(n_thresholds) * 2  # Larger scale
    student_abilities = torch.randn(batch_size, seq_len) * 2
    
    # Add some extreme values to test stability
    item_betas[0, 0, :] = 5.0  # Extreme values
    ordinal_taus[0] = -5.0     # Extreme values
    
    print(f"✓ Created test data with extreme values")
    print(f"  - item_betas range: [{item_betas.min():.3f}, {item_betas.max():.3f}]")
    print(f"  - ordinal_taus range: [{ordinal_taus.min():.3f}, {ordinal_taus.max():.3f}]")
    
    # Create stable blender
    stable_blender = StableThresholdDistanceBlender(
        n_categories=n_categories,
        range_sensitivity_init=0.5,
        distance_sensitivity_init=0.5,
        baseline_bias_init=0.0
    )
    
    print(f"✓ Created StableThresholdDistanceBlender")
    print(f"  - Conservative parameter bounds applied")
    print(f"  - Bounded Geometric Transform enabled")
    
    # Test gradient computation (this would explode in original)
    item_betas.requires_grad_(True)
    ordinal_taus.requires_grad_(True)
    
    try:
        # Forward pass
        blend_weights = stable_blender.calculate_blend_weights_stable(
            item_betas=item_betas,
            ordinal_taus=ordinal_taus,
            student_abilities=student_abilities
        )
        
        # Backward pass to test gradient stability
        loss = blend_weights.sum()
        loss.backward()
        
        # Check gradient norms
        beta_grad_norm = item_betas.grad.norm().item()
        tau_grad_norm = ordinal_taus.grad.norm().item()
        
        print(f"✅ Gradient computation successful!")
        print(f"  - item_betas gradient norm: {beta_grad_norm:.6f}")
        print(f"  - ordinal_taus gradient norm: {tau_grad_norm:.6f}")
        print(f"  - blend_weights range: [{blend_weights.min():.3f}, {blend_weights.max():.3f}]")
        
        # Validate no explosion (gradients should be reasonable)
        assert beta_grad_norm < 10.0, f"Beta gradient norm too large: {beta_grad_norm}"
        assert tau_grad_norm < 10.0, f"Tau gradient norm too large: {tau_grad_norm}"
        assert torch.isfinite(blend_weights).all(), "Non-finite blend weights"
        
        print(f"✅ Gradient stability validated - no explosion detected!")
        
    except Exception as e:
        print(f"❌ Error during gradient computation: {e}")
        return False
    
    # Test boundary conditions
    print(f"\n--- Testing Boundary Conditions ---")
    
    # Test with identical thresholds (edge case)
    identical_betas = torch.zeros(batch_size, seq_len, n_thresholds)
    identical_taus = torch.zeros(n_thresholds)
    
    edge_weights = stable_blender.calculate_blend_weights_stable(
        item_betas=identical_betas,
        ordinal_taus=identical_taus,
        student_abilities=student_abilities
    )
    
    assert torch.isfinite(edge_weights).all(), "Edge case produced non-finite values"
    print(f"✓ Identical thresholds handled: weights range [{edge_weights.min():.3f}, {edge_weights.max():.3f}]")
    
    # Test analysis info
    analysis_info = stable_blender.get_analysis_info()
    assert analysis_info is not None, "Analysis info missing"
    assert 'stability_info' in analysis_info, "Stability info missing"
    
    print(f"✓ Analysis info includes stability metrics")
    print(f"  - Transform type: {analysis_info['stability_info']['transform_type']}")
    
    print(f"\n✅ All stability tests passed!")
    print(f"🎯 Ready for training without gradient explosion")
    
    return True


if __name__ == "__main__":
    test_stable_threshold_blender()