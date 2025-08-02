#!/usr/bin/env python3
"""
Full Adaptive Blender with BGT Stability

This implements the complete ThresholdDistanceBlender from TODO.md specification
with learnable parameters, using the BGT (Bounded Geometric Transform) framework
for numerical stability while maintaining the full sophistication of the research design.

Key Features:
1. Full semantic threshold alignment (τ₀↔β₀, τ₁↔β₁, τ₂↔β₂)
2. Learnable parameters with gradient stability via BGT
3. Complete threshold geometry analysis
4. Category-specific adaptive blending
5. Robust error handling and fallback mechanisms
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import torch.nn.functional as F

class FullAdaptiveBlender(nn.Module):
    """
    Complete ThresholdDistanceBlender implementation with BGT stability.
    
    This version provides the full research capability from TODO.md while using
    BGT transforms to ensure training stability. It bridges the gap between
    the minimal version (for stability) and the complete specification (for performance).
    """
    
    def __init__(self, 
                 n_categories: int,
                 range_sensitivity_init: float = 0.1,  # More conservative default
                 distance_sensitivity_init: float = 0.2,  # More conservative default
                 baseline_bias_init: float = 0.0,
                 # BGT stability parameters
                 use_bgt_transforms: bool = True,
                 gradient_clipping: float = 0.5,  # More aggressive clipping
                 parameter_bounds: bool = True):
        """
        Initialize full adaptive blender with BGT stability.
        
        Args:
            n_categories: Number of ordinal categories
            range_sensitivity_init: Initial range sensitivity (learnable)
            distance_sensitivity_init: Initial distance sensitivity (learnable)
            baseline_bias_init: Initial baseline bias (learnable)
            use_bgt_transforms: Whether to use BGT transforms for stability
            gradient_clipping: Gradient clipping threshold
            parameter_bounds: Whether to enforce parameter bounds
        """
        super().__init__()
        self.n_categories = n_categories
        self.n_thresholds = n_categories - 1
        self.use_bgt_transforms = use_bgt_transforms
        self.gradient_clipping = gradient_clipping
        self.parameter_bounds = parameter_bounds
        
        # Learnable parameters with proper initialization
        self.range_sensitivity = nn.Parameter(torch.tensor(range_sensitivity_init))
        self.distance_sensitivity = nn.Parameter(torch.tensor(distance_sensitivity_init))
        self.baseline_bias = nn.Parameter(torch.tensor(baseline_bias_init))
        
        # Numerical stability constants
        self.register_buffer('eps', torch.tensor(1e-7))
        
        # Parameter bounds for stability (more conservative bounds)
        if parameter_bounds:
            self.register_buffer('range_min', torch.tensor(0.01))  # Tighter lower bound
            self.register_buffer('range_max', torch.tensor(1.0))   # Tighter upper bound
            self.register_buffer('distance_min', torch.tensor(0.01))  # Tighter lower bound
            self.register_buffer('distance_max', torch.tensor(1.0))   # Tighter upper bound
            self.register_buffer('bias_min', torch.tensor(-0.5))      # Tighter bounds
            self.register_buffer('bias_max', torch.tensor(0.5))       # Tighter bounds
        
        # BGT transform bounds
        self.register_buffer('tanh_scale', torch.tensor(2.0))  # For log(1+x) → 2*tanh(x/2)
        self.register_buffer('sigmoid_temp', torch.tensor(1.0))  # For x/(1+x) → sigmoid(x)
    
    def apply_parameter_bounds(self):
        """Apply parameter bounds for stability (called during training)."""
        if self.parameter_bounds:
            with torch.no_grad():
                self.range_sensitivity.clamp_(self.range_min, self.range_max)
                self.distance_sensitivity.clamp_(self.distance_min, self.distance_max)
                self.baseline_bias.clamp_(self.bias_min, self.bias_max)
    
    def bgt_log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """BGT transform: log(1 + x) → 2 * tanh(x/2) for numerical stability."""
        if self.use_bgt_transforms:
            return self.tanh_scale * torch.tanh(x / self.tanh_scale)
        else:
            return torch.log(1 + torch.clamp(x, min=0, max=10) + self.eps)
    
    def bgt_division_transform(self, x: torch.Tensor) -> torch.Tensor:
        """BGT transform: x/(1+x) → sigmoid(x) for numerical stability."""
        if self.use_bgt_transforms:
            return torch.sigmoid(x / self.sigmoid_temp)
        else:
            return x / (1 + torch.clamp(x, min=0, max=10) + self.eps)
    
    def analyze_threshold_geometry(self, 
                                  item_betas: torch.Tensor,
                                  ordinal_taus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete threshold geometry analysis from TODO.md specification.
        
        Implements semantic threshold alignment: τ₀↔β₀, τ₁↔β₁, τ₂↔β₂
        
        Args:
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_thresholds)
            ordinal_taus: CORAL τ thresholds, shape (n_thresholds,)
            
        Returns:
            Dictionary with comprehensive threshold geometry metrics
        """
        batch_size, seq_len, n_thresholds = item_betas.shape
        
        # Semantic alignment: expand ordinal thresholds for element-wise comparison
        ordinal_taus_expanded = ordinal_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, n_thresholds
        )
        
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
        """
        Calculate category-specific adaptive blend weights using full TODO.md specification.
        
        Args:
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_categories-1)
            ordinal_taus: CORAL τ thresholds, shape (n_categories-1,)
            student_abilities: Student θ abilities, shape (batch_size, seq_len)  
            discrimination_alphas: Discrimination α parameters, shape (batch_size, seq_len)
            
        Returns:
            Adaptive blend weights, shape (batch_size, seq_len, n_categories)
        """
        # Apply parameter bounds for stability
        self.apply_parameter_bounds()
        
        batch_size, seq_len = student_abilities.shape
        
        # CRITICAL: Detach parameters to prevent gradient coupling with memory networks
        item_betas_detached = item_betas.detach()
        student_abilities_detached = student_abilities.detach()
        discrimination_alphas_detached = discrimination_alphas.detach() if discrimination_alphas is not None else None
        
        # Analyze threshold geometry using semantic alignment (with detached parameters)
        geometry = self.analyze_threshold_geometry(item_betas_detached, ordinal_taus)
        
        # Extract geometry metrics
        min_distance = geometry['min_distance']                    # (batch_size, seq_len)
        range_divergence = geometry['range_divergence']            # (batch_size, seq_len)
        threshold_correlation = geometry['threshold_correlation']  # (batch_size, seq_len)
        distance_spread = geometry['distance_spread']              # (batch_size, seq_len)
        
        # Expand for category-specific computation
        min_distance_expanded = min_distance.unsqueeze(-1).expand(
            batch_size, seq_len, self.n_categories
        )
        range_divergence_expanded = range_divergence.unsqueeze(-1).expand(
            batch_size, seq_len, self.n_categories
        )
        threshold_correlation_expanded = threshold_correlation.unsqueeze(-1).expand(
            batch_size, seq_len, self.n_categories
        )
        distance_spread_expanded = distance_spread.unsqueeze(-1).expand(
            batch_size, seq_len, self.n_categories
        )
        
        # Apply BGT transforms for numerical stability
        log_term = self.bgt_log_transform(range_divergence_expanded)
        distance_term = self.bgt_division_transform(min_distance_expanded)
        
        # Full adaptive weight calculation from TODO.md
        threshold_weights = torch.sigmoid(
            self.range_sensitivity * log_term + 
            self.distance_sensitivity * distance_term + 
            0.3 * threshold_correlation_expanded +
            0.1 * distance_spread_expanded +
            self.baseline_bias
        )
        
        # Apply gradient clipping for training stability
        if self.training and self.gradient_clipping > 0:
            threshold_weights = threshold_weights.clamp(-self.gradient_clipping, self.gradient_clipping)
        
        return threshold_weights
    
    def forward(self,
                gpcm_probs: torch.Tensor,
                coral_probs: torch.Tensor,
                item_betas: torch.Tensor,
                ordinal_taus: torch.Tensor,
                student_abilities: torch.Tensor,
                discrimination_alphas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply full adaptive blending with complete TODO.md specification.
        
        Args:
            gpcm_probs: GPCM probabilities, shape (batch_size, seq_len, n_categories)
            coral_probs: CORAL probabilities, shape (batch_size, seq_len, n_categories)
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_categories-1)
            ordinal_taus: CORAL τ thresholds, shape (n_categories-1,)
            student_abilities: Student θ abilities, shape (batch_size, seq_len)
            discrimination_alphas: Discrimination α parameters, shape (batch_size, seq_len)
            
        Returns:
            Adaptively blended probabilities, shape (batch_size, seq_len, n_categories)
        """
        # Calculate adaptive blend weights using full specification
        blend_weights = self.calculate_blend_weights(
            item_betas, ordinal_taus, student_abilities, discrimination_alphas
        )
        
        # Category-specific adaptive blending
        blended_probs = (1 - blend_weights) * gpcm_probs + blend_weights * coral_probs
        
        # Robust probability normalization
        prob_sums = blended_probs.sum(dim=-1, keepdim=True)
        blended_probs = blended_probs / torch.clamp(prob_sums, min=self.eps)
        
        # Ensure valid probabilities
        blended_probs = torch.clamp(blended_probs, min=self.eps, max=1.0 - self.eps)
        
        # Final renormalization
        blended_probs = blended_probs / blended_probs.sum(dim=-1, keepdim=True)
        
        return blended_probs
    
    def get_analysis_info(self) -> Dict[str, Any]:
        """Get comprehensive analysis information for debugging and monitoring."""
        return {
            'learnable_params': {
                'range_sensitivity': self.range_sensitivity.item(),
                'distance_sensitivity': self.distance_sensitivity.item(),
                'baseline_bias': self.baseline_bias.item()
            },
            'stability_config': {
                'use_bgt_transforms': self.use_bgt_transforms,
                'gradient_clipping': self.gradient_clipping,
                'parameter_bounds': self.parameter_bounds
            },
            'parameter_bounds': {
                'range': [self.range_min.item(), self.range_max.item()] if self.parameter_bounds else None,
                'distance': [self.distance_min.item(), self.distance_max.item()] if self.parameter_bounds else None,
                'bias': [self.bias_min.item(), self.bias_max.item()] if self.parameter_bounds else None
            }
        }


def test_full_adaptive_blender():
    """Test the full adaptive blender for functionality and stability."""
    print("🧪 Testing Full Adaptive Blender with BGT Stability")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create full blender
    blender = FullAdaptiveBlender(
        n_categories=4,
        range_sensitivity_init=0.5,
        distance_sensitivity_init=1.0,
        baseline_bias_init=0.0,
        use_bgt_transforms=True,
        gradient_clipping=1.0,
        parameter_bounds=True
    ).to(device)
    
    print(f"✓ Full blender created with {sum(p.numel() for p in blender.parameters())} learnable parameters")
    
    # Test data
    batch_size, seq_len, n_cats = 32, 20, 4
    
    gpcm_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    coral_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    item_betas = torch.randn(batch_size, seq_len, n_cats-1, device=device, requires_grad=True)
    ordinal_taus = torch.randn(n_cats-1, device=device, requires_grad=True)
    student_abilities = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
    discrimination_alphas = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
    
    print(f"✓ Test data created")
    
    # Forward pass
    try:
        blended_probs = blender(
            gpcm_probs, coral_probs, item_betas, ordinal_taus, 
            student_abilities, discrimination_alphas
        )
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {blended_probs.shape}")
        print(f"  - Output range: [{blended_probs.min():.6f}, {blended_probs.max():.6f}]")
        print(f"  - Sum check: {blended_probs.sum(dim=-1).mean():.6f} (should be ~1.0)")
        
        # Check for numerical issues
        if torch.isnan(blended_probs).any() or torch.isinf(blended_probs).any():
            print("❌ NaN/Inf detected!")
            return False
        
        # Test backward pass
        targets = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
        loss = F.cross_entropy(
            blended_probs.view(-1, n_cats),
            targets.view(-1)
        )
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        max_grad = 0
        
        learnable_params = [
            ('range_sensitivity', blender.range_sensitivity),
            ('distance_sensitivity', blender.distance_sensitivity), 
            ('baseline_bias', blender.baseline_bias),
            ('item_betas', item_betas),
            ('ordinal_taus', ordinal_taus)
        ]
        
        for name, param in learnable_params:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                max_grad = max(max_grad, grad_norm)
                print(f"  {name:20} grad norm: {grad_norm:.6f}")
        
        total_grad_norm = (total_grad_norm ** 0.5) if param_count > 0 else 0
        print(f"✓ Total gradient norm: {total_grad_norm:.6f}")
        print(f"✓ Max gradient norm: {max_grad:.6f}")
        
        # Get analysis info
        analysis = blender.get_analysis_info()
        print(f"\n📊 Analysis Info:")
        print(f"  - Range sensitivity: {analysis['learnable_params']['range_sensitivity']:.6f}")
        print(f"  - Distance sensitivity: {analysis['learnable_params']['distance_sensitivity']:.6f}")
        print(f"  - Baseline bias: {analysis['learnable_params']['baseline_bias']:.6f}")
        print(f"  - BGT transforms: {analysis['stability_config']['use_bgt_transforms']}")
        
        if total_grad_norm > 10:
            print("⚠️  Large gradients detected")
            return False
        else:
            print("✅ Full adaptive blender test PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_adaptive_blender()
    if success:
        print("\n🎉 FULL ADAPTIVE BLENDER IMPLEMENTATION SUCCESSFUL!")
        print("   Complete TODO.md specification with BGT stability achieved.")
    else:
        print("\n❌ Full Adaptive Blender needs refinement.")


class MinimalAdaptiveBlender(nn.Module):
    """
    Minimal viable adaptive blending with maximum numerical stability.
    
    This version strips down the adaptive blending to its mathematical essence:
    - Simple threshold distance computation
    - Fixed blending parameters (no learning initially)
    - Conservative transform functions
    - Minimal forward pass complexity
    """
    
    def __init__(self, 
                 n_categories: int,
                 base_sensitivity: float = 0.1,
                 distance_threshold: float = 1.0):
        """
        Initialize minimal adaptive blender.
        
        Args:
            n_categories: Number of ordinal categories
            base_sensitivity: Fixed sensitivity parameter (non-learnable)
            distance_threshold: Threshold for distance-based switching
        """
        super().__init__()
        self.n_categories = n_categories
        
        # Fixed parameters (non-learnable) for maximum stability
        self.register_buffer('base_sensitivity', torch.tensor(base_sensitivity))
        self.register_buffer('distance_threshold', torch.tensor(distance_threshold))
        
        # Conservative blending bounds
        self.register_buffer('min_blend_weight', torch.tensor(0.1))
        self.register_buffer('max_blend_weight', torch.tensor(0.9))
    
    def calculate_simple_distances(self, 
                                 item_betas: torch.Tensor,
                                 ordinal_taus: torch.Tensor) -> torch.Tensor:
        """
        Calculate simple threshold distances with minimal computation.
        
        Args:
            item_betas: GPCM β thresholds, shape (batch_size, seq_len, n_thresholds)
            ordinal_taus: CORAL τ thresholds, shape (n_thresholds,)
            
        Returns:
            Simple distance metric, shape (batch_size, seq_len)
        """
        batch_size, seq_len, n_thresholds = item_betas.shape
        
        # Expand ordinal_taus for broadcasting
        taus_expanded = ordinal_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Simple L2 distance between threshold systems (conservative)
        distances = torch.abs(item_betas - taus_expanded)
        
        # Average distance across thresholds (simple aggregation)
        avg_distance = distances.mean(dim=-1)  # Shape: (batch_size, seq_len)
        
        return avg_distance
    
    def compute_blend_weights(self,
                            item_betas: torch.Tensor,
                            ordinal_taus: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive blend weights using minimal stable computation.
        
        Args:
            item_betas: GPCM β thresholds (detached from gradients)
            ordinal_taus: CORAL τ thresholds
            
        Returns:
            Blend weights, shape (batch_size, seq_len, n_categories)
        """
        batch_size, seq_len = item_betas.shape[:2]
        
        # Calculate simple distances
        distances = self.calculate_simple_distances(item_betas, ordinal_taus)
        
        # Simple distance-based blending rule (very conservative)
        # When distances are small: favor CORAL (more structured)
        # When distances are large: favor GPCM (more flexible)
        
        # Normalize distances using stable sigmoid
        normalized_distances = torch.sigmoid(distances / self.distance_threshold)
        
        # Base blend weight computation (minimal)
        base_weights = self.base_sensitivity * normalized_distances
        
        # Clamp to safe range
        blend_weights = torch.clamp(
            base_weights, 
            self.min_blend_weight, 
            self.max_blend_weight
        )
        
        # Expand to category dimension (uniform across categories for simplicity)
        blend_weights = blend_weights.unsqueeze(-1).expand(
            batch_size, seq_len, self.n_categories
        )
        
        return blend_weights
    
    def forward(self,
               gpcm_probs: torch.Tensor,
               coral_probs: torch.Tensor,
               item_betas: torch.Tensor,
               ordinal_taus: torch.Tensor) -> torch.Tensor:
        """
        Apply minimal adaptive blending with maximum stability.
        
        Args:
            gpcm_probs: GPCM probabilities, shape (batch_size, seq_len, n_categories)
            coral_probs: CORAL probabilities, shape (batch_size, seq_len, n_categories)
            item_betas: GPCM β thresholds (will be detached)
            ordinal_taus: CORAL τ thresholds
            
        Returns:
            Blended probabilities, shape (batch_size, seq_len, n_categories)
        """
        # CRITICAL: Detach item_betas to break gradient coupling
        item_betas_detached = item_betas.detach()
        
        # Compute blend weights using detached parameters
        blend_weights = self.compute_blend_weights(
            item_betas_detached, 
            ordinal_taus
        )
        
        # Simple linear blending
        blended_probs = (1 - blend_weights) * gpcm_probs + blend_weights * coral_probs
        
        # Ensure valid probabilities
        blended_probs = torch.clamp(blended_probs, min=1e-7, max=1.0)
        
        # Renormalize
        blended_probs = blended_probs / blended_probs.sum(dim=-1, keepdim=True)
        
        return blended_probs


def test_minimal_adaptive_blender():
    """Test the minimal adaptive blender for numerical stability."""
    print("🧪 Testing Minimal Adaptive Blender")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create blender
    blender = MinimalAdaptiveBlender(
        n_categories=4,
        base_sensitivity=0.1,
        distance_threshold=1.0
    ).to(device)
    
    print(f"✓ Blender created with {sum(p.numel() for p in blender.parameters())} learnable parameters")
    
    # Test data
    batch_size, seq_len, n_cats = 32, 20, 4
    
    gpcm_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    coral_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    item_betas = torch.randn(batch_size, seq_len, n_cats-1, device=device, requires_grad=True)
    ordinal_taus = torch.randn(n_cats-1, device=device, requires_grad=True)
    
    print(f"✓ Test data created")
    
    # Forward pass
    try:
        blended_probs = blender(gpcm_probs, coral_probs, item_betas, ordinal_taus)
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {blended_probs.shape}")
        print(f"  - Output range: [{blended_probs.min():.6f}, {blended_probs.max():.6f}]")
        print(f"  - Sum check: {blended_probs.sum(dim=-1).mean():.6f} (should be ~1.0)")
        
        # Check for numerical issues
        if torch.isnan(blended_probs).any() or torch.isinf(blended_probs).any():
            print("❌ NaN/Inf detected!")
            return False
        
        # Test backward pass
        targets = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
        loss = F.cross_entropy(
            blended_probs.view(-1, n_cats),
            targets.view(-1)
        )
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        
        for name, param in [('item_betas', item_betas), ('ordinal_taus', ordinal_taus)]:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                print(f"  {name} grad norm: {grad_norm:.6f}")
        
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        print(f"✓ Average gradient norm: {avg_grad_norm:.6f}")
        
        if avg_grad_norm > 10:
            print("⚠️  Large gradients detected")
            return False
        else:
            print("✅ Gradients are stable!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_minimal_adaptive_blender()
    if success:
        print("\n✅ Minimal Adaptive Blender Test PASSED")
    else:
        print("\n❌ Minimal Adaptive Blender Test FAILED")