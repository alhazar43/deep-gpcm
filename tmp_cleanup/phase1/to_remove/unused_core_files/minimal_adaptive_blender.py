#!/usr/bin/env python3
"""
Minimal Viable Adaptive Blending Solution

This implements a simplified, numerically stable version of threshold-distance-based 
adaptive blending that preserves the core BGT mathematics while eliminating 
forward pass instabilities.

Key Design Principles:
1. Minimal computation graph complexity
2. Fixed parameters (no learnable components initially)
3. Conservative threshold geometry analysis
4. Gradient-isolated implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import torch.nn.functional as F

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