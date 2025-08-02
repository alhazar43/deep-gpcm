"""
Safe threshold blender with robust numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class SafeThresholdDistanceBlender(nn.Module):
    """Ultra-conservative threshold blender for debugging gradient explosion."""
    
    def __init__(self, 
                 n_categories: int = 4,
                 range_sensitivity_init: float = 0.01,  # Very small
                 distance_sensitivity_init: float = 0.01,  # Very small
                 baseline_bias_init: float = 0.0):
        super().__init__()
        self.n_categories = n_categories
        
        # Extremely conservative learnable parameters
        self.range_sensitivity = nn.Parameter(torch.tensor(range_sensitivity_init))
        self.distance_sensitivity = nn.Parameter(torch.tensor(distance_sensitivity_init))
        self.baseline_bias = nn.Parameter(torch.tensor(baseline_bias_init))
        
        # Learnable category-specific biases - let the model learn what works best
        self.category_biases = nn.Parameter(torch.zeros(n_categories))
        
        # Fixed bounds
        self.eps = 1e-6
    
    def safe_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Ultra-safe sigmoid with strict clamping."""
        return torch.sigmoid(torch.clamp(x, -5.0, 5.0))
    
    def safe_std(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Safe standard deviation that can't return 0."""
        std_val = torch.std(x, dim=dim)
        return torch.clamp(std_val, min=self.eps)
    
    def calculate_blend_weights(self, 
                                item_betas: torch.Tensor,
                                ordinal_taus: torch.Tensor,
                                student_abilities: torch.Tensor,
                                discrimination_alphas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ultra-safe blend weight calculation."""
        
        # Input validation
        if torch.isnan(item_betas).any() or torch.isinf(item_betas).any():
            print("Warning: item_betas contains NaN/Inf")
            batch_size, seq_len = student_abilities.shape
            return torch.full((batch_size, seq_len, self.n_categories), 0.5, 
                            device=student_abilities.device, dtype=student_abilities.dtype)
        
        if torch.isnan(ordinal_taus).any() or torch.isinf(ordinal_taus).any():
            print("Warning: ordinal_taus contains NaN/Inf")
            batch_size, seq_len = student_abilities.shape
            return torch.full((batch_size, seq_len, self.n_categories), 0.5, 
                            device=student_abilities.device, dtype=student_abilities.dtype)
        
        batch_size, seq_len = student_abilities.shape
        
        # Clamp parameters to very conservative range
        range_sens = torch.clamp(self.range_sensitivity, 0.001, 0.1)
        dist_sens = torch.clamp(self.distance_sensitivity, 0.001, 0.1)
        bias = torch.clamp(self.baseline_bias, -0.5, 0.5)
        
        # Simple threshold distance (much safer)
        ordinal_taus_expanded = ordinal_taus.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )
        
        # Basic distance calculation
        distances = torch.abs(ordinal_taus_expanded - item_betas)
        avg_distance = torch.mean(distances, dim=-1, keepdim=True)  # (B, S, 1)
        
        # LEARNABLE ADAPTIVE CATEGORY-SPECIFIC LOGIC
        # Let the model learn optimal category-specific biases through gradient descent
        
        # Calculate per-category adaptive weights
        category_weights = torch.zeros(batch_size, seq_len, self.n_categories, 
                                     device=student_abilities.device, dtype=student_abilities.dtype)
        
        # Base weight calculation using distance and parameters
        base_distance_effect = range_sens * torch.clamp(avg_distance.squeeze(-1), 0.0, 1.0)  # (B, S)
        
        # Apply learnable category-specific biases
        for cat in range(self.n_categories):
            # Each category gets its own learnable bias that the model can optimize
            category_input = base_distance_effect + bias + self.category_biases[cat]
            category_weights[:, :, cat] = torch.clamp(self.safe_sigmoid(category_input), 0.1, 0.9)
        
        return category_weights