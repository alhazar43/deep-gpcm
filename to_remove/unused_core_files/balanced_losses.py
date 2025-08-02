"""Enhanced loss functions for addressing prediction imbalance in middle categories."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class MiddleCategoryBoostLoss(nn.Module):
    """Cross-entropy loss with specific boosting for middle categories.
    
    This loss addresses the common problem where models predict extreme 
    categories well but struggle with middle categories in ordinal tasks.
    """
    
    def __init__(self, n_cats: int, middle_boost: float = 2.0, extreme_penalty: float = 0.8):
        """Initialize middle category boost loss.
        
        Args:
            n_cats: Number of ordinal categories
            middle_boost: Multiplier for middle category losses (>1.0 to boost)
            extreme_penalty: Multiplier for extreme category losses (<1.0 to reduce)
        """
        super().__init__()
        self.n_cats = n_cats
        self.middle_boost = middle_boost
        self.extreme_penalty = extreme_penalty
        
        # Create category weights - boost middle, reduce extremes
        weights = torch.ones(n_cats)
        if n_cats >= 4:
            # Boost middle categories
            for i in range(1, n_cats - 1):
                weights[i] = middle_boost
            # Reduce extreme categories  
            weights[0] = extreme_penalty
            weights[-1] = extreme_penalty
        
        self.register_buffer('category_weights', weights)
    
    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute middle-category-boosted cross-entropy loss."""
        # Convert to logits if probabilities given
        if pred_probs.dim() == 3:
            batch_size, seq_len, n_cats = pred_probs.shape
            pred_probs_flat = pred_probs.view(-1, n_cats)
            targets_flat = targets.view(-1)
        else:
            pred_probs_flat = pred_probs
            targets_flat = targets
        
        # Compute cross-entropy loss without reduction
        log_probs = torch.log(pred_probs_flat + 1e-8)
        ce_loss = F.nll_loss(log_probs, targets_flat, reduction='none')
        
        # Apply category-specific weights
        target_weights = self.category_weights[targets_flat]
        weighted_loss = ce_loss * target_weights
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            weighted_loss = weighted_loss * mask_flat
            return weighted_loss.sum() / (mask_flat.sum() + 1e-7)
        else:
            return weighted_loss.mean()


class AdaptiveThresholdLoss(nn.Module):
    """Loss that adapts decision thresholds during training.
    
    Learns optimal decision boundaries for each category transition
    to better separate middle categories.
    """
    
    def __init__(self, n_cats: int, initial_temp: float = 1.0):
        """Initialize adaptive threshold loss.
        
        Args:
            n_cats: Number of ordinal categories
            initial_temp: Initial temperature for threshold adaptation
        """
        super().__init__()
        self.n_cats = n_cats
        
        # Learnable thresholds for category boundaries
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, n_cats - 1))
        self.temperature = nn.Parameter(torch.tensor(initial_temp))
    
    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adaptive threshold loss."""
        if pred_probs.dim() == 3:
            batch_size, seq_len, n_cats = pred_probs.shape
            pred_probs_flat = pred_probs.view(-1, n_cats)
            targets_flat = targets.view(-1)
        else:
            pred_probs_flat = pred_probs
            targets_flat = targets
        
        # Convert probabilities to cumulative form
        cum_probs = torch.cumsum(pred_probs_flat, dim=1)
        
        # Apply learnable thresholds with temperature scaling
        threshold_logits = self.thresholds / torch.clamp(self.temperature, min=0.1)
        threshold_probs = torch.sigmoid(threshold_logits)
        
        # Create target cumulative probabilities
        targets_cum = torch.zeros_like(cum_probs)
        for i in range(self.n_cats - 1):
            targets_cum[:, i] = (targets_flat > i).float()
        
        # MSE loss between predicted and target cumulative probabilities
        mse_loss = F.mse_loss(cum_probs[:, :-1], targets_cum[:, :-1], reduction='none')
        
        # Weight by inverse frequency to boost rare categories
        category_counts = torch.bincount(targets_flat, minlength=self.n_cats).float()
        category_weights = 1.0 / (category_counts + 1e-7)
        category_weights = category_weights / category_weights.sum() * self.n_cats
        
        target_weights = category_weights[targets_flat]
        weighted_loss = (mse_loss.mean(dim=1) * target_weights)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            weighted_loss = weighted_loss * mask_flat
            return weighted_loss.sum() / (mask_flat.sum() + 1e-7)
        else:
            return weighted_loss.mean()


class FocalOrdinalLoss(nn.Module):
    """Focal loss adapted for ordinal classification with middle category focus.
    
    Combines focal loss (to handle easy/hard examples) with ordinal distance
    weighting to specifically improve middle category predictions.
    """
    
    def __init__(self, n_cats: int, gamma: float = 2.0, alpha: float = 1.0, 
                 ordinal_weight: float = 1.0):
        """Initialize focal ordinal loss.
        
        Args:
            n_cats: Number of ordinal categories
            gamma: Focal loss gamma parameter (higher = more focus on hard examples)
            alpha: Focal loss alpha parameter (class balancing)
            ordinal_weight: Weight for ordinal distance penalty
        """
        super().__init__()
        self.n_cats = n_cats
        self.gamma = gamma
        self.alpha = alpha
        self.ordinal_weight = ordinal_weight
        
        # Create distance matrix for ordinal penalty
        distance_matrix = torch.zeros((n_cats, n_cats))
        for i in range(n_cats):
            for j in range(n_cats):
                distance_matrix[i, j] = abs(i - j)
        self.register_buffer('distance_matrix', distance_matrix)
    
    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal ordinal loss."""
        if pred_probs.dim() == 3:
            batch_size, seq_len, n_cats = pred_probs.shape
            pred_probs_flat = pred_probs.view(-1, n_cats)
            targets_flat = targets.view(-1)
        else:
            pred_probs_flat = pred_probs
            targets_flat = targets
        
        # Compute standard cross-entropy
        log_probs = torch.log(pred_probs_flat + 1e-8)
        ce_loss = F.nll_loss(log_probs, targets_flat, reduction='none')
        
        # Focal loss component
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Ordinal distance penalty
        pred_cats = torch.argmax(pred_probs_flat, dim=1)
        ordinal_distances = self.distance_matrix[targets_flat, pred_cats]
        ordinal_penalty = self.ordinal_weight * ordinal_distances
        
        # Combined loss
        total_loss = focal_loss + ordinal_penalty
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            total_loss = total_loss * mask_flat
            return total_loss.sum() / (mask_flat.sum() + 1e-7)
        else:
            return total_loss.mean()


class BalancedTripleLoss(nn.Module):
    """Enhanced triple loss (CE + QWK + CORAL) with middle category balancing."""
    
    def __init__(self, n_cats: int, 
                 ce_weight: float = 0.4,
                 qwk_weight: float = 0.3, 
                 coral_weight: float = 0.3,
                 middle_boost: float = 1.5):
        """Initialize balanced triple loss.
        
        Args:
            n_cats: Number of ordinal categories
            ce_weight: Weight for cross-entropy component
            qwk_weight: Weight for QWK component  
            coral_weight: Weight for CORAL component
            middle_boost: Boost factor for middle categories
        """
        super().__init__()
        self.n_cats = n_cats
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        self.coral_weight = coral_weight
        
        # Components
        self.middle_ce = MiddleCategoryBoostLoss(n_cats, middle_boost=middle_boost)
        
        from .ordinal_losses import DifferentiableQWKLoss
        self.qwk_loss = DifferentiableQWKLoss(n_cats)
        
    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor,
                coral_info: Optional[Dict[str, Any]] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute balanced triple loss."""
        
        # 1. Middle-category-boosted CE loss
        ce_loss = self.middle_ce(pred_probs, targets, mask)
        
        # 2. QWK loss for ordinal consistency
        qwk_loss = self.qwk_loss(pred_probs, targets, mask)
        
        # 3. CORAL loss if available
        coral_loss = 0.0
        if coral_info is not None and 'logits' in coral_info:
            from .ordinal_losses import compute_coral_loss
            coral_loss = compute_coral_loss(coral_info['logits'], targets, mask)
        
        # Combine losses
        total_loss = (self.ce_weight * ce_loss + 
                     self.qwk_weight * qwk_loss + 
                     self.coral_weight * coral_loss)
        
        return total_loss


def test_balanced_losses():
    """Test balanced loss functions."""
    print("Testing Middle Category Boost Loss...")
    
    # Test data
    batch_size, seq_len, n_cats = 32, 10, 4
    pred_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats), dim=-1)
    targets = torch.randint(0, n_cats, (batch_size, seq_len))
    
    # Test middle category boost
    boost_loss = MiddleCategoryBoostLoss(n_cats, middle_boost=2.0)
    loss_val = boost_loss(pred_probs, targets)
    print(f"Middle Boost Loss: {loss_val:.4f}")
    
    # Test focal ordinal
    focal_loss = FocalOrdinalLoss(n_cats, gamma=2.0)
    loss_val = focal_loss(pred_probs, targets)
    print(f"Focal Ordinal Loss: {loss_val:.4f}")
    
    # Test adaptive threshold
    adapt_loss = AdaptiveThresholdLoss(n_cats)
    loss_val = adapt_loss(pred_probs, targets)
    print(f"Adaptive Threshold Loss: {loss_val:.4f}")
    
    print("✅ All balanced losses working!")


if __name__ == "__main__":
    test_balanced_losses()