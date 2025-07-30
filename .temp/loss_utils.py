"""
Loss Function Utilities for Deep-GPCM
Consolidated loss function creation and testing utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class OrdinalLoss(nn.Module):
    """
    Ordinal loss function for polytomous responses.
    
    Respects the ordering of categories by penalizing predictions
    that violate ordinal relationships.
    """
    
    def __init__(self, n_cats: int, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_cats = n_cats
        self.weight = weight
        
        # Create ordinal penalty matrix
        penalty_matrix = torch.zeros(n_cats, n_cats)
        for i in range(n_cats):
            for j in range(n_cats):
                penalty_matrix[i, j] = abs(i - j)
        
        self.register_buffer('penalty_matrix', penalty_matrix)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal loss.
        
        Args:
            predictions: Model predictions [batch_size, n_cats]
            targets: Target categories [batch_size]
            
        Returns:
            Ordinal loss value
        """
        batch_size = predictions.size(0)
        
        # Get penalty weights for each prediction
        target_penalties = self.penalty_matrix[targets]  # [batch_size, n_cats]
        
        # Compute weighted cross-entropy with ordinal penalties
        log_probs = F.log_softmax(predictions, dim=-1)
        weighted_log_probs = log_probs * target_penalties
        
        # Sum over categories and average over batch
        loss = -weighted_log_probs.sum(dim=-1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    
    Focuses learning on hard examples by down-weighting
    well-classified examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions [batch_size, n_cats]
            targets: Target categories [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, weight=self.weight, reduction='none')
        
        # Compute probabilities and focal weights
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class MSELossWrapper(nn.Module):
    """
    MSE loss wrapper for compatibility with classification interface.
    
    Treats categories as continuous values for regression-like training.
    """
    
    def __init__(self, n_cats: int):
        super().__init__()
        self.n_cats = n_cats
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, n_cats] or [batch_size, n_cats]
            targets: Target categories [batch_size, seq_len] or [batch_size]
            
        Returns:
            MSE loss value
        """
        # Handle different input shapes
        if predictions.dim() == 3:  # [batch_size, seq_len, n_cats]
            batch_size, seq_len, n_cats = predictions.shape
            predictions = predictions.view(-1, n_cats)
            targets = targets.view(-1)
        
        # Convert predictions to expected values
        category_values = torch.arange(self.n_cats, dtype=predictions.dtype, device=predictions.device)
        predicted_values = torch.sum(F.softmax(predictions, dim=-1) * category_values, dim=-1)
        
        # Convert targets to float
        target_values = targets.float()
        
        return self.mse_loss(predicted_values, target_values)


class CrossEntropyWrapper(nn.Module):
    """
    Cross-entropy loss wrapper for unified interface.
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, n_cats] or [batch_size, n_cats]
            targets: Target categories [batch_size, seq_len] or [batch_size]
            
        Returns:
            Cross-entropy loss value
        """
        # Handle different input shapes
        if predictions.dim() == 3:  # [batch_size, seq_len, n_cats]
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets.view(-1)
        
        return F.cross_entropy(predictions, targets, weight=self.weight)


def create_loss_function(loss_type: str, n_cats: int, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
        n_cats: Number of categories
        **kwargs: Additional parameters for loss function
        
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'crossentropy':
        return CrossEntropyWrapper(weight=kwargs.get('weight'))
    
    elif loss_type == 'ordinal':
        return OrdinalLoss(n_cats, weight=kwargs.get('weight'))
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha, gamma, weight=kwargs.get('weight'))
    
    elif loss_type == 'mse':
        return MSELossWrapper(n_cats)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compare_loss_functions(predictions: torch.Tensor, targets: torch.Tensor, 
                          n_cats: int) -> dict:
    """
    Compare different loss functions on the same data.
    
    Args:
        predictions: Model predictions [batch_size, n_cats]
        targets: Target categories [batch_size]
        n_cats: Number of categories
        
    Returns:
        Dictionary of loss values
    """
    results = {}
    
    # Test different loss functions
    loss_configs = [
        ('CrossEntropy', 'crossentropy', {}),
        ('Ordinal', 'ordinal', {}),
        ('Focal (γ=2)', 'focal', {'gamma': 2.0}),
        ('Focal (γ=5)', 'focal', {'gamma': 5.0}),
        ('MSE', 'mse', {})
    ]
    
    for name, loss_type, kwargs in loss_configs:
        try:
            loss_fn = create_loss_function(loss_type, n_cats, **kwargs)
            loss_value = loss_fn(predictions, targets).item()
            results[name] = loss_value
        except Exception as e:
            results[name] = f"Error: {e}"
    
    return results


def test_loss_functions():
    """Test all loss functions with synthetic data."""
    print("🧪 Testing loss functions...")
    
    # Create test data
    batch_size, n_cats = 32, 4
    predictions = torch.randn(batch_size, n_cats)
    targets = torch.randint(0, n_cats, (batch_size,))
    
    print(f"Test data: {batch_size} samples, {n_cats} categories")
    
    # Test each loss function
    loss_results = compare_loss_functions(predictions, targets, n_cats)
    
    print("\nLoss Function Comparison:")
    for name, value in loss_results.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    # Test ordinal properties
    print("\n🔍 Testing ordinal properties...")
    
    # Create predictions that should have different ordinal losses
    perfect_pred = torch.zeros(1, n_cats)
    perfect_pred[0, 2] = 10.0  # Perfect prediction for category 2
    
    wrong_adjacent = torch.zeros(1, n_cats)
    wrong_adjacent[0, 1] = 10.0  # Predict adjacent category
    
    wrong_distant = torch.zeros(1, n_cats)
    wrong_distant[0, 0] = 10.0  # Predict distant category
    
    target = torch.tensor([2])
    
    ordinal_loss = create_loss_function('ordinal', n_cats)
    
    perfect_loss = ordinal_loss(perfect_pred, target).item()
    adjacent_loss = ordinal_loss(wrong_adjacent, target).item()
    distant_loss = ordinal_loss(wrong_distant, target).item()
    
    print(f"Perfect prediction loss: {perfect_loss:.4f}")
    print(f"Adjacent category loss: {adjacent_loss:.4f}")
    print(f"Distant category loss: {distant_loss:.4f}")
    
    # Verify ordinal property
    assert perfect_loss < adjacent_loss < distant_loss, "Ordinal property not satisfied!"
    print("✅ Ordinal property verified!")
    
    print("✅ All loss function tests passed!")


def get_loss_recommendations(n_cats: int, class_distribution: np.ndarray = None) -> dict:
    """
    Get loss function recommendations based on data characteristics.
    
    Args:
        n_cats: Number of categories
        class_distribution: Distribution of classes (optional)
        
    Returns:
        Dictionary with recommendations
    """
    recommendations = {}
    
    # Check class imbalance
    if class_distribution is not None:
        proportions = class_distribution / class_distribution.sum()
        max_prop = proportions.max()
        min_prop = proportions.min()
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        if imbalance_ratio > 3.0:
            recommendations['primary'] = 'focal'
            recommendations['reason'] = f'High class imbalance (ratio: {imbalance_ratio:.1f})'
            recommendations['params'] = {'gamma': 2.0}
        else:
            recommendations['primary'] = 'crossentropy'
            recommendations['reason'] = 'Balanced classes'
            recommendations['params'] = {}
    else:
        recommendations['primary'] = 'crossentropy'
        recommendations['reason'] = 'Standard choice for classification'
        recommendations['params'] = {}
    
    # Always recommend ordinal as secondary for polytomous data
    if n_cats > 2:
        recommendations['secondary'] = 'ordinal'
        recommendations['ordinal_reason'] = 'Respects category ordering in polytomous data'
    
    # Alternative suggestions
    recommendations['alternatives'] = [
        ('focal', 'For handling class imbalance'),
        ('mse', 'For regression-like training'),
        ('ordinal', 'For respecting category ordering')
    ]
    
    return recommendations


if __name__ == "__main__":
    # Run comprehensive tests
    test_loss_functions()
    
    # Show recommendations
    print("\n📋 Loss Function Recommendations:")
    print("-" * 40)
    
    # Balanced case
    balanced_dist = np.array([25, 25, 25, 25])
    rec_balanced = get_loss_recommendations(4, balanced_dist)
    print(f"Balanced data: {rec_balanced['primary']} ({rec_balanced['reason']})")
    
    # Imbalanced case
    imbalanced_dist = np.array([80, 15, 3, 2])
    rec_imbalanced = get_loss_recommendations(4, imbalanced_dist)
    print(f"Imbalanced data: {rec_imbalanced['primary']} ({rec_imbalanced['reason']})")