"""Loss functions for knowledge tracing models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """Ordinal loss function for GPCM."""
    
    def __init__(self, n_cats: int):
        super().__init__()
        self.n_cats = n_cats
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, n_cats)
            targets: Ground truth labels, shape (batch_size,)
            
        Returns:
            Ordinal loss value
        """
        batch_size, K = predictions.shape
        
        # Create cumulative probabilities P(Y <= k)
        cum_probs = torch.cumsum(predictions, dim=1)
        
        # Create mask for I(y <= k)
        mask = torch.arange(K, device=targets.device).expand(batch_size, K) <= targets.unsqueeze(1)
        
        # Calculate ordinal loss
        loss = -torch.sum(
            mask * torch.log(cum_probs + 1e-9) + 
            (1 - mask.float()) * torch.log(1 - cum_probs + 1e-9)
        )
        
        return loss / (batch_size * (K - 1))


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, n_cats)
            targets: Ground truth labels, shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Convert to probabilities
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for class imbalance."""
    
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.
        
        Args:
            predictions: Predicted logits, shape (batch_size, n_cats)
            targets: Ground truth labels, shape (batch_size,)
            
        Returns:
            Weighted cross-entropy loss
        """
        return F.cross_entropy(predictions, targets, weight=self.class_weights)