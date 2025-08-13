"""
Loss function utilities for Deep-GPCM training.
Comprehensive loss functions compatible with the optimized pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple


# Removed UnifiedLossWrapper - using direct loss functions for better performance


def create_loss_function(loss_type: str, n_cats: int, **kwargs) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'qwk', 'ordinal_ce', 'coral', 'combined')
        n_cats: Number of categories
        **kwargs: Additional loss parameters
        
    Returns:
        Direct loss function without wrapper overhead
    """
    
    loss_type = loss_type.lower()
    
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 1.0),
            gamma=kwargs.get('focal_gamma', 2.0)
        )
    elif loss_type == 'qwk':
        return QWKLoss(n_cats)
    elif loss_type == 'ordinal_ce':
        return OrdinalCrossEntropyLoss(
            n_cats,
            ordinal_weight=kwargs.get('ordinal_weight', 1.0)
        )
    elif loss_type == 'coral':
        return CORALLoss(n_cats)
    elif loss_type == 'combined':
        return CombinedLoss(n_cats, **kwargs)
    else:
        return nn.CrossEntropyLoss()  # Default fallback


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [batch_size, n_classes] or [batch_size, seq_len, n_classes]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]
        """
        # Handle different input shapes
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class QWKLoss(nn.Module):
    """Quadratic Weighted Kappa Loss for ordinal classification."""
    
    def __init__(self, n_cats: int, reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.reduction = reduction
        
        # Create QWK weight matrix
        self.register_buffer('qwk_weights', self._create_qwk_weights())
        
    def _create_qwk_weights(self) -> torch.Tensor:
        """Create QWK weight matrix using vectorized operations."""
        # Vectorized computation - much faster than nested loops
        i_grid, j_grid = torch.meshgrid(
            torch.arange(self.n_cats, dtype=torch.float32),
            torch.arange(self.n_cats, dtype=torch.float32),
            indexing='ij'
        )
        weights = 1.0 - ((i_grid - j_grid) ** 2) / ((self.n_cats - 1) ** 2)
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute QWK loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            loss: 1 - QWK (to minimize)
        """
        # Handle different input shapes
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        # Get predictions
        probs = F.softmax(inputs, dim=-1)
        preds = probs.argmax(dim=-1)
        
        # Calculate QWK
        qwk = self._quadratic_weighted_kappa(targets, preds)
        
        # Return 1 - QWK to minimize
        return 1.0 - qwk
    
    def _quadratic_weighted_kappa(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate QWK using PyTorch tensors with vectorized confusion matrix."""
        n = len(y_true)
        if n == 0:
            return torch.tensor(0.0, device=y_true.device)
        
        # Vectorized confusion matrix computation - MUCH faster!
        # Convert to indices for bincount
        indices = y_true * self.n_cats + y_pred
        bincount = torch.bincount(indices, minlength=self.n_cats * self.n_cats)
        confusion_matrix = bincount.view(self.n_cats, self.n_cats).float()
        
        # Normalize
        confusion_matrix = confusion_matrix / n
        
        # Expected matrix (outer product of marginals)
        observed_marginal_true = confusion_matrix.sum(dim=1)
        observed_marginal_pred = confusion_matrix.sum(dim=0)
        expected_matrix = torch.outer(observed_marginal_true, observed_marginal_pred)
        
        # Ensure QWK weights are on the same device as the confusion matrix
        qwk_weights = self.qwk_weights.to(confusion_matrix.device)
        
        # Calculate QWK using correct formula: κ = (Po - Pe) / (1 - Pe)
        Po = (qwk_weights * confusion_matrix).sum()  # Observed agreement
        Pe = (qwk_weights * expected_matrix).sum()   # Expected agreement
        
        # Add numerical stability with epsilon
        eps = 1e-7
        Pe = torch.clamp(Pe, 0.0, 1.0 - eps)  # Prevent Pe from being too close to 1.0
        
        # Additional safety check
        if Pe >= (1.0 - eps):
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
        
        qwk = (Po - Pe) / (1.0 - Pe + eps)  # Add epsilon for numerical stability
        
        # Clamp QWK to reasonable range to prevent gradient explosion
        qwk = torch.clamp(qwk, -1.0, 1.0)
        return qwk


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with ordinal distance weighting."""
    
    def __init__(self, n_cats: int, ordinal_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.ordinal_weight = ordinal_weight
        self.reduction = reduction
        
        # Pre-compute category indices for efficiency
        self.register_buffer('categories', torch.arange(n_cats, dtype=torch.float32))
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss with optimized operations.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            loss: Scalar loss value
        """
        # Handle different input shapes
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        # Standard cross-entropy (already optimized in PyTorch)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Ordinal penalty with optimized operations
        if self.ordinal_weight > 0:
            # Use log_softmax for numerical stability, then exp for probabilities
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            
            # Vectorized expected category calculation
            pred_categories = torch.sum(probs * self.categories, dim=-1)
            
            # Optimized ordinal distance penalty
            ordinal_penalty = torch.abs(pred_categories - targets.float())
            total_loss = ce_loss + self.ordinal_weight * ordinal_penalty
        else:
            total_loss = ce_loss
        
        # Efficient reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class CombinedLoss(nn.Module):
    """Optimized combined loss function with weighted components."""
    
    def __init__(self, n_cats: int, 
                 ce_weight: float = 0.0,
                 focal_weight: float = 0.0,
                 qwk_weight: float = 0.0,
                 ordinal_weight: float = 0.0,
                 coral_weight: float = 0.0,
                 **kwargs):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.qwk_weight = qwk_weight
        self.ordinal_weight = ordinal_weight
        self.coral_weight = coral_weight
        
        # Pre-compute active components for efficiency
        self.active_components = []
        
        # Only initialize loss functions that will be used (weight > 0)
        if ce_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss()
            self.active_components.append(('ce', ce_weight))
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=kwargs.get('focal_alpha', 1.0),
                gamma=kwargs.get('focal_gamma', 2.0)
            )
            self.active_components.append(('focal', focal_weight))
        
        if qwk_weight > 0:
            self.qwk_loss = QWKLoss(n_cats)
            self.active_components.append(('qwk', qwk_weight))
        
        if ordinal_weight > 0:
            self.ordinal_loss = OrdinalCrossEntropyLoss(
                n_cats, 
                ordinal_weight=kwargs.get('ordinal_penalty', 1.0)
            )
            self.active_components.append(('ordinal', ordinal_weight))
        
        if coral_weight > 0:
            self.coral_loss = CORALLoss(n_cats)
            self.active_components.append(('coral', coral_weight))
        
        # Safety check: ensure at least one component is active
        if not self.active_components:
            # Default to CrossEntropy if no weights specified
            self.ce_loss = nn.CrossEntropyLoss()
            self.active_components.append(('ce', 1.0))
            self.ce_weight = 1.0
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute optimized combined loss using only active components.
        
        Args:
            logits: Model predictions/logits [batch_size, n_classes] or [batch_size, seq_len, n_classes]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]
            
        Returns:
            Combined loss value
        """
        
        # Handle different input shapes
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        
        # Fast path for single component
        if len(self.active_components) == 1:
            component_name, weight = self.active_components[0]
            loss_fn = getattr(self, f'{component_name}_loss')
            return weight * loss_fn(logits, targets)
        
        # Optimized loop only over active components
        total_loss = 0.0
        for component_name, weight in self.active_components:
            loss_fn = getattr(self, f'{component_name}_loss')
            total_loss += weight * loss_fn(logits, targets)
        
        return total_loss


class CORALLoss(nn.Module):
    """CORAL (COnsistent RAnk Logits) Loss for ordinal classification."""
    
    def __init__(self, n_cats: int, reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss using cumulative logits.
        
        Args:
            logits: Model predictions/logits [batch_size, n_classes] or [batch_size, seq_len, n_classes]
            targets: Ground truth labels [batch_size] or [batch_size, seq_len]
            
        Returns:
            CORAL loss value
        """
        
        # Handle different input shapes
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        
        # Convert to cumulative logits if needed
        if logits.size(-1) == self.n_cats:
            # Convert from category logits to cumulative logits
            logits = self._to_cumulative_logits(logits)
        
        # Convert targets to cumulative format
        cum_targets = self._to_cumulative_labels(targets).float()
        
        # Binary cross-entropy on each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            cum_targets, 
            reduction='none'
        )
        
        # Average across thresholds
        loss = loss.mean(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _to_cumulative_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert category logits to cumulative logits for ordinal classification."""
        # For CORAL, cumulative logits represent P(y > k) for each threshold k
        probs = F.softmax(logits, dim=-1)
        
        # Vectorized cumulative probabilities: P(y > k) for k = 0, 1, ..., n_cats-2
        # Use cumulative sum in reverse direction and slice appropriately
        reversed_probs = torch.flip(probs, dims=[-1])
        reversed_cumsum = torch.cumsum(reversed_probs, dim=-1)
        cum_probs = torch.flip(reversed_cumsum, dims=[-1])[:, 1:]  # Exclude last category
        
        # Convert to logits: logit(p) = log(p / (1 - p))
        cum_probs = torch.clamp(cum_probs, min=1e-7, max=1-1e-7)  # Numerical stability
        cum_logits = torch.log(cum_probs / (1 - cum_probs))
        
        return cum_logits
    
    def _to_cumulative_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert ordinal targets to cumulative binary labels using vectorized operations."""
        device = targets.device
        n_samples = targets.size(0)
        
        # Vectorized cumulative label computation - much faster!
        # Create threshold matrix: [batch_size, n_thresholds]
        thresholds = torch.arange(self.n_cats - 1, device=device, dtype=targets.dtype)
        targets_expanded = targets.unsqueeze(1)  # [batch_size, 1]
        
        # Vectorized comparison: targets > thresholds
        cum_labels = (targets_expanded > thresholds).float()
        
        return cum_labels



# Legacy wrapper functions removed - all training scripts now use direct loss functions