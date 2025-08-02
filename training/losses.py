"""Unified loss functions for Deep-GPCM models.

This module provides a comprehensive collection of loss functions for knowledge
tracing and ordinal regression tasks, including:
- Standard losses (CrossEntropy, Weighted CE, Focal)
- Ordinal-specific losses (QWK, EMD, Ordinal CE)
- Combined losses for multi-objective optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple


# ============================================================================
# Base Loss Functions
# ============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for class imbalance."""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
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


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in ordinal/categorical problems.
    
    The focal loss is designed to address class imbalance by down-weighting
    easy examples and focusing training on hard negatives.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, 
                 gamma: float = 2.0, 
                 alpha: Optional[Union[float, torch.Tensor]] = None,
                 n_cats: Optional[int] = None,
                 reduction: str = 'mean'):
        """Initialize focal loss.
        
        Args:
            gamma: Focusing parameter (γ ≥ 0). Higher γ increases focus on hard examples.
            alpha: Weighting factor in [0, 1] to balance positive/negative examples.
                   Can be a scalar or a tensor of size n_cats for per-class weights.
            n_cats: Number of categories (required if alpha is a scalar)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Handle alpha parameter
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                # Create balanced alpha weights
                if n_cats is None:
                    raise ValueError("n_cats required when alpha is scalar")
                self.alpha = torch.ones(n_cats) * alpha
            else:
                # Use provided per-class weights
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            predictions: Predicted logits, shape (..., n_cats)
            targets: Ground truth labels, shape (...)
            mask: Optional mask for valid positions
            
        Returns:
            Focal loss value
        """
        # Get device
        device = predictions.device
        
        # Move alpha to correct device if needed
        if self.alpha is not None and self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        
        # Compute cross entropy loss without reduction
        ce_loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), 
                                  targets.view(-1), 
                                  reduction='none')
        
        # Get predicted probabilities for true class
        p = torch.exp(-ce_loss)
        
        # Compute focal term: (1 - p)^gamma
        focal_term = (1 - p) ** self.gamma
        
        # Apply focal term to cross entropy loss
        focal_loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Gather alpha values for each target
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            focal_loss = focal_loss * mask_flat
            
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask_flat.sum() + 1e-7)
            elif self.reduction == 'sum':
                return focal_loss.sum()
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
        
        return focal_loss


# ============================================================================
# Ordinal-Specific Loss Functions
# ============================================================================

class DifferentiableQWKLoss(nn.Module):
    """Differentiable Quadratic Weighted Kappa loss for ordinal regression.
    
    This loss directly optimizes QWK by computing a soft confusion matrix
    from predicted probabilities and using it to calculate a differentiable
    approximation of QWK.
    """
    
    def __init__(self, n_cats: int, epsilon: float = 1e-7):
        """Initialize QWK loss.
        
        Args:
            n_cats: Number of ordinal categories
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.n_cats = n_cats
        self.epsilon = epsilon
        
        # Pre-compute weight matrix
        self.weight_matrix = self._create_weight_matrix(n_cats)
    
    def _create_weight_matrix(self, n_cats: int) -> torch.Tensor:
        """Create quadratic weight matrix for QWK.
        
        Args:
            n_cats: Number of categories
            
        Returns:
            Weight matrix of shape (n_cats, n_cats)
        """
        weights = torch.zeros((n_cats, n_cats))
        for i in range(n_cats):
            for j in range(n_cats):
                weights[i, j] = (i - j) ** 2 / (n_cats - 1) ** 2
        return weights
    
    def forward(self, 
                pred_probs: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute differentiable QWK loss.
        
        Args:
            pred_probs: Predicted probabilities, shape (batch_size, seq_len, n_cats)
            targets: True categories, shape (batch_size, seq_len)
            mask: Optional mask for valid positions
            
        Returns:
            Negative QWK as loss (to minimize)
        """
        # Move weight matrix to correct device
        if self.weight_matrix.device != pred_probs.device:
            self.weight_matrix = self.weight_matrix.to(pred_probs.device)
        
        # Flatten predictions and targets
        batch_size, seq_len = targets.shape
        pred_flat = pred_probs.view(-1, self.n_cats)
        targets_flat = targets.view(-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            pred_flat = pred_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets_flat, num_classes=self.n_cats).float()
        
        # Compute soft confusion matrix
        # conf_matrix[i, j] = sum over samples of P(pred=j) * I(true=i)
        conf_matrix = torch.matmul(targets_onehot.T, pred_flat)
        
        # Normalize confusion matrix
        conf_matrix = conf_matrix / (conf_matrix.sum() + self.epsilon)
        
        # Compute observed and expected agreement
        po = self._compute_observed_agreement(conf_matrix)
        pe = self._compute_expected_agreement(conf_matrix)
        
        # Compute QWK
        qwk = (po - pe) / (1 - pe + self.epsilon)
        
        # Return negative QWK as loss
        return -qwk
    
    def _compute_observed_agreement(self, conf_matrix: torch.Tensor) -> torch.Tensor:
        """Compute weighted observed agreement.
        
        Args:
            conf_matrix: Normalized confusion matrix
            
        Returns:
            Observed agreement
        """
        # Weight the confusion matrix
        weighted_conf = conf_matrix * (1 - self.weight_matrix)
        
        # Sum all weighted agreements
        return weighted_conf.sum()
    
    def _compute_expected_agreement(self, conf_matrix: torch.Tensor) -> torch.Tensor:
        """Compute expected agreement by chance.
        
        Args:
            conf_matrix: Normalized confusion matrix
            
        Returns:
            Expected agreement
        """
        # Marginal probabilities
        row_marginals = conf_matrix.sum(dim=1)
        col_marginals = conf_matrix.sum(dim=0)
        
        # Expected confusion matrix
        expected_conf = torch.outer(row_marginals, col_marginals)
        
        # Weight the expected matrix
        weighted_expected = expected_conf * (1 - self.weight_matrix)
        
        return weighted_expected.sum()


class OrdinalEMDLoss(nn.Module):
    """Earth Mover's Distance (EMD) loss for ordinal regression.
    
    EMD measures the minimum cost to transform predicted distribution
    to true distribution, naturally handling ordinal relationships.
    """
    
    def __init__(self, n_cats: int):
        """Initialize EMD loss.
        
        Args:
            n_cats: Number of ordinal categories
        """
        super().__init__()
        self.n_cats = n_cats
    
    def forward(self, 
                pred_probs: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute EMD loss.
        
        Args:
            pred_probs: Predicted probabilities, shape (batch_size, seq_len, n_cats)
            targets: True categories, shape (batch_size, seq_len)
            mask: Optional mask for valid positions
            
        Returns:
            EMD loss
        """
        # Convert targets to cumulative format
        batch_size, seq_len = targets.shape
        device = pred_probs.device
        
        # Create cumulative target distribution
        # cum_target[i] = 1 if true_class <= i, else 0
        cum_targets = torch.zeros(batch_size, seq_len, self.n_cats, device=device)
        for k in range(self.n_cats):
            cum_targets[..., k] = (targets <= k).float()
        
        # Compute cumulative predicted distribution
        cum_preds = torch.cumsum(pred_probs, dim=-1)
        
        # EMD is the L1 distance between cumulative distributions
        emd = torch.abs(cum_preds - cum_targets).sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            emd = emd * mask
            return emd.sum() / (mask.sum() + 1e-7)
        else:
            return emd.mean()


class OrdinalCrossEntropyLoss(nn.Module):
    """Ordinal-aware cross-entropy loss with distance weighting.
    
    This loss weights prediction errors by their ordinal distance,
    penalizing predictions far from the true category more heavily.
    """
    
    def __init__(self, n_cats: int, alpha: float = 1.0):
        """Initialize ordinal cross-entropy loss.
        
        Args:
            n_cats: Number of ordinal categories
            alpha: Weight factor for ordinal distance (higher = more penalty)
        """
        super().__init__()
        self.n_cats = n_cats
        self.alpha = alpha
        
        # Pre-compute distance matrix
        self.distance_matrix = self._create_distance_matrix(n_cats)
    
    def _create_distance_matrix(self, n_cats: int) -> torch.Tensor:
        """Create ordinal distance matrix.
        
        Args:
            n_cats: Number of categories
            
        Returns:
            Distance matrix of shape (n_cats, n_cats)
        """
        distances = torch.zeros((n_cats, n_cats))
        for i in range(n_cats):
            for j in range(n_cats):
                distances[i, j] = abs(i - j)
        return distances
    
    def forward(self, 
                pred_probs: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute ordinal-aware cross-entropy loss.
        
        Args:
            pred_probs: Predicted probabilities, shape (batch_size, seq_len, n_cats)
            targets: True categories, shape (batch_size, seq_len)
            mask: Optional mask for valid positions
            
        Returns:
            Ordinal cross-entropy loss
        """
        # Move distance matrix to correct device
        if self.distance_matrix.device != pred_probs.device:
            self.distance_matrix = self.distance_matrix.to(pred_probs.device)
        
        # Get batch dimensions
        batch_size, seq_len = targets.shape
        
        # Create weight matrix based on true labels
        # weights[i, j] = 1 + alpha * distance(true_label[i], j)
        weights = torch.zeros_like(pred_probs)
        for b in range(batch_size):
            for s in range(seq_len):
                true_cat = targets[b, s]
                weights[b, s] = 1 + self.alpha * self.distance_matrix[true_cat]
        
        # Compute weighted cross-entropy
        # Avoid log(0) by adding small epsilon
        log_probs = torch.log(pred_probs + 1e-8)
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes=self.n_cats).float()
        
        # Weighted negative log-likelihood
        loss = -torch.sum(weights * targets_onehot * log_probs, dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-7)
        else:
            return loss.mean()


# ============================================================================
# Combined Loss Functions
# ============================================================================

class CombinedOrdinalLoss(nn.Module):
    """Combined loss function for ordinal regression.
    
    Combines multiple loss components with learnable or fixed weights.
    """
    
    def __init__(self, 
                 n_cats: int,
                 ce_weight: float = 1.0,
                 qwk_weight: float = 0.5,
                 coral_weight: float = 0.0,
                 focal_weight: float = 0.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: Optional[float] = None):
        """Initialize combined loss.
        
        Args:
            n_cats: Number of ordinal categories
            ce_weight: Weight for cross-entropy loss
            qwk_weight: Weight for QWK loss
            coral_weight: Weight for CORAL-specific loss
            focal_weight: Weight for focal loss
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss
        """
        super().__init__()
        self.n_cats = n_cats
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.qwk_loss = DifferentiableQWKLoss(n_cats) if qwk_weight > 0 else None
        self.focal_loss = FocalLoss(focal_gamma, focal_alpha, n_cats) if focal_weight > 0 else None
        
        # Weights
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        self.coral_weight = coral_weight
        self.focal_weight = focal_weight
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                coral_info: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            predictions: Either logits or probabilities, shape (batch_size, seq_len, n_cats)
            targets: True categories, shape (batch_size, seq_len)
            mask: Optional mask for valid positions
            coral_info: Optional CORAL-specific information (logits, etc.)
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Convert logits to probabilities if needed
        if predictions.shape[-1] == self.n_cats and not (predictions >= 0).all():
            # Likely logits
            probs = F.softmax(predictions, dim=-1)
            logits = predictions
        else:
            # Already probabilities
            probs = predictions
            logits = torch.log(probs + 1e-8)
        
        losses = {}
        total_loss = 0
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(logits.view(-1, self.n_cats), targets.view(-1))
            if mask is not None:
                ce_loss = ce_loss * mask.view(-1)
                ce_loss = ce_loss.sum() / (mask.sum() + 1e-7)
            else:
                ce_loss = ce_loss.mean()
            
            losses['ce_loss'] = ce_loss
            total_loss += self.ce_weight * ce_loss
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal_loss = self.focal_loss(logits, targets, mask)
            losses['focal_loss'] = focal_loss
            total_loss += self.focal_weight * focal_loss
        
        # QWK loss
        if self.qwk_weight > 0 and self.qwk_loss is not None:
            qwk_loss = self.qwk_loss(probs, targets, mask)
            losses['qwk_loss'] = qwk_loss
            total_loss += self.qwk_weight * qwk_loss
        
        # EMD loss
        
        # CORAL-specific loss
        if self.coral_weight > 0 and coral_info is not None:
            # Use CORAL cumulative logits if available
            try:
                from models.components.coral_layers import CORALCompatibleLoss
                coral_loss_fn = CORALCompatibleLoss(self.n_cats)
                coral_loss = coral_loss_fn((probs, coral_info), targets, mask)
                losses['coral_loss'] = coral_loss
                total_loss += self.coral_weight * coral_loss
            except ImportError:
                # CORAL layers not available, skip
                pass
        
        losses['total_loss'] = total_loss
        return losses


# ============================================================================
# Factory Functions
# ============================================================================

def create_loss_function(loss_type: str, n_cats: int, **kwargs) -> nn.Module:
    """Create loss function based on type.
    
    Args:
        loss_type: Type of loss function
        n_cats: Number of categories
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(kwargs.get('class_weights'))
    elif loss_type == 'focal':
        return FocalLoss(
            gamma=kwargs.get('gamma', 2.0),
            alpha=kwargs.get('alpha', None),
            n_cats=n_cats
        )
    elif loss_type == 'qwk':
        return DifferentiableQWKLoss(n_cats)
    elif loss_type == 'emd':
        return OrdinalEMDLoss(n_cats)
    elif loss_type == 'ordinal_ce':
        return OrdinalCrossEntropyLoss(n_cats, alpha=kwargs.get('alpha', 1.0))
    elif loss_type == 'combined':
        return CombinedOrdinalLoss(
            n_cats,
            ce_weight=kwargs.get('ce_weight', 1.0),
            qwk_weight=kwargs.get('qwk_weight', 0.5),
            coral_weight=kwargs.get('coral_weight', 0.4),
            focal_weight=kwargs.get('focal_weight', 0.0),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            focal_alpha=kwargs.get('focal_alpha', None)
        )
    elif loss_type == 'triple_coral':
        # Triple CORAL loss: CE + QWK + CORAL with balanced weights
        return CombinedOrdinalLoss(
            n_cats,
            ce_weight=0.4,
            qwk_weight=0.3,
            coral_weight=0.3
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# Testing Functions
# ============================================================================

def test_all_losses():
    """Test all loss functions."""
    print("Testing all loss functions...")
    
    # Test data
    batch_size, seq_len, n_cats = 2, 5, 4
    logits = torch.randn(batch_size, seq_len, n_cats, requires_grad=True)
    pred_probs = F.softmax(logits, dim=-1)
    targets = torch.randint(0, n_cats, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    mask[0, 3:] = 0  # Mask out some positions
    
    # Test all loss types
    loss_types = ['ce', 'weighted_ce', 'focal', 'qwk', 'emd', 'ordinal_ce', 'combined', 'triple_coral']
    
    for loss_type in loss_types:
        print(f"\nTesting {loss_type} loss...")
        
        # Create loss function
        kwargs = {}
        if loss_type == 'weighted_ce':
            kwargs['class_weights'] = torch.ones(n_cats)
        
        loss_fn = create_loss_function(loss_type, n_cats, **kwargs)
        
        # Compute loss
        if loss_type in ['ce', 'weighted_ce', 'focal']:
            # These expect logits
            loss = loss_fn(logits.view(-1, n_cats), targets.view(-1))
        elif loss_type in ['qwk', 'emd', 'ordinal_ce']:
            # These expect probabilities and mask
            loss = loss_fn(pred_probs, targets, mask)
        elif loss_type in ['combined', 'triple_coral']:
            # Combined loss returns dictionary
            loss_dict = loss_fn(logits, targets, mask)
            loss = loss_dict['total_loss']
            print(f"  Components: {', '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items() if k != 'total_loss')}")
        
        print(f"  Loss value: {loss.item():.4f}")
        
        # Test backward pass
        if logits.grad is not None:
            logits.grad.zero_()
        loss.backward(retain_graph=True)
        print(f"  Gradient norm: {logits.grad.norm().item():.4f}")
    
    print("\n✓ All losses tested successfully!")


if __name__ == "__main__":
    test_all_losses()