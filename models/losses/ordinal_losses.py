"""
Ordinal-aware loss functions for GPCM models.
Implements losses that respect ordinal relationships between categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with ordinal distance weighting."""
    
    def __init__(self, n_cats: int, ordinal_weight: float = 1.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.ordinal_weight = ordinal_weight
        self.reduction = reduction
        
        # Create ordinal distance matrix
        self.register_buffer('ordinal_distances', self._create_ordinal_distances())
        
    def _create_ordinal_distances(self) -> torch.Tensor:
        """Create matrix of ordinal distances between categories."""
        distances = torch.zeros(self.n_cats, self.n_cats)
        for i in range(self.n_cats):
            for j in range(self.n_cats):
                distances[i, j] = abs(i - j)
        return distances
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal cross-entropy loss.
        
        Args:
            logits: [batch, seq, n_cats] predicted logits
            targets: [batch, seq] true categories
            
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, n_cats = logits.shape
        
        # Reshape for processing
        logits_flat = logits.view(-1, n_cats)  # [batch*seq, n_cats]
        targets_flat = targets.view(-1)  # [batch*seq]
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Ordinal penalty: encourage predictions close to true category
        probs = F.softmax(logits_flat, dim=-1)  # [batch*seq, n_cats]
        
        # Calculate expected predicted category
        categories = torch.arange(n_cats, device=logits.device, dtype=torch.float)
        pred_categories = (probs * categories).sum(dim=-1)  # [batch*seq]
        
        # Ordinal distance penalty
        ordinal_penalty = (pred_categories - targets_flat.float()).abs()
        
        # Combined loss
        total_loss = ce_loss + self.ordinal_weight * ordinal_penalty
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss.view(batch_size, seq_len)


class QWKLoss(nn.Module):
    """Loss function that directly optimizes Quadratic Weighted Kappa."""
    
    def __init__(self, n_cats: int, reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.reduction = reduction
        
        # Create QWK weight matrix
        self.register_buffer('qwk_weights', self._create_qwk_weights())
        
    def _create_qwk_weights(self) -> torch.Tensor:
        """Create QWK weight matrix."""
        weights = torch.zeros(self.n_cats, self.n_cats)
        for i in range(self.n_cats):
            for j in range(self.n_cats):
                weights[i, j] = 1.0 - ((i - j) ** 2) / ((self.n_cats - 1) ** 2)
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute QWK loss.
        
        Args:
            logits: [batch, seq, n_cats] predicted logits
            targets: [batch, seq] true categories
            
        Returns:
            loss: 1 - QWK (to minimize)
        """
        batch_size, seq_len, n_cats = logits.shape
        
        # Get predictions
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # Remove padding (assuming 0 is padding)
        mask = targets_flat > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        preds_masked = preds_flat[mask]
        targets_masked = targets_flat[mask]
        
        # Calculate QWK
        qwk = self._quadratic_weighted_kappa(targets_masked, preds_masked)
        
        # Return 1 - QWK to minimize
        return 1.0 - qwk
    
    def _quadratic_weighted_kappa(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate QWK using PyTorch tensors."""
        n = len(y_true)
        if n == 0:
            return torch.tensor(0.0, device=y_true.device)
        
        # Create confusion matrix
        confusion_matrix = torch.zeros(self.n_cats, self.n_cats, device=y_true.device)
        for i in range(n):
            confusion_matrix[y_true[i], y_pred[i]] += 1
        
        # Normalize
        confusion_matrix = confusion_matrix / n
        
        # Expected matrix (outer product of marginals)
        observed_marginal_true = confusion_matrix.sum(dim=1)
        observed_marginal_pred = confusion_matrix.sum(dim=0)
        expected_matrix = torch.outer(observed_marginal_true, observed_marginal_pred)
        
        # Move qwk_weights to correct device
        qwk_weights = self.qwk_weights.to(y_true.device)
        
        # Calculate QWK
        numerator = (qwk_weights * confusion_matrix).sum()
        denominator = (qwk_weights * expected_matrix).sum()
        
        if denominator == 0:
            return torch.tensor(0.0, device=y_true.device)
        
        return numerator / denominator


class OrdinalFocalLoss(nn.Module):
    """Focal loss adapted for ordinal classification."""
    
    def __init__(self, n_cats: int, alpha: float = 1.0, gamma: float = 2.0,
                 ordinal_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.n_cats = n_cats
        self.alpha = alpha
        self.gamma = gamma
        self.ordinal_weight = ordinal_weight
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal focal loss."""
        batch_size, seq_len, n_cats = logits.shape
        
        # Reshape
        logits_flat = logits.view(-1, n_cats)
        targets_flat = targets.view(-1)
        
        # Calculate probabilities
        probs = F.softmax(logits_flat, dim=-1)
        target_probs = probs[range(len(targets_flat)), targets_flat]
        
        # Focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Cross-entropy
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Ordinal penalty for hard examples
        pred_categories = probs.argmax(dim=-1)
        ordinal_penalty = (pred_categories - targets_flat).abs().float()
        
        # Combine losses
        total_loss = focal_loss + self.ordinal_weight * focal_weight * ordinal_penalty
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss.view(batch_size, seq_len)


class CombinedOrdinalLoss(nn.Module):
    """Combines multiple ordinal-aware losses."""
    
    def __init__(self, n_cats: int, 
                 ce_weight: float = 1.0,
                 qwk_weight: float = 0.5,
                 ordinal_weight: float = 0.3):
        super().__init__()
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        self.ordinal_weight = ordinal_weight
        
        self.ce_loss = OrdinalCrossEntropyLoss(n_cats, ordinal_weight=ordinal_weight)
        self.qwk_loss = QWKLoss(n_cats)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss and return components."""
        ce = self.ce_loss(logits, targets)
        qwk = self.qwk_loss(logits, targets)
        
        total_loss = self.ce_weight * ce + self.qwk_weight * qwk
        
        loss_dict = {
            'total': total_loss,
            'cross_entropy': ce,
            'qwk_loss': qwk
        }
        
        return total_loss, loss_dict


def create_ordinal_loss(loss_type: str, n_cats: int, **kwargs) -> nn.Module:
    """Factory function to create ordinal losses."""
    if loss_type == 'ordinal_ce':
        return OrdinalCrossEntropyLoss(n_cats, **kwargs)
    elif loss_type == 'qwk':
        return QWKLoss(n_cats, **kwargs)
    elif loss_type == 'ordinal_focal':
        return OrdinalFocalLoss(n_cats, **kwargs)
    elif loss_type == 'combined':
        return CombinedOrdinalLoss(n_cats, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")