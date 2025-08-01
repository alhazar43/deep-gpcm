"""Advanced ordinal loss functions for Deep-GPCM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class CombinedOrdinalLoss(nn.Module):
    """Combined loss function for ordinal regression.
    
    Combines multiple loss components with learnable or fixed weights.
    """
    
    def __init__(self, 
                 n_cats: int,
                 ce_weight: float = 1.0,
                 qwk_weight: float = 0.5,
                 emd_weight: float = 0.0,
                 coral_weight: float = 0.0):
        """Initialize combined loss.
        
        Args:
            n_cats: Number of ordinal categories
            ce_weight: Weight for cross-entropy loss
            qwk_weight: Weight for QWK loss
            emd_weight: Weight for EMD loss
            coral_weight: Weight for CORAL-specific loss
        """
        super().__init__()
        self.n_cats = n_cats
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.qwk_loss = DifferentiableQWKLoss(n_cats) if qwk_weight > 0 else None
        self.emd_loss = OrdinalEMDLoss(n_cats) if emd_weight > 0 else None
        
        # Weights
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        self.emd_weight = emd_weight
        self.coral_weight = coral_weight
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                coral_info: Optional[dict] = None) -> dict:
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
        
        # QWK loss
        if self.qwk_weight > 0 and self.qwk_loss is not None:
            qwk_loss = self.qwk_loss(probs, targets, mask)
            losses['qwk_loss'] = qwk_loss
            total_loss += self.qwk_weight * qwk_loss
        
        # EMD loss
        if self.emd_weight > 0 and self.emd_loss is not None:
            emd_loss = self.emd_loss(probs, targets, mask)
            losses['emd_loss'] = emd_loss
            total_loss += self.emd_weight * emd_loss
        
        # CORAL-specific loss
        if self.coral_weight > 0 and coral_info is not None:
            # Use CORAL cumulative logits if available
            from core.coral_layer import CORALCompatibleLoss
            coral_loss_fn = CORALCompatibleLoss(self.n_cats)
            coral_loss = coral_loss_fn((probs, coral_info), targets, mask)
            losses['coral_loss'] = coral_loss
            total_loss += self.coral_weight * coral_loss
        
        losses['total_loss'] = total_loss
        return losses


def test_ordinal_losses():
    """Test ordinal loss functions."""
    print("Testing ordinal loss functions...")
    
    # Test data
    batch_size, seq_len, n_cats = 2, 5, 4
    pred_probs = F.softmax(torch.randn(batch_size, seq_len, n_cats), dim=-1)
    targets = torch.randint(0, n_cats, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    mask[0, 3:] = 0  # Mask out some positions
    
    # Test QWK loss
    qwk_loss = DifferentiableQWKLoss(n_cats)
    qwk_value = qwk_loss(pred_probs, targets, mask)
    print(f"QWK Loss: {qwk_value.item():.4f}")
    
    # Test EMD loss
    emd_loss = OrdinalEMDLoss(n_cats)
    emd_value = emd_loss(pred_probs, targets, mask)
    print(f"EMD Loss: {emd_value.item():.4f}")
    
    # Test ordinal CE loss
    ord_ce_loss = OrdinalCrossEntropyLoss(n_cats, alpha=0.5)
    ord_ce_value = ord_ce_loss(pred_probs, targets, mask)
    print(f"Ordinal CE Loss: {ord_ce_value.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedOrdinalLoss(
        n_cats, 
        ce_weight=1.0, 
        qwk_weight=0.5, 
        emd_weight=0.2
    )
    losses = combined_loss(pred_probs, targets, mask)
    print(f"\nCombined Loss Components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n✓ All ordinal losses tested successfully!")


if __name__ == "__main__":
    test_ordinal_losses()