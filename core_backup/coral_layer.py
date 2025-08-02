"""CORAL (COnsistent RAnk Logits) layer for ordinal regression in knowledge tracing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class CORALLayer(nn.Module):
    """CORAL layer for ordinal knowledge tracing.
    
    This layer enforces rank consistency in ordinal predictions by using
    a shared representation with K-1 binary classifiers for K categories.
    
    Reference:
        Cao et al. "Rank consistent ordinal regression for neural networks 
        with application to age estimation." Pattern Recognition Letters (2020).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 n_cats: int, 
                 use_bias: bool = True, 
                 dropout_rate: float = 0.0,
                 shared_hidden_dim: Optional[int] = None):
        """Initialize CORAL layer.
        
        Args:
            input_dim: Input feature dimension
            n_cats: Number of ordinal categories
            use_bias: Whether to use bias in linear layers
            dropout_rate: Dropout rate for regularization
            shared_hidden_dim: Hidden dimension for shared layer (default: input_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_cats = n_cats
        self.use_bias = use_bias
        
        hidden_dim = shared_hidden_dim or input_dim
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # K-1 binary classifiers with shared base weights
        # Each classifier predicts P(Y > k) for k = 0, ..., K-2
        self.rank_classifier = nn.Linear(hidden_dim, n_cats - 1, bias=use_bias)
        
        # Optional: learnable ordinal thresholds for better calibration
        self.use_thresholds = True
        if self.use_thresholds:
            self.ordinal_thresholds = nn.Parameter(torch.zeros(n_cats - 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability and proper ordering."""
        # Shared layer initialization
        for module in self.shared_layer:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Rank classifier initialization
        nn.init.kaiming_normal_(self.rank_classifier.weight)
        if self.use_bias:
            # Initialize biases to enforce natural ordering
            # Decreasing biases encourage P(Y > k) to decrease with k
            bias_init = torch.linspace(1, -1, self.n_cats - 1)
            self.rank_classifier.bias.data = bias_init
        
        # Ordinal thresholds - initialize with equal spacing
        if self.use_thresholds:
            threshold_init = torch.linspace(-0.5, 0.5, self.n_cats - 1)
            self.ordinal_thresholds.data = threshold_init
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through CORAL layer.
        
        Args:
            features: Input features, shape (batch_size, seq_len, input_dim)
            
        Returns:
            tuple:
                - probs: Ordinal probabilities, shape (batch_size, seq_len, n_cats)
                - info: Dictionary with additional outputs (logits, cumulative_probs)
        """
        # Shared representation
        shared_features = self.shared_layer(features)
        
        # Binary classifiers for P(Y > k) for k = 0, ..., K-2
        logits = self.rank_classifier(shared_features)
        
        # Add ordinal thresholds if enabled
        if self.use_thresholds:
            logits = logits + self.ordinal_thresholds.unsqueeze(0).unsqueeze(0)
        
        # Convert to cumulative probabilities using sigmoid
        # cum_probs[..., k] represents P(Y > k)
        cum_probs = torch.sigmoid(logits)
        
        # Ensure rank consistency by clipping
        # P(Y > k) should be >= P(Y > k+1)
        cum_probs = self._ensure_rank_consistency(cum_probs)
        
        # Convert cumulative probabilities to category probabilities
        probs = self._cumulative_to_categorical(cum_probs)
        
        # Additional info for loss computation
        info = {
            'logits': logits,
            'cumulative_probs': cum_probs,
            'thresholds': self.ordinal_thresholds if self.use_thresholds else None
        }
        
        return probs, info
    
    def _ensure_rank_consistency(self, cum_probs: torch.Tensor) -> torch.Tensor:
        """Ensure cumulative probabilities are monotonically decreasing.
        
        Args:
            cum_probs: Cumulative probabilities P(Y > k), shape (..., K-1)
            
        Returns:
            Rank-consistent cumulative probabilities
        """
        # Method 1: Simple clipping (used during inference)
        if not self.training:
            # Ensure P(Y > k) >= P(Y > k+1)
            for k in range(cum_probs.size(-1) - 1):
                cum_probs[..., k + 1] = torch.min(
                    cum_probs[..., k + 1], 
                    cum_probs[..., k]
                )
        
        # During training, let gradients flow naturally
        return cum_probs
    
    def _cumulative_to_categorical(self, cum_probs: torch.Tensor) -> torch.Tensor:
        """Convert cumulative probabilities to categorical probabilities.
        
        Args:
            cum_probs: Cumulative probabilities P(Y > k), shape (..., K-1)
            
        Returns:
            Categorical probabilities P(Y = k), shape (..., K)
        """
        # P(Y = 0) = 1 - P(Y > 0)
        p0 = 1 - cum_probs[..., 0:1]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        pk = cum_probs[..., :-1] - cum_probs[..., 1:]
        
        # P(Y = K-1) = P(Y > K-2)
        pK = cum_probs[..., -1:]
        
        # Concatenate all probabilities
        probs = torch.cat([p0, pk, pK], dim=-1)
        
        # Ensure probabilities sum to 1 and are non-negative
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs


class CORALCompatibleLoss(nn.Module):
    """Loss function optimized for CORAL predictions."""
    
    def __init__(self, n_cats: int, reduction: str = 'mean'):
        """Initialize CORAL-compatible loss.
        
        Args:
            n_cats: Number of ordinal categories
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.n_cats = n_cats
        self.reduction = reduction
    
    def forward(self, 
                coral_output: Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute CORAL loss using cumulative logits.
        
        Args:
            coral_output: Tuple of (probs, info) from CORAL layer
            targets: True categories, shape (batch_size, seq_len)
            mask: Optional mask for valid positions
            
        Returns:
            Loss value
        """
        probs, info = coral_output
        logits = info['logits']
        
        # Convert targets to cumulative format
        cum_targets = self._to_cumulative_labels(targets)
        
        # Flatten for loss computation
        batch_size, seq_len = targets.shape
        logits_flat = logits.view(-1, self.n_cats - 1)
        cum_targets_flat = cum_targets.view(-1, self.n_cats - 1)
        
        # Binary cross-entropy on each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits_flat, 
            cum_targets_flat, 
            reduction='none'
        )
        
        # Average across thresholds
        loss = loss.mean(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            
            if self.reduction == 'mean':
                return loss.sum() / (mask_flat.sum() + 1e-7)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss
    
    def _to_cumulative_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert categorical targets to cumulative format.
        
        Args:
            targets: Categorical targets, shape (...,)
            
        Returns:
            Cumulative targets, shape (..., K-1)
        """
        # Create cumulative labels: cum_targets[..., k] = 1 if Y > k
        cum_targets = torch.zeros(
            *targets.shape, self.n_cats - 1, 
            device=targets.device, dtype=torch.float32
        )
        
        for k in range(self.n_cats - 1):
            cum_targets[..., k] = (targets > k).float()
        
        return cum_targets


def test_coral_layer():
    """Simple test for CORAL layer functionality."""
    # Test parameters
    batch_size, seq_len = 2, 5
    input_dim = 32
    n_cats = 4
    
    # Create layer
    coral = CORALLayer(input_dim, n_cats)
    
    # Test forward pass
    features = torch.randn(batch_size, seq_len, input_dim)
    probs, info = coral(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Logits shape: {info['logits'].shape}")
    
    # Check probability constraints
    print(f"\nProbabilities sum to 1: {torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))}")
    print(f"Probabilities non-negative: {(probs >= 0).all()}")
    
    # Check rank consistency
    cum_probs = info['cumulative_probs']
    is_consistent = True
    for k in range(n_cats - 2):
        if not (cum_probs[..., k] >= cum_probs[..., k + 1]).all():
            is_consistent = False
            break
    print(f"Rank consistency maintained: {is_consistent}")
    
    # Test loss computation
    loss_fn = CORALCompatibleLoss(n_cats)
    targets = torch.randint(0, n_cats, (batch_size, seq_len))
    loss = loss_fn((probs, info), targets)
    print(f"\nLoss value: {loss.item():.4f}")
    
    return coral, probs, info


if __name__ == "__main__":
    # Run simple test
    test_coral_layer()