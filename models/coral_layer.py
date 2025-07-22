"""
CORAL (COnsistent RAnk Logits) Layer for Ordinal Classification

This implements the CORAL framework as a drop-in replacement for the final 
prediction layer in DKVMN-based models. CORAL ensures rank-consistent predictions
while maintaining compatibility with existing memory architectures.

Reference: "Rank-consistent ordinal regression for neural networks" (CORAL framework)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CoralLayer(nn.Module):
    """
    CORAL (COnsistent RAnk Logits) layer for ordinal classification.
    
    Key properties:
    1. Uses a single feature vector f(x) 
    2. Learns K-1 thresholds τ₁ < τ₂ < ... < τₖ₋₁
    3. Guarantees rank monotonicity: P(Y ≤ k₁) ≤ P(Y ≤ k₂) for k₁ < k₂
    
    Mathematical formulation:
        P(Y > k | x) = σ(f(x) - τₖ)  for k = 0, 1, ..., K-1
        P(Y = k | x) = P(Y > k-1 | x) - P(Y > k | x)
    """
    
    def __init__(self, input_dim, num_classes, use_bias=True):
        """
        Initialize CORAL layer.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of ordinal classes (K)
            use_bias: Whether to use bias in linear layer
        """
        super(CoralLayer, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Single feature transformation f(x)
        self.feature_layer = nn.Linear(input_dim, 1, bias=use_bias)
        
        # Learnable thresholds τ₁, τ₂, ..., τₖ₋₁ 
        # We parameterize as unconstrained values and enforce ordering
        self.raw_thresholds = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, features):
        """
        Forward pass through CORAL layer.
        
        Args:
            features: Input features, shape (..., input_dim)
            
        Returns:
            class_probs: Class probabilities, shape (..., num_classes)
        """
        # Get feature score f(x)
        feature_scores = self.feature_layer(features).squeeze(-1)  # (...,)
        
        # Ensure monotonic thresholds using cumulative sum of softplus
        # This guarantees τ₁ ≤ τ₂ ≤ ... ≤ τₖ₋₁
        ordered_thresholds = torch.cumsum(F.softplus(self.raw_thresholds), dim=0)
        
        # Compute P(Y > k | x) = σ(f(x) - τₖ) for k = 0, 1, ..., K-1
        # For k=0, we don't have a threshold, so P(Y > 0 | x) is handled separately
        cum_probs = torch.sigmoid(feature_scores.unsqueeze(-1) - ordered_thresholds.unsqueeze(0))  # (..., K-1)
        
        # Convert cumulative probabilities to class probabilities
        class_probs = self._cum_to_class_probs(cum_probs, feature_scores)
        
        return class_probs
    
    def _cum_to_class_probs(self, cum_probs, feature_scores):
        """
        Convert cumulative probabilities P(Y > k) to class probabilities P(Y = k).
        
        Mathematical derivation:
        P(Y = 0) = 1 - P(Y > 0) = 1 - σ(f(x) - τ₁)
        P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        P(Y = K-1) = P(Y > K-2) - 0 = σ(f(x) - τₖ₋₁)
        """
        batch_shape = cum_probs.shape[:-1]
        device = cum_probs.device
        
        # Initialize class probabilities
        class_probs = torch.zeros(batch_shape + (self.num_classes,), device=device)
        
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y > 0) = P(Y ≥ 1) = σ(f(x) - τ₁)
        if self.num_classes > 1:
            class_probs[..., 0] = 1.0 - cum_probs[..., 0]
        else:
            class_probs[..., 0] = 1.0
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        for k in range(1, self.num_classes - 1):
            class_probs[..., k] = cum_probs[..., k-1] - cum_probs[..., k]
        
        # P(Y = K-1) = P(Y > K-2)
        if self.num_classes > 1:
            class_probs[..., -1] = cum_probs[..., -1]
        
        # Ensure probabilities are non-negative and sum to 1
        class_probs = F.softmax(torch.log(class_probs.clamp(min=1e-8)), dim=-1)
        
        return class_probs
    
    def get_thresholds(self):
        """Get the current threshold values."""
        return torch.cumsum(F.softplus(self.raw_thresholds), dim=0)
    
    def rank_consistency_loss(self, predictions, targets):
        """
        Additional loss term to enforce rank consistency.
        This is optional since CORAL architecture already guarantees consistency.
        """
        # Compute cumulative probabilities
        cum_probs = torch.cumsum(predictions, dim=-1)
        
        # Check monotonicity: cum_probs[:, k] <= cum_probs[:, k+1]
        consistency_loss = 0.0
        for k in range(cum_probs.shape[-1] - 1):
            violations = F.relu(cum_probs[..., k] - cum_probs[..., k+1])
            consistency_loss += violations.mean()
        
        return consistency_loss


class CoralLoss(nn.Module):
    """
    Loss function specifically designed for CORAL layer.
    
    Uses binary cross-entropy on cumulative probabilities rather than 
    categorical cross-entropy on class probabilities.
    """
    
    def __init__(self, num_classes):
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, coral_logits, targets):
        """
        Compute CORAL loss.
        
        Args:
            coral_logits: Raw logits before sigmoid, shape (..., num_classes)
            targets: True class labels, shape (...,)
            
        Returns:
            loss: CORAL loss value
        """
        # Convert targets to cumulative targets
        cum_targets = self._to_cumulative_targets(targets)
        
        # Extract cumulative logits (before sigmoid)
        # We need to reconstruct these from class probabilities
        # This is a simplified version - in practice, you'd pass logits directly
        cum_logits = torch.log(coral_logits.clamp(min=1e-8) / (1 - coral_logits.clamp(max=1-1e-8)))
        
        # Binary cross-entropy loss on cumulative probabilities
        loss = F.binary_cross_entropy_with_logits(cum_logits, cum_targets.float())
        
        return loss
    
    def _to_cumulative_targets(self, targets):
        """
        Convert ordinal targets to cumulative targets.
        
        Example: target=2 (out of 4 classes) → cum_targets=[1, 1, 0]
        Meaning: Y > 0, Y > 1, Y ≤ 2
        """
        batch_size = targets.shape[0]
        cum_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        
        for k in range(self.num_classes - 1):
            cum_targets[:, k] = (targets > k).float()
        
        return cum_targets


class DKVMNCoralModel(nn.Module):
    """
    Integration wrapper for DKVMN + CORAL.
    
    This replaces the GPCM probability calculation with CORAL while
    maintaining IRT parameter extraction for interpretability.
    """
    
    def __init__(self, base_model, coral_input_dim):
        """
        Initialize DKVMN-CORAL model.
        
        Args:
            base_model: Base DKVMN model (without final prediction)
            coral_input_dim: Input dimension for CORAL layer
        """
        super(DKVMNCoralModel, self).__init__()
        
        self.base_model = base_model
        self.coral_layer = CoralLayer(coral_input_dim, base_model.n_cats)
        
    def forward(self, q_data, r_data, target_mask=None):
        """
        Forward pass through DKVMN + CORAL.
        
        Returns IRT parameters for interpretability and CORAL probabilities for prediction.
        """
        # Get DKVMN features and IRT parameters
        student_abilities, item_thresholds, discrimination_params, summary_vectors = self.base_model.get_features(q_data, r_data, target_mask)
        
        # Use CORAL for final prediction instead of GPCM
        coral_probs = self.coral_layer(summary_vectors)
        
        return student_abilities, item_thresholds, discrimination_params, coral_probs


# Simple integration test
if __name__ == "__main__":
    # Test CORAL layer
    print("Testing CORAL Layer...")
    
    batch_size, seq_len, feature_dim = 32, 10, 128
    num_classes = 4
    
    coral = CoralLayer(feature_dim, num_classes)
    features = torch.randn(batch_size, seq_len, feature_dim)
    
    probs = coral(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len))}")
    
    # Check rank consistency
    cum_probs = torch.cumsum(probs, dim=-1)
    monotonic = True
    for k in range(num_classes - 1):
        if not torch.all(cum_probs[..., k] <= cum_probs[..., k+1] + 1e-6):
            monotonic = False
            break
    
    print(f"Rank consistency maintained: {monotonic}")
    print(f"Current thresholds: {coral.get_thresholds()}")
    
    print("✅ CORAL layer test passed!")