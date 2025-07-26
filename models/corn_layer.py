"""
CORN (Conditional Ordinal Regression for Neural Networks) Layer

This implements the CORN framework, which converts ordinal regression into a
series of binary classification tasks. It's designed as a drop-in replacement
for the final prediction layer in DKVMN-based models.

Reference: "Deep Ordinal Regression: A Survey" and the original CORN paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CornLayer(nn.Module):
    """
    CORN (Conditional Ordinal Regression for Neural Networks) layer.

    Key properties:
    1. Transforms ordinal regression into K-1 binary classification tasks.
    2. Each binary classifier predicts P(Y > k | Y > k-1).
    3. Guarantees probability monotonicity through its architecture.

    Mathematical formulation:
        P(Y > k | x) = P(Y > 0 | x) * P(Y > 1 | Y > 0, x) * ... * P(Y > k | Y > k-1, x)
    """

    def __init__(self, input_dim, num_classes, use_bias=True):
        """
        Initialize CORN layer.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of ordinal classes (K)
            use_bias: Whether to use bias in the linear layers
        """
        super(CornLayer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # K-1 binary classifiers, one for each rank transition
        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=use_bias) for _ in range(num_classes - 1)
        ])

    def forward(self, features):
        """
        Forward pass through CORN layer.

        Args:
            features: Input features, shape (..., input_dim)

        Returns:
            class_probs: Class probabilities, shape (..., num_classes)
        """
        # Get logits from each binary classifier
        logits = [clf(features) for clf in self.classifiers]
        logits = torch.cat(logits, dim=-1)  # (..., K-1)

        # Apply sigmoid to get conditional probabilities P(Y > k | Y > k-1)
        cond_probs = torch.sigmoid(logits)

        # Calculate cumulative probabilities P(Y > k)
        cum_probs = torch.cumprod(cond_probs, dim=-1)

        # Convert cumulative probabilities to class probabilities
        class_probs = self._cum_to_class_probs(cum_probs)

        return class_probs, logits

    def _cum_to_class_probs(self, cum_probs):
        """
        Convert cumulative probabilities P(Y > k) to class probabilities P(Y = k).
        """
        batch_shape = cum_probs.shape[:-1]
        device = cum_probs.device

        # Initialize class probabilities
        class_probs = torch.zeros(batch_shape + (self.num_classes,), device=device)

        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[..., 0] = 1.0 - cum_probs[..., 0]

        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        for k in range(1, self.num_classes - 1):
            class_probs[..., k] = cum_probs[..., k - 1] - cum_probs[..., k]

        # P(Y = K-1) = P(Y > K-2)
        if self.num_classes > 1:
            class_probs[..., -1] = cum_probs[..., -2] if self.num_classes > 2 else cum_probs[..., -1]


        # Clamp and re-normalize to ensure valid probability distribution
        class_probs = torch.clamp(class_probs, min=1e-8)
        class_probs = class_probs / torch.sum(class_probs, dim=-1, keepdim=True)
        
        return class_probs


class CornLoss(nn.Module):
    """
    Loss function for CORN, based on binary cross-entropy for each classifier.
    """

    def __init__(self, num_classes):
        super(CornLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        Compute CORN loss.

        Args:
            logits: Raw logits from the CORN layer, shape (..., num_classes - 1)
            targets: True class labels, shape (...,)

        Returns:
            loss: CORN loss value
        """
        # Convert targets to a set of binary targets for each classifier
        # target_k = 1 if Y > k, else 0
        binary_targets = self._to_binary_targets(targets)

        # Binary cross-entropy loss for each classifier
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets.float(), reduction='mean')

        return loss

    def _to_binary_targets(self, targets):
        """
        Convert ordinal targets to binary targets for each of the K-1 classifiers.
        """
        batch_size = targets.shape[0]
        binary_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)

        for i in range(batch_size):
            for k in range(self.num_classes - 1):
                if targets[i] > k:
                    binary_targets[i, k] = 1.0
        
        return binary_targets
