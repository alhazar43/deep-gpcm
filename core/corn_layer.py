#!/usr/bin/env python3
"""
CORN (Conditional Ordinal Regression for Neural Networks) Layer
Superior alternative to CORAL with better categorical accuracy balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CORNLayer(nn.Module):
    """CORN layer for ordinal knowledge tracing.
    
    CORN addresses CORAL's main limitations:
    - No weight-sharing constraints → Better expressivity
    - Superior class imbalance handling
    - Maintains rank consistency
    - Better categorical accuracy preservation
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3, 
                 architecture: str = 'standard', hidden_multiplier: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.architecture = architecture
        
        # Adaptive hidden dimension based on complexity
        hidden_dim = max(16, int(input_dim * hidden_multiplier))
        
        if architecture == 'deep':
            # Deeper networks for better expressivity
            self.binary_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout * 0.5),  # Less aggressive dropout
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in range(num_classes - 1)
            ])
        elif architecture == 'residual':
            # Residual connections for better gradient flow
            self.binary_classifiers = nn.ModuleList([
                self._create_residual_classifier(input_dim, hidden_dim, dropout)
                for _ in range(num_classes - 1)
            ])
        else:  # standard
            # Standard architecture but optimized
            self.binary_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout),  
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),  # Reduced dropout after activation
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(num_classes - 1)
            ])
        
        # Initialize with proper ordering bias
        self._initialize_thresholds()
    
    def _create_residual_classifier(self, input_dim: int, hidden_dim: int, dropout: float):
        """Create a residual classifier with skip connection."""
        class ResidualClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
                self.projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
                self.dropout = nn.Dropout(dropout * 0.5)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                identity = self.projection(x)
                out = self.activation(self.fc1(x))
                out = self.dropout(out)
                out = self.fc2(out)
                out += identity  # Residual connection
                out = self.activation(out)
                out = self.dropout(out)
                return self.fc3(out)
        
        return ResidualClassifier()
    
    def _initialize_thresholds(self):
        """Initialize thresholds to maintain ordinal structure with better spacing."""
        with torch.no_grad():
            for i, classifier in enumerate(self.binary_classifiers):
                # Use logistic function for more natural threshold spacing
                # This creates better separation between adjacent classes
                progress = (i + 1) / (self.num_classes)  # 0.25, 0.5, 0.75 for 4 classes
                
                # Map to logistic scale with better spacing
                # Higher thresholds should be progressively harder to pass
                bias_value = -2.5 + (progress * 5.0)  # Range: -2.5 to 2.5
                
                # Apply sigmoid-like spacing for more natural progression
                import math
                bias_value = math.log(progress / (1 - progress + 1e-8))
                
                classifier[-1].bias.data.fill_(bias_value)
                
                # Also improve weight initialization for better convergence
                nn.init.xavier_uniform_(classifier[-1].weight)
                for layer in classifier[:-1]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CORN layer.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Class probabilities [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Get binary predictions for each threshold
        binary_logits = []
        for classifier in self.binary_classifiers:
            logit = classifier(x)
            binary_logits.append(logit.squeeze(-1))
        
        binary_logits = torch.stack(binary_logits, dim=-1)  # [batch, num_classes-1]
        binary_probs = torch.sigmoid(binary_logits)
        
        # Convert to ordinal class probabilities
        class_probs = self._binary_to_ordinal_probs(binary_probs)
        
        return class_probs
    
    def _binary_to_ordinal_probs(self, binary_probs: torch.Tensor) -> torch.Tensor:
        """Convert binary threshold predictions to ordinal class probabilities.
        
        For K classes, we have K-1 binary classifiers predicting P(Y > k).
        Class probabilities are computed as:
        P(Y = 0) = 1 - P(Y > 0)
        P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        P(Y = K-1) = P(Y > K-2)
        """
        batch_size = binary_probs.size(0)
        num_thresholds = binary_probs.size(1)
        num_classes = num_thresholds + 1
        
        # Ensure monotonicity: P(Y > k) should decrease with k
        # This is crucial for ordinal consistency
        # Use cumulative minimum to enforce monotonicity without in-place operations
        binary_probs_list = [binary_probs[:, 0:1]]  # First threshold unchanged
        for i in range(1, num_thresholds):
            current_prob = binary_probs[:, i:i+1]
            prev_min = binary_probs_list[-1]
            monotonic_prob = torch.minimum(current_prob, prev_min)
            binary_probs_list.append(monotonic_prob)
        
        binary_probs_monotonic = torch.cat(binary_probs_list, dim=1)
        
        # Convert to class probabilities
        class_probs = torch.zeros(batch_size, num_classes, device=binary_probs.device)
        
        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[:, 0] = 1.0 - binary_probs_monotonic[:, 0]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) for middle classes
        for k in range(1, num_classes - 1):
            class_probs[:, k] = binary_probs_monotonic[:, k-1] - binary_probs_monotonic[:, k]
        
        # P(Y = K-1) = P(Y > K-2)
        class_probs[:, -1] = binary_probs_monotonic[:, -1]
        
        # Ensure probabilities sum to 1 and are non-negative
        class_probs = torch.clamp(class_probs, min=1e-8)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)
        
        return class_probs


class CORNLoss(nn.Module):
    """Optimized loss function for CORN predictions."""
    
    def __init__(self, num_classes: int, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute CORN loss.
        
        Args:
            logits: Raw binary logits from CORN classifiers [batch, seq, num_classes-1]
            targets: True class labels [batch, seq]  
            mask: Optional mask for sequence padding [batch, seq]
        """
        if logits.dim() == 3:  # Sequence data
            batch_size, seq_len, _ = logits.shape
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            if mask is not None:
                mask = mask.view(-1)
        
        # Create binary targets for each threshold
        binary_targets = self._create_binary_targets(targets)
        
        # Compute binary cross-entropy for each classifier
        loss = 0.0
        for i in range(self.num_classes - 1):
            binary_loss = F.binary_cross_entropy_with_logits(
                logits[:, i], binary_targets[:, i].float(), reduction='none'
            )
            loss = loss + binary_loss
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == 'mean':
                loss = loss.sum() / mask.float().sum().clamp(min=1)
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss
    
    def _create_binary_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Create binary targets for threshold classifiers.
        
        For class k, binary_target[i] = 1 if targets > i, else 0
        """
        batch_size = targets.size(0)
        binary_targets = torch.zeros(batch_size, self.num_classes - 1, 
                                   device=targets.device, dtype=targets.dtype)
        
        for i in range(self.num_classes - 1):
            binary_targets[:, i] = (targets > i).long()
        
        return binary_targets


class ProgressiveWeightScheduler:
    """Progressive dynamic weighting for balanced categorical-ordinal training."""
    
    def __init__(self, total_epochs: int, 
                 early_categorical: float = 0.8,
                 late_ordinal: float = 0.6,
                 transition_point: float = 0.3):
        """
        Args:
            total_epochs: Total training epochs
            early_categorical: Initial categorical weight (0.7-0.9 recommended)
            late_ordinal: Final ordinal weight (0.5-0.7 recommended)  
            transition_point: Fraction of epochs for transition (0.2-0.4)
        """
        self.total_epochs = total_epochs
        self.early_categorical = early_categorical
        self.late_ordinal = late_ordinal
        self.transition_point = transition_point
        self.transition_epoch = int(total_epochs * transition_point)
    
    def get_weights(self, epoch: int) -> dict:
        """Get categorical and ordinal weights for current epoch."""
        progress = epoch / self.total_epochs
        
        if progress < self.transition_point:
            # Early phase: Focus on categorical accuracy
            categorical_weight = self.early_categorical
            ordinal_weight = 1.0 - categorical_weight
        elif progress < 0.7:
            # Middle phase: Gradual transition
            transition_progress = (progress - self.transition_point) / (0.7 - self.transition_point)
            categorical_weight = self.early_categorical - transition_progress * 0.2
            ordinal_weight = 1.0 - categorical_weight
        else:
            # Late phase: Emphasize ordinal consistency
            categorical_weight = 1.0 - self.late_ordinal
            ordinal_weight = self.late_ordinal
        
        return {
            'categorical': categorical_weight,
            'ordinal': ordinal_weight,
            'epoch': epoch,
            'progress': progress
        }


class EarthMoversDistance(nn.Module):
    """Earth Mover's Distance loss for ordinal regression."""
    
    def __init__(self, num_classes: int, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute EMD loss between predicted and target distributions."""
        # Convert logits to probabilities
        if logits.dim() == 3:  # Sequence data
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1) if mask is not None else None
        else:
            logits_flat = logits
            targets_flat = targets
            mask_flat = mask
        
        pred_probs = F.softmax(logits_flat, dim=-1)
        target_probs = F.one_hot(targets_flat, self.num_classes).float()
        
        # Compute cumulative distributions
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)
        
        # EMD is the L1 distance between CDFs
        emd = torch.sum(torch.abs(pred_cdf - target_cdf), dim=-1)
        
        # Apply mask if provided
        if mask_flat is not None:
            emd = emd * mask_flat.float()
            if self.reduction == 'mean':
                emd = emd.sum() / mask_flat.float().sum().clamp(min=1)
            elif self.reduction == 'sum':
                emd = emd.sum()
        else:
            if self.reduction == 'mean':
                emd = emd.mean()
            elif self.reduction == 'sum':
                emd = emd.sum()
        
        return emd


class HybridOrdinalLoss(nn.Module):
    """Multi-objective combined loss balancing categorical accuracy and ordinal consistency."""
    
    def __init__(self, num_classes: int, 
                 ce_weight: float = 0.5,
                 emd_weight: float = 0.3,
                 corn_weight: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.emd_weight = emd_weight  
        self.corn_weight = corn_weight
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.emd_loss = EarthMoversDistance(num_classes, reduction='none')
        self.corn_loss = CORNLoss(num_classes, reduction='none')
    
    def forward(self, corn_logits: torch.Tensor, class_probs: torch.Tensor,
                targets: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Args:
            corn_logits: Raw binary logits from CORN [batch, seq, num_classes-1]
            class_probs: Class probabilities from CORN [batch, seq, num_classes]
            targets: True labels [batch, seq]
            mask: Optional sequence mask [batch, seq]
        """
        if corn_logits.dim() == 3:
            batch_size, seq_len, _ = corn_logits.shape
            corn_logits_flat = corn_logits.view(-1, corn_logits.size(-1))
            class_probs_flat = class_probs.view(-1, class_probs.size(-1))
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1) if mask is not None else None
        else:
            corn_logits_flat = corn_logits
            class_probs_flat = class_probs
            targets_flat = targets
            mask_flat = mask
        
        # Compute individual losses
        ce_loss = self.ce_loss(torch.log(class_probs_flat + 1e-8), targets_flat)
        emd_loss = self.emd_loss(torch.log(class_probs_flat + 1e-8), targets_flat, mask_flat)
        corn_loss = self.corn_loss(corn_logits_flat, targets_flat, mask_flat)
        
        # Apply mask and reduce
        if mask_flat is not None:
            ce_loss = (ce_loss * mask_flat.float()).sum() / mask_flat.float().sum().clamp(min=1)
        else:
            ce_loss = ce_loss.mean()
        
        # Combined loss
        total_loss = (self.ce_weight * ce_loss + 
                     self.emd_weight * emd_loss + 
                     self.corn_weight * corn_loss)
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'emd_loss': emd_loss,
            'corn_loss': corn_loss
        }