"""
GPCM utilities including custom ordinal loss and evaluation metrics.
Implements ordinal loss from paper reference for ordered categorical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score


class OrdinalLoss(nn.Module):
    """
    Custom ordinal loss for GPCM following paper formulation:
    L = -Σ Σ Σ [I(y≤k)log(P(Y≤k)) + I(y>k)log(1-P(Y≤k))]
    
    Respects the ordering of categories unlike CrossEntropyLoss.
    """
    
    def __init__(self, n_cats, reduction='mean'):
        super(OrdinalLoss, self).__init__()
        self.n_cats = n_cats
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Compute ordinal loss.
        
        Args:
            predictions: GPCM probabilities, shape (batch_size, seq_len, K)
            targets: True categories, shape (batch_size, seq_len) with values 0 to K-1
            
        Returns:
            loss: Ordinal loss value
        """
        batch_size, seq_len, K = predictions.shape
        device = predictions.device
        
        # Compute cumulative probabilities P(Y ≤ k)
        cum_probs = torch.cumsum(predictions, dim=-1)  # (batch_size, seq_len, K)
        
        loss = 0.0
        
        # For each threshold k = 0, ..., K-2
        for k in range(K - 1):
            # Indicator I(y ≤ k): 1 if target ≤ k, 0 otherwise
            indicator_leq = (targets <= k).float()  # (batch_size, seq_len)
            
            # Indicator I(y > k): 1 if target > k, 0 otherwise  
            indicator_gt = (targets > k).float()  # (batch_size, seq_len)
            
            # Get cumulative probability P(Y ≤ k)
            cum_prob_k = cum_probs[:, :, k]  # (batch_size, seq_len)
            
            # Avoid log(0) by clamping
            cum_prob_k = torch.clamp(cum_prob_k, min=1e-8, max=1-1e-8)
            
            # Compute loss components
            loss_leq = indicator_leq * torch.log(cum_prob_k)
            loss_gt = indicator_gt * torch.log(1 - cum_prob_k)
            
            loss = loss - (loss_leq + loss_gt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GpcmMetrics:
    """
    Evaluation metrics for GPCM including categorical and ordinal accuracy.
    """
    
    @staticmethod
    def categorical_accuracy(predictions, targets):
        """Exact category match accuracy."""
        pred_cats = torch.argmax(predictions, dim=-1)
        correct = (pred_cats == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def ordinal_accuracy(predictions, targets, tolerance=1):
        """Accuracy within tolerance categories."""
        pred_cats = torch.argmax(predictions, dim=-1)
        diff = torch.abs(pred_cats - targets)
        within_tolerance = (diff <= tolerance).float()
        return within_tolerance.mean().item()
    
    @staticmethod
    def mean_absolute_error(predictions, targets):
        """MAE treating categories as ordinal values."""
        pred_cats = torch.argmax(predictions, dim=-1)
        mae = torch.abs(pred_cats.float() - targets.float()).mean()
        return mae.item()
    
    @staticmethod
    def quadratic_weighted_kappa(predictions, targets, n_cats):
        """
        Quadratic Weighted Kappa for ordinal data.
        Uses sklearn implementation on flattened arrays.
        """
        pred_cats = torch.argmax(predictions, dim=-1)
        
        # Flatten and convert to numpy
        y_true = targets.cpu().numpy().flatten()
        y_pred = pred_cats.cpu().numpy().flatten()
        
        # Compute QWK
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        return qwk
    
    @staticmethod
    def per_category_accuracy(predictions, targets, n_cats):
        """Per-category accuracy scores."""
        pred_cats = torch.argmax(predictions, dim=-1)
        
        accuracies = {}
        for k in range(n_cats):
            mask = (targets == k)
            if mask.sum() > 0:
                correct = (pred_cats[mask] == k).float().mean()
                accuracies[f'cat_{k}_acc'] = correct.item()
            else:
                accuracies[f'cat_{k}_acc'] = 0.0
        
        return accuracies


def detect_n_categories(data_path):
    """
    Auto-detect number of categories from dataset.
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        n_cats: Number of categories detected
    """
    responses = []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
        # Read response sequences (every 3rd line starting from line 2)
        for i in range(2, len(lines), 3):
            response_line = lines[i].strip()
            if response_line:
                # Handle both integer and float responses
                try:
                    # Try parsing as integers first (OC format)
                    responses.extend([int(float(r)) for r in response_line.split(',')])
                except ValueError:
                    # Try parsing as floats (PC format)
                    float_responses = [float(r) for r in response_line.split(',')]
                    # Convert to categories by rounding
                    responses.extend([int(round(r * 3)) for r in float_responses])
    
    if not responses:
        return 4  # Default fallback
    
    n_cats = max(responses) + 1  # Categories are 0-indexed
    return n_cats


def load_gpcm_data(data_path, n_cats=None):
    """
    Load GPCM dataset and auto-detect categories if needed.
    
    Args:
        data_path: Path to dataset file
        n_cats: Number of categories (auto-detect if None)
        
    Returns:
        tuple: (sequences, questions, responses, n_cats)
    """
    if n_cats is None:
        n_cats = detect_n_categories(data_path)
    
    sequences = []
    questions = []
    responses = []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            
            # Parse sequence length, questions, and responses
            seq_len = int(lines[i].strip())
            q_seq = [int(q) for q in lines[i + 1].strip().split(',')]
            
            # Handle both OC and PC formats
            response_line = lines[i + 2].strip()
            try:
                # OC format (integers)
                r_seq = [int(float(r)) for r in response_line.split(',')]
            except ValueError:
                # PC format (floats) - convert to categories
                float_responses = [float(r) for r in response_line.split(',')]
                r_seq = [int(round(r * (n_cats - 1))) for r in float_responses]
            
            sequences.append(seq_len)
            questions.append(q_seq)
            responses.append(r_seq)
    
    return sequences, questions, responses, n_cats


def create_gpcm_batch(questions, responses, max_seq_len=None, padding_value=0):
    """
    Create batched tensors from GPCM data.
    
    Args:
        questions: List of question sequences
        responses: List of response sequences  
        max_seq_len: Maximum sequence length (auto-detect if None)
        padding_value: Value for padding
        
    Returns:
        tuple: (q_batch, r_batch, mask_batch)
    """
    batch_size = len(questions)
    
    if max_seq_len is None:
        max_seq_len = max(len(seq) for seq in questions)
    
    # Initialize tensors
    q_batch = torch.full((batch_size, max_seq_len), padding_value, dtype=torch.long)
    r_batch = torch.full((batch_size, max_seq_len), padding_value, dtype=torch.long)
    mask_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    # Fill tensors
    for i, (q_seq, r_seq) in enumerate(zip(questions, responses)):
        seq_len = min(len(q_seq), max_seq_len)
        q_batch[i, :seq_len] = torch.tensor(q_seq[:seq_len])
        r_batch[i, :seq_len] = torch.tensor(r_seq[:seq_len])
        mask_batch[i, :seq_len] = True
    
    return q_batch, r_batch, mask_batch


# Alternative loss functions for comparison

class CrossEntropyLossWrapper(nn.Module):
    """Wrapper for standard CrossEntropyLoss for comparison."""
    
    def __init__(self, reduction='mean'):
        super(CrossEntropyLossWrapper, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, predictions, targets):
        # Reshape for CrossEntropyLoss: (N, C, ...) format
        batch_size, seq_len, n_cats = predictions.shape
        predictions_flat = predictions.view(-1, n_cats)  # (N*seq_len, n_cats)
        targets_flat = targets.view(-1)  # (N*seq_len,)
        
        return self.loss_fn(predictions_flat, targets_flat)


class MSELossWrapper(nn.Module):
    """MSE loss treating ordinal categories as continuous values."""
    
    def __init__(self, reduction='mean'):
        super(MSELossWrapper, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions, targets):
        # Convert predictions to expected category values
        pred_cats = torch.argmax(predictions, dim=-1).float()
        targets_float = targets.float()
        
        return self.loss_fn(pred_cats, targets_float)