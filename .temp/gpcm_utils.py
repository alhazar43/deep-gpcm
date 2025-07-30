"""
GPCM utilities including custom ordinal loss and evaluation metrics.
Implements ordinal loss from paper reference for ordered categorical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr


class OrdinalLoss(nn.Module):
    """
    Custom ordinal loss function for GPCM.
    Respects the ordering of categories.
    """
    
    def __init__(self, n_cats):
        super(OrdinalLoss, self).__init__()
        self.n_cats = n_cats
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted probabilities, shape (batch_size, n_cats)
            targets: Ground truth labels, shape (batch_size,)
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


class GpcmDataset(torch.utils.data.Dataset):
    """Dataset class for GPCM data compatible with DataLoader."""
    
    def __init__(self, questions, responses):
        # Ensure inputs are lists of sequences
        if isinstance(questions, torch.Tensor):
            # Convert flat tensor back to sequences - this is a simplified approach
            # In practice, you'd need proper sequence length information
            self.questions = questions.unsqueeze(0) if questions.dim() == 1 else questions
            self.responses = responses.unsqueeze(0) if responses.dim() == 1 else responses
        else:
            # Convert list of sequences to padded tensors
            max_len = max(len(seq) for seq in questions)
            self.questions = torch.zeros((len(questions), max_len), dtype=torch.long)
            self.responses = torch.zeros((len(responses), max_len), dtype=torch.long)
            
            for i, (q_seq, r_seq) in enumerate(zip(questions, responses)):
                seq_len = len(q_seq)
                self.questions[i, :seq_len] = torch.tensor(q_seq)
                self.responses[i, :seq_len] = torch.tensor(r_seq)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.responses[idx]
