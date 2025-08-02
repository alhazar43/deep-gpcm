#!/usr/bin/env python3
"""
Unified prediction system for Deep-GPCM models.
Provides hard, soft, and threshold-based predictions with proper handling of edge cases.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass

from .monitoring import monitor_prediction, get_monitor

@dataclass
class PredictionConfig:
    """Configuration for unified prediction system."""
    # Threshold settings
    thresholds: List[float] = None  # Default: evenly spaced
    adaptive_thresholds: bool = False
    threshold_percentiles: List[float] = None  # e.g., [25, 50, 75]
    
    # Numerical stability
    epsilon: float = 1e-7
    
    # Performance settings
    use_gpu: bool = True
    batch_size: Optional[int] = None  # For large-scale processing
    
    # Caching
    cache_cumulative: bool = True
    
    def __post_init__(self):
        """Initialize default thresholds if not provided."""
        if self.thresholds is None and self.threshold_percentiles is None:
            # Default to evenly spaced percentiles
            self.threshold_percentiles = [25.0, 50.0, 75.0]


def categorical_to_cumulative(probs: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Convert categorical probabilities to cumulative probabilities.
    
    For K categories, returns K-1 cumulative probabilities:
    P(Y > k) for k = 0, 1, ..., K-2
    
    Args:
        probs: Categorical probabilities, shape (..., K)
        epsilon: Small value for numerical stability
        
    Returns:
        Cumulative probabilities, shape (..., K-1)
    """
    # Ensure probabilities are valid
    probs = torch.clamp(probs, min=epsilon, max=1.0 - epsilon)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Calculate cumulative probabilities P(Y > k)
    # P(Y > 0) = 1 - P(Y = 0) = P(Y = 1) + P(Y = 2) + ... + P(Y = K-1)
    # P(Y > 1) = P(Y = 2) + P(Y = 3) + ... + P(Y = K-1)
    # ...
    # P(Y > K-2) = P(Y = K-1)
    
    # Method: sum probabilities for categories > k
    # For each k, sum probabilities from k+1 to K-1
    cum_probs = []
    for k in range(probs.shape[-1] - 1):
        # P(Y > k) = sum of probabilities for categories k+1, k+2, ..., K-1
        cum_prob_k = probs[..., k+1:].sum(dim=-1)
        cum_probs.append(cum_prob_k)
    
    cum_probs = torch.stack(cum_probs, dim=-1)
    
    return cum_probs


@monitor_prediction('hard')
def compute_hard_predictions(probabilities: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute hard predictions using argmax.
    
    Args:
        probabilities: Probability distributions, shape (..., K)
        mask: Optional boolean mask for valid positions
        
    Returns:
        Hard predictions (category indices), same shape as input minus last dimension
    """
    # Handle numerical edge cases
    if (probabilities.sum(dim=-1) == 0).any():
        monitor = get_monitor()
        monitor.add_warning("Zero probability vectors detected in hard predictions", "warning")
        # Add small epsilon and renormalize
        probabilities = probabilities + 1e-10
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    predictions = probabilities.argmax(dim=-1)
    
    # Apply mask if provided
    if mask is not None:
        predictions = predictions * mask.long()
    
    return predictions


@monitor_prediction('soft')
def compute_soft_predictions(probabilities: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           categories: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute soft predictions using expected value.
    
    The expected value treats categories as ordinal values.
    E[Y] = sum(k * P(Y = k)) for k in {0, 1, ..., K-1}
    
    Args:
        probabilities: Probability distributions, shape (..., K)
        mask: Optional boolean mask for valid positions
        categories: Optional custom category values, shape (K,)
        
    Returns:
        Soft predictions (expected values), same shape as input minus last dimension
    """
    n_cats = probabilities.shape[-1]
    
    # Default categories are 0, 1, 2, ..., K-1
    if categories is None:
        categories = torch.arange(n_cats, dtype=probabilities.dtype, 
                                device=probabilities.device)
    
    # Ensure probabilities are normalized
    prob_sums = probabilities.sum(dim=-1, keepdim=True)
    if (prob_sums == 0).any():
        monitor = get_monitor()
        monitor.add_warning("Zero probability vectors detected in soft predictions", "warning")
        probabilities = probabilities + 1e-10
        prob_sums = probabilities.sum(dim=-1, keepdim=True)
    
    probabilities = probabilities / prob_sums
    
    # Compute expected value
    expected_values = (probabilities * categories.view(1, -1)).sum(dim=-1)
    
    # Apply mask if provided
    if mask is not None:
        expected_values = expected_values * mask.float()
    
    return expected_values


@monitor_prediction('threshold') 
def compute_threshold_predictions(probabilities: torch.Tensor,
                                thresholds: Optional[Union[List[float], torch.Tensor]] = None,
                                mask: Optional[torch.Tensor] = None,
                                adaptive: bool = False,
                                percentiles: Optional[List[float]] = None) -> torch.Tensor:
    """Compute threshold-based predictions using cumulative probabilities.
    
    For ordinal data, we find the category k such that:
    P(Y ≤ k) ≥ threshold and P(Y ≤ k-1) < threshold
    
    Args:
        probabilities: Probability distributions, shape (..., K)
        thresholds: Threshold values for K-1 cumulative probabilities
        mask: Optional boolean mask for valid positions
        adaptive: Whether to compute thresholds from data
        percentiles: Percentiles for adaptive threshold computation
        
    Returns:
        Threshold predictions (category indices), same shape as input minus last dimension
    """
    n_cats = probabilities.shape[-1]
    device = probabilities.device
    
    # Convert to cumulative probabilities P(Y > k)
    cum_probs = categorical_to_cumulative(probabilities)
    
    # Determine thresholds
    if thresholds is None:
        if adaptive and percentiles is not None:
            # Compute adaptive thresholds from data
            thresholds = compute_adaptive_thresholds(
                probabilities, percentiles, mask
            )
        else:
            # Default to median (0.5) for all categories
            thresholds = torch.full((n_cats - 1,), 0.5, device=device)
    else:
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds, device=device)
    
    # Ensure thresholds have correct shape
    if thresholds.dim() == 1 and cum_probs.dim() > 2:
        # Broadcast thresholds to match cum_probs shape
        shape = [1] * (cum_probs.dim() - 1) + [n_cats - 1]
        thresholds = thresholds.view(*shape)
    
    # Find category where cumulative probability crosses threshold
    # We compare P(Y > k) with thresholds
    # If P(Y > k) < threshold[k], then Y ≤ k
    
    # Create predictions starting from highest category
    predictions = torch.zeros_like(probabilities[..., 0], dtype=torch.long)
    
    # Work backwards from highest to lowest category
    for k in range(n_cats - 2, -1, -1):
        # If P(Y > k) >= threshold[k], then predict > k
        mask_gt_k = cum_probs[..., k] >= thresholds[..., k]
        predictions = torch.where(mask_gt_k, k + 1, predictions)
    
    # Apply mask if provided
    if mask is not None:
        predictions = predictions * mask.long()
    
    return predictions


def compute_adaptive_thresholds(probabilities: torch.Tensor,
                              percentiles: List[float] = [25.0, 50.0, 75.0],
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute data-driven thresholds based on probability distributions.
    
    Args:
        probabilities: Probability distributions, shape (..., K)
        percentiles: Percentiles for threshold computation
        mask: Optional boolean mask for valid positions
        
    Returns:
        Adaptive thresholds, shape (K-1,) or matching probabilities shape
    """
    # Convert to cumulative probabilities
    cum_probs = categorical_to_cumulative(probabilities)
    
    # Flatten for percentile computation
    if mask is not None:
        # Only use valid positions
        flat_cum_probs = cum_probs[mask]
    else:
        flat_cum_probs = cum_probs.view(-1, cum_probs.shape[-1])
    
    # Compute percentiles for each cumulative probability
    thresholds = []
    for k in range(cum_probs.shape[-1]):
        k_probs = flat_cum_probs[:, k]
        k_thresholds = torch.quantile(k_probs, 
                                     torch.tensor(percentiles, device=k_probs.device) / 100.0)
        thresholds.append(k_thresholds.mean())  # Use mean of percentiles
    
    return torch.stack(thresholds)


def compute_unified_predictions(probabilities: torch.Tensor,
                              mask: Optional[torch.Tensor] = None,
                              config: Optional[PredictionConfig] = None) -> Dict[str, torch.Tensor]:
    """Compute all three types of predictions in one call.
    
    Args:
        probabilities: Probability distributions, shape (..., K)
        mask: Optional boolean mask for valid positions
        config: Prediction configuration
        
    Returns:
        Dictionary with 'hard', 'soft', and 'threshold' predictions
    """
    if config is None:
        config = PredictionConfig()
    
    # Validate input
    if probabilities.dim() < 2:
        raise ValueError(f"Probabilities must have at least 2 dimensions, got {probabilities.dim()}")
    
    # Move to GPU if requested and available
    if config.use_gpu and torch.cuda.is_available() and not probabilities.is_cuda:
        probabilities = probabilities.cuda()
        if mask is not None:
            mask = mask.cuda()
    elif not config.use_gpu and probabilities.is_cuda:
        # Move to CPU if GPU not requested but tensor is on GPU
        probabilities = probabilities.cpu()
        if mask is not None and mask.is_cuda:
            mask = mask.cpu()
    
    # Cache cumulative probabilities if multiple methods need them
    cum_probs = None
    if config.cache_cumulative:
        cum_probs = categorical_to_cumulative(probabilities, config.epsilon)
    
    # Compute predictions
    predictions = {}
    
    # Hard predictions (argmax)
    predictions['hard'] = compute_hard_predictions(probabilities, mask)
    
    # Soft predictions (expected value)
    predictions['soft'] = compute_soft_predictions(probabilities, mask)
    
    # Threshold predictions
    predictions['threshold'] = compute_threshold_predictions(
        probabilities,
        thresholds=config.thresholds,
        mask=mask,
        adaptive=config.adaptive_thresholds,
        percentiles=config.threshold_percentiles
    )
    
    # If not using GPU, ensure all predictions are on CPU
    if not config.use_gpu:
        for key in ['hard', 'soft', 'threshold']:
            if predictions[key].is_cuda:
                predictions[key] = predictions[key].cpu()
    
    # Add auxiliary information (ensure CPU for compatibility)
    predictions['probabilities'] = probabilities.cpu() if probabilities.is_cuda else probabilities
    predictions['mask'] = mask.cpu() if mask is not None and mask.is_cuda else mask
    
    # Compare predictions for monitoring
    monitor = get_monitor()
    # Ensure all predictions are on CPU for monitoring
    monitor_preds = {
        'hard': predictions['hard'].cpu() if predictions['hard'].is_cuda else predictions['hard'],
        'soft': predictions['soft'].round().long().cpu() if predictions['soft'].is_cuda else predictions['soft'].round().long(),
        'threshold': predictions['threshold'].cpu() if predictions['threshold'].is_cuda else predictions['threshold']
    }
    monitor.compare_predictions(monitor_preds)
    
    return predictions


# Model output adapters for different architectures
def extract_probabilities_from_model_output(model_output: Union[Tuple, torch.Tensor], 
                                          model_type: str = "auto") -> torch.Tensor:
    """Extract probability tensor from model output.
    
    All Deep-GPCM models return:
    (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
    
    Args:
        model_output: Output from model forward pass
        model_type: Model type for validation
        
    Returns:
        Probability tensor with shape (batch_size, seq_len, n_cats)
    """
    if isinstance(model_output, tuple) and len(model_output) == 4:
        # Standard Deep-GPCM output format
        _, _, _, probabilities = model_output
        return probabilities
    elif isinstance(model_output, torch.Tensor):
        # Direct probability tensor
        return model_output
    else:
        raise ValueError(f"Unknown model output format: {type(model_output)}")


# Vectorized operations for efficiency
def compute_predictions_batched(probabilities: torch.Tensor,
                              mask: Optional[torch.Tensor] = None,
                              config: Optional[PredictionConfig] = None,
                              batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Compute predictions in batches for memory efficiency.
    
    Useful for very large datasets that don't fit in memory.
    
    Args:
        probabilities: Large probability tensor
        mask: Optional mask tensor
        config: Prediction configuration
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with batched predictions
    """
    if batch_size is None:
        batch_size = config.batch_size if config else 1000
    
    n_samples = probabilities.shape[0]
    all_predictions = {
        'hard': [],
        'soft': [],
        'threshold': []
    }
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_probs = probabilities[i:end_idx]
        batch_mask = mask[i:end_idx] if mask is not None else None
        
        batch_preds = compute_unified_predictions(batch_probs, batch_mask, config)
        
        for method in ['hard', 'soft', 'threshold']:
            all_predictions[method].append(batch_preds[method])
    
    # Concatenate results
    for method in ['hard', 'soft', 'threshold']:
        all_predictions[method] = torch.cat(all_predictions[method], dim=0)
    
    # Add full tensors
    all_predictions['probabilities'] = probabilities
    all_predictions['mask'] = mask
    
    return all_predictions