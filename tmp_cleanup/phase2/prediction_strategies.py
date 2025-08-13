#!/usr/bin/env python3
"""
Prediction strategy classes for the unified prediction system.
Provides clean abstraction for different prediction methods.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np

from .predictions import (
    categorical_to_cumulative,
    compute_adaptive_thresholds,
    PredictionConfig
)

class PredictionStrategy(ABC):
    """Abstract base class for prediction strategies."""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.name = self.__class__.__name__
        
    @abstractmethod
    def predict(self, probabilities: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate predictions from probability distributions."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the prediction strategy."""
        pass
    
    def validate_input(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Validate and clean input probabilities."""
        # Ensure minimum dimensions
        if probabilities.dim() < 2:
            raise ValueError(f"Probabilities must have at least 2 dimensions, got {probabilities.dim()}")
        
        # Handle numerical issues
        probabilities = torch.clamp(probabilities, min=self.config.epsilon)
        
        # Normalize if needed
        prob_sums = probabilities.sum(dim=-1, keepdim=True)
        if (prob_sums == 0).any():
            probabilities = probabilities + self.config.epsilon
            prob_sums = probabilities.sum(dim=-1, keepdim=True)
        
        probabilities = probabilities / prob_sums
        
        return probabilities


class HardPredictionStrategy(PredictionStrategy):
    """Argmax-based hard prediction strategy."""
    
    def predict(self, probabilities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate hard predictions using argmax."""
        probabilities = self.validate_input(probabilities)
        
        # Argmax prediction
        predictions = probabilities.argmax(dim=-1)
        
        # Apply mask
        if mask is not None:
            predictions = predictions * mask.long()
        
        return predictions
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "HardPrediction",
            "method": "argmax",
            "description": "Maximum probability category selection",
            "properties": {
                "deterministic": True,
                "discrete": True,
                "ordinal_aware": False
            }
        }


class SoftPredictionStrategy(PredictionStrategy):
    """Expected value-based soft prediction strategy."""
    
    def __init__(self, config: Optional[PredictionConfig] = None,
                 categories: Optional[torch.Tensor] = None):
        super().__init__(config)
        self.categories = categories
        
    def predict(self, probabilities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate soft predictions using expected value."""
        probabilities = self.validate_input(probabilities)
        
        n_cats = probabilities.shape[-1]
        device = probabilities.device
        
        # Use custom or default categories
        if self.categories is None:
            categories = torch.arange(n_cats, dtype=probabilities.dtype, device=device)
        else:
            categories = self.categories.to(device)
        
        # Expected value calculation
        predictions = (probabilities * categories.view(1, -1)).sum(dim=-1)
        
        # Apply mask
        if mask is not None:
            predictions = predictions * mask.float()
        
        return predictions
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SoftPrediction",
            "method": "expected_value",
            "description": "Probability-weighted average of categories",
            "properties": {
                "deterministic": True,
                "discrete": False,
                "ordinal_aware": True,
                "continuous_output": True
            }
        }


class ThresholdPredictionStrategy(PredictionStrategy):
    """Cumulative threshold-based prediction strategy."""
    
    def __init__(self, config: Optional[PredictionConfig] = None,
                 thresholds: Optional[torch.Tensor] = None,
                 adaptive: bool = False):
        super().__init__(config)
        self.thresholds = thresholds
        self.adaptive = adaptive or (config and config.adaptive_thresholds)
        self._last_thresholds = None
        
    def predict(self, probabilities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate threshold-based predictions."""
        probabilities = self.validate_input(probabilities)
        
        n_cats = probabilities.shape[-1]
        device = probabilities.device
        
        # Convert to cumulative probabilities
        cum_probs = categorical_to_cumulative(probabilities, self.config.epsilon)
        
        # Determine thresholds
        if self.thresholds is not None:
            thresholds = self.thresholds.to(device)
        elif self.adaptive:
            thresholds = compute_adaptive_thresholds(
                probabilities,
                self.config.threshold_percentiles or [25.0, 50.0, 75.0],
                mask
            )
        else:
            # Default evenly spaced thresholds (descending for P(Y > k))
            thresholds = torch.linspace(0.75, 0.25, n_cats - 1, device=device)
        
        self._last_thresholds = thresholds
        
        # Ensure correct shape
        if thresholds.dim() == 1 and cum_probs.dim() > 2:
            shape = [1] * (cum_probs.dim() - 1) + [n_cats - 1]
            thresholds = thresholds.view(*shape)
        
        # Threshold-based prediction
        predictions = torch.zeros_like(probabilities[..., 0], dtype=torch.long)
        
        # Work backwards from highest to lowest category
        for k in range(n_cats - 2, -1, -1):
            mask_gt_k = cum_probs[..., k] >= thresholds[..., k]
            predictions = torch.where(mask_gt_k, k + 1, predictions)
        
        # Apply mask
        if mask is not None:
            predictions = predictions * mask.long()
        
        return predictions
    
    def get_thresholds(self) -> Optional[torch.Tensor]:
        """Get the last used thresholds."""
        return self._last_thresholds
    
    def get_info(self) -> Dict[str, Any]:
        info = {
            "name": "ThresholdPrediction",
            "method": "cumulative_threshold",
            "description": "Ordinal prediction based on cumulative probability thresholds",
            "properties": {
                "deterministic": True,
                "discrete": True,
                "ordinal_aware": True,
                "adaptive": self.adaptive
            }
        }
        
        if self._last_thresholds is not None:
            info["last_thresholds"] = self._last_thresholds.tolist()
        
        return info


class EnsemblePredictionStrategy(PredictionStrategy):
    """Ensemble multiple prediction strategies."""
    
    def __init__(self, strategies: List[PredictionStrategy],
                 weights: Optional[List[float]] = None,
                 voting: str = "weighted"):
        super().__init__()
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.voting = voting  # "weighted", "majority", "unanimous"
        
    def predict(self, probabilities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate ensemble predictions."""
        predictions = []
        
        # Get predictions from all strategies
        for strategy in self.strategies:
            pred = strategy.predict(probabilities, mask)
            predictions.append(pred)
        
        # Stack predictions
        pred_stack = torch.stack(predictions, dim=0)  # (n_strategies, ...)
        
        if self.voting == "weighted":
            # For continuous predictions (soft), use weighted average
            if any(isinstance(s, SoftPredictionStrategy) for s in self.strategies):
                weights = torch.tensor(self.weights, device=pred_stack.device)
                weights = weights.view(-1, *([1] * (pred_stack.dim() - 1)))
                ensemble_pred = (pred_stack.float() * weights).sum(dim=0)
            else:
                # For discrete predictions, use weighted voting
                ensemble_pred = self._weighted_vote(pred_stack, self.weights)
        
        elif self.voting == "majority":
            ensemble_pred = torch.mode(pred_stack, dim=0)[0]
        
        elif self.voting == "unanimous":
            # All strategies must agree
            first_pred = pred_stack[0]
            unanimous = (pred_stack == first_pred.unsqueeze(0)).all(dim=0)
            # Use first prediction where unanimous, otherwise use weighted vote
            ensemble_pred = torch.where(
                unanimous,
                first_pred,
                self._weighted_vote(pred_stack, self.weights)
            )
        
        return ensemble_pred
    
    def _weighted_vote(self, predictions: torch.Tensor, 
                      weights: List[float]) -> torch.Tensor:
        """Perform weighted voting for discrete predictions."""
        n_cats = predictions.max() + 1
        device = predictions.device
        
        # Create one-hot encodings
        one_hot = F.one_hot(predictions, num_classes=n_cats)  # (n_strat, ..., n_cats)
        
        # Apply weights
        weights_tensor = torch.tensor(weights, device=device)
        weights_tensor = weights_tensor.view(-1, *([1] * (one_hot.dim() - 1)))
        
        # Weighted sum
        weighted_votes = (one_hot.float() * weights_tensor).sum(dim=0)
        
        # Get category with most votes
        return weighted_votes.argmax(dim=-1)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "EnsemblePrediction",
            "method": f"ensemble_{self.voting}",
            "description": f"Ensemble of {len(self.strategies)} strategies",
            "strategies": [s.get_info()["name"] for s in self.strategies],
            "weights": self.weights,
            "voting": self.voting
        }


class StochasticPredictionStrategy(PredictionStrategy):
    """Stochastic sampling-based prediction strategy."""
    
    def __init__(self, config: Optional[PredictionConfig] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None):
        super().__init__(config)
        self.temperature = temperature
        self.top_k = top_k
        
    def predict(self, probabilities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate predictions by sampling from probability distribution."""
        probabilities = self.validate_input(probabilities)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            # Convert to logits, scale, convert back
            logits = torch.log(probabilities + self.config.epsilon)
            logits = logits / self.temperature
            probabilities = torch.softmax(logits, dim=-1)
        
        # Apply top-k filtering if specified
        if self.top_k is not None and self.top_k < probabilities.shape[-1]:
            values, indices = probabilities.topk(self.top_k, dim=-1)
            probabilities_filtered = torch.zeros_like(probabilities)
            probabilities_filtered.scatter_(-1, indices, values)
            probabilities = probabilities_filtered / probabilities_filtered.sum(dim=-1, keepdim=True)
        
        # Sample from distribution
        predictions = torch.multinomial(probabilities.view(-1, probabilities.shape[-1]), 1)
        predictions = predictions.view(probabilities.shape[:-1])
        
        # Apply mask
        if mask is not None:
            predictions = predictions * mask.long()
        
        return predictions
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "StochasticPrediction",
            "method": "sampling",
            "description": "Sample from probability distribution",
            "properties": {
                "deterministic": False,
                "discrete": True,
                "ordinal_aware": False,
                "temperature": self.temperature,
                "top_k": self.top_k
            }
        }


# Factory function for creating strategies
def create_prediction_strategy(method: str, 
                             config: Optional[PredictionConfig] = None,
                             **kwargs) -> PredictionStrategy:
    """Factory function to create prediction strategies.
    
    Args:
        method: Strategy name ('hard', 'soft', 'threshold', 'ensemble', 'stochastic')
        config: Prediction configuration
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        PredictionStrategy instance
    """
    method = method.lower()
    
    if method == 'hard':
        return HardPredictionStrategy(config)
    
    elif method == 'soft':
        categories = kwargs.get('categories', None)
        return SoftPredictionStrategy(config, categories)
    
    elif method == 'threshold':
        thresholds = kwargs.get('thresholds', None)
        adaptive = kwargs.get('adaptive', False)
        return ThresholdPredictionStrategy(config, thresholds, adaptive)
    
    elif method == 'ensemble':
        strategies = kwargs.get('strategies', [])
        weights = kwargs.get('weights', None)
        voting = kwargs.get('voting', 'weighted')
        return EnsemblePredictionStrategy(strategies, weights, voting)
    
    elif method == 'stochastic':
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', None)
        return StochasticPredictionStrategy(config, temperature, top_k)
    
    else:
        raise ValueError(f"Unknown prediction strategy: {method}")