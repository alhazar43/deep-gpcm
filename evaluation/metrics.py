"""
Evaluation metrics for the Deep-GPCM model.
"""

import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

class GpcmMetrics:
    """
    Evaluation metrics for GPCM including categorical and ordinal accuracy.
    """
    
    @staticmethod
    def categorical_accuracy(predictions, targets):
        """Exact category match accuracy using argmax."""
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
    def prediction_consistency_accuracy(predictions, targets, method='cumulative'):
        """
        New prediction accuracy metric that measures consistency 
        between ordinal training loss and prediction method.
        
        Args:
            predictions: GPCM probabilities, shape (..., K)
            targets: True categories, shape (...,)
            method: Prediction method to simulate ('argmax', 'cumulative', 'expected')
            
        Returns:
            Consistency accuracy score
        """
        if method == 'argmax':
            pred_cats = torch.argmax(predictions, dim=-1)
        elif method == 'cumulative':
            # Simulate cumulative prediction
            cum_probs = torch.cumsum(predictions, dim=-1)
            pred_cats = torch.zeros_like(cum_probs[..., 0])
            for k in range(predictions.shape[-1]):
                mask = cum_probs[..., k] > 0.5
                pred_cats = torch.where(mask & (pred_cats == 0), 
                                      torch.tensor(k, dtype=pred_cats.dtype, device=pred_cats.device),
                                      pred_cats)
        elif method == 'expected':
            # Expected value prediction
            n_cats = predictions.shape[-1]
            categories = torch.arange(n_cats, dtype=predictions.dtype, device=predictions.device)
            expected_values = torch.sum(predictions * categories, dim=-1)
            pred_cats = torch.round(expected_values).clamp(0, n_cats - 1)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        # Calculate accuracy
        correct = (pred_cats == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def ordinal_ranking_accuracy(predictions, targets):
        """
        Measures how well the model preserves ordinal ranking.
        Computes Spearman correlation between predicted and true ordinal values.
        """
        
        # Use expected values as continuous predictions
        n_cats = predictions.shape[-1]
        categories = torch.arange(n_cats, dtype=predictions.dtype, device=predictions.device)
        expected_values = torch.sum(predictions * categories, dim=-1)
        
        # Flatten arrays
        pred_vals = expected_values.cpu().numpy().flatten()
        true_vals = targets.cpu().numpy().flatten()
        
        # Calculate Spearman correlation
        correlation, _ = spearmanr(pred_vals, true_vals)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def distribution_consistency_score(predictions, targets):
        """
        Measures consistency between predicted probability distribution
        and the ordinal structure implied by true categories.
        """
        n_cats = predictions.shape[-1]
        
        # For each sample, check if probability mass aligns with ordinal structure
        consistency_scores = []
        
        for pred, target in zip(predictions, targets):
            target_int = int(target.item())
            
            # Expected: higher probabilities for categories closer to target
            distances = torch.abs(torch.arange(n_cats, device=pred.device) - target_int)
            
            # Compute weighted probability by inverse distance (closer = higher weight)
            weights = 1.0 / (distances.float() + 1.0)  # +1 to avoid division by zero
            weighted_prob = (pred * weights).sum()
            
            consistency_scores.append(weighted_prob.item())
        
        return np.mean(consistency_scores)
    
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
