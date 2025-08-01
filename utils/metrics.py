#!/usr/bin/env python3
"""
Simple Metrics System for Deep-GPCM Models
Configurable metrics list with direct JSON handling.
"""

import json
import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, cohen_kappa_score, log_loss
)
from scipy import stats
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class OrdinalMetrics:
    """Comprehensive ordinal prediction metrics for Deep-GPCM evaluation."""
    
    def __init__(self, n_cats: int = 4):
        """Initialize with number of categories."""
        self.n_cats = n_cats
        self.categories = list(range(n_cats))
    
    def categorical_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Standard categorical accuracy."""
        return accuracy_score(y_true, y_pred)
    
    def ordinal_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Ordinal accuracy - correct predictions within ±1 category."""
        diff = np.abs(y_true - y_pred)
        return np.mean(diff <= 1)
    
    def adjacent_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Adjacent accuracy - exact or adjacent category predictions."""
        return self.ordinal_accuracy(y_true, y_pred)
    
    def quadratic_weighted_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Quadratic weighted kappa for ordinal data."""
        def quadratic_weight_matrix(n_cats):
            weights = np.zeros((n_cats, n_cats))
            for i in range(n_cats):
                for j in range(n_cats):
                    weights[i, j] = (i - j) ** 2 / (n_cats - 1) ** 2
            return weights
        
        weights = quadratic_weight_matrix(self.n_cats)
        conf_mat = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        # Normalize confusion matrix
        conf_mat = conf_mat.astype(float)
        if conf_mat.sum() == 0:
            return 0.0
        
        # Calculate expected matrix
        n_scored_items = conf_mat.sum()
        hist_true = conf_mat.sum(axis=1) / n_scored_items
        hist_pred = conf_mat.sum(axis=0) / n_scored_items
        expected = np.outer(hist_true, hist_pred) * n_scored_items
        
        # Calculate kappa
        observed = (weights * conf_mat).sum()
        expected_weighted = (weights * expected).sum()
        
        if expected_weighted == 0:
            return 0.0
        
        return 1 - observed / expected_weighted
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error treating categories as ordinal."""
        return mean_absolute_error(y_true, y_pred)
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error treating categories as ordinal.""" 
        return mean_squared_error(y_true, y_pred)
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(self.mean_squared_error(y_true, y_pred))
    
    def kendall_tau(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Kendall's tau correlation coefficient."""
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return 0.0
        tau, _ = stats.kendalltau(y_true, y_pred)
        return tau if not np.isnan(tau) else 0.0
    
    def spearman_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Spearman rank correlation coefficient."""
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return 0.0
        rho, _ = stats.spearmanr(y_true, y_pred)
        return rho if not np.isnan(rho) else 0.0
    
    def pearson_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        r, _ = stats.pearsonr(y_true, y_pred)
        return r if not np.isnan(r) else 0.0
    
    def cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cohen's kappa coefficient."""
        return cohen_kappa_score(y_true, y_pred)
    
    def ordinal_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Ordinal loss - sum of absolute differences."""
        return np.sum(np.abs(y_true - y_pred))
    
    def category_accuracy_breakdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Per-category accuracy breakdown."""
        breakdown = {}
        for cat in self.categories:
            mask = y_true == cat
            if mask.sum() > 0:
                cat_acc = (y_pred[mask] == cat).mean()
                breakdown[f'cat_{cat}_accuracy'] = cat_acc
            else:
                breakdown[f'cat_{cat}_accuracy'] = 0.0
        return breakdown
    
    def category_count_breakdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Per-category count breakdown."""
        breakdown = {}
        for cat in self.categories:
            breakdown[f'cat_{cat}_true_count'] = (y_true == cat).sum()
            breakdown[f'cat_{cat}_pred_count'] = (y_pred == cat).sum()
        return breakdown
    
    def confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Confusion matrix and derived metrics."""
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        
        # Per-class precision, recall, F1
        report = classification_report(y_true, y_pred, labels=self.categories, 
                                     output_dict=True, zero_division=0)
        
        metrics = {}
        # Overall metrics
        metrics['macro_precision'] = report['macro avg']['precision']
        metrics['macro_recall'] = report['macro avg']['recall']
        metrics['macro_f1'] = report['macro avg']['f1-score']
        metrics['weighted_precision'] = report['weighted avg']['precision']
        metrics['weighted_recall'] = report['weighted avg']['recall']
        metrics['weighted_f1'] = report['weighted avg']['f1-score']
        
        # Per-category metrics
        for cat in self.categories:
            cat_str = str(cat)
            if cat_str in report:
                metrics[f'cat_{cat}_precision'] = report[cat_str]['precision']
                metrics[f'cat_{cat}_recall'] = report[cat_str]['recall']
                metrics[f'cat_{cat}_f1'] = report[cat_str]['f1-score']
        
        return metrics
    
    def probability_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Metrics based on probability predictions."""
        metrics = {}
        
        # Cross-entropy loss
        try:
            metrics['cross_entropy'] = log_loss(y_true, y_prob, labels=self.categories)
        except:
            metrics['cross_entropy'] = float('inf')
        
        # Confidence metrics
        max_probs = np.max(y_prob, axis=1)
        metrics['mean_confidence'] = np.mean(max_probs)
        metrics['confidence_std'] = np.std(max_probs)
        
        # Entropy
        epsilon = 1e-15
        y_prob_safe = np.clip(y_prob, epsilon, 1 - epsilon)
        entropy = -np.sum(y_prob_safe * np.log(y_prob_safe), axis=1)
        metrics['mean_entropy'] = np.mean(entropy)
        metrics['entropy_std'] = np.std(entropy)
        
        # Calibration - expected vs actual probabilities
        y_pred = np.argmax(y_prob, axis=1)
        pred_probs = max_probs
        correct = (y_pred == y_true).astype(float)
        
        # Bin calibration
        n_bins = min(10, len(np.unique(pred_probs)))
        if n_bins > 1:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correct[in_bin].mean()
                    avg_confidence_in_bin = pred_probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['expected_calibration_error'] = ece
        else:
            metrics['expected_calibration_error'] = 0.0
        
        return metrics
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute all available metrics."""
        metrics = {}
        
        # Basic ordinal metrics
        metrics['categorical_accuracy'] = self.categorical_accuracy(y_true, y_pred)
        metrics['ordinal_accuracy'] = self.ordinal_accuracy(y_true, y_pred)
        metrics['adjacent_accuracy'] = self.adjacent_accuracy(y_true, y_pred)
        metrics['quadratic_weighted_kappa'] = self.quadratic_weighted_kappa(y_true, y_pred)
        metrics['mean_absolute_error'] = self.mean_absolute_error(y_true, y_pred)
        metrics['mean_squared_error'] = self.mean_squared_error(y_true, y_pred)
        metrics['root_mean_squared_error'] = self.root_mean_squared_error(y_true, y_pred)
        metrics['kendall_tau'] = self.kendall_tau(y_true, y_pred)
        metrics['spearman_correlation'] = self.spearman_correlation(y_true, y_pred)
        metrics['pearson_correlation'] = self.pearson_correlation(y_true, y_pred)
        metrics['cohen_kappa'] = self.cohen_kappa(y_true, y_pred)
        metrics['ordinal_loss'] = self.ordinal_loss(y_true, y_pred)
        
        # Category breakdowns
        metrics.update(self.category_accuracy_breakdown(y_true, y_pred))
        metrics.update(self.category_count_breakdown(y_true, y_pred))
        metrics.update(self.confusion_matrix_metrics(y_true, y_pred))
        
        # Probability-based metrics if available
        if y_prob is not None:
            metrics.update(self.probability_metrics(y_true, y_prob))
        
        return metrics


# Configurable metrics list - easy to modify
DEFAULT_METRICS = [
    'categorical_accuracy',
    'ordinal_accuracy', 
    'quadratic_weighted_kappa',
    'mean_absolute_error',
    'kendall_tau',
    'spearman_correlation',
    'cohen_kappa',
    'cross_entropy',
    'mean_confidence'
]

# Simple utility functions
def ensure_results_dirs(results_dir: str = "results"):
    """Create results directory structure."""
    dirs = ['train', 'valid', 'test', 'plots']
    for d in dirs:
        os.makedirs(os.path.join(results_dir, d), exist_ok=True)

def compute_metrics(y_true: Union[torch.Tensor, np.ndarray], 
                   y_pred: Union[torch.Tensor, np.ndarray],
                   y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
                   n_cats: int = 4,
                   metrics_list: List[str] = None) -> Dict[str, Any]:
    """Compute specified metrics."""
    if metrics_list is None:
        metrics_list = DEFAULT_METRICS
    
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_prob is not None:
        if y_prob.ndim > 2:
            y_prob = y_prob.reshape(-1, y_prob.shape[-1])
    
    ordinal_metrics = OrdinalMetrics(n_cats)
    all_metrics = ordinal_metrics.compute_all_metrics(y_true, y_pred, y_prob)
    
    # Return only requested metrics
    return {k: v for k, v in all_metrics.items() if k in metrics_list or k.startswith('cat_')}

def save_results(data: Dict[str, Any], filepath: str) -> str:
    """Simple JSON save with timestamp."""
    data['timestamp'] = datetime.now().isoformat()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath

def compute_model_metrics(model, data_loader, device, n_cats: int = 4, 
                         metrics_list: List[str] = None) -> Dict[str, Any]:
    """Compute metrics for a trained model."""
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for questions, responses, masks in data_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            masks = masks.to(device)
            
            # Forward pass
            _, _, _, probs = model(questions, responses)
            
            # Flatten and apply mask to exclude padding tokens
            probs_flat = probs.view(-1, probs.size(-1))
            targets_flat = responses.view(-1)
            masks_flat = masks.view(-1).bool()
            
            # Filter out padding tokens
            valid_probs = probs_flat[masks_flat]
            valid_targets = targets_flat[masks_flat]
            valid_preds = valid_probs.argmax(dim=-1)
            
            all_targets.append(valid_targets.cpu())
            all_predictions.append(valid_preds.cpu())
            all_probabilities.append(valid_probs.cpu())
    
    # Combine all batches
    all_targets = torch.cat(all_targets)
    all_predictions = torch.cat(all_predictions)
    all_probabilities = torch.cat(all_probabilities)
    
    # Compute metrics
    return compute_metrics(all_targets, all_predictions, all_probabilities, 
                          n_cats, metrics_list)