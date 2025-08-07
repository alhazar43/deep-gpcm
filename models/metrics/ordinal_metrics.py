"""
Ordinal-aware metrics for evaluating GPCM models.
Implements metrics that consider ordinal relationships.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class OrdinalMetrics:
    """Collection of ordinal-aware metrics."""
    
    def __init__(self, n_cats: int):
        self.n_cats = n_cats
    
    def calculate_all_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                            y_probs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate all ordinal metrics.
        
        Args:
            y_true: True labels [batch*seq] or [batch, seq]
            y_pred: Predicted labels [batch*seq] or [batch, seq]
            y_probs: Predicted probabilities [batch*seq, n_cats] or [batch, seq, n_cats]
            
        Returns:
            Dictionary of metric values
        """
        # Flatten if necessary
        if y_true.dim() > 1:
            y_true = y_true.flatten()
        if y_pred.dim() > 1:
            y_pred = y_pred.flatten()
        if y_probs is not None and y_probs.dim() > 2:
            y_probs = y_probs.view(-1, y_probs.size(-1))
        
        # Remove padding (assuming 0 is padding)
        mask = y_true > 0
        if mask.sum() == 0:
            return {metric: 0.0 for metric in self._get_metric_names()}
        
        y_true_clean = y_true[mask].cpu().numpy()
        y_pred_clean = y_pred[mask].cpu().numpy()
        
        if y_probs is not None:
            y_probs_clean = y_probs[mask].cpu().numpy()
        else:
            y_probs_clean = None
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true_clean, y_pred_clean)
        
        # Ordinal metrics
        metrics['qwk'] = self._quadratic_weighted_kappa(y_true_clean, y_pred_clean)
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
        metrics['ordinal_accuracy'] = self._ordinal_accuracy(y_true_clean, y_pred_clean)
        metrics['adjacent_accuracy'] = self._adjacent_accuracy(y_true_clean, y_pred_clean)
        
        # Distance-based metrics
        metrics['mean_ordinal_distance'] = self._mean_ordinal_distance(y_true_clean, y_pred_clean)
        metrics['max_ordinal_distance'] = self._max_ordinal_distance(y_true_clean, y_pred_clean)
        
        # Distribution metrics
        if y_probs_clean is not None:
            metrics['cross_entropy'] = self._cross_entropy(y_true_clean, y_probs_clean)
            metrics['brier_score'] = self._brier_score(y_true_clean, y_probs_clean)
            metrics['expected_calibration_error'] = self._expected_calibration_error(
                y_true_clean, y_pred_clean, y_probs_clean)
        
        return metrics
    
    def _quadratic_weighted_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Quadratic Weighted Kappa."""
        try:
            return cohen_kappa_score(y_true, y_pred, weights='quadratic')
        except:
            return 0.0
    
    def _ordinal_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy considering ordinal nature (exact + adjacent matches)."""
        exact_matches = (y_true == y_pred).sum()
        adjacent_matches = (np.abs(y_true - y_pred) <= 1).sum()
        return adjacent_matches / len(y_true)
    
    def _adjacent_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy within one category."""
        return (np.abs(y_true - y_pred) <= 1).mean()
    
    def _mean_ordinal_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute ordinal distance."""
        return np.abs(y_true - y_pred).mean()
    
    def _max_ordinal_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Maximum ordinal distance."""
        return np.abs(y_true - y_pred).max()
    
    def _cross_entropy(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Cross-entropy loss."""
        epsilon = 1e-15
        y_probs = np.clip(y_probs, epsilon, 1 - epsilon)
        return -np.log(y_probs[range(len(y_true)), y_true]).mean()
    
    def _brier_score(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Brier score for ordinal classification."""
        # Convert to one-hot
        y_true_onehot = np.eye(self.n_cats)[y_true]
        return np.mean((y_probs - y_true_onehot) ** 2)
    
    def _expected_calibration_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_probs: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        # Get confidence scores
        confidences = y_probs.max(axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in bin
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        return [
            'accuracy', 'qwk', 'mae', 'ordinal_accuracy', 'adjacent_accuracy',
            'mean_ordinal_distance', 'max_ordinal_distance', 'cross_entropy',
            'brier_score', 'expected_calibration_error'
        ]


class MetricsTracker:
    """Tracks metrics over training epochs."""
    
    def __init__(self, n_cats: int):
        self.n_cats = n_cats
        self.ordinal_metrics = OrdinalMetrics(n_cats)
        self.history = {'train': {}, 'val': {}}
        
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
               y_probs: Optional[torch.Tensor] = None, split: str = 'train'):
        """Update metrics for current epoch."""
        metrics = self.ordinal_metrics.calculate_all_metrics(y_true, y_pred, y_probs)
        
        # Initialize metric lists if first time
        for metric_name, value in metrics.items():
            if metric_name not in self.history[split]:
                self.history[split][metric_name] = []
            self.history[split][metric_name].append(value)
        
        return metrics
    
    def get_best_metrics(self, split: str = 'val', metric: str = 'qwk') -> Dict[str, float]:
        """Get best metrics based on a specific metric."""
        if split not in self.history or metric not in self.history[split]:
            return {}
        
        # Find best epoch
        metric_values = self.history[split][metric]
        if metric in ['cross_entropy', 'brier_score', 'mae', 'mean_ordinal_distance', 'max_ordinal_distance']:
            best_epoch = np.argmin(metric_values)  # Lower is better
        else:
            best_epoch = np.argmax(metric_values)  # Higher is better
        
        # Get all metrics for best epoch
        best_metrics = {}
        for metric_name, values in self.history[split].items():
            if len(values) > best_epoch:
                best_metrics[metric_name] = values[best_epoch]
        
        best_metrics['best_epoch'] = best_epoch
        return best_metrics
    
    def get_current_metrics(self, split: str = 'val') -> Dict[str, float]:
        """Get metrics for current (last) epoch."""
        current_metrics = {}
        for metric_name, values in self.history[split].items():
            if values:
                current_metrics[metric_name] = values[-1]
        return current_metrics
    
    def print_metrics(self, split: str, epoch: int):
        """Print formatted metrics."""
        current = self.get_current_metrics(split)
        if not current:
            return
        
        print(f"\nEpoch {epoch} - {split.upper()} Metrics:")
        print(f"  Accuracy: {current.get('accuracy', 0):.3f}")
        print(f"  QWK: {current.get('qwk', 0):.3f}")
        print(f"  MAE: {current.get('mae', 0):.3f}")
        print(f"  Ordinal Acc: {current.get('ordinal_accuracy', 0):.3f}")
        print(f"  Adjacent Acc: {current.get('adjacent_accuracy', 0):.3f}")
        if 'cross_entropy' in current:
            print(f"  Cross-Entropy: {current['cross_entropy']:.3f}")


def calculate_ordinal_improvement(baseline_metrics: Dict[str, float], 
                                ordinal_metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate improvement of ordinal model over baseline."""
    improvements = {}
    
    for metric in ['qwk', 'ordinal_accuracy', 'adjacent_accuracy']:
        if metric in baseline_metrics and metric in ordinal_metrics:
            baseline_val = baseline_metrics[metric]
            ordinal_val = ordinal_metrics[metric]
            
            if baseline_val > 0:
                improvement = (ordinal_val - baseline_val) / baseline_val * 100
            else:
                improvement = ordinal_val * 100
            
            improvements[f'{metric}_improvement'] = improvement
    
    # For MAE and distances (lower is better)
    for metric in ['mae', 'mean_ordinal_distance']:
        if metric in baseline_metrics and metric in ordinal_metrics:
            baseline_val = baseline_metrics[metric]
            ordinal_val = ordinal_metrics[metric]
            
            if baseline_val > 0:
                improvement = (baseline_val - ordinal_val) / baseline_val * 100
            else:
                improvement = -ordinal_val * 100
            
            improvements[f'{metric}_improvement'] = improvement
    
    return improvements