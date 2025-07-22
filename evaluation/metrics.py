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
    Enhanced with multiple prediction methods for proper train/inference alignment.
    """
    
    @staticmethod
    def categorical_accuracy(predictions, targets, method='argmax'):
        """
        Exact category match accuracy with configurable prediction method.
        
        Args:
            predictions: GPCM probabilities, shape (..., K)
            targets: True categories, shape (...,)
            method: Prediction method ('argmax', 'cumulative', 'expected')
        """
        pred_cats = GpcmMetrics._get_predictions(predictions, method)
        correct = (pred_cats == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def _get_predictions(predictions, method='argmax'):
        """
        Get category predictions using specified method.
        
        This is the critical fix for training/inference alignment.
        """
        if method == 'argmax':
            return torch.argmax(predictions, dim=-1)
        elif method == 'cumulative':
            # FIXED: Proper cumulative prediction aligned with OrdinalLoss
            # The correct interpretation: predict the first category where P(Y ≤ k) > threshold
            # But this means P(Y = 0) = P(Y ≤ 0), so we need to handle category 0 correctly
            
            cum_probs = torch.cumsum(predictions, dim=-1)
            n_cats = predictions.shape[-1]
            
            # Initialize predictions to the highest category (default)
            pred_cats = torch.full_like(cum_probs[..., 0], n_cats - 1, dtype=torch.long)
            
            # Work backwards: find the first threshold that is NOT exceeded
            for k in range(n_cats - 1, -1, -1):
                if k == 0:
                    # Category 0: predict if P(Y=0) > 0.5 directly
                    mask = predictions[..., 0] > 0.5
                else:
                    # Category k: predict if P(Y ≤ k-1) ≤ 0.5 AND P(Y ≤ k) > 0.5
                    mask = (cum_probs[..., k-1] <= 0.5) & (cum_probs[..., k] > 0.5)
                
                pred_cats = torch.where(mask, 
                                      torch.tensor(k, dtype=pred_cats.dtype, device=pred_cats.device),
                                      pred_cats)
            
            return pred_cats
        elif method == 'expected':
            # Expected value prediction
            n_cats = predictions.shape[-1]
            categories = torch.arange(n_cats, dtype=predictions.dtype, device=predictions.device)
            expected_values = torch.sum(predictions * categories, dim=-1)
            return torch.round(expected_values).clamp(0, n_cats - 1)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
    
    @staticmethod
    def ordinal_accuracy(predictions, targets, tolerance=1, method='argmax'):
        """Accuracy within tolerance categories."""
        pred_cats = GpcmMetrics._get_predictions(predictions, method)
        diff = torch.abs(pred_cats - targets)
        within_tolerance = (diff <= tolerance).float()
        return within_tolerance.mean().item()
    
    @staticmethod
    def mean_absolute_error(predictions, targets, method='argmax'):
        """MAE treating categories as ordinal values."""
        pred_cats = GpcmMetrics._get_predictions(predictions, method)
        mae = torch.abs(pred_cats.float() - targets.float()).mean()
        return mae.item()
    
    @staticmethod
    def quadratic_weighted_kappa(predictions, targets, n_cats, method='argmax'):
        """
        Quadratic Weighted Kappa for ordinal data.
        Uses sklearn implementation on flattened arrays.
        """
        pred_cats = GpcmMetrics._get_predictions(predictions, method)
        
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
    def per_category_accuracy(predictions, targets, n_cats, method='argmax'):
        """Per-category accuracy scores."""
        pred_cats = GpcmMetrics._get_predictions(predictions, method)
        
        accuracies = {}
        for k in range(n_cats):
            mask = (targets == k)
            if mask.sum() > 0:
                correct = (pred_cats[mask] == k).float().mean()
                accuracies[f'cat_{k}_acc'] = correct.item()
            else:
                accuracies[f'cat_{k}_acc'] = 0.0
        
        return accuracies
    
    @staticmethod
    def benchmark_prediction_methods(predictions, targets, n_cats, methods=['argmax', 'cumulative', 'expected']):
        """
        Comprehensive benchmarking of different prediction methods.
        
        This function implements Phase 1 benchmarking as requested.
        Returns detailed comparison of argmax vs cumulative vs expected value predictions.
        """
        results = {}
        
        for method in methods:
            method_results = {
                'method': method,
                'categorical_accuracy': GpcmMetrics.categorical_accuracy(predictions, targets, method=method),
                'ordinal_accuracy': GpcmMetrics.ordinal_accuracy(predictions, targets, method=method),
                'mean_absolute_error': GpcmMetrics.mean_absolute_error(predictions, targets, method=method),
                'quadratic_weighted_kappa': GpcmMetrics.quadratic_weighted_kappa(predictions, targets, n_cats, method=method),
                'prediction_consistency': GpcmMetrics.prediction_consistency_accuracy(predictions, targets, method=method),
                'ordinal_ranking': GpcmMetrics.ordinal_ranking_accuracy(predictions, targets),
                'distribution_consistency': GpcmMetrics.distribution_consistency_score(predictions, targets),
            }
            
            # Add per-category breakdown
            per_cat = GpcmMetrics.per_category_accuracy(predictions, targets, n_cats, method=method)
            method_results.update(per_cat)
            
            results[method] = method_results
        
        # Calculate improvement metrics
        if 'argmax' in results and 'cumulative' in results:
            baseline = results['argmax']
            improved = results['cumulative']
            
            results['improvement_analysis'] = {
                'categorical_accuracy_improvement': improved['categorical_accuracy'] - baseline['categorical_accuracy'],
                'categorical_accuracy_improvement_pct': ((improved['categorical_accuracy'] - baseline['categorical_accuracy']) / baseline['categorical_accuracy']) * 100,
                'prediction_consistency_improvement': improved['prediction_consistency'] - baseline['prediction_consistency'],
                'prediction_consistency_improvement_pct': ((improved['prediction_consistency'] - baseline['prediction_consistency']) / baseline['prediction_consistency']) * 100,
                'ordinal_accuracy_improvement': improved['ordinal_accuracy'] - baseline['ordinal_accuracy'],
                'mae_improvement': baseline['mean_absolute_error'] - improved['mean_absolute_error'],  # Lower is better
            }
        
        return results
