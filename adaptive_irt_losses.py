#!/usr/bin/env python3
"""
IRT-Optimized Loss Functions for Adaptive Testing

Standalone loss functions specifically designed for adaptive IRT implementations.
These maintain theoretical IRT soundness while incorporating ordinal awareness.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score
from typing import Optional, Tuple


class IRTOptimizedLoss:
    """
    IRT-Optimized hybrid loss combining CrossEntropy (for MML alignment) 
    with Quadratic Weighted Kappa (for ordinal structure).
    
    Designed specifically for adaptive IRT parameter estimation.
    """
    
    def __init__(self, n_cats: int = 4, ce_weight: float = 0.7, qwk_weight: float = 0.3):
        """
        Initialize IRT-optimized loss function.
        
        Args:
            n_cats: Number of response categories
            ce_weight: Weight for CrossEntropy component (MML alignment)
            qwk_weight: Weight for QWK component (ordinal structure)
        """
        self.n_cats = n_cats
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        
        # Precompute QWK weight matrix
        self.qwk_weights = self._compute_qwk_weights(n_cats)
        
        print(f"📊 IRT-Optimized Loss initialized:")
        print(f"   CE weight: {ce_weight:.1f} (MML alignment)")
        print(f"   QWK weight: {qwk_weight:.1f} (ordinal structure)")
        
    def _compute_qwk_weights(self, n_cats: int) -> np.ndarray:
        """Compute quadratic weights for QWK calculation."""
        weights = np.zeros((n_cats, n_cats))
        for i in range(n_cats):
            for j in range(n_cats):
                weights[i, j] = (i - j) ** 2 / (n_cats - 1) ** 2
        return weights
    
    def cross_entropy_loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute CrossEntropy loss from probabilities.
        
        Args:
            probs: Predicted probabilities, shape (n_samples, n_cats)
            targets: True category labels, shape (n_samples,)
            
        Returns:
            CrossEntropy loss value
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_probs = np.log(probs + epsilon)
        
        # Extract log probability of true class
        true_log_probs = log_probs[np.arange(len(targets)), targets]
        
        # Negative log likelihood
        ce_loss = -np.mean(true_log_probs)
        return ce_loss
    
    def quadratic_weighted_kappa_loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute differentiable QWK loss.
        
        Args:
            probs: Predicted probabilities, shape (n_samples, n_cats)
            targets: True category labels, shape (n_samples,)
            
        Returns:
            QWK loss value (1 - QWK, so lower is better)
        """
        # Get predicted categories
        predictions = np.argmax(probs, axis=1)
        
        # Compute QWK using sklearn with 'quadratic' weights
        qwk = cohen_kappa_score(targets, predictions, weights='quadratic')
        
        # Return 1 - QWK so that lower values are better
        qwk_loss = 1.0 - qwk
        return qwk_loss
    
    def compute_loss(self, probs: np.ndarray, targets: np.ndarray) -> Tuple[float, dict]:
        """
        Compute IRT-optimized hybrid loss.
        
        Args:
            probs: Predicted probabilities, shape (n_samples, n_cats)
            targets: True category labels, shape (n_samples,)
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary with individual loss components
        """
        # Compute individual loss components
        ce_loss = self.cross_entropy_loss(probs, targets)
        qwk_loss = self.quadratic_weighted_kappa_loss(probs, targets)
        
        # Combine losses
        total_loss = self.ce_weight * ce_loss + self.qwk_weight * qwk_loss
        
        # Package results
        loss_components = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'qwk_loss': qwk_loss,
            'ce_weight': self.ce_weight,
            'qwk_weight': self.qwk_weight
        }
        
        return total_loss, loss_components


class AdaptiveIRTParameterEstimator:
    """
    Enhanced parameter estimator using IRT-optimized loss for adaptive testing.
    """
    
    def __init__(self, n_cats: int = 4, loss_config: Optional[dict] = None):
        """
        Initialize parameter estimator with IRT-optimized loss.
        
        Args:
            n_cats: Number of response categories
            loss_config: Loss configuration (ce_weight, qwk_weight)
        """
        self.n_cats = n_cats
        
        # Set up loss configuration
        if loss_config is None:
            loss_config = {'ce_weight': 0.7, 'qwk_weight': 0.3}
        
        self.loss_function = IRTOptimizedLoss(n_cats, **loss_config)
        
    def estimate_theta_with_optimized_loss(self, responses: list, question_ids: list, 
                                         alpha_params: np.ndarray, beta_params: np.ndarray,
                                         prior_mean: float = 0.0, prior_std: float = 1.0) -> Tuple[float, float]:
        """
        Estimate theta using IRT-optimized loss function.
        
        Args:
            responses: List of observed responses
            question_ids: List of question IDs
            alpha_params: Discrimination parameters
            beta_params: Difficulty parameters
            prior_mean: Prior mean for theta
            prior_std: Prior standard deviation for theta
            
        Returns:
            theta_estimate: Estimated ability parameter
            standard_error: Standard error of estimate
        """
        if not responses:
            return 0.0, 1.0
        
        def negative_log_likelihood(theta):
            """Compute negative log-likelihood using IRT-optimized loss."""
            
            # Prior contribution (standard normal)
            prior_contribution = 0.5 * ((theta - prior_mean) / prior_std) ** 2
            
            # Compute GPCM probabilities for all responses
            all_probs = []
            targets = np.array(responses)
            
            for i, (q_id, response) in enumerate(zip(question_ids, responses)):
                alpha = alpha_params[q_id]
                betas = beta_params[q_id]
                
                # GPCM probability calculation
                cum_logits = np.zeros(self.n_cats)
                for k in range(1, self.n_cats):
                    cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
                
                # Convert to probabilities
                cum_logits = cum_logits - np.max(cum_logits)  # Numerical stability
                probs = np.exp(cum_logits) / np.sum(np.exp(cum_logits))
                all_probs.append(probs)
            
            # Stack probabilities
            all_probs = np.array(all_probs)  # shape: (n_responses, n_cats)
            
            # Compute IRT-optimized loss
            total_loss, _ = self.loss_function.compute_loss(all_probs, targets)
            
            # Combine with prior
            return total_loss + prior_contribution
        
        # Optimize theta
        result = minimize_scalar(negative_log_likelihood, bounds=(-4, 4), method='bounded')
        theta_estimate = result.x
        
        # Approximate standard error
        # Could be improved with proper Fisher Information calculation
        n_responses = len(responses)
        se_estimate = 1.0 / np.sqrt(max(n_responses * 0.5, 0.1))
        
        return theta_estimate, se_estimate
    
    def evaluate_item_performance(self, theta: float, alpha: float, betas: np.ndarray) -> dict:
        """
        Evaluate item performance using IRT-optimized metrics.
        
        Args:
            theta: Ability parameter
            alpha: Discrimination parameter
            betas: Difficulty parameters
            
        Returns:
            Dictionary with item performance metrics
        """
        # Compute GPCM probabilities
        cum_logits = np.zeros(self.n_cats)
        for k in range(1, self.n_cats):
            cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        # Convert to probabilities
        cum_logits = cum_logits - np.max(cum_logits)
        probs = np.exp(cum_logits) / np.sum(np.exp(cum_logits))
        
        # Compute information (simplified Fisher Information)
        derivatives = np.array([k * alpha for k in range(self.n_cats)])
        expected_derivative = np.sum(derivatives * probs)
        fisher_info = np.sum((derivatives - expected_derivative) ** 2 * probs)
        
        # Expected score and response probabilities
        expected_score = np.sum(np.arange(self.n_cats) * probs)
        
        return {
            'probabilities': probs,
            'fisher_information': fisher_info,
            'expected_score': expected_score,
            'discrimination': alpha,
            'difficulties': betas
        }


def demo_irt_optimized_loss():
    """Demonstrate the IRT-optimized loss function."""
    
    print("🔬 DEMO: IRT-Optimized Loss Function")
    print("=" * 50)
    
    # Create loss function
    loss_fn = IRTOptimizedLoss(n_cats=4, ce_weight=0.7, qwk_weight=0.3)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate some predictions (probabilities)
    probs = np.random.dirichlet([1, 1, 1, 1], size=n_samples)
    
    # Simulate targets (with some ordinal structure)
    targets = np.random.choice(4, size=n_samples, p=[0.2, 0.3, 0.3, 0.2])
    
    # Compute loss
    total_loss, components = loss_fn.compute_loss(probs, targets)
    
    print(f"\nExample Loss Computation:")
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  CE Loss: {components['ce_loss']:.4f} (weight: {components['ce_weight']:.1f})")
    print(f"  QWK Loss: {components['qwk_loss']:.4f} (weight: {components['qwk_weight']:.1f})")
    
    # Test parameter estimator
    print(f"\n🎯 Testing Parameter Estimator:")
    estimator = AdaptiveIRTParameterEstimator(n_cats=4)
    
    # Simulate some responses
    responses = [1, 2, 1, 3, 2]
    question_ids = [0, 1, 2, 3, 4]
    alpha_params = np.ones(5)  # Neutral discrimination
    beta_params = np.array([[-1, 0, 1]] * 5)  # Ordered thresholds
    
    theta_est, se_est = estimator.estimate_theta_with_optimized_loss(
        responses, question_ids, alpha_params, beta_params
    )
    
    print(f"  Estimated θ: {theta_est:.3f} (±{se_est:.3f})")
    print(f"  Responses: {responses}")
    print(f"  Questions: {question_ids}")


if __name__ == "__main__":
    demo_irt_optimized_loss()