#!/usr/bin/env python3
"""
Static GPCM Probability Computation Script

Computes GPCM probabilities using static IRT parameters from synthetic dataset
and generates confusion matrix comparing predicted vs actual responses.

This script implements the exact same GPCM probability computation as data_gen.py
but uses static parameters to predict responses and compare against actual data.

Usage:
    # Test with legacy format (synthetic_OC)
    python static_gpcm_test.py --data_dir ./data/synthetic_OC
    
    # Test with new format datasets
    python static_gpcm_test.py --data_dir ./data/synthetic_4000_200_3
    python static_gpcm_test.py --data_dir ./data/synthetic_4000_200_2
    python static_gpcm_test.py --data_dir ./data/synthetic_4000_200_5
    
    # Custom output path
    python static_gpcm_test.py --data_dir ./data/synthetic_4000_200_3 --output ./custom_confusion.png

Output Files:
    - static_predict.png: Comprehensive metrics plot (similar to test.png format)
    - static_gpcm_confusion_matrix.png: Confusion matrix (counts and normalized)

Results Summary:
    Categories | Exact Acc | Ordinal Acc | QWK   | MAE   | Kendall | Spearman
    2          | 61.1%     | 100%        | 0.219 | 0.389 | 0.219   | 0.219
    3          | 47.1%     | 80.1%       | 0.281 | 0.727 | 0.255   | 0.282  
    4          | 38.4%     | 67.0%       | 0.269 | 1.087 | 0.233   | 0.271
    5          | 36.7%     | 61.6%       | 0.362 | 1.336 | 0.308   | 0.367

Performance Pattern:
    - Exact accuracy decreases as categories increase (61% → 37%)
    - Ordinal accuracy remains reasonable (62-100%) 
    - QWK shows good ordinal prediction quality (0.22-0.36)
    - MAE increases proportionally with category count
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import OrdinalMetrics
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

class StaticGPCMTester:
    def __init__(self, data_dir="./data/synthetic_OC"):
        self.data_dir = Path(data_dir)
        self.load_irt_parameters()
        self.load_data()
    
    def load_irt_parameters(self):
        """Load static IRT parameters from true_irt_parameters.json"""
        params_file = self.data_dir / "true_irt_parameters.json"
        
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        self.theta = np.array(params['student_abilities']['theta'])
        self.alpha = np.array(params['question_params']['discrimination']['alpha'])
        self.beta = np.array(params['question_params']['difficulties']['beta'])
        self.n_cats = params['model_info']['n_cats']
        
        print(f"Loaded IRT parameters:")
        print(f"  Students: {len(self.theta)}")
        print(f"  Questions: {len(self.alpha)}")
        print(f"  Categories: {self.n_cats}")
        print(f"  Beta shape: {self.beta.shape}")
    
    def load_data(self):
        """Load synthetic data for testing"""
        # Auto-detect format based on available files
        oc_file = self.data_dir / "synthetic_oc_test.txt"
        
        # Check for new format first
        dataset_name = self.data_dir.name
        new_format_file = self.data_dir / f"{dataset_name}_test.txt"
        
        if new_format_file.exists():
            test_file = new_format_file
            print(f"Using new format: {test_file}")
        elif oc_file.exists():
            test_file = oc_file
            print(f"Using legacy format: {test_file}")
        else:
            raise FileNotFoundError(f"No test file found in {self.data_dir}")
        
        self.sequences = []
        with open(test_file, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            seq_len = int(lines[i].strip())
            questions = list(map(int, lines[i+1].strip().split(',')))
            responses = list(map(int, lines[i+2].strip().split(',')))
            
            self.sequences.append({
                'seq_len': seq_len,
                'questions': questions,
                'responses': responses
            })
            
            i += 3
        
        print(f"Loaded {len(self.sequences)} test sequences")
    
    def gpcm_prob(self, theta, alpha, betas):
        """
        Compute GPCM probabilities using static parameters
        Same formula as in data_gen.py
        """
        K = len(betas) + 1  # Number of categories
        cum_logits = np.zeros(K)
        cum_logits[0] = 0
        
        # Compute cumulative logits
        for k in range(1, K):
            cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        # Apply softmax
        exp_logits = np.exp(cum_logits - np.max(cum_logits))
        return exp_logits / np.sum(exp_logits)
    
    def predict_responses(self):
        """Predict responses for all student-question pairs"""
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        
        for student_id, seq in enumerate(self.sequences):
            theta = self.theta[student_id]
            
            for q_id, actual_response in zip(seq['questions'], seq['responses']):
                alpha = self.alpha[q_id]
                betas = self.beta[q_id]
                
                # Compute GPCM probabilities
                probs = self.gpcm_prob(theta, alpha, betas)
                
                # Predict using argmax (softmax approach)
                predicted_response = np.argmax(probs)
                
                all_predictions.append(predicted_response)
                all_actuals.append(actual_response)
                all_probabilities.append(probs)
        
        return np.array(all_predictions), np.array(all_actuals), np.array(all_probabilities)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_cats)))
        
        # Compute normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=range(self.n_cats), yticklabels=range(self.n_cats))
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Response')
        ax1.set_ylabel('Actual Response')
        
        # Plot normalized percentages
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax2,
                   xticklabels=range(self.n_cats), yticklabels=range(self.n_cats))
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Response')
        ax2.set_ylabel('Actual Response')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm, cm_norm
    
    def plot_comprehensive_metrics(self, metrics, save_path=None):
        """Plot comprehensive metrics similar to test.png format"""
        # Select key metrics for plotting (similar to plot_metrics.py)
        key_metrics = [
            'categorical_accuracy', 'ordinal_accuracy', 'quadratic_weighted_kappa',
            'mean_absolute_error', 'kendall_tau', 'spearman_correlation'
        ]
        
        # Filter available metrics
        available_metrics = [m for m in key_metrics if m in metrics]
        
        if not available_metrics:
            print("No key metrics available for plotting")
            return
        
        # Calculate subplot layout
        n_metrics = len(available_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Define metric properties
        higher_is_better = {
            'categorical_accuracy': True,
            'ordinal_accuracy': True,
            'quadratic_weighted_kappa': True,
            'mean_absolute_error': False,
            'kendall_tau': True,
            'spearman_correlation': True,
            'pearson_correlation': True,
            'cohen_kappa': True,
            'cross_entropy': False,
            'mean_confidence': True
        }
        
        # Color for single model (static prediction)
        bar_color = '#2E86AB'
        
        # Plot each metric
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            value = metrics[metric]
            
            # Create single bar
            bar = ax.bar([0], [value], color=bar_color, alpha=0.7, width=0.6)
            
            # Highlight with green edge (best performer style)
            bar[0].set_edgecolor('green')
            bar[0].set_linewidth(3)
            bar[0].set_alpha(1.0)
            
            # Add value annotation
            ax.text(0, value + value * 0.05, f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Formatting
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add directional arrow based on metric
            is_higher_better = higher_is_better.get(metric, True)
            arrow = '↑' if is_higher_better else '↓'
            current_title = ax.get_title()
            ax.set_title(f"{current_title} {arrow}")
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Static GPCM Prediction Metrics', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """Compute comprehensive metrics using OrdinalMetrics"""
        # Initialize metrics calculator
        metrics_calc = OrdinalMetrics(n_cats=self.n_cats)
        
        # Compute all metrics
        all_metrics = metrics_calc.compute_all_metrics(y_true, y_pred, y_prob)
        
        # Add alias for exact accuracy
        all_metrics['exact_accuracy'] = all_metrics['categorical_accuracy']
        
        return all_metrics
    
    def run_test(self, save_path=None):
        """Run complete static GPCM test"""
        print("Computing GPCM probabilities using static parameters...")
        
        # Predict responses
        y_pred, y_true, y_prob = self.predict_responses()
        
        print(f"Total predictions: {len(y_pred)}")
        print(f"Response distribution (actual): {np.bincount(y_true)}")
        print(f"Response distribution (predicted): {np.bincount(y_pred)}")
        
        # Compute comprehensive metrics
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        print("\nKey Metrics:")
        key_metrics_display = [
            ('exact_accuracy', 'Exact Accuracy'),
            ('ordinal_accuracy', 'Ordinal Accuracy (±1)'),
            ('quadratic_weighted_kappa', 'Quadratic Weighted Kappa'),
            ('mean_absolute_error', 'Mean Absolute Error'),
            ('kendall_tau', 'Kendall Tau'),
            ('spearman_correlation', 'Spearman Correlation')
        ]
        
        for metric_key, metric_name in key_metrics_display:
            if metric_key in metrics:
                print(f"  {metric_name}: {metrics[metric_key]:.3f}")
        
        # Define save paths
        if save_path is None:
            confusion_path = self.data_dir / "static_gpcm_confusion_matrix.png"
            metrics_path = self.data_dir / "static_predict.png"
        else:
            save_path = Path(save_path)
            confusion_path = save_path.parent / "static_gpcm_confusion_matrix.png"
            metrics_path = save_path
        
        # Plot confusion matrix
        cm, cm_norm = self.plot_confusion_matrix(y_true, y_pred, confusion_path)
        
        # Plot comprehensive metrics
        self.plot_comprehensive_metrics(metrics, metrics_path)
        
        return {
            'predictions': y_pred,
            'actuals': y_true,
            'probabilities': y_prob,
            'metrics': metrics,
            'confusion_matrix': cm,
            'confusion_matrix_norm': cm_norm
        }


class DynamicGPCMTester:
    """Dynamic GPCM implementation with MML optimization for adaptive testing."""
    
    def __init__(self, data_dir="./data/synthetic_OC", device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = device
        self.load_data()
        self.load_true_parameters()  # For comparison
        self.initialize_parameters()
    
    def load_data(self):
        """Load both training and test data"""
        # Auto-detect format
        dataset_name = self.data_dir.name
        
        # Load training data
        train_file_new = self.data_dir / f"{dataset_name}_train.txt"
        train_file_legacy = self.data_dir / "synthetic_oc_train.txt"
        
        if train_file_new.exists():
            train_file = train_file_new
            test_file = self.data_dir / f"{dataset_name}_test.txt"
        elif train_file_legacy.exists():
            train_file = train_file_legacy
            test_file = self.data_dir / "synthetic_oc_test.txt"
        else:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        print(f"Loading training data from: {train_file}")
        print(f"Loading test data from: {test_file}")
        
        self.train_sequences = self._load_sequences(train_file)
        self.test_sequences = self._load_sequences(test_file)
        
        # Extract metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.n_cats = metadata['n_cats']
        self.n_questions = metadata['n_questions']
        self.n_students_train = len(self.train_sequences)
        self.n_students_test = len(self.test_sequences)
        
        print(f"Loaded data: {self.n_students_train} train + {self.n_students_test} test students")
        print(f"Questions: {self.n_questions}, Categories: {self.n_cats}")
    
    def _load_sequences(self, file_path):
        """Load sequences from file"""
        sequences = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            seq_len = int(lines[i].strip())
            questions = list(map(int, lines[i+1].strip().split(',')))
            responses = list(map(int, lines[i+2].strip().split(',')))
            
            sequences.append({
                'seq_len': seq_len,
                'questions': questions,
                'responses': responses
            })
            i += 3
        
        return sequences
    
    def load_true_parameters(self):
        """Load true parameters for comparison"""
        params_file = self.data_dir / "true_irt_parameters.json"
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Extract true parameters (for comparison only)
        self.true_theta = np.array(params['student_abilities']['theta'])
        self.true_alpha = np.array(params['question_params']['discrimination']['alpha'])
        self.true_beta = np.array(params['question_params']['difficulties']['beta'])
    
    def initialize_parameters(self):
        """Initialize learnable parameters"""
        # Initialize student abilities (theta) - one per training student
        self.theta = nn.Parameter(torch.randn(self.n_students_train, device=self.device) * 0.5)
        
        # Initialize discrimination parameters (alpha) - one per question
        self.alpha = nn.Parameter(torch.ones(self.n_questions, device=self.device) + 
                                 torch.randn(self.n_questions, device=self.device) * 0.2)
        
        # Initialize difficulty parameters (beta) - (n_questions, n_cats-1)
        self.beta = nn.Parameter(torch.randn(self.n_questions, self.n_cats-1, device=self.device) * 0.5)
        
        # Ensure beta parameters are ordered (ascending for each question)
        with torch.no_grad():
            self.beta.data = torch.sort(self.beta.data, dim=1)[0]
        
        print(f"Initialized parameters:")
        print(f"  Theta: {self.theta.shape} (student abilities)")
        print(f"  Alpha: {self.alpha.shape} (discrimination)")
        print(f"  Beta: {self.beta.shape} (difficulties)")
    
    def gpcm_prob_torch(self, theta, alpha, betas):
        """
        Compute GPCM probabilities using PyTorch for automatic differentiation
        
        Args:
            theta: Student ability (scalar or tensor)
            alpha: Question discrimination (scalar)
            betas: Question difficulties (tensor of length K-1)
        
        Returns:
            Probability distribution over K categories
        """
        K = len(betas) + 1
        
        # Compute cumulative logits
        cum_logits = torch.zeros(K, device=self.device)
        cum_logits[0] = 0.0  # Reference category
        
        for k in range(1, K):
            # Sum alpha * (theta - beta_h) for h from 0 to k-1
            cum_logits[k] = torch.sum(alpha * (theta - betas[:k]))
        
        # Apply softmax
        probs = torch.softmax(cum_logits, dim=0)
        return probs
    
    def compute_likelihood(self, train_data=True):
        """Compute negative log-likelihood for current parameters"""
        sequences = self.train_sequences if train_data else self.test_sequences
        total_nll = 0.0
        n_observations = 0
        
        for student_id, seq in enumerate(sequences):
            if train_data:
                theta = self.theta[student_id]
            else:
                # For test data, use mean theta (or could optimize per test student)
                theta = torch.mean(self.theta)
            
            for q_id, response in zip(seq['questions'], seq['responses']):
                alpha = self.alpha[q_id]
                betas = self.beta[q_id]
                
                # Compute GPCM probabilities
                probs = self.gpcm_prob_torch(theta, alpha, betas)
                
                # Add negative log-likelihood of observed response
                total_nll -= torch.log(probs[response] + 1e-8)  # Add small epsilon for numerical stability
                n_observations += 1
        
        return total_nll / n_observations  # Average NLL
    
    def train_model(self, max_epochs=500, lr=0.01, patience=50, min_delta=1e-4):
        """Train the GPCM model using MML optimization"""
        print(f"Training dynamic GPCM model...")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Patience: {patience}")
        
        # Set up optimizer
        optimizer = optim.Adam([self.theta, self.alpha, self.beta], lr=lr)
        
        # Training history
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            # Ensure beta parameters remain ordered
            with torch.no_grad():
                self.beta.data = torch.sort(self.beta.data, dim=1)[0]
            
            # Compute loss
            loss = self.compute_likelihood(train_data=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.theta, self.alpha, self.beta], max_norm=1.0)
            
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Check for improvement
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: Loss = {loss.item():.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        print(f"Training completed. Final loss: {train_losses[-1]:.4f}")
        return train_losses
    
    def predict_responses(self, use_test_data=True):
        """Generate predictions using learned parameters"""
        sequences = self.test_sequences if use_test_data else self.train_sequences
        
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        
        with torch.no_grad():
            for student_id, seq in enumerate(sequences):
                if not use_test_data:
                    theta = self.theta[student_id]
                else:
                    # For test data, use mean theta (simple approach)
                    # In practice, could optimize theta for each test student
                    theta = torch.mean(self.theta)
                
                for q_id, actual_response in zip(seq['questions'], seq['responses']):
                    alpha = self.alpha[q_id]
                    betas = self.beta[q_id]
                    
                    # Compute GPCM probabilities
                    probs = self.gpcm_prob_torch(theta, alpha, betas)
                    
                    # Convert to numpy for metrics computation
                    probs_np = probs.cpu().numpy()
                    predicted_response = np.argmax(probs_np)
                    
                    all_predictions.append(predicted_response)
                    all_actuals.append(actual_response)
                    all_probabilities.append(probs_np)
        
        return np.array(all_predictions), np.array(all_actuals), np.array(all_probabilities)
    
    def analyze_parameter_recovery(self):
        """Analyze how well learned parameters match true parameters"""
        with torch.no_grad():
            learned_alpha = self.alpha.cpu().numpy()
            learned_beta = self.beta.cpu().numpy()
            learned_theta = self.theta.cpu().numpy()
            
            # Correlations with true parameters
            alpha_corr = np.corrcoef(self.true_alpha, learned_alpha)[0, 1]
            
            # For beta, compute correlation for each threshold level
            beta_corrs = []
            for k in range(self.n_cats - 1):
                beta_corr = np.corrcoef(self.true_beta[:, k], learned_beta[:, k])[0, 1]
                beta_corrs.append(beta_corr)
            
            # For theta, use subset that exists in true parameters
            n_compare = min(len(self.true_theta), len(learned_theta))
            theta_corr = np.corrcoef(self.true_theta[:n_compare], learned_theta[:n_compare])[0, 1]
            
            recovery_stats = {
                'alpha_correlation': alpha_corr,
                'beta_correlations': beta_corrs,
                'beta_correlation_mean': np.mean(beta_corrs),
                'theta_correlation': theta_corr,
                'learned_alpha_mean': np.mean(learned_alpha),
                'learned_alpha_std': np.std(learned_alpha),
                'learned_theta_mean': np.mean(learned_theta),
                'learned_theta_std': np.std(learned_theta)
            }
            
            return recovery_stats
    
    def run_dynamic_test(self, train_epochs=200, save_prefix=None):
        """Run complete dynamic GPCM test with training and evaluation"""
        print("=" * 60)
        print("DYNAMIC GPCM TEST WITH MML OPTIMIZATION")
        print("=" * 60)
        
        # Train the model
        train_losses = self.train_model(max_epochs=train_epochs)
        
        # Analyze parameter recovery
        recovery_stats = self.analyze_parameter_recovery()
        
        print(f"\nParameter Recovery Analysis:")
        print(f"  Alpha correlation: {recovery_stats['alpha_correlation']:.3f}")
        print(f"  Beta correlation (mean): {recovery_stats['beta_correlation_mean']:.3f}")
        print(f"  Theta correlation: {recovery_stats['theta_correlation']:.3f}")
        
        # Generate predictions on test data
        y_pred, y_true, y_prob = self.predict_responses(use_test_data=True)
        
        print(f"\nTest Data Predictions:")
        print(f"  Total predictions: {len(y_pred)}")
        print(f"  Response distribution (actual): {np.bincount(y_true)}")
        print(f"  Response distribution (predicted): {np.bincount(y_pred)}")
        
        # Compute comprehensive metrics
        metrics_calc = OrdinalMetrics(n_cats=self.n_cats)
        metrics = metrics_calc.compute_all_metrics(y_true, y_pred, y_prob)
        
        # Add recovery stats to metrics
        metrics.update(recovery_stats)
        
        print(f"\nDynamic GPCM Performance:")
        key_metrics_display = [
            ('categorical_accuracy', 'Exact Accuracy'),
            ('ordinal_accuracy', 'Ordinal Accuracy (±1)'),
            ('quadratic_weighted_kappa', 'Quadratic Weighted Kappa'),
            ('mean_absolute_error', 'Mean Absolute Error'),
            ('kendall_tau', 'Kendall Tau'),
            ('spearman_correlation', 'Spearman Correlation')
        ]
        
        for metric_key, metric_name in key_metrics_display:
            if metric_key in metrics:
                print(f"  {metric_name}: {metrics[metric_key]:.3f}")
        
        # Generate plots
        if save_prefix is None:
            confusion_path = self.data_dir / "dynamic_gpcm_confusion_matrix.png"
            metrics_path = self.data_dir / "dynamic_predict.png"
            training_path = self.data_dir / "dynamic_training.png"
        else:
            save_prefix = Path(save_prefix)
            confusion_path = save_prefix.parent / "dynamic_gpcm_confusion_matrix.png"
            metrics_path = save_prefix
            training_path = save_prefix.parent / "dynamic_training.png"
        
        # Plot training curve
        self.plot_training_curve(train_losses, training_path)
        
        # Plot confusion matrix (reuse StaticGPCMTester method)
        static_tester = StaticGPCMTester(self.data_dir)
        static_tester.n_cats = self.n_cats
        cm, cm_norm = static_tester.plot_confusion_matrix(y_true, y_pred, confusion_path)
        
        # Plot comprehensive metrics (reuse StaticGPCMTester method)
        static_tester.plot_comprehensive_metrics(metrics, metrics_path)
        
        print(f"\nDynamic GPCM test completed!")
        print(f"Plots saved:")
        print(f"  Training curve: {training_path}")
        print(f"  Confusion matrix: {confusion_path}")
        print(f"  Metrics plot: {metrics_path}")
        
        return {
            'predictions': y_pred,
            'actuals': y_true,
            'probabilities': y_prob,
            'metrics': metrics,
            'training_losses': train_losses,
            'recovery_stats': recovery_stats,
            'confusion_matrix': cm,
            'confusion_matrix_norm': cm_norm
        }
    
    def plot_training_curve(self, losses, save_path):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Dynamic GPCM Training Curve (MML Optimization)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Annotate final loss
        final_loss = losses[-1]
        plt.annotate(f'Final: {final_loss:.3f}', 
                    xy=(len(losses)-1, final_loss),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test GPCM probability computation (static or dynamic)')
    parser.add_argument('--data_dir', default='./data/synthetic_OC',
                       help='Directory containing synthetic data')
    parser.add_argument('--output', default=None,
                       help='Output path for plots')
    parser.add_argument('--mode', choices=['static', 'dynamic', 'both'], default='static',
                       help='Test mode: static (true params), dynamic (MML training), or both')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs for dynamic mode')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for dynamic mode')
    
    args = parser.parse_args()
    
    if args.mode in ['static', 'both']:
        print("="*60)
        print("STATIC GPCM TEST (TRUE PARAMETERS)")
        print("="*60)
        
        # Create static tester
        static_tester = StaticGPCMTester(data_dir=args.data_dir)
        
        # Run static test
        static_results = static_tester.run_test(save_path=args.output)
        
        print(f"\nStatic GPCM test completed!")
        print(f"Results summary:")
        print(f"  Exact accuracy: {static_results['metrics']['exact_accuracy']:.3f}")
        print(f"  Ordinal accuracy: {static_results['metrics']['ordinal_accuracy']:.3f}")
        print(f"  QWK: {static_results['metrics']['quadratic_weighted_kappa']:.3f}")
        print(f"  MAE: {static_results['metrics']['mean_absolute_error']:.3f}")
        
        if args.output is None:
            print(f"Static plots saved:")
            print(f"  Confusion matrix: {static_tester.data_dir}/static_gpcm_confusion_matrix.png")
            print(f"  Metrics plot: {static_tester.data_dir}/static_predict.png")
    
    if args.mode in ['dynamic', 'both']:
        if args.mode == 'both':
            print("\n" + "="*60)
        else:
            print("="*60)
        
        # Create dynamic tester
        dynamic_tester = DynamicGPCMTester(data_dir=args.data_dir)
        
        # Run dynamic test
        dynamic_results = dynamic_tester.run_dynamic_test(
            train_epochs=args.epochs, 
            save_prefix=args.output
        )
    
    if args.mode == 'both':
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"Static vs Dynamic Performance:")
        print(f"  Exact Accuracy:  Static {static_results['metrics']['exact_accuracy']:.3f} | Dynamic {dynamic_results['metrics']['categorical_accuracy']:.3f}")
        print(f"  Ordinal Accuracy: Static {static_results['metrics']['ordinal_accuracy']:.3f} | Dynamic {dynamic_results['metrics']['ordinal_accuracy']:.3f}")
        print(f"  QWK:             Static {static_results['metrics']['quadratic_weighted_kappa']:.3f} | Dynamic {dynamic_results['metrics']['quadratic_weighted_kappa']:.3f}")
        print(f"  MAE:             Static {static_results['metrics']['mean_absolute_error']:.3f} | Dynamic {dynamic_results['metrics']['mean_absolute_error']:.3f}")
        
        print(f"\nParameter Recovery (Dynamic):")
        print(f"  Alpha correlation: {dynamic_results['recovery_stats']['alpha_correlation']:.3f}")
        print(f"  Beta correlation:  {dynamic_results['recovery_stats']['beta_correlation_mean']:.3f}")
        print(f"  Theta correlation: {dynamic_results['recovery_stats']['theta_correlation']:.3f}")


if __name__ == "__main__":
    main()