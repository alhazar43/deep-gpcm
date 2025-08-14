"""
Enhanced Adaptive Hyperparameter Optimization

Integrates the adaptive scheduler with the existing Bayesian optimization system.
Provides seamless fallback to original system if adaptive features fail.
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original components
from optimization.adaptive_hyperopt import (
    HyperoptConfig, HyperparameterOptimizer, BayesianOptimizer, 
    AdaptiveSearchSpace, LossWeightOptimizer, TrialResult
)

# Import new adaptive components
from optimization.adaptive_scheduler import (
    AdaptiveConfig, AdaptiveHyperoptIntegration, create_adaptive_integration
)


class EnhancedHyperparameterOptimizer(HyperparameterOptimizer):
    """Enhanced hyperparameter optimizer with adaptive epoch scheduling and expanded search space."""
    
    def __init__(self, model_name: str, dataset: str, config: HyperoptConfig, 
                 adaptive_config: Optional[AdaptiveConfig] = None):
        
        # Initialize base optimizer
        super().__init__(model_name, dataset, config)
        
        # Initialize adaptive components
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.adaptive_integration = create_adaptive_integration(
            model_name, dataset, self.adaptive_config
        )
        
        # Enhanced search space
        if (self.adaptive_config.enable_architectural_params or 
            self.adaptive_config.enable_learning_params):
            self._enhance_search_space()
        
        # Performance tracking
        self.adaptive_enabled = True
        self.fallback_reason = None
        
        self.logger.info(f"Enhanced optimizer initialized with adaptive features: "
                        f"epochs={self.adaptive_config.enable_adaptive_epochs}, "
                        f"arch={self.adaptive_config.enable_architectural_params}, "
                        f"learning={self.adaptive_config.enable_learning_params}")
    
    def _enhance_search_space(self):
        """Enhance search space with adaptive parameters."""
        try:
            # Get original search space
            original_space = self.search_space.base_space.copy()
            
            # Create enhanced search space
            enhanced_space = self.adaptive_integration.create_enhanced_search_space(original_space)
            
            # Update search space
            self.search_space = AdaptiveSearchSpace(enhanced_space)
            
            added_params = set(enhanced_space.keys()) - set(original_space.keys())
            if added_params:
                self.logger.info(f"Enhanced search space with {len(added_params)} new parameters: {added_params}")
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance search space: {e}. Using original space.")
            self.fallback_reason = f"Search space enhancement failed: {e}"
    
    def _evaluate_params(self, params: Dict[str, Any]) -> TrialResult:
        """Evaluate parameters with adaptive epoch scheduling."""
        import time
        start_time = time.time()
        
        trial_id = len(self.trials) + 1
        
        # Get adaptive epoch suggestion
        if self.adaptive_config.enable_adaptive_epochs and self.adaptive_enabled:
            try:
                suggested_epochs, reasoning = self.adaptive_integration.suggest_trial_epochs(
                    trial_id, params, [t.__dict__ for t in self.trials]
                )
                self.logger.info(f"Trial {trial_id}: Using {suggested_epochs} epochs - {reasoning}")
            except Exception as e:
                self.logger.warning(f"Adaptive epoch suggestion failed: {e}. Using default.")
                suggested_epochs = self.config.cv_folds * 10  # Fallback
                reasoning = f"Fallback due to error: {e}"
        else:
            suggested_epochs = self.config.cv_folds * 10  # Original behavior
            reasoning = "Adaptive epochs disabled"
        
        # Convert loss weight parameters back to loss config
        loss_override = LossWeightOptimizer.params_to_loss_config(params)
        
        # Filter out loss weight logits and adaptive-specific params from model params
        model_params = {k: v for k, v in params.items() 
                       if not k.endswith('_logit') and k not in ['epochs']}
        
        self.logger.info(f"Trial {trial_id}: Evaluating {model_params}")
        if loss_override:
            self.logger.info(f"Loss weights: ce={loss_override['ce_weight']:.3f}, "
                           f"focal={loss_override['focal_weight']:.3f}, "
                           f"qwk={loss_override['qwk_weight']:.3f}")
        
        try:
            # Perform cross-validation with adaptive epochs
            cv_scores = self._run_adaptive_cross_validation(model_params, loss_override, suggested_epochs)
            
            metric_value = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            training_time = time.time() - start_time
            
            result = TrialResult(
                params=params,
                metric_value=metric_value,
                cv_scores=cv_scores,
                cv_std=cv_std,
                trial_id=trial_id,
                early_stopped=False,
                training_time=training_time
            )
            
            # Update adaptive components
            self.adaptive_integration.update_trial_result({
                'trial_id': trial_id,
                'params': params,
                'score': metric_value,
                'cv_std': cv_std,
                'early_stopped': False,
                'training_time': training_time,
                'epochs_used': suggested_epochs
            })
            
            self.logger.info(f"Trial {trial_id} completed: {self.config.metric}={metric_value:.4f} ±{cv_std:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Trial {trial_id} failed: {e}")
            
            # Check if we should fallback
            if self.adaptive_integration.should_fallback():
                self.logger.warning("Adaptive features failing, considering fallback")
                self.adaptive_enabled = False
                self.fallback_reason = f"Consecutive failures: {e}"
            
            # Return worst possible score
            worst_score = -1.0 if self.config.maximize else 1.0
            result = TrialResult(
                params=params,
                metric_value=worst_score,
                cv_scores=[worst_score] * self.config.cv_folds,
                cv_std=0.0,
                trial_id=trial_id,
                early_stopped=True,
                training_time=time.time() - start_time
            )
            
            # Update adaptive components with failure
            self.adaptive_integration.update_trial_result({
                'trial_id': trial_id,
                'params': params,
                'score': worst_score,
                'cv_std': 0.0,
                'early_stopped': True,
                'training_time': time.time() - start_time,
                'error': str(e)
            })
            
            return result
    
    def _run_adaptive_cross_validation(self, model_params: Dict[str, Any], 
                                     loss_override: Dict[str, Any], epochs: int) -> List[float]:
        """Run cross-validation with adaptive epoch count."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from sklearn.model_selection import KFold
        from train import load_simple_data, create_data_loaders, create_model, train_single_fold
        import torch
        
        try:
            # Load dataset (same as original)
            if self.dataset.startswith('synthetic_') and '_' in self.dataset[10:]:
                train_path = f"data/{self.dataset}/{self.dataset}_train.txt"
                test_path = f"data/{self.dataset}/{self.dataset}_test.txt"
            else:
                train_path = f"data/{self.dataset}/{self.dataset.lower()}_train.txt"
                test_path = f"data/{self.dataset}/{self.dataset.lower()}_test.txt"
            
            train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
            all_data = train_data + test_data
            
            # Set up cross-validation
            kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            cv_scores = []
            
            # Device setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_data), 1):
                try:
                    # Split data for this fold
                    fold_train_data = [all_data[i] for i in train_idx]
                    fold_val_data = [all_data[i] for i in val_idx]
                    
                    # Create data loaders with current batch size
                    batch_size = int(model_params.get('batch_size', 64))
                    train_loader, val_loader = create_data_loaders(fold_train_data, fold_val_data, batch_size)
                    
                    # Create model with current hyperparameters
                    filtered_model_params = {k: v for k, v in model_params.items() 
                                           if k not in ['lr', 'batch_size', 'weight_decay', 'grad_clip', 'label_smoothing']}
                    model = create_model(self.model_name, n_questions, n_cats, device, **filtered_model_params)
                    
                    # Prepare loss kwargs with enhanced parameters
                    loss_kwargs = {
                        'lr': model_params.get('lr', 0.001),
                        'weight_decay': model_params.get('weight_decay', 0.0)
                    }
                    if loss_override:
                        loss_kwargs.update(loss_override)
                    
                    # Add additional training parameters if available
                    if 'grad_clip' in model_params:
                        loss_kwargs['grad_clip'] = model_params['grad_clip']
                    if 'label_smoothing' in model_params:
                        loss_kwargs['label_smoothing'] = model_params['label_smoothing']
                    
                    # Train with adaptive epochs
                    _, metrics, _ = train_single_fold(
                        model, train_loader, val_loader, device, epochs,
                        self.model_name, fold=None, loss_type='combined',
                        loss_kwargs=loss_kwargs
                    )
                    
                    # Extract the metric we're optimizing for
                    score = metrics.get(self.config.metric, 0.0)
                    cv_scores.append(score)
                    
                    # Log fold progress with epoch info
                    self.logger.debug(f"Fold {fold_idx}: {self.config.metric}={score:.4f} ({epochs} epochs)")
                    
                except Exception as e:
                    self.logger.warning(f"Fold {fold_idx} failed: {e}")
                    # Use a poor score for failed folds
                    cv_scores.append(0.1 if self.config.maximize else 1.0)
            
            # Ensure we have scores for all folds
            while len(cv_scores) < self.config.cv_folds:
                cv_scores.append(0.1 if self.config.maximize else 1.0)
            
            return cv_scores
            
        except Exception as e:
            self.logger.error(f"Adaptive CV evaluation failed completely: {e}")
            # Return worst possible scores for all folds
            worst_score = 0.1 if self.config.maximize else 1.0
            return [worst_score] * self.config.cv_folds
    
    def optimize(self) -> Dict[str, Any]:
        """Run enhanced Bayesian hyperparameter optimization."""
        self.logger.info(f"Starting enhanced Bayesian optimization for {self.model_name}")
        self.logger.info(f"Adaptive features: epochs={self.adaptive_config.enable_adaptive_epochs}, "
                        f"arch={self.adaptive_config.enable_architectural_params}, "
                        f"learning={self.adaptive_config.enable_learning_params}")
        
        # Run original optimization logic
        results = super().optimize()
        
        # Add adaptive-specific results
        results['adaptive_features'] = {
            'adaptive_epochs_enabled': self.adaptive_config.enable_adaptive_epochs,
            'architectural_params_enabled': self.adaptive_config.enable_architectural_params,
            'learning_params_enabled': self.adaptive_config.enable_learning_params,
            'adaptive_enabled': self.adaptive_enabled,
            'fallback_triggered': not self.adaptive_enabled,
            'fallback_reason': self.fallback_reason,
            'performance_comparison': self.adaptive_integration.get_performance_comparison()
        }
        
        # Save adaptive optimization report
        try:
            report_path = self.adaptive_integration.save_performance_report()
            results['adaptive_report_path'] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save adaptive report: {e}")
        
        return results


def run_enhanced_hyperopt(model_name: str, dataset: str, n_trials: int = 50, 
                         metric: str = 'quadratic_weighted_kappa', cv_folds: int = 5,
                         adaptive_config: Optional[AdaptiveConfig] = None) -> Dict[str, Any]:
    """
    Run enhanced hyperparameter optimization with adaptive features.
    
    Args:
        model_name: Name of the model to optimize
        dataset: Dataset name
        n_trials: Number of optimization trials
        metric: Metric to optimize
        cv_folds: Number of CV folds
        adaptive_config: Configuration for adaptive features
    
    Returns:
        Optimization results with adaptive features
    """
    # Standard hyperopt config
    config = HyperoptConfig(
        n_trials=n_trials,
        metric=metric,
        cv_folds=cv_folds,
        maximize=True,  # Assuming QWK and accuracy metrics
        random_state=42
    )
    
    # Default adaptive config if not provided
    if adaptive_config is None:
        adaptive_config = AdaptiveConfig(
            enable_adaptive_epochs=True,
            enable_architectural_params=True,
            enable_learning_params=True,
            min_epochs=5,
            intermediate_epochs=15,
            max_epochs=40
        )
    
    # Create and run enhanced optimizer
    optimizer = EnhancedHyperparameterOptimizer(model_name, dataset, config, adaptive_config)
    results = optimizer.optimize()
    
    return results


if __name__ == "__main__":
    # Example usage with enhanced features
    adaptive_config = AdaptiveConfig(
        enable_adaptive_epochs=True,
        enable_architectural_params=True,
        enable_learning_params=True,
        min_epochs=3,
        intermediate_epochs=10,
        max_epochs=30
    )
    
    results = run_enhanced_hyperopt(
        model_name='deep_gpcm',
        dataset='synthetic_500_200_4', 
        n_trials=20,
        metric='quadratic_weighted_kappa',
        cv_folds=3,
        adaptive_config=adaptive_config
    )
    
    print("\\nEnhanced Hyperparameter Optimization Results:")
    print(f"Best {results['metric']}: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Adaptive features: {results['adaptive_features']}")
    if 'best_loss_weights' in results:
        print(f"Best loss weights: {results['best_loss_weights']}")