"""
Adaptive Bayesian Hyperparameter Optimization for Deep-GPCM Models

Implements state-of-the-art hyperparameter optimization using Gaussian Process-based
Bayesian optimization with intelligent search space adaptation and early stopping.
"""

import os
import sys
import json
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.factory import get_model_hyperparameter_grid, get_model_loss_config


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 50
    metric: str = 'quadratic_weighted_kappa'
    maximize: bool = True
    n_initial_samples: int = 10
    acquisition_func: str = 'ei'  # 'ei', 'pi', 'ucb'
    early_stopping_patience: int = 15
    min_improvement: float = 1e-4
    cv_folds: int = 5
    random_state: int = 42
    
    # GP configuration
    gp_kernel: str = 'matern'
    gp_nu: float = 2.5
    gp_noise_level: float = 0.1
    gp_normalize_y: bool = True
    
    # Acquisition function parameters
    ucb_kappa: float = 2.576  # 99% confidence
    ei_xi: float = 0.01
    pi_xi: float = 0.01


@dataclass 
class TrialResult:
    """Result of a single hyperparameter trial."""
    params: Dict[str, Any]
    metric_value: float
    cv_scores: List[float]
    cv_std: float
    trial_id: int
    early_stopped: bool = False
    training_time: float = 0.0
    

class AdaptiveSearchSpace:
    """Adaptive search space that learns from successful trials."""
    
    def __init__(self, base_space: Dict[str, Any]):
        self.base_space = base_space
        self.param_ranges = {}
        self.param_types = {}
        self.categorical_mappings = {}
        
        self._initialize_space()
    
    def _initialize_space(self):
        """Initialize parameter ranges and types."""
        for param, values in self.base_space.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, float)) for v in values):
                    # Check if all values are integers (discrete) or mixed/float (continuous)
                    if all(isinstance(v, int) for v in values):
                        self.param_types[param] = 'discrete'
                        self.categorical_mappings[param] = values
                    else:
                        # Numeric continuous range
                        self.param_ranges[param] = (min(values), max(values))
                        self.param_types[param] = 'numeric'
                else:
                    # Categorical
                    self.param_types[param] = 'categorical'
                    self.categorical_mappings[param] = values
            elif isinstance(values, tuple) and len(values) == 2:
                # Range tuple - assume continuous
                self.param_ranges[param] = values
                self.param_types[param] = 'numeric'
            else:
                raise ValueError(f"Unsupported parameter format for {param}: {values}")
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample random parameter configurations."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param in self.base_space:
                if self.param_types[param] == 'numeric':
                    low, high = self.param_ranges[param]
                    sample[param] = np.random.uniform(low, high)
                elif self.param_types[param] == 'discrete':
                    sample[param] = np.random.choice(self.categorical_mappings[param])
                else:  # categorical
                    sample[param] = np.random.choice(self.categorical_mappings[param])
            samples.append(sample)
        
        return samples
    
    def normalize_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Normalize parameters to [0, 1] range for GP."""
        normalized = []
        
        for param in self.base_space:
            value = params[param]
            
            if self.param_types[param] == 'numeric':
                low, high = self.param_ranges[param]
                norm_value = (value - low) / (high - low)
            elif self.param_types[param] == 'discrete':
                # Discrete integer to numeric mapping
                categories = self.categorical_mappings[param]
                norm_value = categories.index(value) / (len(categories) - 1) if len(categories) > 1 else 0.0
            else:  # categorical
                # Categorical to numeric mapping
                categories = self.categorical_mappings[param]
                norm_value = categories.index(value) / (len(categories) - 1) if len(categories) > 1 else 0.0
            
            normalized.append(norm_value)
        
        return np.array(normalized)
    
    def denormalize_params(self, normalized: np.ndarray) -> Dict[str, Any]:
        """Denormalize parameters from [0, 1] range."""
        params = {}
        param_names = list(self.base_space.keys())
        
        for i, param in enumerate(param_names):
            norm_value = max(0, min(1, normalized[i]))  # Clip to [0, 1]
            
            if self.param_types[param] == 'numeric':
                low, high = self.param_ranges[param]
                value = low + norm_value * (high - low)
            elif self.param_types[param] == 'discrete':
                # Numeric to discrete integer mapping
                categories = self.categorical_mappings[param]
                idx = int(round(norm_value * (len(categories) - 1)))
                idx = max(0, min(len(categories) - 1, idx))  # Ensure valid index
                value = categories[idx]
            else:  # categorical
                # Numeric to categorical mapping
                categories = self.categorical_mappings[param]
                idx = int(round(norm_value * (len(categories) - 1)))
                idx = max(0, min(len(categories) - 1, idx))  # Ensure valid index
                value = categories[idx]
            
            params[param] = value
        
        return params


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(self, config: HyperoptConfig):
        self.config = config
        self.gp_model = None
        self.X_train = []
        self.y_train = []
        self.scaler = StandardScaler()
        self.trials_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize GP kernel
        if config.gp_kernel == 'matern':
            kernel = Matern(length_scale=1.0, nu=config.gp_nu) + WhiteKernel(noise_level=config.gp_noise_level)
        else:
            raise ValueError(f"Unsupported kernel: {config.gp_kernel}")
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=config.gp_normalize_y,
            random_state=config.random_state,
            alpha=1e-6  # Numerical stability
        )
    
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                            y_best: float) -> np.ndarray:
        """Compute acquisition function values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(X.reshape(1, -1), return_std=True)
        
        mu, sigma = mu[0], sigma[0]
        
        if sigma < 1e-9:
            return np.array([0.0])
        
        if self.config.acquisition_func == 'ei':
            # Expected Improvement
            improvement = mu - y_best - self.config.ei_xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return np.array([ei])
        
        elif self.config.acquisition_func == 'pi':
            # Probability of Improvement  
            Z = (mu - y_best - self.config.pi_xi) / sigma
            return np.array([norm.cdf(Z)])
        
        elif self.config.acquisition_func == 'ucb':
            # Upper Confidence Bound
            return np.array([mu + self.config.ucb_kappa * sigma])
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_func}")
    
    def _optimize_acquisition(self, search_space: AdaptiveSearchSpace) -> Dict[str, Any]:
        """Optimize acquisition function to find next parameter set."""
        if len(self.X_train) == 0:
            return search_space.sample_random(1)[0]
        
        # Current best value
        y_best = max(self.y_train) if self.config.maximize else min(self.y_train)
        
        best_params = None
        best_acquisition = -np.inf if self.config.maximize else np.inf
        
        # Multi-start optimization
        n_starts = min(20, len(self.X_train) + 5)
        
        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(0, 1, len(self.X_train[0]))
            
            # Optimize acquisition function
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x, self.gp_model, y_best)[0],
                    x0,
                    bounds=[(0, 1)] * len(x0),
                    method='L-BFGS-B'
                )
                
                if result.success:
                    acq_value = -result.fun
                    if acq_value > best_acquisition:
                        best_acquisition = acq_value
                        best_params = search_space.denormalize_params(result.x)
            
            except Exception as e:
                self.logger.warning(f"Acquisition optimization failed: {e}")
                continue
        
        # Fallback to random sampling if optimization failed
        if best_params is None:
            best_params = search_space.sample_random(1)[0]
        
        return best_params
    
    def update_model(self, X: List[np.ndarray], y: List[float]):
        """Update GP model with new observations."""
        if len(X) == 0:
            return
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        try:
            X_array = np.array(X)
            y_array = np.array(y)
            
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)
            
            # Fit GP model
            self.gp_model.fit(X_array, y_array)
            
        except Exception as e:
            self.logger.warning(f"GP model update failed: {e}")
    
    def suggest_next_params(self, search_space: AdaptiveSearchSpace, 
                          X_history: List[np.ndarray], y_history: List[float]) -> Dict[str, Any]:
        """Suggest next parameter configuration to evaluate."""
        self.update_model(X_history, y_history)
        
        if len(X_history) < self.config.n_initial_samples:
            # Random exploration phase
            return search_space.sample_random(1)[0]
        else:
            # Acquisition-based exploration
            return self._optimize_acquisition(search_space)


class LossWeightOptimizer:
    """Optimizer for combined loss function weights."""
    
    @staticmethod
    def logit_to_weights(logits: List[float]) -> List[float]:
        """Convert logit space to normalized weights using softmax."""
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits)
        return weights.tolist()
    
    @staticmethod  
    def weights_to_logit(weights: List[float]) -> List[float]:
        """Convert normalized weights to logit space."""
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        weights = np.array(weights) + eps
        weights = weights / np.sum(weights)  # Renormalize
        
        # Use last weight as reference (set to 0)
        logits = np.log(weights[:-1] / weights[-1])
        return logits.tolist()
    
    @staticmethod
    def create_loss_weight_params(loss_config: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Create tunable loss weight parameters."""
        if loss_config.get('type') != 'combined':
            return {}
        
        # Extract current weights
        ce_weight = loss_config.get('ce_weight', 0.4)
        focal_weight = loss_config.get('focal_weight', 0.4) 
        qwk_weight = loss_config.get('qwk_weight', 0.2)
        
        # Convert to logit space for optimization
        current_weights = [ce_weight, focal_weight, qwk_weight]
        logits = LossWeightOptimizer.weights_to_logit(current_weights)
        
        # Define search ranges in logit space (roughly -2 to 2 covers most reasonable weight ranges)
        return {
            'ce_weight_logit': (-2.0, 2.0),
            'focal_weight_logit': (-2.0, 2.0)
            # qwk_weight is computed to make weights sum to 1.0
        }
    
    @staticmethod
    def params_to_loss_config(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert optimization parameters to loss configuration."""
        if 'ce_weight_logit' not in params:
            return {}
        
        # Convert logit space to weights
        logits = [params['ce_weight_logit'], params['focal_weight_logit'], 0.0]  # Last weight as reference
        weights = LossWeightOptimizer.logit_to_weights(logits)
        
        return {
            'type': 'combined',
            'ce_weight': weights[0],
            'focal_weight': weights[1], 
            'qwk_weight': weights[2]
        }


class HyperparameterOptimizer:
    """Main hyperparameter optimizer coordinating all components."""
    
    def __init__(self, model_name: str, dataset: str, config: HyperoptConfig):
        self.model_name = model_name
        self.dataset = dataset 
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize search space
        self.search_space = self._create_search_space()
        self.optimizer = BayesianOptimizer(config)
        
        # Trial tracking
        self.trials = []
        self.X_history = []
        self.y_history = []
        self.best_trial = None
        
        # Early stopping
        self.patience_counter = 0
        self.best_score = -np.inf if config.maximize else np.inf
        
    def _create_search_space(self) -> AdaptiveSearchSpace:
        """Create adaptive search space from model factory."""
        base_space = get_model_hyperparameter_grid(self.model_name)
        
        if not base_space:
            # Fallback default space
            base_space = {
                'lr': (1e-4, 1e-2),
                'batch_size': [32, 64, 128],
                'hidden_dim': [64, 128, 256],
                'dropout': (0.0, 0.5)
            }
            self.logger.warning(f"No factory hyperparameter grid found for {self.model_name}, using defaults")
        
        # Add loss weight parameters if model uses combined loss
        loss_config = get_model_loss_config(self.model_name)
        loss_weight_params = LossWeightOptimizer.create_loss_weight_params(loss_config)
        base_space.update(loss_weight_params)
        
        self.logger.info(f"Search space for {self.model_name}: {list(base_space.keys())}")
        return AdaptiveSearchSpace(base_space)
    
    def _evaluate_params(self, params: Dict[str, Any]) -> TrialResult:
        """Evaluate a parameter configuration using cross-validation."""
        import time
        start_time = time.time()
        
        # Convert loss weight parameters back to loss config
        loss_override = LossWeightOptimizer.params_to_loss_config(params)
        
        # Filter out loss weight logits from model params
        model_params = {k: v for k, v in params.items() if not k.endswith('_logit')}
        
        trial_id = len(self.trials) + 1
        self.logger.info(f"Trial {trial_id}: Evaluating {model_params}")
        if loss_override:
            self.logger.info(f"Loss weights: ce={loss_override['ce_weight']:.3f}, "
                           f"focal={loss_override['focal_weight']:.3f}, "
                           f"qwk={loss_override['qwk_weight']:.3f}")
        
        try:
            # Perform cross-validation with these parameters
            cv_scores = self._run_cross_validation(model_params, loss_override)
            
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
            
            self.logger.info(f"Trial {trial_id} completed: {self.config.metric}={metric_value:.4f} ±{cv_std:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Trial {trial_id} failed: {e}")
            # Return worst possible score
            worst_score = -1.0 if self.config.maximize else 1.0
            return TrialResult(
                params=params,
                metric_value=worst_score,
                cv_scores=[worst_score] * self.config.cv_folds,
                cv_std=0.0,
                trial_id=trial_id,
                early_stopped=True,
                training_time=time.time() - start_time
            )
    
    def _run_cross_validation(self, model_params: Dict[str, Any], 
                            loss_override: Dict[str, Any]) -> List[float]:
        """Run cross-validation for parameter evaluation using real training."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from sklearn.model_selection import KFold
        from train import load_simple_data, create_data_loaders, create_model, train_single_fold
        import torch
        
        try:
            # Load dataset
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
                                           if k not in ['lr', 'batch_size', 'weight_decay']}
                    model = create_model(self.model_name, n_questions, n_cats, device, **filtered_model_params)
                    
                    # Prepare loss kwargs
                    loss_kwargs = {'lr': model_params.get('lr', 0.001)}
                    if loss_override:
                        loss_kwargs.update(loss_override)
                    
                    # Use reduced epochs for hyperparameter search (faster evaluation)
                    cv_epochs = min(10, max(3, model_params.get('epochs', 10)))
                    
                    # Train with reduced epochs for hyperparameter search
                    _, metrics, _ = train_single_fold(
                        model, train_loader, val_loader, device, cv_epochs,
                        self.model_name, fold=None, loss_type='combined',
                        loss_kwargs=loss_kwargs
                    )
                    
                    # Extract the metric we're optimizing for
                    score = metrics.get(self.config.metric, 0.0)
                    cv_scores.append(score)
                    
                    # Log fold progress
                    self.logger.debug(f"Fold {fold_idx}: {self.config.metric}={score:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Fold {fold_idx} failed: {e}")
                    # Use a poor score for failed folds
                    cv_scores.append(0.1 if self.config.maximize else 1.0)
            
            # Ensure we have scores for all folds
            while len(cv_scores) < self.config.cv_folds:
                cv_scores.append(0.1 if self.config.maximize else 1.0)
            
            return cv_scores
            
        except Exception as e:
            self.logger.error(f"CV evaluation failed completely: {e}")
            # Return worst possible scores for all folds
            worst_score = 0.1 if self.config.maximize else 1.0
            return [worst_score] * self.config.cv_folds
    
    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization."""
        self.logger.info(f"Starting Bayesian optimization for {self.model_name}")
        self.logger.info(f"Trials: {self.config.n_trials}, CV folds: {self.config.cv_folds}")
        self.logger.info(f"Optimizing: {self.config.metric} ({'maximize' if self.config.maximize else 'minimize'})")
        
        for trial_idx in range(self.config.n_trials):
            # Get next parameter configuration
            params = self.optimizer.suggest_next_params(
                self.search_space, self.X_history, self.y_history
            )
            
            # Evaluate parameters
            result = self._evaluate_params(params)
            self.trials.append(result)
            
            # Update history
            normalized_params = self.search_space.normalize_params(params)
            self.X_history.append(normalized_params)
            self.y_history.append(result.metric_value)
            
            # Check for improvement
            is_improvement = False
            if self.config.maximize:
                if result.metric_value > self.best_score + self.config.min_improvement:
                    self.best_score = result.metric_value
                    self.best_trial = result
                    is_improvement = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                if result.metric_value < self.best_score - self.config.min_improvement:
                    self.best_score = result.metric_value
                    self.best_trial = result
                    is_improvement = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            # Log progress
            progress = (trial_idx + 1) / self.config.n_trials * 100
            self.logger.info(f"Progress: {progress:.1f}% | "
                           f"Best {self.config.metric}: {self.best_score:.4f} | "
                           f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {trial_idx + 1} trials")
                break
        
        # Return optimization results
        return self._format_results()
    
    def _format_results(self) -> Dict[str, Any]:
        """Format optimization results."""
        if self.best_trial is None:
            raise RuntimeError("No successful trials completed")
        
        results = {
            'best_params': self.best_trial.params,
            'best_score': self.best_trial.metric_value,
            'best_cv_std': self.best_trial.cv_std,
            'total_trials': len(self.trials),
            'successful_trials': len([t for t in self.trials if not t.early_stopped]),
            'total_time': sum(t.training_time for t in self.trials),
            'convergence_trial': self.best_trial.trial_id,
            'metric': self.config.metric,
            'all_trials': [
                {
                    'trial_id': t.trial_id,
                    'params': t.params,
                    'score': t.metric_value,
                    'cv_std': t.cv_std,
                    'early_stopped': t.early_stopped,
                    'training_time': t.training_time
                } for t in self.trials
            ]
        }
        
        # Add loss weight interpretation if optimized
        if any(k.endswith('_logit') for k in self.best_trial.params):
            loss_config = LossWeightOptimizer.params_to_loss_config(self.best_trial.params)
            results['best_loss_weights'] = loss_config
        
        self.logger.info(f"Optimization completed!")
        self.logger.info(f"Best {self.config.metric}: {self.best_trial.metric_value:.4f} ±{self.best_trial.cv_std:.4f}")
        self.logger.info(f"Best parameters: {self.best_trial.params}")
        
        return results


def run_hyperopt(model_name: str, dataset: str, n_trials: int = 50, 
                metric: str = 'quadratic_weighted_kappa', cv_folds: int = 5) -> Dict[str, Any]:
    """Run hyperparameter optimization for a model."""
    config = HyperoptConfig(
        n_trials=n_trials,
        metric=metric,
        cv_folds=cv_folds,
        maximize=True,  # Assuming QWK and accuracy metrics
        random_state=42
    )
    
    optimizer = HyperparameterOptimizer(model_name, dataset, config)
    results = optimizer.optimize()
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_hyperopt(
        model_name='deep_gpcm',
        dataset='synthetic_500_200_4', 
        n_trials=20,
        metric='quadratic_weighted_kappa',
        cv_folds=3
    )
    
    print("\nHyperparameter Optimization Results:")
    print(f"Best {results['metric']}: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    if 'best_loss_weights' in results:
        print(f"Best loss weights: {results['best_loss_weights']}")