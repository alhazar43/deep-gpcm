"""
Adaptive Hyperparameter Optimization Scheduler

Implements Phase 1: Adaptive epoch allocation with fallback system and expanded search space.
Safely detachable system that can fall back to original fixed-epoch optimization.
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from scipy.stats import pearsonr

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Conditional imports to avoid heavy dependencies during testing
try:
    from models.factory import get_model_hyperparameter_grid, get_model_loss_config
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False
    def get_model_hyperparameter_grid(model_name):
        return {}
    def get_model_loss_config(model_name):
        return {}


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive hyperparameter optimization."""
    
    # Adaptive epoch allocation
    enable_adaptive_epochs: bool = True
    min_epochs: int = 5
    intermediate_epochs: int = 15
    max_epochs: int = 40
    early_promotion_threshold: float = 0.02  # Score improvement needed for promotion
    
    # Expanded search space
    enable_architectural_params: bool = True
    enable_learning_params: bool = True
    
    # Safety and fallback
    fallback_on_failure: bool = True
    max_consecutive_failures: int = 3
    validation_threshold: float = 0.1  # Minimum performance for adaptive mode
    
    # A/B testing configuration
    enable_ab_testing: bool = False  # Phase 2 feature
    ab_split_ratio: float = 0.5
    
    # Performance monitoring
    performance_window: int = 5  # Trials to consider for performance trends
    improvement_threshold: float = 0.01  # Minimum improvement rate


class AdaptiveEpochScheduler:
    """Adaptive epoch allocation with progressive evaluation strategy."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.trial_history = []
        self.consecutive_failures = 0
        self.adaptive_enabled = config.enable_adaptive_epochs
        
        # Performance tracking
        self.recent_performances = []
        self.baseline_performance = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def suggest_epochs(self, trial_id: int, params: Dict[str, Any], 
                      trial_history: List[Dict[str, Any]]) -> Tuple[int, str]:
        """
        Suggest number of epochs for next trial with reasoning.
        
        Returns:
            Tuple of (epochs, reasoning)
        """
        if not self.adaptive_enabled:
            return self.config.intermediate_epochs, "Adaptive scheduling disabled"
        
        # Safety check: fallback if too many failures
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.warning(f"Falling back to fixed epochs after {self.consecutive_failures} failures")
            return self.config.intermediate_epochs, "Fallback due to consecutive failures"
        
        # Phase 1: Start with minimum epochs for exploration
        if trial_id <= 5:
            return self.config.min_epochs, "Early exploration phase - quick evaluation"
        
        # Analyze recent performance trends
        recent_performance = self._analyze_recent_performance(trial_history)
        
        # Phase 2: Intermediate epochs for most trials
        if trial_id <= 15:
            if recent_performance and recent_performance > 0.6:
                return self.config.intermediate_epochs, "Good performance - standard evaluation"
            else:
                return self.config.min_epochs, "Low performance - continue exploration"
        
        # Phase 3: Adaptive allocation based on promise
        promise_score = self._calculate_promise_score(params, trial_history)
        
        if promise_score > 0.8:
            return self.config.max_epochs, f"High promise score ({promise_score:.3f}) - full evaluation"
        elif promise_score > 0.6:
            return self.config.intermediate_epochs, f"Medium promise score ({promise_score:.3f}) - standard evaluation"
        else:
            return self.config.min_epochs, f"Low promise score ({promise_score:.3f}) - quick evaluation"
    
    def _analyze_recent_performance(self, trial_history: List[Dict[str, Any]]) -> Optional[float]:
        """Analyze recent trial performance."""
        if not trial_history:
            return None
        
        # Get last N trials
        recent_trials = trial_history[-self.config.performance_window:]
        scores = [trial.get('score', 0) for trial in recent_trials if not trial.get('early_stopped', False)]
        
        if not scores:
            return None
        
        return np.mean(scores)
    
    def _calculate_promise_score(self, params: Dict[str, Any], 
                                trial_history: List[Dict[str, Any]]) -> float:
        """
        Calculate promise score for parameter configuration.
        
        Based on:
        1. Similarity to successful configurations
        2. Parameter importance weights
        3. Recent performance trends
        """
        if not trial_history:
            return 0.5  # Neutral score for first trials
        
        # Get successful trials (top quartile)
        successful_trials = sorted(
            [t for t in trial_history if not t.get('early_stopped', False)],
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        if not successful_trials:
            return 0.5
        
        top_quartile_count = max(1, len(successful_trials) // 4)
        top_trials = successful_trials[:top_quartile_count]
        
        # Calculate similarity to successful configurations
        similarity_scores = []
        for top_trial in top_trials:
            top_params = top_trial.get('params', {})
            similarity = self._calculate_parameter_similarity(params, top_params)
            similarity_scores.append(similarity)
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.5
        
        # Adjust based on recent performance trends
        recent_perf = self._analyze_recent_performance(trial_history)
        if recent_perf:
            # If recent performance is good, be more conservative
            # If recent performance is poor, be more exploratory
            trend_adjustment = min(recent_perf, 0.8)  # Cap at 0.8
            final_score = (avg_similarity * 0.7) + (trend_adjustment * 0.3)
        else:
            final_score = avg_similarity
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], 
                                      params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter configurations."""
        if not params1 or not params2:
            return 0.0
        
        # Get common parameters
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0
        
        similarities = []
        for param in common_params:
            val1, val2 = params1[param], params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                if val1 == val2:
                    sim = 1.0
                else:
                    # Normalized distance
                    max_val = max(abs(val1), abs(val2), 1e-6)
                    sim = 1.0 - (abs(val1 - val2) / max_val)
                    sim = max(0, sim)  # Ensure non-negative
            else:
                # Categorical similarity
                sim = 1.0 if val1 == val2 else 0.0
            
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def update_trial_result(self, trial_result: Dict[str, Any]):
        """Update scheduler with trial result."""
        self.trial_history.append(trial_result)
        
        # Track failures for fallback mechanism
        if trial_result.get('early_stopped', False) or trial_result.get('score', 0) < self.config.validation_threshold:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Update performance tracking
        if not trial_result.get('early_stopped', False):
            score = trial_result.get('score', 0)
            self.recent_performances.append(score)
            
            # Keep only recent performances
            if len(self.recent_performances) > self.config.performance_window:
                self.recent_performances.pop(0)
        
        # Check if adaptive mode should be disabled
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.warning("Disabling adaptive epochs due to consecutive failures")
            self.adaptive_enabled = False


class ExpandedSearchSpace:
    """Expanded hyperparameter search space with architectural parameters."""
    
    def __init__(self, base_space: Dict[str, Any], config: AdaptiveConfig, model_name: str = None):
        self.base_space = base_space.copy()
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        if config.enable_architectural_params:
            self._add_architectural_params()
        
        if config.enable_learning_params:
            self._add_learning_params()
    
    def _add_architectural_params(self):
        """Add architectural parameters to search space based on model capabilities."""
        # Common memory network parameters (all models support these)
        common_params = {
            'key_dim': [32, 50, 64, 128],
            'value_dim': [128, 200, 256, 512],
        }
        
        # Attention-specific parameters (only for attention models)
        attention_params = {
            'embed_dim': [32, 64, 128, 256],
            'n_heads': [2, 4, 8],
            'n_cycles': [1, 2, 3],
        }
        
        # Advanced network parameters (for complex models)
        advanced_params = {
            'num_layers': [1, 2, 3],
            'hidden_dim': [64, 128, 256],
        }
        
        # Add common parameters for all models
        for param, values in common_params.items():
            if param not in self.base_space:
                self.base_space[param] = values
                self.logger.info(f"Added common architectural parameter: {param}")
        
        # Add attention parameters only for attention models
        if self.model_name and ('attn' in self.model_name or 'attention' in self.model_name):
            for param, values in attention_params.items():
                if param not in self.base_space:
                    self.base_space[param] = values
                    self.logger.info(f"Added attention parameter: {param}")
        
        # Add advanced parameters for temporal/complex models
        if self.model_name and ('temporal' in self.model_name or 'stable' in self.model_name):
            for param, values in advanced_params.items():
                if param not in self.base_space:
                    self.base_space[param] = values
                    self.logger.info(f"Added advanced parameter: {param}")
    
    def _add_learning_params(self):
        """Add learning-related parameters to search space."""
        learning_params = {
            # Learning rate and optimization
            'lr': (1e-4, 1e-2),
            'weight_decay': (1e-6, 1e-3),
            'batch_size': [32, 64, 128, 256],
            
            # Regularization
            'grad_clip': (0.5, 5.0),
            'label_smoothing': (0.0, 0.2),
        }
        
        # Only add parameters that don't conflict with existing ones
        for param, values in learning_params.items():
            if param not in self.base_space:
                self.base_space[param] = values
                self.logger.info(f"Added learning parameter: {param}")
    
    def get_expanded_space(self) -> Dict[str, Any]:
        """Get the expanded search space."""
        return self.base_space.copy()


class AdaptiveHyperoptIntegration:
    """Integration layer for adaptive hyperparameter optimization."""
    
    def __init__(self, model_name: str, dataset: str, adaptive_config: Optional[AdaptiveConfig] = None):
        self.model_name = model_name
        self.dataset = dataset
        self.config = adaptive_config or AdaptiveConfig()
        
        # Initialize components
        self.epoch_scheduler = AdaptiveEpochScheduler(self.config)
        self.expanded_space = None
        
        # Safety and monitoring
        self.original_hyperopt_results = None
        self.adaptive_hyperopt_results = None
        self.fallback_triggered = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_enhanced_search_space(self, base_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced search space with adaptive parameters."""
        if not self.expanded_space:
            self.expanded_space = ExpandedSearchSpace(base_space, self.config, self.model_name)
        
        enhanced_space = self.expanded_space.get_expanded_space()
        
        self.logger.info(f"Enhanced search space: {len(enhanced_space)} parameters")
        self.logger.info(f"New parameters: {set(enhanced_space.keys()) - set(base_space.keys())}")
        
        return enhanced_space
    
    def suggest_trial_epochs(self, trial_id: int, params: Dict[str, Any], 
                           trial_history: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Suggest epochs for trial with adaptive scheduling."""
        return self.epoch_scheduler.suggest_epochs(trial_id, params, trial_history)
    
    def update_trial_result(self, trial_result: Dict[str, Any]):
        """Update adaptive components with trial result."""
        self.epoch_scheduler.update_trial_result(trial_result)
    
    def should_fallback(self) -> bool:
        """Check if system should fallback to original optimization."""
        return (self.epoch_scheduler.consecutive_failures >= self.config.max_consecutive_failures or 
                not self.epoch_scheduler.adaptive_enabled)
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Compare adaptive vs original performance."""
        if not self.original_hyperopt_results or not self.adaptive_hyperopt_results:
            return {}
        
        original_score = self.original_hyperopt_results.get('best_score', 0)
        adaptive_score = self.adaptive_hyperopt_results.get('best_score', 0)
        
        improvement = adaptive_score - original_score
        improvement_percent = (improvement / original_score * 100) if original_score > 0 else 0
        
        return {
            'original_score': original_score,
            'adaptive_score': adaptive_score,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'adaptive_enabled': self.epoch_scheduler.adaptive_enabled,
            'fallback_triggered': self.fallback_triggered,
            'consecutive_failures': self.epoch_scheduler.consecutive_failures
        }
    
    def save_performance_report(self, results_dir: str = "results/adaptive_hyperopt"):
        """Save adaptive optimization performance report."""
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'config': {
                'adaptive_epochs': self.config.enable_adaptive_epochs,
                'architectural_params': self.config.enable_architectural_params,
                'learning_params': self.config.enable_learning_params,
                'min_epochs': self.config.min_epochs,
                'intermediate_epochs': self.config.intermediate_epochs,
                'max_epochs': self.config.max_epochs
            },
            'performance_comparison': self.get_performance_comparison(),
            'trial_history': self.epoch_scheduler.trial_history[-10:],  # Last 10 trials
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        report_path = results_path / f"adaptive_report_{self.model_name}_{self.dataset}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Adaptive optimization report saved: {report_path}")
        return str(report_path)


def create_adaptive_integration(model_name: str, dataset: str, 
                              adaptive_config: Optional[AdaptiveConfig] = None) -> AdaptiveHyperoptIntegration:
    """Factory function to create adaptive hyperparameter optimization integration."""
    return AdaptiveHyperoptIntegration(model_name, dataset, adaptive_config)


# Import pandas for timestamp
import pandas as pd

if __name__ == "__main__":
    # Example usage and testing
    config = AdaptiveConfig(
        enable_adaptive_epochs=True,
        enable_architectural_params=True,
        enable_learning_params=True,
        min_epochs=5,
        intermediate_epochs=15,
        max_epochs=40
    )
    
    # Test with synthetic example
    integration = create_adaptive_integration('deep_gpcm', 'synthetic_500_200_4', config)
    
    # Example search space enhancement
    base_space = {
        'memory_size': [20, 50, 100],
        'final_fc_dim': [50, 100],
        'dropout_rate': [0.0, 0.1, 0.2]
    }
    
    enhanced_space = integration.create_enhanced_search_space(base_space)
    print(f"Enhanced search space: {enhanced_space}")
    
    # Example epoch suggestion
    epochs, reason = integration.suggest_trial_epochs(1, {'memory_size': 50}, [])
    print(f"Trial 1: {epochs} epochs - {reason}")