"""
Hyperparameter optimization for ordinal-aware attention mechanisms.
Provides systematic tuning of attention parameters for optimal performance.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import itertools
import time
from dataclasses import dataclass

# Import training components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ordinal_trainer import OrdinalTrainer


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search."""
    learning_rate: float
    ordinal_weight: float
    qwk_weight: float
    distance_penalty: float
    n_heads: int
    n_cycles: int
    dropout_rate: float


class HyperparameterOptimizer:
    """Hyperparameter optimizer for ordinal attention mechanisms."""
    
    def __init__(self, n_questions: int = 30, n_cats: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.device = device
        
        # Define search spaces
        self.search_spaces = {
            'learning_rate': [0.0005, 0.001, 0.002],
            'ordinal_weight': [0.2, 0.5, 1.0],
            'qwk_weight': [0.3, 0.5, 0.7],
            'distance_penalty': [0.05, 0.1, 0.2],
            'n_heads': [4, 8],
            'n_cycles': [1, 2, 3],
            'dropout_rate': [0.1, 0.2]
        }
        
        self.results = {}
    
    def create_hyperparameter_configs(self, max_configs: int = 20) -> List[HyperparameterConfig]:
        """Create list of hyperparameter configurations to test."""
        # Generate all combinations
        param_names = list(self.search_spaces.keys())
        param_values = [self.search_spaces[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit number of configurations
        if len(all_combinations) > max_configs:
            # Sample random subset
            np.random.seed(42)
            selected_indices = np.random.choice(len(all_combinations), max_configs, replace=False)
            selected_combinations = [all_combinations[i] for i in selected_indices]
        else:
            selected_combinations = all_combinations
        
        # Convert to config objects
        configs = []
        for combo in selected_combinations:
            config_dict = dict(zip(param_names, combo))
            configs.append(HyperparameterConfig(**config_dict))
        
        return configs
    
    def evaluate_configuration(self, config: HyperparameterConfig, 
                             n_samples: int = 400, n_epochs: int = 8) -> Dict[str, float]:
        """Evaluate a single hyperparameter configuration."""
        # Create trainer
        trainer = OrdinalTrainer(self.n_questions, self.n_cats, self.device)
        
        # Override trainer configs with optimized parameters
        trainer.configs = {
            'optimized': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned'],
                'loss_type': 'combined'
            }
        }
        
        # Generate dataset
        train_loader, val_loader = trainer.create_synthetic_dataset(n_samples)
        
        # Create model with hyperparameters
        model, loss_fn = trainer.create_model_and_loss('optimized')
        
        # Apply hyperparameters to model
        if hasattr(model, 'attention_refinement') and hasattr(model.attention_refinement, 'attention_pipeline'):
            for mechanism in model.attention_refinement.attention_pipeline.mechanisms:
                if hasattr(mechanism, 'distance_penalty'):
                    mechanism.distance_penalty = config.distance_penalty
                if hasattr(mechanism, 'dropout'):
                    mechanism.dropout = config.dropout_rate
        
        # Apply to loss function
        if hasattr(loss_fn, 'ordinal_weight'):
            loss_fn.ordinal_weight = config.ordinal_weight
        if hasattr(loss_fn, 'qwk_weight'):
            loss_fn.qwk_weight = config.qwk_weight
        
        # Train with hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
        
        from models.metrics.ordinal_metrics import MetricsTracker
        metrics_tracker = MetricsTracker(self.n_cats)
        
        best_qwk = -1.0
        best_metrics = None
        
        try:
            # Training loop
            for epoch in range(n_epochs):
                # Train epoch
                model.train()
                for batch_idx, (questions, responses) in enumerate(train_loader):
                    questions, responses = questions.to(self.device), responses.to(self.device)
                    
                    optimizer.zero_grad()
                    _, _, _, probs = model(questions, responses)
                    logits = torch.log(probs + 1e-8)
                    loss, _ = loss_fn(logits, responses)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                val_preds, val_targets, val_probs = [], [], []
                with torch.no_grad():
                    for questions, responses in val_loader:
                        questions, responses = questions.to(self.device), responses.to(self.device)
                        _, _, _, probs = model(questions, responses)
                        preds = probs.argmax(dim=-1)
                        val_preds.append(preds)
                        val_targets.append(responses)
                        val_probs.append(probs)
                
                val_preds = torch.cat(val_preds, dim=0)
                val_targets = torch.cat(val_targets, dim=0)
                val_probs = torch.cat(val_probs, dim=0)
                
                val_metrics = metrics_tracker.update(val_targets, val_preds, val_probs, split='val')
                scheduler.step(val_metrics['qwk'])
                
                if val_metrics['qwk'] > best_qwk:
                    best_qwk = val_metrics['qwk']
                    best_metrics = val_metrics.copy()
        
        except Exception as e:
            # Handle training failures
            return {
                'qwk': 0.0,
                'accuracy': 0.0,
                'mae': 10.0,
                'ordinal_accuracy': 0.0,
                'training_failed': True,
                'error': str(e)
            }
        
        if best_metrics is None:
            best_metrics = {
                'qwk': 0.0,
                'accuracy': 0.0,
                'mae': 10.0,
                'ordinal_accuracy': 0.0
            }
        
        return best_metrics
    
    def run_hyperparameter_search(self, max_configs: int = 12, n_samples: int = 300, 
                                 n_epochs: int = 6) -> Dict[HyperparameterConfig, Dict[str, float]]:
        """Run hyperparameter search across configurations."""
        print("="*60)
        print("HYPERPARAMETER OPTIMIZATION FOR ORDINAL ATTENTION")
        print("="*60)
        
        configs = self.create_hyperparameter_configs(max_configs)
        print(f"\nTesting {len(configs)} hyperparameter configurations")
        print(f"Dataset: {n_samples} samples, {n_epochs} epochs each")
        
        results = {}
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Testing configuration:")
            print(f"  LR: {config.learning_rate}, Ord.Weight: {config.ordinal_weight}")
            print(f"  QWK.Weight: {config.qwk_weight}, Dist.Penalty: {config.distance_penalty}")
            print(f"  Heads: {config.n_heads}, Cycles: {config.n_cycles}, Dropout: {config.dropout_rate}")
            
            start_time = time.time()
            metrics = self.evaluate_configuration(config, n_samples, n_epochs)
            end_time = time.time()
            
            metrics['training_time'] = end_time - start_time
            results[config] = metrics
            
            print(f"  → QWK: {metrics['qwk']:.3f}, Acc: {metrics['accuracy']:.3f}, "
                  f"MAE: {metrics['mae']:.3f}, Time: {metrics['training_time']:.1f}s")
        
        self.results = results
        return results
    
    def find_best_configuration(self, metric: str = 'qwk') -> Tuple[HyperparameterConfig, Dict[str, float]]:
        """Find the best hyperparameter configuration."""
        if not self.results:
            raise ValueError("No results available. Run hyperparameter search first.")
        
        # Filter out failed configurations
        valid_results = {config: metrics for config, metrics in self.results.items() 
                        if not metrics.get('training_failed', False)}
        
        if not valid_results:
            raise ValueError("All configurations failed during training.")
        
        if metric in ['mae']:
            # Lower is better
            best_config, best_metrics = min(valid_results.items(), key=lambda x: x[1][metric])
        else:
            # Higher is better
            best_config, best_metrics = max(valid_results.items(), key=lambda x: x[1][metric])
        
        return best_config, best_metrics
    
    def print_optimization_summary(self):
        """Print hyperparameter optimization summary."""
        if not self.results:
            print("No optimization results available.")
            return
        
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        # Filter valid results
        valid_results = {config: metrics for config, metrics in self.results.items() 
                        if not metrics.get('training_failed', False)}
        
        print(f"\nValid configurations: {len(valid_results)}/{len(self.results)}")
        
        if not valid_results:
            print("No valid configurations found.")
            return
        
        # Best configurations
        best_qwk_config, best_qwk_metrics = self.find_best_configuration('qwk')
        best_acc_config, best_acc_metrics = self.find_best_configuration('accuracy')
        
        print(f"\nBest QWK Configuration:")
        print(f"  QWK: {best_qwk_metrics['qwk']:.3f}")
        print(f"  LR: {best_qwk_config.learning_rate}")
        print(f"  Ordinal Weight: {best_qwk_config.ordinal_weight}")
        print(f"  QWK Weight: {best_qwk_config.qwk_weight}")
        print(f"  Distance Penalty: {best_qwk_config.distance_penalty}")
        print(f"  Heads: {best_qwk_config.n_heads}, Cycles: {best_qwk_config.n_cycles}")
        print(f"  Dropout: {best_qwk_config.dropout_rate}")
        
        print(f"\nBest Accuracy Configuration:")
        print(f"  Accuracy: {best_acc_metrics['accuracy']:.3f}")
        print(f"  LR: {best_acc_config.learning_rate}")
        print(f"  Ordinal Weight: {best_acc_config.ordinal_weight}")
        print(f"  QWK Weight: {best_acc_config.qwk_weight}")
        
        # Performance distribution
        qwks = [metrics['qwk'] for metrics in valid_results.values()]
        print(f"\nPerformance Distribution:")
        print(f"  QWK - Mean: {np.mean(qwks):.3f}, Std: {np.std(qwks):.3f}")
        print(f"  QWK - Min: {np.min(qwks):.3f}, Max: {np.max(qwks):.3f}")
        
        # Parameter analysis
        print(f"\nParameter Impact Analysis:")
        param_impact = self._analyze_parameter_impact()
        for param, correlation in param_impact.items():
            print(f"  {param}: {correlation:+.3f} correlation with QWK")
        
        print("="*60)
    
    def _analyze_parameter_impact(self) -> Dict[str, float]:
        """Analyze correlation between parameters and performance."""
        valid_results = {config: metrics for config, metrics in self.results.items() 
                        if not metrics.get('training_failed', False)}
        
        if len(valid_results) < 3:
            return {}
        
        # Extract parameter values and QWK scores
        qwks = [metrics['qwk'] for metrics in valid_results.values()]
        
        correlations = {}
        for param_name in self.search_spaces.keys():
            param_values = [getattr(config, param_name) for config in valid_results.keys()]
            
            # Calculate correlation
            if len(set(param_values)) > 1:  # Only if there's variation
                correlation = np.corrcoef(param_values, qwks)[0, 1]
                if not np.isnan(correlation):
                    correlations[param_name] = correlation
        
        return correlations
    
    def get_recommended_config(self) -> HyperparameterConfig:
        """Get recommended hyperparameter configuration."""
        if not self.results:
            # Return default configuration
            return HyperparameterConfig(
                learning_rate=0.001,
                ordinal_weight=0.5,
                qwk_weight=0.5,
                distance_penalty=0.1,
                n_heads=8,
                n_cycles=2,
                dropout_rate=0.2
            )
        
        best_config, _ = self.find_best_configuration('qwk')
        return best_config


def run_quick_optimization():
    """Run quick hyperparameter optimization."""
    print("Running quick hyperparameter optimization...")
    
    optimizer = HyperparameterOptimizer(n_questions=25, n_cats=4)
    
    # Run optimization
    results = optimizer.run_hyperparameter_search(max_configs=8, n_samples=250, n_epochs=5)
    optimizer.print_optimization_summary()
    
    # Get recommended config
    recommended = optimizer.get_recommended_config()
    print(f"\nRecommended Configuration:")
    print(f"  Learning Rate: {recommended.learning_rate}")
    print(f"  Ordinal Weight: {recommended.ordinal_weight}")
    print(f"  QWK Weight: {recommended.qwk_weight}")
    print(f"  Distance Penalty: {recommended.distance_penalty}")
    print(f"  Attention Heads: {recommended.n_heads}")
    print(f"  Refinement Cycles: {recommended.n_cycles}")
    print(f"  Dropout Rate: {recommended.dropout_rate}")
    
    return optimizer


if __name__ == "__main__":
    optimizer = run_quick_optimization()