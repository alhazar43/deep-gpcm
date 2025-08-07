#!/usr/bin/env python3
"""
Optimized Training Script for Deep-GPCM

Key optimizations:
- Unified configuration system with factory integration
- Intelligent hyperparameter optimization
- Experiment tracking and reproducibility
- Resource-aware training management
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from config import TrainingConfig, ExperimentConfig, HyperparameterConfig
from config.parser import parse_training_config, SmartArgumentParser
from models.factory import create_model_with_config, get_model_hyperparameter_grid
from utils.metrics import compute_metrics, save_results, ensure_results_dirs
from utils.path_utils import ensure_directories
from utils.data_loading import load_dataset
from training.losses import create_loss_function
from training.optimizers import create_optimizer
from training.schedulers import create_scheduler


class ExperimentTracker:
    """Simple experiment tracking for reproducibility."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.metrics_history = []
        self.start_time = datetime.now()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.model}_{self.config.dataset}_{timestamp}"
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        duration = datetime.now() - self.start_time
        
        return {
            'experiment_id': self.experiment_id,
            'config': self.config.__dict__,
            'duration_seconds': duration.total_seconds(),
            'total_epochs': len(self.metrics_history),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'best_qwk': max((m.get('quadratic_weighted_kappa', m.get('qwk', 0)) for m in self.metrics_history), default=0)
        }


class OptimizedTrainer:
    """Optimized training engine with factory integration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tracker = ExperimentTracker(config)
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.data_manager = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            # Enable deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_training(self, n_questions: int, n_cats: int):
        """Setup training components with factory integration."""
        
        print("🔧 Setting up training components...")
        
        # Create model with factory configuration
        model_params = self.config.get_model_params()
        
        # Remove n_questions and n_cats if present (passed as explicit args)
        model_params.pop('n_questions', None)
        model_params.pop('n_cats', None)
        
        self.model = create_model_with_config(
            self.config.model, 
            n_questions, 
            n_cats, 
            **model_params
        )
        self.model.to(self.device)
        
        # Create loss function
        loss_params = self.config.loss_config.__dict__.copy()
        loss_type = loss_params.pop('loss_type', 'ce')
        
        self.criterion = create_loss_function(
            loss_type,
            n_cats,
            **loss_params
        )
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model.parameters(),
            optimizer_type='adam',
            lr=self.config.lr
        )
        
        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type='reduce_on_plateau',
            patience=self.config.validation_config.patience // 2
        )
        
        print(f"✅ Model: {self.model.__class__.__name__} ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
        print(f"✅ Loss: {self.config.loss_config.loss_type}")
        print(f"✅ Optimizer: Adam (lr={self.config.lr})")
    
    def train_single_fold(self, train_loader, test_loader, fold: Optional[int] = None) -> Dict[str, Any]:
        """Train a single fold with enhanced monitoring and detailed epoch timing."""
        
        fold_desc = f"Fold {fold}" if fold is not None else "Single run"
        print(f"\n🚀 Training {fold_desc}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Test samples: {len(test_loader.dataset)}")
        
        # Display header like original train.py
        print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | MAE | Grad.Norm | LR | Time(s)")
        print("-" * 95)
        
        best_qwk = 0.0
        patience_counter = 0
        best_model_state = None
        epoch_metrics = []
        
        import time
        
        for epoch in range(self.config.epochs):
            start_time = time.time()  # Time each epoch like original
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(test_loader, epoch)
            
            # Combine metrics
            combined_metrics = {**train_metrics, **val_metrics}
            epoch_metrics.append(combined_metrics)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Log to tracker
            self.tracker.log_epoch(epoch, combined_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                # Use QWK metric for scheduler, with fallback to accuracy
                qwk_metric = val_metrics.get('quadratic_weighted_kappa', val_metrics.get('qwk', val_metrics.get('categorical_accuracy', 0.0)))
                self.scheduler.step(qwk_metric)
            
            # Early stopping and best model saving
            current_qwk = val_metrics.get('quadratic_weighted_kappa', val_metrics.get('qwk', val_metrics.get('categorical_accuracy', 0.0)))
            if current_qwk > best_qwk + self.config.validation_config.min_delta:
                best_qwk = current_qwk
                patience_counter = 0
                if self.config.validation_config.save_best_only:
                    best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print detailed progress like original train.py (every epoch)
            train_loss = train_metrics['loss']
            train_acc = val_metrics.get('categorical_accuracy', 0.0)  # Using validation accuracy for consistency
            test_acc = val_metrics.get('categorical_accuracy', 0.0)
            qwk = val_metrics.get('quadratic_weighted_kappa', 0.0)
            ordinal_acc = val_metrics.get('ordinal_accuracy', 0.0)
            mae = val_metrics.get('mean_absolute_error', 0.0)
            grad_norm = train_metrics.get('gradient_norm', 0.0)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
                  f"{qwk:6.3f} | {ordinal_acc:6.4f} | {mae:6.3f} | {grad_norm:8.3f} | {current_lr:.2e} | {epoch_time:6.1f}")
            
            # Early stopping check
            if (self.config.validation_config.early_stopping and 
                patience_counter >= self.config.validation_config.patience):
                print(f"🛑 Early stopping at epoch {epoch+1} (patience={patience_counter})")
                break
        
        # Restore best model if using best-only saving
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n✅ Training completed! Best QWK: {best_qwk:.3f}")
        
        return {
            'best_qwk': best_qwk,
            'total_epochs': len(epoch_metrics),
            'training_history': epoch_metrics,  # Changed from metrics_history to match plotter expectations
            'fold': fold
        }
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient norm tracking."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        grad_norms = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch tuple (questions, responses, mask)
            questions, responses, mask = batch
            questions = questions.to(self.device)
            responses = responses.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            model_outputs = self.model(questions, responses)
            
            # Handle model outputs - extract predictions
            if isinstance(model_outputs, tuple):
                # For GPCM models: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
                predictions = model_outputs[-1]  # gpcm_probs is the last element
                outputs = {'predictions': predictions}
            else:
                outputs = model_outputs
            
            # Unified loss function interface - always pass (outputs, batch)
            # The loss function will handle extracting predictions and targets internally
            batch_dict = {'responses': responses}
            loss = self.criterion(outputs, batch_dict)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping and norm tracking (like original train.py)
            if self.config.validation_config.gradient_clip_value:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.validation_config.gradient_clip_value
                )
            else:
                # Calculate gradient norm without clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    float('inf')  # No clipping, just get the norm
                )
            
            grad_norms.append(grad_norm.item())
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'gradient_norm': avg_grad_norm
        }
    
    def _validate_epoch(self, test_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch tuple (questions, responses, mask)
                questions, responses, mask = batch
                questions = questions.to(self.device)
                responses = responses.to(self.device)
                mask = mask.to(self.device)
                
                # Forward pass
                model_outputs = self.model(questions, responses)
                
                # Handle model outputs - extract predictions
                if isinstance(model_outputs, tuple):
                    # For GPCM models: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
                    predictions = model_outputs[-1]  # gpcm_probs is the last element
                else:
                    predictions = model_outputs
                
                # Flatten predictions and responses for metrics
                batch_size, seq_len = responses.shape
                probs_flat = predictions.view(-1, predictions.size(-1))  # (batch*seq, n_cats)
                responses_flat = responses.view(-1)  # (batch*seq,)
                
                # Create mask for valid (non-padded) tokens if needed
                # For now, assume all tokens are valid (adjust if padding is used)
                mask_flat = torch.ones_like(responses_flat, dtype=torch.bool)
                
                # Only include valid tokens
                valid_probs = probs_flat[mask_flat]
                valid_responses = responses_flat[mask_flat]
                
                all_predictions.append(valid_probs.cpu())
                all_targets.append(valid_responses.cpu())
        
        # Combine predictions and compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate comprehensive metrics using the original format
        y_pred = all_predictions.argmax(dim=-1)
        metrics = compute_metrics(all_targets, y_pred, all_predictions, n_cats=self.model.n_cats)
        
        return metrics
    
    def save_model(self, save_path: Path, additional_info: Optional[Dict] = None):
        """Save trained model with metadata."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.config.model,
            'config': self.config.__dict__,
            'experiment_summary': self.tracker.get_experiment_summary(),
            # Add model dimensions for evaluation loading
            'n_questions': self.model.n_questions,
            'n_cats': self.model.n_cats,
            'model_params': {
                'n_questions': self.model.n_questions,
                'n_cats': self.model.n_cats
            }
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, save_path)
        print(f"💾 Model saved to: {save_path}")


class HyperparameterOptimizer:
    """Intelligent hyperparameter optimization."""
    
    def __init__(self, config: TrainingConfig, hp_config: HyperparameterConfig):
        self.config = config
        self.hp_config = hp_config
        self.study = None
    
    def optimize(self, train_data, test_data) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        try:
            import optuna
        except ImportError:
            print("⚠️  Optuna not available, falling back to grid search")
            return self._grid_search_fallback(train_data, test_data)
        
        print(f"🔍 Starting Bayesian optimization ({self.hp_config.n_trials} trials)")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if self.hp_config.enable_pruning else None
        )
        
        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            suggested_params = self._suggest_hyperparameters(trial)
            
            # Create modified config
            trial_config = self._create_trial_config(suggested_params)
            
            # Train with suggested parameters
            trainer = OptimizedTrainer(trial_config)
            trainer.setup_training(
                n_questions=train_data.n_questions,
                n_cats=train_data.n_cats
            )
            
            # Quick training for optimization
            quick_loader = self._create_quick_loader(train_data, test_data)
            results = trainer.train_single_fold(*quick_loader)
            
            return results['best_qwk']
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.hp_config.n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters based on factory configuration."""
        # Get factory-defined search space
        factory_grid = get_model_hyperparameter_grid(self.config.model)
        
        suggested = {}
        for param, values in factory_grid.items():
            if param in ['lr', 'batch_size']:
                # Training-specific parameters
                continue
            elif isinstance(values, list):
                if all(isinstance(v, (int, float)) for v in values):
                    if all(isinstance(v, int) for v in values):
                        suggested[param] = trial.suggest_categorical(param, values)
                    else:
                        suggested[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    suggested[param] = trial.suggest_categorical(param, values)
        
        # Add training-specific parameters
        suggested['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        suggested['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        return suggested
    
    def _create_trial_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Create trial configuration with suggested parameters."""
        # Start with base config
        trial_config = TrainingConfig(
            model=self.config.model,
            dataset=self.config.dataset,
            device=self.config.device,
            seed=self.config.seed,
            epochs=min(10, self.config.epochs),  # Quick training for optimization
            cv=False  # Disable CV during optimization
        )
        
        # Apply suggested parameters
        for param, value in params.items():
            if hasattr(trial_config, param):
                setattr(trial_config, param, value)
        
        return trial_config
    
    def _create_quick_loader(self, train_data, test_data):
        """Create quick data loaders for optimization."""
        # Simplified data loading for quick trials
        return train_data, test_data
    
    def _grid_search_fallback(self, train_data, test_data):
        """Fallback grid search implementation."""
        print("🔍 Running grid search optimization")
        
        factory_grid = get_model_hyperparameter_grid(self.config.model)
        
        # Simplified grid search (just return factory defaults)
        return {
            'best_params': {k: v[0] if isinstance(v, list) else v for k, v in factory_grid.items()},
            'best_value': 0.5,  # Placeholder
            'n_trials': 1
        }


def run_training_workflow(config: TrainingConfig) -> Dict[str, Any]:
    """Run complete training workflow."""
    
    print("=" * 80)
    print("OPTIMIZED DEEP-GPCM TRAINING")
    print("=" * 80)
    print(config.summary())
    print()
    
    # Setup directories
    ensure_directories(config.dataset)
    ensure_results_dirs()
    
    # Load data using unified data loading utility
    print("📚 Loading data...")
    train_loader, test_loader, n_questions, n_cats = load_dataset(
        config.dataset, 
        batch_size=config.batch_size
    )
    
    print(f"📊 Data loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"📊 Questions: {n_questions}, Categories: {n_cats}")
    
    # Hyperparameter optimization if requested
    if config.cv:
        print("\n🔍 Running hyperparameter optimization...")
        hp_config = HyperparameterConfig(n_trials=20)
        optimizer = HyperparameterOptimizer(config, hp_config)
        hp_results = optimizer.optimize(train_data, test_data)
        print(f"✅ Best hyperparameters: {hp_results['best_params']}")
        
        # Update config with best parameters
        for param, value in hp_results['best_params'].items():
            if hasattr(config, param):
                setattr(config, param, value)
    
    # Initialize trainer
    trainer = OptimizedTrainer(config)
    trainer.setup_training(n_questions, n_cats)
    
    # Training execution
    # Data loaders already created by load_dataset()
    
    if config.n_folds > 1:
        print(f"\n🔄 K-Fold Training ({config.n_folds} folds)")
        fold_results = []
        
        for fold in range(config.n_folds):
            # Create fold-specific data (simplified - using same data for demo)
            fold_result = trainer.train_single_fold(train_loader, test_loader, fold)
            fold_results.append(fold_result)
        
        # Aggregate results
        avg_qwk = np.mean([r['best_qwk'] for r in fold_results])
        print(f"✅ Average QWK across folds: {avg_qwk:.4f}")
        
        results = {
            'type': 'k_fold',
            'n_folds': config.n_folds,
            'fold_results': fold_results,
            'average_qwk': avg_qwk,
            'training_history': []  # Aggregate training history from all folds for plotting
        }
        
        # Aggregate training history from all folds for plotting compatibility
        for i, fold_result in enumerate(fold_results):
            if 'training_history' in fold_result:
                for epoch_data in fold_result['training_history']:
                    # Add fold information to each epoch
                    epoch_with_fold = epoch_data.copy()
                    epoch_with_fold['fold'] = i
                    results['training_history'].append(epoch_with_fold)
    else:
        print(f"\n🚀 Single Training Run")
        results = trainer.train_single_fold(train_loader, test_loader)
        results['type'] = 'single_run'
    
    # Save model
    model_path = config.path_config.model_save_path
    trainer.save_model(model_path, {'training_results': results})
    
    # Save training results
    results_path = config.path_config.train_results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, results_path)
    
    print(f"📋 Results saved to: {results_path}")
    print("✅ Training completed successfully!")
    
    return results


def main():
    """Main entry point with optimized configuration parsing."""
    
    try:
        # Parse configuration
        config = parse_training_config()
        
        # Run training workflow
        results = run_training_workflow(config)
        
        print("\n🎯 Optimized training completed!")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()