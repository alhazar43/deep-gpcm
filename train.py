#!/usr/bin/env python3
"""
Unified Training Script for Deep-GPCM Models
Supports both baseline and Deep Integration models with optional k-fold cross-validation.
"""

import os

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn.functional as F
import json
import time
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

from utils.metrics import compute_metrics, save_results, ensure_results_dirs
from utils.path_utils import get_path_manager, ensure_directories
from models.factory import get_all_model_types, get_model_hyperparameter_grid, validate_model_type
from training.losses import create_loss_function
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


def generate_hyperopt_analysis(optimization_results, best_params, analysis_data):
    """Generate automated analysis and recommendations from hyperparameter optimization results."""
    
    # Extract trial data
    all_trials = optimization_results.get('hyperopt_summary', {}).get('all_trials', [])
    if not all_trials:
        return {"⚠️ Error": ["No trial data available for analysis"]}
    
    # Convert to structured format for analysis
    trials_data = []
    for trial in all_trials:
        trial_info = {
            'score': trial.get('score', 0),
            'params': trial.get('params', {}),
            'cv_std': trial.get('cv_std', 0),
            'trial_id': trial.get('trial_id', 0)
        }
        trials_data.append(trial_info)
    
    # Sort by performance
    trials_data.sort(key=lambda x: x['score'], reverse=True)
    best_trial = trials_data[0]
    worst_trial = trials_data[-1]
    
    analysis_results = {}
    
    # 1. Performance Analysis
    performance_analysis = []
    performance_analysis.append(f"Best performance: {best_trial['score']:.4f} QWK (Trial #{best_trial['trial_id']})")
    performance_analysis.append(f"Performance range: {worst_trial['score']:.4f} - {best_trial['score']:.4f} ({(best_trial['score'] - worst_trial['score']):.4f} spread)")
    
    # Stability analysis
    if best_trial['cv_std'] < 0.01:
        stability = "Excellent"
    elif best_trial['cv_std'] < 0.02:
        stability = "Good"
    else:
        stability = "Moderate"
    performance_analysis.append(f"Best model stability: {stability} (CV std: {best_trial['cv_std']:.4f})")
    
    analysis_results["📊 Performance Summary"] = performance_analysis
    
    # 2. Parameter Pattern Analysis
    pattern_analysis = []
    
    # Analyze top 5 vs bottom 5 trials
    top_5 = trials_data[:5]
    bottom_5 = trials_data[-5:]
    
    # Memory size analysis
    top_memory = [trial['params'].get('memory_size', 50) for trial in top_5]
    bottom_memory = [trial['params'].get('memory_size', 50) for trial in bottom_5]
    
    # Convert string memory sizes to int
    top_memory = [int(m) if isinstance(m, str) else m for m in top_memory]
    bottom_memory = [int(m) if isinstance(m, str) else m for m in bottom_memory]
    
    avg_top_memory = np.mean(top_memory)
    avg_bottom_memory = np.mean(bottom_memory)
    
    if avg_top_memory < avg_bottom_memory:
        pattern_analysis.append(f"Smaller memory networks perform better (top-5 avg: {avg_top_memory:.0f} vs bottom-5 avg: {avg_bottom_memory:.0f})")
    else:
        pattern_analysis.append(f"Larger memory networks perform better (top-5 avg: {avg_top_memory:.0f} vs bottom-5 avg: {avg_bottom_memory:.0f})")
    
    # Dropout analysis
    top_dropout = [trial['params'].get('dropout_rate', 0.1) for trial in top_5]
    bottom_dropout = [trial['params'].get('dropout_rate', 0.1) for trial in bottom_5]
    avg_top_dropout = np.mean(top_dropout)
    avg_bottom_dropout = np.mean(bottom_dropout)
    
    if avg_top_dropout < 0.05:
        dropout_rec = "Very light regularization"
    elif avg_top_dropout < 0.1:
        dropout_rec = "Light regularization"
    else:
        dropout_rec = "Moderate regularization"
    pattern_analysis.append(f"Optimal dropout: {dropout_rec} (top-5 avg: {avg_top_dropout:.3f})")
    
    analysis_results["🔍 Parameter Patterns"] = pattern_analysis
    
    # 3. Loss Weight Analysis (if available)
    loss_analysis = []
    if 'ce_weight_logit' in best_params and 'focal_weight_logit' in best_params:
        # Convert logits to weights
        ce_logit = best_params['ce_weight_logit']
        focal_logit = best_params['focal_weight_logit']
        
        # Softmax conversion (approximation)
        ce_weight = np.exp(ce_logit) / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
        focal_weight = np.exp(focal_logit) / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
        qwk_weight = 1 / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
        
        loss_analysis.append(f"Optimal loss combination: CE={ce_weight:.2f}, Focal={focal_weight:.2f}, QWK={qwk_weight:.2f}")
        
        if focal_weight > ce_weight:
            loss_analysis.append("Focal loss dominance suggests class imbalance handling is critical")
        if qwk_weight > 0.2:
            loss_analysis.append("High QWK weight indicates ordinal structure is important")
    
    if loss_analysis:
        analysis_results["⚖️ Loss Function Insights"] = loss_analysis
    
    # 4. Recommendations
    recommendations = []
    
    # Based on parameter importance
    importance = analysis_data.get('parameter_importance', {})
    if importance:
        top_param = max(importance.items(), key=lambda x: x[1])
        if top_param[1] > 50:  # >50% importance
            recommendations.append(f"{top_param[0]} is critical ({top_param[1]:.1f}% importance) - focus optimization here")
    
    # Based on patterns
    if avg_top_memory <= 20:
        recommendations.append("Consider even smaller memory sizes (10-15) to prevent overfitting")
    
    if avg_top_dropout < 0.03:
        recommendations.append("Very low dropout works best - try 0.01-0.05 range")
    
    # Convergence analysis
    convergence_trial = analysis_data.get('convergence_analysis', {}).get('convergence_trial', None)
    if convergence_trial and convergence_trial < len(all_trials) * 0.7:
        recommendations.append(f"Early convergence (trial {convergence_trial}) - could reduce trials for efficiency")
    
    # Search space recommendations
    current_params = set(best_params.keys())
    missing_important = {'embed_dim', 'key_dim', 'value_dim', 'n_heads'} - current_params
    if missing_important:
        recommendations.append(f"Expand search space: add {', '.join(sorted(missing_important))} for better optimization")
    
    analysis_results["🚀 Actionable Recommendations"] = recommendations
    
    # 5. Next Steps
    next_steps = []
    best_score = best_trial['score']
    
    if best_score < 0.65:
        next_steps.append("Performance below 65% QWK - expand architectural parameters (embed_dim, attention params)")
    elif best_score < 0.70:
        next_steps.append("Good performance - fine-tune with extended search space and adaptive epochs")
    else:
        next_steps.append("Excellent performance - focus on transfer learning to other datasets")
    
    next_steps.append("Implement adaptive epoch allocation (5→20→40 epochs based on performance)")
    next_steps.append("Add learning rate scheduling and optimizer parameters to search space")
    
    analysis_results["📋 Next Steps"] = next_steps
    
    return analysis_results


def load_simple_data(train_path, test_path):
    """Simple data loading function."""
    def read_data(file_path):
        sequences = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i + 2 >= len(lines):
                    break
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(int, lines[i+2].strip().split(',')))
                
                # Ensure lengths match
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    # Find number of questions and categories
    all_questions = []
    all_responses = []
    for q, r in train_data + test_data:
        all_questions.extend(q)
        all_responses.extend(r)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats


def pad_sequence_batch(batch):
    """Collate function for padding sequences."""
    questions_batch, responses_batch = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(seq) for seq in questions_batch)
    
    # Pad sequences
    questions_padded = []
    responses_padded = []
    masks = []
    
    for q, r in zip(questions_batch, responses_batch):
        q_len = len(q)
        # Pad questions and responses
        q_pad = q + [0] * (max_len - q_len)
        r_pad = r + [0] * (max_len - q_len)
        mask = [True] * q_len + [False] * (max_len - q_len)
        
        questions_padded.append(q_pad)
        responses_padded.append(r_pad)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks, dtype=torch.bool))


def create_data_loaders(train_data, test_data, batch_size=32):
    """Create data loaders."""
    class SequenceDataset(data_utils.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = SequenceDataset(train_data)
    test_dataset = SequenceDataset(test_data)
    
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_batch
    )
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_batch
    )
    
    return train_loader, test_loader


def create_model(model_type, n_questions, n_cats, device, **model_kwargs):
    """Create model based on type."""
    from models import create_model as factory_create_model
    
    model = factory_create_model(model_type, n_questions, n_cats, **model_kwargs)
    return model.to(device)


# Removed duplicate create_loss_function - using direct import from training.losses for better performance


def get_hyperparameter_grid(model_name):
    """Get hyperparameter grid for cross-validation from factory registry."""
    # Validate model exists in factory
    if not validate_model_type(model_name):
        raise ValueError(f"Model '{model_name}' not found in factory registry")
    
    # Get factory-defined hyperparameter grid
    factory_grid = get_model_hyperparameter_grid(model_name)
    
    # Base grid for all models (training-specific parameters)
    base_grid = {
        'lr': [0.001, 0.005, 0.01],
        'batch_size': [32, 64, 128],
    }
    
    # Merge factory grid with base grid
    base_grid.update(factory_grid)
    
    return base_grid


def train_single_fold(model, train_loader, test_loader, device, epochs, model_name, fold=None, loss_type='ce', loss_kwargs=None):
    """Train a single model (one fold or no CV)."""
    print(f"\\n🚀 TRAINING: {model_name}" + (f" (Fold {fold})" if fold is not None else ""))
    print("-" * 60)
    
    # Fresh model initialization (like old working code)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)
    
    model.apply(init_weights)
    
    # Create loss function
    loss_kwargs = loss_kwargs or {}
    criterion = create_loss_function(loss_type, model.n_cats, **loss_kwargs)
    
    # Get learning rate from loss_kwargs or use default
    learning_rate = loss_kwargs.get('lr', 0.001) if loss_kwargs else 0.001
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    training_history = []
    best_qwk = -1.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
    print(f"Loss function: {loss_type} {loss_kwargs if loss_kwargs else ''}")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | MAE | Grad.Norm | LR | Time(s)")
    print("-" * 95)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        for batch_idx, (questions, responses, mask) in enumerate(train_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with training mode configuration
            # Get logits for training to fix double log-softmax bug
            if hasattr(model, 'get_logits'):
                # Model has separate logits method
                student_abilities, item_thresholds, discrimination_params, gpcm_logits = model.get_logits(questions, responses)
                gpcm_probs = F.softmax(gpcm_logits, dim=-1)  # For metrics computation
            else:
                # Modify model forward to return logits during training
                student_abilities, item_thresholds, discrimination_params, gpcm_output = model(questions, responses)
                if hasattr(model, 'gpcm_layer') and hasattr(model.gpcm_layer, 'forward'):
                    # Get logits from GPCM layer for proper loss computation
                    gpcm_logits = model.gpcm_layer(student_abilities, discrimination_params, item_thresholds, return_logits=True)
                    gpcm_probs = F.softmax(gpcm_logits, dim=-1)
                else:
                    # Fallback: treat output as probabilities (suboptimal but functional)
                    gpcm_probs = gpcm_output
                    gpcm_logits = torch.log(gpcm_probs + 1e-8)
            
            # All loss functions now use standard (logits, targets) interface
            
            # Flatten for loss computation and apply mask
            logits_flat = gpcm_logits.view(-1, gpcm_logits.size(-1))
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1).bool()
            
            # Only compute loss on valid (non-padded) tokens
            valid_logits = logits_flat[mask_flat]
            valid_probs = probs_flat[mask_flat]
            valid_responses = responses_flat[mask_flat]
            
            # Compute loss with proper configuration for IRT
            if loss_type == 'ce':
                # FIXED: Use raw logits with CrossEntropyLoss (no double log-softmax)
                loss = criterion(valid_logits, valid_responses)
            elif loss_type == 'focal':
                # Focal loss expects logits
                loss = criterion(valid_logits, valid_responses)
            elif loss_type in ['qwk', 'ordinal_ce']:
                # These losses work with logits and handle masking internally
                loss = criterion(valid_logits, valid_responses)
            elif loss_type == 'combined':
                # Combined loss handles multiple loss components
                loss = criterion(valid_logits, valid_responses)
            elif loss_type == 'coral':
                # CORAL loss for ordinal classification
                loss = criterion(valid_logits, valid_responses)
            else:
                # Default: use logits for better numerical stability
                loss = criterion(valid_logits, valid_responses)
            
            # Check for NaN loss or invalid probabilities
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"🚨 WARNING: Invalid loss ({loss.item()}) in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Check for NaN in probabilities
            if torch.isnan(valid_probs).any() or torch.isinf(valid_probs).any():
                print(f"🚨 WARNING: Invalid probabilities in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Monitor gradient norms with reasonable clipping  
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            # Skip update if gradients are still too large
            if grad_norm > 50.0:
                print(f"🚨 WARNING: Large gradient norm ({grad_norm:.1f}) in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = valid_probs.argmax(dim=-1)
            correct += (predicted == valid_responses).sum().item()
            total += valid_responses.numel()
        
        # Handle case where no valid batches were processed
        if total == 0:
            print(f"🚨 ERROR: No valid batches in epoch {epoch+1} - all batches had NaN/Inf values")
            print("⚠️  Model training failed - returning early with default metrics")
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Create minimal training history entry to prevent IndexError
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': float('inf'),
                'train_accuracy': 0.0,
                'gradient_norm': float('inf'),
                'learning_rate': current_lr,
                'categorical_accuracy': 0.0,
                'ordinal_accuracy': 0.0,
                'quadratic_weighted_kappa': 0.0,
                'mean_absolute_error': float('inf'),
                'kendall_tau': 0.0,
                'spearman_correlation': 0.0,
                'cohen_kappa': 0.0,
                'cross_entropy': float('inf')
            })
            break
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        # Evaluate
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for questions, responses, mask in test_loader:
                questions = questions.to(device)
                responses = responses.to(device)
                mask = mask.to(device)
                
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                
                # Apply mask filtering to exclude padding tokens
                probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
                responses_flat = responses.view(-1)
                mask_flat = mask.view(-1).bool()
                
                # Only include valid (non-padded) tokens
                valid_probs = probs_flat[mask_flat]
                valid_responses = responses_flat[mask_flat]
                
                all_predictions.append(valid_probs.cpu())
                all_targets.append(valid_responses.cpu())
        
        # Combine predictions and compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate comprehensive metrics using simplified system
        y_pred = all_predictions.argmax(dim=-1)
        eval_metrics = compute_metrics(all_targets, y_pred, all_predictions, n_cats=model.n_cats)
        
        # Extract key metrics for display
        test_acc = eval_metrics.get('categorical_accuracy', 0.0)
        qwk = eval_metrics.get('quadratic_weighted_kappa', 0.0)
        ordinal_acc = eval_metrics.get('ordinal_accuracy', 0.0)
        mae = eval_metrics.get('mean_absolute_error', 0.0)
        
        # Update learning rate
        scheduler.step(qwk)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
              f"{qwk:6.3f} | {ordinal_acc:6.4f} | {mae:6.3f} | {avg_grad_norm:8.3f} | {current_lr:.2e} | {epoch_time:6.1f}")
        
        # Save best model
        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        # Record training history with comprehensive metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'gradient_norm': avg_grad_norm,
            'learning_rate': current_lr
        }
        # Add all evaluation metrics
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                epoch_metrics[key] = value
        
        training_history.append(epoch_metrics)
        
        # Early stopping on gradient explosion
        if avg_grad_norm > 100:
            print(f"\\n🚨 STOPPING: Gradient explosion detected (norm: {avg_grad_norm:.1f})")
            break
    
    print(f"\\n✅ Training completed! Best QWK: {best_qwk:.3f} at epoch {best_epoch}")
    
    # Final metrics from best epoch - handle failed training
    if training_history and best_epoch > 0 and best_epoch <= len(training_history):
        best_epoch_data = training_history[best_epoch-1]
        best_metrics = best_epoch_data.copy()
    else:
        # Training failed completely - create default metrics
        best_metrics = {
            'epoch': 0,
            'train_loss': float('inf'),
            'train_accuracy': 0.0,
            'gradient_norm': float('inf'),
            'learning_rate': 0.001,
            'categorical_accuracy': 0.0,
            'ordinal_accuracy': 0.0,
            'quadratic_weighted_kappa': 0.0,
            'mean_absolute_error': float('inf'),
            'kendall_tau': 0.0,
            'spearman_correlation': 0.0,
            'cohen_kappa': 0.0,
            'cross_entropy': float('inf')
        }
    
    best_metrics.update({
        'best_epoch': best_epoch,
        'total_epochs': len(training_history),
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_type': type(model).__name__
    })
    
    return best_model_state, best_metrics, training_history


def perform_cross_validation(model_name, all_data, n_questions, n_cats, device, args):
    """Perform cross-validation with hyperparameter tuning."""
    from itertools import product
    
    print(f"\n{'='*20} CROSS-VALIDATION WITH HYPERPARAMETER TUNING {'='*20}")
    print(f"Model: {model_name}")
    
    # Get hyperparameter grid
    param_grid = get_hyperparameter_grid(model_name)
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))
    
    print(f"Hyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    print(f"Total combinations: {len(param_combinations)}")
    
    # Outer CV loop
    outer_kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    outer_fold_results = []
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_kfold.split(all_data), 1):
        print(f"\n{'='*15} OUTER FOLD {outer_fold}/{args.n_folds} {'='*15}")
        
        # Split data for outer fold
        outer_train_data = [all_data[i] for i in train_idx]
        outer_test_data = [all_data[i] for i in test_idx]
        
        # Inner CV for hyperparameter selection
        inner_kfold = KFold(n_splits=3, shuffle=True, random_state=args.seed + outer_fold)
        best_params = None
        best_inner_score = -1.0
        
        print("\nHyperparameter search:")
        for param_combo in param_combinations:
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            # Skip if batch_size incompatible with data size
            if params['batch_size'] > len(outer_train_data) // 3:
                continue
            
            inner_scores = []
            
            # Inner CV loop
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(outer_train_data), 1):
                inner_train = [outer_train_data[i] for i in inner_train_idx]
                inner_val = [outer_train_data[i] for i in inner_val_idx]
                
                # Create data loaders with current batch size
                train_loader, val_loader = create_data_loaders(inner_train, inner_val, params['batch_size'])
                
                # Create model with current hyperparameters
                model_kwargs = {k: v for k, v in params.items() if k not in ['lr', 'batch_size']}
                model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
                
                # Train with reduced epochs for hyperparameter search
                search_epochs = min(10, args.epochs)  # Use fewer epochs for search
                _, metrics, _ = train_single_fold(
                    model, train_loader, val_loader, device, search_epochs, 
                    model_name, fold=None, loss_type=args.loss,
                    loss_kwargs={'lr': params['lr']}  # Pass learning rate
                )
                
                inner_scores.append(metrics.get('quadratic_weighted_kappa', 0.0))
            
            # Average inner CV score
            avg_inner_score = np.mean(inner_scores)
            
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_params = params.copy()
                print(f"  New best: {params} -> QWK: {avg_inner_score:.3f}")
        
        print(f"\nBest hyperparameters for outer fold {outer_fold}:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        
        # Train final model for this outer fold with best hyperparameters
        outer_train_loader, outer_test_loader = create_data_loaders(
            outer_train_data, outer_test_data, best_params['batch_size']
        )
        
        model_kwargs = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size']}
        model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
        
        # Full training with best parameters
        best_model_state, best_metrics, training_history = train_single_fold(
            model, outer_train_loader, outer_test_loader, device, args.epochs,
            model_name, fold=outer_fold, loss_type=args.loss,
            loss_kwargs={'lr': best_params['lr']}
        )
        
        # Store results
        outer_fold_results.append({
            'fold': outer_fold,
            'best_params': best_params,
            'metrics': best_metrics,
            'training_history': training_history,
            'model_state': best_model_state
        })
    
    return outer_fold_results


def main():
    # Use unified argument parser with fallback to legacy
    try:
        from utils.args import create_parser, validate_args
        parser = create_parser('train', multi_dataset=True)
        
        # Add train-specific arguments
        train_group = parser.parser.add_argument_group('Training Specific')
        available_models = get_all_model_types()
        train_group.add_argument('--model', choices=available_models,
                                help='Single model to train (from factory registry)')
        train_group.add_argument('--models', nargs='+', choices=available_models,
                                help='Multiple models to train sequentially')
        train_group.add_argument('--loss', type=str, default='factory',
                                help='Loss function type (factory=use model default, ce, focal, qwk, combined)')
        # Legacy arguments for backward compatibility
        train_group.add_argument('--no_cv', action='store_true', 
                                help='Disable k-fold training (deprecated, use --n_folds 0)')
        train_group.add_argument('--cv', action='store_true', 
                                help='Enable basic CV (deprecated, use --hyperopt for advanced optimization)')
        
        # Adaptive hyperparameter optimization arguments
        train_group.add_argument('--adaptive', action='store_true', default=True, 
                                help='Enable adaptive optimization (default: True)')
        train_group.add_argument('--no_adaptive', action='store_true', 
                                help='Disable adaptive optimization features')
        train_group.add_argument('--adaptive_epochs', type=str, default='5,15,40', 
                                help='Adaptive epoch allocation (min,mid,max)')
        train_group.add_argument('--adaptive_arch', action='store_true', default=True, 
                                help='Enable architectural parameter optimization')
        train_group.add_argument('--adaptive_learning', action='store_true', default=True, 
                                help='Enable learning parameter optimization')
        
        args = parser.parse_args()
        validate_args(args, required_fields=['dataset'])
        
        # Map unified args to legacy format
        if hasattr(args, 'learning_rate') and hasattr(args, 'lr') and args.learning_rate != args.lr:
            args.lr = args.learning_rate
            
        # Handle legacy CV flags
        if args.no_cv:
            args.n_folds = 0
        elif args.cv and not hasattr(args, 'hyperopt'):
            args.n_folds = max(args.n_folds, 3)  # Ensure at least 3-fold for basic CV
            
        # Handle adaptive optimization flags
        if args.no_adaptive:
            args.adaptive = False
            args.adaptive_arch = False
            args.adaptive_learning = False
            
    except Exception as e:
        print(f"Warning: Unified parser failed ({e}), falling back to legacy parser")
        
        # Legacy parser as fallback
        available_models = get_all_model_types()
        
        parser = argparse.ArgumentParser(description='Unified Deep-GPCM Training')
        parser.add_argument('--model', choices=available_models, 
                            help='Single model to train (for backward compatibility)')
        parser.add_argument('--models', nargs='+', choices=available_models,
                            help='Multiple models to train sequentially')
        parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for k-fold training (0 = no CV, 1 = single run, >1 = K-fold)')
        parser.add_argument('--no_cv', action='store_true', help='Disable k-fold training (deprecated, use --n_folds 0)')
        parser.add_argument('--cv', action='store_true', help='Enable basic cross-validation (deprecated)')
        parser.add_argument('--hyperopt', action='store_true', help='Enable Bayesian hyperparameter optimization')
        parser.add_argument('--hyperopt_trials', type=int, default=50, help='Number of hyperparameter optimization trials')
        parser.add_argument('--hyperopt_metric', type=str, default='quadratic_weighted_kappa', help='Metric to optimize')
        parser.add_argument('--adaptive', action='store_true', default=True, help='Enable adaptive optimization (default: True)')
        parser.add_argument('--no_adaptive', action='store_true', help='Disable adaptive optimization features')
        parser.add_argument('--adaptive_epochs', type=str, default='5,15,40', help='Adaptive epoch allocation (min,mid,max)')
        parser.add_argument('--adaptive_arch', action='store_true', default=True, help='Enable architectural parameter optimization')
        parser.add_argument('--adaptive_learning', action='store_true', default=True, help='Enable learning parameter optimization')
        parser.add_argument('--loss', type=str, default='factory', help='Loss function override (factory=use model default)')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
        
        args = parser.parse_args()
        
        # Handle legacy CV flags for fallback parser
        if args.no_cv:
            args.n_folds = 0
        elif args.cv and not args.hyperopt:
            args.n_folds = max(args.n_folds, 3)  # Ensure at least 3-fold for basic CV
        
        # Handle adaptive optimization flags
        if args.no_adaptive:
            args.adaptive = False
            args.adaptive_arch = False
            args.adaptive_learning = False
    
    # Get available models from factory registry for validation
    available_models = get_all_model_types()
    
    # Validate model arguments
    if args.model and args.models:
        parser.error("Cannot specify both --model and --models. Use one or the other.")
    if not args.model and not args.models:
        parser.error("Must specify either --model or --models.")
    
    # Determine models to train
    if args.model:
        models_to_train = [args.model]
    else:
        models_to_train = args.models
    
    # Validate all models exist in factory registry
    validated_models = []
    for model in models_to_train:
        if validate_model_type(model):
            validated_models.append(model)
        else:
            print(f"⚠️  Warning: Model '{model}' not found in factory registry, skipping")
    
    models_to_train = validated_models
    if not models_to_train:
        raise ValueError("No valid models found in factory registry")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print(f"UNIFIED DEEP-GPCM TRAINING")
    print("=" * 80)
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    
    # Determine training mode
    if hasattr(args, 'hyperopt') and args.hyperopt:
        print(f"Hyperparameter optimization: {args.hyperopt_trials} trials with {args.n_folds}-fold CV")
        print(f"Optimization metric: {args.hyperopt_metric}")
    elif args.n_folds == 0:
        print(f"Training: Single run (no cross-validation)")
    elif args.n_folds == 1:
        print(f"Training: Single run")
    elif args.cv:
        print(f"Cross-validation: {args.n_folds}-fold with hyperparameter tuning (legacy)")
    else:
        print(f"K-fold training: {args.n_folds}-fold (no hyperparameter tuning)")
    
    print(f"Loss configuration: {'Command-line override' if args.loss != 'factory' else 'Per-model factory settings'}")
    print()
    
    # Create directories
    path_manager = get_path_manager()
    ensure_directories(args.dataset)  # Ensure dataset-specific dirs exist
    # Note: Directory creation now handled by path_manager in new structure
    
    # Load data - support both old and new naming formats
    if args.dataset.startswith('synthetic_') and '_' in args.dataset[10:]:
        # New format: synthetic_4000_200_2
        train_path = f"data/{args.dataset}/{args.dataset}_train.txt"
        test_path = f"data/{args.dataset}/{args.dataset}_test.txt"
    else:
        # Legacy format: synthetic_OC -> synthetic_oc_train.txt
        train_path = f"data/{args.dataset}/{args.dataset.lower()}_train.txt"
        test_path = f"data/{args.dataset}/{args.dataset.lower()}_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"📊 Data loaded: {len(train_data)} train, {len(test_data)} test")
        print(f"Questions: {n_questions}, Categories: {n_cats}")
    except FileNotFoundError:
        print(f"❌ Dataset {args.dataset} not found at {train_path}")
        return
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*20} TRAINING {model_name.upper()} {'='*20}")
        
        # Get factory loss configuration
        from models.factory import get_model_loss_config
        loss_config = get_model_loss_config(model_name)
        
        # Determine loss function - allow command line override
        if hasattr(args, 'loss') and args.loss != 'factory':
            current_loss = args.loss
            print(f"🔧 Using command-line loss override: {current_loss}")
            # Use factory parameters but override type
            factory_loss_kwargs = {}
            for param, value in loss_config.items():
                if param != 'type':
                    factory_loss_kwargs[param] = value
        else:
            # Use factory configuration
            current_loss = loss_config.get('type', 'ce')
            print(f"🔧 Using factory loss configuration: {current_loss}")
            # Use all factory loss parameters
            factory_loss_kwargs = {}
            for param, value in loss_config.items():
                if param != 'type':
                    factory_loss_kwargs[param] = value
        
        # Determine training approach
        if hasattr(args, 'hyperopt') and args.hyperopt:
            # Parse adaptive epoch configuration
            try:
                epoch_parts = args.adaptive_epochs.split(',')
                min_epochs, mid_epochs, max_epochs = map(int, epoch_parts)
            except:
                min_epochs, mid_epochs, max_epochs = 5, 15, 40
                print(f"Warning: Invalid adaptive_epochs format, using defaults: {min_epochs},{mid_epochs},{max_epochs}")
            
            # Use adaptive optimization by default
            if args.adaptive:
                print("🚀 ADAPTIVE Bayesian hyperparameter optimization")
                from optimization.enhanced_adaptive_hyperopt import run_enhanced_hyperopt
                from optimization.adaptive_scheduler import AdaptiveConfig
                
                adaptive_config = AdaptiveConfig(
                    enable_adaptive_epochs=True,
                    enable_architectural_params=args.adaptive_arch,
                    enable_learning_params=args.adaptive_learning,
                    min_epochs=min_epochs,
                    intermediate_epochs=mid_epochs,
                    max_epochs=max_epochs,
                    fallback_on_failure=True,
                    max_consecutive_failures=3
                )
                
                print(f"Adaptive config:")
                print(f"  Epochs: {min_epochs}→{mid_epochs}→{max_epochs}")
                print(f"  Architectural params: {args.adaptive_arch}")
                print(f"  Learning params: {args.adaptive_learning}")
                print(f"  Trials: {args.hyperopt_trials}")
                print(f"  CV folds: {max(args.n_folds, 3)}")
                print(f"Starting enhanced optimization...")
                
                # Run enhanced optimization
                optimization_results = run_enhanced_hyperopt(
                    model_name=model_name,
                    dataset=args.dataset,
                    n_trials=args.hyperopt_trials,
                    metric=args.hyperopt_metric,
                    cv_folds=max(args.n_folds, 3),
                    adaptive_config=adaptive_config
                )
            else:
                print("📊 ORIGINAL Bayesian hyperparameter optimization")
                from optimization.adaptive_hyperopt import HyperparameterOptimizer, HyperoptConfig
                
                # Create optimizer configuration
                config = HyperoptConfig(
                    n_trials=args.hyperopt_trials,
                    metric=args.hyperopt_metric,
                    maximize=True,  # All our metrics are better when higher
                    n_initial_samples=max(10, args.hyperopt_trials // 5),
                    early_stopping_patience=max(10, args.hyperopt_trials // 3),
                    cv_folds=max(args.n_folds, 3),  # Ensure at least 3-fold for hyperopt
                    random_state=args.seed
                )
                
                # Create optimizer
                optimizer = HyperparameterOptimizer(model_name, args.dataset, config)
                
                # Run optimization
                print(f"Starting {args.hyperopt_trials} trials with {config.cv_folds}-fold CV...")
                optimization_results = optimizer.optimize()
            
            # Extract best parameters
            best_params = optimization_results['best_params']
            best_score = optimization_results['best_score']
            
            # Show adaptive features if enabled
            if args.adaptive and 'adaptive_features' in optimization_results:
                features = optimization_results['adaptive_features']
                print(f"Adaptive status: {'✅ Active' if features.get('adaptive_enabled', False) else '⚠️ Fallback'}")
                if features.get('fallback_triggered', False):
                    print(f"Fallback reason: {features.get('fallback_reason', 'Unknown')}")
            
            print(f"\n🏆 Optimization completed!")
            print(f"Best {args.hyperopt_metric}: {best_score:.4f}")
            print(f"Best parameters:")
            for param, value in best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
            
            # Create data loaders with optimized batch size
            final_batch_size = int(best_params.get('batch_size', args.batch_size))
            train_loader, test_loader = create_data_loaders(train_data, test_data, final_batch_size)
            
            # Create model with optimized parameters (filter out training-only parameters)
            model_params = {k: v for k, v in best_params.items() 
                           if k not in ['lr', 'batch_size', 'weight_decay', 'grad_clip', 'label_smoothing'] and not k.endswith('_logit')}
            model = create_model(model_name, n_questions, n_cats, device, **model_params)
            
            # Handle optimized loss weights
            optimized_loss_kwargs = factory_loss_kwargs.copy()
            
            # Override with optimized loss weights if present
            from optimization.adaptive_hyperopt import LossWeightOptimizer
            if any(k.endswith('_logit') for k in best_params):
                loss_config = LossWeightOptimizer.params_to_loss_config(best_params)
                if loss_config:
                    optimized_loss_kwargs.update(loss_config)
                    print(f"Optimized loss weights: ce={loss_config.get('ce_weight', 0.33):.3f}, "
                          f"focal={loss_config.get('focal_weight', 0.33):.3f}, "
                          f"qwk={loss_config.get('qwk_weight', 0.34):.3f}")
            
            # Add optimized learning rate
            optimized_loss_kwargs['lr'] = best_params.get('lr', args.lr)
            
            # Train final model
            print(f"\n🚀 Training final model with optimized hyperparameters...")
            best_model_state, best_metrics, training_history = train_single_fold(
                model, train_loader, test_loader, device, args.epochs, model_name,
                loss_type=current_loss, loss_kwargs=optimized_loss_kwargs
            )
            
            # Save optimized model
            model_path = path_manager.get_model_path(model_name, args.dataset, is_best=True)
            torch.save({
                'model_state_dict': best_model_state,
                'config': {
                    'model_type': model_name,
                    'n_questions': n_questions,
                    'n_cats': n_cats,
                    'dataset': args.dataset,
                    'optimized_hyperparameters': best_params,
                    'optimization_trials': args.hyperopt_trials,
                    'optimization_metric': args.hyperopt_metric
                },
                'metrics': best_metrics,
                'hyperopt_results': {
                    'best_score': best_score,
                    'total_trials': optimization_results['total_trials'],
                    'successful_trials': optimization_results['successful_trials'],
                    'total_optimization_time': optimization_results['total_time']
                }
            }, model_path)
            
            # Save training results with optimization summary
            training_results = {
                'config': {**vars(args), 'optimized_hyperparameters': best_params},
                'metrics': best_metrics,
                'training_history': training_history,
                'hyperopt_summary': optimization_results
            }
            log_path = path_manager.get_result_path('train', model_name, args.dataset)
            save_results(training_results, str(log_path))
            
            # Generate comprehensive hyperparameter optimization analysis and visualizations
            print(f"\n📊 Generating hyperparameter optimization analysis...")
            try:
                from optimization.hyperopt_visualization import create_hyperopt_visualizer
                
                visualizer = create_hyperopt_visualizer()
                analysis = visualizer.analyze_optimization_results(
                    optimization_results, model_name, args.dataset
                )
                
                print(f"📈 Optimization analysis completed!")
                print(f"   Best parameters: {len(best_params)} parameters optimized")
                print(f"   Parameter importance: {len(analysis.get('parameter_importance', {}))} parameters analyzed")
                print(f"   Convergence: Trial {analysis['convergence_analysis'].get('convergence_trial', 'N/A')}")
                print(f"   Visualizations: {len(analysis.get('visualization_paths', {}))} plots generated")
                print(f"   📋 Full report: {analysis.get('report_path', 'N/A')}")
                
                # Display top 3 most important parameters
                importance = analysis.get('parameter_importance', {})
                if importance:
                    top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"   🎯 Top parameters: {', '.join([f'{p}({v:.1f}%)' for p, v in top_params])}")
                
                # Generate automated analysis and recommendations
                try:
                    automated_analysis = generate_hyperopt_analysis(optimization_results, best_params, analysis)
                    print(f"\n🤖 AUTOMATED ANALYSIS & RECOMMENDATIONS")
                    print("=" * 60)
                    for section, content in automated_analysis.items():
                        print(f"\n{section}:")
                        for item in content:
                            print(f"  • {item}")
                except Exception as analysis_error:
                    print(f"⚠️  Automated analysis failed: {analysis_error}")
                
            except Exception as e:
                print(f"⚠️  Warning: Hyperopt visualization failed: {e}")
            
            print(f"💾 Optimized model saved to: {model_path}")
            print(f"📋 Training logs saved to: {log_path}")
            
        elif args.n_folds <= 1:
            print("📈 Single training (no cross-validation)")
            
            # Create data loaders
            train_loader, test_loader = create_data_loaders(train_data, test_data, args.batch_size)
            
            # Create and train model
            # Prepare model kwargs
            model_kwargs = {}
                
            
            model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
            
            # Detect and display actual model features
            print(f"\n🔍 MODEL CONFIGURATION:")
            print(f"  - Model type: {type(model).__name__}")
            print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Detect adaptive blending
            if hasattr(model, 'enable_adaptive_blending'):
                if model.enable_adaptive_blending:
                    print(f"  ⚡ Adaptive blending: ENABLED")
                    if hasattr(model, 'threshold_blender') and model.threshold_blender is not None:
                        blender_type = type(model.threshold_blender).__name__
                        print(f"     Blender: {blender_type}")
                        # Handle different blender types with different parameter names
                        if hasattr(model.threshold_blender, 'range_sensitivity'):
                            print(f"     Range sensitivity: {model.threshold_blender.range_sensitivity.item():.3f}")
                            print(f"     Distance sensitivity: {model.threshold_blender.distance_sensitivity.item():.3f}")
                            print(f"     Baseline bias: {model.threshold_blender.baseline_bias.item():.3f}")
                        elif hasattr(model.threshold_blender, 'base_sensitivity'):
                            print(f"     Base sensitivity: {model.threshold_blender.base_sensitivity.item():.3f}")
                            print(f"     Distance threshold: {model.threshold_blender.distance_threshold.item():.3f}")
                        else:
                            print("     Parameters: (blender-specific)")
                else:
                    print(f"  ⚡ Adaptive blending: DISABLED")
            
            # Detect threshold coupling
            if hasattr(model, 'enable_threshold_coupling'):
                if model.enable_threshold_coupling:
                    print(f"  🔗 Threshold coupling: ENABLED")
                else:
                    print(f"  🔗 Threshold coupling: DISABLED")
            
            # Detect CORAL integration
            if hasattr(model, 'coral_layer'):
                print(f"  🎯 CORAL layer: PRESENT")
            
            print()
            
            # Use factory loss configuration for this model
            best_model_state, best_metrics, training_history = train_single_fold(
                model, train_loader, test_loader, device, args.epochs, model_name,
                loss_type=current_loss, loss_kwargs=factory_loss_kwargs
            )
            
            # Save model using new path structure
            model_path = path_manager.get_model_path(model_name, args.dataset, is_best=True)
            torch.save({
                'model_state_dict': best_model_state,
                'config': {
                    'model_type': model_name,
                    'n_questions': n_questions,
                    'n_cats': n_cats,
                    'dataset': args.dataset
                },
                'metrics': best_metrics
            }, model_path)
        
            # Save training results using new path structure
            training_results = {
                'config': vars(args),
                'metrics': best_metrics,
                'training_history': training_history
            }
            log_path = path_manager.get_result_path('train', model_name, args.dataset)
            save_results(training_results, str(log_path))
            
            print(f"💾 Model saved to: {model_path}")
            print(f"📋 Logs saved to: {log_path}")
        
        elif args.cv:
            # Cross-validation with hyperparameter tuning
            print(f"📈 {args.n_folds}-fold cross-validation with hyperparameter tuning")
            
            # Combine all data for CV
            all_data = train_data + test_data
            
            # Perform cross-validation
            cv_results = perform_cross_validation(model_name, all_data, n_questions, n_cats, device, args)
            
            # Compute and display CV summary
            print(f"\n{'='*20} CV SUMMARY {'='*20}")
            
            metrics_names = [
                'categorical_accuracy',
                'ordinal_accuracy', 
                'quadratic_weighted_kappa',
                'mean_absolute_error',
                'kendall_tau',
                'spearman_correlation',
                'cohen_kappa',
                'cross_entropy'
            ]
            cv_summary = {}
            
            for metric in metrics_names:
                values = [fold['metrics'][metric] for fold in cv_results]
                cv_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            
            # Also summarize best hyperparameters
            hyperparam_summary = {}
            for param in cv_results[0]['best_params'].keys():
                param_values = [fold['best_params'][param] for fold in cv_results]
                # Check if all values are the same
                if len(set(param_values)) == 1:
                    hyperparam_summary[param] = param_values[0]
                else:
                    hyperparam_summary[param] = {
                        'values': param_values,
                        'most_common': max(set(param_values), key=param_values.count)
                    }
            
            print(f"{'Metric':<25} {'Mean':<8} {'Std':<8} {'Folds'}")
            print("-" * 60)
            for metric, stats in cv_summary.items():
                fold_str = " ".join([f"{v:.3f}" for v in stats['values']])
                print(f"{metric:<25} {stats['mean']:<8.3f} {stats['std']:<8.3f} [{fold_str}]")
            
            print(f"\nBest hyperparameters across folds:")
            for param, value in hyperparam_summary.items():
                if isinstance(value, dict):
                    print(f"  {param}: varies by fold - most common: {value['most_common']}")
                else:
                    print(f"  {param}: {value}")
            
            # Save CV results
            cv_summary_results = {
                'config': vars(args),
                'cv_summary': cv_summary,
                'hyperparameter_summary': hyperparam_summary,
                'fold_results': [{
                    'fold': r['fold'],
                    'best_params': r['best_params'],
                    'metrics': r['metrics']
                } for r in cv_results]
            }
            cv_log_path = path_manager.get_result_path('validation', model_name, args.dataset)
            save_results(cv_summary_results, str(cv_log_path))
            
            print(f"\n📋 CV summary saved to: {cv_log_path}")
            
            # Find and save best fold model as main model
            best_fold_idx = np.argmax([fold['metrics']['quadratic_weighted_kappa'] for fold in cv_results])
            best_fold = cv_results[best_fold_idx]['fold']
            best_model_state = cv_results[best_fold_idx]['model_state']
            
            # Save best model
            main_model_path = path_manager.get_model_path(model_name, args.dataset, is_best=True)
            torch.save({
                'model_state_dict': best_model_state,
                'config': {
                    'model_type': model_name,
                    'n_questions': n_questions,
                    'n_cats': n_cats,
                    'dataset': args.dataset,
                    'best_hyperparameters': cv_results[best_fold_idx]['best_params']
                },
                'metrics': cv_results[best_fold_idx]['metrics']
            }, main_model_path)
            
            print(f"\n🏆 Best fold: {best_fold} (QWK: {cv_results[best_fold_idx]['metrics']['quadratic_weighted_kappa']:.3f})")
            print(f"💾 Best model saved to: {main_model_path}")
        
        elif args.n_folds > 1:
            # Standard k-fold training (no hyperparameter tuning)
            print(f"📈 {args.n_folds}-fold training (no hyperparameter tuning)")
        
            # Combine all data for k-fold splitting
            all_data = train_data + test_data
            all_indices = np.arange(len(all_data))
            
            kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(all_indices), 1):
                print(f"\\n{'='*20} FOLD {fold}/{args.n_folds} {'='*20}")
                
                # Split data for this fold
                fold_train_data = [all_data[i] for i in train_idx]
                fold_test_data = [all_data[i] for i in test_idx]
                
                # Create data loaders
                train_loader, test_loader = create_data_loaders(fold_train_data, fold_test_data, args.batch_size)
                
                # Create and train model
                # Prepare model kwargs
                model_kwargs = {}
                    
                
                model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
                
                # Detect and display actual model features (first fold only)
                if fold == 1:
                    print(f"\n🔍 MODEL CONFIGURATION:")
                    print(f"  - Model type: {type(model).__name__}")
                    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
                    
                    # Detect adaptive blending
                    if hasattr(model, 'enable_adaptive_blending'):
                        if model.enable_adaptive_blending:
                            print(f"  ⚡ Adaptive blending: ENABLED")
                            if hasattr(model, 'threshold_blender') and model.threshold_blender is not None:
                                blender_type = type(model.threshold_blender).__name__
                                print(f"     Blender: {blender_type}")
                                # Handle different blender types with different parameter names
                                if hasattr(model.threshold_blender, 'range_sensitivity'):
                                    print(f"     Range sensitivity: {model.threshold_blender.range_sensitivity.item():.3f}")
                                    print(f"     Distance sensitivity: {model.threshold_blender.distance_sensitivity.item():.3f}")
                                    print(f"     Baseline bias: {model.threshold_blender.baseline_bias.item():.3f}")
                                elif hasattr(model.threshold_blender, 'base_sensitivity'):
                                    print(f"     Base sensitivity: {model.threshold_blender.base_sensitivity.item():.3f}")
                                    print(f"     Distance threshold: {model.threshold_blender.distance_threshold.item():.3f}")
                                else:
                                    print("     Parameters: (blender-specific)")
                        else:
                            print(f"  ⚡ Adaptive blending: DISABLED")
                    
                    # Detect threshold coupling
                    if hasattr(model, 'enable_threshold_coupling'):
                        if model.enable_threshold_coupling:
                            print(f"  🔗 Threshold coupling: ENABLED")
                        else:
                            print(f"  🔗 Threshold coupling: DISABLED")
                    
                    # Detect CORAL integration
                    if hasattr(model, 'coral_layer'):
                        print(f"  🎯 CORAL layer: PRESENT")
                    
                    print()
            
                # Use factory loss configuration for this model
                best_model_state, best_metrics, training_history = train_single_fold(
                    model, train_loader, test_loader, device, args.epochs, model_name, fold,
                    loss_type=current_loss, loss_kwargs=factory_loss_kwargs
                )
            
                # Save fold results
                fold_results.append({
                    'fold': fold,
                    'metrics': best_metrics,
                    'training_history': training_history
                })
                
                # Save model for this fold using new path structure
                model_path = path_manager.get_model_path(model_name, args.dataset, fold=fold, is_best=False)
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': {
                        'model_type': model_name,
                        'n_questions': n_questions,
                        'n_cats': n_cats,
                        'dataset': args.dataset,
                        'fold': fold
                    },
                    'metrics': best_metrics
                }, model_path)
                
                # Save fold logs using new path structure
                fold_training_results = {
                    'config': {**vars(args), 'fold': fold},
                    'metrics': best_metrics,
                    'training_history': training_history
                }
                log_path = path_manager.get_result_path('train', model_name, args.dataset, fold=fold)
                save_results(fold_training_results, str(log_path))
                
        else:
            # n_folds == 0: No cross-validation, just single train/test split
            print("📈 Single train/test split (no cross-validation)")
            print("✅ Training completed successfully!")
        
        if not args.no_cv and args.n_folds > 1 and not args.cv and not (hasattr(args, 'hyperopt') and args.hyperopt):
            # Compute and display k-fold summary
            print(f"\\n{'='*20} K-FOLD SUMMARY {'='*20}")
            
            metrics_names = [
                'categorical_accuracy',
                'ordinal_accuracy', 
                'quadratic_weighted_kappa',
                'mean_absolute_error',
                'kendall_tau',
                'spearman_correlation',
                'cohen_kappa',
                'cross_entropy'
            ]
            cv_summary = {}
            
            for metric in metrics_names:
                values = [fold['metrics'][metric] for fold in fold_results]
                cv_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            
            print(f"{'Metric':<25} {'Mean':<8} {'Std':<8} {'Folds'}")
            print("-" * 60)
            for metric, stats in cv_summary.items():
                fold_str = " ".join([f"{v:.3f}" for v in stats['values']])
                print(f"{metric:<25} {stats['mean']:<8.3f} {stats['std']:<8.3f} [{fold_str}]")
            
            # Save CV summary using new path structure
            cv_summary_results = {
                'config': vars(args),
                'cv_summary': cv_summary,
                'fold_results': fold_results
            }
            cv_log_path = path_manager.get_result_path('validation', model_name, args.dataset)
            save_results(cv_summary_results, str(cv_log_path))
            
            print(f"\\n📋 CV summary saved to: {cv_log_path}")
            
            # Find and save best fold model as main model
            best_fold_idx = np.argmax([fold['metrics']['quadratic_weighted_kappa'] for fold in fold_results])
            best_fold = fold_results[best_fold_idx]['fold']
            
            best_fold_model_path = path_manager.get_model_path(model_name, args.dataset, fold=best_fold, is_best=False)
            main_model_path = path_manager.get_model_path(model_name, args.dataset, is_best=True)
            
            # Copy best fold model as main model
            import shutil
            if best_fold_model_path.exists():
                shutil.copy2(best_fold_model_path, main_model_path)
            else:
                print(f"⚠️  Warning: Best fold model not found at {best_fold_model_path}")
                print("   Using last saved model instead")
            
            print(f"\\n🏆 Best fold: {best_fold} (QWK: {fold_results[best_fold_idx]['metrics']['quadratic_weighted_kappa']:.3f})")
            print(f"💾 Best model copied to: {main_model_path}")
    
    print("\\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()