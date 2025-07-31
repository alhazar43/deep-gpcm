#!/usr/bin/env python3
"""
Unified Training Script for Deep-GPCM Models
Supports both baseline and Deep Integration models with optional k-fold cross-validation.
"""

import os

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import json
import time
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

from utils.metrics import compute_metrics, save_results, ensure_results_dirs
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


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
    from core import create_model as factory_create_model
    
    model = factory_create_model(model_type, n_questions, n_cats, **model_kwargs)
    return model.to(device)


def create_loss_function(loss_type, n_cats, **kwargs):
    """Create loss function based on type."""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'qwk':
        from training.ordinal_losses import DifferentiableQWKLoss
        return DifferentiableQWKLoss(n_cats)
    elif loss_type == 'emd':
        from training.ordinal_losses import OrdinalEMDLoss
        return OrdinalEMDLoss(n_cats)
    elif loss_type == 'ordinal_ce':
        from training.ordinal_losses import OrdinalCrossEntropyLoss
        return OrdinalCrossEntropyLoss(n_cats, alpha=kwargs.get('ordinal_alpha', 1.0))
    elif loss_type == 'combined':
        from training.ordinal_losses import CombinedOrdinalLoss
        return CombinedOrdinalLoss(
            n_cats,
            ce_weight=kwargs.get('ce_weight', 1.0),
            qwk_weight=kwargs.get('qwk_weight', 0.5),
            emd_weight=kwargs.get('emd_weight', 0.0),
            coral_weight=kwargs.get('coral_weight', 0.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_single_fold(model, train_loader, test_loader, device, epochs, model_name, fold=None, loss_type='ce', loss_kwargs=None):
    """Train a single model (one fold or no CV)."""
    print(f"\\n🚀 TRAINING: {model_name}" + (f" (Fold {fold})" if fold is not None else ""))
    print("-" * 60)
    
    # Fresh model initialization
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
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    training_history = []
    best_qwk = -1.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
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
            
            # Forward pass
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            # Flatten for loss computation and apply mask
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1).bool()
            
            # Only compute loss on valid (non-padded) tokens
            valid_probs = probs_flat[mask_flat]
            valid_responses = responses_flat[mask_flat]
            
            # Compute loss only on valid tokens
            if loss_type == 'ce':
                # CrossEntropyLoss expects log probabilities
                valid_log_probs = torch.log(valid_probs + 1e-8)
                loss = criterion(valid_log_probs, valid_responses)
            elif loss_type in ['qwk', 'emd', 'ordinal_ce']:
                # These losses work with probabilities directly
                # Reshape back to include sequence dimension for proper masking
                loss = criterion(gpcm_probs, responses, mask)
            elif loss_type == 'combined':
                # Combined loss handles both probabilities and logits
                loss_dict = criterion(gpcm_probs, responses, mask)
                loss = loss_dict['total_loss']
            else:
                loss = criterion(valid_probs, valid_responses)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"🚨 WARNING: NaN loss in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Monitor gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = valid_probs.argmax(dim=-1)
            correct += (predicted == valid_responses).sum().item()
            total += valid_responses.numel()
        
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
    
    # Final metrics from best epoch
    best_epoch_data = training_history[best_epoch-1]
    best_metrics = best_epoch_data.copy()
    best_metrics.update({
        'best_epoch': best_epoch,
        'total_epochs': len(training_history),
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_type': type(model).__name__
    })
    
    return best_model_state, best_metrics, training_history


def main():
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Training')
    parser.add_argument('--model', choices=['deep_gpcm', 'attn_gpcm', 'coral', 'coral_gpcm'], 
                        help='Single model to train (for backward compatibility)')
    parser.add_argument('--models', nargs='+', choices=['deep_gpcm', 'attn_gpcm', 'coral', 'coral_gpcm'],
                        help='Multiple models to train sequentially')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds (0 = no CV)')
    parser.add_argument('--no_cv', action='store_true', help='Disable cross-validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'qwk', 'emd', 'ordinal_ce', 'combined'],
                        help='Loss function type (default: ce)')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Weight for CE loss in combined loss')
    parser.add_argument('--qwk_weight', type=float, default=0.5,
                        help='Weight for QWK loss in combined loss')
    parser.add_argument('--emd_weight', type=float, default=0.0,
                        help='Weight for EMD loss in combined loss')
    parser.add_argument('--coral_weight', type=float, default=0.0,
                        help='Weight for CORAL loss in combined loss')
    parser.add_argument('--ordinal_alpha', type=float, default=1.0,
                        help='Alpha parameter for ordinal CE loss')
    
    # Threshold coupling arguments
    parser.add_argument('--enable_threshold_coupling', action='store_true',
                        help='Enable threshold coupling for CORAL models')
    parser.add_argument('--coupling_type', type=str, default='linear',
                        choices=['linear'], help='Type of threshold coupling')
    parser.add_argument('--threshold_gpcm_weight', type=float, default=0.7,
                        help='Weight for GPCM thresholds in coupling')
    parser.add_argument('--threshold_coral_weight', type=float, default=0.3,
                        help='Weight for CORAL thresholds in coupling')
    
    args = parser.parse_args()
    
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
    print(f"Cross-validation: {'Disabled' if args.no_cv else f'{args.n_folds}-fold'}")
    print(f"Loss function: {args.loss}")
    if args.loss == 'combined':
        print(f"  - CE weight: {args.ce_weight}")
        print(f"  - QWK weight: {args.qwk_weight}")
        print(f"  - EMD weight: {args.emd_weight}")
        print(f"  - CORAL weight: {args.coral_weight}")
    elif args.loss == 'ordinal_ce':
        print(f"  - Ordinal alpha: {args.ordinal_alpha}")
    print()
    
    # Create directories
    os.makedirs('save_models', exist_ok=True)
    ensure_results_dirs()
    
    # Load data
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
        
        # Cross-validation or single training
        if args.no_cv or args.n_folds == 0:
            print("📈 Single training (no cross-validation)")
            
            # Create data loaders
            train_loader, test_loader = create_data_loaders(train_data, test_data, args.batch_size)
            
            # Create and train model
            # Prepare model kwargs for threshold coupling
            model_kwargs = {}
            if model_name in ['coral', 'coral_gpcm']:
                model_kwargs.update({
                    'enable_threshold_coupling': args.enable_threshold_coupling or (model_name == 'coral_gpcm'),
                    'coupling_type': args.coupling_type,
                    'gpcm_weight': args.threshold_gpcm_weight,
                    'coral_weight': args.threshold_coral_weight
                })
            
            model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
            
            # Prepare loss kwargs
            loss_kwargs = {
                'ce_weight': args.ce_weight,
                'qwk_weight': args.qwk_weight,
                'emd_weight': args.emd_weight,
                'coral_weight': args.coral_weight,
                'ordinal_alpha': args.ordinal_alpha
            }
            
            best_model_state, best_metrics, training_history = train_single_fold(
                model, train_loader, test_loader, device, args.epochs, model_name,
                loss_type=args.loss, loss_kwargs=loss_kwargs
            )
            
            # Save model and logs
            model_path = f"save_models/best_{model_name}_{args.dataset}.pth"
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
        
            # Save training results using simplified system
            training_results = {
                'config': vars(args),
                'metrics': best_metrics,
                'training_history': training_history
            }
            log_path = save_results(
                training_results, 
                f"results/train/train_results_{model_name}_{args.dataset}.json"
            )
            
            print(f"💾 Model saved to: {model_path}")
            print(f"📋 Logs saved to: {log_path}")
        
        else:
            print(f"📈 {args.n_folds}-fold cross-validation")
        
            # Combine all data for CV splitting
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
                # Prepare model kwargs for threshold coupling
                model_kwargs = {}
                if model_name in ['coral', 'coral_gpcm']:
                    model_kwargs.update({
                        'enable_threshold_coupling': args.enable_threshold_coupling or (model_name == 'coral_gpcm'),
                        'coupling_type': args.coupling_type,
                        'gpcm_weight': args.threshold_gpcm_weight,
                        'coral_weight': args.threshold_coral_weight
                    })
                
                model = create_model(model_name, n_questions, n_cats, device, **model_kwargs)
            
                # Prepare loss kwargs (same as above)
                loss_kwargs = {
                    'ce_weight': args.ce_weight,
                    'qwk_weight': args.qwk_weight,
                    'emd_weight': args.emd_weight,
                    'coral_weight': args.coral_weight,
                    'ordinal_alpha': args.ordinal_alpha
                }
                
                best_model_state, best_metrics, training_history = train_single_fold(
                    model, train_loader, test_loader, device, args.epochs, model_name, fold,
                    loss_type=args.loss, loss_kwargs=loss_kwargs
                )
            
                # Save fold results
                fold_results.append({
                    'fold': fold,
                    'metrics': best_metrics,
                    'training_history': training_history
                })
                
                # Save model for this fold
                model_path = f"save_models/best_{model_name}_{args.dataset}_fold_{fold}.pth"
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
            
            # Save fold logs using simplified system
            fold_training_results = {
                'config': {**vars(args), 'fold': fold},
                'metrics': best_metrics,
                'training_history': training_history
            }
            log_path = save_results(
                fold_training_results,
                f"results/train/train_results_{model_name}_{args.dataset}_fold_{fold}.json"
            )
        
        if args.n_folds > 0:
            # Compute and display CV summary
            print(f"\\n{'='*20} CV SUMMARY {'='*20}")
            
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
            
            # Save CV summary using simplified system
            cv_summary_results = {
                'config': vars(args),
                'cv_summary': cv_summary,
                'fold_results': fold_results
            }
            cv_log_path = save_results(
                cv_summary_results,
                f"results/train/train_results_{model_name}_{args.dataset}_cv_summary.json"
            )
            
            print(f"\\n📋 CV summary saved to: {cv_log_path}")
            
            # Find and save best fold model as main model
            best_fold_idx = np.argmax([fold['metrics']['quadratic_weighted_kappa'] for fold in fold_results])
            best_fold = fold_results[best_fold_idx]['fold']
            
            best_fold_model_path = f"save_models/best_{model_name}_{args.dataset}_fold_{best_fold}.pth"
            main_model_path = f"save_models/best_{model_name}_{args.dataset}.pth"
            
            # Copy best fold model as main model
            import shutil
            shutil.copy2(best_fold_model_path, main_model_path)
            
            print(f"\\n🏆 Best fold: {best_fold} (QWK: {fold_results[best_fold_idx]['metrics']['quadratic_weighted_kappa']:.3f})")
            print(f"💾 Best model copied to: {main_model_path}")
    
    print("\\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()