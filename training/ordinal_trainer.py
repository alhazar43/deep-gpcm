"""
Training pipeline for ordinal-aware attention mechanisms.
Provides systematic training and comparison of different attention types.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import models and components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations.attention_gpcm import AttentionGPCM
from models.losses.ordinal_losses import create_ordinal_loss
from models.metrics.ordinal_metrics import MetricsTracker, calculate_ordinal_improvement


class OrdinalTrainer:
    """Trainer for ordinal-aware attention models."""
    
    def __init__(self, n_questions: int = 50, n_cats: int = 4, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.device = device
        
        # Training configurations
        self.configs = {
            'baseline': {
                'use_ordinal_attention': False,
                'attention_types': None,
                'loss_type': 'standard_ce'
            },
            'ordinal_aware': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware'],
                'loss_type': 'ordinal_ce'
            },
            'qwk_optimized': {
                'use_ordinal_attention': True,
                'attention_types': ['qwk_aligned'],
                'loss_type': 'combined'
            },
            'combined': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned'],
                'loss_type': 'combined'
            }
        }
        
        self.results = {}
    
    def create_synthetic_dataset(self, n_samples: int = 1000, seq_len: int = 15, 
                               validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create synthetic dataset with ordinal structure."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate student abilities
        abilities = np.random.normal(0, 1, n_samples)
        
        # Generate question difficulties
        difficulties = np.random.normal(0, 1, self.n_questions)
        
        all_questions = []
        all_responses = []
        
        for i in range(n_samples):
            student_ability = abilities[i]
            student_questions = []
            student_responses = []
            
            # Random question selection
            question_indices = np.random.choice(self.n_questions, seq_len, replace=True)
            
            for q_idx in question_indices:
                # GPCM-like response generation
                difficulty = difficulties[q_idx]
                
                # Create threshold parameters for ordinal categories
                base_difficulty = difficulty
                thresholds = np.linspace(base_difficulty - 1.5, base_difficulty + 1.5, self.n_cats)
                
                # Calculate cumulative probabilities
                cum_probs = []
                for k in range(self.n_cats):
                    # IRT-like probability
                    if k == 0:
                        prob = 1.0  # Everyone can achieve category 0
                    else:
                        prob = 1.0 / (1.0 + np.exp(-1.7 * (student_ability - thresholds[k-1])))
                    cum_probs.append(prob)
                
                # Convert to category probabilities
                cat_probs = []
                for k in range(self.n_cats):
                    if k == 0:
                        cat_probs.append(1.0 - cum_probs[1] if len(cum_probs) > 1 else 1.0)
                    elif k == self.n_cats - 1:
                        cat_probs.append(cum_probs[k])
                    else:
                        cat_probs.append(cum_probs[k] - cum_probs[k+1])
                
                # Ensure valid probabilities
                cat_probs = np.array(cat_probs)
                cat_probs = np.abs(cat_probs)  # Ensure positive
                cat_probs = cat_probs / cat_probs.sum()  # Normalize
                
                # Sample response
                response = np.random.choice(self.n_cats, p=cat_probs)
                
                student_questions.append(q_idx + 1)  # 1-indexed
                student_responses.append(response)
            
            all_questions.append(student_questions)
            all_responses.append(student_responses)
        
        # Convert to tensors
        questions = torch.tensor(all_questions, dtype=torch.long)
        responses = torch.tensor(all_responses, dtype=torch.long)
        
        # Train/validation split
        n_train = int(n_samples * (1 - validation_split))
        
        train_dataset = TensorDataset(questions[:n_train], responses[:n_train])
        val_dataset = TensorDataset(questions[n_train:], responses[n_train:])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model_and_loss(self, config_name: str) -> Tuple[AttentionGPCM, nn.Module]:
        """Create model and loss function based on configuration."""
        config = self.configs[config_name]
        
        # Create model
        model = AttentionGPCM(
            n_questions=self.n_questions,
            n_cats=self.n_cats,
            embed_dim=64,
            memory_size=30,
            key_dim=50,
            value_dim=128,
            final_fc_dim=64,
            n_heads=8,
            n_cycles=2,
            dropout_rate=0.2,
            use_ordinal_attention=config['use_ordinal_attention'],
            attention_types=config['attention_types']
        ).to(self.device)
        
        # Create loss function
        loss_type = config['loss_type']
        if loss_type == 'standard_ce':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = create_ordinal_loss(loss_type, self.n_cats)
        
        return model, loss_fn
    
    def train_epoch(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                   train_loader: DataLoader, metrics_tracker: MetricsTracker, 
                   epoch: int, use_combined_loss: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for batch_idx, (questions, responses) in enumerate(train_loader):
            questions, responses = questions.to(self.device), responses.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, _, _, probs = model(questions, responses)
            
            # Calculate loss
            if use_combined_loss:
                logits = torch.log(probs + 1e-8)
                loss, loss_dict = loss_fn(logits, responses)
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                # Standard cross-entropy expects logits
                logits = torch.log(probs + 1e-8)
                loss = loss_fn(logits.view(-1, self.n_cats), responses.view(-1))
            else:
                # Ordinal losses expect logits
                logits = torch.log(probs + 1e-8)
                loss = loss_fn(logits, responses)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = probs.argmax(dim=-1)
            all_preds.append(preds.detach())
            all_targets.append(responses.detach())
            all_probs.append(probs.detach())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        
        train_metrics = metrics_tracker.update(all_targets, all_preds, all_probs, split='train')
        
        return {
            'loss': total_loss / len(train_loader),
            'metrics': train_metrics
        }
    
    def validate_epoch(self, model: nn.Module, loss_fn: nn.Module, val_loader: DataLoader,
                      metrics_tracker: MetricsTracker, use_combined_loss: bool = False) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for questions, responses in val_loader:
                questions, responses = questions.to(self.device), responses.to(self.device)
                
                # Forward pass
                _, _, _, probs = model(questions, responses)
                
                # Calculate loss
                if use_combined_loss:
                    logits = torch.log(probs + 1e-8)
                    loss, _ = loss_fn(logits, responses)
                elif isinstance(loss_fn, nn.CrossEntropyLoss):
                    logits = torch.log(probs + 1e-8)
                    loss = loss_fn(logits.view(-1, self.n_cats), responses.view(-1))
                else:
                    logits = torch.log(probs + 1e-8)
                    loss = loss_fn(logits, responses)
                
                total_loss += loss.item()
                
                # Collect predictions
                preds = probs.argmax(dim=-1)
                all_preds.append(preds)
                all_targets.append(responses)
                all_probs.append(probs)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        
        val_metrics = metrics_tracker.update(all_targets, all_preds, all_probs, split='val')
        
        return {
            'loss': total_loss / len(val_loader),
            'metrics': val_metrics
        }
    
    def train_model(self, config_name: str, train_loader: DataLoader, val_loader: DataLoader,
                   n_epochs: int = 20, lr: float = 0.001, verbose: bool = True) -> Dict:
        """Train a single model configuration."""
        if verbose:
            print(f"\nTraining {config_name} model...")
        
        # Create model and loss
        model, loss_fn = self.create_model_and_loss(config_name)
        use_combined_loss = self.configs[config_name]['loss_type'] == 'combined'
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5)
        
        # Metrics tracker
        metrics_tracker = MetricsTracker(self.n_cats)
        
        # Training loop
        best_qwk = -1.0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Train
            train_results = self.train_epoch(model, loss_fn, optimizer, train_loader, 
                                           metrics_tracker, epoch, use_combined_loss)
            
            # Validate
            val_results = self.validate_epoch(model, loss_fn, val_loader, 
                                            metrics_tracker, use_combined_loss)
            
            # Update learning rate
            scheduler.step(val_results['metrics']['qwk'])
            
            # Save best model
            if val_results['metrics']['qwk'] > best_qwk:
                best_qwk = val_results['metrics']['qwk']
                best_model_state = model.state_dict().copy()
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: "
                      f"Val QWK: {val_results['metrics']['qwk']:.3f}, "
                      f"Val Acc: {val_results['metrics']['accuracy']:.3f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_val_results = self.validate_epoch(model, loss_fn, val_loader, 
                                              metrics_tracker, use_combined_loss)
        
        return {
            'model': model,
            'metrics_tracker': metrics_tracker,
            'final_val_metrics': final_val_results['metrics'],
            'best_qwk': best_qwk,
            'config': self.configs[config_name]
        }
    
    def run_comparison_experiment(self, n_samples: int = 800, n_epochs: int = 15,
                                lr: float = 0.001) -> Dict:
        """Run comparison experiment across all configurations."""
        print("Starting ordinal attention training comparison...")
        print(f"Dataset: {n_samples} samples, {n_epochs} epochs, lr={lr}")
        
        # Create dataset
        train_loader, val_loader = self.create_synthetic_dataset(n_samples)
        print(f"Created dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
        
        results = {}
        
        for config_name in self.configs.keys():
            start_time = time.time()
            config_results = self.train_model(config_name, train_loader, val_loader, n_epochs, lr)
            end_time = time.time()
            
            config_results['training_time'] = end_time - start_time
            results[config_name] = config_results
            
            print(f"✓ {config_name}: QWK={config_results['best_qwk']:.3f}, "
                  f"Time={config_results['training_time']:.1f}s")
        
        self.results = results
        return results
    
    def print_comparison_results(self):
        """Print detailed comparison results."""
        if not self.results:
            print("No training results available. Run experiment first.")
            return
        
        print("\n" + "="*70)
        print("ORDINAL ATTENTION TRAINING COMPARISON RESULTS")
        print("="*70)
        
        # Get baseline metrics
        baseline_metrics = self.results.get('baseline', {}).get('final_val_metrics', {})
        
        print(f"\nModel Performance Comparison:")
        print(f"{'Model':<15} {'QWK':<8} {'Accuracy':<8} {'MAE':<8} {'Ord.Acc':<8} {'Time':<8}")
        print("-" * 70)
        
        for config_name, result in self.results.items():
            metrics = result['final_val_metrics']
            time_taken = result['training_time']
            
            print(f"{config_name:<15} {metrics['qwk']:<8.3f} {metrics['accuracy']:<8.3f} "
                  f"{metrics['mae']:<8.3f} {metrics['ordinal_accuracy']:<8.3f} {time_taken:<8.1f}s")
        
        # Improvement analysis
        print(f"\nImprovement over Baseline:")
        print(f"{'Model':<15} {'QWK':<12} {'Ord.Acc':<12} {'MAE':<12}")
        print("-" * 60)
        
        for config_name, result in self.results.items():
            if config_name == 'baseline':
                continue
            
            metrics = result['final_val_metrics']
            improvements = calculate_ordinal_improvement(baseline_metrics, metrics)
            
            qwk_imp = improvements.get('qwk_improvement', 0)
            ord_acc_imp = improvements.get('ordinal_accuracy_improvement', 0)
            mae_imp = improvements.get('mae_improvement', 0)
            
            print(f"{config_name:<15} {qwk_imp:<+12.1f}% {ord_acc_imp:<+12.1f}% {mae_imp:<+12.1f}%")
        
        print("\n" + "="*70)
    
    def save_results(self, filepath: str):
        """Save training results."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract serializable results
        serializable_results = {}
        for config_name, result in self.results.items():
            serializable_results[config_name] = {
                'final_val_metrics': result['final_val_metrics'],
                'best_qwk': result['best_qwk'],
                'training_time': result['training_time'],
                'config': result['config']
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training results saved to {filepath}")


def run_quick_training_experiment():
    """Run a quick training experiment."""
    print("Running quick ordinal attention training experiment...")
    
    trainer = OrdinalTrainer(n_questions=30, n_cats=4)
    results = trainer.run_comparison_experiment(n_samples=400, n_epochs=10, lr=0.001)
    trainer.print_comparison_results()
    
    return trainer


if __name__ == "__main__":
    trainer = run_quick_training_experiment()