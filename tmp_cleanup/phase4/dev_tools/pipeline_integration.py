"""
Pipeline integration for ordinal-aware attention mechanisms.
Integrates with existing Deep-GPCM training and evaluation pipeline.
"""

import torch
import json
import time
import argparse
from pathlib import Path

# Import existing pipeline components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations.attention_gpcm import AttentionGPCM
from models.losses.ordinal_losses import create_ordinal_loss
from models.metrics.ordinal_metrics import OrdinalMetrics, MetricsTracker
from optimization.hyperparameter_tuning import HyperparameterOptimizer

# Import existing pipeline utilities
from utils.metrics import compute_metrics, save_results, ensure_results_dirs
from utils.path_utils import get_path_manager, ensure_directories


class OrdinalAttentionPipelineIntegrator:
    """Integrates ordinal-aware attention with existing Deep-GPCM pipeline."""
    
    def __init__(self):
        self.path_manager = get_path_manager()
        
        # Enhanced model configurations for ordinal attention
        self.ordinal_model_configs = {
            'ordinal_attn_gpcm': {
                'base_model': 'attn_gpcm',
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware'],
                'loss_type': 'ordinal_ce',
                'description': 'Attention-GPCM with ordinal-aware attention',
                'color': '#2ca02c'  # Green
            },
            'qwk_attn_gpcm': {
                'base_model': 'attn_gpcm',
                'use_ordinal_attention': True,
                'attention_types': ['qwk_aligned'],
                'loss_type': 'combined',
                'description': 'Attention-GPCM with QWK-aligned attention',
                'color': '#d62728'  # Red
            },
            'combined_ordinal_gpcm': {
                'base_model': 'attn_gpcm',
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned'],
                'loss_type': 'combined',
                'description': 'Attention-GPCM with combined ordinal attention',
                'color': '#ff7f0e'  # Orange
            },
            'full_ordinal_gpcm': {
                'base_model': 'attn_gpcm',
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned', 'response_conditioned'],
                'loss_type': 'combined',
                'description': 'Attention-GPCM with full ordinal attention suite',
                'color': '#9467bd'  # Purple
            }
        }
    
    def create_ordinal_model(self, model_type: str, n_questions: int, n_cats: int, **kwargs):
        """Create ordinal-aware model based on configuration."""
        if model_type not in self.ordinal_model_configs:
            raise ValueError(f"Unknown ordinal model type: {model_type}")
        
        config = self.ordinal_model_configs[model_type]
        
        # Create base AttentionGPCM with ordinal enhancements
        model = AttentionGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=kwargs.get('embed_dim', 64),
            memory_size=kwargs.get('memory_size', 50),
            key_dim=kwargs.get('key_dim', 50),
            value_dim=kwargs.get('value_dim', 200),
            final_fc_dim=kwargs.get('final_fc_dim', 50),
            n_heads=kwargs.get('n_heads', 8),
            n_cycles=kwargs.get('n_cycles', 2),
            embedding_strategy=kwargs.get('embedding_strategy', 'linear_decay'),
            ability_scale=kwargs.get('ability_scale', 1.0),
            dropout_rate=kwargs.get('dropout_rate', 0.2),
            use_ordinal_attention=config['use_ordinal_attention'],
            attention_types=config['attention_types']
        )
        
        # Attach metadata
        model.model_type = model_type
        model.model_color = config['color']
        model.display_name = model_type.replace('_', '-').upper()
        model.description = config['description']
        
        return model
    
    def create_ordinal_loss(self, loss_type: str, n_cats: int, **kwargs):
        """Create ordinal-aware loss function."""
        return create_ordinal_loss(loss_type, n_cats, **kwargs)
    
    def load_existing_data(self, dataset: str):
        """Load data using existing pipeline format."""
        data_path = Path("data") / dataset
        
        # Handle naming conventions - try lowercase version first
        dataset_lower = dataset.lower()
        train_path = data_path / f"{dataset_lower}_train.txt"
        test_path = data_path / f"{dataset_lower}_test.txt"
        
        # Fallback to original naming if lowercase doesn't exist
        if not train_path.exists() or not test_path.exists():
            train_path = data_path / f"{dataset}_train.txt"
            test_path = data_path / f"{dataset}_test.txt"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Dataset {dataset} not found at {data_path}")
        
        # Use existing data loading function
        from train import load_simple_data
        return load_simple_data(str(train_path), str(test_path))
    
    def train_ordinal_model(self, model_type: str, dataset: str, epochs: int = 30,
                           n_folds: int = 5, **kwargs):
        """Train ordinal model using existing pipeline structure."""
        print(f"\n🚀 Training {model_type} on {dataset}")
        print(f"   Epochs: {epochs}, Folds: {n_folds}")
        
        # Load data
        train_data, test_data, n_questions, n_cats = self.load_existing_data(dataset)
        print(f"   Questions: {n_questions}, Categories: {n_cats}")
        print(f"   Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Create model and loss
        model = self.create_ordinal_model(model_type, n_questions, n_cats, **kwargs)
        
        config = self.ordinal_model_configs[model_type]
        loss_fn = self.create_ordinal_loss(config['loss_type'], n_cats)
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create data loaders with smaller batch size for integration test
        from train import create_data_loaders
        train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=16)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5)
        
        # Metrics tracking
        ordinal_metrics = OrdinalMetrics(n_cats)
        metrics_tracker = MetricsTracker(n_cats)
        
        # Training loop
        best_qwk = -1.0
        best_model_state = None
        training_history = {'train': [], 'val': []}
        
        # Simplified metrics tracking for pipeline integration  
        simple_metrics = OrdinalMetrics(n_cats)
        
        print(f"   🔄 Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds, train_targets, train_probs = [], [], []
            
            max_train_batches = 20  # Limit batches for integration test
            for batch_idx, (questions, responses, masks) in enumerate(train_loader):
                if batch_idx >= max_train_batches:
                    break
                    
                questions, responses = questions.to(device), responses.to(device)
                # masks are available but not used in our ordinal models
                
                optimizer.zero_grad()
                
                # Forward pass
                _, _, _, probs = model(questions, responses)
                
                # Calculate loss
                if config['loss_type'] == 'combined':
                    logits = torch.log(probs + 1e-8)
                    loss, loss_dict = loss_fn(logits, responses)
                else:
                    logits = torch.log(probs + 1e-8)
                    loss = loss_fn(logits, responses)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Collect predictions
                preds = probs.argmax(dim=-1)
                train_preds.append(preds.detach())
                train_targets.append(responses.detach())
                train_probs.append(probs.detach())
                
                # Debug first batch to understand dimensions
                if batch_idx == 0:
                    print(f"   Debug - Batch {batch_idx}: Q {questions.shape}, R {responses.shape}, P {probs.shape}")
            
            # Calculate training metrics - flatten all sequences
            train_preds = torch.cat([pred.flatten() for pred in train_preds], dim=0)
            train_targets = torch.cat([target.flatten() for target in train_targets], dim=0)
            train_probs = torch.cat([prob.view(-1, prob.size(-1)) for prob in train_probs], dim=0)
            
            # Use simplified metrics to avoid memory issues
            train_metrics = simple_metrics.calculate_all_metrics(train_targets, train_preds, train_probs)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds, val_targets, val_probs = [], [], []
            
            max_val_batches = 10  # Limit validation batches too
            with torch.no_grad():
                for val_batch_idx, (questions, responses, masks) in enumerate(test_loader):
                    if val_batch_idx >= max_val_batches:
                        break
                        
                    questions, responses = questions.to(device), responses.to(device)
                    
                    _, _, _, probs = model(questions, responses)
                    
                    if config['loss_type'] == 'combined':
                        logits = torch.log(probs + 1e-8)
                        loss, _ = loss_fn(logits, responses)
                    else:
                        logits = torch.log(probs + 1e-8)
                        loss = loss_fn(logits, responses)
                    
                    val_loss += loss.item()
                    
                    preds = probs.argmax(dim=-1)
                    val_preds.append(preds)
                    val_targets.append(responses)
                    val_probs.append(probs)
            
            val_preds = torch.cat([pred.flatten() for pred in val_preds], dim=0)
            val_targets = torch.cat([target.flatten() for target in val_targets], dim=0)
            val_probs = torch.cat([prob.view(-1, prob.size(-1)) for prob in val_probs], dim=0)
            
            # Use simplified metrics for validation too
            val_metrics = simple_metrics.calculate_all_metrics(val_targets, val_preds, val_probs)
            
            # Update scheduler
            scheduler.step(val_metrics['qwk'])
            
            # Save best model
            if val_metrics['qwk'] > best_qwk:
                best_qwk = val_metrics['qwk']
                best_model_state = model.state_dict().copy()
            
            # Store history
            training_history['train'].append({
                'epoch': epoch + 1,
                'loss': train_loss / len(train_loader),
                'metrics': train_metrics
            })
            training_history['val'].append({
                'epoch': epoch + 1,
                'loss': val_loss / len(test_loader),
                'metrics': val_metrics
            })
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: "
                      f"QWK: {val_metrics['qwk']:.3f}, "
                      f"Acc: {val_metrics['accuracy']:.3f}, "
                      f"MAE: {val_metrics['mae']:.3f}")
        
        training_time = time.time() - start_time
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        final_preds, final_targets, final_probs = [], [], []
        
        with torch.no_grad():
            for final_batch_idx, (questions, responses, masks) in enumerate(test_loader):
                if final_batch_idx >= 10:  # Limit final evaluation too
                    break
                    
                questions, responses = questions.to(device), responses.to(device)
                _, _, _, probs = model(questions, responses)
                preds = probs.argmax(dim=-1)
                
                final_preds.append(preds)
                final_targets.append(responses)
                final_probs.append(probs)
        
        final_preds = torch.cat([pred.flatten() for pred in final_preds], dim=0)
        final_targets = torch.cat([target.flatten() for target in final_targets], dim=0)
        final_probs = torch.cat([prob.view(-1, prob.size(-1)) for prob in final_probs], dim=0)
        
        final_metrics = ordinal_metrics.calculate_all_metrics(final_targets, final_preds, final_probs)
        
        print(f"   ✅ Training completed in {training_time:.1f}s")
        print(f"   📊 Final QWK: {final_metrics['qwk']:.3f}, Accuracy: {final_metrics['accuracy']:.3f}")
        
        # Save results in pipeline format
        results = {
            'model_type': model_type,
            'dataset': dataset,
            'final_metrics': final_metrics,
            'best_qwk': best_qwk,
            'training_time': training_time,
            'epochs': epochs,
            'training_history': training_history,
            'config': config
        }
        
        # Save to results directory
        ensure_directories(dataset)
        ensure_results_dirs()
        
        results_path = Path("results") / "train" / dataset / f"train_{model_type}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save model
        models_path = Path("saved_models") / dataset / f"best_{model_type}.pth"
        models_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), models_path)
        
        print(f"   💾 Results saved to {results_path}")
        print(f"   💾 Model saved to {models_path}")
        
        return results
    
    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # Scalar tensors
            return obj.item()
        else:
            return obj
    
    def run_ordinal_comparison(self, dataset: str = 'synthetic_4000_200_2', epochs: int = 20):
        """Run comparison between baseline and ordinal attention models."""
        print("="*80)
        print("ORDINAL ATTENTION PIPELINE INTEGRATION")
        print("="*80)
        
        models_to_test = [
            'ordinal_attn_gpcm',
            'qwk_attn_gpcm', 
            'combined_ordinal_gpcm'
        ]
        
        results = {}
        
        for model_type in models_to_test:
            print(f"\n{'='*20} {model_type.upper()} {'='*20}")
            
            try:
                model_results = self.train_ordinal_model(
                    model_type=model_type,
                    dataset=dataset,
                    epochs=epochs,
                    lr=0.001,
                    embed_dim=64,
                    n_heads=8,
                    n_cycles=2
                )
                results[model_type] = model_results
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                results[model_type] = {'error': str(e)}
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_comparison_summary(self, results: dict):
        """Print comparison summary of ordinal attention results."""
        print("\n" + "="*80)
        print("ORDINAL ATTENTION COMPARISON SUMMARY")
        print("="*80)
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful training runs")
            return
        
        print(f"\nSuccessful models: {len(successful_results)}/{len(results)}")
        
        print(f"\n{'Model':<25} {'QWK':<8} {'Accuracy':<10} {'MAE':<8} {'Time(s)':<8}")
        print("-" * 60)
        
        for model_type, result in successful_results.items():
            metrics = result['final_metrics']
            time_taken = result['training_time']
            
            print(f"{model_type:<25} {metrics['qwk']:<8.3f} {metrics['accuracy']:<10.3f} "
                  f"{metrics['mae']:<8.3f} {time_taken:<8.1f}")
        
        # Find best model
        best_model = max(successful_results.items(), key=lambda x: x[1]['final_metrics']['qwk'])
        print(f"\n🏆 Best Model: {best_model[0]} (QWK: {best_model[1]['final_metrics']['qwk']:.3f})")
        
        print("="*80)


def run_pipeline_integration():
    """Run the pipeline integration test."""
    integrator = OrdinalAttentionPipelineIntegrator()
    results = integrator.run_ordinal_comparison(dataset='synthetic_4000_200_2', epochs=3)
    return integrator, results


if __name__ == "__main__":
    integrator, results = run_pipeline_integration()