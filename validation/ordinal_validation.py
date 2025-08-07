"""
Validation framework for ordinal-aware attention mechanisms.
Provides systematic testing and comparison of different attention types.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path

# Import models and components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations.attention_gpcm import AttentionGPCM
from models.implementations.deep_gpcm import DeepGPCM
from models.losses.ordinal_losses import create_ordinal_loss
from models.metrics.ordinal_metrics import OrdinalMetrics, MetricsTracker, calculate_ordinal_improvement


class AttentionMechanismValidator:
    """Validates different attention mechanisms systematically."""
    
    def __init__(self, n_questions: int = 50, n_cats: int = 4, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.device = device
        
        # Model configurations to test
        self.model_configs = {
            'baseline': {
                'use_ordinal_attention': False,
                'attention_types': None
            },
            'ordinal_aware': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware']
            },
            'response_conditioned': {
                'use_ordinal_attention': True,
                'attention_types': ['response_conditioned']
            },
            'qwk_aligned': {
                'use_ordinal_attention': True,
                'attention_types': ['qwk_aligned']
            },
            'combined': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned']
            },
            'full_suite': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'response_conditioned', 'qwk_aligned']
            }
        }
        
        # Tracking
        self.results = {}
        self.timing_results = {}
    
    def create_model(self, config_name: str, **kwargs) -> AttentionGPCM:
        """Create model based on configuration."""
        config = self.model_configs[config_name].copy()
        config.update(kwargs)
        
        model = AttentionGPCM(
            n_questions=self.n_questions,
            n_cats=self.n_cats,
            embed_dim=32,  # Smaller for faster validation
            memory_size=20,
            key_dim=32,
            value_dim=64,
            final_fc_dim=32,
            n_heads=4,
            n_cycles=2,
            dropout_rate=0.1,
            **config
        ).to(self.device)
        
        return model
    
    def generate_synthetic_data(self, n_samples: int = 200, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic ordinal data for testing."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate questions and responses with ordinal patterns
        questions = torch.randint(1, self.n_questions + 1, (n_samples, seq_len))
        
        # Create responses with ordinal structure
        responses = torch.zeros(n_samples, seq_len, dtype=torch.long)
        
        for i in range(n_samples):
            # Student ability
            ability = np.random.normal(0, 1)
            
            for j in range(seq_len):
                # Question difficulty
                difficulty = (questions[i, j].item() - self.n_questions/2) / (self.n_questions/4)
                
                # Probability of each response category
                logits = ability - difficulty + np.random.normal(0, 0.5, self.n_cats)
                probs = torch.softmax(torch.tensor(logits), dim=0)
                
                # Sample response
                responses[i, j] = torch.multinomial(probs, 1).item()
        
        return questions.to(self.device), responses.to(self.device)
    
    def validate_forward_pass(self, config_name: str, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Validate forward pass and measure timing."""
        questions, responses = data
        model = self.create_model(config_name)
        
        # Warmup
        with torch.no_grad():
            _ = model(questions[:5], responses[:5])
        
        # Timing test
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            abilities, thresholds, discriminations, probs = model(questions, responses)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Validate outputs
        batch_size, seq_len = questions.shape
        assert abilities.shape == (batch_size, seq_len)
        assert thresholds.shape == (batch_size, seq_len, self.n_cats - 1)
        assert discriminations.shape == (batch_size, seq_len)
        assert probs.shape == (batch_size, seq_len, self.n_cats)
        
        # Check probability constraints
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
        
        return {
            'forward_time': end_time - start_time,
            'memory_allocated': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'output_shapes_valid': True,
            'probability_constraints_satisfied': True
        }
    
    def validate_loss_computation(self, config_name: str, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Validate loss computation with different loss functions."""
        questions, responses = data
        model = self.create_model(config_name)
        
        # Test different loss functions
        loss_functions = {
            'standard_ce': nn.CrossEntropyLoss(),
            'ordinal_ce': create_ordinal_loss('ordinal_ce', self.n_cats),
            'qwk_loss': create_ordinal_loss('qwk', self.n_cats),
            'combined': create_ordinal_loss('combined', self.n_cats)
        }
        
        results = {}
        
        with torch.no_grad():
            _, _, _, probs = model(questions, responses)
            logits = torch.log(probs + 1e-8)  # Convert to logits
        
        for loss_name, loss_fn in loss_functions.items():
            try:
                if loss_name == 'standard_ce':
                    loss = loss_fn(logits.view(-1, self.n_cats), responses.view(-1))
                elif loss_name == 'combined':
                    loss, loss_dict = loss_fn(logits, responses)
                    results[f'{loss_name}_components'] = loss_dict
                else:
                    loss = loss_fn(logits, responses)
                
                results[f'{loss_name}_value'] = loss.item()
                results[f'{loss_name}_valid'] = torch.isfinite(loss).item()
                
            except Exception as e:
                results[f'{loss_name}_error'] = str(e)
                results[f'{loss_name}_valid'] = False
        
        return results
    
    def validate_metrics(self, config_name: str, data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Validate metrics computation."""
        questions, responses = data
        model = self.create_model(config_name)
        
        with torch.no_grad():
            _, _, _, probs = model(questions, responses)
            preds = probs.argmax(dim=-1)
        
        # Calculate metrics
        metrics_calculator = OrdinalMetrics(self.n_cats)
        metrics = metrics_calculator.calculate_all_metrics(responses, preds, probs)
        
        # Validate metric ranges
        validations = {}
        validations['accuracy_in_range'] = 0 <= metrics['accuracy'] <= 1
        validations['qwk_in_range'] = -1 <= metrics['qwk'] <= 1
        validations['mae_non_negative'] = metrics['mae'] >= 0
        validations['ordinal_accuracy_in_range'] = 0 <= metrics['ordinal_accuracy'] <= 1
        
        # Add metrics
        validations.update(metrics)
        
        return validations
    
    def run_comprehensive_validation(self, n_samples: int = 200) -> Dict[str, Dict]:
        """Run comprehensive validation across all configurations."""
        print("Starting comprehensive validation of ordinal attention mechanisms...")
        
        # Generate test data
        data = self.generate_synthetic_data(n_samples)
        print(f"Generated synthetic data: {data[0].shape[0]} samples, {data[0].shape[1]} sequence length")
        
        results = {}
        
        for config_name in self.model_configs.keys():
            print(f"\nValidating configuration: {config_name}")
            config_results = {}
            
            try:
                # Forward pass validation
                print("  - Forward pass validation...")
                forward_results = self.validate_forward_pass(config_name, data)
                config_results['forward_pass'] = forward_results
                
                # Loss computation validation
                print("  - Loss computation validation...")
                loss_results = self.validate_loss_computation(config_name, data)
                config_results['loss_computation'] = loss_results
                
                # Metrics validation
                print("  - Metrics validation...")
                metrics_results = self.validate_metrics(config_name, data)
                config_results['metrics'] = metrics_results
                
                # Overall status
                config_results['validation_successful'] = True
                print(f"  ✓ {config_name} validation successful")
                
            except Exception as e:
                config_results['validation_error'] = str(e)
                config_results['validation_successful'] = False
                print(f"  ✗ {config_name} validation failed: {e}")
            
            results[config_name] = config_results
        
        self.results = results
        return results
    
    def compare_attention_mechanisms(self) -> Dict[str, float]:
        """Compare performance of different attention mechanisms."""
        if not self.results:
            self.run_comprehensive_validation()
        
        comparison = {}
        baseline_metrics = self.results.get('baseline', {}).get('metrics', {})
        
        for config_name, config_results in self.results.items():
            if config_name == 'baseline' or not config_results.get('validation_successful'):
                continue
            
            config_metrics = config_results.get('metrics', {})
            if not config_metrics:
                continue
            
            # Calculate improvements
            improvements = calculate_ordinal_improvement(baseline_metrics, config_metrics)
            comparison[config_name] = improvements
            
            # Add timing comparison
            baseline_time = self.results['baseline']['forward_pass']['forward_time']
            config_time = config_results['forward_pass']['forward_time']
            comparison[config_name]['time_overhead'] = ((config_time - baseline_time) / baseline_time) * 100
        
        return comparison
    
    def print_validation_summary(self):
        """Print comprehensive validation summary."""
        if not self.results:
            print("No validation results available. Run validation first.")
            return
        
        print("\n" + "="*60)
        print("ORDINAL ATTENTION VALIDATION SUMMARY")
        print("="*60)
        
        # Success rate
        successful = sum(1 for r in self.results.values() if r.get('validation_successful', False))
        total = len(self.results)
        print(f"\nValidation Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        # Performance comparison
        print(f"\nPerformance Comparison (vs baseline):")
        comparison = self.compare_attention_mechanisms()
        
        for config_name, improvements in comparison.items():
            print(f"\n{config_name.upper()}:")
            for metric, improvement in improvements.items():
                if 'improvement' in metric:
                    print(f"  {metric}: {improvement:+.1f}%")
                elif metric == 'time_overhead':
                    print(f"  {metric}: {improvement:+.1f}%")
        
        # Timing results
        print(f"\nTiming Results:")
        for config_name, results in self.results.items():
            if results.get('validation_successful'):
                forward_time = results['forward_pass']['forward_time']
                memory = results['forward_pass']['memory_allocated']
                print(f"  {config_name}: {forward_time:.3f}s, {memory:.1f}MB")
        
        print("\n" + "="*60)
    
    def save_results(self, filepath: str):
        """Save validation results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for config_name, config_results in self.results.items():
            serializable_config = {}
            for key, value in config_results.items():
                if isinstance(value, dict):
                    serializable_config[key] = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                                              for k, v in value.items()}
                else:
                    serializable_config[key] = float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
            serializable_results[config_name] = serializable_config
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Validation results saved to {filepath}")


def run_quick_validation():
    """Run a quick validation test."""
    print("Running quick ordinal attention validation...")
    
    validator = AttentionMechanismValidator(n_questions=20, n_cats=4)
    results = validator.run_comprehensive_validation(n_samples=50)
    validator.print_validation_summary()
    
    return validator


if __name__ == "__main__":
    validator = run_quick_validation()