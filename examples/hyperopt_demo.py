#!/usr/bin/env python3
"""
Demonstration of Advanced Hyperparameter Optimization for Deep-GPCM Models

This script shows how to use the new Bayesian hyperparameter optimization system
to find optimal hyperparameters for Deep-GPCM models with tunable loss weights.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.adaptive_hyperopt import create_optimizer, OptimizationConfig
from models.factory import get_all_model_types
import torch

def demo_basic_hyperopt():
    """Basic hyperparameter optimization demo."""
    print("=" * 80)
    print("BASIC HYPERPARAMETER OPTIMIZATION DEMO")
    print("=" * 80)
    
    # Configuration
    model_type = 'deep_gpcm'
    n_questions = 50
    n_cats = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Model: {model_type}")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    print(f"Device: {device}")
    
    # Create optimizer with moderate settings for demo
    config = OptimizationConfig(
        n_trials=20,              # Small number for demo
        n_initial_random=5,       # Initial random trials
        cv_folds=3,              # 3-fold CV
        cv_epochs=8,             # Few epochs for CV
        final_epochs=20,         # Final training epochs
        metric_to_optimize='quadratic_weighted_kappa',
        maximize=True,
        early_stopping_patience=8,
        save_intermediate=True
    )
    
    optimizer = create_optimizer(
        model_type=model_type,
        n_questions=n_questions,
        n_cats=n_cats,
        device=device,
        n_trials=config.n_trials,
        **config.__dict__
    )
    
    print(f"\nSearch space: {len(optimizer.search_space)} parameters")
    for space in optimizer.search_space:
        print(f"  - {space.name}: {space.param_type} {space.bounds}")
    
    # Generate synthetic data for demo
    train_data, test_data = generate_synthetic_data(n_questions, n_cats, n_samples=500)
    
    print(f"\nData: {len(train_data)} train, {len(test_data)} test sequences")
    
    # Run optimization
    print(f"\nStarting optimization with {config.n_trials} trials...")
    best_trial = optimizer.optimize(train_data, test_data)
    
    # Print results
    optimizer.print_summary()
    
    return optimizer, best_trial

def demo_multi_model_comparison():
    """Compare hyperparameter optimization across multiple models."""
    print("\n" + "=" * 80)
    print("MULTI-MODEL HYPERPARAMETER OPTIMIZATION COMPARISON")
    print("=" * 80)
    
    models_to_compare = ['deep_gpcm', 'attn_gpcm_linear']
    n_questions, n_cats = 30, 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Quick optimization settings for comparison
    config = OptimizationConfig(
        n_trials=15,
        n_initial_random=5,
        cv_folds=3,
        cv_epochs=6,
        metric_to_optimize='quadratic_weighted_kappa',
        early_stopping_patience=6
    )
    
    # Generate shared data
    train_data, test_data = generate_synthetic_data(n_questions, n_cats, n_samples=400)
    
    results = {}
    
    for model_type in models_to_compare:
        print(f"\n{'='*15} OPTIMIZING {model_type.upper()} {'='*15}")
        
        optimizer = create_optimizer(
            model_type=model_type,
            n_questions=n_questions,
            n_cats=n_cats,
            device=device,
            **config.__dict__
        )
        
        best_trial = optimizer.optimize(train_data, test_data)
        results[model_type] = {
            'optimizer': optimizer,
            'best_trial': best_trial,
            'best_score': optimizer.best_score
        }
    
    # Compare results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best {config.metric_to_optimize}: {result['best_score']:.4f}")
        print(f"  Trials completed: {len(result['optimizer'].trials)}")
        print(f"  Convergence rate: {sum(1 for t in result['optimizer'].trials if t.converged) / len(result['optimizer'].trials):.2%}")
    
    # Find overall best
    best_model = max(results.keys(), key=lambda m: results[m]['best_score'])
    print(f"\n🏆 WINNER: {best_model} with {config.metric_to_optimize} = {results[best_model]['best_score']:.4f}")
    
    return results

def demo_loss_weight_tuning():
    """Demonstrate tuning of loss function weights."""
    print("\n" + "=" * 80) 
    print("LOSS WEIGHT TUNING DEMONSTRATION")
    print("=" * 80)
    
    # Use a model with combined loss
    model_type = 'attn_gpcm_learn'  # This uses combined loss by default
    n_questions, n_cats = 25, 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = OptimizationConfig(
        n_trials=12,
        n_initial_random=4,
        cv_folds=3,
        cv_epochs=5,
        metric_to_optimize='quadratic_weighted_kappa'
    )
    
    optimizer = create_optimizer(
        model_type=model_type,
        n_questions=n_questions,
        n_cats=n_cats,
        device=device,
        **config.__dict__
    )
    
    # Show loss weight parameters in search space
    print("Loss weight parameters in search space:")
    loss_params = [s for s in optimizer.search_space if s.name.endswith('_logit')]
    for param in loss_params:
        print(f"  - {param.name}: {param.bounds}")
    
    # Generate data and optimize
    train_data, test_data = generate_synthetic_data(n_questions, n_cats, n_samples=300)
    best_trial = optimizer.optimize(train_data, test_data)
    
    # Show optimized loss weights
    print(f"\nOptimized loss weights:")
    best_params = best_trial.hyperparameters
    weight_params = {k: v for k, v in best_params.items() if k.endswith('_weight')}
    for param, value in weight_params.items():
        print(f"  - {param}: {value:.3f}")
    
    # Verify they sum to 1
    total_weight = sum(weight_params.values())
    print(f"  Total weight: {total_weight:.3f} (should be ≈ 1.0)")
    
    return optimizer, best_trial

def generate_synthetic_data(n_questions, n_cats, n_samples=500):
    """Generate synthetic sequential data for testing."""
    import numpy as np
    
    np.random.seed(42)  # Reproducible
    
    train_data = []
    test_data = []
    
    # Generate sequences
    for i in range(n_samples):
        seq_length = np.random.randint(5, 20)  # Variable sequence lengths
        
        # Random questions (with replacement)
        questions = np.random.choice(n_questions, size=seq_length, replace=True).tolist()
        
        # Responses with some pattern (easier questions -> higher responses)
        difficulty = np.array(questions) / n_questions  # Normalize
        ability = np.random.normal(0.5, 0.2)  # Student ability
        
        responses = []
        for q_idx, diff in enumerate(difficulty):
            # Probability of higher response decreases with difficulty
            prob = max(0.1, min(0.9, ability - diff + 0.3))
            response = np.random.binomial(n_cats - 1, prob)
            responses.append(response)
        
        # Split 80-20 train-test
        if i < int(0.8 * n_samples):
            train_data.append((questions, responses))
        else:
            test_data.append((questions, responses))
    
    return train_data, test_data

def main():
    """Run all demos."""
    print("Deep-GPCM Advanced Hyperparameter Optimization Demo")
    print("This demo requires synthetic data generation for testing purposes.")
    print("In practice, you would load your real educational assessment data.")
    
    try:
        # Demo 1: Basic optimization
        optimizer1, best_trial1 = demo_basic_hyperopt()
        
        # Demo 2: Multi-model comparison (optional, can be time-consuming)
        response = input("\nRun multi-model comparison demo? (y/n): ").lower()
        if response.startswith('y'):
            results = demo_multi_model_comparison()
        
        # Demo 3: Loss weight tuning
        response = input("\nRun loss weight tuning demo? (y/n): ").lower()
        if response.startswith('y'):
            optimizer3, best_trial3 = demo_loss_weight_tuning()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nTo use hyperparameter optimization in your training:")
        print("  python train.py --model deep_gpcm --dataset your_dataset --hyperopt --hyperopt_trials 50")
        print("\nAdvanced options:")
        print("  --hyperopt_metric quadratic_weighted_kappa  # Metric to optimize")
        print("  --n_folds 5                                 # Cross-validation folds")
        print("  --hyperopt_trials 100                       # Number of optimization trials")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing dependencies or data issues.")
        raise

if __name__ == "__main__":
    main()