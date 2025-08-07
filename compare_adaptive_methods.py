#!/usr/bin/env python3
"""
Compare Adaptive IRT Methods: Standard MML vs IRT-Optimized Loss

Direct comparison of the two theta estimation approaches.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from adaptive_irt_test import AdaptiveGPCMTest, load_synthetic_data
import warnings
warnings.filterwarnings('ignore')


def evaluate_method(data, use_irt_optimized=True, max_questions=15, n_sequences=25):
    """Evaluate adaptive IRT method."""
    
    method_name = "IRT-Optimized" if use_irt_optimized else "Standard MML"
    print(f"\n🔬 Evaluating {method_name} Method")
    print("=" * 50)
    
    all_predictions = []
    all_targets = []
    theta_errors = []
    
    # Use subset of sequences for efficiency
    sequences = data['sequences'][:n_sequences]
    
    for seq_idx, sequence in enumerate(sequences):
        if seq_idx % 5 == 0:
            print(f"Processing sequence {seq_idx}/{len(sequences)}...")
            
        questions = sequence['questions']
        responses = sequence['responses']
        seq_len = min(len(questions), max_questions)
        
        # Initialize adaptive test
        adaptive_test = AdaptiveGPCMTest(
            len(data['true_alpha']), 
            data['n_cats'],
            use_irt_optimized_loss=use_irt_optimized
        )
        
        # Process sequence
        for i in range(seq_len):
            q_id = questions[i]
            true_response = responses[i]
            
            # Get prediction before seeing response
            pred_probs = adaptive_test.predict_response(q_id)
            predicted_response = np.argmax(pred_probs)
            
            all_predictions.append(predicted_response)
            all_targets.append(true_response)
            
            # Update with true response
            adaptive_test.administer_question(q_id, true_response)
        
        # Compute theta error (using first student for simplicity)
        if seq_idx == 0:
            true_theta = data['true_theta'][0]  # Use first student's theta
            theta_error = abs(adaptive_test.current_theta - true_theta)
            theta_errors.append(theta_error)
    
    # Compute metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    accuracy = accuracy_score(targets, predictions)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(targets, predictions, weights='quadratic')
    theta_mae = np.mean(theta_errors) if theta_errors else 0.0
    
    results = {
        'method': method_name,
        'n_predictions': len(predictions),
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'qwk': qwk,
        'theta_mae': theta_mae,
        'category_distribution': {
            'targets': np.bincount(targets).tolist(),
            'predictions': np.bincount(predictions).tolist()
        }
    }
    
    return results


def main():
    """Compare both methods."""
    
    dataset_path = "data/synthetic_OC"
    print(f"🎯 ADAPTIVE IRT METHOD COMPARISON")
    print(f"Dataset: {dataset_path}")
    print(f"=" * 60)
    
    # Load data
    data = load_synthetic_data(dataset_path)
    print(f"Loaded {len(data['sequences'])} sequences")
    
    # Evaluate both methods
    results_standard = evaluate_method(data, use_irt_optimized=False, n_sequences=10)
    results_optimized = evaluate_method(data, use_irt_optimized=True, n_sequences=10)
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print(f"=" * 60)
    
    metrics = ['accuracy', 'f1_weighted', 'qwk', 'theta_mae']
    
    for metric in metrics:
        std_val = results_standard[metric]
        opt_val = results_optimized[metric]
        
        if metric == 'theta_mae':
            # Lower is better for MAE
            improvement = (std_val - opt_val) / std_val * 100 if std_val > 0 else 0
            direction = "lower is better"
        else:
            # Higher is better for accuracy, F1, QWK
            improvement = (opt_val - std_val) / std_val * 100 if std_val > 0 else 0
            direction = "higher is better"
        
        print(f"\n{metric.upper()} ({direction}):")
        print(f"  Standard MML:     {std_val:.3f}")
        print(f"  IRT-Optimized:    {opt_val:.3f}")
        print(f"  Change:           {improvement:+.1f}%")
    
    print(f"\nCategory Distributions:")
    print(f"  Targets:          {results_standard['category_distribution']['targets']}")
    print(f"  Standard preds:   {results_standard['category_distribution']['predictions']}")
    print(f"  Optimized preds:  {results_optimized['category_distribution']['predictions']}")
    
    # Summary
    qwk_change = (results_optimized['qwk'] - results_standard['qwk']) / results_standard['qwk'] * 100
    print(f"\n🎯 SUMMARY:")
    print(f"   Standard MML QWK:    {results_standard['qwk']:.3f}")
    print(f"   IRT-Optimized QWK:   {results_optimized['qwk']:.3f}")
    print(f"   QWK Change:          {qwk_change:+.1f}%")
    
    if qwk_change > 0:
        print(f"   ✅ IRT-Optimized loss improves performance")
    else:
        print(f"   ⚠️ IRT-Optimized loss decreases performance")
        print(f"   This suggests the hybrid loss may not be optimal for this task")


if __name__ == "__main__":
    main()