#!/usr/bin/env python3
"""
Compare Adaptive CORAL GPCM Models

Compare performance between minimal and full adaptive blending implementations
to demonstrate the benefits of the full learnable parameter approach.
"""

import sys
import os
sys.path.append('/home/steph/dirt-new/deep-gpcm')

import subprocess
import json
import pandas as pd
from datetime import datetime

def run_training(model_type, epochs=5):
    """Run training for a specific model type."""
    print(f"\n🚀 Training {model_type}...")
    cmd = [
        'python', 'train.py',
        '--model', model_type,
        '--dataset', 'synthetic_OC',
        '--epochs', str(epochs),
        '--no_cv'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/steph/dirt-new/deep-gpcm')
    
    if result.returncode != 0:
        print(f"❌ Training failed for {model_type}")
        print(f"Error: {result.stderr}")
        return None
    
    # Read results
    results_file = f'/home/steph/dirt-new/deep-gpcm/results/train/train_results_{model_type}_synthetic_OC.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    else:
        print(f"⚠️  Results file not found for {model_type}")
        return None

def compare_models():
    """Compare minimal vs full adaptive models.""" 
    print("🔬 ADAPTIVE CORAL GPCM MODEL COMPARISON")
    print("=" * 60)
    print("Comparing minimal vs full adaptive blending implementations")
    print()
    
    models = [
        ('adaptive_coral_gpcm', 'Minimal Adaptive Blender'),
        ('full_adaptive_coral_gpcm', 'Full Adaptive Blender')
    ]
    
    results = {}
    epochs = 5
    
    # Set conda environment
    env = os.environ.copy()
    env['PATH'] = '/home/steph/anaconda3/envs/vrec-env/bin:' + env['PATH']
    
    for model_type, description in models:
        print(f"\n{'='*20} {description} {'='*20}")
        
        # Run training with conda environment
        cmd = f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env && python train.py --model {model_type} --dataset synthetic_OC --epochs {epochs} --no_cv"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/steph/dirt-new/deep-gpcm')
        
        if result.returncode != 0:
            print(f"❌ Training failed for {model_type}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            results[model_type] = None
            continue
        
        # Read results
        results_file = f'/home/steph/dirt-new/deep-gpcm/results/train/train_results_{model_type}_synthetic_OC.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
            results[model_type] = data
            
            # Print summary
            metrics = data['metrics']
            print(f"✅ Training completed successfully")
            print(f"  - Parameters: {metrics['parameters']:,}")
            print(f"  - Best QWK: {metrics['quadratic_weighted_kappa']:.3f}")
            print(f"  - Categorical Acc: {metrics['categorical_accuracy']:.3f}")
            print(f"  - Ordinal Acc: {metrics['ordinal_accuracy']:.3f}")
            print(f"  - Gradient Norm: {metrics['gradient_norm']:.3f}")
            print(f"  - Training Loss: {metrics['train_loss']:.3f}")
        else:
            print(f"⚠️  Results file not found for {model_type}")
            results[model_type] = None
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if results['adaptive_coral_gpcm'] and results['full_adaptive_coral_gpcm']:
        minimal = results['adaptive_coral_gpcm']['metrics']
        full = results['full_adaptive_coral_gpcm']['metrics']
        
        print(f"\n{'Metric':<25} {'Minimal':<12} {'Full':<12} {'Improvement':<12}")
        print("-" * 60)
        
        metrics_to_compare = [
            ('Parameters', 'parameters', 'higher'),
            ('QWK', 'quadratic_weighted_kappa', 'higher'),
            ('Categorical Acc', 'categorical_accuracy', 'higher'),
            ('Ordinal Acc', 'ordinal_accuracy', 'higher'),
            ('Training Loss', 'train_loss', 'lower'),
            ('Gradient Norm', 'gradient_norm', 'stable'),
        ]
        
        for metric_name, key, direction in metrics_to_compare:
            min_val = minimal[key]
            full_val = full[key]
            
            if direction == 'higher':
                improvement = ((full_val - min_val) / min_val * 100) if min_val != 0 else 0
                improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            elif direction == 'lower':
                improvement = ((min_val - full_val) / min_val * 100) if min_val != 0 else 0
                improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            else:  # stable
                improvement_str = "stable" if abs(full_val - min_val) < 0.1 else f"{full_val:.3f}"
            
            print(f"{metric_name:<25} {min_val:<12.3f} {full_val:<12.3f} {improvement_str:<12}")
        
        print(f"\n🎯 KEY FINDINGS:")
        qwk_improvement = ((full['quadratic_weighted_kappa'] - minimal['quadratic_weighted_kappa']) / minimal['quadratic_weighted_kappa'] * 100) if minimal['quadratic_weighted_kappa'] != 0 else 0
        cat_improvement = ((full['categorical_accuracy'] - minimal['categorical_accuracy']) / minimal['categorical_accuracy'] * 100) if minimal['categorical_accuracy'] != 0 else 0
        
        print(f"  - Parameter scaling: {full['parameters']/minimal['parameters']:.1f}x increase")
        print(f"  - QWK improvement: {qwk_improvement:+.1f}%")
        print(f"  - Categorical accuracy: {cat_improvement:+.1f}%")
        print(f"  - Gradient stability: {'✅ Stable' if full['gradient_norm'] < 1.0 else '⚠️ Unstable'}")
        
        # Architecture comparison
        print(f"\n🏗️ ARCHITECTURE COMPARISON:")
        print(f"  Minimal Blender:")
        print(f"    - Fixed parameters (non-learnable)")
        print(f"    - Simple L2 distance computation")
        print(f"    - Conservative gradient isolation")
        print(f"    - Parameters: {minimal['parameters']:,}")
        
        print(f"  Full Blender:")
        print(f"    - Learnable parameters with BGT stability")
        print(f"    - Complete semantic threshold analysis")
        print(f"    - Category-specific adaptive weights")
        print(f"    - Parameters: {full['parameters']:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if qwk_improvement > 5:
            print(f"  ✅ Full blender shows significant improvement ({qwk_improvement:+.1f}% QWK)")
            print(f"     Recommend using full_adaptive_coral_gpcm for production")
        elif qwk_improvement > 0:
            print(f"  ⚖️  Full blender shows modest improvement ({qwk_improvement:+.1f}% QWK)")
            print(f"     Consider computational cost vs performance gain")
        else:
            print(f"  ⚠️  Minimal blender performing better - full model may need tuning")
        
        # Save comparison results
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': [
                {'type': 'adaptive_coral_gpcm', 'description': 'Minimal Adaptive Blender'},
                {'type': 'full_adaptive_coral_gpcm', 'description': 'Full Adaptive Blender'}
            ],
            'metrics_comparison': {
                'minimal': minimal,
                'full': full,
                'improvements': {
                    'qwk_improvement_percent': qwk_improvement,
                    'categorical_improvement_percent': cat_improvement,
                    'parameter_scaling': full['parameters'] / minimal['parameters']
                }
            }
        }
        
        with open('/home/steph/dirt-new/deep-gpcm/results/adaptive_model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n📁 Results saved to: results/adaptive_model_comparison.json")
        
    else:
        print("❌ Cannot compare - one or more models failed to train")
        
    print(f"\n🎉 COMPARISON COMPLETE!")

if __name__ == "__main__":
    compare_models()