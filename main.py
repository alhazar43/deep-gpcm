#!/usr/bin/env python3
"""
Unified Main Script for Deep-GPCM Models
Orchestrates training, evaluation, and plotting for complete pipeline.
"""

import os
import sys
import argparse
import subprocess
import glob
from datetime import datetime


def run_command(cmd, description, check=True):
    """Run a command with error handling."""
    print(f"\\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        print(f"✅ {description} completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def find_model_files(models, dataset):
    """Find trained model files."""
    model_files = {}
    
    for model in models:
        # Look for main model file first
        main_path = f"save_models/best_{model}_{dataset}.pth"
        if os.path.exists(main_path):
            model_files[model] = main_path
            continue
        
        # Look for fold-specific files
        fold_pattern = f"save_models/best_{model}_{dataset}_fold_*.pth"
        fold_files = glob.glob(fold_pattern)
        if fold_files:
            # Use the first fold file (or could pick best based on metrics)
            model_files[model] = fold_files[0]
            continue
        
        print(f"⚠️  No trained model found for {model}")
    
    return model_files


def find_result_files(models, dataset, result_type='train'):
    """Find training or test result files."""
    result_files = {}
    
    for model in models:
        # Look for main result file first
        main_path = f"logs/{result_type}_results_{model}_{dataset}.json"
        if os.path.exists(main_path):
            result_files[model] = main_path
            continue
        
        # Look for fold-specific files 
        if result_type == 'train':
            fold_pattern = f"logs/train_results_{model}_{dataset}_fold_*.json"
            fold_files = glob.glob(fold_pattern)
            if fold_files:
                # Use the first fold file
                result_files[model] = fold_files[0]
                continue
        
        # Look for test results
        if result_type == 'test':
            test_path = f"results/test/test_results_{model}_{dataset}.json"
            if os.path.exists(test_path):
                result_files[model] = test_path
                continue
        
        print(f"⚠️  No {result_type} results found for {model}")
    
    return result_files


def main():
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Pipeline')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'deep_integration'], 
                        default=['baseline', 'deep_integration'], help='Models to train/evaluate')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='CV folds (0 for no CV)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only evaluate/plot')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation, only plot')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIFIED DEEP-GPCM PIPELINE")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Epochs: {args.epochs}")
    print(f"CV folds: {args.n_folds}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print()
    
    # Create directories
    os.makedirs('save_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/test', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    success_count = 0
    total_operations = 0
    
    # Phase 1: Training
    if not args.skip_training:
        print(f"\\n{'='*20} PHASE 1: TRAINING {'='*20}")
        
        for model in args.models:
            total_operations += 1
            
            # Build training command
            train_cmd = [
                'python', 'train.py',
                '--model', model,
                '--dataset', args.dataset,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--n_folds', str(args.n_folds)
            ]
            
            if args.device:
                train_cmd.extend(['--device', args.device])
            
            # Run training
            if run_command(train_cmd, f"Training {model} model"):
                success_count += 1
                print(f"✅ {model} training completed")
            else:
                print(f"❌ {model} training failed")
                continue
    else:
        print(f"\\n⏭️  SKIPPING TRAINING (using existing models)")
    
    # Phase 2: Evaluation
    if not args.skip_evaluation:
        print(f"\\n{'='*20} PHASE 2: EVALUATION {'='*20}")
        
        # Find trained models
        model_files = find_model_files(args.models, args.dataset)
        
        if not model_files:
            print("❌ No trained models found for evaluation")
            if not args.skip_training:
                print("Training may have failed for all models")
            else:
                print("Run training first or check save_models/ directory")
            sys.exit(1)
        
        print(f"📂 Found models: {list(model_files.keys())}")
        
        for model, model_path in model_files.items():
            total_operations += 1
            
            # Build evaluation command
            eval_cmd = [
                'python', 'evaluate.py',
                '--model_path', model_path,
                '--dataset', args.dataset,
                '--batch_size', str(args.batch_size)
            ]
            
            if args.device:
                eval_cmd.extend(['--device', args.device])
            
            # Run evaluation
            if run_command(eval_cmd, f"Evaluating {model} model"):
                success_count += 1
                print(f"✅ {model} evaluation completed")
            else:
                print(f"❌ {model} evaluation failed")
    else:
        print(f"\\n⏭️  SKIPPING EVALUATION (using existing results)")
    
    # Phase 3: Plotting
    print(f"\\n{'='*20} PHASE 3: PLOTTING {'='*20}")
    
    # Find result files
    train_files = find_result_files(args.models, args.dataset, 'train')
    test_files = find_result_files(args.models, args.dataset, 'test')
    
    if not train_files and not test_files:
        print("❌ No result files found for plotting")
        print("Run training and/or evaluation first")
        sys.exit(1)
    
    # Build plotting command
    plot_cmd = ['python', 'plot_metrics.py', '--output_dir', 'results/plots']
    
    if train_files:
        plot_cmd.extend(['--train_results'] + list(train_files.values()))
        print(f"📊 Training results: {list(train_files.keys())}")
    
    if test_files:
        plot_cmd.extend(['--test_results'] + list(test_files.values()))
        print(f"🧪 Test results: {list(test_files.keys())}")
    
    # Run plotting
    total_operations += 1
    if run_command(plot_cmd, "Generating comprehensive plots"):
        success_count += 1
        print("✅ Plotting completed")
    else:
        print("❌ Plotting failed")
    
    # Phase 4: Summary
    print(f"\\n{'='*20} PIPELINE SUMMARY {'='*20}")
    
    print(f"📊 Operations: {success_count}/{total_operations} successful")
    print(f"🎯 Success rate: {success_count/total_operations*100:.1f}%")
    
    # Display results summary
    if train_files or test_files:
        print(f"\\n📈 RESULTS SUMMARY:")
        
        # Load and display key metrics
        for model in args.models:
            print(f"\\n🔹 {model.upper()}:")
            
            # Training metrics
            if model in train_files:
                try:
                    import json
                    with open(train_files[model], 'r') as f:
                        train_data = json.load(f)
                    
                    if 'metrics' in train_data:
                        metrics = train_data['metrics']
                        print(f"  Training - Accuracy: {metrics.get('categorical_accuracy', 'N/A'):.3f}, "
                              f"QWK: {metrics.get('quadratic_weighted_kappa', 'N/A'):.3f}")
                    elif 'cv_summary' in train_data:
                        cv = train_data['cv_summary']
                        acc_mean = cv.get('categorical_accuracy', {}).get('mean', 'N/A')
                        qwk_mean = cv.get('quadratic_weighted_kappa', {}).get('mean', 'N/A')
                        print(f"  CV Training - Accuracy: {acc_mean:.3f}, QWK: {qwk_mean:.3f}")
                except:
                    print(f"  Training - Could not load metrics")
            
            # Test metrics
            if model in test_files:
                try:
                    with open(test_files[model], 'r') as f:
                        test_data = json.load(f)
                    
                    if 'evaluation_results' in test_data and 'argmax' in test_data['evaluation_results']:
                        metrics = test_data['evaluation_results']['argmax']
                        print(f"  Test - Accuracy: {metrics.get('categorical_accuracy', 'N/A'):.3f}, "
                              f"QWK: {metrics.get('quadratic_weighted_kappa', 'N/A'):.3f}")
                except:
                    print(f"  Test - Could not load metrics")
    
    # Expected performance check
    print(f"\\n🎯 PERFORMANCE CHECK:")
    print(f"Expected baseline: ~70% accuracy, ~0.67 QWK")
    print(f"Expected deep integration: ~70% accuracy, ~0.69 QWK")
    print(f"Expected improvement: 1-4% across metrics")
    print(f"⚠️  If you see >20% improvements, investigate for bugs!")
    
    # File locations
    print(f"\\n📁 OUTPUT LOCATIONS:")
    print(f"  Models: save_models/")
    print(f"  Training logs: logs/")
    print(f"  Test results: results/test/")
    print(f"  Plots: results/plots/")
    
    if success_count == total_operations:
        print(f"\\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\\n⚠️  PIPELINE COMPLETED WITH SOME FAILURES")
        return 1


if __name__ == "__main__":
    exit(main())