#!/usr/bin/env python3
"""
Main Runner for Deep-GPCM Complete Pipeline
By default: trains both models with CV, evaluates, and plots everything.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from utils.metrics import ensure_results_dirs


def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def run_complete_pipeline(models=['baseline', 'akvmn'], dataset='synthetic_OC', 
                         epochs=30, cv_folds=5, **kwargs):
    """Run the complete Deep-GPCM pipeline."""
    
    print("=" * 80)
    print("DEEP-GPCM COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"CV Folds: {cv_folds}")
    print()
    
    # Ensure results directories exist
    ensure_results_dirs()
    
    results = {'training': {}, 'evaluation': {}}
    
    # 1. Training phase with CV
    print(f"\n{'='*20} TRAINING PHASE {'='*20}")
    for model in models:
        cmd = [
            sys.executable, "train.py",
            "--model", model,
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--n_folds", str(cv_folds)
        ]
        
        # Add common arguments
        for key, value in kwargs.items():
            if key in ["batch_size", "lr", "seed"] and value is not None:
                cmd.extend([f"--{key}", str(value)])
            elif key == "device" and value is not None:
                cmd.extend([f"--{key}", value])
        
        success = run_command(cmd, f"Training {model.upper()}")
        results['training'][model] = success
    
    # 2. Evaluation phase
    print(f"\n{'='*20} EVALUATION PHASE {'='*20}")
    for model in models:
        if results['training'][model]:
            model_path = f"save_models/best_{model}_{dataset}.pth"
            
            if os.path.exists(model_path):
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", model_path,
                    "--dataset", dataset
                ]
                
                # Add additional arguments
                for key, value in kwargs.items():
                    if key == "batch_size" and value is not None:
                        cmd.extend([f"--{key}", str(value)])
                    elif key == "device" and value is not None:
                        cmd.extend([f"--{key}", value])
                
                success = run_command(cmd, f"Evaluating {model.upper()}")
                results['evaluation'][model] = success
            else:
                print(f"❌ Model file not found: {model_path}")
                results['evaluation'][model] = False
        else:
            print(f"⏭️  Skipping evaluation for {model} (training failed)")
            results['evaluation'][model] = False
    
    # 3. Plotting phase
    print(f"\n{'='*20} PLOTTING PHASE {'='*20}")
    if any(results['training'].values()):
        cmd = [
            sys.executable, "utils/plot_metrics.py"
        ]
        success = run_command(cmd, "Generating plots")
        results['plotting'] = success
    else:
        print("⏭️  Skipping plotting (no successful training)")
        results['plotting'] = False
    
    # 4. IRT Analysis phase
    print(f"\n{'='*20} IRT ANALYSIS PHASE {'='*20}")
    if any(results['training'].values()):
        cmd = [
            sys.executable, "analysis/irt_analysis.py",
            "--dataset", dataset,
            "--analysis_types", "recovery", "temporal"
        ]
        success = run_command(cmd, "Running IRT analysis")
        results['irt_analysis'] = success
    else:
        print("⏭️  Skipping IRT analysis (no successful training)")
        results['irt_analysis'] = False
    
    # 5. Summary
    print(f"\n{'='*20} PIPELINE SUMMARY {'='*20}")
    train_success = sum(results['training'].values())
    eval_success = sum(results['evaluation'].values())
    
    print(f"Training: {train_success}/{len(models)} models successful")
    print(f"Evaluation: {eval_success}/{len(models)} models successful")
    print(f"Plotting: {'✅' if results.get('plotting', False) else '❌'}")
    print(f"IRT Analysis: {'✅' if results.get('irt_analysis', False) else '❌'}")
    
    all_success = (train_success == len(models) and eval_success == len(models) and 
                   results.get('plotting', False) and results.get('irt_analysis', False))
    
    if all_success:
        print("🎉 Complete pipeline executed successfully!")
    else:
        print("⚠️  Pipeline completed with some failures")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Deep-GPCM Complete Pipeline')
    
    # By default, run everything
    parser.add_argument('--action', choices=['pipeline', 'train', 'evaluate'], 
                       default='pipeline', help='Action to perform (default: complete pipeline)')
    
    # Model and dataset selection
    parser.add_argument('--models', nargs='+', choices=['baseline', 'akvmn'], 
                       default=['baseline', 'akvmn'], help='Models to train/evaluate')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--cv_folds', type=int, default=5, help='CV folds (0 = no CV)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    
    # Individual control
    parser.add_argument('--model_path', help='Specific model path for evaluation')
    
    args = parser.parse_args()
    
    if args.action == 'pipeline':
        # Run complete pipeline by default
        run_complete_pipeline(
            models=args.models,
            dataset=args.dataset,
            epochs=args.epochs,
            cv_folds=args.cv_folds,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device
        )
    
    elif args.action == 'train':
        # Training only
        print(f"\n🚀 TRAINING ONLY")
        ensure_results_dirs()
        
        for model in args.models:
            cmd = [
                sys.executable, "train.py",
                "--model", model,
                "--dataset", args.dataset,
                "--epochs", str(args.epochs),
                "--n_folds", str(args.cv_folds),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed)
            ]
            
            if args.device:
                cmd.extend(["--device", args.device])
            
            run_command(cmd, f"Training {model.upper()}")
    
    elif args.action == 'evaluate':
        # Evaluation only
        print(f"\n🧪 EVALUATION ONLY")
        
        for model in args.models:
            model_path = args.model_path or f"save_models/best_{model}_{args.dataset}.pth"
            
            if os.path.exists(model_path):
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", model_path,
                    "--dataset", args.dataset,
                    "--batch_size", str(args.batch_size)
                ]
                if args.device:
                    cmd.extend(["--device", args.device])
                
                run_command(cmd, f"Evaluating {model.upper()}")
            else:
                print(f"❌ Model not found: {model_path}")
    
    
    print("\n🎯 Main runner completed!")


if __name__ == "__main__":
    main()