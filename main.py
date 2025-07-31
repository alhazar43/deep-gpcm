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
    
    # Print loss function info
    loss_type = kwargs.get('loss', 'ce')
    if loss_type != 'ce':
        print(f"Loss function: {loss_type}")
        if loss_type == 'combined':
            print(f"  - CE weight: {kwargs.get('ce_weight', 1.0)}")
            print(f"  - QWK weight: {kwargs.get('qwk_weight', 0.5)}")
            print(f"  - EMD weight: {kwargs.get('emd_weight', 0.0)}")
            print(f"  - CORAL weight: {kwargs.get('coral_weight', 0.0)}")
        elif loss_type == 'ordinal_ce':
            print(f"  - Ordinal alpha: {kwargs.get('ordinal_alpha', 1.0)}")
    
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
            elif key == "loss" and value is not None:
                cmd.extend([f"--{key}", value])
            elif key in ["ce_weight", "qwk_weight", "emd_weight", "coral_weight", "ordinal_alpha"] and value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        success = run_command(cmd, f"Training {model.upper()}")
        results['training'][model] = success
    
    # 2. Evaluation phase (enhanced with advanced data collection)
    print(f"\n{'='*20} EVALUATION PHASE {'='*20}")
    for model in models:
        if results['training'][model]:
            model_path = f"save_models/best_{model}_{dataset}.pth"
            
            if os.path.exists(model_path):
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", model_path,
                    "--dataset", dataset,
                    "--regenerate_plots", "True"  # Ensure fresh plot data generation
                ]
                
                # Add additional arguments
                for key, value in kwargs.items():
                    if key == "batch_size" and value is not None:
                        cmd.extend([f"--{key}", str(value)])
                    elif key == "device" and value is not None:
                        cmd.extend([f"--{key}", value])
                
                success = run_command(cmd, f"Evaluating {model.upper()} with enhanced metrics")
                results['evaluation'][model] = success
            else:
                print(f"❌ Model file not found: {model_path}")
                results['evaluation'][model] = False
        else:
            print(f"⏭️  Skipping evaluation for {model} (training failed)")
            results['evaluation'][model] = False
    
    # 3. Plotting phase (enhanced visualization system)
    print(f"\n{'='*20} PLOTTING PHASE {'='*20}")
    if any(results['evaluation'].values()):  # Check evaluation success, not just training
        cmd = [
            sys.executable, "utils/plot_metrics.py"
        ]
        success = run_command(cmd, "Generating comprehensive visualizations (9 plots)")
        results['plotting'] = success
        
        if success:
            print("📊 Generated plots:")
            print("  • Training metrics with highlighted means and IQR bands")
            print("  • Test metrics comparison")
            print("  • Training vs test performance")
            print("  • Categorical breakdown")
            print("  • Confusion matrices with percentage coloring")
            print("  • Ordinal distance distribution")
            print("  • Category transition matrices")
            print("  • ROC curves per category")
            print("  • Calibration curves")
    else:
        print("⏭️  Skipping plotting (no successful evaluation)")
        results['plotting'] = False
    
    # 4. IRT Analysis phase (temporal analysis with static hit rate)
    print(f"\n{'='*20} IRT ANALYSIS PHASE {'='*20}")
    if any(results['evaluation'].values()):  # Require successful evaluation
        cmd = [
            sys.executable, "analysis/irt_analysis.py",
            "--dataset", dataset,
            "--analysis_types", "recovery", "temporal"
        ]
        success = run_command(cmd, "Running IRT temporal analysis with static hit rate selection")
        results['irt_analysis'] = success
        
        if success:
            print("🧠 Generated IRT analysis:")
            print("  • Parameter recovery analysis")
            print("  • Temporal heatmaps (theta parameters)")
            print("  • GPCM probability heatmaps")
            print("  • Static hit rate student selection")
            print("  • Temporal parameter trajectories")
    else:
        print("⏭️  Skipping IRT analysis (no successful evaluation)")
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
    parser = argparse.ArgumentParser(description='Deep-GPCM Enhanced Pipeline - Training, Evaluation, Visualization & IRT Analysis')
    
    # By default, run everything with enhanced features
    parser.add_argument('--action', choices=['pipeline', 'train', 'evaluate', 'plot', 'irt'], 
                       default='pipeline', 
                       help='Action: pipeline=complete enhanced pipeline, train=training only, '
                            'evaluate=enhanced evaluation+plots, plot=visualization only, irt=IRT analysis only')
    
    # Model and dataset selection
    parser.add_argument('--models', nargs='+', 
                       choices=['deep_gpcm', 'attn_gpcm', 'coral', 'coral_gpcm'], 
                       default=['deep_gpcm', 'attn_gpcm'], 
                       help='Models to train/evaluate (supports CORAL models)')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--cv_folds', type=int, default=5, help='CV folds (0 = no CV)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
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
            device=args.device,
            loss=args.loss,
            ce_weight=args.ce_weight,
            qwk_weight=args.qwk_weight,
            emd_weight=args.emd_weight,
            coral_weight=args.coral_weight,
            ordinal_alpha=args.ordinal_alpha
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
            
            # Add loss function arguments
            if args.loss != 'ce':
                cmd.extend(["--loss", args.loss])
            if args.loss == 'combined':
                cmd.extend(["--ce_weight", str(args.ce_weight)])
                cmd.extend(["--qwk_weight", str(args.qwk_weight)])
                cmd.extend(["--emd_weight", str(args.emd_weight)])
                cmd.extend(["--coral_weight", str(args.coral_weight)])
            elif args.loss == 'ordinal_ce':
                cmd.extend(["--ordinal_alpha", str(args.ordinal_alpha)])
            
            run_command(cmd, f"Training {model.upper()}")
    
    elif args.action == 'evaluate':
        # Evaluation only (enhanced with advanced data collection)
        print(f"\n🧪 ENHANCED EVALUATION ONLY")
        
        for model in args.models:
            model_path = args.model_path or f"save_models/best_{model}_{args.dataset}.pth"
            
            if os.path.exists(model_path):
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", model_path,
                    "--dataset", args.dataset,
                    "--batch_size", str(args.batch_size),
                    "--regenerate_plots", "True"  # Enhanced data collection for plotting
                ]
                if args.device:
                    cmd.extend(["--device", args.device])
                
                run_command(cmd, f"Enhanced evaluation: {model.upper()} (with advanced metrics)")
            else:
                print(f"❌ Model not found: {model_path}")
        
        # Auto-generate plots after evaluation
        print(f"\n📊 AUTO-GENERATING PLOTS")
        cmd = [sys.executable, "utils/plot_metrics.py"]
        run_command(cmd, "Generating comprehensive visualizations")
    
    elif args.action == 'plot':
        # Plotting only (enhanced visualization system)
        print(f"\n📊 ENHANCED PLOTTING ONLY")
        cmd = [sys.executable, "utils/plot_metrics.py"]
        success = run_command(cmd, "Generating 9 comprehensive visualizations")
        if success:
            print("✅ Generated all plots with enhanced features:")
            print("  • Training metrics with highlighted means and IQR bands")
            print("  • Confusion matrices with percentage coloring")
            print("  • ROC curves and calibration analysis")
    
    elif args.action == 'irt':
        # IRT Analysis only (temporal analysis with static hit rate)
        print(f"\n🧠 IRT TEMPORAL ANALYSIS ONLY")
        cmd = [
            sys.executable, "analysis/irt_analysis.py",
            "--dataset", args.dataset,
            "--analysis_types", "recovery", "temporal"
        ]
        success = run_command(cmd, "Running IRT analysis with static hit rate selection")
        if success:
            print("✅ Generated IRT analysis with enhanced features:")
            print("  • Static hit rate student selection (not temporal)")
            print("  • Clean temporal heatmaps without 'cherry-picked' references")
    
    print("\n🎯 Enhanced main runner completed!")


if __name__ == "__main__":
    main()