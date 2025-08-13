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

# from utils.metrics import ensure_results_dirs  # No longer needed with new structure
from utils.path_utils import get_path_manager, find_best_model, ensure_directories
from utils.clean_res import ResultsCleaner
from models.factory import get_all_model_types, get_model_loss_config, validate_model_type


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


def run_complete_pipeline(models=None, dataset='synthetic_OC', 
                         epochs=30, n_folds=5, cv=False, **kwargs):
    """Run the complete Deep-GPCM pipeline with dynamic model discovery."""
    
    # Use all available models if none specified
    if models is None:
        models = get_all_model_types()
        print("📋 Using all available models from factory registry")
    
    # Validate all models exist in factory
    validated_models = []
    for model in models:
        if validate_model_type(model):
            validated_models.append(model)
        else:
            print(f"⚠️  Warning: Model '{model}' not found in factory registry, skipping")
    
    models = validated_models
    if not models:
        raise ValueError("No valid models found in factory registry")
    
    print("=" * 80)
    print("DEEP-GPCM COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    if n_folds > 0:
        if cv:
            print(f"Cross-Validation: {n_folds}-fold with hyperparameter tuning")
        else:
            print(f"K-Fold Training: {n_folds}-fold (no hyperparameter tuning)")
    else:
        print("Training: Single run (no folds)")
    
    # Print model-specific loss configurations from factory
    print("\n🔧 Model Configurations from Factory:")
    for model in models:
        loss_config = get_model_loss_config(model)
        if loss_config:
            loss_type = loss_config.get('type', 'ce')
            print(f"  • {model}: {loss_type}", end="")
            if loss_type == 'combined':
                weights = []
                for weight_key in ['ce_weight', 'focal_weight', 'qwk_weight', 'coral_weight']:
                    if weight_key in loss_config:
                        weights.append(f"{weight_key.replace('_weight', '')}={loss_config[weight_key]}")
                if weights:
                    print(f" ({', '.join(weights)})")
                else:
                    print()
            else:
                print()
        else:
            print(f"  • {model}: ce (default)")
    
    print()
    
    # Ensure results directories exist (new structure)
    path_manager = get_path_manager()
    ensure_directories(dataset)
    
    results = {'training': {}, 'evaluation': {}}
    
    # 1. Training phase with CV (factory-based loss configurations)
    print(f"\n{'='*20} TRAINING PHASE {'='*20}")
    for model in models:
        cmd = [
            sys.executable, "train.py",
            "--model", model,
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--n_folds", str(n_folds)
        ]
        
        # Loss configuration is now handled automatically by train.py factory system
        loss_config = get_model_loss_config(model)
        loss_type = loss_config.get('type', 'ce')
        print(f"  ⚙️  {model}: Using factory {loss_type} loss configuration")
        
        # Add cv flag if enabled
        if cv:
            cmd.append("--cv")
        
        # Add common arguments (but skip loss-related if not specified by user)
        for key, value in kwargs.items():
            if key in ["batch_size", "lr", "seed"] and value is not None:
                cmd.extend([f"--{key}", str(value)])
            elif key == "device" and value is not None:
                cmd.extend([f"--{key}", value])
            # Skip loss arguments - handled by factory configuration above
        
        success = run_command(cmd, f"Training {model.upper()}")
        results['training'][model] = success
    
    # 2. Evaluation phase (enhanced with advanced data collection)
    print(f"\n{'='*20} EVALUATION PHASE {'='*20}")
    for model in models:
        if results['training'][model]:
            # Try new structure first, fall back to legacy
            model_path = find_best_model(model, dataset)
            
            if model_path and model_path.exists():
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", str(model_path),
                    "--dataset", dataset,
                    "--regenerate_plots"  # Ensure fresh plot data generation
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
            sys.executable, "utils/plot_metrics.py", "--dataset", dataset
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
    parser.add_argument('--action', choices=['pipeline', 'train', 'evaluate', 'plot', 'irt', 'clean'], 
                       default='pipeline', 
                       help='Action: pipeline=complete enhanced pipeline, train=training only, '
                            'evaluate=enhanced evaluation+plots, plot=visualization only, irt=IRT analysis only, '
                            'clean=cleanup results and models')
    
    # Model and dataset selection (dynamic from factory registry)
    available_models = get_all_model_types()
    parser.add_argument('--models', nargs='+', 
                       choices=available_models, 
                       default=available_models, 
                       help='Pipeline models from factory registry')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name (e.g., synthetic_OC, synthetic_4000_200_2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for k-fold training (0 = no folds)')
    parser.add_argument('--cv', action='store_true', help='Enable cross-validation with hyperparameter tuning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    # Loss configurations are now managed automatically by factory system
    
    
    # Individual control
    parser.add_argument('--model_path', help='Specific model path for evaluation')
    
    # Cleanup options
    parser.add_argument('--clean', action='store_true', help='Clean existing results before running pipeline')
    parser.add_argument('--clean-only', action='store_true', help='Only clean results without running pipeline')
    parser.add_argument('--clean-all', action='store_true', help='Clean results for all datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation during cleanup')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt during cleanup')
    
    args = parser.parse_args()
    
    # Handle --clean-only flag first
    if args.clean_only:
        print(f"\n🧹 CLEANUP ONLY MODE")
        cleaner = ResultsCleaner()
        
        if args.clean_all:
            # Clean all datasets
            if not args.dry_run and not args.no_confirm:
                datasets = cleaner.get_all_datasets()
                print(f"This will delete results for ALL {len(datasets)} datasets: {', '.join(sorted(datasets))}")
                response = input("\nAre you sure? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Cleanup cancelled.")
                    return
            
            cleaner.clean_all_datasets(dry_run=args.dry_run, backup=not args.no_backup)
        else:
            if not args.dataset:
                print("Please specify a dataset with --dataset or use --clean-all for all datasets")
            else:
                # Clean specific dataset
                if not args.dry_run and not args.no_confirm:
                    print(f"This will delete all results for dataset: {args.dataset}")
                    response = input("\nAre you sure? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print("Cleanup cancelled.")
                        return
                
                cleaner.clean_dataset(args.dataset, dry_run=args.dry_run, backup=not args.no_backup)
        return  # Exit after cleanup
    
    if args.action == 'pipeline':
        # Handle --clean flag BEFORE pipeline execution
        if args.clean and args.dataset:
            print(f"\n🧹 PRE-PIPELINE CLEANUP")
            cleaner = ResultsCleaner()
            
            if not args.dry_run and not args.no_confirm:
                print(f"This will delete all existing results for dataset: {args.dataset}")
                response = input("\nAre you sure? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Cleanup cancelled.")
                    return
            
            cleaner.clean_dataset(args.dataset, dry_run=args.dry_run, backup=not args.no_backup)
            print("\n✅ Cleanup complete. Starting fresh pipeline...\n")
        
        # Run complete pipeline
        run_complete_pipeline(
            models=args.models,
            dataset=args.dataset,
            epochs=args.epochs,
            n_folds=args.n_folds,
            cv=args.cv,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        )
    
    elif args.action == 'train':
        # Training only
        print(f"\n🚀 TRAINING ONLY")
        # Note: Directory creation handled by path_manager
        
        for model in args.models:
            cmd = [
                sys.executable, "train.py",
                "--model", model,
                "--dataset", args.dataset,
                "--epochs", str(args.epochs),
                "--n_folds", str(args.n_folds),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed)
            ]
            
            if args.device:
                cmd.extend(["--device", args.device])
            
            # Add cv flag if enabled
            if args.cv:
                cmd.append("--cv")
            
            # Loss configuration is now handled automatically by train.py factory system
            loss_config = get_model_loss_config(model)
            loss_type = loss_config.get('type', 'ce')
            print(f"  ⚙️  {model}: Using factory {loss_type} loss configuration")
            
            run_command(cmd, f"Training {model.upper()}")
    
    elif args.action == 'evaluate':
        # Evaluation only (enhanced with advanced data collection)
        print(f"\n🧪 ENHANCED EVALUATION ONLY")
        
        for model in args.models:
            if args.model_path:
                model_path = Path(args.model_path)
            else:
                model_path = find_best_model(model, args.dataset)
            
            if model_path and model_path.exists():
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", str(model_path),
                    "--dataset", args.dataset,
                    "--batch_size", str(args.batch_size),
                    "--regenerate_plots"  # Enhanced data collection for plotting
                ]
                if args.device:
                    cmd.extend(["--device", args.device])
                
                run_command(cmd, f"Enhanced evaluation: {model.upper()} (with advanced metrics)")
            else:
                print(f"❌ Model not found: {model_path}")
        
        # Auto-generate plots after evaluation
        print(f"\n📊 AUTO-GENERATING PLOTS")
        cmd = [sys.executable, "utils/plot_metrics.py", "--dataset", args.dataset]
        run_command(cmd, "Generating comprehensive visualizations")
    
    elif args.action == 'plot':
        # Plotting only (enhanced visualization system)
        print(f"\n📊 ENHANCED PLOTTING ONLY")
        cmd = [sys.executable, "utils/plot_metrics.py", "--dataset", args.dataset]
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
    
    elif args.action == 'clean':
        # Cleanup action
        print(f"\n🧹 CLEANUP MODE")
        cleaner = ResultsCleaner()
        
        if args.clean_all:
            # Clean all datasets
            if not args.dry_run and not args.no_confirm:
                datasets = cleaner.get_all_datasets()
                print(f"This will delete results for ALL {len(datasets)} datasets: {', '.join(sorted(datasets))}")
                response = input("\nAre you sure? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Cleanup cancelled.")
                    return
            
            cleaner.clean_all_datasets(dry_run=args.dry_run, backup=not args.no_backup)
        else:
            if not args.dataset:
                print("Please specify a dataset with --dataset or use --clean-all for all datasets")
            else:
                # Clean specific dataset
                if not args.dry_run and not args.no_confirm:
                    print(f"This will delete all results for dataset: {args.dataset}")
                    response = input("\nAre you sure? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print("Cleanup cancelled.")
                        return
                
                cleaner.clean_dataset(args.dataset, dry_run=args.dry_run, backup=not args.no_backup)
    
    
    print("\n🎯 Enhanced main runner completed!")


if __name__ == "__main__":
    main()