#!/usr/bin/env python3
"""
Organize existing plots into dataset-specific directories
"""

import os
import shutil
from pathlib import Path
import argparse


def organize_plots(results_dir="results", dataset="synthetic_OC", move=False):
    """
    Organize plots into dataset-specific directories.
    
    Args:
        results_dir: Base results directory
        dataset: Dataset name for organization
        move: If True, move files. If False, copy files.
    """
    results_path = Path(results_dir)
    
    # Create dataset-specific directories
    plots_dataset_dir = results_path / "plots" / dataset
    irt_dataset_dir = results_path / "irt_plots" / dataset
    
    plots_dataset_dir.mkdir(parents=True, exist_ok=True)
    irt_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Move/copy main plots
    main_plots_dir = results_path / "plots"
    moved_files = []
    
    # List of plot files in main plots directory
    plot_patterns = [
        "training_metrics.png",
        "test_metrics.png", 
        "training_vs_test_comparison.png",
        "categorical_breakdown.png",
        "confusion_matrices_test.png",
        "ordinal_distance_distribution_test.png",
        "category_transitions_test.png",
        "roc_curves_per_category_test.png",
        "calibration_curves_test.png",
        "attention_weights_test.png",
        "time_series_performance_test.png"
    ]
    
    for pattern in plot_patterns:
        src = main_plots_dir / pattern
        if src.exists() and src.is_file():
            dst = plots_dataset_dir / pattern
            if move:
                shutil.move(str(src), str(dst))
                action = "Moved"
            else:
                shutil.copy2(str(src), str(dst))
                action = "Copied"
            moved_files.append(f"{action}: {src} -> {dst}")
    
    # Move/copy IRT plots
    irt_plots_dir = results_path / "irt_plots"
    irt_patterns = [
        "parameter_recovery.png",
        "temporal_analysis.png",
        "temporal_theta_heatmap.png",
        "temporal_gpcm_probs_heatmap.png",
        "temporal_parameters_combined.png",
        "irt_analysis_summary.txt"
    ]
    
    for pattern in irt_patterns:
        src = irt_plots_dir / pattern
        if src.exists() and src.is_file():
            dst = irt_dataset_dir / pattern
            if move:
                shutil.move(str(src), str(dst))
                action = "Moved"
            else:
                shutil.copy2(str(src), str(dst))
                action = "Copied"
            moved_files.append(f"{action}: {src} -> {dst}")
    
    # Move/copy model-specific plots (now in model_specs subdirectory)
    model_specs_subdir = irt_plots_dir / "model_specs"
    if model_specs_subdir.exists() and model_specs_subdir.is_dir():
        dst_subdir = irt_dataset_dir / "model_specs"
        dst_subdir.mkdir(parents=True, exist_ok=True)
        
        # Move/copy each model's subdirectory
        for model_dir in model_specs_subdir.iterdir():
            if model_dir.is_dir():
                model_dst = dst_subdir / model_dir.name
                if move:
                    if model_dst.exists():
                        shutil.rmtree(model_dst)
                    shutil.move(str(model_dir), str(model_dst))
                    action = "Moved"
                else:
                    shutil.copytree(str(model_dir), str(model_dst), dirs_exist_ok=True)
                    action = "Copied"
                moved_files.append(f"{action} model plots: {model_dir.name} -> {model_dst}")
    
    print(f"\nOrganized {len(moved_files)} items for dataset '{dataset}':")
    for item in moved_files:
        print(f"  {item}")
    
    print(f"\nNew structure:")
    print(f"  - Main plots: {plots_dataset_dir}")
    print(f"  - IRT plots: {irt_dataset_dir}")
    print(f"  - Model specs: {irt_dataset_dir}/model_specs/")
    
    return moved_files


def list_dataset_plots(results_dir="results"):
    """List all plots organized by dataset."""
    results_path = Path(results_dir)
    
    # Check plots directory
    plots_dir = results_path / "plots"
    if plots_dir.exists():
        print("\nPlots organized by dataset:")
        for dataset_dir in sorted(plots_dir.iterdir()):
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                plot_files = list(dataset_dir.glob("*.png"))
                if plot_files:
                    print(f"\n  {dataset_dir.name}/ ({len(plot_files)} plots)")
                    for plot in sorted(plot_files)[:5]:  # Show first 5
                        print(f"    - {plot.name}")
                    if len(plot_files) > 5:
                        print(f"    ... and {len(plot_files) - 5} more")
    
    # Check IRT directory
    irt_dir = results_path / "irt_plots"
    if irt_dir.exists():
        print("\n\nIRT plots organized by dataset:")
        for dataset_dir in sorted(irt_dir.iterdir()):
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.') and dataset_dir.name != 'model_specs':
                plot_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.txt"))
                if plot_files:
                    print(f"\n  {dataset_dir.name}/ ({len(plot_files)} files)")
                    for plot in sorted(plot_files)[:5]:  # Show first 5
                        print(f"    - {plot.name}")
                    if len(plot_files) > 5:
                        print(f"    ... and {len(plot_files) - 5} more")
                
                # Check for model_specs subdirectory
                model_specs_subdir = dataset_dir / "model_specs"
                if model_specs_subdir.exists() and model_specs_subdir.is_dir():
                    model_dirs = [d for d in model_specs_subdir.iterdir() if d.is_dir()]
                    if model_dirs:
                        print(f"    model_specs/ ({len(model_dirs)} model directories)")
                        for model_dir in sorted(model_dirs):
                            model_plots = list(model_dir.glob("*.png"))
                            print(f"      - {model_dir.name}/ ({len(model_plots)} plots)")


def main():
    parser = argparse.ArgumentParser(description='Organize plots by dataset')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name for organization')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory path')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying')
    parser.add_argument('--list', action='store_true',
                        help='List current dataset organization')
    
    args = parser.parse_args()
    
    if args.list:
        list_dataset_plots(args.results_dir)
    else:
        organize_plots(args.results_dir, args.dataset, args.move)
        print("\nAfter organization:")
        list_dataset_plots(args.results_dir)


if __name__ == "__main__":
    main()