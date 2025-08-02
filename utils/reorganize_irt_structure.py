#!/usr/bin/env python3
"""
Reorganize IRT directory structure:
- results/irt/ -> results/irt_plots/
- dataset/irt_plots/ -> dataset/model_specs/
"""

import os
import shutil
from pathlib import Path
import argparse


def reorganize_irt_structure(results_dir="results", dry_run=False):
    """
    Reorganize the IRT directory structure.
    
    Args:
        results_dir: Base results directory
        dry_run: If True, only show what would be done without making changes
    """
    results_path = Path(results_dir)
    old_irt_dir = results_path / "irt"
    new_irt_dir = results_path / "irt_plots"
    
    if not old_irt_dir.exists():
        print(f"Source directory {old_irt_dir} does not exist. Nothing to do.")
        return
    
    print(f"Reorganizing IRT directory structure...")
    print(f"From: {old_irt_dir}")
    print(f"To: {new_irt_dir}")
    
    if dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")
    
    # Step 1: Create new irt_plots directory if it doesn't exist
    if not dry_run:
        new_irt_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 2: Process each item in the old irt directory
    for item in old_irt_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if item.name == "irt_plots":
                # Skip the old irt_plots directory at root level - we'll handle it specially
                continue
            else:
                # This is a dataset directory
                dataset_name = item.name
                old_dataset_dir = item
                new_dataset_dir = new_irt_dir / dataset_name
                
                print(f"\nProcessing dataset: {dataset_name}")
                
                if not dry_run:
                    new_dataset_dir.mkdir(exist_ok=True, parents=True)
                
                # Move all files from old dataset dir to new dataset dir
                for file in old_dataset_dir.iterdir():
                    if file.is_file():
                        src = file
                        dst = new_dataset_dir / file.name
                        print(f"  Moving: {file.name}")
                        if not dry_run:
                            shutil.move(str(src), str(dst))
                    elif file.is_dir() and file.name == "irt_plots":
                        # Rename irt_plots to model_specs
                        old_irt_plots = file
                        new_model_specs = new_dataset_dir / "model_specs"
                        print(f"  Renaming: irt_plots/ -> model_specs/")
                        if not dry_run:
                            if new_model_specs.exists():
                                shutil.rmtree(new_model_specs)
                            shutil.move(str(old_irt_plots), str(new_model_specs))
        
        elif item.is_file():
            # Move root-level files
            src = item
            dst = new_irt_dir / item.name
            print(f"\nMoving root file: {item.name}")
            if not dry_run:
                shutil.move(str(src), str(dst))
    
    # Step 3: Handle the old root-level irt_plots directory if it exists
    old_root_irt_plots = old_irt_dir / "irt_plots"
    if old_root_irt_plots.exists() and old_root_irt_plots.is_dir():
        # These should be moved to a default dataset directory or handled specially
        print(f"\nFound root-level irt_plots directory - moving to 'default' dataset")
        default_dataset_dir = new_irt_dir / "default" / "model_specs"
        if not dry_run:
            default_dataset_dir.mkdir(exist_ok=True, parents=True)
            for model_dir in old_root_irt_plots.iterdir():
                if model_dir.is_dir():
                    shutil.move(str(model_dir), str(default_dataset_dir / model_dir.name))
    
    # Step 4: Remove the old irt directory if empty
    if not dry_run:
        try:
            old_irt_dir.rmdir()
            print(f"\nRemoved empty directory: {old_irt_dir}")
        except OSError:
            print(f"\nCould not remove {old_irt_dir} - not empty")
    
    print("\nReorganization complete!")
    
    # Show new structure
    print("\nNew structure:")
    if new_irt_dir.exists():
        for dataset_dir in sorted(new_irt_dir.iterdir()):
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                print(f"\n  {new_irt_dir.name}/{dataset_dir.name}/")
                
                # List files
                files = [f for f in dataset_dir.iterdir() if f.is_file()]
                if files:
                    for f in sorted(files)[:3]:
                        print(f"    - {f.name}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more files")
                
                # Check for model_specs
                model_specs_dir = dataset_dir / "model_specs"
                if model_specs_dir.exists():
                    model_dirs = [d for d in model_specs_dir.iterdir() if d.is_dir()]
                    print(f"    model_specs/ ({len(model_dirs)} models)")
                    for model_dir in sorted(model_dirs):
                        print(f"      - {model_dir.name}/")


def main():
    parser = argparse.ArgumentParser(description='Reorganize IRT directory structure')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    reorganize_irt_structure(args.results_dir, args.dry_run)


if __name__ == "__main__":
    main()