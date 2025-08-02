#!/usr/bin/env python3
"""
Cleanup and migration script for Deep-GPCM project
Removes existing files and performs migration to new structure
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.migrate_structure import StructureMigrator


def cleanup_existing_files(base_dir: str = ".", dry_run: bool = True):
    """Remove existing model and result files before migration"""
    base_path = Path(base_dir)
    
    print("="*60)
    print("CLEANING UP EXISTING FILES")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL CLEANUP'}")
    print()
    
    files_to_remove = []
    
    # Find model files
    save_models_dir = base_path / 'save_models'
    if save_models_dir.exists():
        model_files = list(save_models_dir.glob("*.pth"))
        files_to_remove.extend(model_files)
        print(f"Found {len(model_files)} model files in save_models/")
    
    saved_models_dir = base_path / 'saved_models'
    if saved_models_dir.exists():
        new_model_files = []
        for dataset_dir in saved_models_dir.iterdir():
            if dataset_dir.is_dir():
                new_model_files.extend(list(dataset_dir.glob("*.pth")))
        files_to_remove.extend(new_model_files)
        print(f"Found {len(new_model_files)} model files in saved_models/")
    
    # Find result files
    results_dir = base_path / 'results'
    result_files = []
    
    if results_dir.exists():
        # Direct JSON files in train/
        train_dir = results_dir / 'train'
        if train_dir.exists():
            for file in train_dir.iterdir():
                if file.is_file() and file.suffix == '.json':
                    result_files.append(file)
        
        # Files in subdirectories
        for subdir in ['train', 'validation', 'test']:
            sub_path = results_dir / subdir
            if sub_path.exists():
                for dataset_dir in sub_path.iterdir():
                    if dataset_dir.is_dir():
                        dataset_files = list(dataset_dir.glob("*.json"))
                        result_files.extend(dataset_files)
        
    files_to_remove.extend(result_files)
    print(f"Found {len(result_files)} result files")
    
    print(f"\nTotal files to remove: {len(files_to_remove)}")
    
    if files_to_remove:
        print("\nFiles to be removed:")
        for i, file in enumerate(files_to_remove[:10]):  # Show first 10
            print(f"  {file.relative_to(base_path)}")
        if len(files_to_remove) > 10:
            print(f"  ... and {len(files_to_remove) - 10} more files")
    
    if not dry_run and files_to_remove:
        # Create backup before removal
        backup_dir = base_path / 'cleanup_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nBacking up files to: {backup_dir}")
        
        removed_count = 0
        for file in files_to_remove:
            try:
                # Backup file
                rel_path = file.relative_to(base_path)
                backup_path = backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, backup_path)
                
                # Remove file
                file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
        
        print(f"\nRemoved {removed_count} files")
        
        # Clean up empty directories
        for dir_path in [save_models_dir, saved_models_dir]:
            if dir_path.exists():
                try:
                    # Remove empty subdirectories
                    for subdir in dir_path.iterdir():
                        if subdir.is_dir() and not any(subdir.iterdir()):
                            subdir.rmdir()
                    # Remove main dir if empty
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        print(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error cleaning directory {dir_path}: {e}")
    
    return len(files_to_remove)


def main():
    parser = argparse.ArgumentParser(description='Clean up and migrate Deep-GPCM files')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--cleanup-only', action='store_true', help='Only clean up, no migration')
    parser.add_argument('--migrate-only', action='store_true', help='Only migrate, no cleanup')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backups')
    
    args = parser.parse_args()
    
    if not args.migrate_only:
        # Step 1: Clean up existing files
        print("STEP 1: CLEANUP")
        print("-"*60)
        files_removed = cleanup_existing_files(args.base_dir, args.dry_run)
        print()
    
    if not args.cleanup_only:
        # Step 2: Run migration
        print("STEP 2: MIGRATION")
        print("-"*60)
        
        migrator = StructureMigrator(
            base_dir=args.base_dir,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
        
        migrator.run_migration()
    
    print("\n" + "="*60)
    print("CLEANUP AND MIGRATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()