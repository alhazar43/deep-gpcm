#!/usr/bin/env python3
"""
Cleanup utility for removing dataset-specific results.
Supports both new structure (saved_models/dataset/) and legacy structure.
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Optional
import json
import sys


class ResultsCleaner:
    """Handles cleanup of dataset-specific results across all directories."""
    
    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir)
        self.result_dirs = {
            'train': self.base_dir / 'results' / 'train',
            'validation': self.base_dir / 'results' / 'validation',
            'valid': self.base_dir / 'results' / 'valid',  # Support both naming conventions
            'test': self.base_dir / 'results' / 'test',
            'plots': self.base_dir / 'results' / 'plots',
            'irt_plots': self.base_dir / 'results' / 'irt_plots',
            'models_new': self.base_dir / 'saved_models',  # New structure
            'models_legacy': self.base_dir / 'save_models'  # Legacy structure
        }
        
    def find_dataset_files(self, dataset: str) -> Dict[str, List[Path]]:
        """Find all files related to a specific dataset."""
        files_to_clean = {
            'train': [],
            'validation': [],
            'test': [],
            'plots': [],
            'irt_plots': [],
            'models': []
        }
        
        # Find train/validation/test results (nested structure)
        for result_type in ['train', 'validation', 'valid', 'test']:
            result_dir = self.result_dirs.get(result_type)
            if result_dir and result_dir.exists():
                dataset_dir = result_dir / dataset
                if dataset_dir.exists() and dataset_dir.is_dir():
                    # Map 'valid' to 'validation' for consistency
                    key = 'validation' if result_type == 'valid' else result_type
                    files_to_clean[key].append(dataset_dir)
        
        # Find plot files containing dataset name
        plots_dir = self.result_dirs['plots']
        if plots_dir.exists():
            plot_files = list(plots_dir.glob(f'*_{dataset}_*.png'))
            plot_files.extend(list(plots_dir.glob(f'*_{dataset}.png')))
            plot_files.extend(list(plots_dir.glob(f'{dataset}_*.png')))
            files_to_clean['plots'].extend(plot_files)
        
        # Find IRT plots directory
        irt_dir = self.result_dirs['irt_plots'] / dataset
        if irt_dir.exists() and irt_dir.is_dir():
            files_to_clean['irt_plots'].append(irt_dir)
        
        # Find model files (new structure)
        models_new_dir = self.result_dirs['models_new'] / dataset
        if models_new_dir.exists() and models_new_dir.is_dir():
            files_to_clean['models'].append(models_new_dir)
        
        # Find model files (legacy structure)
        models_legacy_dir = self.result_dirs['models_legacy']
        if models_legacy_dir.exists():
            model_files = list(models_legacy_dir.glob(f'*_{dataset}.pth'))
            model_files.extend(list(models_legacy_dir.glob(f'*_{dataset}_*.pth')))
            files_to_clean['models'].extend(model_files)
        
        return files_to_clean
    
    def get_all_datasets(self) -> Set[str]:
        """Detect all datasets present in the results."""
        datasets = set()
        
        # Check train/validation/test directories
        for result_type in ['train', 'validation', 'valid', 'test']:
            result_dir = self.result_dirs.get(result_type)
            if result_dir and result_dir.exists():
                for item in result_dir.iterdir():
                    if item.is_dir():
                        datasets.add(item.name)
        
        # Check saved_models directory
        models_new_dir = self.result_dirs['models_new']
        if models_new_dir.exists():
            for item in models_new_dir.iterdir():
                if item.is_dir():
                    datasets.add(item.name)
        
        # Check IRT plots directory
        irt_dir = self.result_dirs['irt_plots']
        if irt_dir.exists():
            for item in irt_dir.iterdir():
                if item.is_dir():
                    datasets.add(item.name)
        
        return datasets
    
    def create_backup(self, files_dict: Dict[str, List[Path]], dataset: str) -> Optional[Path]:
        """Create a backup of files before deletion."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.base_dir / 'backups' / f'cleanup_{dataset}_{timestamp}'
        
        # Check if there's anything to backup
        total_files = sum(len(files) for files in files_dict.values())
        if total_files == 0:
            return None
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifest file
        manifest = {
            'dataset': dataset,
            'timestamp': timestamp,
            'files_backed_up': {}
        }
        
        for category, paths in files_dict.items():
            if paths:
                category_dir = backup_dir / category
                category_dir.mkdir(exist_ok=True)
                manifest['files_backed_up'][category] = []
                
                for path in paths:
                    if path.exists():
                        if path.is_dir():
                            dest = category_dir / path.name
                            shutil.copytree(path, dest)
                            manifest['files_backed_up'][category].append(str(path))
                        else:
                            dest = category_dir / path.name
                            shutil.copy2(path, dest)
                            manifest['files_backed_up'][category].append(str(path))
        
        # Save manifest
        with open(backup_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return backup_dir
    
    def clean_dataset(self, dataset: str, dry_run: bool = False, backup: bool = True) -> Dict[str, int]:
        """Clean all results for a specific dataset."""
        files_to_clean = self.find_dataset_files(dataset)
        
        # Count files/directories
        counts = {
            'directories': 0,
            'files': 0,
            'total_size': 0
        }
        
        # Calculate what would be deleted
        for category, paths in files_to_clean.items():
            for path in paths:
                if path.exists():
                    if path.is_dir():
                        counts['directories'] += 1
                        # Calculate directory size
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = Path(root) / file
                                if file_path.exists():
                                    counts['total_size'] += file_path.stat().st_size
                    else:
                        counts['files'] += 1
                        counts['total_size'] += path.stat().st_size
        
        if dry_run:
            print(f"\nDRY RUN - Would delete for dataset '{dataset}':")
            for category, paths in files_to_clean.items():
                if paths:
                    print(f"\n{category.upper()}:")
                    for path in paths:
                        if path.exists():
                            print(f"  - {path}")
            
            print(f"\nSummary: {counts['directories']} directories, {counts['files']} files")
            print(f"Total size: {counts['total_size'] / (1024*1024):.2f} MB")
            return counts
        
        # Create backup if requested
        backup_path = None
        if backup and (counts['directories'] > 0 or counts['files'] > 0):
            print(f"\nCreating backup...")
            backup_path = self.create_backup(files_to_clean, dataset)
            if backup_path:
                print(f"Backup created at: {backup_path}")
        
        # Perform actual deletion
        for category, paths in files_to_clean.items():
            for path in paths:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    else:
                        path.unlink()
                        print(f"Removed file: {path}")
        
        return counts
    
    def clean_all_datasets(self, dry_run: bool = False, backup: bool = True) -> Dict[str, Dict[str, int]]:
        """Clean results for all detected datasets."""
        datasets = self.get_all_datasets()
        
        if not datasets:
            print("No datasets found to clean.")
            return {}
        
        print(f"Found {len(datasets)} datasets: {', '.join(sorted(datasets))}")
        
        results = {}
        for dataset in sorted(datasets):
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset}")
            print('='*60)
            results[dataset] = self.clean_dataset(dataset, dry_run=dry_run, backup=backup)
        
        return results


def main():
    """Standalone CLI for cleanup utility."""
    parser = argparse.ArgumentParser(
        description='Clean dataset-specific results from Deep-GPCM project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run for specific dataset
  python clean_res.py --dataset synthetic_OC --dry-run
  
  # Clean specific dataset with backup
  python clean_res.py --dataset synthetic_OC
  
  # Clean all datasets without confirmation
  python clean_res.py --all --no-confirm
  
  # Clean without creating backup
  python clean_res.py --dataset synthetic_OC --no-backup
        """
    )
    
    parser.add_argument('--dataset', type=str, help='Specific dataset to clean')
    parser.add_argument('--all', action='store_true', help='Clean all detected datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup before deletion')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory of the project')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all must be specified")
    
    cleaner = ResultsCleaner(base_dir=args.base_dir)
    
    # Confirmation prompt
    if not args.dry_run and not args.no_confirm:
        if args.all:
            datasets = cleaner.get_all_datasets()
            print(f"This will delete results for ALL {len(datasets)} datasets: {', '.join(sorted(datasets))}")
        else:
            print(f"This will delete all results for dataset: {args.dataset}")
        
        response = input("\nAre you sure? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            sys.exit(0)
    
    # Perform cleanup
    try:
        if args.all:
            results = cleaner.clean_all_datasets(dry_run=args.dry_run, backup=not args.no_backup)
            
            if not args.dry_run:
                print("\n" + "="*60)
                print("CLEANUP SUMMARY")
                print("="*60)
                total_dirs = sum(r['directories'] for r in results.values())
                total_files = sum(r['files'] for r in results.values())
                total_size = sum(r['total_size'] for r in results.values())
                print(f"Total: {total_dirs} directories, {total_files} files")
                print(f"Total size freed: {total_size / (1024*1024):.2f} MB")
        else:
            counts = cleaner.clean_dataset(args.dataset, dry_run=args.dry_run, backup=not args.no_backup)
            
            if not args.dry_run and (counts['directories'] > 0 or counts['files'] > 0):
                print(f"\nSuccessfully cleaned dataset '{args.dataset}'")
                print(f"Removed: {counts['directories']} directories, {counts['files']} files")
                print(f"Size freed: {counts['total_size'] / (1024*1024):.2f} MB")
            elif not args.dry_run:
                print(f"\nNo files found for dataset '{args.dataset}'")
                
    except Exception as e:
        print(f"\nError during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()