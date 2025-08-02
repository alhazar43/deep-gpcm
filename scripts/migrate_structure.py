#!/usr/bin/env python3
"""
Migration script for Deep-GPCM file structure
Migrates from legacy structure to new organized structure
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional, Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.path_utils import PathManager


class StructureMigrator:
    """Handles migration from legacy to new file structure"""
    
    def __init__(self, base_dir: str = ".", dry_run: bool = True, backup: bool = True):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.backup = backup
        self.path_manager = PathManager(base_dir)
        self.migration_log = []
        
    def log_action(self, action: str, details: str):
        """Log migration actions"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.migration_log.append(entry)
        print(f"[{action}] {details}")
    
    def backup_file(self, filepath: Path) -> Optional[Path]:
        """Create backup of file before migration"""
        if not self.backup or self.dry_run:
            return None
        
        backup_dir = self.base_dir / 'migration_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve relative path structure in backup
        rel_path = filepath.relative_to(self.base_dir)
        backup_path = backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def migrate_model_file(self, old_path: Path) -> bool:
        """Migrate a single model file"""
        info = self.path_manager.parse_legacy_filename(old_path)
        
        if not all(k in info for k in ['model', 'dataset']):
            self.log_action("SKIP", f"Cannot parse filename: {old_path.name}")
            return False
        
        # Determine new path
        new_path = self.path_manager.get_model_path(
            info['model'], 
            info['dataset'],
            fold=info.get('fold'),
            is_best=info.get('is_best', True),
            legacy=False
        )
        
        # Log the migration
        self.log_action("MIGRATE", f"{old_path.name} -> {new_path.relative_to(self.base_dir)}")
        
        if not self.dry_run:
            # Backup if requested
            if self.backup:
                backup_path = self.backup_file(old_path)
                if backup_path:
                    self.log_action("BACKUP", f"Created backup at {backup_path.relative_to(self.base_dir)}")
            
            # Perform migration
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            
        return True
    
    def migrate_result_file(self, old_path: Path) -> bool:
        """Migrate a single result file"""
        info = self.path_manager.parse_legacy_filename(old_path)
        
        if not all(k in info for k in ['model', 'dataset']):
            self.log_action("SKIP", f"Cannot parse filename: {old_path.name}")
            return False
        
        # Determine new path based on file type
        if info.get('is_test'):
            new_path = self.path_manager.get_result_path(
                'test', info['model'], info['dataset']
            )
        elif info.get('is_cv_summary'):
            new_path = self.path_manager.get_result_path(
                'validation', info['model'], info['dataset']
            )
        elif 'fold' in info:
            new_path = self.path_manager.get_result_path(
                'train', info['model'], info['dataset'], fold=info['fold']
            )
        else:
            new_path = self.path_manager.get_result_path(
                'train', info['model'], info['dataset']
            )
        
        # Log the migration
        self.log_action("MIGRATE", f"{old_path.name} -> {new_path.relative_to(self.base_dir)}")
        
        if not self.dry_run:
            # Backup if requested
            if self.backup:
                backup_path = self.backup_file(old_path)
                if backup_path:
                    self.log_action("BACKUP", f"Created backup at {backup_path.relative_to(self.base_dir)}")
            
            # Perform migration
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            
        return True
    
    def find_legacy_files(self) -> Dict[str, List[Path]]:
        """Find all files that need migration"""
        return self.path_manager.get_all_legacy_files()
    
    def generate_migration_report(self) -> Dict:
        """Generate a report of what will be migrated"""
        legacy_files = self.find_legacy_files()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_dir': str(self.base_dir),
            'dry_run': self.dry_run,
            'backup_enabled': self.backup,
            'summary': {
                'total_models': len(legacy_files['models']),
                'total_results': len(legacy_files['results']),
                'total_files': len(legacy_files['models']) + len(legacy_files['results'])
            },
            'models': [],
            'results': []
        }
        
        # Analyze model files
        for old_path in legacy_files['models']:
            info = self.path_manager.parse_legacy_filename(old_path)
            if all(k in info for k in ['model', 'dataset']):
                new_path = self.path_manager.get_model_path(
                    info['model'], 
                    info['dataset'],
                    fold=info.get('fold'),
                    is_best=info.get('is_best', True),
                    legacy=False
                )
                report['models'].append({
                    'old': str(old_path.relative_to(self.base_dir)),
                    'new': str(new_path.relative_to(self.base_dir)),
                    'info': info
                })
        
        # Analyze result files
        for old_path in legacy_files['results']:
            info = self.path_manager.parse_legacy_filename(old_path)
            if all(k in info for k in ['model', 'dataset']):
                if info.get('is_test'):
                    new_path = self.path_manager.get_result_path(
                        'test', info['model'], info['dataset']
                    )
                elif info.get('is_cv_summary'):
                    new_path = self.path_manager.get_result_path(
                        'validation', info['model'], info['dataset']
                    )
                elif 'fold' in info:
                    new_path = self.path_manager.get_result_path(
                        'train', info['model'], info['dataset'], fold=info['fold']
                    )
                else:
                    new_path = self.path_manager.get_result_path(
                        'train', info['model'], info['dataset']
                    )
                
                report['results'].append({
                    'old': str(old_path.relative_to(self.base_dir)),
                    'new': str(new_path.relative_to(self.base_dir)),
                    'info': info
                })
        
        return report
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        print("="*60)
        print("DEEP-GPCM FILE STRUCTURE MIGRATION")
        print("="*60)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'ACTUAL MIGRATION'}")
        print(f"Backup: {'ENABLED' if self.backup else 'DISABLED'}")
        print()
        
        # Find legacy files
        legacy_files = self.find_legacy_files()
        total_files = len(legacy_files['models']) + len(legacy_files['results'])
        
        if total_files == 0:
            print("No legacy files found. Migration not needed.")
            return True
        
        print(f"Found {len(legacy_files['models'])} model files")
        print(f"Found {len(legacy_files['results'])} result files")
        print(f"Total: {total_files} files to migrate")
        print()
        
        # Migrate models
        if legacy_files['models']:
            print("Migrating model files...")
            print("-"*40)
            success_count = 0
            for old_path in legacy_files['models']:
                if self.migrate_model_file(old_path):
                    success_count += 1
            print(f"Successfully migrated {success_count}/{len(legacy_files['models'])} model files")
            print()
        
        # Migrate results
        if legacy_files['results']:
            print("Migrating result files...")
            print("-"*40)
            success_count = 0
            for old_path in legacy_files['results']:
                if self.migrate_result_file(old_path):
                    success_count += 1
            print(f"Successfully migrated {success_count}/{len(legacy_files['results'])} result files")
            print()
        
        # Save migration log
        if not self.dry_run and self.migration_log:
            log_path = self.base_dir / 'migration_log.json'
            with open(log_path, 'w') as f:
                json.dump({
                    'migration_date': datetime.now().isoformat(),
                    'actions': self.migration_log
                }, f, indent=2)
            print(f"Migration log saved to: {log_path}")
        
        # Clean up empty legacy directories (only if not dry run)
        if not self.dry_run:
            # Check if save_models is empty
            legacy_model_dir = self.base_dir / 'save_models'
            if legacy_model_dir.exists() and not any(legacy_model_dir.iterdir()):
                legacy_model_dir.rmdir()
                self.log_action("CLEANUP", "Removed empty save_models directory")
        
        print()
        print("Migration completed successfully!" if not self.dry_run else "Dry run completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Migrate Deep-GPCM file structure')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without making changes')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backups')
    parser.add_argument('--report-only', action='store_true', help='Generate migration report only')
    
    args = parser.parse_args()
    
    migrator = StructureMigrator(
        base_dir=args.base_dir,
        dry_run=args.dry_run or args.report_only,
        backup=not args.no_backup
    )
    
    if args.report_only:
        # Generate and display migration report
        report = migrator.generate_migration_report()
        print(json.dumps(report, indent=2))
    else:
        # Run migration
        migrator.run_migration()


if __name__ == "__main__":
    main()