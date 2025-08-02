#!/usr/bin/env python3
"""
Recovery script to restore files from backup and migrate to new structure
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.migrate_structure import StructureMigrator


def recover_from_backup(backup_dir: str, base_dir: str = ".", dry_run: bool = True):
    """Recover files from backup directory"""
    backup_path = Path(backup_dir)
    base_path = Path(base_dir)
    
    if not backup_path.exists():
        print(f"❌ Backup directory not found: {backup_path}")
        return False
    
    print("="*60)
    print("RECOVERING FROM BACKUP")
    print("="*60)
    print(f"Backup: {backup_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL RECOVERY'}")
    print()
    
    files_to_recover = []
    
    # Find all files in backup
    for root, dirs, files in os.walk(backup_path):
        for file in files:
            file_path = Path(root) / file
            # Calculate relative path from backup root
            rel_path = file_path.relative_to(backup_path)
            target_path = base_path / rel_path
            files_to_recover.append((file_path, target_path))
    
    print(f"Found {len(files_to_recover)} files to recover")
    
    if not dry_run:
        recovered = 0
        for src, dst in files_to_recover:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                recovered += 1
                print(f"✅ Recovered: {dst.relative_to(base_path)}")
            except Exception as e:
                print(f"❌ Failed to recover {src}: {e}")
        
        print(f"\nRecovered {recovered}/{len(files_to_recover)} files")
    else:
        print("\nFiles to recover:")
        for i, (src, dst) in enumerate(files_to_recover[:10]):
            print(f"  {src.name} -> {dst.relative_to(base_path)}")
        if len(files_to_recover) > 10:
            print(f"  ... and {len(files_to_recover) - 10} more files")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Recover and migrate Deep-GPCM files')
    parser.add_argument('--backup-dir', required=True, help='Backup directory to recover from')
    parser.add_argument('--base-dir', default='.', help='Base directory of the project')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--recover-only', action='store_true', help='Only recover, no migration')
    parser.add_argument('--migrate-only', action='store_true', help='Only migrate existing files')
    
    args = parser.parse_args()
    
    if not args.migrate_only:
        # Step 1: Recover files from backup
        print("STEP 1: RECOVERY")
        print("-"*60)
        success = recover_from_backup(args.backup_dir, args.base_dir, args.dry_run)
        if not success:
            return
        print()
    
    if not args.recover_only and not args.dry_run:
        # Step 2: Run migration
        print("STEP 2: MIGRATION")
        print("-"*60)
        
        migrator = StructureMigrator(
            base_dir=args.base_dir,
            dry_run=False,
            backup=False  # Don't backup again
        )
        
        migrator.run_migration()
    
    print("\n" + "="*60)
    print("RECOVERY AND MIGRATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()