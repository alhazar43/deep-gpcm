"""Migration script to transition from separate loss files to unified losses.py"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path, dry_run: bool = True):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern replacements
    replacements = [
        # Replace ordinal_losses imports
        (r'from training\.ordinal_losses import', 'from training.losses import'),
        (r'from \.ordinal_losses import', 'from .losses import'),
        (r'import training\.ordinal_losses', 'import training.losses'),
        
        # Update any direct module references
        (r'training\.ordinal_losses\.', 'training.losses.'),
        (r'ordinal_losses\.', 'losses.'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        print(f"{'[DRY RUN] ' if dry_run else ''}Updating: {file_path}")
        if not dry_run:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            with open(backup_path, 'w') as f:
                f.write(original_content)
            
            # Write updated content
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"  Created backup: {backup_path}")
        return True
    return False


def migrate_loss_imports(project_root: str = "/home/steph/dirt-new/deep-gpcm", 
                        dry_run: bool = True):
    """Migrate all imports from ordinal_losses to unified losses module."""
    print(f"Migrating loss imports in {project_root}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL MIGRATION'}\n")
    
    updated_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk(project_root):
        # Skip migration scripts and backup files
        if 'migrate' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and not file.endswith('.bak'):
                file_path = Path(root) / file
                
                # Skip the loss files themselves
                if file in ['losses.py', 'ordinal_losses.py', 'losses_unified.py']:
                    continue
                
                if update_imports_in_file(file_path, dry_run):
                    updated_files.append(file_path)
    
    print(f"\nSummary:")
    print(f"  Files to update: {len(updated_files)}")
    if updated_files:
        print("\n  Files:")
        for f in updated_files:
            print(f"    - {f.relative_to(project_root)}")
    
    if dry_run:
        print("\nTo perform actual migration, run with dry_run=False")
    else:
        print("\nMigration complete!")
    
    return updated_files


def create_transition_plan():
    """Create a detailed transition plan."""
    plan = """
# Loss Module Migration Plan

## Phase 1: Preparation (Current)
1. ✓ Created unified loss module (losses_unified.py)
2. ✓ Includes all functionality from both files
3. ✓ Added enhanced focal loss implementation
4. ✓ Maintained backward compatibility

## Phase 2: Testing
1. Run test suite with unified module
2. Verify all loss functions work correctly
3. Check gradient flow and numerical stability

## Phase 3: Migration
1. Backup original files
2. Replace losses.py with losses_unified.py
3. Update all imports automatically
4. Remove ordinal_losses.py

## Phase 4: Cleanup
1. Update __init__.py exports
2. Update documentation
3. Remove backup files after verification

## Import Changes Required:
- `from training.ordinal_losses import X` → `from training.losses import X`
- All loss classes remain the same
- Factory function `create_loss_function` now included in main module

## New Features Added:
1. Enhanced FocalLoss with:
   - Per-class alpha weights support
   - Mask support for sequence data
   - Proper device handling
   
2. Unified factory function:
   - `create_loss_function(loss_type, n_cats, **kwargs)`
   - Supports all loss types including 'focal'
   
3. Comprehensive testing:
   - `test_all_losses()` function included
"""
    
    with open("/home/steph/dirt-new/deep-gpcm/training/MIGRATION_PLAN.md", 'w') as f:
        f.write(plan)
    
    print("Created migration plan: MIGRATION_PLAN.md")


if __name__ == "__main__":
    # Create migration plan
    create_transition_plan()
    
    # Perform dry run
    print("\nPerforming dry run analysis...")
    migrate_loss_imports(dry_run=True)
    
    # Uncomment to perform actual migration
    # print("\n" + "="*60 + "\n")
    # migrate_loss_imports(dry_run=False)