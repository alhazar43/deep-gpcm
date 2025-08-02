#!/usr/bin/env python3
"""
Path utilities for Deep-GPCM project
Handles both legacy and new file structures with backward compatibility
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json
from datetime import datetime


class PathManager:
    """Manages file paths for models and results with dual structure support"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.legacy_mode = False  # Can be toggled for backward compatibility
        
        # Define directory structures
        self.dirs = {
            'models': self.base_dir / 'saved_models',
            'results': self.base_dir / 'results',
            'train': self.base_dir / 'results' / 'train',
            'validation': self.base_dir / 'results' / 'validation',
            'test': self.base_dir / 'results' / 'test',
            'plots': self.base_dir / 'results' / 'plots',
            'legacy_models': self.base_dir / 'save_models'
        }
    
    def ensure_directories(self, dataset: Optional[str] = None):
        """Create all necessary directories"""
        # Base directories
        for dir_type, path in self.dirs.items():
            if dir_type != 'legacy_models':  # Don't create legacy dir
                path.mkdir(parents=True, exist_ok=True)
        
        # Dataset-specific directories if provided
        if dataset:
            (self.dirs['models'] / dataset).mkdir(parents=True, exist_ok=True)
            (self.dirs['train'] / dataset).mkdir(parents=True, exist_ok=True)
            (self.dirs['validation'] / dataset).mkdir(parents=True, exist_ok=True)
            (self.dirs['test'] / dataset).mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str, dataset: str, fold: Optional[int] = None, 
                      is_best: bool = True, legacy: bool = False) -> Path:
        """Get model save path
        
        Args:
            model_name: Name of model (e.g., 'deep_gpcm')
            dataset: Dataset name
            fold: Fold number (None for no CV or best model)
            is_best: Whether this is the best model
            legacy: Use legacy path structure
            
        Returns:
            Path object for model file
        """
        if legacy or self.legacy_mode:
            # Legacy format: save_models/best_modelname_dataset.pth
            if fold is not None:
                return self.dirs['legacy_models'] / f"best_{model_name}_{dataset}_fold_{fold}.pth"
            else:
                return self.dirs['legacy_models'] / f"best_{model_name}_{dataset}.pth"
        
        # New format: saved_models/dataset/modelname_fold_x.pth or best_modelname.pth
        model_dir = self.dirs['models'] / dataset
        
        if fold is not None:
            return model_dir / f"{model_name}_fold_{fold}.pth"
        elif is_best:
            return model_dir / f"best_{model_name}.pth"
        else:
            return model_dir / f"{model_name}.pth"
    
    def get_result_path(self, result_type: str, model_name: str, dataset: str, 
                       fold: Optional[int] = None, suffix: str = 'json') -> Path:
        """Get result file path
        
        Args:
            result_type: Type of result ('train', 'validation', 'test')
            model_name: Name of model
            dataset: Dataset name
            fold: Fold number (None for summary or single run)
            suffix: File extension (default: json)
            
        Returns:
            Path object for result file
        """
        if result_type not in ['train', 'validation', 'test']:
            raise ValueError(f"Invalid result_type: {result_type}")
        
        result_dir = self.dirs[result_type] / dataset
        
        if result_type == 'train':
            if fold is not None:
                filename = f"train_{model_name}_fold_{fold}.{suffix}"
            else:
                filename = f"train_{model_name}.{suffix}"
        elif result_type == 'validation':
            filename = f"cv_{model_name}_summary.{suffix}"
        else:  # test
            filename = f"test_{model_name}.{suffix}"
        
        return result_dir / filename
    
    def get_legacy_result_path(self, model_name: str, dataset: str, 
                              fold: Optional[int] = None, is_cv_summary: bool = False) -> Path:
        """Get legacy result path for backward compatibility"""
        if is_cv_summary:
            filename = f"train_results_{model_name}_{dataset}_cv_summary.json"
        elif fold is not None:
            filename = f"train_results_{model_name}_{dataset}_fold_{fold}.json"
        else:
            filename = f"train_results_{model_name}_{dataset}.json"
        
        return self.dirs['train'] / filename
    
    def find_model_files(self, model_name: str, dataset: str) -> Dict[str, Path]:
        """Find all model files for a given model and dataset
        
        Returns:
            Dictionary with keys like 'best', 'fold_1', etc.
        """
        files = {}
        
        # Check new structure
        new_dir = self.dirs['models'] / dataset
        if new_dir.exists():
            # Best model
            best_path = new_dir / f"best_{model_name}.pth"
            if best_path.exists():
                files['best'] = best_path
            
            # Fold models
            for fold_file in new_dir.glob(f"{model_name}_fold_*.pth"):
                fold_num = int(fold_file.stem.split('_')[-1])
                files[f'fold_{fold_num}'] = fold_file
        
        # Check legacy structure if no files found
        if not files and self.dirs['legacy_models'].exists():
            # Best model
            legacy_best = self.dirs['legacy_models'] / f"best_{model_name}_{dataset}.pth"
            if legacy_best.exists():
                files['best'] = legacy_best
            
            # Fold models
            for fold_file in self.dirs['legacy_models'].glob(f"best_{model_name}_{dataset}_fold_*.pth"):
                fold_num = int(fold_file.stem.split('_')[-1])
                files[f'fold_{fold_num}'] = fold_file
        
        return files
    
    def migrate_file(self, old_path: Path, new_path: Path, copy: bool = True) -> bool:
        """Migrate a file from old to new location
        
        Args:
            old_path: Source path
            new_path: Destination path
            copy: If True, copy file; if False, move file
            
        Returns:
            True if successful
        """
        if not old_path.exists():
            return False
        
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        if copy:
            import shutil
            shutil.copy2(old_path, new_path)
        else:
            old_path.rename(new_path)
        
        return True
    
    def get_all_legacy_files(self) -> Dict[str, List[Path]]:
        """Find all legacy files that need migration"""
        legacy_files = {
            'models': [],
            'results': []
        }
        
        # Find legacy model files
        if self.dirs['legacy_models'].exists():
            legacy_files['models'] = list(self.dirs['legacy_models'].glob("*.pth"))
        
        # Find legacy result files
        if self.dirs['train'].exists():
            # Direct children only (not in subdirectories)
            for file in self.dirs['train'].iterdir():
                if file.is_file() and file.suffix == '.json':
                    legacy_files['results'].append(file)
        
        # Also check test directory
        if self.dirs['test'].exists():
            for file in self.dirs['test'].iterdir():
                if file.is_file() and file.suffix == '.json':
                    legacy_files['results'].append(file)
        
        return legacy_files
    
    def parse_legacy_filename(self, filepath: Path) -> Dict[str, str]:
        """Parse information from legacy filename"""
        name = filepath.stem
        info = {}
        
        if filepath.suffix == '.pth':
            # Model file: best_modelname_dataset[_fold_X].pth
            parts = name.split('_')
            if parts[0] == 'best':
                info['is_best'] = True
                parts = parts[1:]  # Remove 'best'
            
            # Check for fold pattern
            if len(parts) >= 3 and parts[-2] == 'fold':
                info['fold'] = int(parts[-1])
                remaining_parts = parts[:-2]  # Remove 'fold_X'
            else:
                remaining_parts = parts
            
            # Find the model/dataset split
            # Common models: deep_gpcm, attn_gpcm, coral_gpcm_proper
            if len(remaining_parts) >= 3:
                # Check for known model patterns
                if remaining_parts[0] == 'deep' and remaining_parts[1] == 'gpcm':
                    info['model'] = 'deep_gpcm'
                    info['dataset'] = '_'.join(remaining_parts[2:])
                elif remaining_parts[0] == 'attn' and remaining_parts[1] == 'gpcm':
                    info['model'] = 'attn_gpcm'
                    info['dataset'] = '_'.join(remaining_parts[2:])
                elif len(remaining_parts) >= 3 and remaining_parts[0] == 'coral' and remaining_parts[1] == 'gpcm' and remaining_parts[2] == 'proper':
                    info['model'] = 'coral_gpcm_proper'
                    info['dataset'] = '_'.join(remaining_parts[3:])
                else:
                    # Unknown pattern, take first part as model
                    info['model'] = remaining_parts[0]
                    info['dataset'] = '_'.join(remaining_parts[1:])
            else:
                # Fallback
                info['model'] = remaining_parts[0] if remaining_parts else 'unknown'
                info['dataset'] = '_'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else 'unknown'
        
        elif filepath.suffix == '.json':
            # Result file: train_results_modelname_dataset[_fold_X|_cv_summary].json
            # or test_results_modelname_dataset.json
            if name.startswith('train_results_'):
                parts = name[14:].split('_')  # Remove 'train_results_'
                
                # Check for CV summary pattern
                if len(parts) >= 2 and parts[-1] == 'summary' and parts[-2] == 'cv':
                    info['is_cv_summary'] = True
                    remaining_parts = parts[:-2]  # Remove 'cv_summary'
                # Check for fold pattern
                elif len(parts) >= 2 and parts[-2] == 'fold':
                    info['fold'] = int(parts[-1])
                    remaining_parts = parts[:-2]  # Remove 'fold_X'
                else:
                    remaining_parts = parts
                
                # Find the model/dataset split (same logic as above)
                if len(remaining_parts) >= 3:
                    if remaining_parts[0] == 'deep' and remaining_parts[1] == 'gpcm':
                        info['model'] = 'deep_gpcm'
                        info['dataset'] = '_'.join(remaining_parts[2:])
                    elif remaining_parts[0] == 'attn' and remaining_parts[1] == 'gpcm':
                        info['model'] = 'attn_gpcm'
                        info['dataset'] = '_'.join(remaining_parts[2:])
                    elif len(remaining_parts) >= 3 and remaining_parts[0] == 'coral' and remaining_parts[1] == 'gpcm' and remaining_parts[2] == 'proper':
                        info['model'] = 'coral_gpcm_proper'
                        info['dataset'] = '_'.join(remaining_parts[3:])
                    else:
                        info['model'] = remaining_parts[0]
                        info['dataset'] = '_'.join(remaining_parts[1:])
                else:
                    info['model'] = remaining_parts[0] if remaining_parts else 'unknown'
                    info['dataset'] = '_'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else 'unknown'
            
            elif name.startswith('test_results_'):
                parts = name[13:].split('_')  # Remove 'test_results_'
                
                # Find the model/dataset split (same logic as above)
                if len(parts) >= 3:
                    if parts[0] == 'deep' and parts[1] == 'gpcm':
                        info['model'] = 'deep_gpcm'
                        info['dataset'] = '_'.join(parts[2:])
                    elif parts[0] == 'attn' and parts[1] == 'gpcm':
                        info['model'] = 'attn_gpcm'
                        info['dataset'] = '_'.join(parts[2:])
                    elif len(parts) >= 3 and parts[0] == 'coral' and parts[1] == 'gpcm' and parts[2] == 'proper':
                        info['model'] = 'coral_gpcm_proper'
                        info['dataset'] = '_'.join(parts[3:])
                    else:
                        # For simple model names like 'deep' or 'attn'
                        if parts[0] in ['deep', 'attn']:
                            info['model'] = parts[0]
                            info['dataset'] = '_'.join(parts[1:])
                        else:
                            info['model'] = parts[0]
                            info['dataset'] = '_'.join(parts[1:])
                else:
                    info['model'] = parts[0] if parts else 'unknown'
                    info['dataset'] = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                
                info['is_test'] = True
        
        return info


# Convenience functions
_default_manager = None


def get_path_manager(base_dir: str = ".") -> PathManager:
    """Get or create the default PathManager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = PathManager(base_dir)
    return _default_manager


def ensure_directories(dataset: Optional[str] = None):
    """Ensure all directories exist"""
    manager = get_path_manager()
    manager.ensure_directories(dataset)


def get_model_path(model_name: str, dataset: str, **kwargs) -> Path:
    """Get model path using default manager"""
    manager = get_path_manager()
    return manager.get_model_path(model_name, dataset, **kwargs)


def get_result_path(result_type: str, model_name: str, dataset: str, **kwargs) -> Path:
    """Get result path using default manager"""
    manager = get_path_manager()
    return manager.get_result_path(result_type, model_name, dataset, **kwargs)


def find_best_model(model_name: str, dataset: str) -> Optional[Path]:
    """Find the best model file for given model and dataset"""
    manager = get_path_manager()
    files = manager.find_model_files(model_name, dataset)
    return files.get('best')