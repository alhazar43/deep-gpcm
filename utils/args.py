"""Unified argument parser for Deep-GPCM pipeline.

Standardizes command-line arguments across all components while maintaining
backward compatibility with existing scripts.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

class UnifiedArgumentParser:
    """Unified argument parser with standardized naming and validation."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.parser = argparse.ArgumentParser(
            description=f"Deep-GPCM {component_name}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._setup_base_args()
        
    def _setup_base_args(self):
        """Setup base arguments common to all components."""
        # Dataset configuration
        dataset_group = self.parser.add_argument_group('Dataset Configuration')
        dataset_group.add_argument('--dataset', type=str, required=True,
                                 help='Dataset name (auto-detected from data/ directory)')
        dataset_group.add_argument('--data_dir', type=str, default='data',
                                 help='Base data directory')
        
        # Model configuration
        model_group = self.parser.add_argument_group('Model Configuration')
        # Note: Model choices will be dynamically set by components
        model_group.add_argument('--device', type=str, default='auto',
                               help='Device to use (auto, cpu, cuda, cuda:0)')
        
        # Training parameters  
        train_group = self.parser.add_argument_group('Training Parameters')
        train_group.add_argument('--epochs', type=int, default=30,
                               help='Number of training epochs')
        train_group.add_argument('--batch_size', type=int, default=64,
                               help='Training batch size')
        train_group.add_argument('--lr', '--learning_rate', type=float, default=1e-3,
                               help='Learning rate')
        train_group.add_argument('--patience', type=int, default=10,
                               help='Early stopping patience')
        train_group.add_argument('--bucket_by_length', action='store_true',
                               help='Bucket sequences by length to reduce padding')
        
        # Model architecture
        arch_group = self.parser.add_argument_group('Model Architecture')
        arch_group.add_argument('--hidden_dim', type=int, default=128,
                              help='Hidden dimension size')
        arch_group.add_argument('--num_layers', type=int, default=2,
                              help='Number of model layers')
        arch_group.add_argument('--dropout', type=float, default=0.1,
                              help='Dropout rate')
        
        # Cross-validation - streamlined approach
        cv_group = self.parser.add_argument_group('Cross-Validation')
        cv_group.add_argument('--n_folds', type=int, default=5,
                            help='Number of CV folds (0=no CV, 1=single run, >1=K-fold CV)')
        cv_group.add_argument('--fold', type=int, default=None,
                            help='Specific fold to run (1-based, None for all)')
        cv_group.add_argument('--hyperopt', action='store_true',
                            help='Enable Bayesian hyperparameter optimization')
        cv_group.add_argument('--hyperopt_trials', type=int, default=50,
                            help='Number of hyperparameter optimization trials')
        cv_group.add_argument('--hyperopt_metric', type=str, default='quadratic_weighted_kappa',
                            choices=['quadratic_weighted_kappa', 'categorical_accuracy', 'ordinal_accuracy'],
                            help='Metric to optimize during hyperparameter search')
        
        # Output configuration
        output_group = self.parser.add_argument_group('Output Configuration')
        output_group.add_argument('--output_dir', type=str, default=None,
                                help='Output directory (auto-generated if None)')
        output_group.add_argument('--save_model', action='store_true', default=True,
                                help='Save trained models')
        output_group.add_argument('--save_results', action='store_true', default=True,
                                help='Save evaluation results')
        
        # Execution control
        exec_group = self.parser.add_argument_group('Execution Control')
        exec_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output')
        exec_group.add_argument('--debug', action='store_true',
                              help='Debug mode')
        exec_group.add_argument('--seed', type=int, default=42,
                              help='Random seed')
        
    def add_training_args(self):
        """Add training-specific arguments."""
        train_group = self.parser.add_argument_group('Training Options')
        train_group.add_argument('--resume', type=str, default=None,
                               help='Resume from checkpoint path')
        train_group.add_argument('--validate_only', action='store_true',
                               help='Run validation only (no training)')
        train_group.add_argument('--save_freq', type=int, default=5,
                               help='Model save frequency (epochs)')
        return self
        
    def add_evaluation_args(self):
        """Add evaluation-specific arguments."""
        eval_group = self.parser.add_argument_group('Evaluation Options')
        eval_group.add_argument('--model_path', type=str,
                              help='Path to trained model (required for single evaluation)')
        eval_group.add_argument('--metrics', nargs='+', 
                              default=['auc', 'acc', 'rmse'],
                              help='Metrics to compute')
        eval_group.add_argument('--split', type=str, default='test',
                              choices=['train', 'valid', 'test'],
                              help='Data split to evaluate')
        return self
        
    def add_irt_args(self):
        """Add IRT analysis arguments."""
        irt_group = self.parser.add_argument_group('IRT Analysis Options')
        irt_group.add_argument('--ability_method', type=str, default='last',
                             choices=['last', 'mean', 'final'],
                             help='Student ability extraction method')
        irt_group.add_argument('--param_method', type=str, default='average',
                             choices=['average', 'final', 'best'],
                             help='Item parameter extraction method')
        irt_group.add_argument('--analysis_types', nargs='+',
                             default=['recovery', 'temporal'],
                             choices=['recovery', 'temporal', 'convergence'],
                             help='Types of IRT analysis to perform')
        irt_group.add_argument('--num_students', type=int, default=100,
                             help='Number of students for analysis')
        return self
        
    def add_plotting_args(self):
        """Add plotting arguments."""
        plot_group = self.parser.add_argument_group('Plotting Options')
        plot_group.add_argument('--plot_types', nargs='+',
                              default=['learning_curves', 'metrics'],
                              help='Types of plots to generate')
        plot_group.add_argument('--plot_format', type=str, default='png',
                              choices=['png', 'pdf', 'svg'],
                              help='Plot output format')
        plot_group.add_argument('--dpi', type=int, default=300,
                              help='Plot DPI')
        plot_group.add_argument('--figsize', nargs=2, type=float,
                              default=[12, 8],
                              help='Figure size (width height)')
        return self
        
    def add_cleanup_args(self):
        """Add cleanup arguments."""
        cleanup_group = self.parser.add_argument_group('Cleanup Options')
        cleanup_group.add_argument('--backup', action='store_true', default=True,
                                 help='Create backup before cleanup')
        cleanup_group.add_argument('--dry_run', action='store_true',
                                 help='Show what would be cleaned without doing it')
        cleanup_group.add_argument('--force', action='store_true',
                                 help='Force cleanup without confirmation')
        return self
        
    def add_multi_dataset_args(self):
        """Add multi-dataset training arguments."""
        multi_group = self.parser.add_argument_group('Multi-Dataset Options')
        multi_group.add_argument('--datasets', nargs='+', type=str,
                               help='Multiple datasets to process')
        multi_group.add_argument('--all', action='store_true',
                               help='Process all available datasets')
        multi_group.add_argument('--exclude', nargs='+', type=str, default=[],
                               help='Datasets to exclude from --all')
        multi_group.add_argument('--parallel', action='store_true',
                               help='Run datasets in parallel')
        multi_group.add_argument('--max_workers', type=int, default=4,
                               help='Maximum parallel workers')
        return self
        
    def parse_args(self, args=None):
        """Parse arguments with validation and post-processing."""
        args = self.parser.parse_args(args)
        return self._post_process_args(args)
        
    def _post_process_args(self, args):
        """Post-process and validate arguments."""
        # Device handling
        if args.device == 'auto':
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Dataset validation
        if hasattr(args, 'dataset') and args.dataset:
            args.dataset = self._validate_dataset(args.dataset, args.data_dir)
            
        # Multi-dataset handling
        if hasattr(args, 'all') and args.all:
            exclude = getattr(args, 'exclude', [])
            args.datasets = self._get_all_datasets(args.data_dir, exclude)
        elif hasattr(args, 'datasets') and args.datasets:
            args.datasets = [self._validate_dataset(d, args.data_dir) 
                           for d in args.datasets]
            
        # Path handling
        if hasattr(args, 'output_dir') and args.output_dir:
            args.output_dir = Path(args.output_dir).resolve()
            
        return args
        
    def _validate_dataset(self, dataset: str, data_dir: str) -> str:
        """Validate dataset exists."""
        data_path = Path(data_dir)
        dataset_paths = [
            data_path / dataset,
            data_path / f"{dataset}.txt", 
            data_path / f"{dataset}_train.txt"
        ]
        
        for path in dataset_paths:
            if path.exists():
                return dataset
                
        # List available datasets
        available = []
        if data_path.exists():
            for item in data_path.iterdir():
                if item.is_dir():
                    available.append(item.name)
                elif item.suffix == '.txt' and not item.name.endswith(('_train.txt', '_test.txt')):
                    available.append(item.stem)
                    
        raise ValueError(f"Dataset '{dataset}' not found. Available: {available}")
        
    def _get_all_datasets(self, data_dir: str, exclude: List[str] = None) -> List[str]:
        """Get all available datasets."""
        exclude = exclude or []
        data_path = Path(data_dir)
        datasets = []
        
        if not data_path.exists():
            raise ValueError(f"Data directory '{data_dir}' not found")
            
        # Find dataset directories
        for item in data_path.iterdir():
            if item.is_dir() and item.name not in exclude:
                datasets.append(item.name)
                
        # Find single-file datasets
        for item in data_path.iterdir():
            if (item.suffix == '.txt' and 
                not item.name.endswith(('_train.txt', '_test.txt')) and
                item.stem not in exclude and
                item.stem not in datasets):
                datasets.append(item.stem)
                
        return sorted(datasets)


def create_parser(component: str, **kwargs) -> UnifiedArgumentParser:
    """Factory function to create component-specific parsers.
    
    Args:
        component: Component name ('train', 'eval', 'irt', 'plot', etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Configured UnifiedArgumentParser
    """
    parser = UnifiedArgumentParser(component)
    
    # Add component-specific arguments
    if component == 'train':
        parser.add_training_args()
        if kwargs.get('multi_dataset', False):
            parser.add_multi_dataset_args()
            
    elif component == 'evaluate':
        parser.add_evaluation_args()
        
    elif component == 'irt':
        parser.add_irt_args()
        parser.add_plotting_args()
        
    elif component == 'plot':
        parser.add_plotting_args()
        
    elif component == 'cleanup':
        parser.add_cleanup_args()
        
    elif component == 'main':
        parser.add_training_args()
        parser.add_multi_dataset_args()
        
    return parser


# Backward compatibility helpers
def get_legacy_args(args, legacy_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """Convert unified args to legacy format for backward compatibility.
    
    Args:
        args: Parsed arguments from unified parser
        legacy_mapping: Custom mapping for argument names
        
    Returns:
        Dictionary with legacy argument names
    """
    legacy_mapping = legacy_mapping or {}
    
    # Default legacy mappings
    default_mapping = {
        'learning_rate': 'lr',
        'model_type': 'model',
        'num_epochs': 'epochs',
    }
    
    # Merge mappings
    mapping = {**default_mapping, **legacy_mapping}
    
    # Convert args to dict
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    
    # Apply legacy mappings
    legacy_args = {}
    for key, value in args_dict.items():
        legacy_key = mapping.get(key, key)
        legacy_args[legacy_key] = value
        
    return legacy_args


def validate_args(args, required_fields: List[str] = None) -> None:
    """Validate parsed arguments.
    
    Args:
        args: Parsed arguments
        required_fields: List of required field names
        
    Raises:
        ValueError: If validation fails
    """
    required_fields = required_fields or []
    
    for field in required_fields:
        if not hasattr(args, field) or getattr(args, field) is None:
            raise ValueError(f"Required argument '{field}' is missing")
            
    # Validate numeric ranges
    if hasattr(args, 'epochs') and args.epochs <= 0:
        raise ValueError("epochs must be positive")
        
    if hasattr(args, 'batch_size') and args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
        
    if hasattr(args, 'lr') and (args.lr <= 0 or args.lr > 1):
        raise ValueError("learning_rate must be in (0, 1]")
        
    if hasattr(args, 'n_folds') and args.n_folds < 0:
        raise ValueError("n_folds must be >= 0 (0=no CV, 1=single run, >1=K-fold CV)")
        
    if hasattr(args, 'fold') and args.fold is not None and hasattr(args, 'n_folds'):
        if args.fold < 1 or args.fold > args.n_folds:
            raise ValueError(f"fold must be in [1, {args.n_folds}]")


if __name__ == "__main__":
    # Example usage
    parser = create_parser('train', multi_dataset=True)
    args = parser.parse_args()
    validate_args(args, required_fields=['dataset'])
    print(f"Parsed args: {vars(args)}")
