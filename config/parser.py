"""
Intelligent argument parsing with factory integration.
"""

import argparse
from dataclasses import fields, is_dataclass
from typing import Type, TypeVar, Any, List, Dict, Optional
import sys

from models.factory import get_all_model_types, get_model_loss_config, get_model_default_params
from .base import BaseConfig, LossConfig
from .training import TrainingConfig, MultiModelConfig, HyperparameterConfig
from .evaluation import EvaluationConfig, BatchEvaluationConfig, PlottingConfig
from .pipeline import PipelineConfig, BenchmarkConfig

T = TypeVar('T')


class ConfigParser:
    """Unified argument parser with factory integration."""
    
    def __init__(self, config_class: Type[T], description: Optional[str] = None):
        self.config_class = config_class
        self.parser = argparse.ArgumentParser(
            description=description or f"{config_class.__name__} Configuration"
        )
        self._build_parser()
    
    def _build_parser(self):
        """Build argument parser based on config class fields."""
        if not is_dataclass(self.config_class):
            raise ValueError("Config class must be a dataclass")
        
        # Add arguments for each field in the dataclass
        for field in fields(self.config_class):
            self._add_field_argument(field)
        
        # Add factory-aware model choices
        field_names = [field.name for field in fields(self.config_class)]
        if 'model' in field_names:
            self._add_model_arguments()
    
    def _add_field_argument(self, field):
        """Add argument for a specific dataclass field."""
        field_name = field.name
        field_type = field.type
        default_value = field.default if field.default != field.default_factory else None
        
        # Skip complex nested objects
        if field_name in ['loss_config', 'validation_config', 'path_config', 
                         'irt_analysis_config', 'plotting_config']:
            return
        
        # Handle special cases
        if field_name == 'models':
            self._add_models_argument()
            return
        elif field_name == 'model':
            # Will be handled by _add_model_arguments
            return
        
        # Determine argument properties
        kwargs = {'help': f'{field_name.replace("_", " ").title()}'}
        
        if field_type == bool:
            if default_value:
                kwargs['action'] = 'store_false'
                kwargs['dest'] = field_name
            else:
                kwargs['action'] = 'store_true'
        elif field_type == int:
            kwargs['type'] = int
            if default_value is not None:
                kwargs['default'] = default_value
        elif field_type == float:
            kwargs['type'] = float
            if default_value is not None:
                kwargs['default'] = default_value
        elif field_type == str:
            kwargs['type'] = str
            if default_value is not None:
                kwargs['default'] = default_value
        elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
            kwargs['nargs'] = '+'
            if field_type.__args__[0] == str:
                kwargs['type'] = str
        elif hasattr(field_type, '__origin__') and field_type.__origin__ == Optional:
            # Optional fields
            inner_type = field_type.__args__[0]
            if inner_type == str:
                kwargs['type'] = str
            elif inner_type == int:
                kwargs['type'] = int
            elif inner_type == float:
                kwargs['type'] = float
        
        self.parser.add_argument(f'--{field_name}', **kwargs)
    
    def _add_model_arguments(self):
        """Add model-specific arguments with factory integration."""
        available_models = get_all_model_types()
        
        self.parser.add_argument(
            '--model',
            choices=available_models,
            help='Model type from factory registry'
        )
    
    def _add_models_argument(self):
        """Add models argument for multi-model configurations."""
        available_models = get_all_model_types()
        
        self.parser.add_argument(
            '--models',
            nargs='+',
            choices=available_models,
            default=available_models,
            help='Models to train/evaluate'
        )
    
    def parse(self, args: Optional[List[str]] = None) -> T:
        """Parse arguments and return config object."""
        parsed_args = self.parser.parse_args(args)
        
        # Convert namespace to dict and filter None values
        kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
        
        try:
            return self.config_class(**kwargs)
        except TypeError as e:
            # Handle extra arguments gracefully
            valid_fields = {field.name for field in fields(self.config_class)}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            return self.config_class(**filtered_kwargs)
    
    def add_loss_arguments(self):
        """Add loss function arguments."""
        loss_group = self.parser.add_argument_group('Loss Function Configuration')
        
        loss_group.add_argument('--loss', type=str, default='ce',
                               choices=['ce', 'focal', 'qwk', 'ordinal_ce', 'coral', 'combined'],
                               help='Loss function type')
        loss_group.add_argument('--ce_weight', type=float, default=1.0,
                               help='Weight for CE loss in combined loss')
        loss_group.add_argument('--focal_weight', type=float, default=0.0,
                               help='Weight for focal loss in combined loss')
        loss_group.add_argument('--focal_gamma', type=float, default=2.0,
                               help='Focal loss gamma parameter')
        loss_group.add_argument('--focal_alpha', type=float, default=1.0,
                               help='Focal loss alpha parameter')
        loss_group.add_argument('--qwk_weight', type=float, default=0.5,
                               help='Weight for QWK loss in combined loss')
        loss_group.add_argument('--coral_weight', type=float, default=0.0,
                               help='Weight for CORAL loss in combined loss')
        loss_group.add_argument('--ordinal_weight', type=float, default=0.0,
                               help='Weight for ordinal CE loss in combined loss')


class ModelConfigResolver:
    """Resolves model configurations from factory."""
    
    @staticmethod
    def resolve_training_config(config: TrainingConfig) -> TrainingConfig:
        """Apply factory defaults to training configuration."""
        if not config.loss_config_override:
            # Apply factory loss configuration
            factory_loss_config = LossConfig.from_factory(config.model)
            config.loss_config = factory_loss_config
        
        return config
    
    @staticmethod
    def resolve_evaluation_config(config: EvaluationConfig) -> EvaluationConfig:
        """Apply factory defaults to evaluation configuration."""
        # Evaluation configs are mostly user-driven, minimal factory resolution needed
        return config
    
    @staticmethod
    def resolve_pipeline_config(config: PipelineConfig) -> PipelineConfig:
        """Apply factory defaults to pipeline configuration."""
        # Factory resolution happens at the individual config level
        return config
    
    @staticmethod
    def validate_model_compatibility(config: BaseConfig) -> bool:
        """Validate that model exists in factory and is properly configured."""
        from models.factory import validate_model_type
        return validate_model_type(config.model)


class SmartArgumentParser:
    """Smart parser that adapts to different use cases."""
    
    @staticmethod
    def create_training_parser() -> ConfigParser:
        """Create parser optimized for training workflows."""
        parser = ConfigParser(TrainingConfig, "Deep-GPCM Training")
        parser.add_loss_arguments()
        return parser
    
    @staticmethod
    def create_evaluation_parser() -> ConfigParser:
        """Create parser optimized for evaluation workflows."""
        return ConfigParser(EvaluationConfig, "Deep-GPCM Evaluation")
    
    @staticmethod
    def create_pipeline_parser() -> ConfigParser:
        """Create parser optimized for complete pipeline workflows."""
        return ConfigParser(PipelineConfig, "Deep-GPCM Complete Pipeline")
    
    @staticmethod
    def create_benchmark_parser() -> ConfigParser:
        """Create parser optimized for benchmarking workflows."""
        return ConfigParser(BenchmarkConfig, "Deep-GPCM Benchmark Suite")
    
    @staticmethod
    def create_plotting_parser() -> ConfigParser:
        """Create parser optimized for plotting workflows."""
        from .evaluation import PlottingConfig
        return ConfigParser(PlottingConfig, "Deep-GPCM Plotting System")
    
    @staticmethod
    def parse_from_command_line(script_type: str = 'auto') -> Any:
        """Automatically determine parser type from command line context."""
        if script_type == 'auto':
            script_name = sys.argv[0].lower()
            if 'train' in script_name:
                script_type = 'training'
            elif 'eval' in script_name:
                script_type = 'evaluation'
            elif 'plot' in script_name:
                script_type = 'plotting'
            elif 'main' in script_name or 'pipeline' in script_name:
                script_type = 'pipeline'
            elif 'benchmark' in script_name:
                script_type = 'benchmark'
            else:
                script_type = 'pipeline'  # Default fallback
        
        parser_map = {
            'training': SmartArgumentParser.create_training_parser,
            'evaluation': SmartArgumentParser.create_evaluation_parser,
            'pipeline': SmartArgumentParser.create_pipeline_parser,
            'benchmark': SmartArgumentParser.create_benchmark_parser,
            'plotting': SmartArgumentParser.create_plotting_parser
        }
        
        if script_type not in parser_map:
            raise ValueError(f"Unknown script type: {script_type}")
        
        parser = parser_map[script_type]()
        config = parser.parse()
        
        # Apply factory resolution
        if isinstance(config, TrainingConfig):
            config = ModelConfigResolver.resolve_training_config(config)
        elif isinstance(config, EvaluationConfig):
            config = ModelConfigResolver.resolve_evaluation_config(config)
        elif isinstance(config, PipelineConfig):
            config = ModelConfigResolver.resolve_pipeline_config(config)
        
        return config


# Convenience functions for common use cases
def parse_training_config(args: Optional[List[str]] = None) -> TrainingConfig:
    """Parse training configuration from command line."""
    parser = SmartArgumentParser.create_training_parser()
    config = parser.parse(args)
    return ModelConfigResolver.resolve_training_config(config)


def parse_evaluation_config(args: Optional[List[str]] = None) -> EvaluationConfig:
    """Parse evaluation configuration from command line."""
    parser = SmartArgumentParser.create_evaluation_parser()
    config = parser.parse(args)
    return ModelConfigResolver.resolve_evaluation_config(config)


def parse_pipeline_config(args: Optional[List[str]] = None) -> PipelineConfig:
    """Parse pipeline configuration from command line."""
    parser = SmartArgumentParser.create_pipeline_parser()
    config = parser.parse(args)
    return ModelConfigResolver.resolve_pipeline_config(config)