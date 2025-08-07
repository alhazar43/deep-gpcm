"""
Unified Configuration System for Deep-GPCM Pipeline

This module provides a factory-integrated configuration system that eliminates
argument duplication and provides intelligent defaults for ML workflow optimization.
"""

from .base import BaseConfig, LossConfig
from .training import TrainingConfig, MultiModelConfig, HyperparameterConfig, ExperimentConfig
from .evaluation import EvaluationConfig, BatchEvaluationConfig, IRTAnalysisConfig, PlottingConfig
from .pipeline import PipelineConfig, BenchmarkConfig, ExperimentPipelineConfig
from .parser import ConfigParser, ModelConfigResolver, SmartArgumentParser
from .builder import CommandBuilder, PipelineOrchestrator, BatchCommandBuilder

__all__ = [
    'BaseConfig',
    'LossConfig',
    'TrainingConfig',
    'MultiModelConfig', 
    'HyperparameterConfig',
    'ExperimentConfig',
    'EvaluationConfig',
    'BatchEvaluationConfig',
    'IRTAnalysisConfig',
    'PlottingConfig',
    'PipelineConfig',
    'BenchmarkConfig',
    'ExperimentPipelineConfig',
    'ConfigParser',
    'ModelConfigResolver',
    'SmartArgumentParser',
    'CommandBuilder',
    'PipelineOrchestrator',
    'BatchCommandBuilder'
]