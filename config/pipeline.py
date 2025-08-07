"""
Pipeline orchestration configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import BaseConfig
from .training import TrainingConfig, MultiModelConfig
from .evaluation import EvaluationConfig, IRTAnalysisConfig, PlottingConfig
from models.factory import get_all_model_types


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    # Base configuration
    dataset: str = 'synthetic_OC'
    device: Optional[str] = None
    seed: int = 42
    
    # Models to run (defaults to all available models from factory)
    models: List[str] = field(default_factory=get_all_model_types)
    
    # Pipeline phases
    run_training: bool = True
    run_evaluation: bool = True
    run_plotting: bool = True
    run_irt_analysis: bool = True
    
    # Training configuration
    epochs: int = 30
    n_folds: int = 5
    cv: bool = False
    parallel_training: bool = False
    
    # Evaluation configuration
    regenerate_plots: bool = True
    statistical_comparison: bool = True
    
    # Analysis configuration
    irt_analysis_config: IRTAnalysisConfig = field(default_factory=IRTAnalysisConfig)
    plotting_config: PlottingConfig = field(default_factory=PlottingConfig)
    
    # Resource management
    max_parallel_jobs: int = 2
    memory_limit_gb: Optional[float] = None
    
    # Cleanup options
    clean: bool = False
    no_backup: bool = False
    
    def __post_init__(self):
        # Set device if not specified
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validate that we have at least one model
        if not self.models:
            from models.factory import get_all_model_types
            self.models = get_all_model_types()
        
        # Validate models exist
        from models.factory import validate_model_type, get_all_model_types
        available_models = get_all_model_types()
        invalid_models = [m for m in self.models if not validate_model_type(m)]
        if invalid_models:
            raise ValueError(f"Invalid models: {invalid_models}. Available: {available_models}")
    
    def get_training_configs(self) -> List[TrainingConfig]:
        """Generate training configurations for each model."""
        configs = []
        for model in self.models:
            config = TrainingConfig(
                model=model,
                dataset=self.dataset,
                epochs=self.epochs,
                n_folds=self.n_folds,
                cv=self.cv,
                device=self.device,
                seed=self.seed
            )
            configs.append(config)
        return configs
    
    def get_evaluation_configs(self, training_configs: List[TrainingConfig]) -> List[EvaluationConfig]:
        """Generate evaluation configurations from training configs."""
        configs = []
        for train_config in training_configs:
            eval_config = EvaluationConfig.from_training_config(train_config)
            eval_config.regenerate_plots = self.regenerate_plots
            configs.append(eval_config)
        return configs
    
    def summary(self) -> str:
        """Generate pipeline summary."""
        phases = []
        if self.run_training:
            phases.append(f"Training ({len(self.models)} models, {self.epochs} epochs)")
        if self.run_evaluation:
            phases.append("Evaluation")
        if self.run_plotting:
            phases.append("Plotting")
        if self.run_irt_analysis:
            phases.append("IRT Analysis")
        
        return f"""Pipeline Configuration:
  Dataset: {self.dataset}
  Models: {', '.join(self.models)}
  Phases: {' → '.join(phases)}
  Device: {self.device}
  Cross-validation: {'Yes' if self.cv else 'No'}
  Statistical comparison: {'Yes' if self.statistical_comparison else 'No'}"""


@dataclass
class BenchmarkConfig:
    """Configuration for running comprehensive benchmarks."""
    
    # Datasets to benchmark
    datasets: List[str] = field(default_factory=lambda: ['synthetic_OC', 'synthetic_4000_200_2'])
    
    # Models to include
    models: Optional[List[str]] = None
    
    # Benchmark settings
    quick_mode: bool = False  # Reduced epochs for quick testing
    full_evaluation: bool = True
    
    # Comparison settings
    cross_dataset_comparison: bool = True
    statistical_testing: bool = True
    
    # Output
    generate_report: bool = True
    report_format: str = 'html'  # 'html', 'pdf', 'markdown'
    
    def __post_init__(self):
        if self.models is None:
            from models.factory import get_all_model_types
            self.models = get_all_model_types()
        
        # Validate datasets exist
        for dataset in self.datasets:
            dataset_path = Path(f'data/{dataset}')
            if not dataset_path.exists():
                print(f"⚠️  Warning: Dataset directory not found: {dataset_path}")
    
    def get_pipeline_configs(self) -> List[PipelineConfig]:
        """Generate pipeline configs for each dataset."""
        configs = []
        
        for dataset in self.datasets:
            epochs = 5 if self.quick_mode else 30
            
            config = PipelineConfig(
                model='deep_gpcm',  # Will be overridden by models list
                dataset=dataset,
                models=self.models,
                epochs=epochs,
                run_training=True,
                run_evaluation=self.full_evaluation,
                run_plotting=self.full_evaluation,
                run_irt_analysis=self.full_evaluation,
                statistical_comparison=self.statistical_testing
            )
            configs.append(config)
        
        return configs


@dataclass
class ExperimentPipelineConfig:
    """Configuration for research experiment pipelines."""
    
    # Experiment metadata
    experiment_name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Experimental design
    baseline_config: PipelineConfig = field(default_factory=PipelineConfig)
    experimental_configs: List[PipelineConfig] = field(default_factory=list)
    
    # Comparison settings
    control_group: str = 'deep_gpcm'
    statistical_power: float = 0.8
    effect_size_threshold: float = 0.2  # Minimum meaningful effect size
    
    # Reproducibility
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    deterministic: bool = True
    
    # Output and tracking
    results_dir: str = 'experiments'
    save_checkpoints: bool = True
    track_system_metrics: bool = True
    
    def __post_init__(self):
        if not self.experimental_configs:
            # Create default experimental configs with different settings
            for model in ['attn_gpcm', 'coral_gpcm_proper', 'coral_gpcm_fixed']:
                config = PipelineConfig(
                    model=model,
                    dataset=self.baseline_config.dataset,
                    models=[model],
                    epochs=self.baseline_config.epochs,
                    cv=True  # Enable hyperparameter optimization
                )
                self.experimental_configs.append(config)
    
    def get_all_configs(self) -> List[PipelineConfig]:
        """Get all configurations including baseline."""
        return [self.baseline_config] + self.experimental_configs
    
    def run_power_analysis(self) -> Dict[str, Any]:
        """Run statistical power analysis for experiment design."""
        # Simplified power analysis - in practice would use more sophisticated methods
        n_comparisons = len(self.experimental_configs)
        alpha_adjusted = 0.05 / n_comparisons  # Bonferroni correction
        
        return {
            'n_comparisons': n_comparisons,
            'alpha_adjusted': alpha_adjusted,
            'power': self.statistical_power,
            'effect_size_threshold': self.effect_size_threshold,
            'recommended_n_runs': len(self.random_seeds)
        }