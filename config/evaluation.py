"""
Evaluation-specific configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import BaseConfig, PathConfig
from models.factory import get_model_type_from_path


@dataclass
class EvaluationConfig:
    """Evaluation-specific configuration."""
    
    # Base config fields that can be inherited from BaseConfig
    dataset: str = 'synthetic_OC'
    device: Optional[str] = None
    seed: int = 42
    model: Optional[str] = None  # Made optional for evaluation
    
    # Model path
    model_path: Optional[Path] = None
    
    # Evaluation parameters
    batch_size: int = 32
    regenerate_plots: bool = False
    
    # Prediction methods
    prediction_methods: List[str] = field(default_factory=lambda: ['hard', 'soft', 'threshold'])
    
    # Metrics configuration
    compute_multimethod: bool = True
    include_performance_metrics: bool = True
    
    # Analysis options
    run_irt_analysis: bool = True
    analysis_types: List[str] = field(default_factory=lambda: ['recovery', 'temporal'])
    evaluate_train_set: bool = False  # Whether to evaluate on training set for comparison
    detailed_analysis: bool = False  # Whether to run detailed error analysis
    per_category_analysis: bool = True  # Whether to compute per-category metrics
    comparison_plots: bool = False  # Whether to generate comparison plots
    
    # Paths
    path_config: Optional[PathConfig] = None
    
    def __post_init__(self):
        # Convert model_path to Path object if it's a string
        if self.model_path and isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        # Infer model type from model_path if model is not provided
        if self.model_path and not self.model:
            self.model = get_model_type_from_path(self.model_path)
        
        # Set device if not specified
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validate model if provided
        if self.model:
            from models.factory import validate_model_type, get_all_model_types
            if not validate_model_type(self.model):
                available = ', '.join(get_all_model_types())
                raise ValueError(f"Invalid model '{self.model}'. Available: {available}")
        
        # Initialize path config
        if self.path_config is None and self.model:
            self.path_config = PathConfig(dataset=self.dataset, model=self.model)
        
        # Set model path if not provided
        if self.model_path is None and self.path_config:
            self.model_path = self.path_config.model_save_path
    
    @property
    def test_results_path(self) -> Path:
        """Get test results path."""
        if self.path_config:
            return self.path_config.test_results_path
        else:
            return Path(f"results/test/{self.dataset}/test_{self.model}.json")
    
    @classmethod
    def from_training_config(cls, training_config) -> 'EvaluationConfig':
        """Create evaluation config from training config."""
        return cls(
            model=training_config.model,
            dataset=training_config.dataset,
            device=training_config.device,
            seed=training_config.seed,
            model_path=training_config.path_config.model_save_path,
            regenerate_plots=True  # Always regenerate after training
        )
    
    def get_evaluation_args(self) -> List[str]:
        """Generate evaluation arguments for subprocess calls."""
        args = [
            '--model_path', str(self.model_path),
            '--dataset', self.dataset,
            '--batch_size', str(self.batch_size)
        ]
        
        if self.device:
            args.extend(['--device', self.device])
        
        if self.regenerate_plots:
            args.append('--regenerate_plots')
        
        if self.prediction_methods != ['hard', 'soft', 'threshold']:
            args.extend(['--prediction_methods'] + self.prediction_methods)
        
        return args
    
    def summary(self) -> str:
        """Generate configuration summary."""
        return f"""Evaluation Configuration for {self.model}:
  Model path: {self.model_path}
  Dataset: {self.dataset}
  Batch size: {self.batch_size}
  Device: {self.device}
  Prediction methods: {', '.join(self.prediction_methods)}
  Regenerate plots: {self.regenerate_plots}"""


@dataclass
class BatchEvaluationConfig:
    """Configuration for batch evaluation of multiple models."""
    
    dataset: str = 'synthetic_OC'
    models_dir: str = 'saved_models'
    
    # Filtering
    dataset_filter: Optional[str] = None
    model_filter: Optional[List[str]] = None
    
    # Evaluation settings
    batch_size: int = 32
    device: Optional[str] = None
    regenerate_plots: bool = False
    include_cv_folds: bool = False
    
    # Analysis
    statistical_comparison: bool = True
    generate_summary: bool = True
    
    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ComparisonConfig:
    """Configuration for statistical model comparison."""
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Comparison methods
    statistical_tests: List[str] = field(default_factory=lambda: ['t_test', 'wilcoxon', 'bootstrap'])
    effect_size_measures: List[str] = field(default_factory=lambda: ['cohens_d', 'cliff_delta'])
    
    # Multiple comparison correction
    correction_method: str = 'bonferroni'  # 'bonferroni', 'holm', 'fdr'
    
    # Reporting
    report_format: str = 'table'  # 'table', 'latex', 'markdown'
    include_plots: bool = True
    
    def __post_init__(self):
        valid_tests = ['t_test', 'wilcoxon', 'bootstrap', 'mann_whitney']
        invalid_tests = [t for t in self.statistical_tests if t not in valid_tests]
        if invalid_tests:
            raise ValueError(f"Invalid statistical tests: {invalid_tests}")


@dataclass
class IRTAnalysisConfig:
    """Configuration for IRT parameter analysis."""
    
    # Dataset and directories
    dataset: str = 'synthetic_OC'
    output_dir: Optional[Path] = None  # Will use path_utils.get_plot_path() if None
    
    # Analysis types
    analysis_types: List[str] = field(default_factory=lambda: ['recovery', 'temporal'])
    
    # Parameter extraction
    extract_parameters: List[str] = field(default_factory=lambda: ['theta', 'alpha', 'beta'])
    
    # Temporal analysis
    n_students_temporal: int = 20
    student_selection_method: str = 'hit_rate'  # 'hit_rate', 'random', 'ability_range'
    
    # Recovery analysis
    correlation_metrics: List[str] = field(default_factory=lambda: ['pearson', 'spearman'])
    plot_recovery: bool = True
    
    # Visualization
    plot_heatmaps: bool = True
    plot_trajectories: bool = True
    plot_distributions: bool = True
    
    # Output
    save_parameters: bool = True
    save_summary: bool = True
    
    def __post_init__(self):
        valid_analyses = ['recovery', 'temporal', 'difficulty', 'discrimination']
        invalid_analyses = [a for a in self.analysis_types if a not in valid_analyses]
        if invalid_analyses:
            raise ValueError(f"Invalid analysis types: {invalid_analyses}")
        
        # Ensure output_dir is Path object
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class PlottingConfig:
    """Configuration for optimized plot generation."""
    
    # Dataset and directories
    dataset: Optional[str] = None
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Figure settings
    figsize_base: tuple = (10, 8)
    plot_quality: str = 'medium'  # 'low', 'medium', 'high'
    save_formats: List[str] = field(default_factory=lambda: ['png'])
    
    # Plot enablement flags
    enable_detailed_plots: bool = True
    enable_comparison_plots: bool = True
    enable_categorical_breakdown: bool = True
    enable_confusion_matrices: bool = True
    enable_learning_curves: bool = True
    enable_roc_curves: bool = True
    enable_calibration_plots: bool = True
    
    # Metrics customization
    custom_metrics: Optional[List[str]] = None
    
    # Colors and styling
    color_palette: str = 'tab10'
    style: str = 'default'
    
    # Output settings
    save_plots: bool = True
    show_plots: bool = False
    
    def __post_init__(self):
        valid_qualities = ['low', 'medium', 'high']
        if self.plot_quality not in valid_qualities:
            raise ValueError(f"Invalid plot quality: {self.plot_quality}. Valid: {valid_qualities}")
        
        valid_formats = ['png', 'pdf', 'svg', 'jpg']
        for fmt in self.save_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid format: {fmt}. Valid: {valid_formats}")
        
        # Ensure results_dir is Path object
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)