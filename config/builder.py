"""
Intelligent command builders for subprocess orchestration.
"""

import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

from .training import TrainingConfig
from .evaluation import EvaluationConfig
from .pipeline import PipelineConfig


class CommandBuilder:
    """Intelligent command builder with factory integration."""
    
    @staticmethod
    def build_training_command(config: TrainingConfig) -> List[str]:
        """Build optimized training command from configuration."""
        cmd = [sys.executable, "train.py"]
        
        # Core parameters
        cmd.extend(["--model", config.model])
        cmd.extend(["--dataset", config.dataset])
        cmd.extend(["--epochs", str(config.epochs)])
        cmd.extend(["--batch_size", str(config.batch_size)])
        cmd.extend(["--lr", str(config.lr)])
        cmd.extend(["--n_folds", str(config.n_folds)])
        cmd.extend(["--seed", str(config.seed)])
        
        # Device configuration
        if config.device:
            cmd.extend(["--device", config.device])
        
        # Cross-validation
        if config.cv:
            cmd.append("--cv")
        
        # Loss configuration - only include if not factory default
        if not config.loss_config.is_factory_default(config.model):
            cmd.extend(config.loss_config.to_args())
        
        return cmd
    
    @staticmethod
    def build_evaluation_command(config: EvaluationConfig) -> List[str]:
        """Build evaluation command from configuration."""
        cmd = [sys.executable, "evaluate.py"]
        
        # Core parameters
        cmd.extend(["--model_path", str(config.model_path)])
        cmd.extend(["--dataset", config.dataset])
        cmd.extend(["--batch_size", str(config.batch_size)])
        
        # Device configuration
        if config.device:
            cmd.extend(["--device", config.device])
        
        # Optional parameters
        if config.regenerate_plots:
            cmd.append("--regenerate_plots")
        
        if config.prediction_methods != ['hard', 'soft', 'threshold']:
            cmd.extend(["--prediction_methods"] + config.prediction_methods)
        
        return cmd
    
    @staticmethod
    def build_plotting_command(dataset: str, models: Optional[List[str]] = None) -> List[str]:
        """Build plotting command."""
        cmd = [sys.executable, "utils/plot_metrics.py"]
        cmd.extend(["--dataset", dataset])
        
        if models:
            cmd.extend(["--models"] + models)
        
        return cmd
    
    @staticmethod
    def build_irt_analysis_command(dataset: str, analysis_types: Optional[List[str]] = None) -> List[str]:
        """Build IRT analysis command."""
        cmd = [sys.executable, "analysis/irt_analysis.py"]
        cmd.extend(["--dataset", dataset])
        
        if analysis_types:
            cmd.extend(["--analysis_types"] + analysis_types)
        
        return cmd
    
    @staticmethod
    def estimate_command_resources(cmd: List[str]) -> Dict[str, Any]:
        """Estimate computational resources needed for command."""
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        
        # Basic resource estimation
        resources = {
            'estimated_time_minutes': 10,
            'memory_gb': 2.0,
            'gpu_required': True,
            'cpu_cores': 1
        }
        
        # Script-specific estimates
        if 'train.py' in script_name:
            # Extract epochs from command
            try:
                epochs_idx = cmd.index('--epochs')
                epochs = int(cmd[epochs_idx + 1])
                n_folds_idx = cmd.index('--n_folds')
                n_folds = int(cmd[n_folds_idx + 1])
                
                resources['estimated_time_minutes'] = epochs * max(n_folds, 1) * 2
                resources['memory_gb'] = 4.0
            except (ValueError, IndexError):
                pass
        
        elif 'evaluate.py' in script_name:
            resources['estimated_time_minutes'] = 5
            resources['memory_gb'] = 3.0
        
        elif 'plot_metrics.py' in script_name:
            resources['estimated_time_minutes'] = 2
            resources['memory_gb'] = 1.0
            resources['gpu_required'] = False
        
        elif 'irt_analysis.py' in script_name:
            resources['estimated_time_minutes'] = 3
            resources['memory_gb'] = 2.0
            resources['gpu_required'] = False
        
        return resources


class PipelineOrchestrator:
    """High-level pipeline orchestration with intelligent resource management."""
    
    def __init__(self, max_parallel: int = 2, memory_limit_gb: Optional[float] = None):
        self.max_parallel = max_parallel
        self.memory_limit_gb = memory_limit_gb
        self.command_builder = CommandBuilder()
    
    def build_complete_pipeline(self, config: PipelineConfig) -> Dict[str, List[List[str]]]:
        """Build complete pipeline command sequence."""
        pipeline = {
            'training': [],
            'evaluation': [],
            'plotting': [],
            'irt_analysis': []
        }
        
        # Training phase
        if config.run_training:
            training_configs = config.get_training_configs()
            for train_config in training_configs:
                cmd = self.command_builder.build_training_command(train_config)
                pipeline['training'].append(cmd)
        
        # Evaluation phase
        if config.run_evaluation:
            training_configs = config.get_training_configs()
            evaluation_configs = config.get_evaluation_configs(training_configs)
            for eval_config in evaluation_configs:
                cmd = self.command_builder.build_evaluation_command(eval_config)
                pipeline['evaluation'].append(cmd)
        
        # Plotting phase
        if config.run_plotting:
            cmd = self.command_builder.build_plotting_command(
                config.dataset, config.models
            )
            pipeline['plotting'].append(cmd)
        
        # IRT Analysis phase
        if config.run_irt_analysis:
            cmd = self.command_builder.build_irt_analysis_command(
                config.dataset, config.irt_analysis_config.analysis_types
            )
            pipeline['irt_analysis'].append(cmd)
        
        return pipeline
    
    def optimize_execution_order(self, pipeline: Dict[str, List[List[str]]]) -> List[Dict[str, Any]]:
        """Optimize command execution order based on dependencies and resources."""
        execution_plan = []
        
        # Training phase - can be parallelized
        training_commands = pipeline.get('training', [])
        if training_commands:
            if len(training_commands) > 1 and self.max_parallel > 1:
                # Parallel training
                execution_plan.append({
                    'phase': 'training',
                    'commands': training_commands,
                    'parallel': True,
                    'max_workers': min(self.max_parallel, len(training_commands))
                })
            else:
                # Sequential training
                for cmd in training_commands:
                    execution_plan.append({
                        'phase': 'training',
                        'commands': [cmd],
                        'parallel': False
                    })
        
        # Evaluation phase - depends on training, can be parallelized
        evaluation_commands = pipeline.get('evaluation', [])
        if evaluation_commands:
            execution_plan.append({
                'phase': 'evaluation',
                'commands': evaluation_commands,
                'parallel': len(evaluation_commands) > 1 and self.max_parallel > 1,
                'max_workers': min(self.max_parallel, len(evaluation_commands)) if len(evaluation_commands) > 1 else 1
            })
        
        # Plotting phase - depends on evaluation
        plotting_commands = pipeline.get('plotting', [])
        if plotting_commands:
            execution_plan.append({
                'phase': 'plotting',
                'commands': plotting_commands,
                'parallel': False
            })
        
        # IRT analysis phase - depends on evaluation
        irt_commands = pipeline.get('irt_analysis', [])
        if irt_commands:
            execution_plan.append({
                'phase': 'irt_analysis',
                'commands': irt_commands,
                'parallel': False
            })
        
        return execution_plan
    
    def estimate_total_resources(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate total pipeline resource requirements."""
        total_time = 0
        peak_memory = 0
        total_gpu_time = 0
        
        for phase in execution_plan:
            commands = phase['commands']
            parallel = phase.get('parallel', False)
            max_workers = phase.get('max_workers', 1)
            
            phase_resources = []
            for cmd in commands:
                resources = self.command_builder.estimate_command_resources(cmd)
                phase_resources.append(resources)
            
            if parallel and len(commands) > 1:
                # Parallel execution
                max_time = max(r['estimated_time_minutes'] for r in phase_resources)
                total_memory = sum(r['memory_gb'] for r in phase_resources[:max_workers])
                total_time += max_time
            else:
                # Sequential execution
                phase_time = sum(r['estimated_time_minutes'] for r in phase_resources)
                max_memory = max(r['memory_gb'] for r in phase_resources)
                total_time += phase_time
                total_memory = max_memory
            
            peak_memory = max(peak_memory, total_memory)
            
            # GPU time calculation
            gpu_commands = [r for r in phase_resources if r['gpu_required']]
            if parallel and gpu_commands:
                gpu_time = max(r['estimated_time_minutes'] for r in gpu_commands)
            else:
                gpu_time = sum(r['estimated_time_minutes'] for r in gpu_commands)
            total_gpu_time += gpu_time
        
        return {
            'total_time_minutes': total_time,
            'peak_memory_gb': peak_memory,
            'total_gpu_time_minutes': total_gpu_time,
            'estimated_cost_compute_hours': total_time / 60.0,
            'memory_warning': peak_memory > (self.memory_limit_gb or 8.0)
        }


class BatchCommandBuilder:
    """Builder for batch operations across multiple datasets/models."""
    
    @staticmethod
    def build_benchmark_commands(datasets: List[str], models: List[str], 
                                epochs: int = 30) -> Dict[str, List[str]]:
        """Build commands for comprehensive benchmarking."""
        commands = {}
        
        for dataset in datasets:
            dataset_commands = []
            
            # Training commands for all models
            for model in models:
                cmd = [
                    sys.executable, "train.py",
                    "--model", model,
                    "--dataset", dataset,
                    "--epochs", str(epochs)
                ]
                dataset_commands.append(cmd)
            
            # Evaluation commands
            for model in models:
                model_path = f"saved_models/{dataset}/best_{model}.pth"
                cmd = [
                    sys.executable, "evaluate.py",
                    "--model_path", model_path,
                    "--dataset", dataset,
                    "--regenerate_plots"
                ]
                dataset_commands.append(cmd)
            
            # Analysis commands
            plot_cmd = [sys.executable, "utils/plot_metrics.py", "--dataset", dataset]
            irt_cmd = [sys.executable, "analysis/irt_analysis.py", "--dataset", dataset]
            
            dataset_commands.extend([plot_cmd, irt_cmd])
            commands[dataset] = dataset_commands
        
        return commands
    
    @staticmethod
    def build_ablation_study_commands(base_config: TrainingConfig, 
                                    ablation_targets: List[str]) -> List[List[str]]:
        """Build commands for systematic ablation studies."""
        commands = []
        
        # Baseline command
        baseline_cmd = CommandBuilder.build_training_command(base_config)
        commands.append(baseline_cmd)
        
        # Ablation commands
        for target in ablation_targets:
            # Create modified config for ablation
            ablated_config = TrainingConfig(
                model=base_config.model,
                dataset=f"{base_config.dataset}_ablation_{target}",
                epochs=base_config.epochs,
                cv=base_config.cv
            )
            
            cmd = CommandBuilder.build_training_command(ablated_config)
            commands.append(cmd)
        
        return commands