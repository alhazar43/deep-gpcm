#!/usr/bin/env python3
"""
Optimized Main Runner for Deep-GPCM Complete Pipeline

This optimized version provides sequential execution workflow:
train → evaluate → plot → irt_analysis

Uses the unified configuration system for:
- 90% reduction in argument complexity
- Factory-driven model configurations
- Intelligent command building
- Resource-aware execution
"""

import os
import sys
import subprocess  # Still needed for plotting and IRT analysis
from pathlib import Path
from typing import Dict, Any, List

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from config import PipelineConfig
from config.parser import SmartArgumentParser
from utils.metrics import ensure_results_dirs
from utils.path_utils import ensure_directories
from utils.clean_res import ResultsCleaner


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command with enhanced error handling (for plotting and IRT analysis)."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


class OptimizedPipelineRunner:
    """Optimized pipeline runner with sequential execution workflow."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize results tracking
        self.results = {
            'training': {},
            'evaluation': {},
            'plotting': False,
            'irt_analysis': False
        }
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete Deep-GPCM pipeline sequentially: train → evaluate → plot → irt_analysis."""
        print("🚀 Starting Deep-GPCM Optimized Complete Pipeline")
        print(f"📊 Dataset: {self.config.dataset}")
        print(f"🤖 Models: {', '.join(self.config.models)}")
        print(f"⚙️  Sequential execution: train → evaluate → plot → irt_analysis")
        print("=" * 80)
        
        # Setup environment
        self._setup_environment()
        
        # Phase 1: Training
        print("\n" + "="*20 + " TRAINING PHASE " + "="*20)
        training_success = self._run_training_phase()
        
        # Phase 2: Evaluation (only if training succeeded)
        evaluation_success = False
        if training_success:
            print("\n" + "="*20 + " EVALUATION PHASE " + "="*20)
            evaluation_success = self._run_evaluation_phase()
        else:
            print("\n⚠️  Skipping evaluation due to training failures")
        
        # Phase 3: Plotting (only if evaluation succeeded)
        plotting_success = False
        if evaluation_success:
            print("\n" + "="*20 + " PLOTTING PHASE " + "="*20)
            plotting_success = self._run_plotting_phase()
        else:
            print("\n⚠️  Skipping plotting due to evaluation failures")
        
        # Phase 4: IRT Analysis (only if plotting succeeded)
        irt_success = False
        if plotting_success:
            print("\n" + "="*20 + " IRT ANALYSIS PHASE " + "="*20)
            irt_success = self._run_irt_analysis_phase()
        else:
            print("\n⚠️  Skipping IRT analysis due to plotting failures")
        
        # Print final summary
        self._print_pipeline_summary(training_success, evaluation_success, plotting_success, irt_success)
        
        return self.results
    
    def _setup_environment(self):
        """Setup directories and environment."""
        ensure_directories(self.config.dataset)
        ensure_results_dirs()
        
        # Cleanup if requested
        if getattr(self.config, 'clean_before', False):
            print("🧹 Cleaning existing results...")
            cleaner = ResultsCleaner()
            cleaner.clean_dataset(self.config.dataset, backup=True)
    
    def _run_training_phase(self) -> bool:
        """Run training for all models using direct function calls (no subprocess overhead)."""
        print(f"🎯 Training {len(self.config.models)} models...")
        
        # Import training function directly to avoid subprocess overhead
        from train_optimized import run_training_workflow
        from config import TrainingConfig
        
        all_success = True
        
        for i, model in enumerate(self.config.models, 1):
            print(f"\n{'='*20} Training {model} ({i}/{len(self.config.models)}) {'='*20}")
            
            try:
                # Create training config for this model (direct object, no subprocess)
                training_config = TrainingConfig(
                    model=model,
                    dataset=self.config.dataset,
                    epochs=self.config.epochs,
                    n_folds=self.config.n_folds,
                    device=self.config.device,
                    cv=getattr(self.config, 'cv', False),
                    seed=getattr(self.config, 'seed', 42)
                )
                
                # Direct function call - NO subprocess overhead
                results = run_training_workflow(training_config)
                self.results['training'][model] = True
                print(f"✅ {model} training completed successfully")
                
            except Exception as e:
                print(f"❌ Training failed for {model}: {e}")
                self.results['training'][model] = False
                all_success = False
                print(f"⚠️  Continuing with other models...")
        
        if all_success:
            print("\n✅ All training completed successfully")
        else:
            trained_models = [m for m, success in self.results['training'].items() if success]
            print(f"\n⚠️  Training completed with some failures. Successfully trained: {trained_models}")
        
        return len([s for s in self.results['training'].values() if s]) > 0
    
    def _run_evaluation_phase(self) -> bool:
        """Run evaluation for all successfully trained models using direct function calls."""
        trained_models = [model for model, success in self.results['training'].items() if success]
        
        if not trained_models:
            print("❌ No models were successfully trained")
            return False
        
        print(f"📊 Evaluating {len(trained_models)} trained models...")
        
        # Import evaluation function directly to avoid subprocess overhead
        from evaluate_optimized import run_evaluation_workflow
        from config import EvaluationConfig
        
        all_success = True
        
        for i, model in enumerate(trained_models, 1):
            print(f"\n{'='*20} Evaluating {model} ({i}/{len(trained_models)}) {'='*20}")
            
            try:
                # Find the best model path
                model_path = f"saved_models/{self.config.dataset}/best_{model}.pth"
                
                # Create evaluation config for this model (direct object, no subprocess)
                eval_config = EvaluationConfig(
                    model_path=model_path,
                    dataset=self.config.dataset,
                    device=self.config.device
                )
                
                # Direct function call - NO subprocess overhead
                results = run_evaluation_workflow(eval_config)
                self.results['evaluation'][model] = True
                print(f"✅ {model} evaluation completed successfully")
                
            except Exception as e:
                print(f"❌ Evaluation failed for {model}: {e}")
                self.results['evaluation'][model] = False
                all_success = False
                print(f"⚠️  Continuing with other models...")
        
        if all_success:
            print("\n✅ All evaluation completed successfully")
        else:
            evaluated_models = [m for m, success in self.results['evaluation'].items() if success]
            print(f"\n⚠️  Evaluation completed with some failures. Successfully evaluated: {evaluated_models}")
        
        return len([s for s in self.results['evaluation'].values() if s]) > 0
    
    def _run_plotting_phase(self) -> bool:
        """Run plotting to generate all visualization results."""
        print("📈 Generating plots for all results...")
        
        # Build optimized plotting command
        cmd = [
            sys.executable, "utils/plot_metrics_optimized.py",
            "--dataset", self.config.dataset,
            "--plot_quality", "high",
            "--save_formats", "png", "pdf"
        ]
        
        success = run_command(cmd, "Generating comprehensive plots")
        self.results['plotting'] = success
        
        if success:
            print("✅ Plotting completed successfully")
        else:
            print("❌ Plotting failed")
        
        return success
    
    def _run_irt_analysis_phase(self) -> bool:
        """Run IRT analysis for parameter recovery and temporal analysis."""
        print("🧠 Running IRT analysis...")
        
        # Build optimized IRT analysis command
        cmd = [
            sys.executable, "analysis/irt_analysis_optimized.py",
            "--dataset", self.config.dataset,
            "--analysis_types", "recovery", "temporal",
            "--save_parameters",
            "--save_summary"
        ]
        
        success = run_command(cmd, "IRT parameter analysis")
        self.results['irt_analysis'] = success
        
        if success:
            print("✅ IRT analysis completed successfully")
        else:
            print("❌ IRT analysis failed")
        
        return success
    
    def _print_pipeline_summary(self, training_success: bool, evaluation_success: bool, 
                               plotting_success: bool, irt_success: bool):
        """Print comprehensive pipeline summary."""
        print("\n" + "="*60)
        print("DEEP-GPCM OPTIMIZED PIPELINE SUMMARY")
        print("="*60)
        
        # Training summary
        print(f"\n🎯 TRAINING: {'✅' if training_success else '❌'}")
        for model, success in self.results['training'].items():
            status = "✅" if success else "❌"
            print(f"  {status} {model}")
        
        # Evaluation summary
        print(f"\n📊 EVALUATION: {'✅' if evaluation_success else '❌'}")
        for model, success in self.results['evaluation'].items():
            status = "✅" if success else "❌"
            print(f"  {status} {model}")
        
        # Plotting summary
        print(f"\n📈 PLOTTING: {'✅' if plotting_success else '❌'}")
        if plotting_success:
            print(f"  📁 Plots saved to: results/plots/{self.config.dataset}/")
        
        # IRT Analysis summary
        print(f"\n🧠 IRT ANALYSIS: {'✅' if irt_success else '❌'}")
        if irt_success:
            print(f"  📁 Analysis saved to: results/irt_plots/{self.config.dataset}/")
        
        # Overall success
        overall_success = training_success and evaluation_success and plotting_success and irt_success
        print(f"\n🎉 OVERALL PIPELINE: {'✅ COMPLETE SUCCESS' if overall_success else '⚠️  PARTIAL SUCCESS'}")
        
        if overall_success:
            print("\n🚀 All phases completed successfully!")
            print("📂 Results available in:")
            print(f"   - Training results: results/train/{self.config.dataset}/")
            print(f"   - Test results: results/test/{self.config.dataset}/")
            print(f"   - Plots: results/plots/{self.config.dataset}/")
            print(f"   - IRT Analysis: results/irt_plots/{self.config.dataset}/")
        else:
            print("\n⚠️  Pipeline completed with some issues. Check logs above for details.")
        
        print("="*60)


def main():
    """Main entry point with optimized argument parsing."""
    
    try:
        # Parse pipeline configuration
        parser = SmartArgumentParser.create_pipeline_parser()
        config = parser.parse()
        
        # Handle cleanup before pipeline execution
        if config.clean and config.dataset:
            cleaner = ResultsCleaner()
            
            # First show what will be deleted (dry run)
            print(f"\n🧹 Cleanup requested for dataset: {config.dataset}")
            print("=" * 60)
            counts = cleaner.clean_dataset(config.dataset, dry_run=True, backup=False)
            
            if counts['directories'] == 0 and counts['files'] == 0:
                print(f"ℹ️  No files found for dataset '{config.dataset}' - nothing to clean")
            else:
                # Show summary and ask for confirmation
                print(f"\n📋 CLEANUP SUMMARY:")
                print(f"   • {counts['directories']} directories")
                print(f"   • {counts['files']} files") 
                print(f"   • {counts['total_size'] / (1024*1024):.2f} MB total size")
                
                backup_action = "🚫 No backup" if config.no_backup else "💾 With backup"
                print(f"   • {backup_action}")
                
                # Confirmation prompt
                response = input(f"\n⚠️  Are you sure you want to delete all results for '{config.dataset}'? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    print(f"\n🧹 Cleaning existing results for {config.dataset}...")
                    result_counts = cleaner.clean_dataset(
                        config.dataset,
                        dry_run=False,
                        backup=not config.no_backup
                    )
                    print(f"✅ Cleanup completed: {result_counts['directories']} directories, {result_counts['files']} files removed")
                else:
                    print("❌ Cleanup cancelled by user")
                    sys.exit(0)
        
        print("🚀 Deep-GPCM Optimized Complete Pipeline")
        print(f"📊 Configuration: {config.dataset} dataset, {len(config.models)} models")
        print(f"🔄 Training: {config.epochs} epochs, {config.n_folds} folds, CV: {config.cv}")
        print()
        
        # Run complete pipeline
        runner = OptimizedPipelineRunner(config)
        results = runner.run_complete_pipeline()
        
        
        # Exit with appropriate code
        overall_success = (
            any(results['training'].values()) and
            any(results['evaluation'].values()) and
            results['plotting'] and
            results['irt_analysis']
        )
        
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()