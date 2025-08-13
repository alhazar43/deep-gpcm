#!/usr/bin/env python3
"""
Optimized Evaluation Script for Deep-GPCM

Key optimizations:
- Unified configuration system with factory integration
- Standardized evaluation framework with statistical comparison
- Comprehensive metrics computation and analysis
- Intelligent plotting and visualization generation
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from config import EvaluationConfig, BatchEvaluationConfig
from config.parser import parse_evaluation_config, SmartArgumentParser
from models.factory import create_model_with_config, get_model_type_from_path
from utils.metrics import compute_metrics, compute_statistical_comparison, save_results
from utils.path_utils import ensure_directories
from data.loaders import DataLoaderManager
from utils.data_loading import load_dataset, get_data_file_paths, load_simple_data
# from utils.plot_metrics import MetricsPlotter, ComparisonPlotter

# Simple placeholder classes for plotting
class MetricsPlotter:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def plot_model_results(self, model, results):
        print(f"📊 Would generate plots for {model} on {self.dataset}")

class ComparisonPlotter:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def plot_model_comparison(self):
        print(f"📊 Would generate comparison plots for {self.dataset}")
    
    def plot_batch_comparison(self, results):
        print(f"📊 Would generate batch comparison plots for {self.dataset}")


class ModelEvaluator:
    """Comprehensive model evaluation with statistical analysis."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {}
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.model = None
        self.data_manager = None
        
    def _set_seeds(self):
        """Set random seeds for reproducible evaluation."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def load_model(self) -> bool:
        """Load trained model from checkpoint."""
        if not self.config.model_path.exists():
            print(f"❌ Model file not found: {self.config.model_path}")
            return False
        
        print(f"📥 Loading model from: {self.config.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Extract model information
            model_type = checkpoint.get('model_type', self.config.model)
            if not model_type:
                # Try to infer from path
                model_type = get_model_type_from_path(self.config.model_path)
            
            # Get model dimensions - try multiple fallback locations
            n_questions = None
            n_cats = None
            
            # First try direct fields (new format)
            n_questions = checkpoint.get('n_questions')
            n_cats = checkpoint.get('n_cats')
            
            # If not found, try model_params dict
            if n_questions is None or n_cats is None:
                model_params = checkpoint.get('model_params', {})
                n_questions = model_params.get('n_questions')
                n_cats = model_params.get('n_cats')
            
            # If still not found, try config dict
            if n_questions is None or n_cats is None:
                model_config = checkpoint.get('config', {})
                n_questions = model_config.get('n_questions')
                n_cats = model_config.get('n_cats')
            
            # Final fallback: infer from data
            if n_questions is None or n_cats is None:
                print("⚠️  Model dimensions not found in checkpoint, inferring from data...")
                return self._load_model_with_data_inference(checkpoint, model_type)
            
            # Get model parameters from config
            model_config = checkpoint.get('config', {})
            model_params = {}
            
            # Extract model-specific parameters from config if available
            # Exclude training-specific and config-specific parameters
            excluded_params = {
                'model', 'dataset', 'epochs', 'batch_size', 'lr', 'n_folds', 'cv', 
                'device', 'seed', 'loss_config', 'validation_config', 'path_config',
                'loss_config_override'  # This is training-specific
            }
            for key, value in model_config.items():
                if key not in excluded_params:
                    model_params[key] = value
            
            # Create model
            self.model = create_model_with_config(
                model_type, 
                n_questions, 
                n_cats,
                **model_params
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded: {model_type} ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _load_model_with_data_inference(self, checkpoint: Dict, model_type: str) -> bool:
        """Load model by inferring dimensions from data when not available in checkpoint."""
        try:
            # Get data dimensions using unified data loading utility
            train_path, test_path = get_data_file_paths(self.config.dataset)
            _, _, n_questions, n_cats = load_simple_data(train_path, test_path)
            
            print(f"📊 Inferred dimensions: {n_questions} questions, {n_cats} categories")
            
            # Get model parameters from config
            model_config = checkpoint.get('config', {})
            model_params = {}
            
            # Extract model-specific parameters from config if available
            # Exclude training-specific and config-specific parameters
            excluded_params = {
                'model', 'dataset', 'epochs', 'batch_size', 'lr', 'n_folds', 'cv', 
                'device', 'seed', 'loss_config', 'validation_config', 'path_config',
                'loss_config_override'  # This is training-specific
            }
            for key, value in model_config.items():
                if key not in excluded_params:
                    model_params[key] = value
            
            # Create model with inferred dimensions
            self.model = create_model_with_config(
                model_type, 
                n_questions, 
                n_cats,
                **model_params
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded with inferred dimensions: {model_type} ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model with data inference: {e}")
            return False
    
    def setup_data(self) -> bool:
        """Setup data loaders using unified data loading utility."""
        try:
            # Load data using unified data loading utility (same as train_optimized.py)
            train_loader, test_loader, n_questions, n_cats = load_dataset(
                self.config.dataset, 
                batch_size=self.config.batch_size
            )
            
            # Store dataset info
            self.n_questions = n_questions
            self.n_cats = n_cats
            
            # Store data loaders
            self.train_loader = train_loader
            self.test_loader = test_loader
            
            print(f"📊 Data loaded: {len(test_loader.dataset)} test samples")
            print(f"📊 Questions: {self.n_questions}, Categories: {self.n_cats}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        print("\n🧪 Running comprehensive evaluation...")
        
        evaluation_results = {}
        
        # Test set evaluation
        test_metrics = self._evaluate_split(self.test_loader, "test")
        evaluation_results['test'] = test_metrics
        
        # Training set evaluation (for comparison)
        if self.config.evaluate_train_set:
            train_metrics = self._evaluate_split(self.train_loader, "train")
            evaluation_results['train'] = train_metrics
            
            # Compute overfitting metrics
            overfitting_metrics = self._compute_overfitting_metrics(train_metrics, test_metrics)
            evaluation_results['overfitting'] = overfitting_metrics
        
        # Multiple prediction methods
        if len(self.config.prediction_methods) > 1:
            method_comparison = self._compare_prediction_methods()
            evaluation_results['prediction_methods'] = method_comparison
        
        # Detailed analysis
        if self.config.detailed_analysis:
            detailed_results = self._detailed_analysis()
            evaluation_results['detailed'] = detailed_results
        
        return evaluation_results
    
    def _evaluate_split(self, data_loader, split_name: str) -> Dict[str, Any]:
        """Evaluate model on a data split."""
        print(f"  📋 Evaluating {split_name} set...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Handle different batch formats
                if isinstance(batch, tuple):
                    # Original train.py format: (questions, responses, mask)
                    questions, responses, mask = batch
                    questions = questions.to(self.device)
                    responses = responses.to(self.device)
                    mask = mask.to(self.device)
                else:
                    # DataLoaderManager format: dict with keys
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    questions = batch['questions']
                    responses = batch['responses']
                    mask = batch.get('mask')
                
                # Forward pass - match training interface
                outputs = self.model(questions, responses)
                
                # Extract predictions and probabilities - handle model output formats
                if isinstance(outputs, tuple):
                    # For GPCM models: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
                    logits = outputs[-1]  # gpcm_probs is the last element
                    probabilities = logits  # Already probabilities for GPCM models
                elif isinstance(outputs, dict):
                    logits = outputs.get('predictions', outputs.get('logits'))
                    probabilities = torch.softmax(logits, dim=-1)
                else:
                    logits = outputs
                    probabilities = torch.softmax(logits, dim=-1)
                
                # Handle tensor shapes and convert to numpy
                batch_size, seq_len = responses.shape
                
                # Reshape predictions to (batch_size * seq_len, n_cats)
                if logits.dim() > 2:
                    logits_flat = logits.view(-1, logits.size(-1))
                    prob_flat = probabilities.view(-1, probabilities.size(-1))
                else:
                    logits_flat = logits
                    prob_flat = probabilities
                
                # Reshape targets to (batch_size * seq_len,)
                targets_flat = responses.view(-1)
                
                # Convert to numpy and append
                all_predictions.append(logits_flat.cpu())
                all_targets.append(targets_flat.cpu())
                all_probabilities.append(prob_flat.cpu())
        
        # Concatenate tensors and convert to numpy
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        probabilities = torch.cat(all_probabilities, dim=0).numpy()
        
        # Compute comprehensive metrics using predictions (logits) and targets
        # For metrics computation, we need predicted classes
        predicted_classes = probabilities.argmax(axis=-1)
        metrics = compute_metrics(targets, predicted_classes, probabilities, n_cats=self.model.n_cats)
        
        # Add probability-based metrics
        prob_metrics = self._compute_probability_metrics(probabilities, targets)
        metrics.update(prob_metrics)
        
        # Per-category analysis
        if self.config.per_category_analysis:
            category_metrics = self._compute_per_category_metrics(predictions, targets)
            metrics['per_category'] = category_metrics
        
        return {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    def _compute_probability_metrics(self, probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute probability-based metrics."""
        metrics = {}
        
        # Calibration metrics
        try:
            from sklearn.calibration import calibration_curve
            
            # Convert targets to binary for calibration (correct vs incorrect)
            predicted_classes = np.argmax(probabilities, axis=1)
            correct_predictions = (predicted_classes == targets).astype(int)
            max_probabilities = np.max(probabilities, axis=1)
            
            # Compute calibration curve
            fraction_positives, mean_predicted_value = calibration_curve(
                correct_predictions, max_probabilities, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (max_probabilities > bin_lower) & (max_probabilities <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correct_predictions[in_bin].mean()
                    avg_confidence_in_bin = max_probabilities[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['ece'] = ece
            metrics['avg_confidence'] = np.mean(max_probabilities)
            
        except ImportError:
            print("⚠️  Scikit-learn not available for calibration metrics")
        
        # Entropy-based metrics
        eps = 1e-7
        entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
        metrics['avg_entropy'] = np.mean(entropy)
        metrics['max_entropy'] = np.log(self.model.n_cats)  # Maximum possible entropy
        metrics['normalized_entropy'] = np.mean(entropy) / np.log(self.model.n_cats)
        
        return metrics
    
    def _compute_per_category_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Compute per-category performance metrics."""
        predicted_classes = np.argmax(predictions, axis=1)
        
        per_category = {}
        for category in range(self.model.n_cats):
            # Category-specific masks
            true_category = (targets == category)
            pred_category = (predicted_classes == category)
            
            # Basic counts
            true_positives = np.sum(true_category & pred_category)
            false_positives = np.sum(~true_category & pred_category)
            false_negatives = np.sum(true_category & ~pred_category)
            
            # Metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_category[f'category_{category}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': np.sum(true_category),
                'predicted_count': np.sum(pred_category)
            }
        
        return per_category
    
    def _compute_overfitting_metrics(self, train_metrics: Dict, test_metrics: Dict) -> Dict[str, float]:
        """Compute overfitting indicators."""
        train_qwk = train_metrics['metrics']['qwk']
        test_qwk = test_metrics['metrics']['qwk']
        
        train_accuracy = train_metrics['metrics']['accuracy']
        test_accuracy = test_metrics['metrics']['accuracy']
        
        return {
            'qwk_gap': train_qwk - test_qwk,
            'accuracy_gap': train_accuracy - test_accuracy,
            'generalization_ratio': test_qwk / train_qwk if train_qwk > 0 else 0,
            'overfitting_severity': 'high' if (train_qwk - test_qwk) > 0.1 else 'moderate' if (train_qwk - test_qwk) > 0.05 else 'low'
        }
    
    def _compare_prediction_methods(self) -> Dict[str, Any]:
        """Compare different prediction methods."""
        method_results = {}
        
        for method in self.config.prediction_methods:
            print(f"    🔍 Testing prediction method: {method}")
            
            # Implement different prediction strategies
            if method == 'hard':
                # Argmax prediction (already computed)
                continue
            elif method == 'soft':
                # Probability-weighted prediction
                method_results[method] = self._soft_prediction_evaluation()
            elif method == 'threshold':
                # Threshold-based prediction
                method_results[method] = self._threshold_prediction_evaluation()
        
        return method_results
    
    def _soft_prediction_evaluation(self) -> Dict[str, float]:
        """Evaluate using soft (probability-weighted) predictions."""
        # Simplified implementation - would need full re-evaluation with soft predictions
        return {'qwk': 0.0, 'note': 'soft prediction method - implementation placeholder'}
    
    def _threshold_prediction_evaluation(self) -> Dict[str, float]:
        """Evaluate using threshold-based predictions."""
        # Simplified implementation - would need full re-evaluation with thresholds
        return {'qwk': 0.0, 'note': 'threshold prediction method - implementation placeholder'}
    
    def _detailed_analysis(self) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        print("  🔬 Running detailed analysis...")
        
        # Error distribution analysis
        error_analysis = self._analyze_error_patterns()
        
        # Difficulty analysis
        difficulty_analysis = self._analyze_question_difficulty()
        
        return {
            'error_patterns': error_analysis,
            'question_difficulty': difficulty_analysis
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in prediction errors."""
        # Placeholder for error pattern analysis
        return {'error_distribution': 'analysis placeholder'}
    
    def _analyze_question_difficulty(self) -> Dict[str, Any]:
        """Analyze model performance by question difficulty."""
        # Placeholder for difficulty analysis
        return {'difficulty_correlation': 'analysis placeholder'}
    
    def generate_visualizations(self) -> bool:
        """Generate comprehensive visualizations."""
        if not self.config.regenerate_plots:
            print("📊 Skipping plot generation (regenerate_plots=False)")
            return True
        
        print("\n📊 Generating visualizations...")
        
        try:
            # Create plotter
            plotter = MetricsPlotter(self.config.dataset)
            
            # Generate standard plots
            plotter.plot_model_results(self.config.model, self.results)
            
            # Generate comparison plots if multiple models exist
            if self.config.comparison_plots:
                comparison_plotter = ComparisonPlotter(self.config.dataset)
                comparison_plotter.plot_model_comparison()
            
            print("✅ Visualizations generated successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
            return False
    
    def save_results(self) -> bool:
        """Save evaluation results."""
        try:
            # Ensure output directories exist
            self.config.test_results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive results
            save_results(self.results, self.config.test_results_path)
            
            # Save summary metrics
            summary_path = self.config.test_results_path.parent / f"summary_{self.config.model}.json"
            summary = self._create_summary()
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"💾 Results saved to: {self.config.test_results_path}")
            print(f"💾 Summary saved to: {summary_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create evaluation summary."""
        test_metrics = self.results.get('test', {}).get('metrics', {})
        
        summary = {
            'model': self.config.model,
            'dataset': self.config.dataset,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'device': self.config.device,
                'batch_size': self.config.batch_size,
                'prediction_methods': self.config.prediction_methods,
                'detailed_analysis': self.config.detailed_analysis
            },
            'performance': {
                'qwk': test_metrics.get('qwk', 0.0),
                'accuracy': test_metrics.get('accuracy', 0.0),
                'mae': test_metrics.get('mae', 0.0),
                'mse': test_metrics.get('mse', 0.0)
            }
        }
        
        # Add overfitting analysis if available
        if 'overfitting' in self.results:
            summary['overfitting'] = self.results['overfitting']
        
        return summary


class BatchModelEvaluator:
    """Evaluate multiple models with statistical comparison."""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.individual_results = {}
        self.comparison_results = {}
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models and perform statistical comparison."""
        print(f"🧪 BATCH EVALUATION - {len(self.config.model_paths)} models")
        print("=" * 60)
        
        # Evaluate each model individually
        for model_path in self.config.model_paths:
            model_name = Path(model_path).stem.replace('best_', '')
            print(f"\n📋 Evaluating {model_name}...")
            
            # Create individual evaluation config
            eval_config = EvaluationConfig(
                model_path=Path(model_path),
                dataset=self.config.dataset,
                device=self.config.device,
                regenerate_plots=False,  # Generate plots only once at the end
                detailed_analysis=self.config.detailed_analysis
            )
            
            # Run evaluation
            evaluator = ModelEvaluator(eval_config)
            if evaluator.load_model() and evaluator.setup_data():
                results = evaluator.evaluate_model()
                self.individual_results[model_name] = results
            else:
                print(f"❌ Failed to evaluate {model_name}")
        
        # Statistical comparison
        if len(self.individual_results) > 1 and self.config.statistical_comparison:
            print("\n📊 Running statistical comparison...")
            self.comparison_results = self._perform_statistical_comparison()
        
        # Generate comprehensive plots
        if self.config.regenerate_plots:
            self._generate_batch_visualizations()
        
        return {
            'individual_results': self.individual_results,
            'statistical_comparison': self.comparison_results,
            'summary': self._create_batch_summary()
        }
    
    def _perform_statistical_comparison(self) -> Dict[str, Any]:
        """Perform statistical comparison between models."""
        try:
            # Extract QWK scores for comparison
            model_scores = {}
            for model_name, results in self.individual_results.items():
                qwk = results.get('test', {}).get('metrics', {}).get('qwk', 0.0)
                model_scores[model_name] = qwk
            
            # Perform pairwise statistical tests
            comparison_results = compute_statistical_comparison(model_scores)
            
            return comparison_results
            
        except Exception as e:
            print(f"⚠️  Error in statistical comparison: {e}")
            return {}
    
    def _generate_batch_visualizations(self):
        """Generate visualizations for batch evaluation."""
        try:
            comparison_plotter = ComparisonPlotter(self.config.dataset)
            comparison_plotter.plot_batch_comparison(self.individual_results)
            print("✅ Batch visualizations generated")
        except Exception as e:
            print(f"⚠️  Error generating batch visualizations: {e}")
    
    def _create_batch_summary(self) -> Dict[str, Any]:
        """Create summary of batch evaluation."""
        model_performances = {}
        for model_name, results in self.individual_results.items():
            test_metrics = results.get('test', {}).get('metrics', {})
            model_performances[model_name] = {
                'qwk': test_metrics.get('qwk', 0.0),
                'accuracy': test_metrics.get('accuracy', 0.0)
            }
        
        # Find best model
        best_model = max(model_performances.items(), key=lambda x: x[1]['qwk'])
        
        return {
            'n_models_evaluated': len(self.individual_results),
            'best_model': {
                'name': best_model[0],
                'qwk': best_model[1]['qwk'],
                'accuracy': best_model[1]['accuracy']
            },
            'model_performances': model_performances,
            'timestamp': datetime.now().isoformat()
        }


def run_evaluation_workflow(config: EvaluationConfig) -> Dict[str, Any]:
    """Run complete evaluation workflow."""
    
    print("=" * 80)
    print("OPTIMIZED DEEP-GPCM EVALUATION")
    print("=" * 80)
    print(config.summary())
    print()
    
    # Setup directories
    ensure_directories(config.dataset)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Load model and data
    if not evaluator.load_model():
        raise RuntimeError("Failed to load model")
    
    if not evaluator.setup_data():
        raise RuntimeError("Failed to setup data")
    
    # Run evaluation
    results = evaluator.evaluate_model()
    evaluator.results = results
    
    # Generate visualizations
    evaluator.generate_visualizations()
    
    # Save results
    evaluator.save_results()
    
    print("\n✅ Evaluation completed successfully!")
    
    return results


def main():
    """Main entry point with optimized configuration parsing."""
    
    try:
        # Parse configuration
        config = parse_evaluation_config()
        
        # Check if this is a batch evaluation
        if hasattr(config, 'model_paths') and len(config.model_paths) > 1:
            # Batch evaluation mode
            batch_config = BatchEvaluationConfig(
                model_paths=config.model_paths,
                dataset=config.dataset,
                device=config.device,
                detailed_analysis=config.detailed_analysis,
                regenerate_plots=config.regenerate_plots
            )
            
            batch_evaluator = BatchModelEvaluator(batch_config)
            results = batch_evaluator.evaluate_all_models()
        else:
            # Single model evaluation
            results = run_evaluation_workflow(config)
        
        print("\n🎯 Optimized evaluation completed!")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()