#!/usr/bin/env python3
"""
Adaptive IRT Evaluation Script
Integrates adaptive IRT into the existing evaluation pipeline to generate
predictions in the same format as other models and create comparison confusion matrices.
"""

import os
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import argparse

from adaptive_irt_test import AdaptiveGPCMTest, load_synthetic_data
from utils.plot_metrics import AdaptivePlotter
from utils.metrics import compute_metrics


class AdaptiveIRTEvaluator:
    """Evaluates adaptive IRT in the same format as other models"""
    
    def __init__(self, dataset_path, max_questions=15):
        self.dataset_path = Path(dataset_path)
        self.max_questions = max_questions
        self.data = load_synthetic_data(dataset_path)
        self.results = {}
        
    def evaluate_all_students(self, use_test_data=True):
        """Evaluate adaptive IRT using actual test data sequences"""
        
        if use_test_data:
            # Load actual test data sequences
            test_file = self.dataset_path / "synthetic_oc_test.txt"
            with open(test_file, 'r') as f:
                lines = f.readlines()
            
            # Parse test sequences
            test_sequences = []
            i = 0
            while i < len(lines):
                if i + 2 >= len(lines):
                    break
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(int, lines[i+2].strip().split(',')))
                
                # Ensure lengths match
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                test_sequences.append((questions, responses))
                i += 3
            
            print(f"Evaluating Adaptive IRT on {len(test_sequences)} test sequences...")
            print(f"Using actual test data sequences (varying lengths)")
        else:
            # Fallback to synthetic generation
            n_students = min(100, len(self.data['true_theta']))
            print(f"Evaluating Adaptive IRT on {n_students} students...")
            print(f"Max questions per student: {self.max_questions}")
        
        print("=" * 60)
        
        all_predictions = []
        all_true_responses = []
        all_probabilities = []
        theta_estimates = []
        theta_errors = []
        
        if use_test_data:
            # Process subset of test sequences for efficiency (25 sequences)
            max_sequences = min(25, len(test_sequences))
            selected_sequences = test_sequences[:max_sequences]
            
            print(f"Processing {max_sequences} sequences (subset for efficiency)...")
            
            for seq_idx, (questions, responses) in enumerate(selected_sequences):
                if seq_idx % 5 == 0:
                    print(f"Processing sequence {seq_idx}/{max_sequences}...")
                
                # Get true theta for this sequence (assume sequential student mapping)
                student_id = seq_idx % len(self.data['true_theta'])
                true_theta = self.data['true_theta'][student_id]
                true_alpha = self.data['true_alpha']
                true_beta = self.data['true_beta']
                
                # Initialize adaptive test
                adaptive_test = AdaptiveGPCMTest(len(true_alpha), self.data['n_cats'])
                
                # Process each question-response pair in the sequence
                for q_idx, (question_id, true_response) in enumerate(zip(questions, responses)):
                    # Get prediction probabilities before seeing response
                    pred_probs = adaptive_test.predict_response(question_id)
                    predicted_response = np.argmax(pred_probs)
                    
                    # Store results
                    all_predictions.append(predicted_response)
                    all_true_responses.append(true_response)
                    all_probabilities.append(pred_probs.tolist())
                    
                    # Update adaptive test with true response
                    adaptive_test.administer_question(question_id, true_response)
                
                # Track theta estimation performance for this sequence
                theta_estimates.append(adaptive_test.current_theta)
                theta_errors.append(abs(adaptive_test.current_theta - true_theta))
        
        else:
            # Original synthetic generation code
            for student_id in range(n_students):
                if student_id % 10 == 0:
                    print(f"Processing student {student_id}/{n_students}...")
                
                # Get true parameters for this student
                true_theta = self.data['true_theta'][student_id]
                true_alpha = self.data['true_alpha']
                true_beta = self.data['true_beta']
                
                # Initialize adaptive test
                adaptive_test = AdaptiveGPCMTest(len(true_alpha), self.data['n_cats'])
                
                # Run adaptive test
                administered_questions = set()
                
                for _ in range(self.max_questions):
                    # Select question
                    next_q = adaptive_test.select_next_question(administered_questions)
                    if next_q is None:
                        break
                    administered_questions.add(next_q)
                    
                    # Generate true response
                    true_probs = adaptive_test.gpcm_prob(true_theta, true_alpha[next_q], true_beta[next_q])
                    true_response = np.random.choice(self.data['n_cats'], p=true_probs)
                    
                    # Get prediction probabilities before seeing response
                    pred_probs = adaptive_test.predict_response(next_q)
                    predicted_response = np.argmax(pred_probs)
                    
                    # Store results
                    all_predictions.append(predicted_response)
                    all_true_responses.append(true_response)
                    all_probabilities.append(pred_probs.tolist())
                    
                    # Update adaptive test with true response
                    adaptive_test.administer_question(next_q, true_response)
                
                # Track theta estimation performance
                theta_estimates.append(adaptive_test.current_theta)
                theta_errors.append(abs(adaptive_test.current_theta - true_theta))
        
        # Compute overall metrics
        accuracy = accuracy_score(all_true_responses, all_predictions)
        f1 = f1_score(all_true_responses, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_true_responses, all_predictions, 
                                     labels=list(range(self.data['n_cats'])))
        
        # Store results in format compatible with existing pipeline
        self.results = {
            'model_type': 'adaptive_irt',
            'config': {
                'model_type': 'adaptive_irt',
                'model': 'adaptive_irt',
                'max_questions': self.max_questions,
                'n_sequences_evaluated': len(theta_estimates)
            },
            'predictions': [int(p) for p in all_predictions],
            'actual': [int(a) for a in all_true_responses],
            'probabilities': all_probabilities,
            'confusion_matrix': conf_matrix.tolist(),
            'evaluation_results': {
                'accuracy': float(accuracy),
                'f1_weighted': float(f1),
                'confusion_matrix': conf_matrix.tolist(),
                'theta_mae': float(np.mean(theta_errors)),
                'theta_estimates': [float(x) for x in theta_estimates],
                'theta_errors': [float(x) for x in theta_errors],
                'n_predictions': len(all_predictions)
            },
            'ordinal_distances': [int(abs(p - a)) for p, a in zip(all_predictions, all_true_responses)]
        }
        
        print(f"\nAdaptive IRT Results:")
        print(f"  Predictions made: {len(all_predictions)}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 (weighted): {f1:.3f}")
        print(f"  Theta MAE: {np.mean(theta_errors):.3f}")
        
        return self.results
    
    def save_results(self, output_path=None):
        """Save results in format compatible with existing pipeline"""
        if not self.results:
            print("No results to save. Run evaluate_all_students() first.")
            return
        
        if output_path is None:
            output_path = self.dataset_path / "results" / "test" / "adaptive_irt_results.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def load_existing_results(self):
        """Load existing model results for comparison"""
        results_dir = self.dataset_path / "results" / "test"
        existing_results = []
        
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return existing_results
        
        # Load all JSON result files
        for result_file in results_dir.glob("*.json"):
            if result_file.name.startswith("adaptive_irt"):
                continue  # Skip our own results
            
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    existing_results.append(result)
                    print(f"Loaded results from: {result_file.name}")
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        
        return existing_results
    
    def create_comparison_plots(self):
        """Create confusion matrix comparison with existing models"""
        
        # Load existing results
        existing_results = self.load_existing_results()
        
        if not existing_results:
            print("No existing results found for comparison")
            return
        
        # Add our results
        if self.results:
            all_results = existing_results + [self.results]
        else:
            print("No adaptive IRT results available")
            return
        
        # Create plot manager
        plots_dir = self.dataset_path / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plotter = AdaptivePlotter(plots_dir)
        
        # Generate confusion matrix comparison
        print("\nGenerating confusion matrix comparison...")
        confusion_plot_path = plotter.plot_confusion_matrices(all_results, "test")
        
        # Also create individual adaptive IRT confusion matrix
        self.plot_adaptive_irt_confusion_matrix()
        
        return confusion_plot_path
    
    def plot_adaptive_irt_confusion_matrix(self):
        """Create standalone confusion matrix for adaptive IRT"""
        if not self.results:
            return
        
        conf_matrix = np.array(self.results['confusion_matrix'])
        accuracy = self.results['evaluation_results']['accuracy']
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['white', '#1f77b4']  # Blue color scheme
        custom_cmap = LinearSegmentedColormap.from_list('adaptive_irt_cmap', colors, N=256)
        
        # Calculate percentages for coloring
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        percentage_matrix = conf_matrix / row_sums * 100
        
        # Plot
        im = plt.imshow(percentage_matrix, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=100)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Row Percentage (%)', rotation=270, labelpad=15)
        
        # Add text annotations
        thresh = 50.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                color = 'white' if percentage_matrix[i, j] > thresh else 'black'
                plt.text(j, i, f'{conf_matrix[i, j]}', 
                        ha='center', va='center', color=color, fontweight='bold')
        
        # Customize plot
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('Actual Category', fontsize=12)
        plt.title(f'Adaptive IRT Confusion Matrix (Accuracy: {accuracy:.3f})', 
                 fontsize=14, fontweight='bold')
        
        # Set ticks
        n_categories = conf_matrix.shape[0]
        plt.xticks(range(n_categories), [f'Cat {i}' for i in range(n_categories)])
        plt.yticks(range(n_categories), [f'Cat {i}' for i in range(n_categories)])
        
        plt.tight_layout()
        
        # Save
        output_path = self.dataset_path / "results" / "plots" / "adaptive_irt_confusion_matrix.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Adaptive IRT confusion matrix saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate Adaptive IRT')
    parser.add_argument('--dataset', default='./data/synthetic_OC',
                       help='Path to dataset')
    parser.add_argument('--max_questions', type=int, default=15,
                       help='Maximum questions per student')
    parser.add_argument('--n_students', type=int, default=100,
                       help='Number of students to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("🎯 ADAPTIVE IRT EVALUATION")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Max questions: {args.max_questions}")
    print(f"Students: {args.n_students}")
    print()
    
    # Initialize evaluator
    evaluator = AdaptiveIRTEvaluator(args.dataset, args.max_questions)
    
    # Run evaluation (use actual test data by default)
    results = evaluator.evaluate_all_students(use_test_data=True)
    
    # Save results
    evaluator.save_results()
    
    # Create comparison plots
    evaluator.create_comparison_plots()
    
    print("\n✅ Evaluation completed!")
    print(f"Check results directory: {evaluator.dataset_path}/results/")


if __name__ == "__main__":
    main()