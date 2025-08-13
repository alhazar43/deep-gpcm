#!/usr/bin/env python3
"""
Adaptive IRT Testing with Dynamic MML Parameter Estimation

Implements real-time adaptive testing using GPCM model with:
- Maximum Marginal Likelihood (MML) optimization
- Dynamic theta estimation
- Adaptive question selection
- Real-time prediction updates
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import the IRT-optimized loss function
from adaptive_irt_losses import IRTOptimizedLoss, AdaptiveIRTParameterEstimator

class AdaptiveGPCMTest:
    def __init__(self, n_questions, n_cats=4, prior_theta_mean=0, prior_theta_std=1, 
                 use_irt_optimized_loss=True):
        """
        Initialize adaptive GPCM testing system with IRT-optimized loss
        
        Args:
            n_questions: Number of available questions
            n_cats: Number of response categories (4 = 0,1,2,3)
            prior_theta_mean: Prior mean for ability
            prior_theta_std: Prior std for ability
            use_irt_optimized_loss: Use IRT-optimized hybrid loss function
        """
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.prior_theta_mean = prior_theta_mean
        self.prior_theta_std = prior_theta_std
        self.use_irt_optimized_loss = use_irt_optimized_loss
        
        # Initialize item parameters (will be estimated via MML)
        self.alpha = np.ones(n_questions)  # Start with neutral discrimination
        self.beta = np.zeros((n_questions, n_cats - 1))  # Start with neutral difficulties
        
        # Initialize beta thresholds as ordered
        for q in range(n_questions):
            self.beta[q] = np.sort(np.linspace(-1.5, 1.5, n_cats - 1))
        
        # Track responses and current estimates
        self.responses = []  # [(question_id, response)]
        self.theta_history = []  # Track theta estimates over time
        self.question_history = []  # Track questions administered
        
        # Current ability estimate
        self.current_theta = 0.0
        self.theta_se = 1.0  # Standard error of theta estimate
        
        # Initialize IRT-optimized parameter estimator if requested
        if self.use_irt_optimized_loss:
            self.irt_estimator = AdaptiveIRTParameterEstimator(
                n_cats=n_cats, 
                loss_config={'ce_weight': 0.7, 'qwk_weight': 0.3}
            )
            print("🎯 Using IRT-Optimized Loss (CE:QWK = 0.7:0.3)")
        
    def gpcm_prob(self, theta, alpha, betas):
        """Compute GPCM category probabilities"""
        K = len(betas) + 1
        cum_logits = np.zeros(K)
        cum_logits[0] = 0
        
        for k in range(1, K):
            cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        # Numerical stability
        cum_logits = cum_logits - np.max(cum_logits)
        exp_logits = np.exp(cum_logits)
        return exp_logits / np.sum(exp_logits)
    
    def item_information(self, theta, question_id):
        """Compute Fisher information for a specific item"""
        alpha = self.alpha[question_id]
        betas = self.beta[question_id]
        
        probs = self.gpcm_prob(theta, alpha, betas)
        
        # Compute score function (derivative of log-likelihood)
        scores = np.zeros(self.n_cats)
        for k in range(self.n_cats):
            if k == 0:
                scores[k] = 0
            else:
                scores[k] = k * alpha
        
        # Expected score
        expected_score = np.sum(scores * probs)
        
        # Variance of score (Fisher information)
        score_variance = np.sum((scores - expected_score)**2 * probs)
        
        return alpha**2 * score_variance
    
    def select_next_question(self, administered_questions):
        """Select next question to maximize information"""
        available_questions = [q for q in range(self.n_questions) if q not in administered_questions]
        
        if not available_questions:
            return None
        
        # Compute information for each available question
        information_values = []
        for q in available_questions:
            info = self.item_information(self.current_theta, q)
            information_values.append(info)
        
        # Select question with maximum information
        best_idx = np.argmax(information_values)
        return available_questions[best_idx]
    
    def log_likelihood_theta(self, theta, responses, question_ids):
        """Compute log-likelihood for theta given responses"""
        log_lik = 0
        
        # Prior contribution
        log_lik += norm.logpdf(theta, self.prior_theta_mean, self.prior_theta_std)
        
        # Response contributions
        for (q_id, response) in zip(question_ids, responses):
            alpha = self.alpha[q_id]
            betas = self.beta[q_id]
            probs = self.gpcm_prob(theta, alpha, betas)
            
            # Add small epsilon to avoid log(0)
            prob = max(probs[response], 1e-10)
            log_lik += np.log(prob)
        
        return log_lik
    
    def estimate_theta_mml(self):
        """Estimate theta using Maximum Marginal Likelihood with optional IRT-optimized loss"""
        if not self.responses:
            return 0.0, 1.0
        
        question_ids = [resp[0] for resp in self.responses]
        responses = [resp[1] for resp in self.responses]
        
        # Use IRT-optimized estimator if available
        if self.use_irt_optimized_loss and hasattr(self, 'irt_estimator'):
            try:
                theta_est, se_est = self.irt_estimator.estimate_theta_with_optimized_loss(
                    responses, question_ids, self.alpha, self.beta,
                    self.prior_theta_mean, self.prior_theta_std
                )
                return theta_est, se_est
            except Exception as e:
                print(f"⚠️ IRT-optimized estimation failed: {e}, falling back to standard MML")
        
        # Fallback to standard MML estimation
        def neg_log_lik(theta):
            return -self.log_likelihood_theta(theta, responses, question_ids)
        
        result = minimize_scalar(neg_log_lik, bounds=(-4, 4), method='bounded')
        theta_est = result.x
        
        # Compute standard error (approximate)
        # SE ≈ 1/sqrt(Fisher Information)
        total_info = sum([self.item_information(theta_est, q_id) for q_id in question_ids])
        se = 1.0 / np.sqrt(max(total_info, 0.1))
        
        return theta_est, se
    
    def update_item_parameters_mml(self):
        """Update item parameters using MML (simplified version)"""
        if len(self.responses) < 3:  # Need minimum responses
            return
        
        # In practice, this would involve EM algorithm or more sophisticated optimization
        # For this demo, we'll do a simplified update
        question_ids = [resp[0] for resp in self.responses]
        responses = [resp[1] for resp in self.responses]
        
        # Group responses by question
        question_responses = {}
        for q_id, resp in zip(question_ids, responses):
            if q_id not in question_responses:
                question_responses[q_id] = []
            question_responses[q_id].append(resp)
        
        # Simple update: adjust discrimination based on response variance
        for q_id, resp_list in question_responses.items():
            if len(resp_list) >= 2:
                response_var = np.var(resp_list)
                # Higher variance suggests better discrimination
                self.alpha[q_id] = max(0.5, min(2.0, 1.0 + response_var * 0.5))
    
    def predict_response(self, question_id, theta=None):
        """Predict response probabilities for a question"""
        if theta is None:
            theta = self.current_theta
        
        alpha = self.alpha[question_id]
        betas = self.beta[question_id]
        return self.gpcm_prob(theta, alpha, betas)
    
    def administer_question(self, question_id, true_response):
        """Administer a question and update estimates"""
        # Record response
        self.responses.append((question_id, true_response))
        self.question_history.append(question_id)
        
        # Update theta estimate
        self.current_theta, self.theta_se = self.estimate_theta_mml()
        self.theta_history.append(self.current_theta)
        
        # Update item parameters (periodically)
        if len(self.responses) % 5 == 0:  # Update every 5 responses
            self.update_item_parameters_mml()
    
    def run_adaptive_test(self, true_theta, true_alpha, true_beta, max_questions=20):
        """
        Run full adaptive test simulation
        
        Args:
            true_theta: True ability of the examinee
            true_alpha: True discrimination parameters
            true_beta: True difficulty parameters
            max_questions: Maximum number of questions to administer
        """
        administered_questions = set()
        predictions = []
        true_responses = []
        
        print(f"Starting adaptive test for examinee with true theta = {true_theta:.3f}")
        print("=" * 60)
        
        for step in range(max_questions):
            # Select next question
            next_q = self.select_next_question(administered_questions)
            if next_q is None:
                print("No more questions available")
                break
            
            administered_questions.add(next_q)
            
            # Generate true response using true parameters
            true_probs = self.gpcm_prob(true_theta, true_alpha[next_q], true_beta[next_q])
            true_response = np.random.choice(self.n_cats, p=true_probs)
            true_responses.append(true_response)
            
            # Get prediction before seeing response
            pred_probs = self.predict_response(next_q)
            predicted_response = np.argmax(pred_probs)
            predictions.append(predicted_response)
            
            # Administer question (update estimates)
            self.administer_question(next_q, true_response)
            
            # Print progress
            print(f"Step {step+1:2d}: Q{next_q:2d} | "
                  f"True: {true_response} | Pred: {predicted_response} | "
                  f"θ_est: {self.current_theta:+.3f} (SE: {self.theta_se:.3f}) | "
                  f"Error: {abs(self.current_theta - true_theta):.3f}")
        
        return predictions, true_responses
    
    def plot_theta_convergence(self, true_theta):
        """Plot theta estimate convergence"""
        if not self.theta_history:
            return
        
        plt.figure(figsize=(10, 6))
        steps = range(1, len(self.theta_history) + 1)
        
        plt.plot(steps, self.theta_history, 'b-o', markersize=4, label='θ estimates')
        plt.axhline(y=true_theta, color='r', linestyle='--', label=f'True θ = {true_theta:.3f}')
        
        # Add confidence bands (approximate)
        theta_array = np.array(self.theta_history)
        se_estimates = [1.0 / np.sqrt(i * 0.5) for i in steps]  # Rough SE approximation
        
        plt.fill_between(steps, 
                        theta_array - np.array(se_estimates), 
                        theta_array + np.array(se_estimates), 
                        alpha=0.3, color='blue', label='±1 SE')
        
        plt.xlabel('Question Number')
        plt.ylabel('Ability Estimate (θ)')
        plt.title('Adaptive Testing: Theta Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_confusion_matrix(self, predictions, true_responses):
        """Plot confusion matrix of predictions vs true responses"""
        if not predictions or not true_responses:
            return
        
        cm = confusion_matrix(true_responses, predictions, labels=range(self.n_cats))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Cat {i}' for i in range(self.n_cats)],
                   yticklabels=[f'Cat {i}' for i in range(self.n_cats)])
        
        plt.title('Adaptive IRT: Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.tight_layout()
        return plt


def load_synthetic_data(dataset_path):
    """Load synthetic dataset and true parameters"""
    data_dir = Path(dataset_path)
    
    # Load true IRT parameters
    param_file = data_dir / 'true_irt_parameters.json'
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    true_theta = np.array(params['student_abilities']['theta'])
    true_alpha = np.array(params['question_params']['discrimination']['alpha'])
    true_beta = np.array(params['question_params']['difficulties']['beta'])
    n_cats = params['model_info']['n_cats']
    
    # Load actual response data (for validation)
    train_file = data_dir / 'synthetic_oc_train.txt'
    if not train_file.exists():
        # Try alternative naming
        train_files = list(data_dir.glob('*train.txt'))
        if train_files:
            train_file = train_files[0]
    
    sequences = []
    if train_file.exists():
        with open(train_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            seq_len = int(lines[i].strip())
            questions = list(map(int, lines[i+1].strip().split(',')))
            responses = list(map(int, lines[i+2].strip().split(',')))
            
            sequences.append({
                'questions': questions,
                'responses': responses,
                'seq_len': seq_len
            })
            i += 3
    
    return {
        'true_theta': true_theta,
        'true_alpha': true_alpha, 
        'true_beta': true_beta,
        'n_cats': n_cats,
        'sequences': sequences
    }


def run_adaptive_test_demo(dataset_path, student_id=0, max_questions=15, save_plots=True):
    """Run adaptive testing demonstration"""
    
    print("Loading synthetic data...")
    data = load_synthetic_data(dataset_path)
    
    n_questions = len(data['true_alpha'])
    n_cats = data['n_cats']
    true_theta = data['true_theta'][student_id]
    true_alpha = data['true_alpha']
    true_beta = data['true_beta']
    
    print(f"Dataset info:")
    print(f"  Questions: {n_questions}")
    print(f"  Categories: {n_cats}")
    print(f"  Selected student {student_id} with true θ = {true_theta:.3f}")
    print()
    
    # Initialize adaptive testing system
    adaptive_test = AdaptiveGPCMTest(n_questions, n_cats)
    
    # Run adaptive test
    predictions, true_responses = adaptive_test.run_adaptive_test(
        true_theta, true_alpha, true_beta, max_questions
    )
    
    print("\n" + "=" * 60)
    print("ADAPTIVE TEST RESULTS")
    print("=" * 60)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_responses))
    print(f"Prediction Accuracy: {accuracy:.3f}")
    
    # Calculate final theta error
    final_theta_error = abs(adaptive_test.current_theta - true_theta)
    print(f"Final θ Error: {final_theta_error:.3f}")
    print(f"Final θ Estimate: {adaptive_test.current_theta:.3f} (±{adaptive_test.theta_se:.3f})")
    print(f"True θ: {true_theta:.3f}")
    
    # Classification report
    if len(set(true_responses)) > 1:  # Only if multiple classes present
        print(f"\nClassification Report:")
        try:
            labels = sorted(list(set(true_responses + predictions)))
            print(classification_report(true_responses, predictions, 
                                      labels=labels,
                                      target_names=[f'Cat {i}' for i in labels]))
        except:
            print(f"Categories used: {sorted(set(true_responses))}")
            print(f"Predictions made: {sorted(set(predictions))}")
    else:
        print(f"\nOnly one response category observed: {set(true_responses)}")
    
    if save_plots:
        # Plot theta convergence
        plt1 = adaptive_test.plot_theta_convergence(true_theta)
        output_dir = Path(dataset_path).parent / 'plots' / Path(dataset_path).name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt1.savefig(output_dir / f'adaptive_theta_convergence_student_{student_id}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved theta convergence plot to {output_dir}")
        plt1.close()
        
        # Plot confusion matrix
        plt2 = adaptive_test.plot_confusion_matrix(predictions, true_responses)
        plt2.savefig(output_dir / f'adaptive_confusion_matrix_student_{student_id}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_dir}")
        plt2.close()
    
    return {
        'adaptive_test': adaptive_test,
        'predictions': predictions,
        'true_responses': true_responses,
        'accuracy': accuracy,
        'theta_error': final_theta_error
    }


def compare_multiple_students(dataset_path, n_students=5, max_questions=15):
    """Compare adaptive testing across multiple students"""
    
    data = load_synthetic_data(dataset_path)
    results = []
    
    print(f"Running adaptive tests for {n_students} students...")
    print("=" * 80)
    
    for student_id in range(min(n_students, len(data['true_theta']))):
        print(f"\nStudent {student_id}:")
        print("-" * 40)
        
        result = run_adaptive_test_demo(dataset_path, student_id, max_questions, save_plots=False)
        results.append({
            'student_id': student_id,
            'true_theta': data['true_theta'][student_id],
            'estimated_theta': result['adaptive_test'].current_theta,
            'theta_error': result['theta_error'],
            'accuracy': result['accuracy'],
            'se': result['adaptive_test'].theta_se
        })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS STUDENTS")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(f"Mean θ Error: {df['theta_error'].mean():.3f} (±{df['theta_error'].std():.3f})")
    print(f"Mean Accuracy: {df['accuracy'].mean():.3f} (±{df['accuracy'].std():.3f})")
    print(f"θ Correlation: {np.corrcoef(df['true_theta'], df['estimated_theta'])[0,1]:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['true_theta'], df['estimated_theta'], alpha=0.7)
    plt.plot([-3, 3], [-3, 3], 'r--', label='Perfect Estimation')
    plt.xlabel('True θ')
    plt.ylabel('Estimated θ')
    plt.title('Theta Estimation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['true_theta'], df['accuracy'], alpha=0.7)
    plt.xlabel('True θ')
    plt.ylabel('Prediction Accuracy')
    plt.title('Accuracy vs Ability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_dir = Path(dataset_path).parent / 'plots' / Path(dataset_path).name
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'adaptive_comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison summary to {output_dir}")
    plt.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Adaptive IRT Testing with MML')
    parser.add_argument('--dataset', default='./data/synthetic_OC', 
                       help='Path to synthetic dataset')
    parser.add_argument('--student_id', type=int, default=0,
                       help='Student ID to test')
    parser.add_argument('--max_questions', type=int, default=15,
                       help='Maximum questions in adaptive test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare across multiple students')
    parser.add_argument('--n_students', type=int, default=5,
                       help='Number of students for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_multiple_students(args.dataset, args.n_students, args.max_questions)
    else:
        run_adaptive_test_demo(args.dataset, args.student_id, args.max_questions)


if __name__ == "__main__":
    main()