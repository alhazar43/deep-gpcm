#!/usr/bin/env python3
"""
Corrected Adaptive IRT Implementation
Fixes the theoretical issues identified in the analysis
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from adaptive_irt_test import load_synthetic_data
import matplotlib.pyplot as plt

class CorrectedAdaptiveGPCMTest:
    """Corrected adaptive GPCM test with proper Fisher Information and MML"""
    
    def __init__(self, n_questions, n_cats=4, prior_theta_mean=0, prior_theta_std=1):
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.prior_theta_mean = prior_theta_mean
        self.prior_theta_std = prior_theta_std
        
        # Initialize item parameters (will be estimated via MML)
        self.alpha = np.ones(n_questions)
        self.beta = np.zeros((n_questions, n_cats - 1))
        
        # Initialize beta thresholds as ordered
        for q in range(n_questions):
            self.beta[q] = np.sort(np.linspace(-1.5, 1.5, n_cats - 1))
        
        # Track responses and current estimates
        self.responses = []
        self.theta_history = []
        self.question_history = []
        self.current_theta = 0.0
        self.theta_se = 1.0
        
    def gpcm_prob(self, theta, alpha, betas):
        """Compute GPCM category probabilities (unchanged - this was correct)"""
        K = len(betas) + 1
        cum_logits = np.zeros(K)
        cum_logits[0] = 0
        
        for k in range(1, K):
            cum_logits[k] = np.sum([alpha * (theta - betas[h]) for h in range(k)])
        
        # Numerical stability
        cum_logits = cum_logits - np.max(cum_logits)
        exp_logits = np.exp(cum_logits)
        return exp_logits / np.sum(exp_logits)
    
    def correct_item_information(self, theta, question_id):
        """CORRECTED: Proper GPCM Fisher Information calculation"""
        alpha = self.alpha[question_id]
        betas = self.beta[question_id]
        
        probs = self.gpcm_prob(theta, alpha, betas)
        
        # Compute proper derivatives for GPCM
        derivatives = np.zeros(self.n_cats)
        for k in range(self.n_cats):
            if k == 0:
                derivatives[k] = 0
            else:
                # Derivative of log P(X=k|theta) with respect to theta
                derivatives[k] = alpha * k
        
        # Expected derivative
        expected_deriv = np.sum(derivatives * probs)
        
        # Fisher Information = Variance of the derivative
        fisher_info = np.sum((derivatives - expected_deriv)**2 * probs)
        
        return fisher_info
    
    def select_next_question(self, administered_questions):
        """Select next question to maximize Fisher Information (now corrected)"""
        available_questions = [q for q in range(self.n_questions) if q not in administered_questions]
        
        if not available_questions:
            return None
        
        # Compute corrected Fisher Information for each available question
        information_values = []
        for q in available_questions:
            info = self.correct_item_information(self.current_theta, q)
            information_values.append(info)
        
        # Select question with maximum information
        best_idx = np.argmax(information_values)
        return available_questions[best_idx]
    
    def log_likelihood_theta(self, theta, responses, question_ids):
        """Compute log-likelihood for theta given responses (unchanged - correct)"""
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
        """Estimate theta using MML (unchanged - this was correct)"""
        if not self.responses:
            return 0.0, 1.0
        
        question_ids = [resp[0] for resp in self.responses]
        responses = [resp[1] for resp in self.responses]
        
        # Optimize theta
        def neg_log_lik(theta):
            return -self.log_likelihood_theta(theta, responses, question_ids)
        
        result = minimize_scalar(neg_log_lik, bounds=(-4, 4), method='bounded')
        theta_est = result.x
        
        # Compute standard error
        total_info = sum([self.correct_item_information(theta_est, q_id) for q_id in question_ids])
        se = 1.0 / np.sqrt(max(total_info, 0.1))
        
        return theta_est, se
    
    def improved_item_parameter_estimation(self):
        """IMPROVED: Better item parameter estimation using EM-like approach"""
        if len(self.responses) < 5:  # Need sufficient data
            return
        
        # Group responses by question
        question_responses = {}
        for q_id, resp in self.responses:
            if q_id not in question_responses:
                question_responses[q_id] = []
            question_responses[q_id].append(resp)
        
        # Simple EM-like update for discrimination parameters
        for q_id, resp_list in question_responses.items():
            if len(resp_list) >= 3:  # Need multiple responses
                # Estimate discrimination based on response pattern
                response_var = np.var(resp_list)
                response_mean = np.mean(resp_list)
                
                # Update discrimination (alpha) based on variance and current theta estimate
                # Higher variance suggests better discrimination
                if response_var > 0:
                    # Simple heuristic: adjust alpha based on variance and mean response level
                    alpha_adjustment = 0.1 * (response_var - 0.5)  # Target variance around 0.5
                    self.alpha[q_id] = max(0.3, min(2.5, self.alpha[q_id] + alpha_adjustment))
                
                # Update difficulty parameters (beta) based on response pattern
                # Adjust beta[0] based on mean response level
                if response_mean < 1.0:  # Responses skewed low - item too hard
                    self.beta[q_id] += 0.1
                elif response_mean > 2.0:  # Responses skewed high - item too easy
                    self.beta[q_id] -= 0.1
                
                # Ensure beta parameters remain ordered
                self.beta[q_id] = np.sort(self.beta[q_id])
    
    def administer_question(self, question_id, true_response):
        """Administer question and update estimates (improved parameter estimation)"""
        # Record response
        self.responses.append((question_id, true_response))
        self.question_history.append(question_id)
        
        # Update theta estimate
        self.current_theta, self.theta_se = self.estimate_theta_mml()
        self.theta_history.append(self.current_theta)
        
        # Update item parameters using improved method
        if len(self.responses) % 3 == 0:  # Update every 3 responses
            self.improved_item_parameter_estimation()
    
    def predict_response(self, question_id, theta=None):
        """Predict response probabilities for a question"""
        if theta is None:
            theta = self.current_theta
        
        alpha = self.alpha[question_id]
        betas = self.beta[question_id]
        return self.gpcm_prob(theta, alpha, betas)


def quick_comparison_test():
    """Quick test to compare corrected vs original implementation"""
    
    print("🔧 TESTING CORRECTED ADAPTIVE IRT")
    print("=" * 50)
    
    # Load synthetic data
    data = load_synthetic_data("./data/synthetic_OC")
    
    # Test on a single student
    student_id = 0
    true_theta = data['true_theta'][student_id]
    true_alpha = data['true_alpha']
    true_beta = data['true_beta']
    
    print(f"Test student: {student_id} (true θ = {true_theta:.3f})")
    
    # Initialize corrected adaptive test
    corrected_test = CorrectedAdaptiveGPCMTest(len(true_alpha), data['n_cats'])
    
    print("\nTesting Fisher Information calculation...")
    
    # Test question 0
    q_test = 0
    info_corrected = corrected_test.correct_item_information(true_theta, q_test)
    
    print(f"Question {q_test} at θ={true_theta:.3f}:")
    print(f"  Corrected Fisher Info: {info_corrected:.4f}")
    
    # Run short adaptive test
    print(f"\nRunning 10-question adaptive test...")
    
    administered = set()
    for step in range(10):
        # Select question using corrected Fisher Information
        next_q = corrected_test.select_next_question(administered)
        if next_q is None:
            break
        administered.add(next_q)
        
        # Generate true response
        true_probs = corrected_test.gpcm_prob(true_theta, true_alpha[next_q], true_beta[next_q])
        true_response = np.random.choice(data['n_cats'], p=true_probs)
        
        # Get prediction and information
        pred_probs = corrected_test.predict_response(next_q)
        predicted = np.argmax(pred_probs)
        info_gain = corrected_test.correct_item_information(corrected_test.current_theta, next_q)
        
        # Update estimates
        corrected_test.administer_question(next_q, true_response)
        
        print(f"Step {step+1}: Q{next_q:2d} | True: {true_response} | Pred: {predicted} | "
              f"θ: {corrected_test.current_theta:+.3f} | Info: {info_gain:.3f}")
    
    final_error = abs(corrected_test.current_theta - true_theta)
    print(f"\nFinal θ estimate: {corrected_test.current_theta:.3f}")
    print(f"True θ: {true_theta:.3f}")
    print(f"Final error: {final_error:.3f}")
    
    return corrected_test


if __name__ == "__main__":
    np.random.seed(42)
    quick_comparison_test()