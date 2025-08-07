#!/usr/bin/env python3
"""
Demo: Adaptive IRT with Real-time MML Parameter Estimation

Shows the key features of adaptive testing:
1. Dynamic theta estimation using MML
2. Adaptive question selection based on information maximization  
3. Real-time parameter updates
4. Convergence monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_irt_test import AdaptiveGPCMTest, load_synthetic_data
from pathlib import Path

def demo_adaptive_features():
    """Demonstrate key adaptive testing features"""
    
    print("🎯 ADAPTIVE IRT DEMO: Real-time MML Parameter Estimation")
    print("=" * 65)
    
    # Load data
    dataset_path = "./data/synthetic_OC"
    data = load_synthetic_data(dataset_path)
    
    # Select a student with moderate ability
    student_id = 50  # Pick one in the middle range
    true_theta = data['true_theta'][student_id]
    true_alpha = data['true_alpha']
    true_beta = data['true_beta']
    
    print(f"👤 Testing Student {student_id}")
    print(f"   True ability (θ): {true_theta:.3f}")
    print(f"   Available questions: {len(true_alpha)}")
    print(f"   Response categories: {data['n_cats']} (0,1,2,3)")
    print()
    
    # Initialize adaptive test
    adaptive_test = AdaptiveGPCMTest(len(true_alpha), data['n_cats'])
    
    print("🔄 ADAPTIVE TESTING PROCESS")
    print("-" * 40)
    print("Step | Question | Response | θ_estimate | Information | Error")
    print("-" * 60)
    
    administered_questions = set()
    theta_estimates = []
    information_gains = []
    errors = []
    
    # Run adaptive test step by step
    for step in range(12):
        # 1. SELECT NEXT QUESTION (Information Maximization)
        next_q = adaptive_test.select_next_question(administered_questions)
        if next_q is None:
            break
            
        administered_questions.add(next_q)
        
        # 2. COMPUTE EXPECTED INFORMATION
        info_gain = adaptive_test.item_information(adaptive_test.current_theta, next_q)
        information_gains.append(info_gain)
        
        # 3. GENERATE TRUE RESPONSE (Simulate student)
        true_probs = adaptive_test.gpcm_prob(true_theta, true_alpha[next_q], true_beta[next_q])
        true_response = np.random.choice(data['n_cats'], p=true_probs)
        
        # 4. UPDATE ESTIMATES (MML Optimization)
        adaptive_test.administer_question(next_q, true_response)
        
        # 5. TRACK PROGRESS
        theta_estimates.append(adaptive_test.current_theta)
        error = abs(adaptive_test.current_theta - true_theta)
        errors.append(error)
        
        print(f"{step+1:4d} | Q{next_q:7d} | {true_response:8d} | "
              f"{adaptive_test.current_theta:+9.3f} | {info_gain:11.3f} | {error:.3f}")
    
    print()
    print("📊 ADAPTIVE TESTING RESULTS")
    print("-" * 30)
    print(f"Final θ estimate: {adaptive_test.current_theta:.3f} (±{adaptive_test.theta_se:.3f})")
    print(f"True θ value: {true_theta:.3f}")
    print(f"Final error: {errors[-1]:.3f}")
    print(f"Initial error: {abs(0.0 - true_theta):.3f} (started at θ=0)")
    print(f"Error reduction: {abs(0.0 - true_theta) - errors[-1]:.3f}")
    print(f"Average information gain: {np.mean(information_gains):.3f}")
    
    # Create visualization
    create_adaptive_demo_plot(theta_estimates, errors, information_gains, 
                            true_theta, student_id)
    
    print(f"\n💾 Saved visualization to: data/plots/synthetic_OC/adaptive_demo_student_{student_id}.png")
    
    return {
        'theta_estimates': theta_estimates,
        'errors': errors, 
        'information_gains': information_gains,
        'true_theta': true_theta,
        'final_error': errors[-1]
    }

def create_adaptive_demo_plot(theta_estimates, errors, information_gains, 
                            true_theta, student_id):
    """Create comprehensive visualization of adaptive testing process"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    steps = range(1, len(theta_estimates) + 1)
    
    # 1. Theta Convergence
    ax1.plot(steps, theta_estimates, 'b-o', markersize=6, linewidth=2, label='θ estimates')
    ax1.axhline(y=true_theta, color='red', linestyle='--', linewidth=2, 
                label=f'True θ = {true_theta:.3f}')
    ax1.set_xlabel('Question Number')
    ax1.set_ylabel('Ability Estimate (θ)')
    ax1.set_title('Real-time Theta Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error Reduction
    ax2.plot(steps, errors, 'r-s', markersize=6, linewidth=2, color='orange')
    ax2.set_xlabel('Question Number')
    ax2.set_ylabel('Absolute Error |θ_est - θ_true|')
    ax2.set_title('Estimation Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Information Gain
    ax3.bar(steps, information_gains, alpha=0.7, color='green', width=0.6)
    ax3.set_xlabel('Question Number')
    ax3.set_ylabel('Fisher Information')
    ax3.set_title('Information Gain per Question')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative Precision
    cumulative_info = np.cumsum(information_gains)
    precision = 1.0 / np.sqrt(cumulative_info)  # Standard error approximation
    
    ax4.plot(steps, precision, 'purple', marker='d', markersize=6, linewidth=2)
    ax4.set_xlabel('Question Number')
    ax4.set_ylabel('Standard Error (θ)')
    ax4.set_title('Measurement Precision')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Adaptive IRT Demo: Student {student_id} (True θ = {true_theta:.3f})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("./data/plots/synthetic_OC")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'adaptive_demo_student_{student_id}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def compare_adaptive_vs_random():
    """Compare adaptive vs random question selection"""
    
    print("\n🎲 COMPARISON: Adaptive vs Random Question Selection")
    print("=" * 55)
    
    dataset_path = "./data/synthetic_OC"
    data = load_synthetic_data(dataset_path)
    
    student_id = 25
    true_theta = data['true_theta'][student_id]
    true_alpha = data['true_alpha']
    true_beta = data['true_beta']
    
    results = {'adaptive': [], 'random': []}
    
    # Run multiple trials
    n_trials = 5
    max_questions = 10
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}:")
        
        # Adaptive selection
        adaptive_test = AdaptiveGPCMTest(len(true_alpha), data['n_cats'])
        administered = set()
        
        for _ in range(max_questions):
            next_q = adaptive_test.select_next_question(administered)
            if next_q is None:
                break
            administered.add(next_q)
            
            true_probs = adaptive_test.gpcm_prob(true_theta, true_alpha[next_q], true_beta[next_q])
            response = np.random.choice(data['n_cats'], p=true_probs)
            adaptive_test.administer_question(next_q, response)
        
        adaptive_error = abs(adaptive_test.current_theta - true_theta)
        results['adaptive'].append(adaptive_error)
        
        # Random selection
        random_test = AdaptiveGPCMTest(len(true_alpha), data['n_cats'])
        questions = np.random.choice(len(true_alpha), max_questions, replace=False)
        
        for q in questions:
            true_probs = random_test.gpcm_prob(true_theta, true_alpha[q], true_beta[q])
            response = np.random.choice(data['n_cats'], p=true_probs)
            random_test.administer_question(q, response)
        
        random_error = abs(random_test.current_theta - true_theta)
        results['random'].append(random_error)
        
        print(f"  Adaptive error: {adaptive_error:.3f}")
        print(f"  Random error:   {random_error:.3f}")
        print(f"  Improvement:    {random_error - adaptive_error:+.3f}")
    
    print(f"\n📈 SUMMARY (True θ = {true_theta:.3f}):")
    print(f"   Adaptive - Mean error: {np.mean(results['adaptive']):.3f} (±{np.std(results['adaptive']):.3f})")
    print(f"   Random   - Mean error: {np.mean(results['random']):.3f} (±{np.std(results['random']):.3f})")
    print(f"   Average improvement: {np.mean(results['random']) - np.mean(results['adaptive']):.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demo 1: Show adaptive testing features
    demo_adaptive_features()
    
    # Demo 2: Compare adaptive vs random selection
    compare_adaptive_vs_random()
    
    print("\n✅ Demo completed! Check the plots directory for visualizations.")