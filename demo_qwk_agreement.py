#!/usr/bin/env python3
"""
QWK Probability Agreement Demonstration

This script demonstrates the QWK-weighted probability agreement measure
for ordinal categorical data, showing how it works and comparing it with
other ordinal agreement measures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our new utility
from utils.ordinal_agreement import (
    qwk_probability_agreement, 
    triangular_probability_agreement,
    earth_movers_distance_agreement,
    visualize_weight_functions,
    generate_agreement_heatmap,
    compare_agreement_measures
)

def demonstrate_agreement_scores():
    """Demonstrate QWK agreement scores with various prediction scenarios."""
    print("QWK Probability Agreement Demonstration")
    print("=" * 50)
    
    # Test scenarios for a 4-category ordinal scale (0=Below Basic, 1=Basic, 2=Proficient, 3=Advanced)
    scenarios = [
        # (true_category, predicted_probabilities, description)
        (0, [1.0, 0.0, 0.0, 0.0], "Perfect prediction: Below Basic"),
        (0, [0.0, 1.0, 0.0, 0.0], "Off by 1: Predicted Basic"),
        (0, [0.0, 0.0, 1.0, 0.0], "Off by 2: Predicted Proficient"), 
        (0, [0.0, 0.0, 0.0, 1.0], "Off by 3: Predicted Advanced"),
        (0, [0.25, 0.25, 0.25, 0.25], "Uniform uncertainty"),
        (1, [0.1, 0.7, 0.2, 0.0], "Confident correct with adjacent uncertainty"),
        (1, [0.0, 0.5, 0.5, 0.0], "Split between correct and adjacent"),
        (2, [0.0, 0.1, 0.8, 0.1], "Highly confident correct"),
        (2, [0.2, 0.2, 0.4, 0.2], "Weak correct with spread uncertainty"),
        (3, [0.0, 0.0, 0.3, 0.7], "Strong correct prediction"),
    ]
    
    print(f"{'Scenario':<45} {'QWK':<6} {'Tri':<6} {'EMD':<6}")
    print("-" * 70)
    
    for true_cat, pred_probs, description in scenarios:
        qwk_score = qwk_probability_agreement(true_cat, pred_probs, K=4)
        tri_score = triangular_probability_agreement(true_cat, pred_probs, K=4)
        emd_score = earth_movers_distance_agreement(true_cat, pred_probs, K=4)
        
        print(f"{description:<45} {qwk_score:.3f} {tri_score:.3f} {emd_score:.3f}")

def create_comparison_visualization():
    """Create visualization comparing QWK agreement with other measures."""
    # Generate synthetic prediction scenarios
    np.random.seed(42)
    n_samples = 100
    
    # Create diverse prediction scenarios
    scenarios = []
    true_categories = []
    
    for _ in range(n_samples):
        true_cat = np.random.randint(0, 4)
        true_categories.append(true_cat)
        
        # Generate realistic prediction probabilities
        if np.random.random() < 0.4:  # 40% good predictions
            probs = np.random.dirichlet([0.1, 0.1, 0.1, 0.1])
            probs[true_cat] *= 5  # Boost correct category
        elif np.random.random() < 0.3:  # 30% adjacent errors
            probs = np.random.dirichlet([0.1, 0.1, 0.1, 0.1])
            if true_cat > 0:
                probs[true_cat - 1] *= 3
            if true_cat < 3:
                probs[true_cat + 1] *= 3
        else:  # 30% random/poor predictions
            probs = np.random.dirichlet([1, 1, 1, 1])
        
        # Normalize
        probs = probs / np.sum(probs)
        scenarios.append(probs)
    
    # Compare agreement measures
    results = compare_agreement_measures(true_categories, scenarios, K=4)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plots comparing measures
    axes[0, 0].scatter(results['qwk_agreement'], results['triangular_agreement'], alpha=0.6)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('QWK Agreement')
    axes[0, 0].set_ylabel('Triangular Agreement')
    axes[0, 0].set_title('QWK vs Triangular Agreement')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(results['qwk_agreement'], results['emd_agreement'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].set_xlabel('QWK Agreement')
    axes[0, 1].set_ylabel('EMD Agreement')
    axes[0, 1].set_title('QWK vs EMD Agreement')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution plots
    axes[1, 0].hist(results['qwk_agreement'], bins=20, alpha=0.7, label='QWK', color='blue')
    axes[1, 0].hist(results['triangular_agreement'], bins=20, alpha=0.7, label='Triangular', color='orange')
    axes[1, 0].set_xlabel('Agreement Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Agreement Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation analysis
    correlations = np.corrcoef([results['qwk_agreement'], results['triangular_agreement'], results['emd_agreement']])
    im = axes[1, 1].imshow(correlations, cmap='RdBu', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{correlations[i, j]:.3f}', 
                           ha='center', va='center', fontweight='bold')
    
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['QWK', 'Triangular', 'EMD'])
    axes[1, 1].set_yticklabels(['QWK', 'Triangular', 'EMD'])
    axes[1, 1].set_title('Agreement Measure Correlations')
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.suptitle('Comparison of Ordinal Agreement Measures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_heatmap_example():
    """Create an example heatmap similar to what would be generated in IRT analysis."""
    # Simulate temporal predictions for students
    np.random.seed(123)
    n_students = 15
    n_questions = 25
    
    # Create realistic student ability levels and question difficulties
    student_abilities = np.random.normal(0, 1, n_students)
    question_difficulties = np.random.normal(0, 0.5, n_questions)
    
    # Generate agreement matrix
    agreement_matrix = np.zeros((n_students, n_questions))
    
    for i, ability in enumerate(student_abilities):
        for j, difficulty in enumerate(question_difficulties):
            # Generate realistic ordinal response probabilities using simple IRT-like model
            logits = np.array([
                ability - difficulty - 1.5,  # P(category 0)
                ability - difficulty - 0.5,  # P(category 1) 
                ability - difficulty + 0.5,  # P(category 2)
                ability - difficulty + 1.5   # P(category 3)
            ])
            
            # Convert to probabilities (simplified GPCM-like)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)
            
            # Generate true response based on ability and difficulty
            true_response_prob = 1 / (1 + np.exp(-(ability - difficulty)))
            if true_response_prob > 0.75:
                true_response = 3
            elif true_response_prob > 0.5:
                true_response = 2
            elif true_response_prob > 0.25:
                true_response = 1
            else:
                true_response = 0
            
            # Calculate QWK agreement
            agreement_matrix[i, j] = qwk_probability_agreement(true_response, probs, K=4)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Configure plot
    ax.set_xlabel('Question/Time Steps')
    ax.set_ylabel('Students (ordered by ability)')
    ax.set_title('QWK Probability Agreement Heatmap\n(Simulated Deep-GPCM Predictions)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('QWK Agreement Score [0,1]', rotation=270, labelpad=15)
    
    # Set reasonable ticks
    ax.set_xticks(np.arange(0, n_questions, 5))
    ax.set_xticklabels(np.arange(1, n_questions+1, 5))
    
    student_labels = [f'S{i+1} ({student_abilities[i]:.1f})' for i in range(n_students)]
    ax.set_yticks(np.arange(0, n_students, 3))
    ax.set_yticklabels([student_labels[i] for i in range(0, n_students, 3)])
    
    plt.tight_layout()
    return fig

def main():
    """Main demonstration function."""
    print("Starting QWK Probability Agreement Demonstration...")
    print()
    
    # Create output directory
    output_dir = Path('demo_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Demonstrate agreement scores
    demonstrate_agreement_scores()
    print()
    
    # 2. Generate weight function visualizations
    print("Generating weight function visualizations...")
    visualize_weight_functions(K=4, save_path=output_dir / 'weight_functions.png')
    
    # 3. Generate agreement matrix heatmap
    print("Generating agreement matrix heatmap...")
    generate_agreement_heatmap(K=4, save_path=output_dir / 'agreement_matrix.png')
    
    # 4. Create comparison visualization
    print("Creating agreement measure comparison...")
    comparison_fig = create_comparison_visualization()
    comparison_fig.savefig(output_dir / 'agreement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(comparison_fig)
    
    # 5. Create example heatmap
    print("Creating example temporal heatmap...")
    heatmap_fig = create_heatmap_example()
    heatmap_fig.savefig(output_dir / 'example_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(heatmap_fig)
    
    print()
    print("Demonstration complete! Generated files:")
    for file_path in output_dir.glob('*.png'):
        print(f"  - {file_path}")
    
    print()
    print("Integration Summary:")
    print("- QWK agreement function: utils/ordinal_agreement.py")
    print("- IRT analysis integration: analysis/irt_analysis.py (qwk_agreement visualization)")
    print("- Documentation: docs/QWK_Agreement_Guide.md")
    print("- Usage: python irt_analysis.py --analysis_types temporal")

if __name__ == "__main__":
    main()