#!/usr/bin/env python3
"""
Ordinal Agreement Measures for Educational Assessment

This module provides functions to compute agreement between predicted probability
distributions and true ordinal categories, respecting ordinal structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, List

def qwk_probability_agreement(y_true: int, y_pred_probs: np.ndarray, K: int = 4) -> float:
    """
    Compute QWK-weighted probability agreement between true ordinal category and predicted probabilities.
    
    This measures "degree of closeness" between probability distributions and true ordinal categories,
    respecting ordinal structure where being wrong by 1 category is better than wrong by 2.
    
    Mathematical formulation:
        Agreement = Σ_k [QWK_weight(y_true, k) × p_k]
        where QWK_weight(i,j) = 1 - (i-j)²/(K-1)²
    
    Args:
        y_true: True ordinal category (0 to K-1)
        y_pred_probs: Predicted probabilities array of length K
        K: Number of ordinal categories
    
    Returns:
        Agreement score in [0,1] where higher = better agreement
        
    Examples:
        >>> # Perfect prediction
        >>> qwk_probability_agreement(0, [1.0, 0.0, 0.0, 0.0], K=4)
        1.0
        
        >>> # Adjacent category error
        >>> qwk_probability_agreement(0, [0.0, 1.0, 0.0, 0.0], K=4)
        0.8888888888888888
        
        >>> # Uniform prediction
        >>> qwk_probability_agreement(0, [0.25, 0.25, 0.25, 0.25], K=4)
        0.6111111111111112
        
        >>> # Worst prediction
        >>> qwk_probability_agreement(0, [0.0, 0.0, 0.0, 1.0], K=4)
        0.0
    """
    if not (0 <= y_true < K) or len(y_pred_probs) != K:
        return 0.0
        
    agreement = 0.0
    max_distance_squared = (K - 1) ** 2
    
    for k in range(K):
        # QWK-style weight: 1 - (distance²) / (max_distance²)
        distance_squared = (y_true - k) ** 2
        weight = 1.0 - distance_squared / max_distance_squared
        agreement += weight * y_pred_probs[k]
    
    return agreement

def triangular_probability_agreement(y_true: int, y_pred_probs: np.ndarray, K: int = 4) -> float:
    """
    Compute triangular-weighted probability agreement (linear decay).
    
    Mathematical formulation:
        Agreement = Σ_k [triangular_weight(y_true, k) × p_k]
        where triangular_weight(i,j) = max(0, 1 - |i-j|/(K-1))
    
    Args:
        y_true: True ordinal category (0 to K-1)
        y_pred_probs: Predicted probabilities array of length K
        K: Number of ordinal categories
    
    Returns:
        Agreement score in [0,1] where higher = better agreement
    """
    if not (0 <= y_true < K) or len(y_pred_probs) != K:
        return 0.0
        
    agreement = 0.0
    max_distance = K - 1
    
    for k in range(K):
        # Triangular weight: max(0, 1 - distance/max_distance)
        distance = abs(y_true - k)
        weight = max(0.0, 1.0 - distance / max_distance)
        agreement += weight * y_pred_probs[k]
    
    return agreement

def earth_movers_distance_agreement(y_true: int, y_pred_probs: np.ndarray, K: int = 4) -> float:
    """
    Compute 1 - normalized Earth Mover's Distance as agreement measure.
    
    Args:
        y_true: True ordinal category (0 to K-1)
        y_pred_probs: Predicted probabilities array of length K
        K: Number of ordinal categories
    
    Returns:
        Agreement score in [0,1] where higher = better agreement
    """
    if not (0 <= y_true < K) or len(y_pred_probs) != K:
        return 0.0
    
    # Create one-hot true distribution
    y_true_dist = np.zeros(K)
    y_true_dist[y_true] = 1.0
    
    # Compute cumulative distributions
    cum_true = np.cumsum(y_true_dist)
    cum_pred = np.cumsum(y_pred_probs)
    
    # Earth Mover's Distance (L1 norm of cumulative difference)
    emd = np.sum(np.abs(cum_true - cum_pred))
    
    # Normalize to [0,1] and convert to agreement (1 - distance)
    max_emd = K - 1  # Maximum possible EMD for ordinal data
    agreement = 1.0 - (emd / max_emd)
    
    return max(0.0, agreement)

def compare_agreement_measures(y_true_list: List[int], y_pred_probs_list: List[np.ndarray], 
                             K: int = 4) -> dict:
    """
    Compare different agreement measures across multiple predictions.
    
    Args:
        y_true_list: List of true categories
        y_pred_probs_list: List of predicted probability arrays
        K: Number of ordinal categories
    
    Returns:
        Dictionary with agreement scores for each measure
    """
    results = {
        'qwk_agreement': [],
        'triangular_agreement': [],
        'emd_agreement': []
    }
    
    for y_true, y_pred_probs in zip(y_true_list, y_pred_probs_list):
        results['qwk_agreement'].append(
            qwk_probability_agreement(y_true, y_pred_probs, K)
        )
        results['triangular_agreement'].append(
            triangular_probability_agreement(y_true, y_pred_probs, K)
        )
        results['emd_agreement'].append(
            earth_movers_distance_agreement(y_true, y_pred_probs, K)
        )
    
    return results

def visualize_weight_functions(K: int = 4, save_path: str = None):
    """
    Visualize the different weighting functions for ordinal agreement.
    
    Args:
        K: Number of ordinal categories
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # For each true category, show weights for predicted categories
    categories = np.arange(K)
    
    for true_cat in range(K):
        # QWK weights
        qwk_weights = [1 - (true_cat - k)**2 / (K-1)**2 for k in range(K)]
        axes[0].plot(categories, qwk_weights, 'o-', label=f'True={true_cat}', linewidth=2)
        
        # Triangular weights
        tri_weights = [max(0, 1 - abs(true_cat - k) / (K-1)) for k in range(K)]
        axes[1].plot(categories, tri_weights, 's-', label=f'True={true_cat}', linewidth=2)
        
        # EMD weights (for reference - shows step function nature)
        emd_weights = [1.0 if k == true_cat else 0.0 for k in range(K)]
        axes[2].plot(categories, emd_weights, '^-', label=f'True={true_cat}', linewidth=2)
    
    # Configure plots
    titles = ['QWK Weights (Quadratic)', 'Triangular Weights (Linear)', 'One-Hot (EMD basis)']
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('Weight')
        ax.set_title(title)
        ax.set_xticks(categories)
        ax.set_xticklabels([f'Cat {k}' for k in categories])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
    
    plt.suptitle(f'Ordinal Agreement Weight Functions (K={K} categories)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Weight functions plot saved to: {save_path}")
    else:
        plt.show()

def generate_agreement_heatmap(K: int = 4, save_path: str = None):
    """
    Generate heatmap showing agreement scores for all true/predicted combinations.
    
    Args:
        K: Number of ordinal categories
        save_path: Optional path to save the plot
    """
    # Create grid of all possible one-hot predictions
    agreement_matrix = np.zeros((K, K))
    
    for true_cat in range(K):
        for pred_cat in range(K):
            # Create one-hot prediction
            pred_probs = np.zeros(K)
            pred_probs[pred_cat] = 1.0
            
            # Compute QWK agreement
            agreement_matrix[true_cat, pred_cat] = qwk_probability_agreement(
                true_cat, pred_probs, K
            )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(K):
        for j in range(K):
            text = ax.text(j, i, f'{agreement_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # Configure plot
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'Cat {k}' for k in range(K)])
    ax.set_yticklabels([f'Cat {k}' for k in range(K)])
    ax.set_xlabel('Predicted Category (one-hot)')
    ax.set_ylabel('True Category')
    ax.set_title('QWK Probability Agreement Matrix\n(Values for one-hot predictions)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('QWK Agreement Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Agreement heatmap saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage and validation
    print("QWK Probability Agreement Examples:")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        (0, [1.0, 0.0, 0.0, 0.0], "Perfect prediction"),
        (0, [0.0, 1.0, 0.0, 0.0], "Adjacent category"),
        (0, [0.0, 0.0, 1.0, 0.0], "Two categories off"),
        (0, [0.0, 0.0, 0.0, 1.0], "Opposite category"),
        (0, [0.25, 0.25, 0.25, 0.25], "Uniform prediction"),
        (1, [0.1, 0.7, 0.2, 0.0], "Realistic soft prediction"),
    ]
    
    for y_true, y_pred_probs, description in test_cases:
        agreement = qwk_probability_agreement(y_true, y_pred_probs, K=4)
        print(f"{description:20s}: {agreement:.4f}")
    
    print("\nGenerating visualizations...")
    
    # Generate visualizations
    visualize_weight_functions(K=4)
    generate_agreement_heatmap(K=4)