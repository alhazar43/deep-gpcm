#!/usr/bin/env python3
"""
Visualization Dashboard for Deep-GPCM

Advanced logging and metrics visualization for training analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from datetime import datetime
import logging


def setup_visualization_logging():
    """Setup logging for visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/visualization_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_training_history(dataset_name):
    """Load training history from JSON file."""
    history_path = f"results/train/training_history_{dataset_name}.json"
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return pd.DataFrame(history)


def load_cv_results(dataset_name, n_folds=5):
    """Load cross-validation results."""
    cv_path = f"results/cv/cv_results_{dataset_name}_{n_folds}fold.json"
    
    if not os.path.exists(cv_path):
        return None
    
    with open(cv_path, 'r') as f:
        cv_results = json.load(f)
    
    return cv_results


def load_analysis_results(dataset_name):
    """Load strategy analysis results."""
    analysis_path = f"results/analysis/strategy_analysis_{dataset_name}.json"
    
    if not os.path.exists(analysis_path):
        return None
    
    with open(analysis_path, 'r') as f:
        analysis_results = json.load(f)
    
    return analysis_results


def plot_training_curves(df, dataset_name, save_path="results/plots"):
    """Plot training curves."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training and validation loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['valid_loss'], label='Valid Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Categorical accuracy
    axes[0, 1].plot(df['epoch'], df['categorical_acc'], color='green', linewidth=2)
    axes[0, 1].set_title('Categorical Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ordinal accuracy
    axes[0, 2].plot(df['epoch'], df['ordinal_acc'], color='orange', linewidth=2)
    axes[0, 2].set_title('Ordinal Accuracy (±1)', fontsize=14)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Mean Absolute Error
    axes[1, 0].plot(df['epoch'], df['mae'], color='red', linewidth=2)
    axes[1, 0].set_title('Mean Absolute Error', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quadratic Weighted Kappa
    axes[1, 1].plot(df['epoch'], df['qwk'], color='purple', linewidth=2)
    axes[1, 1].set_title('Quadratic Weighted Kappa', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('QWK')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 2].plot(df['epoch'], df['learning_rate'], color='brown', linewidth=2)
    axes[1, 2].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Deep-GPCM Training Metrics - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cv_results(cv_results, dataset_name, save_path="results/plots"):
    """Plot cross-validation results."""
    
    if cv_results is None:
        return
    
    # Extract CV statistics
    cv_stats = cv_results['cv_statistics']
    metrics = ['categorical_acc', 'ordinal_acc', 'mae', 'qwk']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot of CV results
    cv_data = []
    for metric in metrics:
        for value in cv_stats[metric]['values']:
            cv_data.append({'Metric': metric.replace('_', ' ').title(), 'Value': value})
    
    cv_df = pd.DataFrame(cv_data)
    
    sns.boxplot(data=cv_df, x='Metric', y='Value', ax=axes[0])
    axes[0].set_title(f'{cv_results["n_folds"]}-Fold Cross-Validation Results')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Error bars plot
    means = [cv_stats[metric]['mean'] for metric in metrics]
    stds = [cv_stats[metric]['std'] for metric in metrics]
    metric_names = [metric.replace('_', ' ').title() for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                color=['blue', 'green', 'red', 'purple'])
    axes[1].set_title('Cross-Validation Summary (Mean ± Std)')
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metric_names, rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Deep-GPCM Cross-Validation Analysis - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_path}/cv_results_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_progression(df, dataset_name, save_path="results/plots"):
    """Plot detailed learning progression analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss convergence analysis
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
    axes[0, 0].plot(df['epoch'], df['valid_loss'], label='Valid Loss', linewidth=2, alpha=0.8)
    
    # Add trend lines
    z_train = np.polyfit(df['epoch'], df['train_loss'], 1)
    p_train = np.poly1d(z_train)
    axes[0, 0].plot(df['epoch'], p_train(df['epoch']), '--', alpha=0.5, color='blue')
    
    z_valid = np.polyfit(df['epoch'], df['valid_loss'], 1)
    p_valid = np.poly1d(z_valid)
    axes[0, 0].plot(df['epoch'], p_valid(df['epoch']), '--', alpha=0.5, color='orange')
    
    axes[0, 0].set_title('Loss Convergence with Trends')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy improvement rate
    cat_acc_diff = df['categorical_acc'].diff().fillna(0)
    axes[0, 1].plot(df['epoch'], cat_acc_diff, color='green', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Categorical Accuracy Improvement Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy Change')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance correlation
    metrics_corr = df[['categorical_acc', 'ordinal_acc', 'mae', 'qwk']].corr()
    sns.heatmap(metrics_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
    axes[1, 0].set_title('Metrics Correlation Matrix')
    
    # Training stability (rolling statistics)
    window_size = max(3, len(df) // 5)
    rolling_loss = df['valid_loss'].rolling(window=window_size).std()
    rolling_acc = df['categorical_acc'].rolling(window=window_size).std()
    
    ax2 = axes[1, 1]
    ax2.plot(df['epoch'], rolling_loss, color='red', linewidth=2, label='Loss Std')
    ax2.set_ylabel('Loss Std', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['epoch'], rolling_acc, color='blue', linewidth=2, label='Acc Std')
    ax2_twin.set_ylabel('Accuracy Std', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')
    
    ax2.set_title(f'Training Stability (Rolling Std, window={window_size})')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Deep-GPCM Learning Progression Analysis - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/learning_progression_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_dashboard(dataset_name, save_path="results/plots"):
    """Create comprehensive performance dashboard."""
    
    logger = setup_visualization_logging()
    logger.info(f"Creating performance dashboard for {dataset_name}")
    
    # Load all available results
    try:
        df = load_training_history(dataset_name)
        logger.info(f"Loaded training history with {len(df)} epochs")
    except FileNotFoundError as e:
        logger.error(f"Training history not found: {e}")
        return
    
    cv_results = load_cv_results(dataset_name)
    if cv_results:
        logger.info("Loaded cross-validation results")
    
    analysis_results = load_analysis_results(dataset_name)
    if analysis_results:
        logger.info("Loaded strategy analysis results")
    
    # Create plots
    os.makedirs(save_path, exist_ok=True)
    
    logger.info("Generating training curves...")
    plot_training_curves(df, dataset_name, save_path)
    
    logger.info("Generating learning progression analysis...")
    plot_learning_progression(df, dataset_name, save_path)
    
    if cv_results:
        logger.info("Generating cross-validation plots...")
        plot_cv_results(cv_results, dataset_name, save_path)
    
    # Create summary report
    create_performance_report(df, cv_results, analysis_results, dataset_name, save_path)
    
    logger.info(f"Dashboard complete! Plots saved to: {save_path}")


def create_performance_report(df, cv_results, analysis_results, dataset_name, save_path):
    """Create detailed performance report."""
    
    report_path = f"{save_path}/performance_report_{dataset_name}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"DEEP-GPCM PERFORMANCE REPORT\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Training Summary
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Epochs: {len(df)}\n")
        f.write(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}\n")
        f.write(f"Final Valid Loss: {df['valid_loss'].iloc[-1]:.4f}\n")
        f.write(f"Best Categorical Accuracy: {df['categorical_acc'].max():.4f}\n")
        f.write(f"Best Ordinal Accuracy: {df['ordinal_acc'].max():.4f}\n")
        f.write(f"Best QWK: {df['qwk'].max():.4f}\n")
        f.write(f"Final Learning Rate: {df['learning_rate'].iloc[-1]:.6f}\n\n")
        
        # Convergence Analysis
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        loss_improvement = df['valid_loss'].iloc[0] - df['valid_loss'].iloc[-1]
        acc_improvement = df['categorical_acc'].iloc[-1] - df['categorical_acc'].iloc[0]
        
        f.write(f"Loss Improvement: {loss_improvement:.4f}\n")
        f.write(f"Accuracy Improvement: {acc_improvement:.4f}\n")
        
        # Find convergence epoch (when improvement becomes minimal)
        loss_diff = df['valid_loss'].diff().abs()
        convergence_threshold = 0.01
        convergence_epochs = df[loss_diff < convergence_threshold]['epoch'].tolist()
        if convergence_epochs:
            f.write(f"Convergence Epoch (±{convergence_threshold}): {convergence_epochs[0]}\n")
        
        f.write("\n")
        
        # Cross-validation summary
        if cv_results:
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-" * 28 + "\n")
            cv_stats = cv_results['cv_statistics']
            
            for metric in ['categorical_acc', 'ordinal_acc', 'mae', 'qwk']:
                stats = cv_stats[metric]
                f.write(f"{metric.replace('_', ' ').title():<20}: "
                       f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                       f"(range: {stats['min']:.4f}-{stats['max']:.4f})\n")
            f.write("\n")
        
        # Strategy analysis summary
        if analysis_results:
            f.write("STRATEGY ANALYSIS SUMMARY\n")
            f.write("-" * 25 + "\n")
            comparison_data = analysis_results['comparison_summary']
            
            # Best configurations
            best_cat = max(comparison_data, key=lambda x: x['categorical_acc'])
            best_qwk = max(comparison_data, key=lambda x: x['qwk'])
            
            f.write(f"Best Categorical Accuracy: {best_cat['config_name']} - {best_cat['categorical_acc']:.4f}\n")
            f.write(f"Best QWK: {best_qwk['config_name']} - {best_qwk['qwk']:.4f}\n")
            
            # Strategy ranking
            strategy_avg = {}
            for item in comparison_data:
                strategy = item['strategy']
                if strategy not in strategy_avg:
                    strategy_avg[strategy] = []
                strategy_avg[strategy].append(item['categorical_acc'])
            
            f.write("\nStrategy Performance Ranking:\n")
            strategy_means = {s: np.mean(scores) for s, scores in strategy_avg.items()}
            for i, (strategy, mean_score) in enumerate(sorted(strategy_means.items(), 
                                                             key=lambda x: x[1], reverse=True), 1):
                f.write(f"{i}. {strategy.title()}: {mean_score:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated by Deep-GPCM Visualization Dashboard\n")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Create Deep-GPCM Performance Dashboard')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--save_path', type=str, default='results/plots',
                        help='Save path for plots (default: results/plots)')
    
    args = parser.parse_args()
    
    # Create dashboard
    create_performance_dashboard(args.dataset, args.save_path)


if __name__ == "__main__":
    main()