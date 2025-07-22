#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Deep-GPCM

Compares embedding strategies across different configurations and K values.
"""

import os
import torch
import numpy as np
import json
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from models.model import DeepGpcmModel
from utils.gpcm_utils import (
    OrdinalLoss, GpcmMetrics, load_gpcm_data, create_gpcm_batch,
    CrossEntropyLossWrapper, MSELossWrapper
)
from train import GpcmDataLoader, train_epoch, evaluate_model, setup_logging


def setup_analysis_logging():
    """Setup logging for analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/analysis_{timestamp}.log"
    
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


def train_single_config(config, train_loader, valid_loader, device, logger):
    """Train a single configuration and return results."""
    
    # Create model with specified configuration
    model = DeepGpcmModel(
        n_questions=config['n_questions'],
        n_cats=config['n_cats'],
        memory_size=config['memory_size'],
        key_dim=config['key_dim'],
        value_dim=config['value_dim'],
        final_fc_dim=config['final_fc_dim'],
        embedding_strategy=config['embedding_strategy']
    ).to(device)
    
    # Create loss function
    if config['loss_type'] == 'ordinal':
        loss_fn = OrdinalLoss(config['n_cats'])
    elif config['loss_type'] == 'crossentropy':
        loss_fn = CrossEntropyLossWrapper()
    elif config['loss_type'] == 'mse':
        loss_fn = MSELossWrapper()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    # Metrics
    metrics = GpcmMetrics()
    
    # Quick training (fewer epochs for comparative analysis)
    n_epochs = config['n_epochs']
    best_valid_acc = 0.0
    training_history = []
    
    for epoch in range(n_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config['n_cats'])
        
        # Validation
        valid_results = evaluate_model(model, valid_loader, loss_fn, metrics, device, config['n_cats'])
        
        # Update learning rate
        scheduler.step(valid_results['loss'])
        
        # Track best
        if valid_results['categorical_acc'] > best_valid_acc:
            best_valid_acc = valid_results['categorical_acc']
        
        # Store results
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_results['loss'],
            'categorical_acc': valid_results['categorical_acc'],
            'ordinal_acc': valid_results['ordinal_acc'],
            'mae': valid_results['mae'],
            'qwk': valid_results['qwk']
        })
    
    # Final evaluation
    final_results = evaluate_model(model, valid_loader, loss_fn, metrics, device, config['n_cats'])
    
    return {
        'config': config,
        'final_results': final_results,
        'best_valid_acc': best_valid_acc,
        'training_history': training_history,
        'model_params': sum(p.numel() for p in model.parameters())
    }


def create_analysis_configs(base_config, strategies, k_values):
    """Create all configuration combinations for analysis."""
    configs = []
    
    for strategy in strategies:
        for k in k_values:
            config = base_config.copy()
            config['embedding_strategy'] = strategy
            config['n_cats'] = k
            config['config_name'] = f"{strategy}_K{k}"
            configs.append(config)
    
    return configs


def analyze_embedding_strategies(dataset_name='synthetic_OC', quick_analysis=True):
    """Comprehensive analysis of embedding strategies."""
    
    logger = setup_analysis_logging()
    logger.info("Starting comprehensive embedding strategy analysis")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs("results/analysis", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Load data
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    logger.info(f"Loading data from {dataset_name}...")
    train_seqs, train_questions, train_responses, detected_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, detected_cats)
    
    # Determine n_questions
    all_questions = []
    for q_seq in train_questions + test_questions:
        all_questions.extend(q_seq)
    n_questions = max(all_questions)
    
    logger.info(f"Data loaded: {len(train_seqs)} train, {len(test_seqs)} test sequences")
    logger.info(f"Questions: {n_questions}, Detected categories: {detected_cats}")
    
    # Base configuration
    base_config = {
        'n_questions': n_questions,
        'memory_size': 50,
        'key_dim': 50,
        'value_dim': 200,
        'final_fc_dim': 50,
        'learning_rate': 0.001,
        'loss_type': 'ordinal',
        'n_epochs': 10 if quick_analysis else 20  # Quick analysis for comparison
    }
    
    # Analysis configurations
    strategies = ['ordered', 'unordered', 'linear_decay']
    k_values = [3, 4, 5] if quick_analysis else [2, 3, 4, 5, 6]  # Different K values
    
    configs = create_analysis_configs(base_config, strategies, k_values)
    
    logger.info(f"Running analysis on {len(configs)} configurations:")
    for config in configs:
        logger.info(f"  - {config['config_name']}")
    
    # Run analysis
    all_results = []
    
    for config in tqdm(configs, desc="Running configurations"):
        logger.info(f"\nTraining configuration: {config['config_name']}")
        
        # Create data loaders (adjust for K categories)
        # For fair comparison, we'll adjust synthetic data to have K categories
        adjusted_train_responses = []
        adjusted_test_responses = []
        
        for r_seq in train_responses:
            # Map responses to K categories: scale from [0, detected_cats-1] to [0, K-1]
            adjusted_seq = [int(r * (config['n_cats'] - 1) / (detected_cats - 1)) for r in r_seq]
            adjusted_train_responses.append(adjusted_seq)
        
        for r_seq in test_responses:
            adjusted_seq = [int(r * (config['n_cats'] - 1) / (detected_cats - 1)) for r in r_seq]
            adjusted_test_responses.append(adjusted_seq)
        
        train_loader = GpcmDataLoader(train_questions, adjusted_train_responses, 32, shuffle=True)
        valid_loader = GpcmDataLoader(test_questions, adjusted_test_responses, 32, shuffle=False)
        
        # Train and evaluate
        try:
            result = train_single_config(config, train_loader, valid_loader, device, logger)
            all_results.append(result)
            
            logger.info(f"  Results - Cat Acc: {result['final_results']['categorical_acc']:.4f}, "
                       f"Ord Acc: {result['final_results']['ordinal_acc']:.4f}, "
                       f"QWK: {result['final_results']['qwk']:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to train {config['config_name']}: {e}")
            continue
    
    # Analysis and visualization
    logger.info("\nGenerating analysis results...")
    
    # Create comparison dataframe
    comparison_data = []
    for result in all_results:
        config = result['config']
        final_res = result['final_results']
        
        comparison_data.append({
            'strategy': config['embedding_strategy'],
            'k_categories': config['n_cats'],
            'config_name': config['config_name'],
            'categorical_acc': final_res['categorical_acc'],
            'ordinal_acc': final_res['ordinal_acc'],
            'mae': final_res['mae'],
            'qwk': final_res['qwk'],
            'model_params': result['model_params'],
            'best_valid_acc': result['best_valid_acc']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Generate plots
    create_analysis_plots(df, dataset_name)
    
    # Save detailed results
    analysis_results = {
        'dataset': dataset_name,
        'base_config': base_config,
        'analysis_configs': [r['config'] for r in all_results],
        'results': all_results,
        'comparison_summary': comparison_data,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = f"results/analysis/strategy_analysis_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print summary
    print_analysis_summary(df, logger)
    
    logger.info(f"Analysis complete! Results saved to: {results_path}")
    
    return analysis_results


def create_analysis_plots(df, dataset_name):
    """Create visualization plots for analysis."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance comparison across strategies and K values
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Categorical Accuracy
    sns.boxplot(data=df, x='strategy', y='categorical_acc', ax=axes[0,0])
    axes[0,0].set_title('Categorical Accuracy by Strategy')
    axes[0,0].set_ylabel('Categorical Accuracy')
    
    # Ordinal Accuracy  
    sns.boxplot(data=df, x='strategy', y='ordinal_acc', ax=axes[0,1])
    axes[0,1].set_title('Ordinal Accuracy by Strategy')
    axes[0,1].set_ylabel('Ordinal Accuracy')
    
    # QWK by K categories
    sns.scatterplot(data=df, x='k_categories', y='qwk', hue='strategy', s=100, ax=axes[1,0])
    axes[1,0].set_title('QWK vs Number of Categories')
    axes[1,0].set_xlabel('Number of Categories (K)')
    axes[1,0].set_ylabel('Quadratic Weighted Kappa')
    
    # Model parameters vs performance
    sns.scatterplot(data=df, x='model_params', y='categorical_acc', hue='strategy', s=100, ax=axes[1,1])
    axes[1,1].set_title('Model Size vs Performance')
    axes[1,1].set_xlabel('Model Parameters')
    axes[1,1].set_ylabel('Categorical Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'results/plots/strategy_analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed heatmap
    pivot_data = df.pivot(index='strategy', columns='k_categories', values='categorical_acc')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Categorical Accuracy'})
    plt.title(f'Performance Heatmap: Categorical Accuracy\nDataset: {dataset_name}')
    plt.xlabel('Number of Categories (K)')
    plt.ylabel('Embedding Strategy')
    plt.tight_layout()
    plt.savefig(f'results/plots/strategy_heatmap_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_analysis_summary(df, logger):
    """Print analysis summary."""
    
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING STRATEGY ANALYSIS SUMMARY")
    logger.info("="*60)
    
    # Overall best performers
    best_cat_acc = df.loc[df['categorical_acc'].idxmax()]
    best_ord_acc = df.loc[df['ordinal_acc'].idxmax()]
    best_qwk = df.loc[df['qwk'].idxmax()]
    
    logger.info(f"\nBest Categorical Accuracy: {best_cat_acc['config_name']} - {best_cat_acc['categorical_acc']:.4f}")
    logger.info(f"Best Ordinal Accuracy:     {best_ord_acc['config_name']} - {best_ord_acc['ordinal_acc']:.4f}")
    logger.info(f"Best QWK:                  {best_qwk['config_name']} - {best_qwk['qwk']:.4f}")
    
    # Strategy comparison
    logger.info(f"\nStrategy Performance Summary:")
    strategy_summary = df.groupby('strategy')[['categorical_acc', 'ordinal_acc', 'qwk']].agg(['mean', 'std'])
    
    for strategy in ['ordered', 'unordered', 'linear_decay']:
        if strategy in strategy_summary.index:
            cat_mean = strategy_summary.loc[strategy, ('categorical_acc', 'mean')]
            cat_std = strategy_summary.loc[strategy, ('categorical_acc', 'std')]
            ord_mean = strategy_summary.loc[strategy, ('ordinal_acc', 'mean')]
            qwk_mean = strategy_summary.loc[strategy, ('qwk', 'mean')]
            
            logger.info(f"  {strategy.title():<12}: Cat Acc = {cat_mean:.4f}±{cat_std:.4f}, "
                       f"Ord Acc = {ord_mean:.4f}, QWK = {qwk_mean:.4f}")
    
    # K value analysis
    logger.info(f"\nPerformance vs Number of Categories:")
    k_summary = df.groupby('k_categories')[['categorical_acc', 'ordinal_acc', 'qwk']].mean()
    
    for k in sorted(k_summary.index):
        logger.info(f"  K={k}: Cat Acc = {k_summary.loc[k, 'categorical_acc']:.4f}, "
                   f"Ord Acc = {k_summary.loc[k, 'ordinal_acc']:.4f}, "
                   f"QWK = {k_summary.loc[k, 'qwk']:.4f}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze Deep-GPCM Embedding Strategies')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick analysis with fewer epochs and K values')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run analysis
    results = analyze_embedding_strategies(
        dataset_name=args.dataset,
        quick_analysis=args.quick
    )
    
    return results


if __name__ == "__main__":
    main()