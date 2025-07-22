#!/usr/bin/env python3
"""
Comprehensive Strategy Comparison Script for Deep-GPCM

Compares embedding strategies across PC and OC formats with epoch-wise analysis.
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
from train import GpcmDataLoader, setup_logging


def setup_comparison_logging():
    """Setup logging for comparison analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/strategy_comparison_{timestamp}.log"
    
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


def load_pc_data(data_path, n_cats):
    """Load Partial Credit format data and convert to integer categories."""
    # Load PC data (decimal format)
    train_path = f"{data_path}/synthetic_pc_train.txt"
    test_path = f"{data_path}/synthetic_pc_test.txt"
    
    # PC format uses decimal scores, need to read differently
    train_sequences = []
    test_sequences = []
    
    # Read PC train data
    with open(train_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  # PC format has 3 lines per sequence
            if i + 2 < len(lines):
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(float, lines[i+2].strip().split(',')))
                # Convert PC scores to categories: 0.0->0, 0.333->1, 0.667->2, 1.0->3
                categories = [round(r * (n_cats - 1)) for r in responses]
                train_sequences.append((questions, categories))
    
    # Read PC test data
    with open(test_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  # PC format has 3 lines per sequence
            if i + 2 < len(lines):
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(float, lines[i+2].strip().split(',')))
                categories = [round(r * (n_cats - 1)) for r in responses]
                test_sequences.append((questions, categories))
    
    # Separate questions and responses
    train_questions = [seq[0] for seq in train_sequences]
    train_responses = [seq[1] for seq in train_sequences]
    test_questions = [seq[0] for seq in test_sequences]
    test_responses = [seq[1] for seq in test_sequences]
    
    return train_sequences, train_questions, train_responses, test_sequences, test_questions, test_responses, n_cats


def train_strategy_comparison(strategy, dataset_path, format_type, n_epochs=5, n_cats=4):
    """Train a single strategy and return epoch-wise results."""
    
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training {strategy} strategy on {format_type} format")
    
    # Load data based on format
    if format_type == 'OC':
        train_seqs, train_questions, train_responses, detected_cats = load_gpcm_data(
            f"{dataset_path}/synthetic_oc_train.txt"
        )
        test_seqs, test_questions, test_responses, _ = load_gpcm_data(
            f"{dataset_path}/synthetic_oc_test.txt", detected_cats
        )
        n_cats = detected_cats
    elif format_type == 'PC':
        train_seqs, train_questions, train_responses, test_seqs, test_questions, test_responses, n_cats = load_pc_data(
            dataset_path, n_cats
        )
    else:
        raise ValueError(f"Unknown format: {format_type}")
    
    # Determine n_questions
    all_questions = []
    for q_seq in train_questions + test_questions:
        all_questions.extend(q_seq)
    n_questions = max(all_questions)
    
    logger.info(f"Data loaded: {len(train_questions)} train, {len(test_questions)} test sequences")
    logger.info(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create data loaders
    train_loader = GpcmDataLoader(train_questions, train_responses, batch_size=64, shuffle=True)
    test_loader = GpcmDataLoader(test_questions, test_responses, batch_size=64, shuffle=False)
    
    # Create model
    model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy=strategy
    ).to(device)
    
    # Loss and optimizer
    loss_fn = OrdinalLoss(n_cats)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = GpcmMetrics()
    
    # Training loop with epoch-wise tracking
    epoch_results = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        
        for q_batch, r_batch, mask_batch in train_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, _, _, _, gpcm_probs = model(q_batch, r_batch)
            
            # Compute loss
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / n_batches
        
        # Evaluation
        model.eval()
        total_valid_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for q_batch, r_batch, mask_batch in test_loader:
                q_batch = q_batch.to(device)
                r_batch = r_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                _, _, _, _, gpcm_probs = model(q_batch, r_batch)
                
                if mask_batch is not None:
                    valid_probs = gpcm_probs[mask_batch]
                    valid_targets = r_batch[mask_batch]
                else:
                    valid_probs = gpcm_probs.view(-1, n_cats)
                    valid_targets = r_batch.view(-1)
                
                loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
                total_valid_loss += loss.item()
                
                all_predictions.append(valid_probs.cpu())
                all_targets.append(valid_targets.cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0).unsqueeze(1)
        all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
        
        valid_loss = total_valid_loss / len(test_loader)
        categorical_acc = metrics.categorical_accuracy(all_predictions, all_targets)
        ordinal_acc = metrics.ordinal_accuracy(all_predictions, all_targets)
        mae = metrics.mean_absolute_error(all_predictions, all_targets)
        qwk = metrics.quadratic_weighted_kappa(all_predictions, all_targets, n_cats)
        
        epoch_result = {
            'epoch': epoch + 1,
            'strategy': strategy,
            'format': format_type,
            'train_loss': avg_train_loss,
            'valid_loss': valid_loss,
            'categorical_acc': categorical_acc,
            'ordinal_acc': ordinal_acc,
            'mae': mae,
            'qwk': qwk
        }
        epoch_results.append(epoch_result)
        
        logger.info(f"  Epoch {epoch + 1}/{n_epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Valid Loss: {valid_loss:.4f}, "
                   f"Cat Acc: {categorical_acc:.4f}, "
                   f"Ord Acc: {ordinal_acc:.4f}")
    
    return epoch_results


def analyze_prediction_targets(dataset_path):
    """Analyze prediction targets for PC and OC formats."""
    
    logger = logging.getLogger(__name__)
    logger.info("Analyzing prediction targets for PC and OC formats...")
    
    # Analyze OC format
    oc_train_path = f"{dataset_path}/synthetic_OC/synthetic_oc_train.txt"
    if os.path.exists(oc_train_path):
        train_seqs, train_questions, train_responses, n_cats_oc = load_gpcm_data(oc_train_path)
        
        # Flatten responses for analysis
        flat_responses_oc = []
        for r_seq in train_responses:
            flat_responses_oc.extend(r_seq)
        
        logger.info(f"\nOC Format Analysis:")
        logger.info(f"  Categories: {n_cats_oc} (discrete integers)")
        logger.info(f"  Value range: {min(flat_responses_oc)} to {max(flat_responses_oc)}")
        logger.info(f"  Unique values: {sorted(set(flat_responses_oc))}")
        logger.info(f"  Distribution: {np.bincount(flat_responses_oc)}")
    
    # Analyze PC format
    pc_train_path = f"{dataset_path}/synthetic_PC/synthetic_pc_train.txt"
    if os.path.exists(pc_train_path):
        with open(pc_train_path, 'r') as f:
            flat_responses_pc = []
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    responses = list(map(float, lines[i+2].strip().split(',')))
                    flat_responses_pc.extend(responses)
        
        logger.info(f"\nPC Format Analysis:")
        logger.info(f"  Format: Continuous scores [0.0, 1.0]")
        logger.info(f"  Value range: {min(flat_responses_pc):.3f} to {max(flat_responses_pc):.3f}")
        logger.info(f"  Unique values: {sorted(set(flat_responses_pc))}")
        
        # Convert to categories for comparison
        categories_pc = [round(r * 3) for r in flat_responses_pc]  # 0.0->0, 0.333->1, 0.667->2, 1.0->3
        logger.info(f"  Converted to categories: {np.bincount(categories_pc)}")
    
    return {
        'oc_format': {'type': 'discrete_integer', 'range': [0, n_cats_oc-1], 'categories': n_cats_oc},
        'pc_format': {'type': 'continuous_score', 'range': [0.0, 1.0], 'mapping': 'linear_to_categories'}
    }


def create_strategy_comparison_plots(all_results, save_path="results/plots"):
    """Create comprehensive comparison plots."""
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Categorical Accuracy over Epochs (by Strategy)
    for strategy in df['strategy'].unique():
        for format_type in df['format'].unique():
            subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
            axes[0, 0].plot(subset['epoch'], subset['categorical_acc'], 
                           marker='o', linewidth=2, alpha=0.8,
                           label=f"{strategy} ({format_type})")
    
    axes[0, 0].set_title('Categorical Accuracy over Epochs', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Categorical Accuracy')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Ordinal Accuracy over Epochs
    for strategy in df['strategy'].unique():
        for format_type in df['format'].unique():
            subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
            axes[0, 1].plot(subset['epoch'], subset['ordinal_acc'], 
                           marker='s', linewidth=2, alpha=0.8,
                           label=f"{strategy} ({format_type})")
    
    axes[0, 1].set_title('Ordinal Accuracy over Epochs', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Ordinal Accuracy (±1)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. QWK over Epochs
    for strategy in df['strategy'].unique():
        for format_type in df['format'].unique():
            subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
            axes[1, 0].plot(subset['epoch'], subset['qwk'], 
                           marker='^', linewidth=2, alpha=0.8,
                           label=f"{strategy} ({format_type})")
    
    axes[1, 0].set_title('Quadratic Weighted Kappa over Epochs', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('QWK')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Loss over Epochs
    for strategy in df['strategy'].unique():
        for format_type in df['format'].unique():
            subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
            axes[1, 1].plot(subset['epoch'], subset['train_loss'], 
                           marker='d', linewidth=2, alpha=0.8,
                           label=f"{strategy} ({format_type})")
    
    axes[1, 1].set_title('Training Loss over Epochs', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Final Performance Comparison (Grouped Bar plot with both metrics)
    final_epoch = df['epoch'].max()
    final_df = df[df['epoch'] == final_epoch]
    
    # Create explicit strategy labels
    strategy_labels = []
    for _, row in final_df.iterrows():
        if row['strategy'] == 'linear_decay':
            strat_name = 'Linear Decay\n(Triangular)'
        elif row['strategy'] == 'ordered':
            strat_name = 'Ordered\n(Binary+Score)'  
        elif row['strategy'] == 'unordered':
            strat_name = 'Unordered\n(One-hot)'
        else:
            strat_name = row['strategy']
        strategy_labels.append(f"{strat_name}\n({row['format']})")
    
    x_pos = np.arange(len(strategy_labels))
    width = 0.35
    
    # Plot both categorical and ordinal accuracy
    bars1 = axes[2, 0].bar(x_pos - width/2, final_df['categorical_acc'], width, 
                          label='Categorical Accuracy', alpha=0.8)
    bars2 = axes[2, 0].bar(x_pos + width/2, final_df['ordinal_acc'], width,
                          label='Ordinal Accuracy (±1)', alpha=0.8)
    
    axes[2, 0].set_title('Final Performance Comparison', fontsize=14)
    axes[2, 0].set_xlabel('Strategy (Format)')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(strategy_labels, rotation=45, ha='right')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        axes[2, 0].text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                       f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
        axes[2, 0].text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
                       f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Strategy-Format Performance Heatmap
    pivot_df = final_df.pivot(index='strategy', columns='format', values='categorical_acc')
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Categorical Accuracy'}, ax=axes[2, 1])
    axes[2, 1].set_title('Strategy vs Format Performance Heatmap', fontsize=14)
    axes[2, 1].set_xlabel('Data Format')
    axes[2, 1].set_ylabel('Embedding Strategy')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/strategy_format_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate plots for clearer visualization
    create_detailed_epoch_plots(df, save_path)


def create_detailed_epoch_plots(df, save_path):
    """Create detailed epoch-wise plots separated by metric."""
    
    metrics = ['categorical_acc', 'ordinal_acc', 'qwk', 'train_loss']
    metric_names = ['Categorical Accuracy', 'Ordinal Accuracy', 'Quadratic Weighted Kappa', 'Training Loss']
    
    for metric, metric_name in zip(metrics, metric_names):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot by strategy
        for strategy in df['strategy'].unique():
            for format_type in df['format'].unique():
                subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
                axes[0].plot(subset['epoch'], subset[metric], 
                           marker='o', linewidth=2.5, alpha=0.8,
                           label=f"{strategy} ({format_type})")
        
        axes[0].set_title(f'{metric_name} by Strategy and Format', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel(metric_name)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot improvement rates
        for strategy in df['strategy'].unique():
            for format_type in df['format'].unique():
                subset = df[(df['strategy'] == strategy) & (df['format'] == format_type)]
                if len(subset) > 1:
                    improvement = subset[metric].diff().fillna(0)
                    axes[1].plot(subset['epoch'], improvement, 
                               marker='s', linewidth=2, alpha=0.8,
                               label=f"{strategy} ({format_type})")
        
        axes[1].set_title(f'{metric_name} Improvement Rate', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(f'{metric_name} Change')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/detailed_{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare Deep-GPCM Embedding Strategies')
    parser.add_argument('--dataset_path', type=str, default='data/large',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--strategies', nargs='+', 
                        default=['ordered', 'unordered', 'linear_decay'],
                        help='Embedding strategies to compare')
    parser.add_argument('--formats', nargs='+', default=['OC', 'PC'],
                        help='Data formats to test')
    
    args = parser.parse_args()
    
    logger = setup_comparison_logging()
    logger.info("Starting comprehensive strategy and format comparison")
    
    # Create directories
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Analyze prediction targets first
    logger.info("="*60)
    logger.info("PREDICTION TARGET ANALYSIS")
    logger.info("="*60)
    
    target_analysis = analyze_prediction_targets(args.dataset_path)
    
    # Run comparison across strategies and formats
    logger.info("\n" + "="*60)
    logger.info("STRATEGY COMPARISON ANALYSIS")
    logger.info("="*60)
    
    all_results = []
    
    total_configs = len(args.strategies) * len(args.formats)
    current_config = 0
    
    for strategy in args.strategies:
        for format_type in args.formats:
            current_config += 1
            logger.info(f"\n[{current_config}/{total_configs}] Running {strategy} on {format_type} format")
            
            try:
                if format_type == 'OC':
                    dataset_path = f"{args.dataset_path}/synthetic_OC"
                elif format_type == 'PC':
                    dataset_path = f"{args.dataset_path}/synthetic_PC"
                else:
                    continue
                
                epoch_results = train_strategy_comparison(
                    strategy, dataset_path, format_type, args.epochs
                )
                all_results.extend(epoch_results)
                
            except Exception as e:
                logger.error(f"Failed to run {strategy} on {format_type}: {e}")
                continue
    
    # Save results
    results_data = {
        'comparison_config': {
            'strategies': args.strategies,
            'formats': args.formats,
            'epochs': args.epochs,
            'dataset_path': args.dataset_path
        },
        'target_analysis': target_analysis,
        'epoch_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = f"results/comparison/strategy_format_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Create plots
    logger.info("\nGenerating comparison plots...")
    create_strategy_comparison_plots(all_results, "results/plots")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    if all_results:
        df = pd.DataFrame(all_results)
        final_results = df[df['epoch'] == df['epoch'].max()]
        
        logger.info(f"\nFinal Performance (Epoch {df['epoch'].max()}):")
        for _, row in final_results.iterrows():
            logger.info(f"  {row['strategy']} ({row['format']}): "
                       f"Cat Acc = {row['categorical_acc']:.4f}, "
                       f"Ord Acc = {row['ordinal_acc']:.4f}, "
                       f"QWK = {row['qwk']:.4f}")
        
        # Best performers
        best_cat = final_results.loc[final_results['categorical_acc'].idxmax()]
        best_ord = final_results.loc[final_results['ordinal_acc'].idxmax()]
        best_qwk = final_results.loc[final_results['qwk'].idxmax()]
        
        logger.info(f"\nBest Performers:")
        logger.info(f"  Categorical Accuracy: {best_cat['strategy']} ({best_cat['format']}) - {best_cat['categorical_acc']:.4f}")
        logger.info(f"  Ordinal Accuracy: {best_ord['strategy']} ({best_ord['format']}) - {best_ord['ordinal_acc']:.4f}")
        logger.info(f"  QWK: {best_qwk['strategy']} ({best_qwk['format']}) - {best_qwk['qwk']:.4f}")
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("Plots saved to: results/plots/")


if __name__ == "__main__":
    main()