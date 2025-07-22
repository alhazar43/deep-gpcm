#!/usr/bin/env python3
"""
Evaluation Script for Deep-GPCM Model

Comprehensive model evaluation with baseline comparisons and statistical analysis.
"""

import os
import torch
import numpy as np
import json
import logging
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.model import DeepGpcmModel
from evaluation.metrics import GpcmMetrics
from utils.gpcm_utils import (
    OrdinalLoss, load_gpcm_data, create_gpcm_batch,
    CrossEntropyLossWrapper, MSELossWrapper
)
from train import GpcmDataLoader


def setup_logging(dataset_name, model_name="deepgpcm"):
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/eval_{model_name}_{dataset_name}_{timestamp}.log"
    
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


class RandomBaseline:
    """Random baseline for comparison."""
    
    def __init__(self, n_cats):
        self.n_cats = n_cats
    
    def predict(self, questions, responses):
        """Generate random predictions."""
        predictions = []
        for q_seq, r_seq in zip(questions, responses):
            seq_preds = np.random.randint(0, self.n_cats, len(q_seq))
            predictions.append(seq_preds)
        return predictions


class FrequencyBaseline:
    """Frequency-based baseline using training distribution."""
    
    def __init__(self, n_cats):
        self.n_cats = n_cats
        self.response_probs = None
    
    def fit(self, train_responses):
        """Fit on training data to get response frequency."""
        all_responses = []
        for r_seq in train_responses:
            all_responses.extend(r_seq)
        
        # Count frequency of each response
        counts = np.bincount(all_responses, minlength=self.n_cats)
        self.response_probs = counts / counts.sum()
    
    def predict(self, questions, responses):
        """Generate predictions based on training frequency."""
        if self.response_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for q_seq, r_seq in zip(questions, responses):
            seq_preds = np.random.choice(self.n_cats, len(q_seq), p=self.response_probs)
            predictions.append(seq_preds)
        return predictions


def evaluate_baselines(train_responses, test_questions, test_responses, n_cats, metrics):
    """Evaluate baseline models."""
    results = {}
    
    # Random baseline
    random_model = RandomBaseline(n_cats)
    random_preds = random_model.predict(test_questions, test_responses)
    
    # Convert to tensor format for metrics
    all_preds = []
    all_targets = []
    for pred_seq, true_seq in zip(random_preds, test_responses):
        all_preds.extend(pred_seq)
        all_targets.extend(true_seq)
    
    pred_tensor = torch.tensor(all_preds).float().unsqueeze(1)
    target_tensor = torch.tensor(all_targets).long().unsqueeze(1)
    
    # Create probability distribution (uniform for random)
    prob_tensor = torch.zeros(len(all_preds), 1, n_cats)
    for i, pred in enumerate(all_preds):
        prob_tensor[i, 0, pred] = 1.0
    
    results['random'] = {
        'categorical_acc': metrics.categorical_accuracy(prob_tensor, target_tensor),
        'ordinal_acc': metrics.ordinal_accuracy(prob_tensor, target_tensor),
        'mae': metrics.mean_absolute_error(prob_tensor, target_tensor),
        'qwk': metrics.quadratic_weighted_kappa(prob_tensor, target_tensor, n_cats)
    }
    
    # Frequency baseline
    freq_model = FrequencyBaseline(n_cats)
    freq_model.fit(train_responses)
    freq_preds = freq_model.predict(test_questions, test_responses)
    
    all_preds = []
    for pred_seq in freq_preds:
        all_preds.extend(pred_seq)
    
    pred_tensor = torch.tensor(all_preds).float().unsqueeze(1)
    prob_tensor = torch.zeros(len(all_preds), 1, n_cats)
    for i, pred in enumerate(all_preds):
        prob_tensor[i, 0, pred] = 1.0
    
    results['frequency'] = {
        'categorical_acc': metrics.categorical_accuracy(prob_tensor, target_tensor),
        'ordinal_acc': metrics.ordinal_accuracy(prob_tensor, target_tensor),
        'mae': metrics.mean_absolute_error(prob_tensor, target_tensor),
        'qwk': metrics.quadratic_weighted_kappa(prob_tensor, target_tensor, n_cats)
    }
    
    return results


def create_confusion_matrix(model, test_loader, device, n_cats, save_path):
    """Create and save confusion matrix."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in test_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            _, _, _, gpcm_probs = model(q_batch, r_batch)
            
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            predictions = torch.argmax(valid_probs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(n_cats))
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(n_cats), yticklabels=range(n_cats))
    plt.title('Confusion Matrix - Deep-GPCM')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def evaluate_model_comprehensive(model_path, dataset_name, device=None):
    """Comprehensive model evaluation."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logging(dataset_name)
    logger.info(f"Evaluating Deep-GPCM on {dataset_name}")
    
    # Create directories
    os.makedirs("results/eval", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Load checkpoint
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    n_cats = checkpoint['n_cats']
    n_questions = checkpoint['n_questions']
    
    logger.info(f"Model config: {n_questions} questions, {n_cats} categories")
    
    # Load data
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    logger.info("Loading data...")
    train_seqs, train_questions, train_responses, _ = load_gpcm_data(train_path, n_cats)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    logger.info(f"Train: {len(train_seqs)} sequences, Test: {len(test_seqs)} sequences")
    
    # Create model
    model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test loader
    test_loader = GpcmDataLoader(test_questions, test_responses, batch_size=64, shuffle=False)
    
    # Loss functions and metrics
    ordinal_loss = OrdinalLoss(n_cats)
    ce_loss = CrossEntropyLossWrapper()
    mse_loss = MSELossWrapper()
    metrics = GpcmMetrics()
    
    logger.info("Evaluating Deep-GPCM model...")
    
    # Evaluate with different loss functions
    model_results = {}
    
    for loss_name, loss_fn in [('ordinal', ordinal_loss), ('crossentropy', ce_loss), ('mse', mse_loss)]:
        logger.info(f"Evaluating with {loss_name} loss...")
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for q_batch, r_batch, mask_batch in test_loader:
                q_batch = q_batch.to(device)
                r_batch = r_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                _, _, _, gpcm_probs = model(q_batch, r_batch)
                
                if mask_batch is not None:
                    valid_probs = gpcm_probs[mask_batch]
                    valid_targets = r_batch[mask_batch]
                else:
                    valid_probs = gpcm_probs.view(-1, n_cats)
                    valid_targets = r_batch.view(-1)
                
                loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
                total_loss += loss.item()
                
                all_predictions.append(valid_probs.cpu())
                all_targets.append(valid_targets.cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0).unsqueeze(1)
        all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
        
        model_results[loss_name] = {
            'loss': total_loss / len(test_loader),
            'categorical_acc': metrics.categorical_accuracy(all_predictions, all_targets),
            'ordinal_acc': metrics.ordinal_accuracy(all_predictions, all_targets),
            'mae': metrics.mean_absolute_error(all_predictions, all_targets),
            'qwk': metrics.quadratic_weighted_kappa(all_predictions, all_targets, n_cats)
        }
        
        per_cat_acc = metrics.per_category_accuracy(all_predictions, all_targets, n_cats)
        model_results[loss_name].update(per_cat_acc)
    
    # Evaluate baselines
    logger.info("Evaluating baseline models...")
    baseline_results = evaluate_baselines(train_responses, test_questions, test_responses, n_cats, metrics)
    
    # Create confusion matrix
    logger.info("Creating confusion matrix...")
    cm_path = f"results/plots/confusion_matrix_{dataset_name}.png"
    cm = create_confusion_matrix(model, test_loader, device, n_cats, cm_path)
    
    # Compile results
    evaluation_results = {
        'dataset': dataset_name,
        'model_path': model_path,
        'n_cats': n_cats,
        'n_questions': n_questions,
        'model_results': model_results,
        'baseline_results': baseline_results,
        'confusion_matrix': cm.tolist(),
        'evaluation_date': datetime.now().isoformat()
    }
    
    # Save results
    results_path = f"results/eval/evaluation_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Categories: {n_cats}")
    logger.info(f"Test Sequences: {len(test_seqs)}")
    
    logger.info("\nModel Performance (Ordinal Loss):")
    ordinal_res = model_results['ordinal']
    logger.info(f"  Categorical Accuracy: {ordinal_res['categorical_acc']:.4f}")
    logger.info(f"  Ordinal Accuracy:     {ordinal_res['ordinal_acc']:.4f}")
    logger.info(f"  Mean Absolute Error:  {ordinal_res['mae']:.4f}")
    logger.info(f"  Quadratic Weighted κ: {ordinal_res['qwk']:.4f}")
    
    logger.info("\nBaseline Comparisons:")
    for baseline_name, baseline_res in baseline_results.items():
        logger.info(f"  {baseline_name.title()} Baseline:")
        logger.info(f"    Categorical Acc: {baseline_res['categorical_acc']:.4f}")
        logger.info(f"    Ordinal Acc:     {baseline_res['ordinal_acc']:.4f}")
        logger.info(f"    QWK:             {baseline_res['qwk']:.4f}")
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    return evaluation_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Evaluate Deep-GPCM Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Evaluate model
    results = evaluate_model_comprehensive(args.model_path, args.dataset)
    
    return results


if __name__ == "__main__":
    main()