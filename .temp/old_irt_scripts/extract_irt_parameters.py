#!/usr/bin/env python3
"""
Extract IRT parameters from trained Deep-GPCM models.

This script loads a trained model and extracts the IRT parameters (theta, alpha, beta)
for analysis without retraining.
"""

import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import DeepGPCM, AttentionGPCM
from train import load_simple_data, create_data_loaders


def extract_irt_parameters(model, data_loader, device):
    """Extract IRT parameters from model for all sequences in the dataset.
    
    Returns:
        dict: Contains the following arrays:
            - student_abilities: (n_students, max_seq_len) - theta values per student over time
            - item_discriminations: (n_students, max_seq_len) - alpha values (item-specific at each time)
            - item_thresholds: (n_students, max_seq_len, n_cats-1) - beta values (item-specific at each time)
            - question_ids: (n_students, max_seq_len) - which question at each position
            - responses: (n_students, max_seq_len) - actual responses
            - sequence_lengths: (n_students,) - actual length of each sequence
            - masks: (n_students, max_seq_len) - valid positions mask
    """
    model.eval()
    
    all_abilities = []
    all_discriminations = []
    all_thresholds = []
    all_questions = []
    all_responses = []
    all_seq_lengths = []
    all_masks = []
    
    # Find maximum sequence length first
    max_seq_len = 0
    for batch_idx, (questions, responses, mask) in enumerate(data_loader):
        max_seq_len = max(max_seq_len, questions.shape[1])
    
    with torch.no_grad():
        for batch_idx, (questions, responses, mask) in enumerate(data_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            batch_size, seq_len = questions.shape
            
            # Forward pass to get IRT parameters
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            # Pad to max_seq_len if needed
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                student_abilities = F.pad(student_abilities, (0, pad_len), value=0)
                discrimination_params = F.pad(discrimination_params, (0, pad_len), value=0)
                item_thresholds = F.pad(item_thresholds, (0, 0, 0, pad_len), value=0)
                questions = F.pad(questions, (0, pad_len), value=0)
                responses = F.pad(responses, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)
            
            # Convert to numpy and store
            all_abilities.append(student_abilities.cpu().numpy())
            all_discriminations.append(discrimination_params.cpu().numpy())
            all_thresholds.append(item_thresholds.cpu().numpy())
            all_questions.append(questions.cpu().numpy())
            all_responses.append(responses.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            
            # Calculate actual sequence lengths from mask
            seq_lengths = mask.sum(dim=1).cpu().numpy()
            all_seq_lengths.append(seq_lengths)
    
    # Concatenate all batches
    abilities = np.concatenate(all_abilities, axis=0)
    discriminations = np.concatenate(all_discriminations, axis=0)
    thresholds = np.concatenate(all_thresholds, axis=0)
    questions = np.concatenate(all_questions, axis=0)
    responses = np.concatenate(all_responses, axis=0)
    seq_lengths = np.concatenate(all_seq_lengths, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    return {
        'student_abilities': abilities,  # (n_students, max_seq_len)
        'item_discriminations': discriminations,  # (n_students, max_seq_len)
        'item_thresholds': thresholds,  # (n_students, max_seq_len, n_cats-1)
        'question_ids': questions,  # (n_students, max_seq_len)
        'responses': responses,  # (n_students, max_seq_len)
        'sequence_lengths': seq_lengths,  # (n_students,)
        'masks': masks  # (n_students, max_seq_len)
    }


def analyze_parameters(params):
    """Analyze extracted IRT parameters and print summary statistics."""
    print("\n" + "="*80)
    print("IRT PARAMETER ANALYSIS")
    print("="*80)
    
    abilities = params['student_abilities']
    discriminations = params['item_discriminations']
    thresholds = params['item_thresholds']
    questions = params['question_ids']
    seq_lengths = params['sequence_lengths']
    
    n_students, max_seq_len = abilities.shape
    n_questions = questions.max() + 1
    n_cats = thresholds.shape[-1] + 1
    
    print(f"\nDataset Statistics:")
    print(f"  Number of students: {n_students}")
    print(f"  Number of questions: {n_questions}")
    print(f"  Number of categories: {n_cats}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Avg sequence length: {seq_lengths.mean():.1f}")
    
    # Student ability analysis
    print(f"\nStudent Ability (θ) Analysis:")
    print(f"  Shape: {abilities.shape} - (n_students, seq_len)")
    print(f"  Range: [{abilities.min():.3f}, {abilities.max():.3f}]")
    print(f"  Mean: {abilities.mean():.3f}, Std: {abilities.std():.3f}")
    
    # Track ability evolution
    initial_abilities = []
    final_abilities = []
    for i in range(n_students):
        seq_len = int(seq_lengths[i])
        if seq_len > 0:
            initial_abilities.append(abilities[i, 0])
            final_abilities.append(abilities[i, seq_len-1])
    
    print(f"  Initial abilities: Mean={np.mean(initial_abilities):.3f}, Std={np.std(initial_abilities):.3f}")
    print(f"  Final abilities: Mean={np.mean(final_abilities):.3f}, Std={np.std(final_abilities):.3f}")
    print(f"  Average change: {np.mean(final_abilities) - np.mean(initial_abilities):.3f}")
    
    # Item discrimination analysis
    print(f"\nItem Discrimination (α) Analysis:")
    print(f"  Shape: {discriminations.shape} - (n_students, seq_len)")
    print(f"  Range: [{discriminations.min():.3f}, {discriminations.max():.3f}]")
    print(f"  Mean: {discriminations.mean():.3f}, Std: {discriminations.std():.3f}")
    
    # Item threshold analysis
    print(f"\nItem Thresholds (β) Analysis:")
    print(f"  Shape: {thresholds.shape} - (n_students, seq_len, n_cats-1)")
    for k in range(n_cats-1):
        beta_k = thresholds[:, :, k]
        print(f"  β_{k}: Range=[{beta_k.min():.3f}, {beta_k.max():.3f}], Mean={beta_k.mean():.3f}, Std={beta_k.std():.3f}")
    
    # Check monotonicity of thresholds
    monotonic = True
    for k in range(n_cats-2):
        if not np.all(thresholds[:, :, k] < thresholds[:, :, k+1]):
            monotonic = False
            break
    print(f"  Thresholds monotonic: {monotonic}")
    
    # Per-question statistics
    print(f"\nPer-Question Statistics:")
    print(f"  Note: Parameters are temporally dynamic, showing averages across all occurrences")
    
    # Collect parameters for each question
    question_stats = {}
    for q_id in range(min(10, n_questions)):  # Show first 10 questions
        q_mask = (questions == q_id)
        if q_mask.sum() > 0:
            q_alpha = discriminations[q_mask].mean()
            q_betas = thresholds[q_mask].mean(axis=0)
            question_stats[q_id] = {
                'count': q_mask.sum(),
                'alpha': q_alpha,
                'betas': q_betas
            }
    
    print(f"  Question | Count | α (avg) | β₀ (avg) | β₁ (avg) | β₂ (avg)")
    print(f"  ---------|-------|---------|----------|----------|----------")
    for q_id, stats in sorted(question_stats.items())[:10]:
        print(f"  {q_id:8d} | {stats['count']:5d} | {stats['alpha']:7.3f} | "
              f"{stats['betas'][0]:8.3f} | {stats['betas'][1]:8.3f} | {stats['betas'][2]:8.3f}")
    
    return params


def save_parameters(params, output_path):
    """Save extracted parameters to file."""
    # Convert numpy arrays to lists for JSON serialization
    save_dict = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value.tolist()
        else:
            save_dict[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nParameters saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract IRT parameters from trained Deep-GPCM models')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--split', default='test', choices=['train', 'test'], 
                        help='Which data split to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', help='Path to save extracted parameters (JSON)')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine model type and create model
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # Try to infer from checkpoint keys
        model_type = 'attention' if 'attention_refinement' in str(checkpoint['model_state_dict'].keys()) else 'baseline'
    
    print(f"Model type: {model_type}")
    
    # Load dataset to get dimensions
    data_dir = Path('data') / args.dataset
    train_path = data_dir / f'{args.dataset.lower()}_train.txt'
    test_path = data_dir / f'{args.dataset.lower()}_test.txt'
    
    train_data, test_data, n_questions, n_cats = load_simple_data(str(train_path), str(test_path))
    data = test_data if args.split == 'test' else train_data
    
    # Create model
    if model_type == 'attention':
        model = AttentionGPCM(n_questions=n_questions, n_cats=n_cats)
    else:
        model = DeepGPCM(n_questions=n_questions, n_cats=n_cats)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loader
    _, data_loader = create_data_loaders(train_data, test_data, batch_size=args.batch_size)
    if args.split == 'train':
        data_loader, _ = create_data_loaders(train_data, test_data, batch_size=args.batch_size)
    
    # Extract parameters
    print(f"\nExtracting IRT parameters from {args.split} set...")
    params = extract_irt_parameters(model, data_loader, device)
    
    # Analyze parameters
    analyzed_params = analyze_parameters(params)
    
    # Save if requested
    if args.save_path:
        save_parameters(analyzed_params, args.save_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()