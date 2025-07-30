#!/usr/bin/env python3
"""
All-in-one IRT analysis tool that automatically:
1. Detects trained models
2. Extracts IRT parameters
3. Compares with true parameters
4. Generates plots and summaries
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from glob import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import DeepGPCM, AttentionGPCM
from train import load_simple_data, create_data_loaders


class IRTAnalyzer:
    """Unified IRT analysis tool."""
    
    def __init__(self, dataset='synthetic_OC', output_dir='results/irt_analysis'):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data_dir = Path('data') / dataset
        train_path = data_dir / f'{dataset.lower()}_train.txt'
        test_path = data_dir / f'{dataset.lower()}_test.txt'
        
        self.train_data, self.test_data, self.n_questions, self.n_cats = load_simple_data(
            str(train_path), str(test_path)
        )
        
        # Load true parameters if available
        self.true_params = self._load_true_parameters(data_dir)
        
    def _load_true_parameters(self, dataset_path):
        """Load true IRT parameters from synthetic dataset."""
        true_params_path = dataset_path / 'true_irt_parameters.json'
        
        if not true_params_path.exists():
            print(f"Warning: No true parameters found at {true_params_path}")
            return None
        
        with open(true_params_path, 'r') as f:
            return json.load(f)
    
    def find_models(self):
        """Automatically find all trained models for the dataset."""
        model_pattern = f"save_models/best_*_{self.dataset}.pth"
        model_files = glob(model_pattern)
        
        # Filter out fold-specific models
        model_files = [f for f in model_files if not any(f'fold_{i}' in f for i in range(1, 10))]
        
        models = {}
        for model_path in model_files:
            # Extract model name from path
            base_name = os.path.basename(model_path)
            model_name = base_name.replace(f'best_', '').replace(f'_{self.dataset}.pth', '')
            models[model_name] = model_path
            
        print(f"Found {len(models)} models: {list(models.keys())}")
        return models
    
    def extract_parameters(self, model_path, split='test'):
        """Extract IRT parameters from a model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine model type
        model_type = 'attention' if 'attention_refinement' in str(checkpoint['model_state_dict'].keys()) else 'baseline'
        
        # Create model
        if model_type == 'attention':
            model = AttentionGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        else:
            model = DeepGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        
        # Load weights with proper handling for wrapped models
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # Handle models saved with wrapper
            if 'gpcm_model' in str(e):
                # Extract the wrapped model state dict
                wrapped_state_dict = checkpoint['model_state_dict']
                unwrapped_state_dict = {}
                for key, value in wrapped_state_dict.items():
                    if key.startswith('gpcm_model.'):
                        new_key = key.replace('gpcm_model.', '')
                        unwrapped_state_dict[new_key] = value
                    else:
                        unwrapped_state_dict[key] = value
                model.load_state_dict(unwrapped_state_dict)
            else:
                raise e
        
        model.to(device)
        model.eval()
        
        # Create data loader
        data = self.test_data if split == 'test' else self.train_data
        if split == 'train':
            data_loader, _ = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        else:
            _, data_loader = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        
        # Extract parameters
        all_abilities = []
        all_discriminations = []
        all_thresholds = []
        all_questions = []
        all_masks = []
        
        # Find max sequence length
        max_seq_len = 0
        for _, (questions, _, _) in enumerate(data_loader):
            max_seq_len = max(max_seq_len, questions.shape[1])
        
        with torch.no_grad():
            for batch_idx, (questions, responses, mask) in enumerate(data_loader):
                questions = questions.to(device)
                responses = responses.to(device)
                batch_size, seq_len = questions.shape
                
                # Forward pass
                student_abilities, item_thresholds, discrimination_params, _ = model(questions, responses)
                
                # Pad to max length
                if seq_len < max_seq_len:
                    pad_len = max_seq_len - seq_len
                    student_abilities = F.pad(student_abilities, (0, pad_len), value=0)
                    discrimination_params = F.pad(discrimination_params, (0, pad_len), value=0)
                    item_thresholds = F.pad(item_thresholds, (0, 0, 0, pad_len), value=0)
                    questions = F.pad(questions, (0, pad_len), value=0)
                    mask = F.pad(mask, (0, pad_len), value=0)
                
                all_abilities.append(student_abilities.cpu().numpy())
                all_discriminations.append(discrimination_params.cpu().numpy())
                all_thresholds.append(item_thresholds.cpu().numpy())
                all_questions.append(questions.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
        
        # Concatenate all batches
        return {
            'student_abilities': np.concatenate(all_abilities, axis=0),
            'item_discriminations': np.concatenate(all_discriminations, axis=0),
            'item_thresholds': np.concatenate(all_thresholds, axis=0),
            'question_ids': np.concatenate(all_questions, axis=0),
            'masks': np.concatenate(all_masks, axis=0),
            'model_type': model_type
        }
    
    def extract_item_parameters(self, params):
        """Extract average item parameters from temporal parameters."""
        questions = params['question_ids']
        discriminations = params['item_discriminations']
        thresholds = params['item_thresholds']
        masks = params['masks']
        
        n_questions = int(np.max(questions)) + 1
        n_cats = thresholds.shape[-1] + 1
        
        # Calculate average parameters for each question
        item_alphas = np.zeros(n_questions)
        item_betas = np.zeros((n_questions, n_cats - 1))
        item_counts = np.zeros(n_questions)
        
        for i in range(questions.shape[0]):
            for j in range(questions.shape[1]):
                if masks[i, j] > 0 and questions[i, j] > 0:
                    q_id = questions[i, j]
                    item_alphas[q_id] += discriminations[i, j]
                    item_betas[q_id] += thresholds[i, j]
                    item_counts[q_id] += 1
        
        # Average
        for q_id in range(n_questions):
            if item_counts[q_id] > 0:
                item_alphas[q_id] /= item_counts[q_id]
                item_betas[q_id] /= item_counts[q_id]
        
        return item_alphas, item_betas, item_counts
    
    def normalize_parameters(self, alphas, betas, prior_alpha_mean=1.0, prior_alpha_std=0.5,
                           prior_beta_mean=0.0, prior_beta_std=1.0):
        """Normalize parameters to standard IRT scale."""
        # Normalize discriminations (log-normal prior)
        alphas_norm = np.exp((np.log(alphas + 1e-6) - np.mean(np.log(alphas + 1e-6))) / 
                            np.std(np.log(alphas + 1e-6)) * prior_alpha_std + np.log(prior_alpha_mean))
        
        # Normalize thresholds (normal prior)
        betas_norm = np.zeros_like(betas)
        for k in range(betas.shape[1]):
            betas_norm[:, k] = (betas[:, k] - np.mean(betas[:, k])) / np.std(betas[:, k]) * prior_beta_std + prior_beta_mean
        
        return alphas_norm, betas_norm
    
    def calculate_correlations(self, true_params, learned_params):
        """Calculate correlations between true and learned parameters."""
        # Extract average item parameters
        learned_alphas, learned_betas, item_counts = self.extract_item_parameters(learned_params)
        
        # Get true parameters
        true_alphas = np.array(true_params['question_params']['discrimination']['alpha'])
        true_betas = np.array(true_params['question_params']['difficulties']['beta'])
        
        # Only consider items seen in data
        valid_items = item_counts > 0
        valid_indices = np.where(valid_items)[0]
        
        # Normalize parameters
        true_alphas_norm, true_betas_norm = self.normalize_parameters(true_alphas, true_betas)
        learned_alphas_norm, learned_betas_norm = self.normalize_parameters(
            learned_alphas[valid_items], learned_betas[valid_items]
        )
        
        # Calculate correlations
        alpha_corr = np.corrcoef(true_alphas_norm[valid_indices], learned_alphas_norm)[0, 1]
        beta_corrs = [np.corrcoef(true_betas_norm[valid_indices, k], learned_betas_norm[:, k])[0, 1] 
                     for k in range(3)]
        
        return {
            'alpha': alpha_corr,
            'betas': beta_corrs,
            'n_items': len(valid_indices),
            'true_alphas_norm': true_alphas_norm,
            'true_betas_norm': true_betas_norm,
            'learned_alphas_norm': learned_alphas_norm,
            'learned_betas_norm': learned_betas_norm,
            'valid_indices': valid_indices
        }
    
    def plot_comparison(self, model_results, save_path):
        """Create comparison plots for all models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models, 4, figsize=(16, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(model_results.items()):
            corr_data = results['correlations']
            
            # Discrimination plot
            ax = axes[idx, 0]
            ax.scatter(corr_data['true_alphas_norm'][corr_data['valid_indices']], 
                      corr_data['learned_alphas_norm'], alpha=0.6)
            ax.plot([0, 3], [0, 3], 'r--', label='Perfect recovery')
            ax.set_xlabel('True α')
            ax.set_ylabel('Learned α')
            ax.set_title(f'{model_name.upper()}\nDiscrimination (r={corr_data["alpha"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Threshold plots
            for k in range(3):
                ax = axes[idx, k + 1]
                ax.scatter(corr_data['true_betas_norm'][corr_data['valid_indices'], k], 
                          corr_data['learned_betas_norm'][:, k], alpha=0.6)
                ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
                ax.set_xlabel(f'True β_{k}')
                ax.set_ylabel(f'Learned β_{k}')
                ax.set_title(f'Threshold {k} (r={corr_data["betas"][k]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('IRT Parameter Recovery Comparison\n' +
                    'Parameters normalized with standard IRT priors: α~LogN(1,0.5), β~N(0,1)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_summary(self, model_results):
        """Generate summary report."""
        summary_path = self.output_dir / 'irt_analysis_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("IRT PARAMETER RECOVERY ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Models analyzed: {len(model_results)}\n")
            f.write("\nNormalization applied:\n")
            f.write("- Discrimination (α): Log-normal prior with mean=1.0, std=0.5\n")
            f.write("- Thresholds (β): Normal prior with mean=0.0, std=1.0\n")
            f.write("\n" + "-"*80 + "\n")
            
            # Detailed results for each model
            for model_name, results in model_results.items():
                params = results['parameters']
                corr = results['correlations']
                
                f.write(f"\nMODEL: {model_name.upper()}\n")
                f.write(f"  Type: {params['model_type']}\n")
                f.write(f"  Items analyzed: {corr['n_items']}/{self.n_questions}\n")
                f.write(f"\n  Parameter correlations with true values:\n")
                f.write(f"    Discrimination (α): {corr['alpha']:.4f}\n")
                f.write(f"    Threshold β₀: {corr['betas'][0]:.4f}\n")
                f.write(f"    Threshold β₁: {corr['betas'][1]:.4f}\n")
                f.write(f"    Threshold β₂: {corr['betas'][2]:.4f}\n")
                f.write(f"    Average β correlation: {np.mean(corr['betas']):.4f}\n")
                
                # Parameter statistics
                abilities = params['student_abilities']
                f.write(f"\n  Student ability statistics:\n")
                f.write(f"    Range: [{abilities.min():.3f}, {abilities.max():.3f}]\n")
                f.write(f"    Mean: {abilities.mean():.3f}, Std: {abilities.std():.3f}\n")
                
                discriminations = params['item_discriminations']
                f.write(f"\n  Discrimination statistics:\n")
                f.write(f"    Range: [{discriminations.min():.3f}, {discriminations.max():.3f}]\n")
                f.write(f"    Mean: {discriminations.mean():.3f}, Std: {discriminations.std():.3f}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("\nINTERPRETATION:\n")
            f.write("- Correlations measure how well the model recovers true IRT parameters\n")
            f.write("- Values closer to 1.0 indicate better parameter recovery\n")
            f.write("- Deep-GPCM learns dynamic parameters that change over time\n")
            f.write("- This temporal adaptation is a key difference from traditional static IRT\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary saved to: {summary_path}")
    
    def run_analysis(self):
        """Run complete IRT analysis pipeline."""
        print("\n" + "="*80)
        print("AUTOMATED IRT ANALYSIS PIPELINE")
        print("="*80)
        
        # Find models
        models = self.find_models()
        if not models:
            print("No trained models found!")
            return
        
        # Extract parameters from each model
        model_results = {}
        
        for model_name, model_path in models.items():
            print(f"\nProcessing {model_name}...")
            
            try:
                # Extract parameters
                params = self.extract_parameters(model_path)
                
                # Save parameters
                param_file = self.output_dir / f'irt_params_{model_name}.json'
                save_params = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in params.items()}
                with open(param_file, 'w') as f:
                    json.dump(save_params, f, indent=2)
                print(f"  Parameters saved to: {param_file}")
                
                # Calculate correlations if true parameters available
                if self.true_params:
                    correlations = self.calculate_correlations(self.true_params, params)
                    model_results[model_name] = {
                        'parameters': params,
                        'correlations': correlations
                    }
                    print(f"  Correlations calculated: α={correlations['alpha']:.3f}, " +
                          f"β_avg={np.mean(correlations['betas']):.3f}")
                else:
                    model_results[model_name] = {'parameters': params}
            except Exception as e:
                print(f"  Error processing {model_name}: {str(e)}")
                print(f"  Skipping this model...")
        
        # Create comparison plots
        if self.true_params and model_results:
            plot_path = self.output_dir / 'irt_parameter_recovery_comparison.png'
            self.plot_comparison(model_results, plot_path)
            print(f"\nComparison plot saved to: {plot_path}")
        
        # Generate summary
        if model_results:
            self.generate_summary(model_results)
        
        print("\nAnalysis complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='All-in-one IRT analysis tool')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--output_dir', default='results/irt_analysis', 
                        help='Output directory for results')
    parser.add_argument('--split', default='test', choices=['train', 'test'],
                        help='Which data split to analyze')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = IRTAnalyzer(dataset=args.dataset, output_dir=args.output_dir)
    
    # Run analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main()