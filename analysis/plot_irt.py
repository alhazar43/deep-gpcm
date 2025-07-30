#!/usr/bin/env python3
"""
IRT Parameter Extraction and Visualization Tool for Deep-GPCM
Comprehensive tool that extracts IRT parameters from trained models and creates visualizations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import DeepGPCM, AttentionGPCM


class IRTParameterExtractor:
    """Extract IRT parameters from trained Deep-GPCM models."""
    
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.config = checkpoint.get('config', {})
        model_type = self.config.get('model_type', 'baseline')
        n_questions = self.config.get('n_questions', 30)
        n_cats = self.config.get('n_cats', 4)
        
        # Create model based on type
        if model_type == 'baseline':
            self.model = DeepGPCM(
                n_questions=n_questions,
                n_cats=n_cats
            )
        elif model_type == 'akvmn':
            self.model = AttentionGPCM(
                n_questions=n_questions,
                n_cats=n_cats
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded {model_type} model with {n_questions} questions, {n_cats} categories")
        return self.model
    
    def extract_from_data(self, data_loader):
        """Extract IRT parameters by running model on data."""
        if self.model is None:
            self.load_model()
        
        all_thetas = []
        all_alphas = []
        all_betas = []
        all_questions = []
        
        print("Extracting IRT parameters from data...")
        with torch.no_grad():
            for batch_idx, (questions, responses, masks) in enumerate(data_loader):
                questions = questions.to(self.device)
                responses = responses.to(self.device)
                
                # Get model outputs
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = self.model(questions, responses)
                
                # Apply mask and collect valid parameters
                valid_mask = masks.bool()
                
                # Extract valid parameters
                for seq_idx in range(questions.shape[0]):
                    seq_mask = valid_mask[seq_idx]
                    seq_len = seq_mask.sum().item()
                    
                    if seq_len > 0:
                        # Student abilities (theta) - per time step
                        theta_seq = student_abilities[seq_idx, :seq_len].cpu().numpy()
                        all_thetas.extend(theta_seq)
                        
                        # Question IDs and item parameters
                        q_seq = questions[seq_idx, :seq_len].cpu().numpy()
                        alpha_seq = discrimination_params[seq_idx, :seq_len].cpu().numpy()
                        beta_seq = item_thresholds[seq_idx, :seq_len].cpu().numpy()
                        
                        all_questions.extend(q_seq)
                        all_alphas.extend(alpha_seq)
                        all_betas.extend(beta_seq)
                
                if batch_idx % 10 == 0:
                    print(f"  Processed batch {batch_idx + 1}")
        
        # Convert to numpy arrays
        all_questions = np.array(all_questions)
        all_thetas = np.array(all_thetas)
        all_alphas = np.array(all_alphas)
        all_betas = np.array(all_betas)
        
        # Aggregate item-level parameters
        n_questions = self.config.get('n_questions', 30)
        n_cats = self.config.get('n_cats', 4)
        
        item_alphas = np.zeros(n_questions)
        item_betas = np.zeros((n_questions, n_cats - 1))
        
        for q_id in range(1, n_questions + 1):  # Questions are 1-indexed
            mask = all_questions == q_id
            if mask.sum() > 0:
                item_alphas[q_id - 1] = all_alphas[mask].mean()
                item_betas[q_id - 1] = all_betas[mask].mean(axis=0)
        
        return {
            'theta': all_thetas,
            'alpha': item_alphas,
            'beta': item_betas,
            'raw_alpha': all_alphas,
            'raw_beta': all_betas,
            'questions': all_questions,
            'n_questions': n_questions,
            'n_cats': n_cats,
            'model_type': self.config.get('model_type', 'baseline')
        }
    
    def extract_network_parameters(self):
        """Extract parameters directly from model networks."""
        if self.model is None:
            self.load_model()
        
        n_questions = self.config.get('n_questions', 30)
        n_cats = self.config.get('n_cats', 4)
        
        # Access model networks based on type
        if hasattr(self.model, 'gpcm_model'):  # BaselineGPCM
            base_model = self.model.gpcm_model
        else:  # AKVMNGPCM
            base_model = self.model
        
        # Extract network weights
        with torch.no_grad():
            # Question embeddings (for reference)
            q_embeds = base_model.question_embed.weight[1:].cpu().numpy()  # Skip padding
            
            # Threshold network weights (beta parameters)
            threshold_weights = base_model.question_threshold_network[0].weight.cpu().numpy()
            threshold_bias = base_model.question_threshold_network[0].bias.cpu().numpy()
            
            # Base thresholds (could be refined with actual question embeddings)
            base_betas = threshold_bias.reshape(1, -1).repeat(n_questions, axis=0)
            
            # Discrimination network weights (for reference)
            discrim_weights = base_model.discrimination_network[0].weight.cpu().numpy()
            
            # Memory key matrix
            memory_keys = base_model.memory.key_memory_matrix.cpu().numpy()
        
        return {
            'question_embeddings': q_embeds,
            'threshold_weights': threshold_weights,
            'threshold_bias': threshold_bias,
            'base_betas': base_betas,
            'discrimination_weights': discrim_weights,
            'memory_keys': memory_keys,
            'n_questions': n_questions,
            'n_cats': n_cats
        }


class IRTVisualizer:
    """Create various IRT-related visualizations."""
    
    def __init__(self, output_dir='irt_plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def gpcm_probability(self, theta, alpha, beta):
        """Compute GPCM probabilities."""
        theta = np.asarray(theta)
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        
        if beta.ndim == 1:
            beta = beta.reshape(1, -1)
        
        K = beta.shape[1] + 1  # Number of categories
        
        # Compute cumulative logits
        cum_logits = np.zeros((*theta.shape, K))
        cum_logits[..., 0] = 0  # First category baseline
        
        # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - beta_h)
        for k in range(1, K):
            if theta.ndim == 0:  # Scalar theta
                cum_logits[k] = np.sum(alpha * (theta - beta[:k]))
            else:  # Array theta
                cum_logits[:, k] = np.sum(alpha * (theta.reshape(-1, 1) - beta[:, :k]), axis=1)
        
        # Convert to probabilities via softmax
        exp_logits = np.exp(cum_logits - np.max(cum_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs
    
    def item_information(self, theta, alpha, beta):
        """Compute item information function."""
        probs = self.gpcm_probability(theta, alpha, beta)
        
        # For GPCM: I(theta) = sum_k P_k * (d log P_k / d theta)^2
        # Simplified approximation
        info = alpha**2 * np.sum(probs * (1 - probs), axis=-1)
        return info
    
    def plot_icc(self, alpha, beta, item_indices=None, n_cats=4, title="Item Characteristic Curves"):
        """Plot Item Characteristic Curves."""
        if item_indices is None:
            item_indices = np.random.choice(len(alpha), min(6, len(alpha)), replace=False)
        
        theta_range = np.linspace(-3, 3, 100)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, item_idx in enumerate(item_indices):
            if i >= 6:
                break
                
            ax = axes[i]
            probs = self.gpcm_probability(theta_range, alpha[item_idx], beta[item_idx])
            
            for k in range(n_cats):
                ax.plot(theta_range, probs[:, k], label=f'Category {k}', linewidth=2)
            
            ax.set_title(f'Item {item_idx + 1} (α={alpha[item_idx]:.2f})')
            ax.set_xlabel('Ability (θ)')
            ax.set_ylabel('Probability')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 1)
        
        # Hide empty subplots
        for i in range(len(item_indices), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'item_characteristic_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ICC plot saved: {output_path}")
        return output_path
    
    def plot_iif(self, alpha, beta, item_indices=None, title="Item Information Functions"):
        """Plot Item Information Functions."""
        if item_indices is None:
            item_indices = np.random.choice(len(alpha), min(10, len(alpha)), replace=False)
        
        theta_range = np.linspace(-3, 3, 100)
        
        plt.figure(figsize=(12, 8))
        
        for item_idx in item_indices:
            info = self.item_information(theta_range, alpha[item_idx], beta[item_idx])
            plt.plot(theta_range, info, label=f'Item {item_idx + 1} (α={alpha[item_idx]:.2f})', linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Ability (θ)', fontsize=12)
        plt.ylabel('Information', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(-3, 3)
        
        output_path = os.path.join(self.output_dir, 'item_information_functions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 IIF plot saved: {output_path}")
        return output_path
    
    def plot_tif(self, alpha, beta, title="Test Information Function"):
        """Plot Test Information Function."""
        theta_range = np.linspace(-3, 3, 100)
        
        # Compute total information
        total_info = np.zeros_like(theta_range)
        for i in range(len(alpha)):
            info = self.item_information(theta_range, alpha[i], beta[i])
            total_info += info
        
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, total_info, linewidth=3, color='darkblue')
        plt.fill_between(theta_range, total_info, alpha=0.3, color='lightblue')
        
        # Add statistics
        max_info = np.max(total_info)
        max_theta = theta_range[np.argmax(total_info)]
        plt.axvline(max_theta, color='red', linestyle='--', alpha=0.7, label=f'Peak: θ={max_theta:.2f}')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Ability (θ)', fontsize=12)
        plt.ylabel('Test Information', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-3, 3)
        
        # Add text with statistics
        plt.text(0.02, 0.98, f'Max Info: {max_info:.2f}\nPeak at θ = {max_theta:.2f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_path = os.path.join(self.output_dir, 'test_information_function.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 TIF plot saved: {output_path}")
        return output_path
    
    def plot_wright_map(self, theta, beta, alpha, title="Wright Map (Item-Person Map)"):
        """Plot Wright Map showing item and person distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        
        # Person distribution (left side)
        ax1.hist(theta, bins=30, orientation='horizontal', alpha=0.7, color='skyblue', density=True)
        ax1.set_xlabel('Person Density')
        ax1.set_title('Person Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Item distribution (right side)
        # Use mean threshold as item difficulty
        item_difficulties = np.mean(beta, axis=1)
        
        # Create scatter plot of items
        y_positions = item_difficulties
        x_positions = alpha  # Use discrimination as x-position
        
        scatter = ax2.scatter(x_positions, y_positions, s=100, alpha=0.7, c=alpha, cmap='viridis')
        
        # Add item labels
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Discrimination (α)')
        ax2.set_title('Item Locations')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Discrimination')
        
        # Set common y-axis
        y_min = min(theta.min(), item_difficulties.min()) - 0.5
        y_max = max(theta.max(), item_difficulties.max()) + 0.5
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        ax1.set_ylabel('Ability / Difficulty (θ / β)')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'wright_map.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Wright Map saved: {output_path}")
        return output_path
    
    def plot_parameter_distributions(self, params, model_name="", title="Parameter Distributions"):
        """Plot distributions of IRT parameters."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Discrimination distribution
        ax = axes[0, 0]
        ax.hist(params['alpha'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Discrimination (α) Distribution')
        ax.set_xlabel('Discrimination')
        ax.set_ylabel('Frequency')
        ax.axvline(params['alpha'].mean(), color='red', linestyle='--', label=f'Mean: {params["alpha"].mean():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ability distribution
        ax = axes[0, 1]
        ax.hist(params['theta'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Student Ability (θ) Distribution')
        ax.set_xlabel('Ability')
        ax.set_ylabel('Frequency')
        ax.axvline(params['theta'].mean(), color='red', linestyle='--', label=f'Mean: {params["theta"].mean():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Threshold matrix (heatmap)
        ax = axes[1, 0]
        im = ax.imshow(params['beta'].T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax.set_title('Threshold Parameters (β) Matrix')
        ax.set_xlabel('Item')
        ax.set_ylabel('Threshold')
        plt.colorbar(im, ax=ax, label='Threshold Value')
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
        Model: {model_name}
        
        Discrimination (α):
        • Mean: {params['alpha'].mean():.3f}
        • Std: {params['alpha'].std():.3f}
        • Range: [{params['alpha'].min():.3f}, {params['alpha'].max():.3f}]
        
        Student Ability (θ):
        • Mean: {params['theta'].mean():.3f}
        • Std: {params['theta'].std():.3f}
        • Range: [{params['theta'].min():.3f}, {params['theta'].max():.3f}]
        
        Thresholds (β):
        • Mean: {params['beta'].mean():.3f}
        • Std: {params['beta'].std():.3f}
        • Range: [{params['beta'].min():.3f}, {params['beta'].max():.3f}]
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'parameter_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Parameter distributions saved: {output_path}")
        return output_path
    
    def plot_comparison(self, params1, params2, label1="Model 1", label2="Model 2"):
        """Compare parameters between two models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Discrimination comparison
        ax = axes[0, 0]
        ax.scatter(params1['alpha'], params2['alpha'], alpha=0.7, s=50)
        ax.plot([params1['alpha'].min(), params1['alpha'].max()], 
                [params1['alpha'].min(), params1['alpha'].max()], 'r--', alpha=0.8)
        ax.set_xlabel(f'{label1} Discrimination (α)')
        ax.set_ylabel(f'{label2} Discrimination (α)')
        ax.set_title('Discrimination Parameter Comparison')
        ax.grid(True, alpha=0.3)
        
        # Correlation
        r_alpha = np.corrcoef(params1['alpha'], params2['alpha'])[0, 1]
        ax.text(0.05, 0.95, f'r = {r_alpha:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Threshold comparison (mean thresholds)
        ax = axes[0, 1]
        beta1_mean = np.mean(params1['beta'], axis=1)
        beta2_mean = np.mean(params2['beta'], axis=1)
        ax.scatter(beta1_mean, beta2_mean, alpha=0.7, s=50)
        ax.plot([beta1_mean.min(), beta1_mean.max()], 
                [beta1_mean.min(), beta1_mean.max()], 'r--', alpha=0.8)
        ax.set_xlabel(f'{label1} Mean Threshold (β)')
        ax.set_ylabel(f'{label2} Mean Threshold (β)')
        ax.set_title('Mean Threshold Comparison')
        ax.grid(True, alpha=0.3)
        
        r_beta = np.corrcoef(beta1_mean, beta2_mean)[0, 1]
        ax.text(0.05, 0.95, f'r = {r_beta:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Ability distribution comparison
        ax = axes[1, 0]
        ax.hist(params1['theta'], bins=30, alpha=0.5, label=label1, density=True)
        ax.hist(params2['theta'], bins=30, alpha=0.5, label=label2, density=True)
        ax.set_xlabel('Student Ability (θ)')
        ax.set_ylabel('Density')
        ax.set_title('Ability Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Test Information Function comparison
        ax = axes[1, 1]
        theta_range = np.linspace(-3, 3, 100)
        
        # Compute TIF for both models
        tif1 = np.zeros_like(theta_range)
        tif2 = np.zeros_like(theta_range)
        
        for i in range(len(params1['alpha'])):
            info1 = self.item_information(theta_range, params1['alpha'][i], params1['beta'][i])
            tif1 += info1
            
        for i in range(len(params2['alpha'])):
            info2 = self.item_information(theta_range, params2['alpha'][i], params2['beta'][i])
            tif2 += info2
        
        ax.plot(theta_range, tif1, label=label1, linewidth=2)
        ax.plot(theta_range, tif2, label=label2, linewidth=2)
        ax.set_xlabel('Ability (θ)')
        ax.set_ylabel('Test Information')
        ax.set_title('Test Information Function Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{label1} vs {label2} Parameter Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model comparison saved: {output_path}")
        return output_path


def load_simple_data(data_path):
    """Load data for parameter extraction."""
    sequences = []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if i + 2 >= len(lines):
                break
            seq_len = int(lines[i].strip())
            questions = list(map(int, lines[i+1].strip().split(',')))
            responses = list(map(int, lines[i+2].strip().split(',')))
            
            questions = questions[:seq_len]
            responses = responses[:seq_len]
            
            sequences.append((questions, responses))
            i += 3
    
    return sequences


def create_data_loader(sequences, batch_size=32):
    """Create data loader from sequences."""
    import torch.utils.data as data_utils
    
    class SequenceDataset(data_utils.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def pad_sequence_batch(batch):
        questions_batch, responses_batch = zip(*batch)
        max_len = max(len(seq) for seq in questions_batch)
        
        questions_padded = []
        responses_padded = []
        masks = []
        
        for q, r in zip(questions_batch, responses_batch):
            q_len = len(q)
            q_pad = q + [0] * (max_len - q_len)
            r_pad = r + [0] * (max_len - q_len)
            mask = [1] * q_len + [0] * (max_len - q_len)
            
            questions_padded.append(q_pad)
            responses_padded.append(r_pad)
            masks.append(mask)
        
        return (torch.tensor(questions_padded), 
                torch.tensor(responses_padded), 
                torch.tensor(masks))
    
    dataset = SequenceDataset(sequences)
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_batch)
    
    return loader


def save_irt_params(params, filepath):
    """Save IRT parameters to file."""
    if filepath.endswith('.npz'):
        np.savez(filepath, **params)
    elif filepath.endswith('.json'):
        # Convert numpy arrays to lists for JSON
        json_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                json_params[key] = value.tolist()
            else:
                json_params[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_params, f, indent=2)
    
    print(f"💾 IRT parameters saved: {filepath}")


def load_irt_params(filepath):
    """Load IRT parameters from file."""
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        params = {key: data[key] for key in data.files}
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            json_params = json.load(f)
        
        # Convert lists back to numpy arrays
        params = {}
        for key, value in json_params.items():
            if isinstance(value, list):
                params[key] = np.array(value)
            else:
                params[key] = value
    
    return params


def main():
    parser = argparse.ArgumentParser(description='IRT Parameter Extraction and Visualization for Deep-GPCM')
    
    # Input options
    parser.add_argument('--model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--irt_file', type=str, help='Path to saved IRT parameters file')
    parser.add_argument('--data_path', type=str, default='data/synthetic_OC/test.txt', help='Path to test data')
    
    # Comparison mode
    parser.add_argument('--compare', nargs=2, metavar=('MODEL1', 'MODEL2'), help='Compare two models')
    parser.add_argument('--labels', nargs=2, default=['Baseline', 'AKVMN'], help='Labels for comparison')
    
    # Plot options
    parser.add_argument('--plot_type', choices=['icc', 'iif', 'tif', 'wright', 'dist', 'all'], 
                        default='all', help='Type of plots to generate')
    parser.add_argument('--items', type=int, nargs='+', help='Specific item indices to plot')
    parser.add_argument('--output_dir', default='irt_plots', help='Output directory for plots')
    
    # Extraction options
    parser.add_argument('--extract', action='store_true', help='Extract IRT parameters from model')
    parser.add_argument('--save_params', type=str, help='Save extracted parameters to file')
    
    # Device
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IRT PARAMETER EXTRACTION AND VISUALIZATION")
    print("=" * 80)
    
    # Initialize visualizer
    visualizer = IRTVisualizer(args.output_dir)
    
    if args.compare:
        # Comparison mode
        print(f"🔄 Comparing models: {args.compare[0]} vs {args.compare[1]}")
        
        # Load test data
        sequences = load_simple_data(args.data_path)
        data_loader = create_data_loader(sequences, batch_size=32)
        
        # Extract parameters from both models
        extractor1 = IRTParameterExtractor(args.compare[0], args.device)
        params1 = extractor1.extract_from_data(data_loader)
        
        extractor2 = IRTParameterExtractor(args.compare[1], args.device)
        params2 = extractor2.extract_from_data(data_loader)
        
        # Create comparison plots
        visualizer.plot_comparison(params1, params2, args.labels[0], args.labels[1])
        
        # Individual plots for each model
        visualizer.plot_parameter_distributions(params1, args.labels[0], f"{args.labels[0]} Parameter Distributions")
        visualizer.plot_parameter_distributions(params2, args.labels[1], f"{args.labels[1]} Parameter Distributions")
        
    elif args.irt_file:
        # Load from saved file
        print(f"📂 Loading IRT parameters from: {args.irt_file}")
        params = load_irt_params(args.irt_file)
        
        # Generate plots
        if args.plot_type in ['all', 'dist']:
            visualizer.plot_parameter_distributions(params, params.get('model_type', ''))
        
        if args.plot_type in ['all', 'icc']:
            visualizer.plot_icc(params['alpha'], params['beta'], args.items, params.get('n_cats', 4))
        
        if args.plot_type in ['all', 'iif']:
            visualizer.plot_iif(params['alpha'], params['beta'], args.items)
        
        if args.plot_type in ['all', 'tif']:
            visualizer.plot_tif(params['alpha'], params['beta'])
        
        if args.plot_type in ['all', 'wright']:
            visualizer.plot_wright_map(params['theta'], params['beta'], params['alpha'])
    
    elif args.model_path:
        # Extract from model
        print(f"🧠 Extracting IRT parameters from: {args.model_path}")
        
        # Load test data
        sequences = load_simple_data(args.data_path)
        data_loader = create_data_loader(sequences, batch_size=32)
        
        # Extract parameters
        extractor = IRTParameterExtractor(args.model_path, args.device)
        params = extractor.extract_from_data(data_loader)
        
        # Save parameters if requested
        if args.save_params:
            save_irt_params(params, args.save_params)
        
        # Generate plots
        model_name = params.get('model_type', 'Unknown')
        
        if args.plot_type in ['all', 'dist']:
            visualizer.plot_parameter_distributions(params, model_name)
        
        if args.plot_type in ['all', 'icc']:
            visualizer.plot_icc(params['alpha'], params['beta'], args.items, params.get('n_cats', 4))
        
        if args.plot_type in ['all', 'iif']:
            visualizer.plot_iif(params['alpha'], params['beta'], args.items)
        
        if args.plot_type in ['all', 'tif']:
            visualizer.plot_tif(params['alpha'], params['beta'])
        
        if args.plot_type in ['all', 'wright']:
            visualizer.plot_wright_map(params['theta'], params['beta'], params['alpha'])
    
    else:
        print("❌ Error: Must provide either --model_path, --irt_file, or --compare")
        return 1
    
    print(f"\n✅ IRT analysis completed! Plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())