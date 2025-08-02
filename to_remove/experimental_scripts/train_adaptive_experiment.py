"""
Phase 2 Experimental Validation: Enhanced CORAL-GPCM with Adaptive Blending

This script trains the enhanced_coral_gpcm_adaptive model with the exact same configuration
as the baseline coral_gpcm to test if threshold-distance-based dynamic blending can solve
the middle category prediction imbalance.

Target Improvements:
- Cat 1: From 28.11% to >35.14% (+25% improvement)
- Cat 2: From 27.37% to >35.58% (+30% improvement)
- Maintain Cat 0 and Cat 3 performance (>80% of baseline)
"""

import os
import sys
# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import json
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

# Import from existing infrastructure
sys.path.append('/home/steph/dirt-new/deep-gpcm')
from models.coral_gpcm import EnhancedCORALGPCM
from utils.metrics import compute_metrics, save_results, ensure_results_dirs
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


def load_data():
    """Load synthetic_OC dataset matching baseline configuration."""
    train_path = 'data/synthetic_OC/synthetic_oc_train.txt'
    test_path = 'data/synthetic_OC/synthetic_oc_test.txt'
    
    def read_data(file_path):
        sequences = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i + 2 >= len(lines):
                    break
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(int, lines[i+2].strip().split(',')))
                
                # Ensure lengths match
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    print(f"✓ Loaded synthetic_OC dataset:")
    print(f"  - Training sequences: {len(train_data)}")
    print(f"  - Test sequences: {len(test_data)}")
    
    return train_data, test_data


def pad_sequences(sequences, max_len=None):
    """Pad sequences to uniform length."""
    if max_len is None:
        max_len = max(len(seq[0]) for seq in sequences)
    
    questions = []
    responses = []
    
    for q_seq, r_seq in sequences:
        q_padded = q_seq + [0] * (max_len - len(q_seq))
        r_padded = r_seq + [0] * (max_len - len(r_seq))
        questions.append(q_padded)
        responses.append(r_padded)
    
    return torch.tensor(questions, dtype=torch.long), torch.tensor(responses, dtype=torch.long)


def create_model():
    """Create EnhancedCORALGPCM with adaptive blending enabled."""
    print("🚀 Creating Enhanced CORAL-GPCM with Adaptive Blending")
    print("="*60)
    
    model = EnhancedCORALGPCM(
        n_questions=400,  # Match baseline
        n_cats=4,         # Match baseline
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy="linear_decay",
        ability_scale=1.0,
        use_discrimination=True,
        dropout_rate=0.0,
        # Enhanced features
        enable_threshold_coupling=True,
        coupling_type="linear",
        gpcm_weight=0.7,
        coral_weight=0.3,
        # *** ADAPTIVE BLENDING ENABLED ***
        enable_adaptive_blending=True,
        blend_weight=0.5,  # Fallback weight
        range_sensitivity_init=1.0,
        distance_sensitivity_init=1.0,
        baseline_bias_init=0.0
    )
    
    print(f"✓ Model Created:")
    print(f"  - Model name: {model.model_name}")
    print(f"  - Model version: {model.model_version}")
    print(f"  - Adaptive blending: {model.enable_adaptive_blending}")
    print(f"  - Threshold coupling: {model.enable_threshold_coupling}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get initial adaptive blending info
    blending_info = model.get_adaptive_blending_info()
    if blending_info and blending_info.get('analysis_available', False):
        params = blending_info['learnable_parameters']
        print(f"  - Initial adaptive parameters:")
        print(f"    * range_sensitivity: {params['range_sensitivity']:.4f}")
        print(f"    * distance_sensitivity: {params['distance_sensitivity']:.4f}")
        print(f"    * baseline_bias: {params['baseline_bias']:.4f}")
    
    return model


def train_single_fold(model, train_data, test_data, fold_num, device):
    """Train model for single fold."""
    print(f"\\n🎯 Training Fold {fold_num}")
    print("-" * 40)
    
    # Prepare data
    max_len = 400  # Reasonable max length
    train_questions, train_responses = pad_sequences(train_data, max_len)
    test_questions, test_responses = pad_sequences(test_data, max_len)
    
    print(f"✓ Data prepared:")
    print(f"  - Training shape: {train_questions.shape}")
    print(f"  - Test shape: {test_questions.shape}")
    print(f"  - Sequence length: {max_len}")
    
    # Create dataloaders
    train_dataset = data_utils.TensorDataset(train_questions, train_responses)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = data_utils.TensorDataset(test_questions, test_responses)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Setup training
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # Training loop
    epochs = 30  # Match baseline
    best_accuracy = 0
    best_epoch = 0
    training_history = []
    
    print(f"\\n📚 Starting Training (30 epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_questions, batch_responses in train_loader:
            batch_questions = batch_questions.to(device)
            batch_responses = batch_responses.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, _, _, predictions = model(batch_questions, batch_responses)
            
            # Compute loss (flatten for cross entropy)
            batch_size, seq_len, n_cats = predictions.shape
            predictions_flat = predictions.view(-1, n_cats)
            responses_flat = batch_responses.view(-1)
            
            loss = criterion(predictions_flat, responses_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_responses = []
            
            for batch_questions, batch_responses in test_loader:
                batch_questions = batch_questions.to(device)
                batch_responses = batch_responses.to(device)
                
                _, _, _, predictions = model(batch_questions, batch_responses)
                
                # Get predicted classes
                pred_classes = torch.argmax(predictions, dim=-1)
                
                # Remove padding (responses == 0)
                mask = batch_responses != 0
                pred_classes_masked = pred_classes[mask]
                responses_masked = batch_responses[mask]
                
                all_predictions.extend(pred_classes_masked.cpu().numpy())
                all_responses.extend(responses_masked.cpu().numpy())
            
            # Compute metrics
            all_predictions = np.array(all_predictions)
            all_responses = np.array(all_responses)
            
            metrics = compute_metrics(all_responses, all_predictions, n_cats=4)
            
            accuracy = metrics['categorical_accuracy']
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'test_accuracy': accuracy,
                'cat_0_accuracy': metrics['cat_0_accuracy'],
                'cat_1_accuracy': metrics['cat_1_accuracy'],
                'cat_2_accuracy': metrics['cat_2_accuracy'],
                'cat_3_accuracy': metrics['cat_3_accuracy']
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                # Save best model
                torch.save(model.state_dict(), f'save_models/enhanced_coral_gpcm_adaptive_synthetic_OC_fold_{fold_num}.pth')
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:2d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f} "
                      f"[Cat1: {metrics['cat_1_accuracy']:.3f}, Cat2: {metrics['cat_2_accuracy']:.3f}]")
                
                # Show adaptive parameter evolution
                if model.enable_adaptive_blending:
                    blending_info = model.get_adaptive_blending_info()
                    if blending_info and blending_info.get('analysis_available'):
                        params = blending_info['learnable_parameters']
                        print(f"    Adaptive params: range={params['range_sensitivity']:.3f}, "
                              f"dist={params['distance_sensitivity']:.3f}, bias={params['baseline_bias']:.3f}")
    
    print(f"\\n✅ Fold {fold_num} completed!")
    print(f"   Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    
    # Final evaluation with best model
    model.load_state_dict(torch.load(f'save_models/enhanced_coral_gpcm_adaptive_synthetic_OC_fold_{fold_num}.pth'))
    model.eval()
    
    with torch.no_grad():
        all_predictions = []
        all_responses = []
        
        for batch_questions, batch_responses in test_loader:
            batch_questions = batch_questions.to(device)
            batch_responses = batch_responses.to(device)
            
            _, _, _, predictions = model(batch_questions, batch_responses)
            pred_classes = torch.argmax(predictions, dim=-1)
            
            mask = batch_responses != 0
            pred_classes_masked = pred_classes[mask]
            responses_masked = batch_responses[mask]
            
            all_predictions.extend(pred_classes_masked.cpu().numpy())
            all_responses.extend(responses_masked.cpu().numpy())
        
        final_metrics = compute_metrics(np.array(all_responses), np.array(all_predictions), n_cats=4)
        
        # Get final adaptive blending analysis
        adaptive_analysis = None
        if model.enable_adaptive_blending:
            adaptive_analysis = model.get_adaptive_blending_info()
    
    return final_metrics, training_history, adaptive_analysis


def main():
    print("🔬 Phase 2: Experimental Validation - Enhanced CORAL-GPCM Adaptive Blending")
    print("=" * 80)
    print("Target: Solve middle category prediction imbalance")
    print("Baseline: Cat 1: 28.11% → >35.14% (+25%), Cat 2: 27.37% → >35.58% (+30%)")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    train_data, test_data = load_data()
    
    # Create model
    model = create_model()
    
    # Run single fold experiment (for now)
    print(f"\\n🧪 Running Single Fold Experiment...")
    final_metrics, training_history, adaptive_analysis = train_single_fold(
        model, train_data, test_data, fold_num=1, device=device
    )
    
    # Results analysis
    print("\\n" + "="*80)
    print("📊 EXPERIMENTAL RESULTS ANALYSIS")
    print("="*80)
    
    print(f"\\n🎯 Per-Category Performance:")
    print(f"  Cat 0: {final_metrics['cat_0_accuracy']:.4f} ({final_metrics['cat_0_accuracy']*100:.2f}%)")
    print(f"  Cat 1: {final_metrics['cat_1_accuracy']:.4f} ({final_metrics['cat_1_accuracy']*100:.2f}%) {'✅' if final_metrics['cat_1_accuracy'] > 0.3514 else '❌'}")
    print(f"  Cat 2: {final_metrics['cat_2_accuracy']:.4f} ({final_metrics['cat_2_accuracy']*100:.2f}%) {'✅' if final_metrics['cat_2_accuracy'] > 0.3558 else '❌'}")
    print(f"  Cat 3: {final_metrics['cat_3_accuracy']:.4f} ({final_metrics['cat_3_accuracy']*100:.2f}%)")
    
    # Compare against baseline
    baseline = {'cat_0': 0.7610, 'cat_1': 0.2811, 'cat_2': 0.2737, 'cat_3': 0.7231}
    
    print(f"\\n📈 Improvement Analysis:")
    for i in range(4):
        current = final_metrics[f'cat_{i}_accuracy']
        baseline_val = baseline[f'cat_{i}']
        improvement = (current - baseline_val) / baseline_val * 100
        print(f"  Cat {i}: {improvement:+.2f}% change ({'✅' if improvement > 0 else '❌'})")
    
    # Check success criteria
    cat1_target = baseline['cat_1'] * 1.25  # +25%
    cat2_target = baseline['cat_2'] * 1.30  # +30%
    
    cat1_success = final_metrics['cat_1_accuracy'] >= cat1_target
    cat2_success = final_metrics['cat_2_accuracy'] >= cat2_target
    
    print(f"\\n🎯 Success Criteria Check:")
    print(f"  Cat 1 target: ≥{cat1_target:.4f} → {'✅ ACHIEVED' if cat1_success else '❌ MISSED'}")
    print(f"  Cat 2 target: ≥{cat2_target:.4f} → {'✅ ACHIEVED' if cat2_success else '❌ MISSED'}")
    
    # Overall metrics
    print(f"\\n📊 Overall Performance:")
    print(f"  Categorical Accuracy: {final_metrics['categorical_accuracy']:.4f}")
    print(f"  Ordinal Accuracy: {final_metrics['ordinal_accuracy']:.4f}")
    print(f"  QWK: {final_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  MAE: {final_metrics['mean_absolute_error']:.4f}")
    
    # Adaptive blending analysis
    if adaptive_analysis and adaptive_analysis.get('analysis_available'):
        print(f"\\n🎛️  Final Adaptive Parameters:")
        params = adaptive_analysis['learnable_parameters']
        print(f"  Range sensitivity: {params['range_sensitivity']:.4f}")
        print(f"  Distance sensitivity: {params['distance_sensitivity']:.4f}")
        print(f"  Baseline bias: {params['baseline_bias']:.4f}")
        
        # Analyze if parameters learned meaningful values
        range_meaningful = abs(params['range_sensitivity'] - 1.0) > 0.1
        dist_meaningful = abs(params['distance_sensitivity'] - 1.0) > 0.1
        bias_meaningful = abs(params['baseline_bias']) > 0.1
        
        print(f"\\n🧠 Parameter Learning Analysis:")
        print(f"  Range sensitivity learned: {'✅' if range_meaningful else '❌'}")
        print(f"  Distance sensitivity learned: {'✅' if dist_meaningful else '❌'}")
        print(f"  Baseline bias learned: {'✅' if bias_meaningful else '❌'}")
    
    # Save results
    results = {
        'model_type': 'enhanced_coral_gpcm_adaptive',
        'model_version': model.model_version,
        'final_metrics': final_metrics,
        'training_history': training_history,
        'adaptive_analysis': adaptive_analysis,
        'success_criteria': {
            'cat_1_target': cat1_target,
            'cat_1_achieved': cat1_success,
            'cat_2_target': cat2_target,
            'cat_2_achieved': cat2_success
        },
        'timestamp': datetime.now().isoformat()
    }
    
    ensure_results_dirs()
    with open('results/test/test_results_enhanced_coral_gpcm_adaptive_synthetic_OC.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: results/test/test_results_enhanced_coral_gpcm_adaptive_synthetic_OC.json")
    
    # Final verdict
    success = cat1_success and cat2_success
    print(f"\\n🏆 EXPERIMENTAL OUTCOME: {'SUCCESS!' if success else 'NEEDS REFINEMENT'}")
    
    if success:
        print("🎉 Threshold-distance-based dynamic blending successfully solved middle category imbalance!")
    else:
        print("🔬 Further parameter tuning or architectural adjustments may be needed.")
    
    return success


if __name__ == "__main__":
    success = main()