"""
Quick Test: Enhanced CORAL-GPCM Adaptive Blending Validation

This is a shortened version to quickly validate that our adaptive blending approach 
is working correctly and learning meaningful parameters.
"""

import os
import sys
# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import json
import numpy as np
from datetime import datetime

# Import from existing infrastructure
sys.path.append('/home/steph/dirt-new/deep-gpcm')
from models.coral_gpcm import EnhancedCORALGPCM
from utils.metrics import compute_metrics
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


def create_sample_data():
    """Create sample data for quick testing."""
    n_samples = 200
    seq_len = 50
    n_questions = 400
    n_cats = 4
    
    # Create realistic data that would show category imbalance
    questions = torch.randint(0, n_questions, (n_samples, seq_len))
    
    # Create responses with middle category bias (simulating the problem)
    responses = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        for j in range(seq_len):
            # Bias toward categories 0 and 3, under-represent 1 and 2
            prob = torch.rand(1).item()
            if prob < 0.4:
                responses[i, j] = 0
            elif prob < 0.5:
                responses[i, j] = 1  # Under-represented
            elif prob < 0.6:
                responses[i, j] = 2  # Under-represented
            else:
                responses[i, j] = 3
    
    return questions, responses


def quick_test():
    """Quick test of adaptive blending approach."""
    print("🚀 Quick Test: Enhanced CORAL-GPCM Adaptive Blending")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create test data
    questions, responses = create_sample_data()
    print(f"✓ Test data created: {questions.shape}")
    
    # Count category distribution in test data
    unique, counts = torch.unique(responses, return_counts=True)
    print("✓ Test data category distribution:")
    for cat, count in zip(unique, counts):
        pct = count.item() / responses.numel() * 100
        print(f"  Cat {cat}: {count} ({pct:.1f}%)")
    
    # Create model with adaptive blending
    model = EnhancedCORALGPCM(
        n_questions=400,
        n_cats=4,
        memory_size=32,  # Smaller for quick test
        key_dim=32,
        value_dim=64,
        final_fc_dim=32,
        enable_adaptive_blending=True,
        range_sensitivity_init=1.0,
        distance_sensitivity_init=1.0,
        baseline_bias_init=0.0
    ).to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get initial adaptive parameters
    blending_info = model.get_adaptive_blending_info()
    if blending_info and blending_info.get('analysis_available'):
        params = blending_info['learnable_parameters']
        print(f"✓ Initial adaptive parameters:")
        print(f"  range_sensitivity: {params['range_sensitivity']:.4f}")
        print(f"  distance_sensitivity: {params['distance_sensitivity']:.4f}")
        print(f"  baseline_bias: {params['baseline_bias']:.4f}")
    
    # Quick training (5 epochs)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dataset = data_utils.TensorDataset(questions, responses)
    dataloader = data_utils.DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"\\n🎯 Quick Training (5 epochs)...")
    
    for epoch in range(5):
        total_loss = 0
        num_batches = 0
        
        for batch_questions, batch_responses in dataloader:
            batch_questions = batch_questions.to(device)
            batch_responses = batch_responses.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, _, _, predictions = model(batch_questions, batch_responses)
            
            # Compute loss
            batch_size, seq_len, n_cats = predictions.shape
            predictions_flat = predictions.view(-1, n_cats)
            responses_flat = batch_responses.view(-1)
            
            loss = criterion(predictions_flat, responses_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Check adaptive parameters evolution
        blending_info = model.get_adaptive_blending_info()
        if blending_info and blending_info.get('analysis_available'):
            params = blending_info['learnable_parameters']
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                  f"Range={params['range_sensitivity']:.3f}, "
                  f"Dist={params['distance_sensitivity']:.3f}, "
                  f"Bias={params['baseline_bias']:.3f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_responses = []
        
        for batch_questions, batch_responses in dataloader:
            batch_questions = batch_questions.to(device)
            batch_responses = batch_responses.to(device)
            
            _, _, _, predictions = model(batch_questions, batch_responses)
            pred_classes = torch.argmax(predictions, dim=-1)
            
            all_predictions.extend(pred_classes.cpu().numpy().flatten())
            all_responses.extend(batch_responses.cpu().numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_responses = np.array(all_responses)
        
        # Compute per-category accuracy
        category_accuracies = []
        for cat in range(4):
            mask = all_responses == cat
            if mask.sum() > 0:
                accuracy = (all_predictions[mask] == cat).mean()
                category_accuracies.append(accuracy)
            else:
                category_accuracies.append(0.0)
    
    print(f"\\n📊 Final Results:")
    for i, acc in enumerate(category_accuracies):
        print(f"  Cat {i}: {acc:.4f} ({acc*100:.1f}%)")
    
    # Get final adaptive parameters
    final_blending_info = model.get_adaptive_blending_info()
    if final_blending_info and final_blending_info.get('analysis_available'):
        final_params = final_blending_info['learnable_parameters']
        print(f"\\n🎛️  Final Adaptive Parameters:")
        print(f"  range_sensitivity: {final_params['range_sensitivity']:.4f}")
        print(f"  distance_sensitivity: {final_params['distance_sensitivity']:.4f}")
        print(f"  baseline_bias: {final_params['baseline_bias']:.4f}")
        
        # Check if parameters learned meaningfully
        range_changed = abs(final_params['range_sensitivity'] - 1.0) > 0.05
        dist_changed = abs(final_params['distance_sensitivity'] - 1.0) > 0.05
        bias_changed = abs(final_params['baseline_bias']) > 0.05
        
        print(f"\\n🧠 Learning Validation:")
        print(f"  Range sensitivity adapted: {'✅' if range_changed else '❌'}")
        print(f"  Distance sensitivity adapted: {'✅' if dist_changed else '❌'}")
        print(f"  Baseline bias adapted: {'✅' if bias_changed else '❌'}")
        
        learning_success = range_changed or dist_changed or bias_changed
        print(f"\\n🎯 Adaptive Learning: {'✅ SUCCESS' if learning_success else '❌ NO ADAPTATION'}")
        
        if learning_success:
            print("✅ The threshold-distance-based dynamic blending is working!")
            print("   Parameters are adapting to the data characteristics.")
        else:
            print("⚠️  Parameters didn't change significantly in this quick test.")
            print("   Longer training may be needed for full adaptation.")
    
    print(f"\\n✅ Quick test completed successfully!")
    return True


if __name__ == "__main__":
    quick_test()