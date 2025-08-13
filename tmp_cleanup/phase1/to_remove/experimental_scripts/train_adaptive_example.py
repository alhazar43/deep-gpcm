"""
Example Training Script for Adaptive Blending Enhancement

This script demonstrates how to safely use the new threshold-distance-based 
dynamic blending feature without affecting existing trained models.

Usage:
  # Train with adaptive blending (NEW feature)
  python train_adaptive_example.py --enable_adaptive_blending
  
  # Train without adaptive blending (EXISTING behavior, default)
  python train_adaptive_example.py

Key Safety Features:
- Adaptive blending is OFF by default (enable_adaptive_blending=False)
- Existing training scripts remain unchanged
- Model versioning distinguishes adaptive vs standard models
- Separate model names prevent checkpoint conflicts
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.coral_gpcm import EnhancedCORALGPCM


def create_model(args):
    """Create model with optional adaptive blending."""
    print(f"🏗️  Creating {'ADAPTIVE' if args.enable_adaptive_blending else 'STANDARD'} EnhancedCORALGPCM model")
    
    model = EnhancedCORALGPCM(
        n_questions=args.n_questions,
        n_cats=args.n_cats,
        memory_size=args.memory_size,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        final_fc_dim=args.final_fc_dim,
        # Adaptive blending configuration (NEW)
        enable_adaptive_blending=args.enable_adaptive_blending,
        range_sensitivity_init=args.range_sensitivity_init,
        distance_sensitivity_init=args.distance_sensitivity_init,
        baseline_bias_init=args.baseline_bias_init,
        # Standard configuration (EXISTING)
        enable_threshold_coupling=True,
        blend_weight=0.5
    )
    
    print(f"✓ Model created:")
    print(f"  - Model name: {model.model_name}")
    print(f"  - Model version: {model.model_version}")
    print(f"  - Adaptive blending: {model.enable_adaptive_blending}")
    print(f"  - Threshold blender: {'Enabled' if model.threshold_blender else 'Disabled'}")
    
    if args.enable_adaptive_blending:
        # Get adaptive blending info
        blending_info = model.get_adaptive_blending_info()
        params = blending_info.get('learnable_parameters', {})
        print(f"  - Learnable parameters: {params}")
    
    return model


def get_save_path(args):
    """Generate safe save path that won't overwrite existing models."""
    base_name = f"enhanced_coral_gpcm_{args.dataset}"
    
    if args.enable_adaptive_blending:
        # Use different naming to avoid conflicts
        save_name = f"{base_name}_adaptive_v2.pth"
        print(f"💾 Adaptive model will save to: {save_name}")
    else:
        # Use standard naming (compatible with existing workflows)
        save_name = f"{base_name}_standard.pth"
        print(f"💾 Standard model will save to: {save_name}")
    
    return f"save_models/{save_name}"


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (questions, responses) in enumerate(dataloader):
        questions, responses = questions.to(device), responses.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        student_abilities, item_thresholds, discrimination_params, predictions = model(questions, responses)
        
        # Reshape for loss computation: (batch_size, seq_len, n_cats) -> (batch_size * seq_len, n_cats)
        batch_size, seq_len, n_cats = predictions.shape
        predictions_flat = predictions.view(-1, n_cats)  # (batch_size * seq_len, n_cats)
        responses_flat = responses.view(-1)  # (batch_size * seq_len,)
        
        # Compute loss
        loss = criterion(predictions_flat, responses_flat)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.6f}")
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train EnhancedCORALGPCM with optional adaptive blending')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--n_questions', type=int, default=50, help='Number of questions')
    parser.add_argument('--n_cats', type=int, default=4, help='Number of categories')
    
    # Model parameters
    parser.add_argument('--memory_size', type=int, default=50, help='Memory size')
    parser.add_argument('--key_dim', type=int, default=50, help='Key dimension')
    parser.add_argument('--value_dim', type=int, default=200, help='Value dimension')
    parser.add_argument('--final_fc_dim', type=int, default=50, help='Final FC dimension')
    
    # Adaptive blending parameters (NEW)
    parser.add_argument('--enable_adaptive_blending', action='store_true', 
                       help='Enable threshold-distance-based adaptive blending (NEW FEATURE)')
    parser.add_argument('--range_sensitivity_init', type=float, default=1.0,
                       help='Initial range sensitivity parameter')
    parser.add_argument('--distance_sensitivity_init', type=float, default=1.0,
                       help='Initial distance sensitivity parameter')
    parser.add_argument('--baseline_bias_init', type=float, default=0.0,
                       help='Initial baseline bias parameter')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 Enhanced CORAL-GPCM Training with Adaptive Blending Support")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Device: {device}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Adaptive blending: {'🟢 ENABLED' if args.enable_adaptive_blending else '🔴 DISABLED (default)'}")
    
    # Create model
    model = create_model(args).to(device)
    
    # Create dummy dataset for demonstration
    print(f"\\n📚 Creating dummy dataset...")
    n_samples = 1000
    seq_len = 20
    
    questions = torch.randint(0, args.n_questions, (n_samples, seq_len))
    responses = torch.randint(0, args.n_cats, (n_samples, seq_len))
    
    dataset = torch.utils.data.TensorDataset(questions, responses)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"✓ Dataset created: {n_samples} samples, sequence length {seq_len}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\\n🎯 Starting training...")
    for epoch in range(args.epochs):
        print(f"\\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        epoch_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"✓ Epoch {epoch+1} completed: Average Loss = {epoch_loss:.6f}")
        
        # Show adaptive blending info if enabled
        if args.enable_adaptive_blending:
            blending_info = model.get_adaptive_blending_info()
            if blending_info and blending_info.get('analysis_available'):
                params = blending_info['learnable_parameters']
                print(f"  📊 Adaptive parameters: range_sens={params['range_sensitivity']:.4f}, "
                      f"dist_sens={params['distance_sensitivity']:.4f}, bias={params['baseline_bias']:.4f}")
    
    # Save model
    save_path = get_save_path(args)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_questions': args.n_questions,
            'n_cats': args.n_cats,
            'enable_adaptive_blending': args.enable_adaptive_blending,
            'model_version': model.model_version,
            'model_name': model.model_name
        },
        'training_config': vars(args)
    }, save_path)
    
    print(f"\\n💾 Model saved successfully to: {save_path}")
    print(f"   - Model version: {model.model_version}")
    print(f"   - Adaptive blending: {model.enable_adaptive_blending}")
    
    print("\\n✅ Training completed successfully!")
    print("\\n📋 Usage Notes:")
    if args.enable_adaptive_blending:
        print("  🟢 You trained an ADAPTIVE model (v2.0)")
        print("  🔬 This model uses threshold-distance-based dynamic blending")
        print("  📈 Target: Improved middle category prediction accuracy")
        print("  ⚠️  This is experimental - compare against standard model")
    else:
        print("  🔴 You trained a STANDARD model (v1.0)")
        print("  ✅ This model uses existing behavior (backward compatible)")
        print("  🔄 Same as your previous trained models")
        print("  💡 Add --enable_adaptive_blending to try the new feature")


if __name__ == "__main__":
    main()