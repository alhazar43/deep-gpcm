#!/usr/bin/env python3
"""Test ordinal attention training loop."""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_ordinal_training():
    """Test ordinal attention training."""
    print("Testing ordinal attention training...")
    
    # Load data
    from train import load_simple_data, create_data_loaders
    train_data, test_data, n_questions, n_cats = load_simple_data(
        "data/synthetic_4000_200_2/synthetic_4000_200_2_train.txt",
        "data/synthetic_4000_200_2/synthetic_4000_200_2_test.txt"
    )
    
    print(f"Data: {len(train_data)} train, {len(test_data)} test")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create data loaders with smaller batch size
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=16)
    print(f"Loaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Create model
    from models.implementations.attention_gpcm import AttentionGPCM
    model = AttentionGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=1,
        use_ordinal_attention=True,
        attention_types=['ordinal_aware']
    )
    
    # Create loss function
    from models.losses.ordinal_losses import create_ordinal_loss
    loss_fn = create_ordinal_loss('ordinal_ce', n_cats)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test single epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.train()
    
    print("\\nStarting training epoch...")
    start_time = time.time()
    
    train_loss = 0.0
    num_batches = 0
    max_batches = 5  # Limit to first 5 batches for testing
    
    for batch_idx, (questions, responses, masks) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
            
        questions, responses = questions.to(device), responses.to(device)
        
        print(f"  Batch {batch_idx+1}: Q {questions.shape}, R {responses.shape}")
        
        optimizer.zero_grad()
        
        # Forward pass
        memory, abilities, difficulty, probs = model(questions, responses)
        
        # Calculate loss
        logits = torch.log(probs + 1e-8)
        loss = loss_fn(logits, responses)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
        
        print(f"    Loss: {loss.item():.4f}")
    
    avg_loss = train_loss / num_batches
    elapsed = time.time() - start_time
    
    print(f"\\n✅ Training test completed!")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Batches processed: {num_batches}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("ORDINAL TRAINING TEST")
    print("=" * 50)
    
    try:
        success = test_ordinal_training()
        if success:
            print("\\n🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()