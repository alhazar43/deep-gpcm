#!/usr/bin/env python3
"""
Simple parameter counter using manual calculation.
"""

def count_baseline_parameters():
    """Manually count baseline parameters."""
    n_questions = 50
    n_cats = 4
    memory_size = 50
    key_dim = 50
    value_dim = 200
    final_fc_dim = 50
    
    # Question embedding: (n_questions + 1) * key_dim
    q_embed_params = (n_questions + 1) * key_dim
    
    # GPCM value embedding: (n_cats * n_questions) * value_dim + value_dim (bias)
    gpcm_embed_dim = n_cats * n_questions
    gpcm_value_embed_params = gpcm_embed_dim * value_dim + value_dim
    
    # DKVMN memory
    # Key memory: memory_size * key_dim
    key_memory_params = memory_size * key_dim
    
    # Query key linear: key_dim * key_dim + key_dim (bias)
    query_key_params = key_dim * key_dim + key_dim
    
    # Memory head groups
    # Write head: erase_linear + add_linear
    erase_linear_params = value_dim * value_dim + value_dim
    add_linear_params = value_dim * value_dim + value_dim
    
    # Summary network: (value_dim + key_dim) * final_fc_dim + final_fc_dim (bias)
    summary_network_params = (value_dim + key_dim) * final_fc_dim + final_fc_dim
    
    # Student ability network: final_fc_dim * 1 + 1 (bias)
    student_ability_params = final_fc_dim * 1 + 1
    
    # Question threshold network: key_dim * (n_cats - 1) + (n_cats - 1) (bias)
    question_threshold_params = key_dim * (n_cats - 1) + (n_cats - 1)
    
    # Discrimination network: (final_fc_dim + key_dim) * 1 + 1 (bias)
    discrimination_params = (final_fc_dim + key_dim) * 1 + 1
    
    # Initial value memory: memory_size * value_dim
    init_value_memory_params = memory_size * value_dim
    
    total_params = (
        q_embed_params +
        gpcm_value_embed_params +
        key_memory_params +
        query_key_params +
        erase_linear_params +
        add_linear_params +
        summary_network_params +
        student_ability_params +
        question_threshold_params +
        discrimination_params +
        init_value_memory_params
    )
    
    print(f"Baseline Parameter Breakdown:")
    print(f"  Question embedding: {q_embed_params:,}")
    print(f"  GPCM value embedding: {gpcm_value_embed_params:,}")
    print(f"  Key memory: {key_memory_params:,}")
    print(f"  Query transform: {query_key_params:,}")
    print(f"  Erase linear: {erase_linear_params:,}")
    print(f"  Add linear: {add_linear_params:,}")
    print(f"  Summary network: {summary_network_params:,}")
    print(f"  Student ability: {student_ability_params:,}")
    print(f"  Question threshold: {question_threshold_params:,}")
    print(f"  Discrimination: {discrimination_params:,}")
    print(f"  Init value memory: {init_value_memory_params:,}")
    print(f"  TOTAL: {total_params:,}")
    
    return total_params

def count_akvmn_parameters():
    """Manually count AKVMN parameters."""
    n_questions = 50
    n_cats = 4
    memory_size = 50
    key_dim = 50
    value_dim = 200
    final_fc_dim = 50
    
    # Question embedding: (n_questions + 1) * key_dim
    q_embed_params = (n_questions + 1) * key_dim
    
    # GPCM value embedding: (n_cats * n_questions) * value_dim + value_dim (bias)
    gpcm_embed_dim = n_cats * n_questions
    gpcm_value_embed_params = gpcm_embed_dim * value_dim + value_dim
    
    # Key memory: memory_size * key_dim
    key_memory_params = memory_size * key_dim
    
    # Query transform: key_dim * key_dim + key_dim (bias)
    query_transform_params = key_dim * key_dim + key_dim
    
    # Key transform: key_dim * key_dim + key_dim (bias)
    key_transform_params = key_dim * key_dim + key_dim
    
    # Write erase: value_dim * value_dim + value_dim (bias)
    write_erase_params = value_dim * value_dim + value_dim
    
    # Write add: value_dim * value_dim + value_dim (bias)
    write_add_params = value_dim * value_dim + value_dim
    
    # Enhancement layer: 2 linear layers (tuned for target)
    # Layer 1: (value_dim + key_dim) * (final_fc_dim * 2) + (final_fc_dim * 2) (bias)
    enhancement_1_params = (value_dim + key_dim) * (final_fc_dim * 2) + (final_fc_dim * 2)
    # Layer 2: (final_fc_dim * 2) * final_fc_dim + final_fc_dim (bias)
    enhancement_2_params = (final_fc_dim * 2) * final_fc_dim + final_fc_dim
    enhancement_params = enhancement_1_params + enhancement_2_params
    
    # Ordinal projection: 1 linear layer (simplified)
    # Layer 1: final_fc_dim * n_cats + n_cats (bias)
    ordinal_params = final_fc_dim * n_cats + n_cats
    
    # Initial value memory: memory_size * value_dim
    init_value_memory_params = memory_size * value_dim
    
    total_params = (
        q_embed_params +
        gpcm_value_embed_params +
        key_memory_params +
        query_transform_params +
        key_transform_params +
        write_erase_params +
        write_add_params +
        enhancement_params +
        ordinal_params +
        init_value_memory_params
    )
    
    print(f"\nAKVMN Parameter Breakdown:")
    print(f"  Question embedding: {q_embed_params:,}")
    print(f"  GPCM value embedding: {gpcm_value_embed_params:,}")
    print(f"  Key memory: {key_memory_params:,}")
    print(f"  Query transform: {query_transform_params:,}")
    print(f"  Key transform: {key_transform_params:,}")
    print(f"  Write erase: {write_erase_params:,}")
    print(f"  Write add: {write_add_params:,}")
    print(f"  Enhancement layer: {enhancement_params:,}")
    print(f"  Ordinal projection: {ordinal_params:,}")
    print(f"  Init value memory: {init_value_memory_params:,}")
    print(f"  TOTAL: {total_params:,}")
    
    return total_params

def main():
    """Main function."""
    print("=== Manual Parameter Count Analysis ===")
    
    baseline_params = count_baseline_parameters()
    akvmn_params = count_akvmn_parameters()
    
    print(f"\n=== Summary ===")
    print(f"Baseline: {baseline_params:,} parameters")
    print(f"AKVMN: {akvmn_params:,} parameters")
    
    diff = akvmn_params - baseline_params
    percent_diff = (diff / baseline_params) * 100
    print(f"Difference: {diff:,} ({percent_diff:+.1f}%)")
    
    # Historical comparison
    target_baseline = 130655
    target_akvmn = 171217
    target_diff = target_akvmn - target_baseline
    
    print(f"\nHistorical Targets:")
    print(f"Baseline: {target_baseline:,}")
    print(f"AKVMN: {target_akvmn:,}")
    print(f"Target difference: {target_diff:,}")
    
    print(f"\nParameter Accuracy:")
    print(f"Baseline match: {'✅' if abs(baseline_params - target_baseline) < 1000 else '❌'} ({baseline_params - target_baseline:+,})")
    print(f"AKVMN match: {'✅' if abs(akvmn_params - target_akvmn) < 1000 else '❌'} ({akvmn_params - target_akvmn:+,})")

if __name__ == "__main__":
    main()