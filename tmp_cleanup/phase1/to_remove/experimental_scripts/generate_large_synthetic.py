#!/usr/bin/env python3
"""
Generate Large-Scale Synthetic Dataset

Creates a realistic synthetic OC dataset with:
- 500 students
- 200 questions  
- 100-200 sequence length per student
- 4 ordered categories
- Realistic IRT parameters
"""

import numpy as np
import json
import os
from pathlib import Path

def generate_irt_parameters(n_questions, n_cats):
    """Generate realistic IRT parameters for GPCM."""
    np.random.seed(42)
    
    # Difficulty parameters (beta) - spread across ability range
    difficulties = np.random.normal(0, 1.2, n_questions)
    
    # Discrimination parameters (alpha) - mostly positive, some variation
    discriminations = np.random.lognormal(0.2, 0.3, n_questions)
    discriminations = np.clip(discriminations, 0.5, 3.0)
    
    # Threshold parameters for each question (for GPCM)
    # These represent the difficulty of achieving each category level
    thresholds = []
    for q in range(n_questions):
        # Generate ordered thresholds relative to item difficulty
        base_difficulty = difficulties[q]
        # Create 3 thresholds for 4 categories (0,1,2,3)
        thresh = np.sort(np.random.normal(base_difficulty, 0.8, n_cats - 1))
        thresholds.append(thresh.tolist())
    
    return {
        'difficulties': difficulties.tolist(),
        'discriminations': discriminations.tolist(), 
        'thresholds': thresholds,
        'n_questions': n_questions,
        'n_cats': n_cats
    }

def generate_student_abilities(n_students):
    """Generate student ability parameters."""
    np.random.seed(43)
    # Abilities from normal distribution
    abilities = np.random.normal(0, 1, n_students)
    return abilities.tolist()

def gpcm_response_probability(ability, difficulty, discrimination, thresholds):
    """Calculate GPCM response probabilities."""
    n_cats = len(thresholds) + 1
    
    # Calculate numerators for each category
    numerators = []
    for k in range(n_cats):
        if k == 0:
            # Category 0: no thresholds to cross
            num = 1.0
        else:
            # Category k: sum of (ability - threshold) for all thresholds up to k
            threshold_sum = sum(discrimination * (ability - thresholds[j]) for j in range(k))
            num = np.exp(threshold_sum)
        numerators.append(num)
    
    # Normalize to get probabilities
    total = sum(numerators)
    probs = [num / total for num in numerators]
    
    return probs

def generate_responses(abilities, irt_params, min_seq=100, max_seq=200):
    """Generate student responses using GPCM."""
    np.random.seed(44)
    
    n_students = len(abilities)
    n_questions = irt_params['n_questions']
    n_cats = irt_params['n_cats']
    
    all_responses = []
    
    for student_id in range(n_students):
        ability = abilities[student_id]
        
        # Random sequence length for this student
        seq_len = np.random.randint(min_seq, max_seq + 1)
        
        # Random questions for this student
        questions = np.random.choice(n_questions, size=seq_len, replace=True)
        
        responses = []
        for q_idx in questions:
            # Get IRT parameters for this question
            difficulty = irt_params['difficulties'][q_idx]
            discrimination = irt_params['discriminations'][q_idx] 
            thresholds = irt_params['thresholds'][q_idx]
            
            # Calculate response probabilities
            probs = gpcm_response_probability(ability, difficulty, discrimination, thresholds)
            
            # Sample response
            response = np.random.choice(n_cats, p=probs)
            responses.append(response)
        
        all_responses.append({
            'student_id': student_id,
            'questions': questions.tolist(),
            'responses': responses,
            'ability': ability
        })
    
    return all_responses

def write_dataset_files(data, output_dir, train_ratio=0.8):
    """Write train/test files in the expected format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_students = len(data)
    n_train = int(n_students * train_ratio)
    
    # Split data
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    # Write train file
    with open(output_dir / 'synthetic_oc_train.txt', 'w') as f:
        for student in train_data:
            for q, r in zip(student['questions'], student['responses']):
                f.write(f"{student['student_id']},{q},{r}\n")
    
    # Write test file  
    with open(output_dir / 'synthetic_oc_test.txt', 'w') as f:
        for student in test_data:
            for q, r in zip(student['questions'], student['responses']):
                f.write(f"{student['student_id']},{q},{r}\n")
    
    print(f"✅ Dataset written to {output_dir}")
    print(f"   - Train: {n_train} students")
    print(f"   - Test: {len(test_data)} students")
    
    return n_train, len(test_data)

def create_metadata(n_students, n_questions, n_cats, min_seq, max_seq, n_train, n_test, output_dir):
    """Create metadata file."""
    metadata = {
        "n_students": n_students,
        "n_questions": n_questions, 
        "n_cats": n_cats,
        "response_type": "ordered_categorical",
        "format": "OC",
        "description": f"Large-scale synthetic GPCM data with {n_cats} ordered categories",
        "train_students": n_train,
        "test_students": n_test,
        "seq_len_range": [min_seq, max_seq]
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved")

def main():
    """Generate large-scale synthetic dataset."""
    print("🔧 Generating Large-Scale Synthetic OC Dataset")
    print("=" * 60)
    
    # Configuration
    n_students = 500
    n_questions = 200
    n_cats = 4
    min_seq = 100
    max_seq = 200
    output_dir = '/home/steph/dirt-new/deep-gpcm/data/synthetic_OC'
    
    print(f"Configuration:")
    print(f"  - Students: {n_students}")
    print(f"  - Questions: {n_questions}")
    print(f"  - Categories: {n_cats}")
    print(f"  - Sequence length: {min_seq}-{max_seq}")
    print(f"  - Output: {output_dir}")
    print()
    
    # Generate IRT parameters
    print("🎯 Generating IRT parameters...")
    irt_params = generate_irt_parameters(n_questions, n_cats)
    
    # Generate student abilities
    print("👥 Generating student abilities...")
    abilities = generate_student_abilities(n_students)
    
    # Generate responses
    print("📝 Generating student responses...")
    response_data = generate_responses(abilities, irt_params, min_seq, max_seq)
    
    # Calculate statistics
    total_responses = sum(len(student['responses']) for student in response_data)
    avg_seq_len = total_responses / n_students
    
    print(f"📊 Dataset statistics:")
    print(f"  - Total responses: {total_responses:,}")
    print(f"  - Average sequence length: {avg_seq_len:.1f}")
    print()
    
    # Write dataset files
    print("💾 Writing dataset files...")
    n_train, n_test = write_dataset_files(response_data, output_dir)
    
    # Create metadata
    create_metadata(n_students, n_questions, n_cats, min_seq, max_seq, n_train, n_test, output_dir)
    
    # Save IRT parameters
    with open(Path(output_dir) / 'true_irt_parameters.json', 'w') as f:
        json.dump({
            'irt_parameters': irt_params,
            'student_abilities': abilities
        }, f, indent=2)
    
    print(f"✅ IRT parameters saved")
    print()
    print("🎉 Large-scale synthetic dataset generation complete!")
    print(f"Dataset scale: {n_students} students × {avg_seq_len:.0f} avg responses = {total_responses:,} total responses")

if __name__ == "__main__":
    main()