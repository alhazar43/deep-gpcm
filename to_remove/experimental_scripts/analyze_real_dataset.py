#!/usr/bin/env python3
"""
Real-World Dataset Analysis

Analyze assist2009 and other real-world datasets to understand:
- Number of students
- Number of questions  
- Sequence lengths per student
- Response patterns
- Data distribution characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json

def parse_sequence_file(file_path):
    """Parse sequence-based data file format."""
    students = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if i >= len(lines):
            break
            
        # Read sequence length
        try:
            seq_len = int(lines[i].strip())
        except:
            i += 1
            continue
            
        if i + 2 >= len(lines):
            break
            
        # Read questions
        question_line = lines[i + 1].strip()
        if question_line.endswith(','):
            question_line = question_line[:-1]
        questions = [int(x) for x in question_line.split(',') if x.strip()]
        
        # Read responses  
        response_line = lines[i + 2].strip()
        if response_line.endswith(','):
            response_line = response_line[:-1]
        responses = [int(x) for x in response_line.split(',') if x.strip()]
        
        if len(questions) == len(responses) == seq_len:
            students.append({
                'questions': questions,
                'responses': responses,
                'seq_len': seq_len
            })
        
        i += 3
    
    return students

def analyze_assist2009():
    """Analyze the assist2009 dataset."""
    print("🔍 ANALYZING ASSIST2009 DATASET")
    print("=" * 50)
    
    data_path = Path("/home/steph/dirt-new/deep-2pl/data/assist2009")
    
    # Read train and test files
    train_file = data_path / "builder_train.csv"
    test_file = data_path / "builder_test.csv"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return None
    
    # Read sequence-based data
    print("📖 Reading sequence-based data files...")
    train_data = parse_sequence_file(train_file)
    test_data = parse_sequence_file(test_file) if test_file.exists() else None
    
    print(f"✅ Train data: {len(train_data)} students")
    if test_data is not None:
        print(f"✅ Test data: {len(test_data)} students")
        all_data = train_data + test_data
    else:
        all_data = train_data
    
    print(f"✅ Total data: {len(all_data)} students")
    print()
    
    # Convert to flat format for analysis
    all_questions = []
    all_responses = []
    sequence_lengths = []
    
    for student in all_data:
        all_questions.extend(student['questions'])
        all_responses.extend(student['responses'])
        sequence_lengths.append(student['seq_len'])
    
    total_interactions = len(all_questions)
    print(f"Total interactions: {total_interactions:,}")
    print()
    
    # Student analysis
    print("👥 STUDENT ANALYSIS")
    print("-" * 20)
    unique_students = len(all_data)
    avg_interactions_per_student = total_interactions / unique_students
    
    print(f"Total students: {unique_students:,}")
    print(f"Total interactions: {total_interactions:,}")
    print(f"Average interactions per student: {avg_interactions_per_student:.1f}")
    
    # Sequence length analysis
    sequence_lengths = np.array(sequence_lengths)
    print(f"Sequence length statistics:")
    print(f"  Min: {sequence_lengths.min()}")
    print(f"  Max: {sequence_lengths.max()}")
    print(f"  Mean: {sequence_lengths.mean():.1f}")
    print(f"  Median: {np.median(sequence_lengths)}")
    print(f"  Std: {sequence_lengths.std():.1f}")
    print()
    
    # Question analysis
    print("❓ QUESTION ANALYSIS")
    print("-" * 20)
    unique_questions = len(set(all_questions))
    question_counts = Counter(all_questions)
    question_frequencies = list(question_counts.values())
    
    print(f"Total unique questions: {unique_questions:,}")
    print(f"Question frequency statistics:")
    print(f"  Min appearances: {min(question_frequencies)}")
    print(f"  Max appearances: {max(question_frequencies)}")
    print(f"  Mean appearances: {np.mean(question_frequencies):.1f}")
    print(f"  Median appearances: {np.median(question_frequencies)}")
    print()
    
    # Response analysis
    print("📝 RESPONSE ANALYSIS")
    print("-" * 20)
    response_counts = Counter(all_responses)
    response_total = sum(response_counts.values())
    
    print("Response distribution:")
    for response in sorted(response_counts.keys()):
        count = response_counts[response]
        rate = count / response_total
        print(f"  Response {response}: {count:,} ({rate:.1%})")
    print()
    
    # Dataset comparison metrics
    print("📈 COMPARISON WITH SYNTHETIC DATASET")
    print("-" * 35)
    print("Assist2009 vs Synthetic_OC (large):")
    print(f"  Students: {unique_students:,} vs 500")
    print(f"  Questions: {unique_questions:,} vs 200") 
    print(f"  Total interactions: {total_interactions:,} vs 75,817")
    print(f"  Avg seq length: {avg_interactions_per_student:.1f} vs 151.6")
    print(f"  Max seq length: {sequence_lengths.max()} vs 200")
    print(f"  Response categories: {len(response_counts)} vs 4")
    
    # Scale factor analysis
    student_scale = unique_students / 500
    question_scale = unique_questions / 200
    interaction_scale = total_interactions / 75817
    
    print(f"\n📊 Scale factors (real vs synthetic):")
    print(f"  Students: {student_scale:.1f}x")
    print(f"  Questions: {question_scale:.1f}x") 
    print(f"  Interactions: {interaction_scale:.1f}x")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR ADAPTIVE BLENDING")
    print("-" * 40)
    
    if len(response_counts) == 2:
        print("⚠️  Binary responses detected - need to adapt for binary classification")
        print("   Consider using binary adaptive thresholding instead of ordinal")
    elif len(response_counts) > 4:
        print(f"✅ {len(response_counts)} response categories - good for ordinal adaptive blending")
    else:
        print(f"✅ {len(response_counts)} response categories - suitable for current adaptive system")
    
    if unique_students > 1000:
        print("✅ Large student population - excellent for training stability")
    else:
        print("⚠️  Small student population - may need careful validation")
    
    if unique_questions > 500:
        print("✅ Large question pool - good for generalization")
    else:
        print("⚠️  Limited question pool - may overfit")
    
    if avg_interactions_per_student > 100:
        print("✅ Long sequences - ideal for memory network training")
    else:
        print("⚠️  Short sequences - may limit memory network effectiveness")
    
    # Calculate response distribution as percentages
    response_distribution = {str(k): v/response_total for k, v in response_counts.items()}
    
    return {
        'dataset': 'assist2009',
        'students': unique_students,
        'questions': unique_questions,
        'interactions': total_interactions,
        'avg_sequence_length': avg_interactions_per_student,
        'max_sequence_length': int(sequence_lengths.max()),
        'response_categories': len(response_counts),
        'response_distribution': response_distribution,
        'binary_responses': len(response_counts) == 2
    }

def analyze_assist2009_updated():
    """Analyze the assist2009_updated dataset."""
    print("\n🔍 ANALYZING ASSIST2009_UPDATED DATASET")
    print("=" * 50)
    
    data_path = Path("/home/steph/dirt-new/deep-2pl/data/assist2009_updated")
    train_file = data_path / "assist2009_updated_train.csv"
    test_file = data_path / "assist2009_updated_test.csv"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return None
    
    print("📖 Reading sequence-based data files...")
    train_data = parse_sequence_file(train_file)
    test_data = parse_sequence_file(test_file) if test_file.exists() else None
    
    print(f"✅ Train data: {len(train_data)} students")
    if test_data is not None:
        print(f"✅ Test data: {len(test_data)} students")
        all_data = train_data + test_data
    else:
        all_data = train_data
    
    # Quick analysis
    all_questions = []
    all_responses = []
    sequence_lengths = []
    
    for student in all_data:
        all_questions.extend(student['questions'])
        all_responses.extend(student['responses'])
        sequence_lengths.append(student['seq_len'])
    
    students = len(all_data)
    questions = len(set(all_questions))
    interactions = len(all_questions)
    correct_rate = np.mean(all_responses)
    
    sequence_lengths = np.array(sequence_lengths)
    
    print(f"\n📊 QUICK STATISTICS:")
    print(f"  Students: {students:,}")
    print(f"  Questions: {questions:,}")
    print(f"  Interactions: {interactions:,}")
    print(f"  Avg sequence length: {sequence_lengths.mean():.1f}")
    print(f"  Max sequence length: {sequence_lengths.max()}")
    print(f"  Correct rate: {correct_rate:.1%}")
    
    return {
        'dataset': 'assist2009_updated',
        'students': students,
        'questions': questions,
        'interactions': interactions,
        'avg_sequence_length': float(sequence_lengths.mean()),
        'max_sequence_length': int(sequence_lengths.max()),
        'correct_rate': correct_rate
    }

def analyze_other_datasets():
    """Quick analysis of other available datasets."""
    datasets = [
        ('assist2015', '/home/steph/dirt-new/deep-2pl/data/assist2015/assist2015_train.txt'),
        ('assist2017', '/home/steph/dirt-new/deep-2pl/data/assist2017/assist2017_train.txt'),
        ('kddcup2010', '/home/steph/dirt-new/deep-2pl/data/kddcup2010/kddcup2010_train.txt'),
        ('statics2011', '/home/steph/dirt-new/deep-2pl/data/statics2011/static2011_train.txt'),
    ]
    
    results = {}
    
    print("\n🔍 QUICK ANALYSIS OF OTHER DATASETS")
    print("=" * 50)
    
    for name, path in datasets:
        if Path(path).exists():
            try:
                # Try to read as simple format first
                with open(path, 'r') as f:
                    lines = f.readlines()
                
                total_lines = len(lines)
                print(f"\n{name.upper()}:")
                print(f"  File: {Path(path).name}")
                print(f"  Lines: {total_lines:,}")
                
                # Try to parse first few lines to understand format
                sample_lines = lines[:5]
                print(f"  Sample data:")
                for i, line in enumerate(sample_lines):
                    print(f"    {i+1}: {line.strip()[:100]}")
                
                results[name] = {'lines': total_lines, 'format': 'text'}
                
            except Exception as e:
                print(f"  ❌ Error reading {name}: {e}")
    
    return results

def main():
    """Main analysis function."""
    print("🔬 REAL-WORLD DATASET ANALYSIS")
    print("=" * 60)
    print("Analyzing knowledge tracing datasets for adaptive blending compatibility")
    print()
    
    results = {}
    
    # Analyze assist2009
    assist2009_results = analyze_assist2009()
    if assist2009_results:
        results['assist2009'] = assist2009_results
    
    # Analyze assist2009_updated  
    assist2009_updated_results = analyze_assist2009_updated()
    if assist2009_updated_results:
        results['assist2009_updated'] = assist2009_updated_results
    
    # Quick analysis of other datasets
    other_results = analyze_other_datasets()
    results.update(other_results)
    
    # Save analysis results
    output_file = '/home/steph/dirt-new/deep-gpcm/results/real_dataset_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis results saved to: {output_file}")
    print("\n🎉 ANALYSIS COMPLETE!")
    
    return results

if __name__ == "__main__":
    results = main()