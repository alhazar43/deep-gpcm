#!/usr/bin/env python3
"""
Convert Large Dataset to Training Format

Converts the CSV format (student_id,question_id,response) to the expected format:
seq_len
question1,question2,...
response1,response2,...
"""

from collections import defaultdict
import json
from pathlib import Path

def convert_csv_to_sequences(csv_file, output_file):
    """Convert CSV format to sequence format."""
    
    # Group by student_id
    student_data = defaultdict(lambda: {'questions': [], 'responses': []})
    
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 3:
                continue
                
            student_id, question_id, response = map(int, parts)
            student_data[student_id]['questions'].append(question_id)
            student_data[student_id]['responses'].append(response)
    
    # Write in expected format
    with open(output_file, 'w') as f:
        for student_id in sorted(student_data.keys()):
            questions = student_data[student_id]['questions']
            responses = student_data[student_id]['responses']
            
            seq_len = len(questions)
            
            # Write sequence length
            f.write(f"{seq_len}\n")
            
            # Write questions
            f.write(','.join(map(str, questions)) + '\n')
            
            # Write responses  
            f.write(','.join(map(str, responses)) + '\n')
    
    print(f"✅ Converted {len(student_data)} students to {output_file}")
    return len(student_data)

def main():
    """Convert the large dataset to the expected format."""
    print("🔄 Converting Large Dataset Format")
    print("=" * 50)
    
    data_dir = Path('/home/steph/dirt-new/deep-gpcm/data/synthetic_OC')
    
    # Convert train file
    print("Converting train file...")
    n_train = convert_csv_to_sequences(
        data_dir / 'synthetic_oc_train.txt',
        data_dir / 'synthetic_oc_train.txt'
    )
    
    # Convert test file
    print("Converting test file...")
    n_test = convert_csv_to_sequences(
        data_dir / 'synthetic_oc_test.txt', 
        data_dir / 'synthetic_oc_test.txt'
    )
    
    # Update metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    metadata['train_students'] = n_train
    metadata['test_students'] = n_test
    metadata['total_students'] = n_train + n_test
    
    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Updated metadata: {n_train} train, {n_test} test students")
    print(f"🎉 Dataset format conversion complete!")

if __name__ == "__main__":
    main()