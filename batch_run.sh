#!/bin/bash
# Activate environment first
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

for k in 2 3 4; do
    echo "=== Training synthetic_10000_200_$k ==="
    python main.py --dataset synthetic_10000_200_$k --epochs 30 --n_folds 0
    if [ $? -eq 0 ]; then
        echo "✅ Success: synthetic_10000_200_$k"
    else
        echo "❌ Failed: synthetic_10000_200_$k"
    fi
done
echo "🎯 All batches completed!"