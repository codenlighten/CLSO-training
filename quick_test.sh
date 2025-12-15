#!/bin/bash

# Quick test run of CLSO training
# This script runs a minimal configuration for testing

echo "=================================="
echo "CLSO Quick Test Run"
echo "=================================="
echo ""
echo "Running minimal configuration for testing..."
echo "This should complete in a few minutes."
echo ""

python src/train_clso.py \
  --n_embd 128 \
  --n_layer 2 \
  --n_head 2 \
  --library_size 32 \
  --pop_size 16 \
  --num_generations 5 \
  --batch_size 4 \
  --eval_batches 10 \
  --seq_length 128 \
  --exp_dir ./experiments/quick_test \
  --exp_name "clso_quick_test"

echo ""
echo "=================================="
echo "Test run complete!"
echo "Check ./experiments/quick_test/ for results"
echo "=================================="
