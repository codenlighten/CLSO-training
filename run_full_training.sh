#!/bin/bash

# Full CLSO training run with recommended parameters
# This will take significantly longer (hours to days depending on hardware)

echo "=================================="
echo "CLSO Full Training Run"
echo "=================================="
echo ""
echo "Starting full-scale CLSO experiment..."
echo "This will run for many hours."
echo ""
echo "Parameters:"
echo "  - Model: 256-dim, 4 layers"
echo "  - Library: 256 basis functions"
echo "  - Population: 128 individuals"
echo "  - Generations: 500"
echo ""
echo "Press Ctrl+C to stop at any time."
echo ""

# Parse arguments for wandb
USE_WANDB=""
if [ "$1" == "--wandb" ]; then
    USE_WANDB="--use_wandb"
    echo "Weights & Biases logging enabled"
fi

python src/train_clso.py \
  --n_embd 256 \
  --n_layer 4 \
  --n_head 4 \
  --seq_length 512 \
  --library_size 256 \
  --pop_size 128 \
  --num_generations 500 \
  --mutation_rate 0.08 \
  --crossover_rate 0.75 \
  --batch_size 8 \
  --eval_batches 50 \
  --surrogate_update_freq 10 \
  --local_search_freq 50 \
  --full_eval_fraction 0.2 \
  --exp_dir ./experiments/full_run_$(date +%Y%m%d_%H%M%S) \
  --exp_name "clso_full_training" \
  $USE_WANDB

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="
