# Crystalline Latent Space Optimization (CLSO)

A novel neuroevolutionary training framework for energy-efficient Large Language Models.

## Overview

CLSO replaces continuous gradient-based optimization with discrete combinatorial search over pre-defined, geometrically structured "crystalline" basis functions. This approach aims to:

- **Reduce training energy** by avoiding intensive backpropagation
- **Enable faster convergence** through structured parameter space
- **Improve robustness** via sparse and low-rank components
- **Discover novel architectures** not accessible via gradient descent

## Project Structure

```
CLSO-ai-training/
├── src/
│   ├── basis_library.py          # Crystalline matrix basis functions
│   ├── crystalline_model.py      # GPT-2 with discrete parameter selection
│   ├── genetic_optimizer.py      # Evolutionary algorithm with surrogate
│   └── train_clso.py            # Main training script
├── experiments/                  # Experiment outputs
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Test Individual Components

1. **Test Basis Library:**
```bash
cd src
python basis_library.py
```

2. **Test Crystalline Model:**
```bash
python crystalline_model.py
```

3. **Test Genetic Optimizer:**
```bash
python genetic_optimizer.py
```

### Run Full Training

**Small-scale experiment (fast, for testing):**
```bash
python src/train_clso.py \
  --n_embd 128 \
  --n_layer 2 \
  --library_size 64 \
  --pop_size 32 \
  --num_generations 50 \
  --exp_dir ./experiments/test_run
```

**Full-scale experiment:**
```bash
python src/train_clso.py \
  --n_embd 256 \
  --n_layer 4 \
  --library_size 256 \
  --pop_size 128 \
  --num_generations 500 \
  --exp_dir ./experiments/full_run \
  --use_wandb
```

## Key Parameters

### Model Architecture
- `--n_embd`: Embedding dimension (default: 256)
- `--n_layer`: Number of transformer layers (default: 4)
- `--n_head`: Number of attention heads (default: 4)
- `--seq_length`: Sequence length (default: 512)

### CLSO Configuration
- `--library_size`: Number of basis functions in library (default: 256)
- `--pop_size`: Evolutionary population size (default: 128)
- `--num_generations`: Training generations (default: 500)
- `--mutation_rate`: Gene mutation probability (default: 0.08)
- `--crossover_rate`: Crossover probability (default: 0.75)
- `--full_eval_fraction`: Fraction of population to fully evaluate (default: 0.2)

### Training
- `--batch_size`: Batch size for evaluation (default: 8)
- `--eval_batches`: Number of batches per evaluation (default: 50)
- `--dataset`: Dataset to use (default: 'wikitext')

## How It Works

### 1. Basis Library Generation
The `BasisLibrary` creates a discrete library of M efficient matrix structures:
- **Block-sparse matrices**: Block-diagonal structures with varying block sizes
- **Quantized low-rank matrices**: U×V^T where U,V have values in {-1, 0, 1}

### 2. Crystalline Model
The `CrystallineGPT2` replaces standard linear layers with `CrystallineLinear` layers that:
- Point to basis functions via indices (genes)
- Retrieve frozen, pre-computed weights
- Eliminate gradient computation for these layers

### 3. Evolutionary Optimization
The `GeneticOptimizer` manages:
- **Population**: Collection of genomes (index vectors)
- **Fitness evaluation**: Validation loss on WikiText-103
- **Surrogate model**: Neural predictor to reduce full evaluations
- **Genetic operators**: Mutation, crossover, tournament selection
- **Local search**: Hill-climbing on top performers

### 4. Energy Monitoring
Uses NVIDIA Management Library (NVML) to track actual GPU power consumption in Watt-hours.

## Outputs

Each experiment creates:
- `best_genome.pt`: Best configuration checkpoint
- `results.json`: Final statistics and metrics
- Logs: Generation-by-generation progress (to console and W&B if enabled)

## Expected Results

Based on the theoretical framework, CLSO should demonstrate:

1. **Energy Efficiency**: Significantly lower Watt-hours compared to baseline GPT-2
2. **Competitive Performance**: Within 5% perplexity of standard training
3. **Fast Convergence**: Fewer evaluations to reach target performance
4. **Robustness**: Stable training through discrete search space

## Baseline Comparison

To compare with standard training, train a baseline GPT-2:

```bash
# TODO: Add baseline training script
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--n_embd` or `--n_layer`
- Reduce `--seq_length`

### Slow Training
- Increase `--full_eval_fraction` gradually
- Reduce `--eval_batches`
- Use `--use_wandb` for better monitoring

### Poor Convergence
- Increase `--library_size` for more expressiveness
- Adjust `--mutation_rate` and `--crossover_rate`
- Enable more frequent `--local_search_freq`

## Citation

```bibtex
@article{clso2025,
  title={Crystalline Latent Space Optimization: A Discretized Neuroevolutionary Approach for Energy-Efficient LLMs},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

Generated by NeuroLab AI Syndicate
Built with PyTorch, Transformers, and love for energy-efficient AI 💚
