"""
Scaling Experiment Configurations for CLSO

This module provides pre-configured settings for scaling experiments:
- Different model sizes (GPT-2 Small, Medium, Large)
- Different library sizes (64, 128, 256 basis functions)
- Different datasets (WikiText-103, OpenWebText, C4)
- Different population sizes and generation counts
"""

from dataclasses import dataclass
from typing import Dict, List
from transformers import GPT2Config

@dataclass
class LibraryConfig:
    """Configuration for basis function library."""
    size: int  # Number of basis functions (M)
    types: List[str]  # Types of matrices to generate
    name: str  # Configuration name
    
    @property
    def description(self):
        return f"{self.name}: {self.size} basis functions ({', '.join(self.types)})"

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: int
    vocab_size: int = 50257
    
    def to_gpt2_config(self) -> GPT2Config:
        """Convert to HuggingFace GPT2Config."""
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=1024,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )
    
    @property
    def description(self):
        return f"{self.name}: {self.n_layer} layers, {self.n_embd}d, {self.n_head} heads"

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    name: str
    pop_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    batch_size: int
    max_length: int
    patience: int
    
    @property
    def description(self):
        return (f"{self.name}: Pop {self.pop_size}, "
                f"{self.generations} gens, "
                f"batch {self.batch_size}")

@dataclass
class DatasetConfig:
    """Configuration for dataset."""
    name: str
    hf_name: str  # HuggingFace dataset name
    subset: str = None
    split_train: str = 'train'
    split_val: str = 'validation'
    
    @property
    def description(self):
        subset_str = f"/{self.subset}" if self.subset else ""
        return f"{self.name} ({self.hf_name}{subset_str})"

# ============================================================================
# Library Configurations
# ============================================================================

LIBRARY_CONFIGS = {
    'tiny': LibraryConfig(
        size=32,
        types=['block_sparse', 'quantized_low_rank'],
        name='Tiny'
    ),
    'small': LibraryConfig(
        size=64,
        types=['block_sparse', 'quantized_low_rank'],
        name='Small'
    ),
    'medium': LibraryConfig(
        size=128,
        types=['block_sparse', 'quantized_low_rank'],
        name='Medium'
    ),
    'large': LibraryConfig(
        size=256,
        types=['block_sparse', 'quantized_low_rank'],
        name='Large'
    ),
}

# ============================================================================
# Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    'gpt2-tiny': ModelConfig(
        name='GPT-2 Tiny',
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=512
    ),
    'gpt2-small': ModelConfig(
        name='GPT-2 Small',
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072
    ),
    'gpt2-medium': ModelConfig(
        name='GPT-2 Medium',
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=4096
    ),
    'gpt2-large': ModelConfig(
        name='GPT-2 Large',
        n_embd=1280,
        n_layer=36,
        n_head=20,
        n_inner=5120
    ),
}

# ============================================================================
# Training Configurations
# ============================================================================

TRAINING_CONFIGS = {
    'quick': TrainingConfig(
        name='Quick Test',
        pop_size=16,
        generations=5,
        mutation_rate=0.08,
        crossover_rate=0.75,
        batch_size=4,
        max_length=128,
        patience=3
    ),
    'standard': TrainingConfig(
        name='Standard',
        pop_size=32,
        generations=50,
        mutation_rate=0.08,
        crossover_rate=0.75,
        batch_size=8,
        max_length=128,
        patience=5
    ),
    'extended': TrainingConfig(
        name='Extended',
        pop_size=64,
        generations=100,
        mutation_rate=0.08,
        crossover_rate=0.75,
        batch_size=8,
        max_length=128,
        patience=10
    ),
    'large_scale': TrainingConfig(
        name='Large Scale',
        pop_size=128,
        generations=200,
        mutation_rate=0.08,
        crossover_rate=0.75,
        batch_size=16,
        max_length=256,
        patience=15
    ),
}

# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIGS = {
    'wikitext103': DatasetConfig(
        name='WikiText-103',
        hf_name='wikitext',
        subset='wikitext-103-v1',
        split_train='train',
        split_val='validation'
    ),
    'wikitext2': DatasetConfig(
        name='WikiText-2',
        hf_name='wikitext',
        subset='wikitext-2-v1',
        split_train='train',
        split_val='validation'
    ),
    'openwebtext': DatasetConfig(
        name='OpenWebText',
        hf_name='openwebtext',
        split_train='train',
        split_val='train'  # No separate validation, will need to split
    ),
}

# ============================================================================
# Experiment Presets
# ============================================================================

EXPERIMENT_PRESETS = {
    'quick_test': {
        'description': 'Quick sanity check (5 generations, tiny model)',
        'model': 'gpt2-tiny',
        'library': 'tiny',
        'training': 'quick',
        'dataset': 'wikitext2'
    },
    'baseline': {
        'description': 'Baseline experiment matching paper results',
        'model': 'gpt2-tiny',
        'library': 'small',
        'training': 'standard',
        'dataset': 'wikitext103'
    },
    'scale_model': {
        'description': 'Scale to GPT-2 Small',
        'model': 'gpt2-small',
        'library': 'medium',
        'training': 'extended',
        'dataset': 'wikitext103'
    },
    'scale_library': {
        'description': 'Test larger library (256 basis functions)',
        'model': 'gpt2-tiny',
        'library': 'large',
        'training': 'extended',
        'dataset': 'wikitext103'
    },
    'scale_both': {
        'description': 'Scale both model and library',
        'model': 'gpt2-medium',
        'library': 'large',
        'training': 'large_scale',
        'dataset': 'wikitext103'
    },
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_experiment_config(preset_name: str) -> Dict:
    """
    Get full configuration for an experiment preset.
    
    Args:
        preset_name: Name of the preset (e.g., 'baseline', 'scale_model')
    
    Returns:
        Dictionary with all configuration objects
    """
    if preset_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(EXPERIMENT_PRESETS.keys())}")
    
    preset = EXPERIMENT_PRESETS[preset_name]
    
    return {
        'preset_name': preset_name,
        'description': preset['description'],
        'model': MODEL_CONFIGS[preset['model']],
        'library': LIBRARY_CONFIGS[preset['library']],
        'training': TRAINING_CONFIGS[preset['training']],
        'dataset': DATASET_CONFIGS[preset['dataset']]
    }

def print_experiment_config(config: Dict):
    """Pretty print an experiment configuration."""
    print("=" * 70)
    print(f"EXPERIMENT: {config['preset_name']}")
    print("=" * 70)
    print(f"\nDescription: {config['description']}\n")
    print("Configuration:")
    print(f"  • Model: {config['model'].description}")
    print(f"  • Library: {config['library'].description}")
    print(f"  • Training: {config['training'].description}")
    print(f"  • Dataset: {config['dataset'].description}")
    print("=" * 70)

def list_available_presets():
    """List all available experiment presets."""
    print("\n📋 Available Experiment Presets:\n")
    for name, preset in EXPERIMENT_PRESETS.items():
        print(f"  • {name:15} - {preset['description']}")
    print()

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CLSO Scaling Experiment Configurations\n")
    
    # List all presets
    list_available_presets()
    
    # Show example configuration
    print("\nExample: 'baseline' preset configuration:")
    config = get_experiment_config('baseline')
    print_experiment_config(config)
    
    # Show scaling progression
    print("\n\n📈 Scaling Progression:\n")
    for preset_name in ['quick_test', 'baseline', 'scale_model', 'scale_library', 'scale_both']:
        config = get_experiment_config(preset_name)
        print(f"{preset_name:15} → {config['model'].name:15} × "
              f"Library {config['library'].size:3} × "
              f"Pop {config['training'].pop_size:3} × "
              f"{config['training'].generations:3} gens")
