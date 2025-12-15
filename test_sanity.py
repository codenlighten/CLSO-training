"""
Quick sanity check for CLSO components.
Tests each module independently before full training.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

import torch
import numpy as np

print("="*80)
print("CLSO Component Sanity Check")
print("="*80)

# Test 1: Basis Library
print("\n[1/3] Testing Basis Library...")
try:
    from basis_library import BasisLibrary
    
    lib = BasisLibrary(M=32, d_in=64, d_out=64, types=['block_sparse', 'quantized_low_rank'])
    matrix = lib.get_matrix(0)
    
    assert matrix.shape == (64, 64), f"Expected shape (64, 64), got {matrix.shape}"
    assert torch.is_tensor(matrix), "Output should be a tensor"
    
    print("  ✓ Basis library creation successful")
    print(f"  ✓ Matrix shape: {matrix.shape}")
    print(f"  ✓ Library size: {lib.M}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Crystalline Model
print("\n[2/3] Testing Crystalline Model...")
try:
    from crystalline_model import CrystallineGPT2, CrystallineLinear
    from transformers import GPT2Config
    
    config = GPT2Config(n_embd=64, n_layer=2, n_head=2, vocab_size=1000)
    lib_attn_qkv = BasisLibrary(M=16, d_in=64, d_out=192)  # n_embd -> 3*n_embd
    lib_attn_out = BasisLibrary(M=16, d_in=64, d_out=64)   # n_embd -> n_embd
    lib_mlp_up = BasisLibrary(M=16, d_in=64, d_out=256)    # n_embd -> 4*n_embd
    lib_mlp_down = BasisLibrary(M=16, d_in=256, d_out=64)  # 4*n_embd -> n_embd
    
    model = CrystallineGPT2(config, lib_attn_qkv, lib_attn_out, lib_mlp_up, lib_mlp_down)
    
    # Count crystalline layers
    num_crystal = len([m for m in model.modules() if isinstance(m, CrystallineLinear)])
    
    # Create genome
    genome = [0] * num_crystal
    model.assemble_weights(genome)
    
    # Forward pass
    dummy_input = torch.randint(0, 1000, (2, 10))
    output = model(dummy_input)
    
    assert output['logits'].shape == (2, 10, 1000), f"Expected shape (2, 10, 1000), got {output['logits'].shape}"
    
    print("  ✓ Model creation successful")
    print(f"  ✓ Crystalline layers: {num_crystal}")
    print(f"  ✓ Output shape: {output['logits'].shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Genetic Optimizer
print("\n[3/3] Testing Genetic Optimizer...")
try:
    from genetic_optimizer import GeneticOptimizer
    
    optimizer = GeneticOptimizer(
        pop_size=20,
        genome_length=num_crystal,
        library_size=16,
        mutation_rate=0.1
    )
    
    # Mock fitness evaluation
    fitnesses = np.random.rand(20) * 10
    
    # Test evolution
    optimizer.evolve(fitnesses)
    
    # Test surrogate update
    real_data = [(optimizer.population[i], fitnesses[i]) for i in range(5)]
    optimizer.update_surrogate(real_data)
    
    # Test prediction
    predictions = optimizer.predict_fitness_batch()
    
    assert len(predictions) == 20, f"Expected 20 predictions, got {len(predictions)}"
    assert optimizer.generation == 1, f"Expected generation 1, got {optimizer.generation}"
    
    print("  ✓ Optimizer creation successful")
    print(f"  ✓ Population size: {optimizer.pop_size}")
    print(f"  ✓ Genome length: {optimizer.genome_length}")
    print(f"  ✓ Evolution working")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Integration check
print("\n[4/4] Testing Full Integration...")
try:
    # Simulate one generation
    for i in range(5):
        genome = optimizer.population[i]
        model.assemble_weights(genome)
        
        dummy_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(dummy_input, labels=dummy_input)
            loss = output['loss']
        
        assert loss is not None, "Loss should not be None"
        assert torch.is_tensor(loss), "Loss should be a tensor"
    
    print("  ✓ Integration test passed")
    print("  ✓ Model + Optimizer working together")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ All Sanity Checks Passed!")
print("="*80)
print("\nSystem is ready for training. Run:")
print("  python src/train_clso.py --help")
print("\nFor a quick test run:")
print("  python src/train_clso.py --n_embd 128 --n_layer 2 --library_size 32 --pop_size 16 --num_generations 5")
