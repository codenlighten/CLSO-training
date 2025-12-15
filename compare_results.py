#!/usr/bin/env python3
"""
Compare CLSO vs Baseline Training Results
"""

import json
import torch
from pathlib import Path


def load_clso_results(exp_dir):
    """Load CLSO experiment results."""
    exp_path = Path(exp_dir)
    
    results_file = exp_path / 'results.json'
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    checkpoint_file = exp_path / 'best_genome.pt'
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        results['checkpoint_gen'] = checkpoint.get('generation', 'N/A')
    
    return results


def load_baseline_results(exp_dir):
    """Load baseline training results."""
    exp_path = Path(exp_dir)
    
    results_file = exp_path / 'results.json'
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def compare_results(clso_dir, baseline_dir):
    """Compare CLSO and baseline results."""
    
    print("\n" + "="*80)
    print("CLSO vs Baseline Comparison")
    print("="*80 + "\n")
    
    clso = load_clso_results(clso_dir)
    baseline = load_baseline_results(baseline_dir)
    
    if not clso:
        print(f"❌ CLSO results not found in {clso_dir}")
        return
    
    if not baseline:
        print(f"❌ Baseline results not found in {baseline_dir}")
        return
    
    # Performance comparison
    print("📊 Performance Metrics\n")
    
    print(f"{'Metric':<20} {'CLSO':>12} {'Baseline':>12} {'Difference':>15}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*15}")
    print(f"{'Best Loss':<20} {clso['best_loss']:>12.4f} {baseline['best_loss']:>12.4f} {clso['best_loss'] - baseline['best_loss']:>+15.4f}")
    print(f"{'Energy (Wh)':<20} {clso.get('total_energy_wh', 0):>12.4f} {baseline.get('total_energy_wh', 0):>12.4f} {clso.get('total_energy_wh', 0) - baseline.get('total_energy_wh', 0):>+15.4f}")
    
    # Training details
    print("\n\n⚙️  Training Configuration\n")
    
    print(f"{'Parameter':<20} {'CLSO':<30} {'Baseline':<30}")
    print(f"{'-'*20} {'-'*30} {'-'*30}")
    print(f"{'Model Size':<20} {clso['config']['n_embd']}d, {clso['config']['n_layer']} layers{'':<15} {baseline['config']['n_embd']}d, {baseline['config']['n_layer']} layers")
    print(f"{'Training Method':<20} {'Evolutionary Search':<30} {'Gradient Descent':<30}")
    print(f"{'Iterations':<20} {clso['num_generations']} generations{'':<17} {baseline.get('total_steps', 'N/A')} steps")
    print(f"{'Population/Batch':<20} {clso['config']['pop_size']} individuals{'':<17} batch_size={baseline['config']['batch_size']}")
    
    # CLSO-specific info
    if 'best_genome' in clso:
        print(f"\n\n🧬 CLSO Best Genome: {clso['best_genome']}")
        print(f"   Unique basis functions: {len(set(clso['best_genome']))}/{len(clso['best_genome'])}")
    
    # Summary
    print("\n\n" + "="*80)
    print("Summary")
    print("="*80 + "\n")
    
    loss_diff = clso['best_loss'] - baseline['best_loss']
    energy_diff = clso.get('total_energy_wh', 0) - baseline.get('total_energy_wh', 0)
    
    if loss_diff < 0:
        print(f"✓ CLSO achieved {abs(loss_diff):.4f} LOWER loss than baseline")
    else:
        print(f"✗ CLSO loss is {loss_diff:.4f} HIGHER than baseline")
    
    if energy_diff < 0:
        energy_savings = (1 - clso.get('total_energy_wh', 0) / baseline.get('total_energy_wh', 1)) * 100
        print(f"✓ CLSO used {abs(energy_diff):.4f} Wh LESS energy ({energy_savings:.1f}% savings)")
    else:
        print(f"✗ CLSO used {energy_diff:.4f} Wh MORE energy")
    
    print("\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <clso_exp_dir> <baseline_exp_dir>")
        print("\nExample:")
        print("  python compare_results.py experiments/extended_run experiments/baseline_comparison")
        sys.exit(1)
    
    compare_results(sys.argv[1], sys.argv[2])
