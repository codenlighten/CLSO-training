"""
Visualize CLSO Training Results

Loads and visualizes the results from a CLSO training run.
"""

import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(exp_dir):
    """Load results from experiment directory."""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Load results JSON
    results_file = exp_path / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = None
    
    # Load best genome
    genome_file = exp_path / 'best_genome.pt'
    if genome_file.exists():
        checkpoint = torch.load(genome_file, map_location='cpu')
    else:
        checkpoint = None
    
    return results, checkpoint


def visualize_genome(genome, library_size):
    """Visualize the genome as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Reshape genome for visualization
    genome_array = np.array(genome).reshape(1, -1)
    
    im = ax.imshow(genome_array, cmap='viridis', aspect='auto', 
                   vmin=0, vmax=library_size-1)
    ax.set_yticks([])
    ax.set_xlabel('Layer Index')
    ax.set_title('Best Genome Configuration')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Basis Function Index')
    
    # Add value labels
    for i, val in enumerate(genome):
        ax.text(i, 0, str(val), ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_summary(results, checkpoint):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("CLSO Training Summary")
    print("="*80)
    
    if results:
        print(f"\n📊 Final Results:")
        print(f"  • Best Loss: {results['best_loss']:.4f}")
        print(f"  • Total Energy: {results['total_energy_wh']:.4f} Wh")
        print(f"  • Generations: {results['num_generations']}")
        
        if 'config' in results:
            config = results['config']
            print(f"\n⚙️  Configuration:")
            print(f"  • Model: {config.get('n_embd', 'N/A')}d, "
                  f"{config.get('n_layer', 'N/A')} layers")
            print(f"  • Library Size: {config.get('library_size', 'N/A')}")
            print(f"  • Population: {config.get('pop_size', 'N/A')}")
            print(f"  • Mutation Rate: {config.get('mutation_rate', 'N/A')}")
        
        if 'best_genome' in results:
            genome = results['best_genome']
            print(f"\n🧬 Best Genome:")
            print(f"  • Length: {len(genome)}")
            print(f"  • Unique basis functions used: {len(set(genome))}")
            print(f"  • Most common basis: {max(set(genome), key=genome.count)} "
                  f"(used {genome.count(max(set(genome), key=genome.count))} times)")
    
    if checkpoint:
        print(f"\n💾 Checkpoint Info:")
        print(f"  • Generation: {checkpoint.get('generation', 'N/A')}")
        print(f"  • Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize CLSO results')
    parser.add_argument('exp_dir', type=str, help='Experiment directory path')
    parser.add_argument('--save', action='store_true', help='Save figure instead of showing')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.exp_dir}")
    results, checkpoint = load_results(args.exp_dir)
    
    # Print summary
    print_summary(results, checkpoint)
    
    # Visualize genome if available
    if results and 'best_genome' in results:
        genome = results['best_genome']
        library_size = results.get('config', {}).get('library_size', 256)
        
        fig = visualize_genome(genome, library_size)
        
        if args.save:
            output_path = Path(args.exp_dir) / 'genome_visualization.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {output_path}")
        else:
            plt.show()
    else:
        print("⚠️  No genome data found to visualize")


if __name__ == "__main__":
    main()
