#!/usr/bin/env python3
"""
Real-time monitor for CLSO training progress.
Watches the experiment directory and displays live updates.
"""

import sys
import time
import json
from pathlib import Path
import argparse


def monitor_training(exp_dir, refresh_interval=5):
    """Monitor training progress in real-time."""
    exp_path = Path(exp_dir)
    
    print("="*80)
    print(f"Monitoring CLSO Training: {exp_dir}")
    print("="*80)
    print("\nPress Ctrl+C to stop monitoring (training will continue)\n")
    
    last_gen = -1
    start_time = time.time()
    
    try:
        while True:
            # Check if results file exists
            results_file = exp_path / 'results.json'
            genome_file = exp_path / 'best_genome.pt'
            
            if genome_file.exists():
                import torch
                checkpoint = torch.load(genome_file, map_location='cpu')
                current_gen = checkpoint.get('generation', 0)
                current_loss = checkpoint.get('loss', float('inf'))
                
                if current_gen > last_gen:
                    elapsed = time.time() - start_time
                    print(f"\n[Gen {current_gen+1}] "
                          f"Best Loss: {current_loss:.4f} | "
                          f"Time: {elapsed:.1f}s")
                    last_gen = current_gen
            
            elif results_file.exists():
                # Training complete
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print("\n" + "="*80)
                print("✓ Training Complete!")
                print("="*80)
                print(f"\nFinal Results:")
                print(f"  • Best Loss: {results['best_loss']:.4f}")
                print(f"  • Total Energy: {results['total_energy_wh']:.4f} Wh")
                print(f"  • Generations: {results['num_generations']}")
                print(f"\nResults saved to: {exp_dir}")
                break
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
        print(f"Check {exp_dir} for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor CLSO training')
    parser.add_argument('exp_dir', type=str, help='Experiment directory to monitor')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    monitor_training(args.exp_dir, args.interval)
