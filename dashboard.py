#!/usr/bin/env python3
"""
Real-time dashboard for monitoring both CLSO and Baseline training.
"""

import time
import json
from pathlib import Path
import sys


def check_progress(exp_dirs):
    """Check progress of multiple experiments."""
    results = {}
    
    for name, exp_dir in exp_dirs.items():
        exp_path = Path(exp_dir)
        
        results_file = exp_path / 'results.json'
        genome_file = exp_path / 'best_genome.pt'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results[name] = json.load(f)
                results[name]['status'] = 'complete'
        elif genome_file.exists():
            import torch
            checkpoint = torch.load(genome_file, map_location='cpu')
            results[name] = {
                'status': 'running',
                'generation': checkpoint.get('generation', 0),
                'loss': checkpoint.get('loss', float('inf'))
            }
        else:
            results[name] = {'status': 'not_started'}
    
    return results


def display_dashboard(results):
    """Display formatted dashboard."""
    print("\033[2J\033[H")  # Clear screen
    print("="*80)
    print(" "*25 + "CLSO TRAINING DASHBOARD")
    print("="*80)
    print(f"\nLast Update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for name, data in results.items():
        status = data.get('status', 'unknown')
        
        print(f"\n📊 {name.upper()}")
        print("-" * 40)
        
        if status == 'complete':
            print(f"  Status: ✓ COMPLETE")
            print(f"  Best Loss: {data.get('best_loss', 'N/A'):.4f}")
            print(f"  Energy: {data.get('total_energy_wh', 0):.4f} Wh")
            if 'num_generations' in data:
                print(f"  Generations: {data['num_generations']}")
            if 'total_steps' in data:
                print(f"  Steps: {data['total_steps']}")
        
        elif status == 'running':
            print(f"  Status: 🔄 RUNNING")
            print(f"  Current Generation: {data.get('generation', 0)}")
            print(f"  Current Loss: {data.get('loss', 'N/A'):.4f}")
        
        else:
            print(f"  Status: ⏳ NOT STARTED or INITIALIZING")
    
    print("\n" + "="*80)
    print("Press Ctrl+C to stop monitoring (trainings will continue)")
    print("="*80)


def main():
    experiments = {
        'Extended CLSO': './experiments/extended_run',
        'Baseline GPT-2': './experiments/baseline_comparison'
    }
    
    print("Starting monitoring...")
    print("Watching:")
    for name, path in experiments.items():
        print(f"  - {name}: {path}")
    
    time.sleep(2)
    
    try:
        while True:
            results = check_progress(experiments)
            display_dashboard(results)
            
            # Check if all complete
            all_complete = all(r.get('status') == 'complete' for r in results.values())
            if all_complete:
                print("\n\n✅ All trainings complete! Run compare_results.py to see full comparison.")
                break
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Trainings continue in background.")
        print("\nTo check results later:")
        print("  python compare_results.py experiments/extended_run experiments/baseline_comparison")


if __name__ == "__main__":
    main()
