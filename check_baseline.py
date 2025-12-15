#!/usr/bin/env python3
"""
Quick monitoring script to check baseline training progress
"""

import time
import json
from pathlib import Path

def check_progress():
    exp_dir = Path("experiments/baseline_comparison")
    results_file = exp_dir / "results.json"
    
    print("🔍 Monitoring Baseline Training Progress\n")
    print("=" * 60)
    
    if not exp_dir.exists():
        print("❌ Experiment directory not found")
        return
    
    if not results_file.exists():
        print("⏳ Training in progress...")
        print("   (Results file not yet created)")
    else:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        best_loss = results.get('best_loss', 'N/A')
        final_loss = results.get('final_loss', 'N/A')
        steps = results.get('total_steps', 'N/A')
        
        if best_loss == float('inf') or best_loss == 'Infinity':
            print("⏳ Training completed but loss not recorded properly")
            print("   Need to rerun with fixed evaluation")
        else:
            print("✅ Training Complete!")
            print(f"   Best Loss: {best_loss}")
            print(f"   Final Loss: {final_loss}")
            print(f"   Total Steps: {steps}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_progress()
