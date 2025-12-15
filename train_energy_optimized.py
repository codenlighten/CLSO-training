"""
Energy-Optimized CLSO Training Script
Implements early stopping to reduce energy consumption while maintaining performance.

Key Features:
- Convergence detection (stops when no improvement for N generations)
- Patience-based early stopping
- Energy tracking and reporting
- Optimal solution preservation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import json
import os
from datetime import datetime
import time

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.basis_library import BasisLibrary
from src.crystalline_model import CrystallineGPT2, CrystallineLinear
from src.genetic_optimizer import GeneticOptimizer
from src.train_clso import EnergyMonitor, prepare_data

class EarlyStoppingCLSOTrainer:
    """
    CLSO Trainer with early stopping for energy optimization.
    """
    def __init__(
        self,
        model,
        libraries,
        train_loader,
        val_loader,
        optimizer,
        device='cpu',
        patience=5,
        min_delta=0.01,
        output_dir='experiments/energy_optimized'
    ):
        self.model = model
        self.libraries = libraries
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.output_dir = output_dir
        
        # Early stopping state
        self.best_loss = float('inf')
        self.best_genome = None
        self.best_generation = 0
        self.generations_without_improvement = 0
        self.converged = False
        
        # Energy monitoring
        self.energy_monitor = EnergyMonitor()
        
        # History
        self.history = {
            'generations': [],
            'best_losses': [],
            'energy_per_gen': [],
            'surrogate_usage': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_genome(self, genome, use_full_eval=True):
        """Evaluate a single genome configuration."""
        self.model.assemble_weights(genome)
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask=attention_mask)
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # For quick eval, use fewer batches
                if not use_full_eval and num_batches >= 5:
                    break
        
        return total_loss / num_batches
    
    def check_convergence(self, current_loss):
        """
        Check if training has converged.
        Returns True if should stop, False otherwise.
        """
        # Check if improvement is significant
        if current_loss < (self.best_loss - self.min_delta):
            # Significant improvement
            self.best_loss = current_loss
            self.best_genome = self.optimizer.population[0].copy()
            self.best_generation = self.optimizer.generation
            self.generations_without_improvement = 0
            return False
        else:
            # No significant improvement
            self.generations_without_improvement += 1
            
            if self.generations_without_improvement >= self.patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"   No improvement for {self.patience} generations")
                print(f"   Best loss: {self.best_loss:.4f} (Generation {self.best_generation})")
                self.converged = True
                return True
        
        return False
    
    def train(self, max_generations=50):
        """
        Train with early stopping.
        """
        print("=" * 70)
        print("ENERGY-OPTIMIZED CLSO TRAINING")
        print("=" * 70)
        print(f"Max generations: {max_generations}")
        print(f"Patience: {self.patience} generations")
        print(f"Min delta: {self.min_delta}")
        print()
        
        self.energy_monitor.start()
        start_time = time.time()
        
        for generation in range(max_generations):
            gen_start_time = time.time()
            
            print(f"\n{'='*70}")
            print(f"Generation {generation + 1}/{max_generations}")
            print(f"{'='*70}")
            
            # Get current population
            population = self.optimizer.population
            fitness_scores = []
            
            # Determine which individuals to fully evaluate
            pop_size = len(population)
            num_full_eval = max(int(pop_size * 0.2), 1)  # At least top 20%
            
            # Evaluate fitness
            surrogate_count = 0
            full_eval_count = 0
            
            for i, genome in enumerate(population):
                # Top individuals get full evaluation
                if i < num_full_eval or generation < 2:
                    loss = self.evaluate_genome(genome, use_full_eval=True)
                    full_eval_count += 1
                else:
                    # Use surrogate for others
                    genome_tensor = torch.tensor([genome], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        predicted_loss = self.optimizer.surrogate(genome_tensor).item()
                    loss = predicted_loss
                    surrogate_count += 1
                
                fitness_scores.append(loss)
            
            # Get best individual this generation
            best_idx = fitness_scores.index(min(fitness_scores))
            current_best_loss = fitness_scores[best_idx]
            current_best_genome = population[best_idx]
            
            # Update surrogate model
            real_fitness_data = [
                (population[i], fitness_scores[i]) 
                for i in range(min(num_full_eval, len(population)))
            ]
            if generation % 5 == 0 and generation > 0:
                self.optimizer.update_surrogate(real_fitness_data)
            
            # Calculate metrics
            gen_time = time.time() - gen_start_time
            gen_energy = self.energy_monitor.get_energy()
            surrogate_pct = (surrogate_count / pop_size) * 100
            
            # Store history
            self.history['generations'].append(generation + 1)
            self.history['best_losses'].append(current_best_loss)
            self.history['energy_per_gen'].append(gen_energy)
            self.history['surrogate_usage'].append(surrogate_pct)
            
            # Print progress
            print(f"\nResults:")
            print(f"  Best Loss: {current_best_loss:.4f}")
            print(f"  Global Best: {self.best_loss:.4f} (Gen {self.best_generation})")
            print(f"  Surrogate Usage: {surrogate_pct:.1f}%")
            print(f"  Time: {gen_time:.1f}s")
            print(f"  Energy: {gen_energy:.4f} Wh")
            
            # Check for convergence
            if self.check_convergence(current_best_loss):
                break
            
            # Evolve population
            self.optimizer.evolve(fitness_scores)
        
        # Final results
        self.energy_monitor.stop()
        total_time = time.time() - start_time
        total_energy = self.energy_monitor.get_energy()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nFinal Results:")
        print(f"  Generations completed: {generation + 1}")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Found at generation: {self.best_generation}")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Total energy: {total_energy:.4f} Wh")
        print(f"  Energy per generation: {total_energy/(generation+1):.4f} Wh")
        
        # Calculate energy savings
        if generation + 1 < max_generations:
            saved_gens = max_generations - (generation + 1)
            projected_full_energy = total_energy * (max_generations / (generation + 1))
            energy_saved = projected_full_energy - total_energy
            savings_pct = (energy_saved / projected_full_energy) * 100
            
            print(f"\n  Energy Savings:")
            print(f"    Stopped {saved_gens} generations early")
            print(f"    Projected full run: {projected_full_energy:.4f} Wh")
            print(f"    Actual energy: {total_energy:.4f} Wh")
            print(f"    Saved: {energy_saved:.4f} Wh ({savings_pct:.1f}%)")
        
        # Save results
        self.save_results(generation + 1, total_energy)
        
        return self.best_loss, self.best_genome
    
    def save_results(self, final_generation, total_energy):
        """Save training results and best model."""
        results = {
            'final_generation': final_generation,
            'best_loss': float(self.best_loss),
            'best_generation': self.best_generation,
            'total_energy': float(total_energy),
            'converged': self.converged,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'history': {
                'generations': self.history['generations'],
                'best_losses': [float(x) for x in self.history['best_losses']],
                'energy_per_gen': [float(x) for x in self.history['energy_per_gen']],
                'surrogate_usage': [float(x) for x in self.history['surrogate_usage']]
            },
            'best_genome': self.best_genome,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")

def main():
    """Main training function."""
    print("🚀 Energy-Optimized CLSO Training\n")
    
    # Configuration
    device = 'cpu'
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=512,
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1
    )
    
    # Load dataset
    print("Loading WikiText-103 dataset...")
    train_loader, val_loader = prepare_data(batch_size=8, max_length=128)
    
    # Create basis libraries
    print("\nInitializing basis libraries...")
    lib_attn_qkv = BasisLibrary(M=64, d_in=128, d_out=384, device=device)
    lib_attn_out = BasisLibrary(M=64, d_in=128, d_out=128, device=device)
    lib_mlp_up = BasisLibrary(M=64, d_in=128, d_out=512, device=device)
    lib_mlp_down = BasisLibrary(M=64, d_in=512, d_out=128, device=device)
    
    libraries = {
        'attn_qkv': lib_attn_qkv,
        'attn_out': lib_attn_out,
        'mlp_up': lib_mlp_up,
        'mlp_down': lib_mlp_down
    }
    
    # Create model
    print("\nCreating Crystalline GPT-2 model...")
    model = CrystallineGPT2(config, libraries)
    model.to(device)
    
    # Count crystalline layers
    num_crystal_layers = len([m for m in model.modules() if isinstance(m, CrystallineLinear)])
    print(f"Number of crystalline layers: {num_crystal_layers}")
    
    # Create genetic optimizer
    print("\nInitializing genetic optimizer...")
    optimizer = GeneticOptimizer(
        pop_size=32,
        genome_length=num_crystal_layers,
        library_size=64,
        mutation_rate=0.08,
        crossover_rate=0.75,
        device=device
    )
    
    # Create trainer
    trainer = EarlyStoppingCLSOTrainer(
        model=model,
        libraries=libraries,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        patience=5,  # Stop if no improvement for 5 generations
        min_delta=0.01,  # Minimum improvement threshold
        output_dir='experiments/energy_optimized'
    )
    
    # Train with early stopping
    best_loss, best_genome = trainer.train(max_generations=50)
    
    print("\n" + "=" * 70)
    print("✅ ENERGY-OPTIMIZED TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
