"""
CLSO Training Script

Main training loop for Crystalline Latent Space Optimization.
Integrates the basis library, crystalline model, and genetic optimizer
with real dataset evaluation and energy monitoring.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import wandb
import json
import time
from pathlib import Path
import argparse

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU energy monitoring disabled.")

# Handle both direct script execution and package imports
try:
    from .basis_library import BasisLibrary
    from .crystalline_model import CrystallineGPT2
    from .genetic_optimizer import GeneticOptimizer
except ImportError:
    # If relative imports fail, try absolute imports
    from basis_library import BasisLibrary
    from crystalline_model import CrystallineGPT2
    from genetic_optimizer import GeneticOptimizer


class EnergyMonitor:
    """Monitor GPU energy consumption using NVIDIA Management Library."""
    
    def __init__(self):
        self.available = NVML_AVAILABLE
        if self.available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.power_readings = []
                print("✓ Energy monitoring initialized")
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
                self.available = False
    
    def start_measurement(self):
        """Start energy measurement period."""
        self.power_readings = []
        self.start_time = time.time()
    
    def record(self):
        """Record current power draw."""
        if self.available:
            try:
                # Power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_readings.append(power_mw / 1000.0)  # Convert to watts
            except:
                pass
    
    def get_energy_consumed(self):
        """Get total energy consumed in Watt-hours."""
        if not self.power_readings:
            return 0.0
        
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        avg_power_watts = np.mean(self.power_readings)
        return avg_power_watts * elapsed_hours
    
    def shutdown(self):
        if self.available:
            pynvml.nvmlShutdown()


class CLSOTrainer:
    """Main trainer for CLSO experiments."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize energy monitor
        self.energy_monitor = EnergyMonitor()
        
        # Setup experiment directory
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if args.use_wandb:
            wandb.init(
                project="clso-training",
                config=vars(args),
                name=args.exp_name
            )
        
        # Load dataset
        print("Loading dataset...")
        self.setup_dataset()
        
        # Setup model and libraries
        print("Initializing CLSO components...")
        self.setup_model()
        
        # Setup optimizer
        self.setup_optimizer()
    
    def setup_dataset(self):
        """Load and prepare WikiText-103 dataset."""
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        if self.args.dataset == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-v1')
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.args.seq_length,
                padding='max_length'
            )
        
        # Tokenize datasets
        print("Tokenizing dataset...")
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        self.val_dataset = self.val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        # Set format
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def setup_model(self):
        """Initialize the crystalline model and basis libraries."""
        # Create config
        self.config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=self.args.n_embd,
            n_layer=self.args.n_layer,
            n_head=self.args.n_head,
            n_positions=self.args.seq_length
        )
        
        # Create basis libraries with proper dimensions
        print(f"Creating basis libraries with M={self.args.library_size} functions...")
        
        # Attention QKV: (n_embd -> 3 * n_embd)
        self.lib_attn_qkv = BasisLibrary(
            M=self.args.library_size,
            d_in=self.args.n_embd,
            d_out=3 * self.args.n_embd,
            device=str(self.device)
        )
        
        # Attention output: (n_embd -> n_embd)
        self.lib_attn_out = BasisLibrary(
            M=self.args.library_size,
            d_in=self.args.n_embd,
            d_out=self.args.n_embd,
            device=str(self.device)
        )
        
        # MLP expansion: (n_embd -> 4 * n_embd)
        self.lib_mlp_up = BasisLibrary(
            M=self.args.library_size,
            d_in=self.args.n_embd,
            d_out=4 * self.args.n_embd,
            device=str(self.device)
        )
        
        # MLP projection: (4 * n_embd -> n_embd)
        self.lib_mlp_down = BasisLibrary(
            M=self.args.library_size,
            d_in=4 * self.args.n_embd,
            d_out=self.args.n_embd,
            device=str(self.device)
        )
        
        # Create model
        self.model = CrystallineGPT2(
            self.config,
            self.lib_attn_qkv,
            self.lib_attn_out,
            self.lib_mlp_up,
            self.lib_mlp_down
        ).to(self.device)
        
        # Count crystalline layers
        from crystalline_model import CrystallineLinear
        self.genome_length = len([
            m for m in self.model.modules() if isinstance(m, CrystallineLinear)
        ])
        print(f"Model has {self.genome_length} crystalline layers")
    
    def setup_optimizer(self):
        """Initialize genetic optimizer."""
        self.genetic_opt = GeneticOptimizer(
            pop_size=self.args.pop_size,
            genome_length=self.genome_length,
            library_size=self.args.library_size,
            mutation_rate=self.args.mutation_rate,
            crossover_rate=self.args.crossover_rate,
            surrogate_update_freq=self.args.surrogate_update_freq,
            device=str(self.device)
        )
    
    def evaluate_genome(self, genome):
        """Evaluate a single genome on validation set."""
        # Assemble model with this genome
        self.model.assemble_weights(genome)
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Use subset of validation data for speed
        max_eval_batches = self.args.eval_batches
        
        with torch.no_grad():
            for i in range(0, min(len(self.val_dataset), max_eval_batches * self.args.batch_size), self.args.batch_size):
                batch_data = self.val_dataset[i:i+self.args.batch_size]
                
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                if num_batches >= max_eval_batches:
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting CLSO Training")
        print("="*80 + "\n")
        
        self.energy_monitor.start_measurement()
        
        best_loss = float('inf')
        best_genome = None
        
        for generation in range(self.args.num_generations):
            gen_start = time.time()
            print(f"\n{'='*80}")
            print(f"Generation {generation + 1}/{self.args.num_generations}")
            print(f"{'='*80}")
            
            # Get current population
            population = self.genetic_opt.population
            
            # Evaluate population
            fitnesses = []
            
            # Determine which individuals to fully evaluate
            num_full_eval = max(int(self.args.full_eval_fraction * self.args.pop_size), 1)
            
            print(f"Evaluating {num_full_eval}/{self.args.pop_size} individuals fully...")
            
            for i, genome in enumerate(tqdm(population, desc="Evaluating")):
                if i < num_full_eval or generation < 5:
                    # Full evaluation
                    loss = self.evaluate_genome(genome)
                    fitnesses.append(loss)
                    
                    # Record energy periodically
                    if i % 5 == 0:
                        self.energy_monitor.record()
                else:
                    # Use surrogate
                    pred_loss = self.genetic_opt.predict_fitness_batch([genome])[0]
                    fitnesses.append(pred_loss)
            
            # Update statistics
            best_idx = np.argmin(fitnesses)
            gen_best_loss = fitnesses[best_idx]
            gen_best_genome = population[best_idx]
            
            if gen_best_loss < best_loss:
                best_loss = gen_best_loss
                best_genome = gen_best_genome.copy()
                
                # Save best model
                checkpoint = {
                    'generation': generation,
                    'genome': best_genome,
                    'loss': best_loss,
                    'config': self.config.to_dict()
                }
                torch.save(checkpoint, self.exp_dir / 'best_genome.pt')
            
            # Update surrogate model
            real_evals = [(population[i], fitnesses[i]) for i in range(min(num_full_eval, len(population)))]
            self.genetic_opt.update_surrogate(real_evals)
            
            # Evolve population
            self.genetic_opt.evolve(fitnesses)
            
            # Logging
            gen_time = time.time() - gen_start
            energy_consumed = self.energy_monitor.get_energy_consumed()
            
            print(f"\nGeneration {generation + 1} Results:")
            print(f"  Best Loss: {gen_best_loss:.4f}")
            print(f"  Overall Best Loss: {best_loss:.4f}")
            print(f"  Mean Loss: {np.mean(fitnesses):.4f}")
            print(f"  Std Loss: {np.std(fitnesses):.4f}")
            print(f"  Time: {gen_time:.2f}s")
            print(f"  Energy: {energy_consumed:.4f} Wh")
            
            if self.args.use_wandb:
                wandb.log({
                    'generation': generation,
                    'best_loss': gen_best_loss,
                    'overall_best_loss': best_loss,
                    'mean_loss': np.mean(fitnesses),
                    'std_loss': np.std(fitnesses),
                    'energy_wh': energy_consumed,
                    'time': gen_time
                })
            
            # Periodic local search
            if (generation + 1) % self.args.local_search_freq == 0:
                print("\nPerforming local search...")
                new_candidates = self.genetic_opt.local_search(
                    top_k=10,
                    num_neighbors=5
                )
                # Evaluate promising candidates
                for candidate in new_candidates[:5]:
                    loss = self.evaluate_genome(candidate)
                    if loss < best_loss:
                        best_loss = loss
                        best_genome = candidate.copy()
                        print(f"  Local search found better genome: {loss:.4f}")
        
        # Final results
        total_energy = self.energy_monitor.get_energy_consumed()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Total Energy Consumed: {total_energy:.4f} Wh")
        print(f"Best genome saved to: {self.exp_dir / 'best_genome.pt'}")
        
        # Save final results
        results = {
            'best_loss': float(best_loss),
            'best_genome': best_genome,
            'total_energy_wh': float(total_energy),
            'num_generations': self.args.num_generations,
            'config': vars(self.args)
        }
        
        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.energy_monitor.shutdown()
        
        if self.args.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train CLSO model')
    
    # Model architecture
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    
    # CLSO specific
    parser.add_argument('--library_size', type=int, default=256, help='Basis library size')
    parser.add_argument('--pop_size', type=int, default=128, help='Population size')
    parser.add_argument('--num_generations', type=int, default=500, help='Number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.08, help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.75, help='Crossover rate')
    parser.add_argument('--surrogate_update_freq', type=int, default=10, help='Surrogate update frequency')
    parser.add_argument('--local_search_freq', type=int, default=50, help='Local search frequency')
    parser.add_argument('--full_eval_fraction', type=float, default=0.2, help='Fraction of pop to fully evaluate')
    
    # Training
    parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--eval_batches', type=int, default=50, help='Number of batches for evaluation')
    
    # Experiment
    parser.add_argument('--exp_dir', type=str, default='./experiments/clso_run', help='Experiment directory')
    parser.add_argument('--exp_name', type=str, default='clso_experiment', help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = CLSOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
