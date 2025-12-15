"""
CLSO Genetic Optimizer

This module implements the evolutionary brain of the system. It manages the 
population of model configurations ("genomes"), handles the genetic operators 
(mutation, crossover), and trains the surrogate model to accelerate search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Dict, Optional


class FitnessPredictor(nn.Module):
    """
    A lightweight surrogate model to estimate the fitness (loss) of a genome.
    This saves energy by predicting which configurations are bad before
    running a full forward pass.
    """
    def __init__(self, num_layers, library_size, embed_dim=16):
        super().__init__()
        # Learnable embeddings for each basis function in the library
        self.basis_embeddings = nn.Embedding(library_size, embed_dim)
        
        # Simple MLP to map concatenated embeddings to a fitness score
        input_dim = num_layers * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicts scalar Loss
        )

    def forward(self, genome_indices_batch):
        # Flatten input: (Batch, N_layers) -> (Batch, N_layers * Embed)
        embeds = self.basis_embeddings(genome_indices_batch)
        flat_embeds = embeds.view(embeds.size(0), -1)
        return self.net(flat_embeds)


class GeneticOptimizer:
    """
    Manages the population of Crystalline LLM configurations.
    """
    def __init__(
        self,
        pop_size: int,
        genome_length: int,
        library_size: int,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        surrogate_update_freq: int = 10,
        device: str = 'cpu'
    ):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.library_size = library_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.device = device
        
        # Initialize Population: List of lists (genomes)
        # Each gene is an index pointing to a Basis Function
        self.population = [
            self._random_genome() for _ in range(pop_size)
        ]
        
        # Surrogate Model
        self.surrogate = FitnessPredictor(genome_length, library_size).to(device)
        self.surrogate_optim = optim.Adam(self.surrogate.parameters(), lr=1e-3)
        self.surrogate_history = []  # Store (genome, real_fitness) for training
        
        self.generation = 0
        self.surrogate_update_freq = surrogate_update_freq
        
        # Best genome tracking
        self.best_genome = None
        self.best_fitness = float('inf')

    def _random_genome(self) -> List[int]:
        return np.random.randint(0, self.library_size, self.genome_length).tolist()

    def get_population_batch(self):
        """Returns the current population as a tensor for efficient surrogate processing."""
        return torch.tensor(self.population, dtype=torch.long, device=self.device)

    def predict_fitness_batch(self, genomes: Optional[List[List[int]]] = None):
        """Use surrogate model to predict fitness for a batch of genomes."""
        if genomes is None:
            genomes = self.population
        
        self.surrogate.eval()
        with torch.no_grad():
            genome_tensor = torch.tensor(genomes, dtype=torch.long, device=self.device)
            predictions = self.surrogate(genome_tensor).cpu().numpy().flatten()
        
        return predictions

    def update_surrogate(self, real_fitness_data: List[Tuple[List[int], float]]):
        """
        Trains the surrogate model on real (Genome, Loss) pairs.
        """
        self.surrogate_history.extend(real_fitness_data)
        
        # Keep history manageable (e.g., last 1000 evaluations)
        if len(self.surrogate_history) > 1000:
            self.surrogate_history = self.surrogate_history[-1000:]
        
        if len(self.surrogate_history) < 10:
            # Not enough data yet
            return
        
        # Prepare batch
        genomes = torch.tensor([x[0] for x in self.surrogate_history], dtype=torch.long, device=self.device)
        targets = torch.tensor([x[1] for x in self.surrogate_history], dtype=torch.float, device=self.device).unsqueeze(1)
        
        # Simple training loop
        self.surrogate.train()
        for _ in range(50):  # 50 gradient steps
            preds = self.surrogate(genomes)
            loss = nn.MSELoss()(preds, targets)
            
            self.surrogate_optim.zero_grad()
            loss.backward()
            self.surrogate_optim.step()

    def evolve(self, fitness_scores: List[float]):
        """
        The Core Genetic Step.
        Args:
            fitness_scores: List of floats corresponding to current population.
                          Lower is better (Loss).
        """
        # Update best genome
        min_idx = np.argmin(fitness_scores)
        if fitness_scores[min_idx] < self.best_fitness:
            self.best_fitness = fitness_scores[min_idx]
            self.best_genome = self.population[min_idx].copy()
        
        # 1. Sort population by fitness (Ascending, since Metric is Loss)
        sorted_indices = np.argsort(fitness_scores)
        sorted_pop = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # 2. Elitism: Keep top 10% exactly as is
        num_elites = int(self.pop_size * 0.1)
        next_gen = [genome.copy() for genome in sorted_pop[:num_elites]]
        
        # 3. Reproduction Loop
        while len(next_gen) < self.pop_size:
            # Tournament Selection (Pick 2 random, take best)
            parent1 = self._tournament_select(sorted_pop, sorted_fitness)
            parent2 = self._tournament_select(sorted_pop, sorted_fitness)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self._mutate(child)
            
            next_gen.append(child)
        
        self.population = next_gen
        self.generation += 1

    def _tournament_select(self, sorted_pop, sorted_fitness):
        # Tournament selection with k=5
        k = min(5, len(sorted_pop))
        tournament_indices = random.sample(range(len(sorted_pop)), k)
        # Return the best (lowest fitness) from tournament
        best_idx = min(tournament_indices, key=lambda i: sorted_fitness[i])
        return sorted_pop[best_idx].copy()

    def _crossover(self, p1, p2):
        # Single point crossover
        point = random.randint(1, self.genome_length - 1)
        return p1[:point] + p2[point:]

    def _mutate(self, genome):
        # Mutate genes with probability `mutation_rate`
        genome = list(genome)  # Copy
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = random.randint(0, self.library_size - 1)
        return genome

    def local_search(self, top_k: int = 10, num_neighbors: int = 5):
        """
        Perform local search on the top K individuals.
        Returns list of (genome, None) tuples for new candidates to evaluate.
        """
        new_candidates = []
        
        for i in range(min(top_k, len(self.population))):
            current_genome = self.population[i]
            
            for _ in range(num_neighbors):
                # Create a neighbor by mutating a single gene
                neighbor = current_genome.copy()
                idx_to_mutate = random.randint(0, self.genome_length - 1)
                neighbor[idx_to_mutate] = random.randint(0, self.library_size - 1)
                
                new_candidates.append(neighbor)
        
        return new_candidates


# ==========================================
# Integration / Main Loop Example
# ==========================================
if __name__ == "__main__":
    # Import previous modules (assuming they are in the same folder)
    from basis_library import BasisLibrary
    from crystalline_model import CrystallineGPT2
    from transformers import GPT2Config
    
    # 1. Setup Environment
    print("--- Setting up CLSO Experiment ---")
    config = GPT2Config(n_embd=64, n_layer=2, n_head=2, vocab_size=500)
    
    # Libraries with proper dimensions
    lib_attn_qkv = BasisLibrary(M=32, d_in=64, d_out=192)
    lib_attn_out = BasisLibrary(M=32, d_in=64, d_out=64)
    lib_mlp_up = BasisLibrary(M=32, d_in=64, d_out=256)
    lib_mlp_down = BasisLibrary(M=32, d_in=256, d_out=64)
    
    # Model wrapper
    model = CrystallineGPT2(config, lib_attn_qkv, lib_attn_out, lib_mlp_up, lib_mlp_down)
    
    # Count targets
    genome_len = len([m for m in model.modules() if hasattr(m, 'basis_index')])
    print(f"Genome Length: {genome_len}")
    
    # 2. Initialize Optimizer
    optimizer = GeneticOptimizer(
        pop_size=20,
        genome_length=genome_len,
        library_size=32
    )
    
    # 3. Mock Training Loop (Simulating 3 Generations)
    for gen in range(3):
        print(f"\nGeneration {gen+1}")
        
        current_pop = optimizer.population
        fitnesses = []
        
        # Evaluate each individual
        for i, genome in enumerate(current_pop):
            # A. Assemble Model
            model.assemble_weights(genome)
            
            # B. Forward Pass (Mock Loss Calculation)
            # In reality: Run a batch of WikiText-103
            dummy_input = torch.randint(0, 500, (1, 10))
            with torch.no_grad():
                out = model(dummy_input)
                # Fake loss: random value + penalty for bad genes (mocking structure)
                loss = torch.randn(1).item() + 10.0
            
            fitnesses.append(loss)
        
        print(f"  Best Loss: {min(fitnesses):.4f}")
        
        # C. Update Surrogate (every gen for demo)
        real_data = list(zip(current_pop, fitnesses))
        optimizer.update_surrogate(real_data)
        
        # D. Evolve
        optimizer.evolve(fitnesses)
    
    print("\nOptimization Complete.")
    print(f"Best Genome: {optimizer.best_genome}")
    print(f"Best Fitness: {optimizer.best_fitness:.4f}")
