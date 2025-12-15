"""
CLSO Basis Library Generator

This module implements the core BasisLibrary class. It generates a "crystalline" 
search space of matrices that are geometrically structured (Block-Sparse) or 
mathematically compressed (Quantized Low-Rank), rather than randomly dense.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Literal


class BasisLibrary:
    """
    A curated library of 'crystalline' matrix basis functions.
    Instead of learning weights continuously, the CLSO framework selects
    pre-fabricated, efficient matrices from this library.
    """
    def __init__(
        self,
        M: int,
        d_in: int,
        d_out: int,
        types: List[str] = ['block_sparse', 'quantized_low_rank'],
        device: str = 'cpu'
    ):
        """
        Args:
            M (int): Total number of basis functions in the library.
            d_in (int): Input dimension of the target layers.
            d_out (int): Output dimension of the target layers.
            types (List[str]): List of generation strategies to use.
            device (str): Device to store the library on.
        """
        self.M = M
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.library = {}
        
        # Partition the library slots among the requested types
        per_type_count = M // len(types)
        
        print(f"Initializing BasisLibrary with {M} slots for shape ({d_in}, {d_out})...")
        
        current_idx = 0
        for gen_type in types:
            count = per_type_count
            # Give any remainder slots to the last type
            if gen_type == types[-1]:
                count += (M % len(types))
            
            print(f"  > Generating {count} matrices of type: {gen_type}")
            
            for _ in range(count):
                if gen_type == 'block_sparse':
                    matrix = self._generate_block_sparse()
                elif gen_type == 'quantized_low_rank':
                    matrix = self._generate_quantized_low_rank()
                else:
                    raise ValueError(f"Unknown generation type: {gen_type}")
                
                self.library[current_idx] = matrix.to(self.device)
                current_idx += 1
        
        print("Initialization complete.")

    def get_matrix(self, index: int) -> torch.Tensor:
        """Retrieves a basis matrix by its index."""
        if index not in self.library:
            raise IndexError(f"Index {index} out of bounds for Library size {self.M}")
        return self.library[index]

    def _generate_block_sparse(self) -> torch.Tensor:
        """
        Generates a block-diagonal sparse matrix.
        Strategy: Randomly partition dimensions into 2-8 blocks and fill them with dense noise.
        """
        # 1. Determine number of blocks (random between 2 and 8)
        num_blocks = torch.randint(2, 9, (1,)).item()
        
        # 2. Partition d_in and d_out roughly equally
        row_splits = self._random_partition(self.d_out, num_blocks)
        col_splits = self._random_partition(self.d_in, num_blocks)
        
        blocks = []
        for r, c in zip(row_splits, col_splits):
            # Create a dense block, normalized to keep variance stable
            block = torch.randn(r, c) * (2.0 / (r + c))**0.5
            blocks.append(block)
        
        # 3. Assemble into a full matrix using torch.block_diag
        full_matrix = torch.block_diag(*blocks)
        
        return full_matrix

    def _generate_quantized_low_rank(self) -> torch.Tensor:
        """
        Generates a Low-Rank matrix W = U * V^T where elements of U and V
        are quantized to {-1, 0, 1}.
        """
        # 1. Choose a random rank r (e.g., between 4 and d_in/4)
        max_rank = max(4, min(self.d_in, self.d_out) // 4)
        rank = torch.randint(2, max_rank + 1, (1,)).item()
        
        # 2. Generate U (d_out x rank) and V (d_in x rank)
        # We sample from {-1, 0, 1} with probabilities.
        # High prob of 0 promotes sparsity within the factors.
        probs = torch.tensor([0.25, 0.5, 0.25])  # prob(-1), prob(0), prob(1)
        
        # Helper to sample -1, 0, 1
        indices_u = torch.multinomial(probs, self.d_out * rank, replacement=True).view(self.d_out, rank)
        U = (indices_u - 1).float()  # map [0,1,2] -> [-1, 0, 1]
        
        indices_v = torch.multinomial(probs, self.d_in * rank, replacement=True).view(self.d_in, rank)
        V = (indices_v - 1).float()
        
        # 3. Compute W = U @ V.T
        W = torch.matmul(U, V.t())
        
        # 4. Scale to maintain signal variance roughly
        # Standard deviation of sum of 'rank' elements with values {-1,0,1}
        scale = 1.0 / (rank ** 0.5)
        return W * scale

    def _random_partition(self, total_size: int, num_parts: int) -> List[int]:
        """Helper to randomly partition an integer into `num_parts` sum components."""
        if num_parts == 1:
            return [total_size]
        
        # Generate random cut points
        cuts = torch.sort(torch.randint(1, total_size, (num_parts - 1,)))[0]
        
        # Calculate differences between cuts to get sizes
        sizes = []
        prev = 0
        for cut in cuts:
            sizes.append(cut.item() - prev)
            prev = cut.item()
        sizes.append(total_size - prev)
        
        # Handle case where a size might be 0 due to duplicate random cuts
        # We force min size 1 by redistributing
        sizes = [max(1, s) for s in sizes]
        # Adjust last element to ensure sum is correct
        diff = sum(sizes) - total_size
        if diff > 0:
            # subtract from largest block
            max_idx = np.argmax(sizes)
            sizes[max_idx] -= diff
        elif diff < 0:
            # add to first block
            sizes[0] -= diff
        
        return sizes


# ==========================================
# Demonstration Block
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Configuration mimicking a GPT-2 Small Projection Layer (768 -> 768)
    D_MODEL = 128  # Reduced for visualization clarity
    LIB_SIZE = 16
    
    print("--- Generating CLSO Basis Library ---")
    lib = BasisLibrary(M=LIB_SIZE, d_in=D_MODEL, d_out=D_MODEL)
    
    # Visualizing one of each type
    print("\n--- Visualization Check ---")
    
    # Find one Block Sparse
    bs_matrix = lib.get_matrix(0)  # Logic above puts block sparse first
    # Find one Quantized Low Rank
    qlr_matrix = lib.get_matrix(LIB_SIZE - 1)  # Logic above puts QLR last
    
    print(f"Block Sparse Shape: {bs_matrix.shape}")
    print(f"Block Sparse Density: {(bs_matrix != 0).float().mean().item():.2%}")
    
    print(f"Quantized Low-Rank Shape: {qlr_matrix.shape}")
    print(f"QLR Unique Values (approx): {torch.unique(torch.round(qlr_matrix, decimals=2))[:10]}")

    # Plotting (requires matplotlib)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].spy(bs_matrix.numpy(), markersize=1)
    ax[0].set_title("Block Sparse Pattern")
    
    # For QLR, we visualize the heatmap as it is dense but low-rank
    im = ax[1].imshow(qlr_matrix.numpy(), cmap='bwr', aspect='auto')
    ax[1].set_title("Quantized Low-Rank (Heatmap)")
    
    plt.tight_layout()
    plt.savefig('basis_library_visualization.png')
    print("Visualization saved to basis_library_visualization.png")
