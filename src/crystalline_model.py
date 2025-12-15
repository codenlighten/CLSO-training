"""
Crystalline Model Assembler

This module implements the CrystallineGPT2 architecture. It replaces the standard 
dense linear layers of a transformer with custom "Crystal" layers that retrieve 
their weights from the BasisLibrary.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from typing import List, Optional
import copy

# Import the library class we defined previously
from .basis_library import BasisLibrary


class CrystallineLinear(nn.Module):
    """
    A 'Virtual' Linear layer.
    Instead of storing a weight matrix, it stores an index pointing to
    a matrix in the shared BasisLibrary.
    """
    def __init__(self, library: BasisLibrary, bias: bool = True):
        super().__init__()
        self.library = library
        self.basis_index = 0  # Default to 0, mutable by the Genome
        
        # Bias remains continuous and trainable in this version (optional)
        # It's very small parameter-wise compared to weights.
        if bias:
            self.bias = nn.Parameter(torch.zeros(library.d_out))
        else:
            self.register_parameter('bias', None)

    def set_basis_index(self, index: int):
        """Updates the pointer to the basis function."""
        self.basis_index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Retrieve the weight matrix (Frozen/Fixed)
        # Note: We detach to ensure no gradients flow back into the library generator
        weight = self.library.get_matrix(self.basis_index).detach()
        
        # 2. Standard Linear transformation: y = xA^T + b
        return nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f'basis_index={self.basis_index}, bias={self.bias is not None}'


class CrystallineGPT2(nn.Module):
    """
    A GPT-2 wrapper where specific linear projection layers are replaced
    by CrystallineLinear layers.
    """
    def __init__(self, config: GPT2Config, library_attn_qkv: BasisLibrary, library_attn_out: BasisLibrary, 
                 library_mlp_up: BasisLibrary, library_mlp_down: BasisLibrary):
        super().__init__()
        self.config = config
        
        # 1. Load standard GPT-2 structure
        # We start with a standard HF model to get embeddings/blocks structure
        self.transformer = GPT2Model(config)
        
        # 2. Store Libraries with different dimensions
        self.lib_attn_qkv = library_attn_qkv    # For c_attn (n_embd -> 3*n_embd)
        self.lib_attn_out = library_attn_out    # For attn c_proj (n_embd -> n_embd)
        self.lib_mlp_up = library_mlp_up        # For mlp c_fc (n_embd -> 4*n_embd)
        self.lib_mlp_down = library_mlp_down    # For mlp c_proj (4*n_embd -> n_embd)
        
        # 3. "Crystalize" the model: Recursively replace layers
        self._replace_layers(self.transformer)
        
        # 4. Standard LM Head (Continuous)
        # As discussed, we keep the head continuous for output stability
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights if requested (standard GPT-2 practice)
        self.lm_head.weight = self.transformer.wte.weight

    def _replace_layers(self, module: nn.Module):
        """
        Recursively traverse the model and replace nn.Conv1D (HF style)
        or nn.Linear with CrystallineLinear.
        """
        for name, child in module.named_children():
            # HF GPT2 uses Conv1D for linear layers usually.
            # We target specific layer names known in GPT2 architecture.
            
            # Target 1: Attention Q,K,V projection (c_attn)
            if name == 'c_attn' and 'attn' in module.__class__.__name__.lower():
                self._swapping_logic(module, name, child, self.lib_attn_qkv)
            
            # Target 2: Attention output projection (c_proj in attention)
            elif name == 'c_proj' and 'attn' in module.__class__.__name__.lower():
                self._swapping_logic(module, name, child, self.lib_attn_out)
            
            # Target 3: MLP expansion (c_fc)
            elif name == 'c_fc' and 'mlp' in module.__class__.__name__.lower():
                self._swapping_logic(module, name, child, self.lib_mlp_up)
            
            # Target 4: MLP projection (c_proj in MLP)
            elif name == 'c_proj' and 'mlp' in module.__class__.__name__.lower():
                self._swapping_logic(module, name, child, self.lib_mlp_down)
            
            else:
                # Recurse deeper
                self._replace_layers(child)

    def _swapping_logic(self, parent, name, child, library):
        """Helper to perform the actual layer swap."""
        # Determine bias
        has_bias = getattr(child, 'bias', None) is not None
        
        # Create new Crystal Layer
        new_layer = CrystallineLinear(library, bias=has_bias)
        
        # Replace in parent
        setattr(parent, name, new_layer)

    def assemble_weights(self, genome: List[int]):
        """
        The 'hydration' step. Takes a genome (list of indices) and assigns
        them to the CrystallineLinear layers in sequential order.
        """
        # 1. Collect all crystal layers
        crystal_layers = [
            m for m in self.modules() if isinstance(m, CrystallineLinear)
        ]
        
        if len(genome) != len(crystal_layers):
            raise ValueError(
                f"Genome length ({len(genome)}) does not match "
                f"model crystal layers ({len(crystal_layers)})"
            )
        
        # 2. Assign indices
        for layer, index in zip(crystal_layers, genome):
            layer.set_basis_index(index)

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': lm_logits}


# ==========================================
# Demonstration Block
# ==========================================
if __name__ == "__main__":
    print("--- Initializing Crystalline GPT-2 ---")
    
    # 1. Setup Config (Tiny for demo)
    config = GPT2Config(
        n_embd=128,
        n_layer=2,
        n_head=2,
        vocab_size=1000
    )
    
    # 2. Create Libraries with proper dimensions
    print("Generating Attention QKV Library (128 -> 384)...")
    lib_attn_qkv = BasisLibrary(M=16, d_in=128, d_out=384)
    
    print("Generating Attention Out Library (128 -> 128)...")
    lib_attn_out = BasisLibrary(M=16, d_in=128, d_out=128)
    
    print("Generating MLP Up Library (128 -> 512)...")
    lib_mlp_up = BasisLibrary(M=16, d_in=128, d_out=512)
    
    print("Generating MLP Down Library (512 -> 128)...")
    lib_mlp_down = BasisLibrary(M=16, d_in=512, d_out=128)

    # 3. Instantiate Model
    model = CrystallineGPT2(config, lib_attn_qkv, lib_attn_out, lib_mlp_up, lib_mlp_down)
    
    # 4. Create a Fake Genome
    # Count how many crystal layers we have
    num_crystal = len([m for m in model.modules() if isinstance(m, CrystallineLinear)])
    print(f"Model has {num_crystal} Crystalline Layers.")
    
    fake_genome = [0] * num_crystal  # Just point everything to index 0
    model.assemble_weights(fake_genome)
    print("Weights assembled from genome.")

    # 5. Forward Pass Check
    dummy_input = torch.randint(0, 1000, (1, 10))  # Batch 1, Seq 10
    output = model(dummy_input)
    
    print(f"Output Shape: {output['logits'].shape} (Batch, Seq, Vocab)")
    print("Forward pass successful. Model is ready for evolution.")
