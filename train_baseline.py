"""
Baseline GPT-2 Training Script

Train a standard GPT-2 model using conventional gradient descent
for comparison with CLSO approach.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
import argparse

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class EnergyMonitor:
    """Monitor energy consumption during training."""
    
    def __init__(self):
        self.has_gpu = False
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_gpu = True
            except:
                pass
        
        self.start_time = None
        self.total_energy_j = 0.0
    
    def start(self):
        """Start energy monitoring."""
        self.start_time = time.time()
        self.total_energy_j = 0.0
    
    def update(self):
        """Update energy consumption (call periodically)."""
        if not self.has_gpu or self.start_time is None:
            return
        
        try:
            # Get power draw in mW
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            # Convert to Watts
            power_w = power_mw / 1000.0
            # Estimate energy since last update (assuming ~1 sec intervals)
            self.total_energy_j += power_w
        except:
            pass
    
    def get_energy_wh(self):
        """Get total energy consumed in Watt-hours."""
        if self.total_energy_j == 0:
            # Estimate based on CPU usage (very rough)
            if self.start_time:
                elapsed_h = (time.time() - self.start_time) / 3600.0
                # Assume ~50W average for CPU training
                return 50.0 * elapsed_h
        return self.total_energy_j / 3600.0  # Joules to Wh


class BaselineTrainer:
    """Standard GPT-2 training for baseline comparison."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup experiment directory
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.setup_dataset()
        
        # Setup model
        print("Initializing model...")
        self.setup_model()
        
        # Setup optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
    
    def setup_dataset(self):
        """Load and prepare WikiText-103 dataset."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dataset = load_dataset('wikitext', 'wikitext-103-v1')
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['validation']
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.args.seq_length,
                padding='max_length'
            )
        
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
        
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    def setup_model(self):
        """Initialize standard GPT-2 model."""
        config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=self.args.n_embd,
            n_layer=self.args.n_layer,
            n_head=self.args.n_head,
            n_positions=self.args.seq_length
        )
        
        self.model = GPT2LMHeadModel(config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, min(len(self.val_dataset), self.args.eval_batches * self.args.batch_size), 
                          self.args.batch_size):
                batch = self.val_dataset[i:i+self.args.batch_size]
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                if num_batches >= self.args.eval_batches:
                    break
        
        return total_loss / max(num_batches, 1)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Baseline GPT-2 Training")
        print("="*80 + "\n")
        
        # Start energy monitoring
        energy_monitor = EnergyMonitor()
        energy_monitor.start()
        
        best_loss = float('inf')
        start_time = time.time()
        global_step = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*80}")
            
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(range(0, len(self.train_dataset), self.args.batch_size), 
                       desc="Training")
            
            for i in pbar:
                batch = self.train_dataset[i:i+self.args.batch_size]
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update energy monitor periodically
                if global_step % 10 == 0:
                    energy_monitor.update()
                
                pbar.set_postfix({'loss': loss.item()})
                
                # Evaluate periodically
                if global_step % self.args.eval_every == 0:
                    val_loss = self.evaluate()
                    print(f"\nStep {global_step} | Val Loss: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_loss,
                        }, self.exp_dir / 'best_model.pt')
                    
                    self.model.train()
                
                if num_batches >= self.args.max_batches_per_epoch:
                    break
            
            # End of epoch evaluation
            val_loss = self.evaluate()
            elapsed = time.time() - start_time
            
            # Update best loss if this is better
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, self.exp_dir / 'best_model.pt')
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {epoch_loss / num_batches:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best Val Loss: {best_loss:.4f}")
            print(f"  Time: {elapsed:.2f}s")
        
        # Final evaluation to ensure we have a recorded loss
        final_loss = self.evaluate()
        if final_loss < best_loss:
            best_loss = final_loss
        
        # Get final energy consumption
        total_energy_wh = energy_monitor.get_energy_wh()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Total Steps: {global_step}")
        print(f"Total Energy: {total_energy_wh:.4f} Wh")
        print(f"{'='*60}\n")
        
        # Save final results
        results = {
            'best_loss': float(best_loss),
            'final_loss': float(final_loss),
            'total_energy_wh': float(total_energy_wh),
            'final_epoch': self.args.num_epochs,
            'total_steps': global_step,
            'config': vars(self.args)
        }
        
        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Best Val Loss: {best_loss:.4f}")
        print(f"Results saved to: {self.exp_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train baseline GPT-2')
    
    # Model architecture
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--eval_batches', type=int, default=50)
    parser.add_argument('--max_batches_per_epoch', type=int, default=10000)
    
    # Experiment
    parser.add_argument('--exp_dir', type=str, default='./experiments/baseline')
    
    args = parser.parse_args()
    
    trainer = BaselineTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
