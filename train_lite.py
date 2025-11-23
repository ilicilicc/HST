# train_ultra.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import json
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import DataLoader, Dataset

# Import the updated model
from hst_v3_ultra import HSTv3Ultra 

# ==========================================================
# 1. Configuration (Set for Wikipedia Training)
# ==========================================================

# Data paths based on the provided directory structure
DATA_DIR = 'data/wiki' 
CHECKPOINT_DIR = 'checkpoints/ultra'
CONFIG_PATH = 'configs/hst_config.json'

# Dummy config content for a ready-to-run script. You must create 'configs/hst_config.json'.
# Assuming a standard small model configuration for testing
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Warning: {CONFIG_PATH} not found. Using default internal config.")
    config = {
        'vocab_size': 50257,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 8,
        'max_seq_len': 512,
        'horizon': 16
    }
    
# Training Specific Hyperparameters
WIKI_TRAIN_CONFIG = {
    'epochs': 1,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'max_seq_len': config['max_seq_len'],
    
    # Loss Weights (Crucial for HST-v3 stability - Section 8)
    'lambda_aux': 0.8,    # Weight for Horizon Loss (high)
    'lambda_closure': 0.01, # Weight for Learned Basis L1 Loss (low)
    
    **config 
}

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================================
# 2. Data Loading Utility (Using .bin files)
# ==========================================================

class WikiDataset(Dataset):
    """Loads tokenized .bin files from the Wikipedia pre-processed data."""
    def __init__(self, file_path: str, block_size: int):
        print(f"Loading data from {file_path}...")
        # Assume .bin files contain a single long sequence of token IDs
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        # The number of blocks that can be extracted from the data
        self.num_blocks = (len(self.data) - 1) // block_size 

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        # We need block_size + 1 elements to get input (block_size) and target (block_size)
        start = idx * self.block_size
        end = start + self.block_size + 1 
        
        # Ensure we don't read beyond the data array
        if end > len(self.data):
             # This should ideally not happen if num_blocks is calculated correctly, 
             # but acts as a safeguard.
             start = len(self.data) - self.block_size - 1
             end = len(self.data)
             
        # Input sequence (x) and target sequence (y)
        # x: data[t...T-1], y: data[t+1...T]
        x = torch.from_numpy(self.data[start:end-1].astype(np.int64))
        y = torch.from_numpy(self.data[start+1:end].astype(np.int64))
        return x, y

# ==========================================================
# 3. Loss and Training Functions
# ==========================================================

def calculate_loss(model_output: Dict, targets: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Calculates the total loss with all auxiliary terms (LM, Horizon, Closure).
    """
    # 1. Standard LM Loss (t+1 prediction)
    logits_t1 = model_output['logits'] 
    
    # Targets are already shifted for the LM task (y = x[1:])
    lm_loss = F.cross_entropy(
        logits_t1[:, :-1, :].reshape(-1, logits_t1.size(-1)), 
        targets[:, 1:].reshape(-1)
    )

    # 2. Auxiliary Horizon Loss (H=16 tokens)
    horizon_logits = model_output['horizon_logits'] # [B, H, V]
    horizon_size = horizon_logits.size(1)
    
    # Targets t+1 to t+H
    # The targets are targets[:, 1:] because targets itself is already shifted (y=x[1:])
    # We take the first 'horizon_size' tokens of the targets array
    horizon_target = targets[:, 1:1 + horizon_size].contiguous() 
    
    if horizon_target.size(1) == horizon_size:
        aux_loss = F.cross_entropy(
            horizon_logits.reshape(-1, horizon_logits.size(-1)),
            horizon_target.reshape(-1)
        )
    else:
        # Should be avoided by correct data handling, but padded to 0 if needed
        aux_loss = torch.tensor(0.0, device=targets.device)
        
    # 3. Algebraic Closure Loss
    closure_loss = model.get_closure_loss() # Retrieves the L1 penalty from LearnedHarmonicBasis

    # Total Loss (Weighted sum)
    total_loss = lm_loss + (config['lambda_aux'] * aux_loss) + (config['lambda_closure'] * closure_loss)
    
    loss_metrics = {
        'lm_loss': lm_loss.item(),
        'aux_loss': aux_loss.item(),
        'closure_loss': closure_loss.item(),
        'total_loss': total_loss.item(),
        'bottom_depth': model_output['bottom_depth']
    }
    
    return total_loss, loss_metrics


def train(model: HSTv3Ultra, train_loader: DataLoader, optimizer: optim.Optimizer, config: Dict):
    model.train()
    device = next(model.parameters()).device
    print(f"Starting Wikipedia Training on device: {device}...")
    
    for epoch in range(config['epochs']):
        for step, (x, y) in enumerate(train_loader):
            start_time = time.time()
            
            # Move data to device
            x, y = x.to(device), y.to(device) 
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(x, target_ids=y) 

            # Calculate Loss
            total_loss, metrics = calculate_loss(output, y, config)
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()
            
            end_time = time.time()
            step_time = end_time - start_time
            
            # Logging
            if step % 50 == 0:
                print(f"E: {epoch+1}, S: {step}/{len(train_loader)}, "
                      f"Time/Step: {step_time:.2f}s, "
                      f"Loss: {metrics['total_loss']:.4f} "
                      f"(LM:{metrics['lm_loss']:.4f} | AUX:{metrics['aux_loss']:.4f} | CLOSURE:{metrics['closure_loss']:.4f}), "
                      f"Depth: {metrics['bottom_depth']}")

        print(f"Epoch {epoch+1} finished. Saving checkpoint...")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'hst_v3_epoch_{epoch+1}.pt'))
        
    print("Training Complete.")


if __name__ == '__main__':
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Initialize Model
    model = HSTv3Ultra(
        vocab_size=WIKI_TRAIN_CONFIG['vocab_size'],
        d_model=WIKI_TRAIN_CONFIG['d_model'],
        n_heads=WIKI_TRAIN_CONFIG['n_heads'],
        n_layers=WIKI_TRAIN_CONFIG['n_layers'],
        max_seq_len=WIKI_TRAIN_CONFIG['max_seq_len'],
        horizon=WIKI_TRAIN_CONFIG['horizon']
    ).to(device)
    
    # 3. Load Data
    train_file = os.path.join(DATA_DIR, 'train.bin')
    train_dataset = WikiDataset(train_file, WIKI_TRAIN_CONFIG['max_seq_len'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=WIKI_TRAIN_CONFIG['batch_size'],
        shuffle=True,
        drop_last=True 
    )
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=WIKI_TRAIN_CONFIG['learning_rate'])
    
    # 5. Start Training
    train(model, train_loader, optimizer, WIKI_TRAIN_CONFIG)