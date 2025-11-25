import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from hst_v5_2_unified import HSTv5_2Unified
import time
import numpy as np
from typing import Dict, Tuple

# ==========================================================
# 1. HIERARCHICAL PREDICTIVE LOSS (from data_mapping.pdf)
# ==========================================================
class HierarchicalPredictiveLoss(nn.Module):
    """
    Calculates a weighted average of three loss components:
    1. Standard Cross-Entropy loss for the next token (t+1).
    2. Horizon loss for future tokens (t+2, t+3, ...).
    3. Consistency loss to ensure short-term and long-term predictions align.
    (THEORY-COMPLIANT IMPLEMENTATION)
    """
    def __init__(self, vocab_size, alpha=0.6, beta=0.3, gamma=0.1, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for standard CE loss
        self.beta = beta    # Weight for horizon loss
        self.gamma = gamma  # Weight for consistency loss
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, model_output: Dict, targets: torch.Tensor, horizon_targets: torch.Tensor) -> Dict:
        logits_t1 = model_output['logits']
        horizon_logits = model_output['horizon_logits']

        # 1. Standard Cross-Entropy Loss (for t+1)
        # Reshape for CrossEntropyLoss: [B * S, V] and [B * S]
        loss_t1 = self.ce_loss(logits_t1.view(-1, self.vocab_size), targets.view(-1))

        # 2. Horizon Loss (for t+2 to t+H)
        B, S, V = logits_t1.shape
        H = horizon_logits.size(2)

        # Reshape horizon logits and targets for loss calculation
        # horizon_logits: [B, S, H, V] -> [B*S, H, V]
        # horizon_targets: [B, S, H] -> [B*S, H]
        horizon_logits_flat = horizon_logits.view(-1, H, self.vocab_size)
        horizon_targets_flat = horizon_targets.view(-1, H)

        loss_horizon = 0.0
        for i in range(H):
            # Apply a discount factor for more distant tokens
            discount = 0.95 ** i
            loss_horizon += discount * self.ce_loss(
                horizon_logits_flat[:, i, :],
                horizon_targets_flat[:, i]
            )
        loss_horizon = loss_horizon / H

        # 3. Consistency Loss (KL Divergence)
        # Ensure the t+2 prediction from the horizon head aligns with the t+1 prediction one step later.
        # This requires a second forward pass in a real scenario, but we can approximate it here.

        # We'll use the prediction for t+2 from the horizon head at time t
        pred_t2_from_horizon = F.log_softmax(horizon_logits[:, :-1, 0, :], dim=-1) # [B, S-1, V]

        # And compare it with the main prediction for t+1 at time t+1 (which is t+2 from t's perspective)
        pred_t1_at_tplus1 = F.softmax(logits_t1[:, 1:, :], dim=-1) # [B, S-1, V]

        loss_consistency = self.kl_loss(pred_t2_from_horizon.reshape(-1, self.vocab_size),
                                      pred_t1_at_tplus1.reshape(-1, self.vocab_size))

        # Combine losses
        total_loss = self.alpha * loss_t1 + self.beta * loss_horizon + self.gamma * loss_consistency

        return {
            'total_loss': total_loss,
            'ce_loss': loss_t1,
            'horizon_loss': loss_horizon,
            'consistency_loss': loss_consistency
        }

# ==========================================================
# 2. DATASET & TRAINING INFRASTRUCTURE
# ==========================================================
class SyntheticDataset(Dataset):
    """Generates synthetic data for training the HST model."""
    def __init__(self, num_samples, seq_len, vocab_size, horizon):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.horizon = horizon

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random sequence
        input_seq = torch.randint(0, self.vocab_size, (self.seq_len,))

        # Target is the input sequence shifted by 1
        targets_t1 = torch.roll(input_seq, -1)
        targets_t1[-1] = 0 # Padding

        # Horizon targets are shifted versions of the input sequence
        horizon_targets = torch.zeros(self.seq_len, self.horizon, dtype=torch.long)
        for i in range(self.horizon):
            # Target for horizon `i` is the sequence shifted by `i+2`
            shifted = torch.roll(input_seq, -(i + 2))
            shifted[-(i+2):] = 0 # Padding
            horizon_targets[:, i] = shifted

        return input_seq, targets_t1, horizon_targets

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler, curriculum_params):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (inputs, targets_t1, horizon_targets) in enumerate(dataloader):
        inputs, targets_t1, horizon_targets = inputs.to(device), targets_t1.to(device), horizon_targets.to(device)

        optimizer.zero_grad()

        # Adjust sequence length based on curriculum
        current_seq_len = int(inputs.size(1) * curriculum_params['seq_len_factor'])
        inputs_curr = inputs[:, :current_seq_len]
        targets_t1_curr = targets_t1[:, :current_seq_len]
        horizon_targets_curr = horizon_targets[:, :current_seq_len, :]

        model_output = model(inputs_curr, horizon_targets=horizon_targets_curr)

        loss_dict = loss_fn(model_output, targets_t1_curr, horizon_targets_curr)
        loss = loss_dict['total_loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss / len(dataloader)

    print(f"Epoch finished in {epoch_time:.2f}s. Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets_t1, horizon_targets in dataloader:
            inputs, targets_t1, horizon_targets = inputs.to(device), targets_t1.to(device), horizon_targets.to(device)
            model_output = model(inputs, horizon_targets=horizon_targets)
            loss_dict = loss_fn(model_output, targets_t1, horizon_targets)
            total_loss += loss_dict['total_loss'].item()

    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return avg_loss

def benchmark_generation(model, device, prompt_len=16, max_new_tokens=128):
    model.eval()
    prompt = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)

    print("\n--- Benchmarking Ultra-Fast Generation ---")
    start_time = time.time()

    # Ensure the model is in 'token' mode for generation
    initial_mode = model.mode
    model.mode = 'token'

    generated_ids, stats = model.generate_ultra_fast(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50
    )

    model.mode = initial_mode # Restore original mode

    end_time = time.time()

    duration = end_time - start_time
    tokens_generated = stats['tokens_generated']
    speed = tokens_generated / duration

    print(f"Generated {tokens_generated} tokens in {duration:.2f}s.")
    print(f"Generation Speed: {speed:.2f} tokens/sec")
    print(f"Acceptance Rate: {stats['acceptance_rate']:.2f}")
    print(f"Effective Speedup: x{stats['effective_speedup']:.2f}")
    print("-" * 40)

    return speed

# ==========================================================
# 3. MAIN TRAINING SCRIPT
# ==========================================================
def main():
    # --- Config ---
    # Model params
    vocab_size = 50257
    d_model = 512
    n_heads = 8
    n_layers = 12
    horizon = 16
    max_seq_len = 2048

    # Training params
    mode = 'chunk' # 'token' or 'chunk'
    chunk_size = 128
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = HSTv5_2Unified(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len if mode == 'token' else max_seq_len // chunk_size,
        horizon=horizon,
        mode=mode,
        chunk_size=chunk_size,
        use_adaptive_processor=(mode == 'chunk') # Use adaptive for chunks
    ).to(device)

    print(f"Model created in '{mode}' mode. Num params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- Data ---
    train_seq_len = max_seq_len if mode == 'token' else max_seq_len * 4 # More data for chunk mode
    train_dataset = SyntheticDataset(num_samples=100, seq_len=train_seq_len, vocab_size=vocab_size, horizon=horizon)
    val_dataset = SyntheticDataset(num_samples=20, seq_len=train_seq_len, vocab_size=vocab_size, horizon=horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                            steps_per_epoch=len(train_loader), epochs=num_epochs)
    loss_fn = HierarchicalPredictiveLoss(vocab_size=vocab_size)

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Curriculum Learning: gradually increase sequence length and reduce loss component weights
        seq_len_factor = min(1.0, 0.5 + 0.5 * (epoch / (num_epochs - 1)))
        loss_fn.alpha = max(0.4, 0.7 - 0.3 * (epoch / (num_epochs - 1)))
        loss_fn.beta = min(0.4, 0.2 + 0.2 * (epoch / (num_epochs - 1)))

        curriculum_params = {'seq_len_factor': seq_len_factor}
        print(f"Curriculum: Seq Len Factor={seq_len_factor:.2f}, Loss Alpha={loss_fn.alpha:.2f}, Beta={loss_fn.beta:.2f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler, curriculum_params)
        val_loss = validate(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best validation loss. Saving checkpoint...")
            torch.save(model.state_dict(), f"hst_v5_2_{mode}_best.pt")

    # --- Final Actions ---
    print("\nTraining complete.")
    # Load best model for final benchmark
    model.load_state_dict(torch.load(f"hst_v5_2_{mode}_best.pt"))
    benchmark_generation(model, device)

if __name__ == '__main__':
    main()
