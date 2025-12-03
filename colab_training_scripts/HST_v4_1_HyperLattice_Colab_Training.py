"""
HST v4.1 HyperLattice - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- Dynamic Hyper-Lattice Block with learnable lattice weights
- Paged KV Cache for efficient memory usage
- Self-Attention with Paged Cache support
- Harmonic Horizon Predictor
- Graceful interrupt handling
"""

# ==================== SETUP & INSTALLATION ====================
# !pip install torch transformers datasets bitsandbytes accelerate -q

import os
import gc
import numpy as np
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from datasets import load_dataset

# ==================== MEMORY OPTIMIZATION FOR T4 ====================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:32'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== T4-OPTIMIZED HYPERPARAMETERS ====================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
LATTICE_DEPTH = 48
MAX_SEQ_LEN = 512
HORIZON = 8
VOCAB_SIZE = 32000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MAX_TRAINING_STEPS = 100000
INITIAL_LR = 2e-4
WARMUP_STEPS = 2000

save_dir = '/content/drive/MyDrive/hst_v4_1_checkpoints'
os.makedirs(save_dir, exist_ok=True)

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ==================== PAGED KV CACHE ====================
class PagedKVCache:
    def __init__(self, head_dim, n_heads, page_size=256, max_pages=32, device='cuda'):
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device
        
        self.k_pages = []
        self.v_pages = []
        self.current_length = 0
    
    def append(self, k_new, v_new):
        B, H, S, D = k_new.shape
        
        if not self.k_pages:
            self.k_pages.append(k_new)
            self.v_pages.append(v_new)
            self.current_length = S
            return
        
        last_k = self.k_pages[-1]
        last_len = last_k.size(2)
        
        if last_len + S <= self.page_size:
            self.k_pages[-1] = torch.cat([last_k, k_new], dim=2)
            self.v_pages[-1] = torch.cat([self.v_pages[-1], v_new], dim=2)
        else:
            self.k_pages.append(k_new)
            self.v_pages.append(v_new)
        
        self.current_length += S
        
        while len(self.k_pages) > self.max_pages:
            self.k_pages.pop(0)
            self.v_pages.pop(0)
    
    def get_full_kv(self):
        if not self.k_pages:
            return None, None
        k = torch.cat(self.k_pages, dim=2)
        v = torch.cat(self.v_pages, dim=2)
        return k, v
    
    def get_length(self):
        return self.current_length

# ==================== HYPER-LATTICE BLOCK ====================
class HyperLatticeBlock(nn.Module):
    def __init__(self, d_model, lattice_depth=48):
        super().__init__()
        self.d_model = d_model
        self.lattice_depth = lattice_depth
        
        self.lattice_weights = nn.Parameter(torch.randn(lattice_depth, lattice_depth) * 0.02)
        
        self.node_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, S, D = x.shape
        
        L = min(S, self.lattice_depth)
        lattice_adj = torch.softmax(self.lattice_weights[:L, :L], dim=-1)
        
        x_transformed = self.node_transform(x[:, :L, :])
        
        messages = []
        for i in range(L):
            neighbors = []
            weights = []
            for j in range(L):
                if lattice_adj[i, j] > 0.01:
                    neighbors.append(j)
                    weights.append(lattice_adj[i, j])
            
            if neighbors:
                neighbor_feats = x_transformed[:, neighbors, :]
                weight_tensor = torch.tensor(weights, device=x.device).view(1, -1, 1)
                weighted_neighbors = (neighbor_feats * weight_tensor).sum(dim=1)
                
                edge_input = torch.cat([x_transformed[:, i, :], weighted_neighbors], dim=-1)
                msg = self.edge_mlp(edge_input)
                messages.append(msg)
            else:
                messages.append(x_transformed[:, i, :])
        
        message_stack = torch.stack(messages, dim=1)
        
        gate_input = torch.cat([x[:, :L, :], message_stack], dim=-1)
        g = self.gate(gate_input)
        
        updated = g * self.output_proj(message_stack) + (1 - g) * x[:, :L, :]
        
        if S > L:
            output = torch.cat([updated, x[:, L:, :]], dim=1)
        else:
            output = updated
        
        return output

# ==================== SELF-ATTENTION WITH PAGED CACHE ====================
class SelfAttentionWithPagedCache(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, cache=None):
        B, S_new, D = x.shape
        
        q = self.q_proj(x).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
        
        if cache is not None:
            cache.append(k_new, v_new)
            k, v = cache.get_full_kv()
        else:
            k, v = k_new, v_new
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        S_full = k.size(2)
        if cache is not None and S_new == 1:
            pass
        else:
            attn_mask = torch.triu(torch.ones(S_new, S_full, dtype=torch.bool, device=x.device), diagonal=S_full - S_new + 1)
            attn_weights.masked_fill_(attn_mask[None, None, :, :], -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S_new, D)
        
        return self.out_proj(attn_output)

# ==================== HARMONIC HORIZON PREDICTOR ====================
class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.horizon_projection = nn.Linear(d_model, d_model * horizon)
        self.prediction_head = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, horizon)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_last = x[:, -1, :]
        
        projected = self.horizon_projection(x_last).view(-1, self.horizon, self.d_model)
        logits_list = self.prediction_head(projected)
        confidence = torch.sigmoid(self.confidence_head(x_last))
        
        return logits_list, confidence

# ==================== HST v4.1 HYPER-LATTICE MODEL ====================
class HSTv4HyperLattice(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, lattice_depth=48, max_seq_len=512, horizon=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.horizon = horizon
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.bottom_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True)
            for _ in range(n_layers // 2)
        ])
        
        self.hyper_lattice = HyperLatticeBlock(d_model, lattice_depth)
        
        self.top_stack = nn.ModuleList([
            SelfAttentionWithPagedCache(d_model, n_heads)
            for _ in range(n_layers // 2)
        ])
        
        self.top_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4*d_model),
                nn.GELU(),
                nn.Linear(4*d_model, d_model),
                nn.Dropout(0.1)
            ) for _ in range(n_layers // 2)
        ])
        self.top_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers // 2)])
        self.top_norms_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers // 2)])

        self.harmonic_horizon_predictor = HarmonicHorizonPredictor(d_model, vocab_size, horizon=horizon)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, caches=None):
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        past_len = caches[0].get_length() if caches else 0
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        for block in self.bottom_stack:
            x = block(x)
        
        x = self.hyper_lattice(x)
        
        if caches is None:
            head_dim = self.d_model // self.n_heads
            caches = [PagedKVCache(head_dim, self.n_heads, device=device) for _ in range(len(self.top_stack))]
        
        for i, (attn, ffn, norm1, norm2) in enumerate(zip(self.top_stack, self.top_ffns, self.top_norms, self.top_norms_2)):
            normed_x = norm1(x)
            attn_out = attn(normed_x, cache=caches[i])
            x = x + attn_out
            x = x + ffn(norm2(x))
        
        h_final = x
        
        logits = self.lm_head(self.ln_f(h_final))
        horizon_logits, confidence = self.harmonic_horizon_predictor(h_final)
        
        return {
            'logits': logits,
            'horizon_logits': horizon_logits,
            'confidence': confidence,
            'caches': caches
        }

# ==================== LOSS FUNCTION ====================
def compute_loss(output, targets, horizon=8, gamma=0.95, pad_id=50256):
    logits = output['logits']
    B, S = targets.shape
    V = logits.size(-1)
    
    logits = logits[:, :S]
    pred_logits = logits[:, :-1].reshape(-1, V)
    pred_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(pred_logits, pred_targets, ignore_index=pad_id)
    
    if 'horizon_logits' in output:
        horizon_logits = output['horizon_logits']
        H = horizon_logits.size(1)
        for k in range(1, min(H + 1, 5)):
            if k < S:
                h_logits_k = horizon_logits[:, k-1, :]
                h_targets_k = targets[:, min(k, S-1)]
                loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
    
    return loss

# ==================== INITIALIZE ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded: {len(tokenizer)} tokens")

print("[2/5] Building HST v4.1 HyperLattice model (T4 optimized)...")
model = HSTv4HyperLattice(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    lattice_depth=LATTICE_DEPTH,
    max_seq_len=MAX_SEQ_LEN,
    horizon=HORIZON
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")
print_memory()

print("\n[3/5] Setting up optimizer...")
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), weight_decay=0.01)
    print("Using 8-bit AdamW")
except ImportError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    print("Using standard AdamW")

scaler = GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_TRAINING_STEPS)

print("[4/5] Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def tokenize(ex):
    t = tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    return {"input_ids": t}

stream = dataset.map(tokenize, batched=True, batch_size=500, remove_columns=dataset.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
loader = DataLoader(stream, batch_size=BATCH_SIZE, collate_fn=collator)

print(f"[5/5] Starting training (Max Steps: {MAX_TRAINING_STEPS})...\n")

# ==================== TRAINING LOOP ====================
model.train()
step = 0
grad_acc_step = 0

try:
    for batch in loader:
        if step >= MAX_TRAINING_STEPS:
            break
        
        ids = batch["input_ids"].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            out = model(ids)
            loss = compute_loss(out, ids, horizon=HORIZON)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        grad_acc_step += 1
        
        if grad_acc_step % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % 100 == 0:
                print(f"Step {step:6d} | Loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | ", end="")
                print_memory()
            
            if step % 5000 == 0 and step > 0:
                torch.save(model.state_dict(), f"{save_dir}/ckpt_step_{step}.pt")
                print(f"  Checkpoint saved at step {step}")

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            step += 1

except KeyboardInterrupt:
    print("\n\n!! Training INTERRUPTED !!")
    torch.save(model.state_dict(), f'{save_dir}/hst_v4_1_interrupt_step_{step}.pt')
    print(f"Model saved at step {step}")

torch.save(model.state_dict(), f'{save_dir}/hst_v4_1_final.pt')
print(f"\nTRAINING COMPLETE - Model saved!")
print_memory()
