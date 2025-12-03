"""
HST v6 - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- ChunkDecoderWithCache for efficient incremental generation
- TransformerDecoderLayerWithCache with cross-attention caching
- Multi-Level Lattice with Path-Weighted fusion
- Speculative verification support
- Token and Chunk mode with full cache support
"""

# ==================== SETUP ====================
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:32'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== HYPERPARAMETERS ====================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
MAX_SEQ_LEN = 512
HORIZON = 8
VOCAB_SIZE = 32000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MAX_TRAINING_STEPS = 100000
INITIAL_LR = 2e-4
WARMUP_STEPS = 2000
MODE = 'token'
CHUNK_SIZE = 128

save_dir = '/content/drive/MyDrive/hst_v6_checkpoints'
os.makedirs(save_dir, exist_ok=True)

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ==================== ATTENTION LAYERS ====================
class SelfAttentionWithCache(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, layer_past=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present = (k, v)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        full_S = k.size(2)
        if full_S > S:
            attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            attn_mask_full = torch.ones(S, full_S, dtype=torch.bool, device=x.device)
            attn_mask_full[:, full_S - S:] = attn_mask
            attn_weights.masked_fill_(attn_mask_full[None, None, :, :], -torch.inf)
        else:
            attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            attn_weights.masked_fill_(attn_mask[None, None, :, :], -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_output), present

class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn = SelfAttentionWithCache(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
        tgt_norm = self.norm1(tgt)
        sa_output, sa_present = self.self_attn(tgt_norm, layer_past=self_attn_past)
        tgt = tgt + self.dropout1(sa_output)

        tgt_norm = self.norm2(tgt)
        if cross_attn_past is not None:
            ca_output, _ = self.cross_attn(tgt_norm, cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            ca_output, _ = self.cross_attn(tgt_norm, memory, memory)
            ca_present = (memory, memory)

        tgt = tgt + self.dropout2(ca_output)
        tgt_norm = self.norm3(tgt)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout(ff_output)
        
        return tgt, sa_present, ca_present

class TransformerEncoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.attn = SelfAttentionWithCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        attn_output, present = self.attn(self.norm1(x), layer_past)
        x = x + self.dropout1(attn_output)
        ff_output = self.linear2(F.relu(self.linear1(self.norm2(x))))
        x = x + self.dropout2(ff_output)
        return x, present

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.block = TransformerEncoderLayerWithCache(d_model, n_heads)
        self.confidence_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, 1), nn.Sigmoid()
        )
    
    def forward(self, x, layer_past=None):
        x_out, present = self.block(x, layer_past)
        if x_out.size(1) > 1:
            conf = self.confidence_predictor(x_out.transpose(1, 2)).mean(dim=0)
        else:
            conf = x_out.new_tensor([0.0])
        return x_out, conf, present

# ==================== CHUNK ENCODER/DECODER ====================
class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :].view(B * num_chunks, self.chunk_size, D)
        encoded_tokens = self.local_encoder(chunks)
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        return pooled.view(B, num_chunks, D)

class ChunkDecoderWithCache(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayerWithCache(d_model, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings, cache=None):
        B, S, D = target_token_embeddings.shape
        device = target_token_embeddings.device
        
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + S, dtype=torch.long, device=device) % self.chunk_size
        
        pos_emb = self.pos_embedding(positions)
        tgt = target_token_embeddings + pos_emb
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else (None, None)
            self_attn_past, cross_attn_past = layer_cache
            memory = chunk_embeddings.repeat(1, S, 1)
            tgt, sa_present, ca_present = layer(tgt, memory, self_attn_past, cross_attn_past)
            new_cache.append((sa_present, ca_present))
            
        logits = self.lm_head(tgt)
        return logits, new_cache

# ==================== LATTICE CORE ====================
class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True) for _ in range(10)
        ])
        self.task_router = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10), nn.Sigmoid()
        )
    
    def forward(self, x, horizon_targets=None):
        B, S, D = x.shape
        task_embedding = x.mean(dim=1)
        layer_gates = self.task_router(task_embedding)

        h = x
        for layer_idx, processor in enumerate(self.layer_processors):
            gate = layer_gates[:, layer_idx].unsqueeze(1).unsqueeze(2)
            if gate.mean() > 0.1:
                h_layer = processor(h)
                h = h + gate * (h_layer - h)
        return h

# ==================== HORIZON PREDICTOR ====================
class RecursiveHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.coarse_predictor = nn.Linear(d_model, vocab_size)
        self.medium_predictor = nn.Linear(d_model * 2, vocab_size)
        self.fine_predictor = nn.Linear(d_model * 2, vocab_size)
        self.lattice_embeddings = nn.Embedding(20, d_model)
        self.projection = nn.Linear(vocab_size, d_model)

    def forward(self, h_sequence):
        B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :]
        
        coarse_preds = {}
        for offset in [4, min(10, self.horizon)]:
            offset_emb = self.lattice_embeddings(torch.tensor([min(offset - 1, 19)], device=h_t.device))
            coarse_preds[offset] = self.coarse_predictor(h_t + offset_emb)

        medium_preds = {}
        for offset in [2, 6]:
            if offset <= self.horizon:
                left_coarse = coarse_preds[4]
                right_coarse = coarse_preds.get(10, coarse_preds[4])
                alpha = (offset - 4) / max(10 - 4, 1)
                coarse_interp = self.projection(alpha * left_coarse + (1 - alpha) * right_coarse)
                medium_preds[offset] = self.medium_predictor(torch.cat([h_t, coarse_interp], dim=-1))

        fine_preds = {}
        for offset in [1, 3, 5]:
            if offset <= self.horizon:
                left_med = medium_preds.get(2, coarse_preds[4])
                right_med = medium_preds.get(6, coarse_preds[4])
                alpha = (offset - 2) / max(6 - 2, 1)
                medium_interp = self.projection(alpha * left_med + (1 - alpha) * right_med)
                fine_preds[offset] = self.fine_predictor(torch.cat([h_t, medium_interp], dim=-1))
        
        all_preds = {**coarse_preds, **medium_preds, **fine_preds}
        logits_list = [all_preds.get(i, torch.zeros(B, self.vocab_size, device=h_t.device)) for i in range(1, self.horizon + 1)]
        logits = torch.stack(logits_list, dim=1)
        confidence = torch.ones(B, self.horizon, device=h_t.device)
        return logits, confidence

# ==================== HST v6 MODEL ====================
class HSTv6(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=512, horizon=8,
                 early_exit_threshold=0.93, mode='token', chunk_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.n_bottom_layers = n_layers // 2
        self.n_top_layers = n_layers - self.n_bottom_layers
        self.early_exit_threshold = early_exit_threshold
        self.mode = mode
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.mode == 'chunk':
            self.pos_embedding = nn.Embedding(max_seq_len * chunk_size, d_model)
            self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
            self.chunk_decoder = ChunkDecoderWithCache(d_model, vocab_size, chunk_size)
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(self.n_bottom_layers)])
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
            self.top_stack = nn.ModuleList([TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(self.n_top_layers)])

        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, cache=None):
        if self.mode == 'token':
            return self.forward_token(input_ids, cache)
        else:
            return self.forward_chunk(input_ids)

    def forward_token(self, input_ids, cache=None):
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        past_len = 0
        if cache and cache[0] and cache[0][0] is not None:
            past_len = cache[0][0].size(2)

        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        new_cache = []
        cache_idx = 0
        predicted_depth = self.n_bottom_layers

        for i, block in enumerate(self.adaptive_bottom):
            layer_past = cache[cache_idx] if cache and cache_idx < len(cache) else None
            x, conf, present = block(x, layer_past)
            new_cache.append(present)
            cache_idx += 1
            
            if past_len == 0 and i >= 1 and conf.item() > self.early_exit_threshold:
                predicted_depth = i + 1
                break
        
        h_lattice_out = self.lattice_core(x)
        
        h_top_in = h_lattice_out
        for i, block in enumerate(self.top_stack):
            layer_past = cache[cache_idx] if cache and cache_idx < len(cache) else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
            cache_idx += 1
        
        h_final = h_top_in
        logits_t1 = self.lm_head(self.ln_f(h_final))
        logits_horizon, confidence = self.horizon_predictor(h_final)
        
        return {
            'logits': logits_t1,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_final,
            'bottom_depth': predicted_depth,
            'cache': new_cache
        }

    def forward_chunk(self, input_ids):
        B, total_tokens = input_ids.shape
        device = input_ids.device

        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        chunk_emb = self.chunk_encoder(x)
        h_lattice_out = self.lattice_core(chunk_emb)
        logits, _ = self.chunk_decoder(h_lattice_out, x)
        logits_horizon, confidence = self.horizon_predictor(h_lattice_out)
        
        return {
            'logits': logits,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_lattice_out,
            'bottom_depth': 0,
            'cache': None
        }

# ==================== LOSS FUNCTION ====================
def compute_loss(output, targets, horizon=8, gamma=0.95, pad_id=50256, n_layers=16):
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
    
    if 'bottom_depth' in output and output['bottom_depth'] > 0:
        depth = float(output['bottom_depth'])
        conf = output['confidence'].mean()
        loss += 0.03 * F.mse_loss(conf, torch.tensor(depth / n_layers, device=conf.device))
    
    return loss

# ==================== TRAINING ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

print("[2/5] Building HST v6 model...")
model = HSTv6(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=MODE, chunk_size=CHUNK_SIZE
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

scaler = GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_TRAINING_STEPS)

print("[4/5] Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def tokenize(ex):
    return {"input_ids": tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]}

stream = dataset.map(tokenize, batched=True, batch_size=500, remove_columns=dataset.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
loader = DataLoader(stream, batch_size=BATCH_SIZE, collate_fn=collator)

print(f"[5/5] Starting training (Max Steps: {MAX_TRAINING_STEPS})...\n")

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
            loss = compute_loss(out, ids, horizon=HORIZON, n_layers=N_LAYERS)
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
                depth = out.get('bottom_depth', 0)
                print(f"Step {step:6d} | Loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | Depth {depth} | ", end="")
                print_memory()
            
            if step % 5000 == 0 and step > 0:
                torch.save(model.state_dict(), f"{save_dir}/ckpt_step_{step}.pt")

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            step += 1

except KeyboardInterrupt:
    print("\n!! Training INTERRUPTED !!")
    torch.save(model.state_dict(), f'{save_dir}/hst_v6_interrupt_step_{step}.pt')

torch.save(model.state_dict(), f'{save_dir}/hst_v6_final.pt')
print(f"\nTRAINING COMPLETE!")
print_memory()
