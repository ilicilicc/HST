"""
HST v7.1.2 Agile - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- Adaptive Bottom Transformer with early exit
- Recursive Descent Lattice Analyzer
- Adaptive Lattice Processor with task routing
- Recursive Horizon Predictor (coarse -> medium -> fine)
- Speculative Verifier for efficient generation
- Chunk mode with decoder cache support
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

save_dir = '/content/drive/MyDrive/hst_v7_1_2_checkpoints'
os.makedirs(save_dir, exist_ok=True)

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ==================== CHUNK ENCODER ====================
class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.pooler = nn.Sequential(
            nn.Linear(d_model * chunk_size, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, token_embeddings):
        B, S, D = token_embeddings.shape
        num_chunks = S // self.chunk_size
        trimmed_S = num_chunks * self.chunk_size
        token_embeddings = token_embeddings[:, :trimmed_S, :]
        reshaped = token_embeddings.view(B, num_chunks, self.chunk_size * D)
        return self.pooler(reshaped)

# ==================== SPECULATIVE VERIFIER ====================
class SpeculativeVerifier(nn.Module):
    def __init__(self, d_model, n_layers, horizon, vocab_size, n_heads=8):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        
        self.draft_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(horizon, d_model)
        
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(min(n_layers, 4))
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            for _ in range(min(n_layers, 4))
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(min(n_layers, 4) * 2)])
        
        self.output_head = nn.Linear(d_model, vocab_size)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
    
    def forward(self, draft_tokens, hidden_states):
        B = draft_tokens.size(0)
        H = min(draft_tokens.size(1), self.horizon)
        
        x = self.draft_embedding(draft_tokens[:, :H])
        pos = self.pos_embedding(torch.arange(H, device=draft_tokens.device))
        x = x + pos
        
        memory = hidden_states[:, -1:, :] if hidden_states.size(1) > 0 else hidden_states
        
        norm_idx = 0
        for attn, ffn in zip(self.cross_attn_layers, self.ffn_layers):
            x_norm = self.layer_norms[norm_idx](x)
            attn_out, _ = attn(x_norm, memory, memory)
            x = x + attn_out
            norm_idx += 1
            
            x_norm = self.layer_norms[norm_idx](x)
            x = x + ffn(x_norm)
            norm_idx += 1
        
        logits = self.output_head(x)
        confidence = self.confidence_head(x).squeeze(-1)
        
        return logits, confidence

# ==================== TRANSFORMER LAYERS WITH CACHE ====================
class TransformerEncoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        B, S, D = x.shape
        
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present = (k, v)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        S_full = k.size(2)
        if S > 1:
            mask = torch.triu(torch.ones(S, S_full, dtype=torch.bool, device=x.device), diagonal=S_full - S + 1)
            attn.masked_fill_(mask[None, None, :, :], float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)
        
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.linear2(F.relu(self.linear1(self.norm2(x)))))
        
        return x, present

# ==================== ADAPTIVE BOTTOM TRANSFORMER ====================
class AdaptiveBottomTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers_max, dropout=0.1, early_exit_threshold=0.93):
        super().__init__()
        self.num_layers_max = num_layers_max
        self.early_exit_threshold = early_exit_threshold
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithCache(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers_max)
        ])
        
        self.confidence_predictors = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, 1), nn.Sigmoid())
            for _ in range(num_layers_max)
        ])
    
    def forward(self, x, cache=None):
        B, S, D = x.shape
        new_cache = []
        predicted_depth = self.num_layers_max
        
        for i, (layer, conf_pred) in enumerate(zip(self.layers, self.confidence_predictors)):
            layer_past = cache[i] if cache and i < len(cache) else None
            x, present = layer(x, layer_past)
            new_cache.append(present)
            
            if S > 1:
                conf = conf_pred(x.transpose(1, 2)).mean()
                if i >= 1 and conf.item() > self.early_exit_threshold:
                    predicted_depth = i + 1
                    break
        
        return x, predicted_depth, new_cache

# ==================== RECURSIVE DESCENT LATTICE ANALYZER ====================
class RecursiveDescentLatticeAnalyzer(nn.Module):
    def __init__(self, max_seq_len=8192):
        super().__init__()
        spine_list = self._generate_spine_list(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine_list, dtype=torch.long))
        self.descent_paths = self._compute_descent_paths()
        self.layer_weights = nn.Parameter(torch.ones(10))

    def _generate_spine_list(self, max_len):
        spine = [0, 2, 4]
        while True:
            next_val = 2 * spine[-1] + 2 * spine[-2] + 2 * spine[-3]
            if next_val >= max_len:
                break
            spine.append(next_val)
        return spine

    def _find_parent(self, pos):
        if pos in self.spine:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx > 0:
                return self.spine[idx-1].item()
        left_spine = self.spine[self.spine < pos]
        if len(left_spine) > 0:
            return left_spine[-1].item()
        return 0

    def _compute_descent_paths(self):
        paths = {}
        for pos_tensor in self.spine:
            pos = pos_tensor.item()
            path = []
            current = pos
            layer = 0
            while current > 0 and layer < 10:
                parent = self._find_parent(current)
                path.append((layer, parent))
                if current == parent:
                    break
                current = parent
                layer += 1
            paths[pos] = path
        return paths

    def compute_predictive_field(self, pos, target_offset):
        try:
            source_spine_idx = (self.spine == pos).nonzero(as_tuple=True)[0]
            target_spine_idx = (self.spine == (pos + target_offset)).nonzero(as_tuple=True)[0]
            spine_distance = abs(target_spine_idx - source_spine_idx)
        except:
            spine_distance = int(np.log2(target_offset + 1))

        layer_importance = torch.zeros(10, device=self.layer_weights.device)
        if spine_distance > 5:
            layer_importance[0:3] = torch.tensor([1.0, 0.8, 0.5])
        elif spine_distance > 2:
            layer_importance[1:5] = torch.tensor([0.5, 1.0, 0.8, 0.3])
        else:
            layer_importance[3:7] = torch.tensor([0.3, 0.8, 1.0, 0.8])
        
        layer_importance = layer_importance * torch.sigmoid(self.layer_weights)
        return layer_importance

# ==================== ADAPTIVE LATTICE PROCESSOR ====================
class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(10)
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

# ==================== RECURSIVE HORIZON PREDICTOR ====================
class RecursiveHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.coarse_predictor = nn.Linear(d_model, vocab_size)
        self.medium_predictor = nn.Linear(d_model + d_model, vocab_size)
        self.fine_predictor = nn.Linear(d_model + d_model, vocab_size)
        self.lattice_embeddings = nn.Embedding(20, d_model)
        self.projection = nn.Linear(vocab_size, d_model)

    def forward(self, h_sequence):
        B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :]
        
        coarse_offsets = [4, min(10, self.horizon)]
        coarse_preds = {}
        for offset in coarse_offsets:
            offset_emb = self.lattice_embeddings(torch.tensor([min(offset - 1, 19)], device=h_t.device))
            h_augmented = h_t + offset_emb
            coarse_preds[offset] = self.coarse_predictor(h_augmented)

        medium_offsets = [2, 6]
        medium_preds = {}
        for offset in medium_offsets:
            if offset <= self.horizon:
                left_coarse = coarse_preds[4]
                right_coarse = coarse_preds.get(10, coarse_preds[4])
                alpha = (offset - 4) / max(10 - 4, 1)
                coarse_interp = self.projection(alpha * left_coarse + (1 - alpha) * right_coarse)
                h_interpolated = torch.cat([h_t, coarse_interp], dim=-1)
                medium_preds[offset] = self.medium_predictor(h_interpolated)

        fine_offsets = [1, 3, 5]
        fine_preds = {}
        for offset in fine_offsets:
            if offset <= self.horizon:
                left_med = medium_preds.get(2, coarse_preds[4])
                right_med = medium_preds.get(6, coarse_preds[4])
                alpha = (offset - 2) / max(6 - 2, 1)
                medium_interp = self.projection(alpha * left_med + (1-alpha) * right_med)
                h_interpolated = torch.cat([h_t, medium_interp], dim=-1)
                fine_preds[offset] = self.fine_predictor(h_interpolated)
        
        all_preds = {**coarse_preds, **medium_preds, **fine_preds}
        
        logits_list = []
        for i in range(1, self.horizon + 1):
            if i in all_preds:
                logits_list.append(all_preds[i])
            else:
                logits_list.append(torch.zeros(B, self.vocab_size, device=h_t.device))
        
        logits = torch.stack(logits_list, dim=1)
        confidence = torch.ones(B, self.horizon, device=h_t.device)
        
        return logits, confidence

# ==================== HST v7.1.2 AGILE MODEL ====================
class HSTv7Agile(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=512, horizon=8,
                 early_exit_confidence_threshold=0.93, mode='token', chunk_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.n_bottom_layers = n_layers // 2
        self.n_top_layers = n_layers - self.n_bottom_layers
        self.early_exit_confidence_threshold = early_exit_confidence_threshold
        self.mode = mode
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.mode == 'chunk':
            self.pos_embedding = nn.Embedding(max_seq_len * chunk_size, d_model)
            self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = AdaptiveBottomTransformer(
                d_model, n_heads, self.n_bottom_layers, early_exit_threshold=early_exit_confidence_threshold
            )
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
            self.top_stack = nn.ModuleList([
                TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(self.n_top_layers)
            ])

        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.speculative_verifier = SpeculativeVerifier(d_model, self.n_top_layers, horizon, vocab_size, n_heads)

    def forward(self, input_ids, cache=None, horizon_targets=None):
        if self.mode == 'token':
            return self.forward_token(input_ids, cache)
        else:
            return self.forward_chunk(input_ids, horizon_targets)

    def forward_token(self, input_ids, cache=None):
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        past_len = 0
        if cache and cache[0] and cache[0][0] is not None:
            past_len = cache[0][0].size(2)

        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        bottom_cache = cache[:self.n_bottom_layers] if cache else None
        h_bottom, predicted_depth, bottom_new_cache = self.adaptive_bottom(x, cache=bottom_cache)

        h_lattice_out = self.lattice_core(h_bottom)
        
        h_top_in = h_lattice_out
        new_cache = bottom_new_cache
        top_stack_cache = cache[self.n_bottom_layers:] if cache else None
        
        for i, block in enumerate(self.top_stack):
            layer_past = top_stack_cache[i] if top_stack_cache and i < len(top_stack_cache) else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
        
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

    def forward_chunk(self, input_ids, horizon_targets=None):
        B, total_tokens = input_ids.shape
        device = input_ids.device

        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        chunk_emb = self.chunk_encoder(x)
        h_lattice_out = self.lattice_core(chunk_emb)
        
        logits = self.lm_head(self.ln_f(h_lattice_out))
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

# ==================== INITIALIZE ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded: {len(tokenizer)} tokens")

print("[2/5] Building HST v7.1.2 Agile model (T4 optimized)...")
model = HSTv7Agile(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    horizon=HORIZON,
    mode=MODE,
    chunk_size=CHUNK_SIZE
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
                print(f"  Checkpoint saved at step {step}")

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            step += 1

except KeyboardInterrupt:
    print("\n\n!! Training INTERRUPTED !!")
    torch.save(model.state_dict(), f'{save_dir}/hst_v7_1_2_interrupt_step_{step}.pt')
    print(f"Model saved at step {step}")

torch.save(model.state_dict(), f'{save_dir}/hst_v7_1_2_final.pt')
print(f"\nTRAINING COMPLETE - Model saved!")
print_memory()
