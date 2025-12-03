"""
HST v4 Unified - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- Complete Lattice Core with Meta-Fusion
- Token Mode and Chunk Mode support
- KV Cache for efficient generation
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
MAX_SEQ_LEN = 512
HORIZON = 8
VOCAB_SIZE = 32000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MAX_TRAINING_STEPS = 100000
INITIAL_LR = 2e-4
WARMUP_STEPS = 2000
CHUNK_SIZE = 128
MODE = 'token'

save_dir = '/content/drive/MyDrive/hst_v4_checkpoints'
os.makedirs(save_dir, exist_ok=True)

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def prune_cache(cache, max_size=2048):
    if cache is None:
        return None
    new_cache = []
    for k, v in cache:
        if k.size(2) > max_size:
            k = k[:, :, -max_size:, :]
            v = v[:, :, -max_size:, :]
        new_cache.append((k, v))
    return new_cache

# ==================== LATTICE FIELD ANALYZER ====================
class FullLatticeFieldAnalyzer:
    def __init__(self, max_seq_len=8192):
        self.max_seq_len = max_seq_len
        spine_list = self._generate_spine_list(max_seq_len)
        self.spine = torch.tensor(spine_list, dtype=torch.long)
        self._structure_cache = {}
    
    def _generate_spine_list(self, max_len):
        spine = [0, 2, 4]
        while True:
            next_val = 2 * spine[-1] + 2 * spine[-2] + 2 * spine[-3]
            if next_val >= max_len:
                break
            spine.append(next_val)
        return spine
    
    def get_structure(self, pos):
        if pos in self._structure_cache:
            return self._structure_cache[pos]
        
        if pos in self.spine:
            structure = self._analyze_spine_position(pos)
        else:
            structure = self._analyze_non_spine(pos)
        
        self._structure_cache[pos] = structure
        return structure
    
    def _analyze_spine_position(self, pos):
        visited = {pos}
        levels = {0: [pos]}
        queue = [(pos, 0)]
        
        while queue:
            current, level = queue.pop(0)
            if level >= 9:
                break
            
            ancestors = self._get_immediate_ancestors(current)
            current_level = []
            
            for anc in ancestors:
                if anc not in visited and anc >= 0:
                    visited.add(anc)
                    current_level.append(anc)
                    queue.append((anc, level + 1))
            
            if current_level:
                levels[level + 1] = current_level.copy()

        max_depth = max(levels.keys()) if levels else 0
        path_counts = self._compute_path_counts(pos, levels, max_depth)
        
        return {
            'levels': levels,
            'path_counts': path_counts,
            'total_ancestors': len(visited) - 1,
            'max_depth': max_depth
        }
    
    def _get_immediate_ancestors(self, pos):
        try:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx >= 3:
                return [self.spine[idx-1].item(), self.spine[idx-2].item(), self.spine[idx-3].item()]
        except:
            pass
        return []
    
    def _analyze_non_spine(self, pos):
        left_spine = self.spine[self.spine < pos]
        ancestors = []
        if len(left_spine) > 0:
            ancestors.append(left_spine[-1].item())
        return {
            'levels': {0: [pos], 1: ancestors},
            'path_counts': {anc: 1 for anc in ancestors},
            'total_ancestors': len(ancestors),
            'max_depth': 1
        }
    
    def _compute_path_counts(self, pos, levels, max_depth):
        path_counts = {pos: 1}
        for level in sorted(levels.keys(), reverse=True):
            for node in levels[level]:
                if node == pos:
                    continue
                if level == max_depth:
                    path_counts[node] = 1
                    continue
                count = 0
                for child in levels.get(level + 1, []):
                    if node in self._get_immediate_ancestors(child):
                        count += path_counts.get(child, 0)
                if level != 0:
                    path_counts[node] = count
        path_counts.pop(pos, None)
        return path_counts

# ==================== CHUNK ENCODER/DECODER ====================
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
        chunk_embeddings = self.pooler(reshaped)
        return chunk_embeddings

class ChunkDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.expander = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * chunk_size)
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, chunk_idx=None):
        B, num_chunks, D = chunk_embeddings.shape
        expanded = self.expander(chunk_embeddings)
        token_hidden = expanded.view(B, num_chunks * self.chunk_size, D)
        logits = self.lm_head(token_hidden)
        return logits

# ==================== TRANSFORMER BLOCKS ====================
class TransformerEncoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, layer_past=None):
        B, S, D = x.shape
        
        if layer_past is not None:
            past_k, past_v = layer_past
            past_len = past_k.size(2)
        else:
            past_len = 0
        
        x_norm = self.norm1(x)
        
        if layer_past is not None:
            k = torch.cat([past_k.transpose(1, 2).reshape(B, -1, D), x_norm], dim=1)
            v = torch.cat([past_v.transpose(1, 2).reshape(B, -1, D), x_norm], dim=1)
        else:
            k = v = x_norm
        
        attn_output, _ = self.self_attn(x_norm, k, v, need_weights=False, is_causal=(layer_past is None))
        x = x + self.dropout1(attn_output)
        
        x_norm2 = self.norm2(x)
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(x_norm2))))
        x = x + self.dropout2(ff_output)
        
        n_heads = self.self_attn.num_heads
        head_dim = D // n_heads
        new_k = x_norm.view(B, S, n_heads, head_dim).transpose(1, 2)
        new_v = x_norm.view(B, S, n_heads, head_dim).transpose(1, 2)
        
        if layer_past is not None:
            new_k = torch.cat([past_k, new_k], dim=2)
            new_v = torch.cat([past_v, new_v], dim=2)
        
        return x, (new_k, new_v)

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.block = TransformerEncoderLayerWithCache(d_model, n_heads)
        self.confidence_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, layer_past=None):
        x_out, present = self.block(x, layer_past)
        if x_out.size(1) > 1:
            conf = self.confidence_predictor(x_out.transpose(1, 2))
            conf = conf.mean(dim=0)
        else:
            conf = x_out.new_tensor([0.0])
        return x_out, conf, present

# ==================== LATTICE PROCESSORS ====================
class MultiLevelLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        self.level_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for _ in range(10)
        ])
        self.level_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
    
    def forward(self, x):
        B, S, D = x.shape
        spine = self.analyzer.spine.to(x.device)
        relevant_spine = spine[spine < S]
        
        updates = {}
        for spine_pos in relevant_spine:
            if spine_pos.item() < 3:
                continue
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            if structure is None:
                continue
            
            level_features = []
            for level in range(1, structure['max_depth'] + 1):
                if level not in structure['levels']:
                    continue
                level_nodes = structure['levels'][level]
                level_h = []
                total_weight = 0.0
                for node in level_nodes:
                    if node < S:
                        weight = structure['path_counts'].get(node, 1)
                        level_h.append(x[:, node, :] * weight)
                        total_weight += weight
                if level_h and total_weight > 0:
                    level_feat = torch.stack(level_h, dim=1).sum(dim=1) / total_weight
                    level_feat = self.level_transforms[min(level, 9)](level_feat)
                    level_features.append(level_feat)

            if not level_features:
                continue

            level_stack = torch.stack(level_features, dim=1)
            query = x[:, pos:pos+1, :]
            attended, _ = self.level_attention(query, level_stack, level_stack)
            combined = torch.cat([attended.squeeze(1), x[:, pos, :]], dim=-1)
            updates[pos] = self.fusion(combined)

        if not updates:
            return x

        sorted_positions = sorted(updates.keys())
        output_slices = []
        last_pos = 0
        for pos in sorted_positions:
            if pos > last_pos:
                output_slices.append(x[:, last_pos:pos, :])
            output_slices.append(updates[pos].unsqueeze(1))
            last_pos = pos + 1
        if last_pos < S:
            output_slices.append(x[:, last_pos:S, :])
        return torch.cat(output_slices, dim=1)

class PathWeightedLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        self.path_weight_net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        self.message_fn = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.aggregate_fn = nn.GRU(d_model, d_model, batch_first=True)
        self.update_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
    
    def forward(self, x):
        B, S, D = x.shape
        spine = self.analyzer.spine.to(x.device)
        relevant_spine = spine[spine < S]
        
        updates = {}
        for spine_pos in relevant_spine:
            if spine_pos.item() < 3:
                continue
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            if structure is None or structure['total_ancestors'] == 0:
                continue
            
            all_ancestors = []
            path_counts = []
            for level in structure['levels']:
                if level > 0:
                    for anc in structure['levels'][level]:
                        if anc < S:
                            all_ancestors.append(anc)
                            path_counts.append(structure['path_counts'].get(anc, 1))

            if not all_ancestors:
                continue

            path_count_tensor = torch.tensor(path_counts, device=x.device).view(-1, 1).float()
            path_weights_tensor = self.path_weight_net(path_count_tensor).squeeze()

            messages = []
            for ancestor_pos in all_ancestors:
                h_anc = x[:, ancestor_pos, :]
                h_curr = x[:, pos, :]
                msg = self.message_fn(torch.cat([h_anc, h_curr], dim=-1))
                messages.append(msg)
            
            msg_stack = torch.stack(messages, dim=1)
            if path_weights_tensor.dim() == 0:
                weights_tensor = path_weights_tensor.view(1, 1, 1).expand(B, -1, D)
            else:
                weights_tensor = path_weights_tensor.view(1, -1, 1).expand(B, -1, D)
            weighted_msgs = msg_stack * weights_tensor
            aggregated, _ = self.aggregate_fn(weighted_msgs)
            aggregated = aggregated[:, -1, :]
            gate = self.update_gate(torch.cat([aggregated, x[:, pos, :]], dim=-1))
            updates[pos] = gate * aggregated + (1 - gate) * x[:, pos, :]

        if not updates:
            return x

        sorted_positions = sorted(updates.keys())
        output_slices = []
        last_pos = 0
        for pos in sorted_positions:
            if pos > last_pos:
                output_slices.append(x[:, last_pos:pos, :])
            output_slices.append(updates[pos].unsqueeze(1))
            last_pos = pos + 1
        if last_pos < S:
            output_slices.append(x[:, last_pos:S, :])
        return torch.cat(output_slices, dim=1)

class CompleteLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.multi_level = MultiLevelLatticeProcessor(d_model, max_seq_len)
        self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        self.meta_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.LayerNorm(d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x):
        h_multi = self.multi_level(x)
        h_path = self.path_weighted(x)
        h_combined = torch.cat([x, h_multi, h_path], dim=-1)
        return self.meta_fusion(h_combined)

# ==================== HARMONIC HORIZON PREDICTOR ====================
class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.horizon_projection = nn.Linear(d_model, d_model * horizon)
        self.prediction_head = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, horizon))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_last = x[:, -1, :]
        projected = self.horizon_projection(x_last).view(-1, self.horizon, self.d_model)
        logits_list = self.prediction_head(projected)
        confidence = torch.sigmoid(self.confidence_head(x_last))
        return logits_list, confidence

# ==================== HST v4 UNIFIED MODEL ====================
class HSTv4Unified(nn.Module):
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
            self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size)
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(self.n_bottom_layers)])
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len)
            self.top_stack = nn.ModuleList([TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(self.n_top_layers)])

        self.harmonic_horizon_predictor = HarmonicHorizonPredictor(d_model, vocab_size, horizon=horizon)
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
        
        past_len = cache[0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        new_cache = []
        cache_idx = 0
        predicted_depth = self.n_bottom_layers

        for i, block in enumerate(self.adaptive_bottom):
            layer_past = cache[cache_idx] if cache else None
            x, conf, present = block(x, layer_past)
            new_cache.append(present)
            cache_idx += 1
            
            if past_len == 0 and i >= 1 and conf.item() > self.early_exit_confidence_threshold:
                predicted_depth = i + 1
                break
        
        h_bottom = x
        h_lattice_out = self.lattice_core(h_bottom)
        
        h_top_in = h_lattice_out
        for i, block in enumerate(self.top_stack):
            layer_past = cache[cache_idx] if cache else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
            cache_idx += 1
            
        h_final = h_top_in
        logits_t1 = self.lm_head(self.ln_f(h_final))
        logits_horizon, confidence = self.harmonic_horizon_predictor(h_final)
        
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
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        chunk_emb = self.chunk_encoder(x)
        h_lattice_out = self.lattice_core(chunk_emb)
        logits = self.chunk_decoder(h_lattice_out, chunk_idx=None)

        last_chunk_rep = h_lattice_out[:, -1:, :]
        logits_horizon, confidence = self.harmonic_horizon_predictor(last_chunk_rep)
        
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
    
    if 'horizon_logits' in output and output['horizon_logits'] is not None:
        horizon_logits = output['horizon_logits']
        H = horizon_logits.size(1)
        for k in range(1, min(H + 1, 5)):
            if k < S:
                h_logits_k = horizon_logits[:, k-1, :]
                h_targets_k = targets[:, min(k, S-1)]
                loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
    
    if 'confidence' in output and output.get('bottom_depth', 0) > 0:
        depth = float(output['bottom_depth'])
        conf = output['confidence'].mean()
        loss += 0.03 * F.mse_loss(conf, torch.tensor(depth / n_layers, device=conf.device))
    
    return loss

# ==================== INITIALIZE ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded: {len(tokenizer)} tokens")

print("[2/5] Building HST v4 Unified model (T4 optimized)...")
model = HSTv4Unified(
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
    torch.save(model.state_dict(), f'{save_dir}/hst_v4_interrupt_step_{step}.pt')
    print(f"Model saved at step {step}")

torch.save(model.state_dict(), f'{save_dir}/hst_v4_final.pt')
print(f"\nTRAINING COMPLETE - Model saved!")
print_memory()
