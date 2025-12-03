"""
HST v3 Ultra - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- Complete Lattice Core (Multi-Level + Path-Weighted Fusion)
- Adaptive Block with Early Exit
- Harmonic Horizon Predictor
- Graceful interrupt handling with checkpoint saving
"""

# ==================== SETUP & INSTALLATION ====================
# !pip install torch transformers datasets bitsandbytes accelerate -q

import os
import gc
import signal
import numpy as np
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

save_dir = '/content/drive/MyDrive/hst_v3_checkpoints'
os.makedirs(save_dir, exist_ok=True)

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

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
                return [
                    self.spine[idx-1].item(),
                    self.spine[idx-2].item(),
                    self.spine[idx-3].item()
                ]
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

# ==================== MULTI-LEVEL LATTICE PROCESSOR ====================
class MultiLevelLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(10)
        ])
        
        self.level_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        spine = self.analyzer.spine.to(x.device)
        relevant_spine = spine[spine < S]
        
        h_out = x.clone()
        
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
            
            index_tensor = torch.tensor([pos], device=x.device, dtype=torch.long)
            new_value = self.fusion(combined)
            h_out = h_out.scatter(1, index_tensor.view(1, 1, 1).expand(B, 1, D), new_value.unsqueeze(1))

        return h_out

# ==================== PATH-WEIGHTED LATTICE CORE ====================
class PathWeightedLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        
        self.path_weight_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.aggregate_fn = nn.GRU(d_model, d_model, batch_first=True)
        
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        spine = self.analyzer.spine.to(x.device)
        relevant_spine = spine[spine < S]
        
        h_out = x.clone()
        
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
                h_curr = h_out[:, pos, :]
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
            
            gate = self.update_gate(torch.cat([aggregated, h_out[:, pos, :]], dim=-1))
            new_value = gate * aggregated + (1 - gate) * h_out[:, pos, :]
            
            index_tensor = torch.tensor([pos], device=x.device, dtype=torch.long)
            h_out = h_out.scatter(1, index_tensor.view(1, 1, 1).expand(B, 1, D), new_value.unsqueeze(1))
        
        return h_out

# ==================== COMPLETE LATTICE CORE ====================
class CompleteLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.multi_level = MultiLevelLatticeProcessor(d_model, max_seq_len)
        self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        
        self.meta_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_multi = self.multi_level(x)
        h_path = self.path_weighted(x)
        
        h_combined = torch.cat([x, h_multi, h_path], dim=-1)
        h_out = self.meta_fusion(h_combined)
        
        return h_out

# ==================== TRANSFORMER BLOCKS ====================
class LightweightAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, attn_dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        nn.init.xavier_uniform_(self.q.weight, gain=0.01)
        nn.init.xavier_uniform_(self.k.weight, gain=0.01)
        nn.init.xavier_uniform_(self.v.weight, gain=0.01)
    
    def forward(self, x):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask[None, None, :, :], neg_inf)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = LightweightAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.conf_pred = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        conf = self.conf_pred(x.transpose(1, 2))
        conf = conf.mean(dim=0)
        return x, conf

# ==================== HARMONIC HORIZON PREDICTOR ====================
class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.horizon_projection = nn.Linear(d_model, d_model * horizon)
        self.prediction_head = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, horizon),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, S, D = x.shape
        
        projected = self.horizon_projection(x).view(B, S, self.horizon, D)
        logits_list = self.prediction_head(projected.reshape(-1, D)).view(B, S, self.horizon, -1)
        
        x_last = x[:, -1, :]
        confidence = self.confidence_head(x_last)
        
        return logits_list.transpose(-1, -2), confidence

# ==================== FULL MODEL ====================
class HSTv3Ultra(nn.Module):
    def __init__(self, vocab_size=32000, d_model=768, n_heads=12, n_layers=16, max_seq_len=512, horizon=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.lattice = CompleteLatticeCore(d_model, max_seq_len)
        
        self.n_layers = n_layers
        self.n_bottom = n_layers // 2
        self.bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(self.n_bottom)])
        self.top = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(n_layers - self.n_bottom)])
        
        self.horizon = HarmonicHorizonPredictor(d_model, vocab_size, horizon)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids):
        B, S = input_ids.shape
        embed = self.embed(input_ids)
        pos_embed = self.pos_embed(torch.arange(S, device=input_ids.device))
        x = embed + pos_embed
        
        x = self.lattice(x)
        
        confs = []
        depth = self.n_bottom
        for i, block in enumerate(self.bottom):
            x, conf = block(x)
            confs.append(conf)
            if i >= 1 and conf.item() > 0.93:
                depth = i + 1
                break
        
        for block in self.top:
            x, _ = block(x)
        
        x = self.norm(x)
        horizon_logits, horizon_conf = self.horizon(x)
        
        return {
            'logits': self.out_proj(x),
            'horizon_logits': horizon_logits,
            'horizon_confidence': horizon_conf,
            'bottom_depth': torch.tensor(depth, device=input_ids.device),
            'confidence': torch.stack(confs).mean() if confs else torch.tensor(0.5, device=input_ids.device)
        }

# ==================== LOSS FUNCTION ====================
def compute_loss(output, targets, horizon=8, gamma=0.95, pad_id=50256, n_layers=16):
    logits = output['logits']
    horizon_logits = output['horizon_logits']
    
    B, S = targets.shape
    V = logits.size(-1)
    
    logits = logits[:, :S]
    
    pred_logits = logits[:, :-1].reshape(-1, V)
    pred_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(pred_logits, pred_targets, ignore_index=pad_id)
    
    H = horizon_logits.size(-1)
    for k in range(1, min(H + 1, S)):
        h_logits_k = horizon_logits[:, :-k, :, k-1]
        h_targets_k = targets[:, k:]
        
        h_logits_k = h_logits_k.reshape(-1, V)
        h_targets_k = h_targets_k.reshape(-1)
        
        loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
    
    depth = output['bottom_depth'].float()
    conf = output['confidence']
    loss += 0.03 * F.mse_loss(conf, depth / n_layers)
    
    return loss

# ==================== INITIALIZE ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded: {len(tokenizer)} tokens")

print("[2/5] Building HST v3 Ultra model (T4 optimized)...")
model = HSTv3Ultra(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    horizon=HORIZON
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {total_params/1e6:.1f}M params")
print(f"Trainable: {trainable_params/1e6:.1f}M params")
print_memory()

print("\n[3/5] Setting up optimizer...")
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), weight_decay=0.01)
    print("Using 8-bit AdamW")
except ImportError:
    print("bitsandbytes not available, using standard AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)

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

print(f"[5/5] Starting training (Max Steps: {MAX_TRAINING_STEPS}, Accumulation: {GRADIENT_ACCUMULATION_STEPS})...\n")

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
                print(f"Step {step:6d} | Loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | Depth {out['bottom_depth'].item():.0f} | Conf {out['confidence'].item():.3f} | ", end="")
                print_memory()
            
            if step % 5000 == 0 and step > 0:
                torch.save(model.state_dict(), f"{save_dir}/ckpt_step_{step}.pt")
                print(f"  Checkpoint saved at step {step}")

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            step += 1

except KeyboardInterrupt:
    print("\n\n!! Training INTERRUPTED by User !!")
    interrupt_path = f'{save_dir}/hst_v3_ultra_interrupt_step_{step}.pt'
    torch.save(model.state_dict(), interrupt_path)
    print(f"Model saved at interruption: {interrupt_path}")
    print_memory()

final_path = f'{save_dir}/hst_v3_ultra_final.pt'
torch.save(model.state_dict(), final_path)
print(f"\nTRAINING COMPLETE - Final Model Saved to: {final_path}")
print_memory()
