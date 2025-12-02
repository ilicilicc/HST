#!/usr/bin/env python3
"""
HST v3 Ultra - FULL ARCHITECTURE Training Script
Uses complete CompleteLatticeCore with meta-fusion
Includes: MultiLevelLatticeProcessor + PathWeightedLatticeCore + Meta-Fusion
Memory optimized for 16GB GPU (Colab/Kaggle)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from datasets import load_dataset
import gc
import psutil

try:
    from google.colab import drive
    drive.mount('/content/drive')
    save_dir = '/content/drive/MyDrive/HST_Training/v3_ultra_full'
except:
    save_dir = './HST_Training/v3_ultra_full'

os.makedirs(save_dir, exist_ok=True)
device = torch.device('cuda')
torch.cuda.empty_cache()

print("=" * 80)
print("HST v3 Ultra - FULL ARCHITECTURE (Complete Lattice Core + Meta-Fusion)")
print("=" * 80)

def print_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    gpu_max = torch.cuda.get_device_properties(device).total_memory / 1e9
    gpu_pct = (gpu_mem / gpu_max) * 100
    ram = psutil.virtual_memory()
    ram_pct = ram.percent
    print(f"GPU: {gpu_mem:.2f}GB / {gpu_max:.2f}GB ({gpu_pct:.1f}%) | RAM: {ram.used/1e9:.1f}GB / {ram.total/1e9:.1f}GB ({ram_pct:.1f}%)")

# ==================== FULL LATTICE CORE IMPLEMENTATION ====================
class FullLatticeFieldAnalyzer(nn.Module):
    """Complete lattice structure analyzer with all levels"""
    def __init__(self, max_seq_len=512):
        super().__init__()
        spine = [0, 2, 4]
        while True:
            next_val = 2*spine[-1] + 2*spine[-2] + 2*spine[-3]
            if next_val >= max_seq_len:
                break
            spine.append(next_val)
        
        self.register_buffer('spine', torch.tensor(spine, dtype=torch.long))
        self.max_depth = len(spine)
        
        # Precompute for spine positions
        self.lattice_structure = {}
        for pos in spine:
            if pos < max_seq_len:
                self.lattice_structure[pos] = self._analyze_position(pos)
        
        self._non_spine_cache = {}
    
    def get_structure(self, pos: int):
        if pos in self.lattice_structure:
            return self.lattice_structure[pos]
        if pos in self._non_spine_cache:
            return self._non_spine_cache[pos]
        structure = self._analyze_non_spine(pos)
        self._non_spine_cache[pos] = structure
        return structure
    
    def _analyze_position(self, pos):
        levels = {0: [pos]}
        visited = {pos}
        current_level = [pos]
        level = 0
        
        while current_level and level < 10:
            next_level = set()
            for node in current_level:
                ancestors = self._get_immediate_ancestors(node)
                for anc in ancestors:
                    if anc not in visited and anc >= 0:
                        visited.add(anc)
                        next_level.add(anc)
            current_level = list(next_level)
            level += 1
            if current_level:
                levels[level] = current_level.copy()
        
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
                if node == pos: continue
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

class MultiLevelLatticeProcessor(nn.Module):
    """Process each lattice level separately with attention fusion"""
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
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        spine = self.analyzer.spine
        relevant_spine = spine[spine < S]
        
        h_out = x.clone()
        
        for spine_pos in relevant_spine:
            if spine_pos.item() < 3: continue
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            
            if structure is None: continue
            
            level_features = []
            for level in range(structure['max_depth'] + 1):
                if level == 0: continue
                if level not in structure['levels']: continue
                
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
                    level_feat = self.level_transforms[level](level_feat)
                    level_features.append(level_feat)
            
            if not level_features: continue
            
            level_stack = torch.stack(level_features, dim=1)
            query = h_out[:, pos:pos+1, :]
            attended, _ = self.level_attention(query, level_stack, level_stack)
            
            combined = torch.cat([attended.squeeze(1), x[:, pos, :]], dim=-1)
            h_out[:, pos, :] = self.fusion(combined)
        
        return h_out

class PathWeightedLatticeCore(nn.Module):
    """GRU-based path-weighted aggregation"""
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
        spine = self.analyzer.spine
        relevant_spine = spine[spine < S]
        
        h_out = x.clone()
        
        for spine_pos in relevant_spine:
            if spine_pos.item() < 3: continue
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            
            if structure is None or structure['total_ancestors'] == 0: continue
            
            all_ancestors = []
            path_counts = []
            
            for level in structure['levels']:
                if level > 0:
                    for anc in structure['levels'][level]:
                        if anc < S: 
                            all_ancestors.append(anc)
                            path_counts.append(structure['path_counts'].get(anc, 1))
            
            if not all_ancestors: continue
            
            # Batch process path weights
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
            h_out[:, pos, :] = gate * aggregated + (1 - gate) * h_out[:, pos, :]
        
        return h_out

class CompleteLatticeCore(nn.Module):
    """FULL IMPLEMENTATION: Meta-fusion of Multi-Level + Path-Weighted"""
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
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
        scores.masked_fill_(mask[None, None, :, :], torch.finfo(scores.dtype).min)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LightweightAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        # Advanced confidence predictor
        self.conf_pred = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # Compute confidence from pooled features
        conf = self.conf_pred(x.transpose(1, 2))
        conf = conf.mean(dim=0)
        return x, conf

class HarmonicHorizonPredictor(nn.Module):
    """Full horizon predictor with projection and confidence"""
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
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_last = x[:, -1, :]
        
        projected = self.horizon_projection(x_last).view(-1, self.horizon, self.d_model)
        logits_list = self.prediction_head(projected)
        confidence = self.confidence_head(x_last)
        
        # Return in format compatible with training
        return logits_list.transpose(1, 2), confidence  # (B, horizon, V) -> (B, V, horizon)

# ==================== FULL MODEL ====================
class HSTv3Ultra(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=8, n_layers=16, max_seq_len=512, horizon=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # FULL LATTICE CORE with meta-fusion
        self.lattice = CompleteLatticeCore(d_model, max_seq_len)
        
        self.n_bottom = n_layers // 2
        self.bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(self.n_bottom)])
        self.top = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(n_layers - self.n_bottom)])
        
        # FULL HORIZON PREDICTOR
        self.horizon = HarmonicHorizonPredictor(d_model, vocab_size, horizon)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight
    
    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed(torch.arange(input_ids.size(1), device=device))
        
        # Apply FULL lattice core
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
            'bottom_depth': torch.tensor(depth, device=device),
            'confidence': torch.stack(confs).mean() if confs else torch.tensor(0.5, device=device)
        }

# ==================== LOSS FUNCTION ====================
def compute_loss(output, targets, horizon=16, gamma=0.95, pad_id=50256):
    logits = output['logits']
    horizon_logits = output['horizon_logits']
    
    B, S = targets.shape
    V = logits.size(-1)
    
    logits = logits[:, :S]
    horizon_logits = horizon_logits[:, :S]
    
    # Main loss
    pred_logits = logits[:, :-1].reshape(-1, V)
    pred_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(pred_logits, pred_targets, ignore_index=pad_id)
    
    # Horizon losses with proper indexing
    H = min(horizon, S - 1)
    for k in range(1, H + 1):
        if k >= horizon_logits.size(2):
            break
        h_logits_k = horizon_logits[:, :S-k, k]
        h_targets_k = targets[:, k:]
        
        if h_logits_k.shape[1] == 0 or h_targets_k.shape[1] == 0:
            continue
        
        min_len = min(h_logits_k.shape[1], h_targets_k.shape[1])
        h_logits_k = h_logits_k[:, :min_len].reshape(-1, V)
        h_targets_k = h_targets_k[:, :min_len].reshape(-1)
        
        loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
    
    # Early-exit consistency
    depth = output['bottom_depth'].float()
    conf = output['confidence']
    loss += 0.05 * F.mse_loss(conf, depth / (len(output) if isinstance(output, dict) else 4.0))
    
    return loss

# ==================== INITIALIZE ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("[2/5] Building FULL ARCHITECTURE model (768-dim, 16-layer, 8-head)...")
model = HSTv3Ultra(
    d_model=768,       # Reduced from 1024 to fit full architecture
    n_heads=8,
    n_layers=16,       # Reduced from 24 to fit full architecture
    max_seq_len=512,
    horizon=16
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {total_params/1e9:.2f}B params ({total_params/1e6:.0f}M)")
print(f"Trainable: {trainable_params/1e9:.2f}B params")
print(f"Final size (FP32): {(total_params * 4) / 1e9:.1f}GB")
print(f"Training uses FP16 autocast for memory efficiency")
print_memory()

print("\n[3/5] Setting up 8-bit optimizer...")
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
    print("✓ Using 8-bit AdamW")
except ImportError:
    print("⚠ bitsandbytes not available, using standard AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scaler = GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, 3000, 100000)

print("[4/5] Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def tokenize(ex):
    t = tokenizer(ex["text"], truncation=True, max_length=512)["input_ids"]
    return {"input_ids": t}

stream = dataset.map(tokenize, batched=True, batch_size=500, remove_columns=dataset.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
loader = DataLoader(stream, batch_size=1, collate_fn=collator)

print("[5/5] Starting training loop with FULL ARCHITECTURE...\n")

# ==================== TRAINING ====================
model.train()
step = 0

for batch in loader:
    if step >= 100000: break
    
    ids = batch["input_ids"].to(device)
    
    optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        out = model(ids)
        loss = compute_loss(out, ids, horizon=16)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    if step % 100 == 0:
        print(f"Step {step:6d} | Loss {loss.item():.4f} | Depth {out['bottom_depth'].item():.0f} | Conf {out['confidence'].item():.3f} | ", end="")
        print_memory()
    
    if step % 5000 == 0 and step > 0:
        torch.save(model.state_dict(), f"{save_dir}/ckpt_step_{step}.pt")
        print(f"  ✓ Checkpoint saved at step {step}")
    
    if step % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    step += 1

torch.save(model.state_dict(), f'{save_dir}/hst_v3_ultra_final.pt')
print("\n✅ TRAINING COMPLETE - Full Architecture Model Saved!")
print_memory()
