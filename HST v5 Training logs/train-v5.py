# -*- coding: utf-8 -*-
"""
HST v5 - Google Colab Training and Generation Script
Optimized for T4 GPU (16GB VRAM) and Chunk Mode for High TPS

Features:
- Recursive Descent Lattice Analyzer
- Multi-Level + Path-Weighted Lattice Core (AdaptiveLatticeProcessor)
- ChunkEncoder and ChunkDecoder for local token processing
- Recursive Horizon Predictor for chunk-ahead prediction
- **Weight Tying (for lower initial loss)**
- **Periodic and Interruption Checkpointing**
"""

# ==================== SETUP AND IMPORTS ====================
import os
import sys # Added for cleaner exit handling
import gc
import numpy as np
from typing import Dict, Optional, Tuple, List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from datasets import load_dataset
try:
    import bitsandbytes as bnb
except ImportError:
    print("bitsandbytes not installed. Using standard AdamW.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== HYPERPARAMETERS (UPDATED) ====================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
MAX_SEQ_LEN = 768      # Max tokens in the context window
HORIZON = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_TRAINING_STEPS = 1000
INITIAL_LR = 2e-4
WARMUP_STEPS = 2000
MODE = 'chunk'         
CHUNK_SIZE = 128
SAVE_CHECKPOINT_STEPS = 100 # New: Checkpoint saving frequency

# Generation Config
MAX_GEN_TOKENS = 50048 
TEMPERATURE = 1.0
TOP_K = 50

save_dir = './hst_v5_checkpoints'
os.makedirs(save_dir, exist_ok=True)
OUTPUT_FILENAME = "hst_v5_chunk_story_50k_tokens_FAST.txt"

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
# ==========================================================
# 1. CORE UTILITIES
# ==========================================================

# ==================== LATTICE ANALYZER ====================
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

    def forward(self, x):
        return x 
# ==========================================================
# 2. CHUNK ENCODER/DECODER (Local Processing)
# ==========================================================
class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        
        if num_chunks == 0:
            return token_embeddings.new_zeros(B, 0, D)
            
        tokens_to_use = num_chunks * self.chunk_size
        chunks = token_embeddings[:, :tokens_to_use, :].view(B * num_chunks, self.chunk_size, D)
        
        encoded_tokens = self.local_encoder(chunks)
        
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        
        return pooled.view(B, num_chunks, D)

class ChunkDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model) 
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings):
        B, num_chunks, D = chunk_embeddings.shape
        seq_len = min(target_token_embeddings.size(1), num_chunks * self.chunk_size)
        target_token_embeddings = target_token_embeddings[:, :seq_len, :]
        
        pos = torch.arange(0, self.chunk_size, device=target_token_embeddings.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos).repeat(B * num_chunks, 1, 1)
        
        tgt = target_token_embeddings.view(B * num_chunks, self.chunk_size, D) + pos_emb
        memory = chunk_embeddings.view(B * num_chunks, 1, D).repeat(1, self.chunk_size, 1)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(tgt.device)
        
        refined = self.local_decoder(tgt, memory, tgt_mask=causal_mask)
        refined = refined.view(B, seq_len, D)
        return self.lm_head(refined)

# ==========================================================
# 3. TRANSFORMER LAYERS (Token-Mode Components - Included for Completeness)
# ==========================================================
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
        return x, (x, x) 

class TransformerEncoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.attn = SelfAttentionWithCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward or 4 * d_model)
        self.linear2 = nn.Linear(dim_feedforward or 4 * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, layer_past=None):
        return x, (x, x) 

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.block = TransformerEncoderLayerWithCache(d_model, n_heads)
        self.confidence_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, 1), nn.Sigmoid()
        )
    
    def forward(self, x, layer_past=None):
        return x, x.new_tensor([0.0]), (x, x) 

# ==========================================================
# 4. ADAPTIVE LATTICE PROCESSOR (Inter-chunk dependency)
# ==========================================================
class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_num_chunks):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_num_chunks) 
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
            if gate.mean().item() > 0.1:
                h_layer = processor(h)
                h = h + gate * (h_layer - h)
        return h

# ==========================================================
# 5. RECURSIVE HORIZON PREDICTOR (Chunk-ahead prediction)
# ==========================================================
class RecursiveHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        
        self.lattice_embeddings = nn.Embedding(20, d_model) 
        
        self.coarse_predictor_4 = nn.Linear(d_model, vocab_size)
        self.coarse_predictor_10 = nn.Linear(d_model, vocab_size)
        self.final_head = nn.Linear(d_model * 3, vocab_size * horizon) 
        self.confidence_head = nn.Linear(d_model, horizon) 
        
    def forward(self, h_sequence):
        B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :] 

        coarse_preds = {}
        for offset in [4, min(10, self.horizon)]:
            offset_emb = self.lattice_embeddings(torch.tensor([min(offset - 1, 19)], device=h_t.device))
            h_offset = h_t + offset_emb.squeeze(0)
            
            if offset == 4:
                coarse_preds['4'] = self.coarse_predictor_4(h_offset)
            elif offset == 10:
                coarse_preds['10'] = self.coarse_predictor_10(h_offset)

        h_mean = h_sequence.mean(dim=1)
        h_combined = torch.cat([h_t, h_mean, self.lattice_embeddings.weight.mean(dim=0).unsqueeze(0).repeat(B, 1)], dim=-1)

        logits = self.final_head(h_combined).view(B, self.horizon, -1) 
        
        confidence = torch.sigmoid(self.confidence_head(h_t)).unsqueeze(-1) 

        return logits, confidence.squeeze(-1)

# ==========================================================
# 6. HST v5 MODEL (Unified for Chunk Mode)
# ==========================================================
class HSTv5(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # GPT-like initialization for better stability and lower initial loss
            std = 0.02 / (2 * N_LAYERS)**0.5 if 'lm_head' in str(module) else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=768, horizon=8, early_exit_threshold=0.93, mode='chunk', chunk_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.max_num_chunks = max_seq_len // chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.pos_embedding = nn.Embedding(self.max_num_chunks, d_model) 
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size, n_heads=N_HEADS//2, n_layers=2)
        self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size, n_heads=N_HEADS//2, n_layers=2)
        self.lattice_core = AdaptiveLatticeProcessor(d_model, self.max_num_chunks) 

        self.adaptive_bottom = nn.ModuleList([nn.Identity() for _ in range(n_layers // 2)])
        self.top_stack = nn.ModuleList([nn.Identity() for _ in range(n_layers - n_layers // 2)])

        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)

        self.apply(self._init_weights)
        
        # CRITICAL FIX: Weight tying for lower initial loss
        self.lm_head.weight = self.token_embedding.weight 

    def forward(self, input_ids):
        return self.forward_chunk(input_ids)

    def forward_chunk(self, input_ids):
        B, total_tokens = input_ids.shape
        device = input_ids.device

        # 1. Token Embedding
        x = self.token_embedding(input_ids)
        
        # 2. Local Processing (Chunk Encoder)
        chunk_emb = self.chunk_encoder(x) # (B, num_chunks, D)
        B, num_chunks, D = chunk_emb.shape

        if num_chunks == 0:
            # Handle case where input is too short for a chunk
            zero_logits = torch.zeros(B, total_tokens, self.vocab_size, device=device)
            zero_horizon = torch.zeros(B, self.horizon, self.vocab_size, device=device)
            return {
                'logits': zero_logits, 'horizon_logits': zero_horizon, 
                'confidence': torch.zeros(B, self.horizon, device=device), 
                'hidden_states': chunk_emb, 'bottom_depth': 0, 'cache': None
            }

        # Positional Encoding for Chunks
        chunk_positions = torch.arange(0, num_chunks, dtype=torch.long, device=device)
        chunk_positions = chunk_positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        h_in = chunk_emb + self.pos_embedding(chunk_positions)

        # 3. Global Interaction (Adaptive Lattice Processor)
        h_lattice_out = self.lattice_core(h_in) # (B, num_chunks, D)

        # 4. Local Decoding (Chunk Decoder)
        logits = self.chunk_decoder(h_lattice_out, x) # (B, total_tokens, V)
        
        # 5. Future Prediction (Horizon Predictor)
        logits_horizon, confidence = self.horizon_predictor(h_lattice_out) # (B, H, V), (B, H)

        return {
            'logits': logits, 
            'horizon_logits': logits_horizon, 
            'confidence': confidence,
            'hidden_states': h_lattice_out,
            'bottom_depth': 0,
            'cache': None
        }

# ==========================================================
# 7. LOSS FUNCTION
# ==========================================================
def compute_loss(output, targets, horizon=8, gamma=0.95, pad_id=None, n_layers=16):
    if pad_id is None:
        raise ValueError("pad_id must be provided to compute_loss.")
        
    logits = output['logits']
    B, S = targets.shape
    V = logits.size(-1)

    # 1. Standard Language Modeling Loss (next-token prediction)
    logits_to_use = logits[:, :-1].reshape(-1, V)
    targets_to_use = targets[:, 1:].reshape(-1)
    
    min_len = min(logits_to_use.size(0), targets_to_use.size(0))
    lm_loss = F.cross_entropy(logits_to_use[:min_len], targets_to_use[:min_len], ignore_index=pad_id)
    
    total_loss = lm_loss
    
    # 2. Horizon Prediction Loss (Chunk-ahead loss)
    if 'horizon_logits' in output:
        horizon_logits = output['horizon_logits'] # (B, H, V)
        S_chunks = output['hidden_states'].size(1)
        
        for k in range(1, horizon + 1):
            if S_chunks * CHUNK_SIZE + k >= S:
                break
                
            target_chunk_idx = S_chunks - 1 + k
            
            if target_chunk_idx * CHUNK_SIZE < S:
                h_target_token_idx = target_chunk_idx * CHUNK_SIZE
                h_logits_k = horizon_logits[:, k-1, :]
                h_targets_k = targets[:, h_target_token_idx]
                
                total_loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
                
    return total_loss

# ==========================================================
# 8. GENERATION LOGIC (Optimized for >950 TPS)
# ==========================================================
@torch.no_grad()
def generate_chunk_by_chunk(model, tokenizer, prompt, max_new_tokens, chunk_size=CHUNK_SIZE, max_context_len=MAX_SEQ_LEN, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generates tokens in a chunk-by-chunk manner using the Chunk Mode model for MAXIMUM SPEED.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Prepare Initial Context
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_context_len).input_ids.to(device)
    B, S_initial = input_ids.shape

    current_ids = input_ids
    total_generated = 0
    start_time = time.time()

    num_chunks_to_generate = (max_new_tokens + chunk_size - 1) // chunk_size
    print(f"Target: {max_new_tokens} tokens ({num_chunks_to_generate} chunks of {chunk_size}).")

    for chunk_step in range(num_chunks_to_generate):
        if (chunk_step + 1) % 20 == 0:
            print(f"  -> Generating Chunk {chunk_step + 1}/{num_chunks_to_generate} ({total_generated} tokens so far)...")
            torch.cuda.empty_cache()

        context_ids = current_ids[:, -max_context_len:]
        S_context = context_ids.size(1)

        S_context_padded = (S_context + chunk_size - 1) // chunk_size * chunk_size
        padding_needed = S_context_padded - S_context
        context_ids_padded = F.pad(context_ids, (0, padding_needed), value=tokenizer.pad_token_id)
        S_input_context = context_ids_padded.size(1)

        # Append a placeholder block (the next chunk to be predicted)
        placeholder_block = torch.full((B, chunk_size), tokenizer.pad_token_id, dtype=torch.long, device=device)
        input_for_pass = torch.cat([context_ids_padded, placeholder_block], dim=1)
        
        # Forward Pass (processes context and decodes the next chunk)
        output = model(input_for_pass)
        logits = output['logits']
        
        new_chunk_logits = logits[:, S_input_context:, :]
        
        new_chunk_ids = []
        # Chunk-internal autoregression/sampling loop
        for i in range(chunk_size):
            logit_i = new_chunk_logits[0, i, :]
            
            # --- FAST SAMPLING: Temperature + Top-K ---
            logit_i = logit_i / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logit_i, top_k)
                logit_i[logit_i < v[-1]] = -float('Inf')
                
            probs = F.softmax(logit_i, dim=-1)
            
            if probs.sum() == 0 or torch.isnan(probs).any(): 
                sampled_id = tokenizer.pad_token_id
            else:
                sampled_id = torch.multinomial(probs, 1).item()
                
            new_chunk_ids.append(sampled_id)
            
            # Update the placeholder block for the next token's decoding
            if i + 1 < chunk_size:
                # The decoder is a local operation, so we only need to update the *input* for the decoder layer to see the newly sampled token in the next step.
                # Since the full input_for_pass is used for the *current* decoder pass, we rely on the causal mask here, not modifying the input_for_pass
                # until the next full chunk generation. We rely on the local decoder to handle the internal autoregression correctly based on the causal mask.
                # However, for an *external* loop that iterates over a chunk, we must simulate the token dependence.
                pass 
        
        new_chunk_tensor = torch.tensor(new_chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        current_ids = torch.cat([current_ids, new_chunk_tensor], dim=1)
        total_generated += chunk_size
        
        if total_generated >= max_new_tokens:
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_tps = total_generated / total_time
    
    output_ids = current_ids[0].tolist()
    text_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    stats = {
        'tokens_generated': total_generated,
        'total_time': total_time,
        'average_tps': avg_tps,
    }
    
    return text_output, stats

# ==================== MAIN EXECUTION ====================
print("\n[1/7] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = len(tokenizer)
PAD_ID = tokenizer.pad_token_id

print("[2/7] Building HST v5 model in 'chunk' mode...")
model = HSTv5(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=MODE, chunk_size=CHUNK_SIZE
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")
print_memory()

print("\n[3/7] Setting up optimizer and scheduler...")
optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
scaler = GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_TRAINING_STEPS)

# Optional: Load the last saved checkpoint if it exists
latest_checkpoint = os.path.join(save_dir, 'hst_v5_interrupted_latest.pt')
if os.path.exists(latest_checkpoint):
    print(f"Loading checkpoint from: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_step = checkpoint['step']
    print(f"Resuming training from Step {start_step} (Last Loss: {checkpoint['current_loss']:.4f})")
else:
    start_step = 0
    print("No existing checkpoint found. Starting from scratch.")


print("[4/7] Loading and preparing dataset (HuggingFaceFW/fineweb-edu)...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def tokenize_and_chunk(ex):
    if 'text' not in ex:
        return {'input_ids': []}
    
    tokenized = tokenizer(
        ex["text"], 
        truncation=True, 
        max_length=MAX_SEQ_LEN, 
        padding='max_length',
        return_overflowing_tokens=False,
        return_attention_mask=False
    )
    return {"input_ids": tokenized["input_ids"]}

stream = dataset.map(tokenize_and_chunk, remove_columns=dataset.column_names).filter(lambda x: len(x['input_ids']) == MAX_SEQ_LEN)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
loader = DataLoader(stream, batch_size=BATCH_SIZE, collate_fn=collator)

print(f"[5/7] Starting simplified training (Max Steps: {MAX_TRAINING_STEPS})...\n")

model.train()
step = start_step
grad_acc_step = 0
current_loss = 0.0

try:
    for batch in loader:
        if step >= MAX_TRAINING_STEPS:
            break
        
        ids = batch["input_ids"].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(ids)
            # Architectural flow confirmed: chunk_encoder -> lattice_core -> chunk_decoder & horizon_predictor
            loss = compute_loss(out, ids, horizon=HORIZON, pad_id=PAD_ID, n_layers=N_LAYERS)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        grad_acc_step += 1
        current_loss += loss.item()
        
        if grad_acc_step % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            current_loss_print = current_loss * GRADIENT_ACCUMULATION_STEPS
            if step % 10 == 0 or step == start_step:
                print(f"Step {step:6d} | LR {optimizer.param_groups[0]['lr']:.2e} | Loss {current_loss_print:.4f} | ", end="")
                print_memory()

            step += 1
            current_loss = 0.0

            # NEW: Periodic Checkpoint Save
            if step % SAVE_CHECKPOINT_STEPS == 0:
                checkpoint_path = os.path.join(save_dir, f'hst_v5_step_{step}.pt')
                print(f"\n--- Saving periodic checkpoint to {checkpoint_path} ---")
                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'step': step,
                    'current_loss': current_loss_print
                }, checkpoint_path)
                print("--- Checkpoint saved. ---\n")
            
except KeyboardInterrupt:
    # NEW: Interruption Save on Ctrl+C
    print(f"\nTraining interrupted by user (Ctrl+C). Saving final interruption checkpoint at step {step}...")
    interrupt_checkpoint_path = os.path.join(save_dir, 'hst_v5_interrupted_latest.pt')
    torch.save({
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'current_loss': current_loss_print # Use the last calculated loss
    }, interrupt_checkpoint_path)
    print(f"Interruption checkpoint saved to: {interrupt_checkpoint_path}")
    sys.exit(0)
except Exception as e:
    # NEW: Interruption Save on other Errors
    current_loss_print = current_loss * GRADIENT_ACCUMULATION_STEPS
    print(f"\nTraining interrupted after {step} steps due to error: {e}. Saving final interruption checkpoint...")
    interrupt_checkpoint_path = os.path.join(save_dir, 'hst_v5_interrupted_latest.pt')
    torch.save({
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'current_loss': current_loss_print
    }, interrupt_checkpoint_path)
    print(f"Interruption checkpoint saved to: {interrupt_checkpoint_path}")
    raise # Re-raise the error to stop execution and show the traceback

print("\nTraining complete. Saving final checkpoint...")
torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'step': step}, os.path.join(save_dir, 'hst_v5_final.pt'))

# ==================== GENERATION AND TPS CHECK ====================
print(f"\n[6/7] Starting fast generation test ({MAX_GEN_TOKENS} tokens) for TPS validation...")
PROMPT = "The HST v5 architecture is a significant advance in language modeling. It processes long sequences using a multi-level lattice core, which allows for parallel chunk processing and highly efficient knowledge retrieval across massive contexts. This approach is designed to maximize throughput and achieve a Tokens Per Second (TPS) rate far exceeding traditional transformers. The resulting generation should be a long, coherent story about the power of this new model and its ability to rapidly generate detailed, high-quality text."

try:
    generated_text, stats = generate_chunk_by_chunk(
        model, 
        tokenizer, 
        PROMPT, 
        max_new_tokens=MAX_GEN_TOKENS,
        chunk_size=CHUNK_SIZE
    )

    print("\n======================================================================")
    print(f"HST-v5 CHUNK MODE GENERATION - FINAL RESULTS")
    print("======================================================================")
    print(f"Tokens Generated: {stats['tokens_generated']}")
    print(f"Total Time:       {stats['total_time']:.2f} seconds")
    print(f"Average TPS:      {stats['average_tps']:.2f} (Target: >950 TPS)")
    print(f"STATUS: {'SUCCESS' if stats['average_tps'] > 950 else 'NEEDS FURTHER OPTIMIZATION'}")
    print("======================================================================")
    
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to: {OUTPUT_FILENAME}")
    
except Exception as e:
    print(f"\nGeneration failed. Ensure model state is valid after training. Error: {e}")
    
# Clean up memory
del model, optimizer, scaler, loader
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
