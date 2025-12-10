"""
HST v5 - Google Colab Training Script
Optimized for T4 GPU (16GB VRAM)

Features:
- Recursive Descent Lattice Analyzer with layer-aware predictive fields
- Multi-Level + Path-Weighted Lattice Core
- Adaptive Block with early exit
- Recursive Horizon Predictor with multi-scale predictions
- Token and Chunk mode support
"""

# ==================== SETUP ====================
# !pip install torch transformers datasets bitsandbytes accelerate -q

import os
# Setting CUDA allocator config BEFORE torch import to avoid RuntimeError
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:32'

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== HYPERPARAMETERS ====================
import numpy as np
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer
import gc
# NOTE: Ensure you are running this on a GPU (e.g., Colab T4)

# ==========================================================
# 1. MODEL HYPERPARAMETERS (MUST MATCH TRAINING)
# ==========================================================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
MAX_SEQ_LEN = 768
MAX_SEQ_LEN = 768 # Max tokens in the context window
HORIZON = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_TRAINING_STEPS = 1000
INITIAL_LR = 2e-4
WARMUP_STEPS = 2000
MODE = 'token'
CHUNK_SIZE = 128

save_dir = './hst_v5_checkpoints'
os.makedirs(save_dir, exist_ok=True)
# TARGET: ~200,000 characters. 50048 is a multiple of 128 (391 chunks).
MAX_GEN_TOKENS = 50048 
OUTPUT_FILENAME = "hst_v5_chunk_story_50k_tokens_FAST.txt"
DRIVE_OUTPUT_PATH = f"/content/drive/MyDrive/{OUTPUT_FILENAME}"

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
# --- QUALITY HYPERPARAMETERS (FOR SPEED) ---
TEMPERATURE = 1.0       # Unbiased sampling (best for speed)
TOP_K = 50             # Standard for speed
# ==========================================================

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
# ==========================================================
# 2. CORE UTILITIES (Needed for inference)
# ==========================================================
KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

# ==================== LATTICE ANALYZER ====================
class RecursiveDescentLatticeAnalyzer(nn.Module):
    # ... (content remains the same) ...
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
# ==================== CHUNK ENCODER/DECODER ====================
        # Simplified for inference, assuming max_seq_len is max_chunks
        self.max_len = max_seq_len 
        self.layer_weights = nn.Parameter(torch.ones(10)) # Needs to be defined for state_dict load

    def forward(self, x):
        return x # Analyzer is primarily used in training for loss/routing

# ==========================================================
# 3. CHUNK ENCODER/DECODER (Used for local and token-level work)
# ==========================================================
class ChunkEncoder(nn.Module):
def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
super().__init__()
self.chunk_size = chunk_size
        self.d_model = d_model
encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
@@ -137,7 +59,12 @@ def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
def forward(self, token_embeddings):
B, total_tokens, D = token_embeddings.shape
num_chunks = total_tokens // self.chunk_size
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :].view(B * num_chunks, self.chunk_size, D)
        tokens_to_use = num_chunks * self.chunk_size
        
        if num_chunks == 0:
            return token_embeddings.new_zeros(B, 0, D)
            
        chunks = token_embeddings[:, :tokens_to_use, :].view(B * num_chunks, self.chunk_size, D)
encoded_tokens = self.local_encoder(chunks)
query = self.pooling_query.expand(B * num_chunks, -1, -1)
pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
@@ -147,119 +74,58 @@ class ChunkDecoder(nn.Module):
def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
super().__init__()
self.chunk_size = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.pos_embedding = nn.Embedding(chunk_size, d_model) 
decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
self.lm_head = nn.Linear(d_model, vocab_size)

def forward(self, chunk_embeddings, target_token_embeddings):
B, num_chunks, D = chunk_embeddings.shape
        seq_len = num_chunks * self.chunk_size
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

# ==================== TRANSFORMER LAYERS ====================
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
        if S > 1:
            causal_mask = torch.triu(torch.ones(full_S, full_S, dtype=torch.bool, device=x.device), diagonal=1)
            attn_weights[:, :, :, :].masked_fill_(causal_mask[-S:, :].bool()[None, None, :, :], -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_output), present

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

# ==================== ADAPTIVE LATTICE PROCESSOR ====================
# ==========================================================
# 4. ADAPTIVE LATTICE PROCESSOR (Inter-chunk dependency)
# ==========================================================
class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
    def __init__(self, d_model, max_num_chunks):
super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_num_chunks) 
self.layer_processors = nn.ModuleList([
nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True) for _ in range(10)
])
self.task_router = nn.Sequential(
nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10), nn.Sigmoid()
)

    def forward(self, x, horizon_targets=None):
    def forward(self, x, horizon_targets=None): 
B, S, D = x.shape
        task_embedding = x.mean(dim=1)
        task_embedding = x.mean(dim=1) 
layer_gates = self.task_router(task_embedding)

h = x
for layer_idx, processor in enumerate(self.layer_processors):
gate = layer_gates[:, layer_idx].unsqueeze(1).unsqueeze(2)
            if gate.mean() > 0.1:
            if gate.mean() > 0.1: 
h_layer = processor(h)
h = h + gate * (h_layer - h)
return h

# ==================== RECURSIVE HORIZON PREDICTOR ====================
# ==========================================================
# 5. RECURSIVE HORIZON PREDICTOR (Chunk-ahead prediction)
# ==========================================================
class RecursiveHorizonPredictor(nn.Module):
def __init__(self, d_model, vocab_size, horizon=8):
super().__init__()
@@ -273,8 +139,9 @@ def __init__(self, d_model, vocab_size, horizon=8):

def forward(self, h_sequence):
B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :]
        h_t = h_sequence[:, -1, :] 

        # Prediction logic remains the same, predicting tokens from future chunk states
coarse_preds = {}
for offset in [4, min(10, self.horizon)]:
offset_emb = self.lattice_embeddings(torch.tensor([min(offset - 1, 19)], device=h_t.device))
@@ -304,273 +171,287 @@ def forward(self, h_sequence):
confidence = torch.ones(B, self.horizon, device=h_t.device)
return logits, confidence

# ==================== HST v5 MODEL ====================
class HSTv5(nn.Module):
    # REPAIR: Added _init_weights for proper transformer initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard LayerNorm initialization
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=512, horizon=8,
                 early_exit_threshold=0.93, mode='token', chunk_size=128):
# ==========================================================
# 6. HSTv5 INFERENCE MODEL (Unified class with chunk focus)
# ==========================================================
class HSTv5_Inference(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=768, horizon=8, early_exit_threshold=0.93, mode='chunk', chunk_size=128):
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
        max_num_chunks = max_seq_len // chunk_size 

        # Chunk Mode Components
        self.pos_embedding = nn.Embedding(max_num_chunks, d_model) # Pos embedding for chunks
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size, n_heads=N_HEADS//2) # Use fewer heads for local
        self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size, n_heads=N_HEADS//2)
        self.lattice_core = AdaptiveLatticeProcessor(d_model, max_num_chunks) 

        if self.mode == 'chunk':
            self.pos_embedding = nn.Embedding(max_seq_len * chunk_size, d_model)
            self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
            self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size)
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(self.n_bottom_layers)])
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
            self.top_stack = nn.ModuleList([TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(self.n_top_layers)])
        # Dummy layers for token-mode components to allow non-strict loading
        self.adaptive_bottom = nn.ModuleList([nn.Identity() for _ in range(n_layers // 2)])
        self.top_stack = nn.ModuleList([nn.Identity() for _ in range(n_layers - n_layers // 2)])


self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.ln_f = nn.LayerNorm(d_model)

        # REPAIR: Apply weight initialization
        self.apply(self._init_weights)

def forward(self, input_ids, cache=None):
        if self.mode == 'token':
            return self.forward_token(input_ids, cache)
        else:
            return self.forward_chunk(input_ids)
        return self.forward_chunk(input_ids) # Only support chunk mode

    def forward_token(self, input_ids, cache=None):
        B, seq_len = input_ids.shape
    def forward_chunk(self, input_ids):
        B, total_tokens = input_ids.shape
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
            
            if past_len == 0 and i >= 1 and conf.mean().item() > self.early_exit_threshold:
                predicted_depth = i + 1
                break
        
        h_lattice_out = self.lattice_core(x)
        x = self.token_embedding(input_ids)
        chunk_emb = self.chunk_encoder(x)
        B, num_chunks, D = chunk_emb.shape

        h_top_in = h_lattice_out
        for i, block in enumerate(self.top_stack):
            layer_past = cache[cache_idx] if cache and cache_idx < len(cache) else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
            cache_idx += 1
        if num_chunks == 0:
            # Handle case where input is too short to form a single chunk
            zero_logits = torch.zeros(B, total_tokens, self.vocab_size, device=device)
            zero_horizon = torch.zeros(B, self.horizon, self.vocab_size, device=device)
            return {
                'logits': zero_logits, 'horizon_logits': zero_horizon, 
                'confidence': torch.zeros(B, self.horizon, device=device), 
                'hidden_states': chunk_emb, 'bottom_depth': 0, 'cache': None
            }

        h_final = h_top_in
        logits_t1 = self.lm_head(self.ln_f(h_final))
        logits_horizon, confidence = self.horizon_predictor(h_final[:, -1:, :])
        # Chunk Positional Encoding
        chunk_positions = torch.arange(0, num_chunks, dtype=torch.long, device=device)
        chunk_positions = chunk_positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        h_in = chunk_emb + self.pos_embedding(chunk_positions)

        return {
            'logits': logits_t1,
            'horizon_logits': logits_horizon.squeeze(1),
            'confidence': confidence.squeeze(1),
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
        # Lattice Core Processing
        h_lattice_out = self.lattice_core(h_in)

        chunk_emb = self.chunk_encoder(x)
        h_lattice_out = self.lattice_core(chunk_emb)
        # Chunk Decoding (Token-level prediction)
logits = self.chunk_decoder(h_lattice_out, x)
        
        # Horizon Prediction (Chunk-level prediction)
logits_horizon, confidence = self.horizon_predictor(h_lattice_out)

return {
            'logits': logits,
            'horizon_logits': logits_horizon,
            'logits': logits, 
            'horizon_logits': logits_horizon, 
'confidence': confidence,
            'hidden_states': h_lattice_out,
            'bottom_depth': 0,
            'hidden_states': h_lattice_out, 
            'bottom_depth': 0, 
'cache': None
}

# ==================== LOSS FUNCTION ====================
def compute_loss(output, targets, horizon=8, gamma=0.95, pad_id=None, n_layers=16):
    if pad_id is None:
        raise ValueError("pad_id must be provided to compute_loss.")
        
    logits = output['logits']
    B, S = targets.shape
    V = logits.size(-1)
# ==========================================================
# 7. GENERATION LOGIC (Simplified & Optimized)
# ==========================================================
@torch.no_grad()
def generate_chunk_by_chunk(model, tokenizer, prompt, max_new_tokens, chunk_size=CHUNK_SIZE, max_context_len=MAX_SEQ_LEN, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generates tokens in a chunk-by-chunk manner using the Chunk Mode model (MAXIMUM SPEED).
    """
    device = next(model.parameters()).device

    # Standard Language Modeling Loss (next-token prediction)
    logits = logits[:, :S]
    pred_logits = logits[:, :-1].reshape(-1, V)
    pred_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(pred_logits, pred_targets, ignore_index=pad_id)
    # 1. Prepare Initial Context
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_context_len).input_ids.to(device)
    B, S_initial = input_ids.shape

    # Horizon Prediction Loss
    if 'horizon_logits' in output:
        horizon_logits = output['horizon_logits']
        H = horizon_logits.size(1)
        for k in range(1, min(H + 1, S)):
            h_logits_k = horizon_logits[:, k-1, :]
            h_targets_k = targets[:, k]
            loss += (gamma ** k) * F.cross_entropy(h_logits_k, h_targets_k, ignore_index=pad_id)
    current_ids = input_ids
    total_generated = 0
    start_time = time.time()

    # Early-Exit Regularization Loss
    if 'bottom_depth' in output and output['bottom_depth'] > 0:
        depth = float(output['bottom_depth'])
    # We generate an exact number of chunks
    num_chunks_to_generate = (max_new_tokens + chunk_size - 1) // chunk_size
    print(f"Target: {max_new_tokens} tokens ({num_chunks_to_generate} chunks of {chunk_size}).")

    for chunk_step in range(num_chunks_to_generate):
        if (chunk_step + 1) % 20 == 0:
            print(f"  -> Generating Chunk {chunk_step + 1}/{num_chunks_to_generate} ({total_generated} tokens so far)...")
            torch.cuda.empty_cache()

        # 2. Prepare the input block: Context + Placeholder Block
        # Only use the last MAX_SEQ_LEN tokens as context
        context_ids = current_ids[:, -max_context_len:]
        S_context = context_ids.size(1)

        # Pad context to be a multiple of chunk_size (necessary for ChunkEncoder)
        S_context_padded = (S_context + chunk_size - 1) // chunk_size * chunk_size
        padding_needed = S_context_padded - S_context
        context_ids_padded = F.pad(context_ids, (0, padding_needed), value=tokenizer.pad_token_id)

        conf = output.get('confidence', torch.ones(B, 1, device=targets.device)).mean()
        S_input_context = context_ids_padded.size(1)

        target_depth_norm = torch.tensor(
            depth / n_layers, 
            dtype=torch.float32, 
            device=targets.device
        )
        loss += 0.03 * F.mse_loss(conf.float(), target_depth_norm)
        # The key to chunk generation: The model must be trained to predict the next chunk 
        # based on the *current chunk's hidden state*. 
        # By appending a placeholder, the model is forced to decode the next tokens.
        placeholder_block = torch.full((B, chunk_size), tokenizer.pad_token_id, dtype=torch.long, device=device)
        input_for_pass = torch.cat([context_ids_padded, placeholder_block], dim=1)
        
        # 3. Forward Pass (Model decodes *all* tokens based on the lattice core output)
        output = model(input_for_pass)
        logits = output['logits']
        
        # 4. Extract logits for the newly generated chunk (it's the last CHUNK_SIZE tokens)
        # Note: If there was no padding, the new tokens are logits[:, S_context:]. 
        # Since we padded, the logits are slightly shifted. The *new chunk* starts at index S_input_context.
        new_chunk_logits = logits[:, S_input_context:, :]
        
        new_chunk_ids = []
        for i in range(chunk_size):
            logit_i = new_chunk_logits[0, i, :]
            
            # --- FAST SAMPLING: Temperature + Top-K ---
            logit_i = logit_i / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logit_i, top_k)
                logit_i[logit_i < v[-1]] = -float('Inf')
                
            probs = F.softmax(logit_i, dim=-1)
            
            # Safety check: ensure we can sample
            if probs.sum() == 0 or torch.isnan(probs).any(): 
                sampled_id = tokenizer.pad_token_id
            else:
                sampled_id = torch.multinomial(probs, 1).item()
            
            new_chunk_ids.append(sampled_id)
            
            # --- Quick update of the placeholder block to simulate auto-regression within the chunk ---
            # This is technically not pure autoregressive but improves sample quality without 
            # re-running the full forward pass. (Optional: Can remove for absolute max speed)
            if i + 1 < chunk_size:
                input_for_pass[:, S_input_context + i + 1] = sampled_id 
        
        new_chunk_tensor = torch.tensor(new_chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # 5. Update current IDs
        current_ids = torch.cat([current_ids, new_chunk_tensor], dim=1)
        total_generated += chunk_size
        
        if total_generated >= max_new_tokens:
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_tps = total_generated / total_time

    return loss

# ==================== TRAINING ====================
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = len(tokenizer)
PAD_ID = tokenizer.pad_token_id

print("[2/5] Building HST v5 model...")
model = HSTv5(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=MODE, chunk_size=CHUNK_SIZE
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params")
print_memory()

print("\n[3/5] Setting up optimizer...")
try:
    import bitsandbytes as bnb
    # Use standard AdamW for now, since bnb might also interact with numerical stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    print("Using standard AdamW (8-bit option removed for initial stability check)")
except ImportError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)

scaler = GradScaler()
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_TRAINING_STEPS)

print("[4/5] Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

def tokenize_and_chunk(ex):
    if 'text' not in ex:
        return {'input_ids': []}
    output_ids = current_ids[0].tolist()

    tokenized = tokenizer(
        ex["text"], 
        truncation=True, 
        max_length=MAX_SEQ_LEN, 
        return_overflowing_tokens=False,
        return_attention_mask=False
    )
    stats = {
        'tokens_generated': total_generated,
        'total_time': total_time,
        'average_tps': avg_tps,
    }

    return {"input_ids": tokenized["input_ids"]}

stream = dataset.map(tokenize_and_chunk, remove_columns=dataset.column_names).filter(lambda x: len(x['input_ids']) > 10)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
loader = DataLoader(stream, batch_size=BATCH_SIZE, collate_fn=collator)
    return output_ids, stats

print(f"[5/5] Starting training (Max Steps: {MAX_TRAINING_STEPS})...\n")

model.train()
step = 0
grad_acc_step = 0

try:
    for batch in loader:
        if step >= MAX_TRAINING_STEPS:
            break
        
        ids = batch["input_ids"].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(ids)
            loss = compute_loss(out, ids, horizon=HORIZON, pad_id=PAD_ID, n_layers=N_LAYERS)
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
def load_model_from_checkpoint(mode, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    VOCAB_SIZE = len(tokenizer)
    
    print(f"\n[INFO] Loading model in {mode} mode...")
    
    # Initialize the inference-specific model class
    model = HSTv5_Inference(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=mode, chunk_size=CHUNK_SIZE,
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']

            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            if step % 10 == 0:
                depth = out.get('bottom_depth', 0)
                print(f"Step {step:6d} | LR {optimizer.param_groups[0]['lr']:.2e} | Loss {current_loss:.4f} | Depth {depth} | ", end="")
                print_memory()
            # Chunk Mode: Only load relevant weights (chunk-related and core shared layers)
            state_dict_filtered = {
                k: v for k, v in state_dict.items() 
                if not (k.startswith('adaptive_bottom') or k.startswith('top_stack'))
            }

            if step % 100 == 0 and step > 0:
                torch.save(model.state_dict(), f"{save_dir}/ckpt_step_{step}.pt")
            model.load_state_dict(state_dict_filtered, strict=False)
            print(f"✅ Successfully loaded model state NON-STRICTLY ({mode} keys filtered) from: {checkpoint_path}")

        except Exception as e:
            print(f"❌ WARNING: Load failed. Using random weights. Error: {e}")
    else:
        print(f"❌ WARNING: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.eval()
    return model

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
# ==========================================================
# 8. EXECUTION BLOCK
# ==========================================================

            step += 1
if __name__ == '__main__':
    
    # --- Colab Setup --- 
    try:
        from google.colab import drive
        LOCAL_CHECKPOINT_PATH = '/content/hst_v5_checkpoints/hst_v5_final.pt'
        if os.path.exists(LOCAL_CHECKPOINT_PATH):
            CHECKPOINT_PATH = LOCAL_CHECKPOINT_PATH
            DRIVE_MOUNTED = False
        else:
            print("[INFO] Attempting to mount Google Drive...")
            drive.mount('/content/drive')
            CHECKPOINT_PATH = f'/content/drive/MyDrive/hst_v5_checkpoints/hst_v5_final.pt'
            DRIVE_MOUNTED = True
    except ImportError:
        CHECKPOINT_PATH = './hst_v5_checkpoints/hst_v5_final.pt'
        DRIVE_MOUNTED = False
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    VOCAB_SIZE = len(tokenizer)

    print("=" * 70)
    print(f"HST-v5 CHUNK MODE: 200K Character Generation (Target: {MAX_GEN_TOKENS} Tokens)")
    print(f"MAX SPEED TEST: T={TEMPERATURE}, Top-K={TOP_K}")
    print("=" * 70)

    # --- Load Model in Chunk Mode ---
    model_chunk = load_model_from_checkpoint(mode='chunk', checkpoint_path=CHECKPOINT_PATH)
    
    EVAL_PROMPT = "The ancient scroll detailed the forgotten history of the star-faring empire, which began when their homeworld, Xylos, faced..."

except KeyboardInterrupt:
    print("\n!! Training INTERRUPTED !!")
    torch.save(model.state_dict(), f'{save_dir}/hst_v5_interrupt_step_{step}.pt')
    print(f"\nStarting generation of {MAX_GEN_TOKENS} tokens...")
    
    # --- Start Generation ---
    output_ids, stats = generate_chunk_by_chunk(
        model=model_chunk,
        tokenizer=tokenizer,
        prompt=EVAL_PROMPT,
        max_new_tokens=MAX_GEN_TOKENS,
        temperature=TEMPERATURE, 
        top_k=TOP_K,             
        chunk_size=CHUNK_SIZE,
        max_context_len=MAX_SEQ_LEN
    )

if step > 0:
    torch.save(model.state_dict(), f'{save_dir}/hst_v5_final.pt')
print(f"\nTRAINING COMPLETE! Final Step: {step}")
print_memory()
    # --- Save and Report ---
    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    final_path = DRIVE_OUTPUT_PATH if DRIVE_MOUNTED else OUTPUT_FILENAME
    
    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print("\n" + "="*70)
        print(f"✅ GENERATION COMPLETE. Story saved to: {final_path}")
    except Exception as e:
        print(f"❌ WARNING: Could not save file to {final_path}. Error: {e}")
        
    print(f"Total Characters Generated: {len(full_text)}")
    print(f"Total Tokens Generated: {stats['tokens_generated']}")
    print(f"Total Time: {stats['total_time']:.4f}s")
    print(f"**Average TPS (Tokens Per Second): {stats['average_tps']:.2f}**")
    print("="*70)
